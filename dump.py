"""
Latent dump / 诊断脚本
----------------------
目的：确认草地(纹理区)的 latent 是不是被 entropy model 压成近常数，
      从而导致 decoder 只能复制同一个 motif → 重复纹理。

用法：
    python dump_latent.py \
        --checkpoint /path/to/ckpt.pth \
        --image /path/to/plane_grass.png \
        --cuda \
        --save_path latent_dump

产出（都在 --save_path 目录）：
    latent_maps.png   : latent energy / entropy scale / 输入图(latent分辨率) 三联图
    rate_map.png      : per-position 真实 bit 分布（空间 rate 图）
    x_hat.png         : 重建图（对照用）
    region_grid.png   : 在重建图上叠网格，方便你读坐标去指定草地/飞机区域
    report.txt        : 数值报告（草地 vs 飞机的 spatial_std / neighbor_diff / bits）

判读（report.txt 里）：
    若 grass.spatial_std << plane.spatial_std（差 ~3x 以上）
       且 grass.bits << plane.bits
       → 机制1：草地 latent 被压成近常数，信息不足 → 任何 loss 都救不了重复，
         得让 noise 补位 或 提高草地 rate。
    若两者接近
       → 信息够，重复是 decoder/loss 的问题，不是信息不足。
"""

import os
import sys
import math
import argparse
import warnings

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from models.dcae import DCAE  # 需与本脚本同目录


# ----------------------------- 工具 -----------------------------
def pad(x, p=128):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    pl = (new_w - w) // 2
    pr = new_w - w - pl
    pt = (new_h - h) // 2
    pb = new_h - h - pt
    x_padded = F.pad(x, (pl, pr, pt, pb), mode="constant", value=0)
    return x_padded, (pl, pr, pt, pb)


def to_np_img(t):
    return t[0].detach().permute(1, 2, 0).cpu().clamp(0, 1).numpy()


# ----------------------------- 核心 dump -----------------------------
def dump(net, x_padded, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)
    lines = []

    def log(s):
        print(s)
        lines.append(s)

    with torch.no_grad():
        out = net.forward(x_padded)
        x_hat = out["x_hat"].clamp(0, 1)
        y = out["para"]["y"]           # (B, M, Hy, Wy)  encoder latent
        means = out["para"]["means"]   # (B, M, Hy, Wy)
        scales = out["para"]["scales"] # (B, M, Hy, Wy)

        B, C, Hy, Wy = y.shape
        H, W = x_padded.shape[2:]
        log(f"input padded: {H}x{W}   latent: {C}x{Hy}x{Wy}  (下采样 {H // Hy}x)")

        # ---- 1) latent 能量热图 & entropy scale ----
        y_energy = y.pow(2).mean(1, keepdim=True)      # (B,1,Hy,Wy) 每位置信息量代理
        scale_map = scales.mean(1, keepdim=True)       # sigma 大 = 花更多 bit / 更不确定

        # ---- 2) per-position 真实 bit ----
        # gaussian_conditional 回传 (quantized, likelihoods)
        _, y_like = net.gaussian_conditional(y, scales, means)
        bits = (-torch.log2(y_like.clamp_min(1e-9))).sum(1, keepdim=True)  # (B,1,Hy,Wy)

        # ---- 存三联图 ----
        x_small = F.interpolate(x_padded, size=(Hy, Wy), mode="bilinear",
                                align_corners=False)
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        im0 = ax[0].imshow(y_energy[0, 0].cpu(), cmap="viridis")
        ax[0].set_title("latent energy (per-position)")
        plt.colorbar(im0, ax=ax[0], fraction=0.046)
        im1 = ax[1].imshow(scale_map[0, 0].cpu(), cmap="viridis")
        ax[1].set_title("entropy scale (rate proxy)")
        plt.colorbar(im1, ax=ax[1], fraction=0.046)
        ax[2].imshow(to_np_img(x_small))
        ax[2].set_title("input @ latent res")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "latent_maps.png"), dpi=120)
        plt.close()
        log(f"saved latent_maps.png")

        # ---- 存 rate map ----
        plt.figure(figsize=(8, 6))
        plt.imshow(bits[0, 0].cpu(), cmap="hot")
        plt.colorbar(fraction=0.046)
        plt.title("per-position bits (spatial rate map)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "rate_map.png"), dpi=120)
        plt.close()
        log(f"saved rate_map.png  (total bits={bits.sum().item():.0f}, "
            f"bpp={bits.sum().item() / (H * W):.4f})")

        # ---- 存重建图 ----
        plt.figure(figsize=(10, 8))
        plt.imshow(to_np_img(x_hat))
        plt.title("x_hat (reconstruction)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "x_hat.png"), dpi=120)
        plt.close()

        # ---- 带网格的重建图,方便读坐标(网格标注的是 latent 坐标) ----
        fig, axg = plt.subplots(figsize=(12, 9))
        axg.imshow(to_np_img(x_hat))
        # 每隔 latent 的若干格画一条线,并标注 latent 坐标
        step_px_y = H / Hy
        step_px_x = W / Wy
        grid = max(1, min(Hy, Wy) // 12)  # 大约画 12 条线
        for ly in range(0, Hy + 1, grid):
            axg.axhline(ly * step_px_y, color="cyan", lw=0.5, alpha=0.6)
            axg.text(2, ly * step_px_y, f"{ly}", color="cyan", fontsize=8, va="top")
        for lx in range(0, Wy + 1, grid):
            axg.axvline(lx * step_px_x, color="cyan", lw=0.5, alpha=0.6)
            axg.text(lx * step_px_x, 2, f"{lx}", color="cyan", fontsize=8, ha="left")
        axg.set_title(f"x_hat with LATENT-coord grid (latent {Hy}x{Wy}). "
                      f"用这些数字去指定 --grass / --plane 区域")
        axg.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "region_grid.png"), dpi=120)
        plt.close()
        log(f"saved region_grid.png  → 打开它读草地/飞机的 latent 坐标")

    return y, means, scales, bits, (Hy, Wy)


def compare_regions(y, bits, regions, save_dir):
    """regions: dict name -> (y1,y2,x1,x2) in latent coords."""
    lines = ["", "=== region comparison (latent coords) ==="]
    results = {}
    for name, (y1, y2, x1, x2) in regions.items():
        patch = y[:, :, y1:y2, x1:x2]
        bpatch = bits[:, :, y1:y2, x1:x2]
        if patch.numel() == 0:
            lines.append(f"[{name}] EMPTY region {(y1, y2, x1, x2)} — 检查坐标")
            continue
        spatial_std = patch.std(dim=[2, 3]).mean().item()   # 空间方差(越小越接近常数)
        nbr_v = (patch[:, :, 1:, :] - patch[:, :, :-1, :]).abs().mean().item()
        nbr_h = (patch[:, :, :, 1:] - patch[:, :, :, :-1]).abs().mean().item()
        mean_bits = bpatch.mean().item()
        results[name] = spatial_std
        lines.append(
            f"[{name}] region(y{y1}:{y2}, x{x1}:{x2})  "
            f"spatial_std={spatial_std:.4f}  "
            f"neighbor_diff(v/h)={nbr_v:.4f}/{nbr_h:.4f}  "
            f"mean_bits/pos={mean_bits:.3f}"
        )

    # 自动判读
    if "grass" in results and "plane" in results:
        r = results["plane"] / max(results["grass"], 1e-8)
        lines.append("")
        lines.append(f"plane_std / grass_std = {r:.2f}")
        if r >= 3.0:
            lines.append(">>> 机制1 成立：草地 latent 明显更接近常数 → 信息不足。")
            lines.append(">>> 结论：重复来自 latent 没信息，需 noise 补位或提高草地 rate；")
            lines.append(">>>       单纯调 WD/spectral/LPIPS 无法根治。")
        elif r <= 1.5:
            lines.append(">>> 机制1 不成立：草地 latent 信息量与结构区相当。")
            lines.append(">>> 结论：重复是 decoder/loss 问题（noise 未被奖励），")
            lines.append(">>>       该动 loss（LPIPS 加权 / sliced-W）或修 NoiseTransform。")
        else:
            lines.append(">>> 介于中间：信息略少但非近常数，两方面可能都有贡献。")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(save_dir, "report.txt"), "a") as f:
        f.write(report + "\n")


# ----------------------------- 额外：noise 敏感度 -----------------------------
def noise_sensitivity(net, x_padded, save_dir):
    """两次 forward(不同随机 noise)看输出差异 + 存两张原图肉眼比对斜纹。"""
    lines = ["", "=== noise sensitivity (seed diff) ==="]
    with torch.no_grad():
        a = net.forward(x_padded)["x_hat"].clamp(0, 1)
        b = net.forward(x_padded)["x_hat"].clamp(0, 1)
        diff = (a - b).abs()
        lines.append(f"seed diff mean={diff.mean().item():.6f}  max={diff.max().item():.6f}")
        if diff.mean().item() < 1e-5:
            lines.append(">>> 输出几乎不随 noise 改变 → noise 被学死。")
        else:
            lines.append(">>> 输出随 noise 改变 → noise 有在参与。")

        # ---- 存两张不同 seed 的重建图(肉眼比对斜纹动不动) ----
        plt.imsave(os.path.join(save_dir, "seed_a.png"), to_np_img(a))
        plt.imsave(os.path.join(save_dir, "seed_b.png"), to_np_img(b))
        lines.append("saved seed_a.png / seed_b.png  (斜纹固定→deterministic; 变→noise)")

        # ---- 并排对比图,直接看差异 ----
        fig, ax = plt.subplots(1, 3, figsize=(21, 7))
        ax[0].imshow(to_np_img(a)); ax[0].set_title("seed A"); ax[0].axis("off")
        ax[1].imshow(to_np_img(b)); ax[1].set_title("seed B"); ax[1].axis("off")
        dmap = diff.mean(1, keepdim=True)[0, 0].cpu()
        im = ax[2].imshow(dmap, cmap="magma")
        ax[2].set_title("|A - B|"); ax[2].axis("off")
        plt.colorbar(im, ax=ax[2], fraction=0.046)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "seed_compare.png"), dpi=120)
        plt.close()
        lines.append("saved seed_compare.png  (A / B / diff 并排)")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(save_dir, "report.txt"), "a") as f:
        f.write(report + "\n")

# ----------------------------- 打印 noise_scale 学到多少 -----------------------------
def dump_noise_scales(net, save_dir):
    lines = ["", "=== NoiseInjectedResBlock noise_scale 值 ==="]
    found = False
    for name, m in net.named_modules():
        if m.__class__.__name__ == "NoiseInjectedResBlock":
            found = True
            s1 = m.noise_scale1.abs().mean().item()
            s2 = m.noise_scale2.abs().mean().item()
            lines.append(f"{name}: |scale1|={s1:.4f}  |scale2|={s2:.4f}")
    if not found:
        lines.append("(没找到 NoiseInjectedResBlock — 确认 decoder_nt 有开)")
    else:
        lines.append(">>> 若这些值都很小(接近 0)→ noise 被学死,loss 没在奖励它。")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(save_dir, "report.txt"), "a") as f:
        f.write(report + "\n")


# ----------------------------- main -----------------------------
def parse_region(s, Hy, Wy):
    """接受 'y1,y2,x1,x2'(latent坐标) 或 'auto_grass'/'auto_plane'。"""
    if s == "auto_grass":
        return (int(Hy * 0.6), Hy, 0, int(Wy * 0.45))
    if s == "auto_plane":
        return (int(Hy * 0.3), int(Hy * 0.65), int(Wy * 0.25), int(Wy * 0.75))
    y1, y2, x1, x2 = [int(v) for v in s.split(",")]
    return (y1, y2, x1, x2)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--save_path", type=str, default="latent_dump")
    ap.add_argument("--grass", type=str, default="auto_grass",
                    help="latent坐标 'y1,y2,x1,x2' 或 auto_grass")
    ap.add_argument("--plane", type=str, default="auto_plane",
                    help="latent坐标 'y1,y2,x1,x2' 或 auto_plane")
    ap.add_argument("--pad", type=int, default=128)
    args = ap.parse_args(argv)

    device = "cuda:0" if args.cuda and torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.enabled = False

    net = DCAE().to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    net.load_state_dict(sd)
    print("loaded checkpoint:", args.checkpoint)

    img = Image.open(args.image).convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    x_padded, _ = pad(x, args.pad)

    os.makedirs(args.save_path, exist_ok=True)
    open(os.path.join(args.save_path, "report.txt"), "w").close()  # 清空

    y, means, scales, bits, (Hy, Wy) = dump(net, x_padded, args.save_path, device)

    grass = parse_region(args.grass, Hy, Wy)
    plane = parse_region(args.plane, Hy, Wy)
    print(f"\n草地区域(latent): {grass}\n飞机区域(latent): {plane}")
    print("※ 若自动区域不准,打开 region_grid.png 读坐标,用 --grass y1,y2,x1,x2 重跑\n")

    compare_regions(y, bits, {"grass": grass, "plane": plane}, args.save_path)
    dump_noise_scales(net, args.save_path)
    noise_sensitivity(net, x_padded, args.save_path)

    print(f"\n全部结果在: {args.save_path}/")
    print("重点看 report.txt 的结论,以及 rate_map.png / noise_seed_diff.png")


if __name__ == "__main__":
    main(sys.argv[1:])