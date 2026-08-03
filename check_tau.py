"""
check_tau.py

驗證 τ 可交換性場 —— 複用你 check_maskA 的可視化閉環。
把 τ 和 輸入 / 頻譜 mask 並排看:確認 τ 在草地亮、天空黑、飛機黑。

用法:
  python check_tau.py --image path/to/img.png [--cuda] [--save_path tau_check]
  # 需要 wa_wd.py 在 PYTHONPATH(取 MultiscaleTruncatedVGG16)
  # 頻譜 mask 對照為選用:給 EMLNet 路徑則一併畫 stochastic_mask_fn

驗證判準:
  τ 草地 明顯 > τ 飛機 且 > τ 天空 (對比 >2x)
  對照頻譜 mask:兩者在草地應大致一致;τ 在飛機/天空應更乾淨(VGG 語義優勢)
"""
import os
import sys
import argparse
import warnings

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from tau_field import compute_tau, _select_scale0, tau_to_log2_sigma


def load_vgg(device):
    """取 WA-WD 的 VGG backbone。"""
    from wa_wd import MultiscaleTruncatedVGG16
    vgg = MultiscaleTruncatedVGG16(
        requires_grad=False, pretrained=True, truncate_slice=5).to(device)
    vgg.eval()
    return vgg


def load_saliency(args, device):
    """EMLNet saliency —— τ 的飛機門控必需。三路徑齊全才載入。"""
    paths = [args.emlnet_imagenet_path, args.emlnet_places_path,
             args.emlnet_decoder_path]
    if any(p is None for p in paths):
        print("[warn] 未給 EMLNet 路徑 → τ 無飛機門控(飛機不會被壓)")
        return None
    import torch.nn as nn
    import emlnet.decoder as eml_decoder
    import emlnet.resnet as eml_resnet

    class EMLNetSaliency(nn.Module):
        def __init__(self, ip, pp, dp, input_size=(480, 640), num_feat=5):
            super().__init__()
            self.input_size = input_size
            self.img_model = eml_resnet.resnet50(ip)
            self.pla_model = eml_resnet.resnet50(pp)
            self.decoder_model = eml_decoder.build_decoder(
                dp, input_size, num_feat, num_feat)
            self.eval()
            for p in self.parameters():
                p.requires_grad = False

        @torch.no_grad()
        def forward(self, image):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            _, _, H, W = image.shape
            r = F.interpolate(image, size=self.input_size,
                              mode='bilinear', align_corners=False)
            imf = self.img_model(r, decode=True)
            plf = self.pla_model(r, decode=True)
            pred = self.decoder_model([imf, plf])
            return F.interpolate(pred, size=(H, W),
                                 mode='bilinear', align_corners=False)

    return EMLNetSaliency(*paths).to(device)


def try_spectral_mask(x, args, device):
    """選用:若給了 EMLNet 路徑,畫頻譜 mask 做對照。否則回 None。"""
    if not (args.emlnet_imagenet_path and args.emlnet_places_path
            and args.emlnet_decoder_path):
        return None
    try:
        from mask_utils import stochastic_mask_fn
        import torch.nn as nn
        import emlnet.decoder as eml_decoder
        import emlnet.resnet as eml_resnet

        class EMLNetSaliency(nn.Module):
            def __init__(self, ip, pp, dp, input_size=(480, 640), num_feat=5):
                super().__init__()
                self.input_size = input_size
                self.img_model = eml_resnet.resnet50(ip)
                self.pla_model = eml_resnet.resnet50(pp)
                self.decoder_model = eml_decoder.build_decoder(
                    dp, input_size, num_feat, num_feat)
                self.eval()
                for p in self.parameters():
                    p.requires_grad = False

            @torch.no_grad()
            def forward(self, image):
                if image.dim() == 3:
                    image = image.unsqueeze(0)
                _, _, H, W = image.shape
                r = F.interpolate(image, size=self.input_size,
                                  mode='bilinear', align_corners=False)
                imf = self.img_model(r, decode=True)
                plf = self.pla_model(r, decode=True)
                pred = self.decoder_model([imf, plf])
                return F.interpolate(pred, size=(H, W),
                                     mode='bilinear', align_corners=False)

        sal = EMLNetSaliency(args.emlnet_imagenet_path,
                             args.emlnet_places_path,
                             args.emlnet_decoder_path).to(device)
        score = stochastic_mask_fn(x, saliency_model=sal,
                                   hf_pct=0.6, hf_temp=0.05, use_aniso=False)
        return score
    except Exception as e:
        print(f"[spectral] skip ({e})")
        return None


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--save_path", default="tau_check")
    ap.add_argument("--num-scales", type=int, default=1,
                    help="VGG 多尺度數(τ 只用 scale0,設 1 即可)")
    ap.add_argument("--box-ks", type=int, default=5)
    ap.add_argument("--sal-strength", type=float, default=1.0,
                    help="saliency 门控强度(压飞机)")
    ap.add_argument("--use-deep", action="store_true",
                    help="启用深层语义门控(仅对照;实测有害)")
    ap.add_argument("--deep-gate-strength", type=float, default=1.0)
    ap.add_argument("--show-spectral", action="store_true",
                    help="额外画频谱 mask 对照")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--sigma-min", type=float, default=1.0)
    ap.add_argument("--sigma-max", type=float, default=16.0)
    # 選用頻譜對照
    ap.add_argument("--emlnet-imagenet-path", default=None)
    ap.add_argument("--emlnet-places-path", default=None)
    ap.add_argument("--emlnet-decoder-path", default=None)
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args(argv)

    device = "cuda:0" if args.cuda and torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_path, exist_ok=True)

    img = Image.open(args.image).convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    B, _, H, W = x.shape

    vgg = load_vgg(device)
    with torch.no_grad():
        feats_all = vgg(x, num_scales=args.num_scales)
    feats = _select_scale0(feats_all, truncate_slice=5)

    sal_model = load_saliency(args, device)
    saliency = sal_model(x) if sal_model is not None else None

    tau, parts = compute_tau(
        feats, saliency=saliency,
        shallow_idx=(0, 1), deep_idx=(3, 4),
        box_ks=args.box_ks, sal_strength=args.sal_strength,
        use_deep=args.use_deep, deep_gate_strength=args.deep_gate_strength,
        gamma=args.gamma, return_parts=True)

    log2_sigma = tau_to_log2_sigma(tau, args.sigma_min, args.sigma_max)

    spec = try_spectral_mask(x, args, device) if args.show_spectral else None

    # numpy 化(τ 在 1/2 分辨率,上採樣到原圖便於疊圖)
    def up(t):
        return F.interpolate(t, size=(H, W), mode='bilinear',
                             align_corners=False)[0, 0].cpu().numpy()
    x_np      = x[0].permute(1, 2, 0).cpu().numpy()
    hf_np     = up(parts["hf_norm"])
    sal_np    = up(parts["sal_norm"])
    sgate_np  = up(parts["sal_gate"])
    tau_np    = up(parts["tau"])
    ls_np     = up(log2_sigma)

    n_panel = 7 if spec is None else 8
    fig, ax = plt.subplots(1, n_panel, figsize=(6 * n_panel, 6))
    ax[0].imshow(x_np); ax[0].set_title("input"); ax[0].axis("off")

    im1 = ax[1].imshow(hf_np, cmap="inferno", vmin=0, vmax=1)
    ax[1].set_title("shallow HF (高=有纹理,排天空)"); ax[1].axis("off")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)

    im2 = ax[2].imshow(sal_np, cmap="jet", vmin=0, vmax=1)
    ax[2].set_title("saliency (亮=飞机)"); ax[2].axis("off")
    plt.colorbar(im2, ax=ax[2], fraction=0.046)

    im3 = ax[3].imshow(sgate_np, cmap="gray", vmin=0, vmax=1)
    ax[3].set_title("sal gate (1-sal, 排飞机)"); ax[3].axis("off")
    plt.colorbar(im3, ax=ax[3], fraction=0.046)

    im4 = ax[4].imshow(tau_np, cmap="magma", vmin=0, vmax=1)
    ax[4].set_title("TAU (可交换性场)"); ax[4].axis("off")
    plt.colorbar(im4, ax=ax[4], fraction=0.046)

    im5 = ax[5].imshow(ls_np, cmap="viridis")
    ax[5].set_title("log2_sigma (→WD)"); ax[5].axis("off")
    plt.colorbar(im5, ax=ax[5], fraction=0.046)

    overlay = x_np.copy()
    a = np.clip(tau_np, 0, 1)[..., None]
    red = np.zeros_like(overlay); red[..., 0] = 1.0
    overlay = overlay * (1 - 0.6 * a) + red * (0.6 * a)
    ax[6].imshow(np.clip(overlay, 0, 1))
    ax[6].set_title("overlay (红=noise 强度)"); ax[6].axis("off")

    if spec is not None:
        spec_np = spec[0, 0].cpu().numpy()
        im7 = ax[7].imshow(spec_np, cmap="magma")
        ax[7].set_title("频谱 mask (对照 baseline)"); ax[7].axis("off")
        plt.colorbar(im7, ax=ax[7], fraction=0.046)

    plt.tight_layout()
    out_png = os.path.join(args.save_path, "tau_overlay.png")
    plt.savefig(out_png, dpi=110)
    plt.close()
    print(f"saved {out_png}")

    # ---- 區域統計(與 check_maskA 同一套 region 定義)----
    regions = {
        "grass": (int(H * 0.80), H,             int(W * 0.20), int(W * 0.60)),
        "plane": (int(H * 0.40), int(H * 0.60), int(W * 0.30), int(W * 0.65)),
        "sky":   (int(H * 0.05), int(H * 0.25), int(W * 0.55), int(W * 0.95)),
    }
    rmean = lambda a, b: float(a[b[0]:b[1], b[2]:b[3]].mean())
    lines = ["=== τ 可交換性場 (浅层HF能量 × (1-saliency)) ===",
             f"box_ks={args.box_ks} sal_strength={args.sal_strength} "
             f"use_deep={args.use_deep} gamma={args.gamma} "
             f"sigma=[{args.sigma_min},{args.sigma_max}]",
             f"τ 全图均值={tau_np.mean():.3f} max={tau_np.max():.3f}", "",
             "区域均值:"]
    tt = {}
    for name, box in regions.items():
        hf_c = rmean(hf_np, box); sl_c = rmean(sal_np, box); t_c = rmean(tau_np, box)
        tt[name] = t_c
        lines.append(f"[{name:6s}] shallow_hf={hf_c:.2f} saliency={sl_c:.2f} "
                     f"TAU={t_c:.3f}")
    lines.append("")
    g, p, s = tt["grass"], tt["plane"], tt["sky"]
    lines.append(f"τ: 草地={g:.3f} 飞机={p:.3f} 天空={s:.3f}")
    lines.append(f"对比: 草地/飞机={g/(p+1e-6):.1f}x 草地/天空={g/(s+1e-6):.1f}x")
    if g > 0.1 and g > p * 2 and g > s * 2:
        lines.append(">>> ✓ τ 集中在纹理区,飞机/天空被压低。")
    else:
        lines.append(">>> 需调整:")
        if g <= p * 2:
            lines.append(">>>   飞机没压下去: 升 --sal-strength;或检查 saliency 是否真高亮飞机")
        if g <= s * 2:
            lines.append(">>>   天空没压下去: 浅层 HF 问题,检查 shallow_idx")
        if g <= 0.1:
            lines.append(">>>   草地 τ 偏低: 降 gamma 或检查 robust_norm 分位")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.save_path, "report.txt"), "w") as f:
        f.write(report + "\n")
    print(f"\n结果在 {args.save_path}/")


if __name__ == "__main__":
    main(sys.argv[1:])