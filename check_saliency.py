"""
检查单张图的 saliency map + 对应 σ-map / realism mask。
----------------------------------------------------------
目的：确认 EMLNet saliency 能不能正确把「草地(realism区)」和
      「飞机/天空(fidelity区)」分开。这决定 masked MSD 可不可行——
      masked MSD 需要一个 mask 让 MSD 只作用在草地。

用法：
    python check_saliency.py \
        --image /path/to/plane_grass.png \
        --emlnet-imagenet-path /path/res_imagenet.pth \
        --emlnet-places-path   /path/res_places.pth \
        --emlnet-decoder-path  /path/res_decoder.pth \
        --sigma-max 16 --p-min 0.5 \
        --cuda \
        --save_path saliency_check

产出（save_path 目录）：
    saliency_overlay.png : 5 联图
        [输入] [raw saliency] [σ-map] [realism mask] [mask 叠在输入上]
    report.txt           : 草地 vs 飞机/天空的 saliency / σ 数值，判读 mask 好不好

判读：
    - realism mask 应该: 草地=1(白), 飞机/天空=0(黑)
    - 若草地和飞机/天空在 mask 上分不开 → saliency 不适合当 mask,
      masked MSD 要换别的 mask 来源(见脚本末注释)。
"""

import os
import sys
import argparse
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# 需与训练脚本相同的 emlnet 模块可 import
import emlnet.decoder as eml_decoder
import emlnet.resnet as eml_resnet


# ---- 直接复用训练脚本里的 EMLNetSaliency ----
class EMLNetSaliency(nn.Module):
    def __init__(self, imagenet_path, places_path, decoder_path,
                 input_size=(480, 640), num_feat=5):
        super().__init__()
        self.input_size = input_size
        self.img_model = eml_resnet.resnet50(imagenet_path)
        self.pla_model = eml_resnet.resnet50(places_path)
        self.decoder_model = eml_decoder.build_decoder(
            decoder_path, input_size, num_feat, num_feat)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, image):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        _, _, H, W = image.shape
        resized = F.interpolate(image, size=self.input_size,
                                mode='bilinear', align_corners=False)
        img_feat = self.img_model(resized, decode=True)
        pla_feat = self.pla_model(resized, decode=True)
        pred = self.decoder_model([img_feat, pla_feat])
        pred = F.interpolate(pred, size=(H, W),
                             mode='bilinear', align_corners=False)
        return pred


def compute_sigma(saliency, sigma_max, p_min, apply_norm, apply_blur):
    """复刻训练脚本 get_wd_sigma_fn 的 saliency 分支。"""
    if apply_blur:
        saliency = TF.gaussian_blur(saliency, kernel_size=[31, 31], sigma=[5.0, 5.0])
    if apply_norm:
        s_min = saliency.amin(dim=(-1, -2), keepdim=True)
        s_max = saliency.amax(dim=(-1, -2), keepdim=True)
        saliency = (saliency - s_min) / (s_max - s_min + 1e-8)
    s = saliency.clamp(min=1e-6)
    s_bar = s.mean(dim=(-1, -2), keepdim=True)
    p = p_min + (1.0 - p_min) * (s / s_bar)
    sigma = sigma_max * p_min / p
    return sigma.clamp(min=1e-6)


def to_np(t):
    return t[0, 0].detach().cpu().numpy()


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--emlnet-imagenet-path", type=str,
                        default="/home/at9529/ycw.cs14/Michael/massive_activation/ICLR2024-FTIC/res_imagenet.pth")
    ap.add_argument("--emlnet-places-path", type=str,
                        default="/home/at9529/ycw.cs14/Michael/massive_activation/ICLR2024-FTIC/res_places.pth")
    ap.add_argument("--emlnet-decoder-path", type=str,
                        default="/home/at9529/ycw.cs14/Michael/massive_activation/ICLR2024-FTIC/res_decoder.pth")
    ap.add_argument("--sigma-max", type=float, default=32.0)
    ap.add_argument("--p-min", type=float, default=0.5)
    ap.add_argument("--norm", action="store_true", help="normalize saliency")
    ap.add_argument("--blur", action="store_true", help="blur saliency")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--save_path", default="saliency_check")
    # realism mask 阈值:σ 大于此值算 realism(草地)。用 log2 或线性都可,这里用线性 σ。
    ap.add_argument("--mask-mode", choices=["median", "sigma_thresh"],
                    default="median",
                    help="median: σ>中位数算realism; sigma_thresh: σ>--mask-sigma")
    ap.add_argument("--mask-sigma", type=float, default=None,
                    help="mask_mode=sigma_thresh 时的阈值(σ 单位)")
    args = ap.parse_args(argv)

    device = "cuda:0" if args.cuda and torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_path, exist_ok=True)

    # 载入图
    img = Image.open(args.image).convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)

    # saliency
    sal_model = EMLNetSaliency(
        args.emlnet_imagenet_path,
        args.emlnet_places_path,
        args.emlnet_decoder_path,
    ).to(device)
    saliency = sal_model(x)                       # (1,1,H,W)
    sigma = compute_sigma(saliency, args.sigma_max, args.p_min,
                          args.norm, args.blur)   # (1,1,H,W)

    # realism mask:σ 大(不显著)= realism = 草地
    if args.mask_mode == "median":
        thr = sigma.median()
    else:
        thr = args.mask_sigma if args.mask_sigma is not None else sigma.median()
    realism_mask = (sigma > thr).float()

    # ---- 存 5 联图 ----
    sal_np = to_np(saliency)
    sig_np = to_np(sigma)
    mask_np = to_np(realism_mask)
    x_np = x[0].permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(1, 5, figsize=(30, 6))
    ax[0].imshow(x_np); ax[0].set_title("input"); ax[0].axis("off")
    im1 = ax[1].imshow(sal_np, cmap="jet")
    ax[1].set_title("raw saliency (亮=显著=飞机)"); ax[1].axis("off")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)
    im2 = ax[2].imshow(sig_np, cmap="viridis")
    ax[2].set_title(f"σ-map (大=realism=草地)"); ax[2].axis("off")
    plt.colorbar(im2, ax=ax[2], fraction=0.046)
    ax[3].imshow(mask_np, cmap="gray")
    ax[3].set_title(f"realism mask (白=草地, thr={thr:.2f})"); ax[3].axis("off")
    # mask 叠在输入上
    overlay = x_np.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + 0.4 * mask_np, 0, 1)  # 草地染红
    ax[4].imshow(overlay); ax[4].set_title("mask overlay (红=会被MSD作用)")
    ax[4].axis("off")
    plt.tight_layout()
    out_png = os.path.join(args.save_path, "saliency_overlay.png")
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"saved {out_png}")

    # ---- 数值报告:手动指定草地/飞机/天空区域看 saliency & σ ----
    H, W = sal_np.shape
    lines = ["=== saliency / sigma region check ==="]
    lines.append(f"σ-map: min={sig_np.min():.3f} max={sig_np.max():.3f} "
                 f"median={np.median(sig_np):.3f}")
    lines.append(f"realism mask covers {100*mask_np.mean():.1f}% of pixels")
    lines.append("")

    regions = {
        "grass(下方)":   (int(H*0.80), H,            int(W*0.20), int(W*0.60)),
        "plane(中间)":   (int(H*0.40), int(H*0.60),  int(W*0.30), int(W*0.65)),
        "sky(右上)":     (int(H*0.05), int(H*0.25),  int(W*0.55), int(W*0.95)),
    }
    for name, (y1, y2, x1, x2) in regions.items():
        s_reg = sal_np[y1:y2, x1:x2].mean()
        sig_reg = sig_np[y1:y2, x1:x2].mean()
        m_reg = mask_np[y1:y2, x1:x2].mean()
        lines.append(f"[{name:12s}] saliency={s_reg:.3f}  σ={sig_reg:.3f}  "
                     f"realism_mask覆盖={100*m_reg:.0f}%")

    # 判读
    lines.append("")
    g_mask = mask_np[int(H*0.80):H, int(W*0.20):int(W*0.60)].mean()
    p_mask = mask_np[int(H*0.40):int(H*0.60), int(W*0.30):int(W*0.65)].mean()
    s_mask = mask_np[int(H*0.05):int(H*0.25), int(W*0.55):int(W*0.95)].mean()
    lines.append(f"草地 realism 覆盖={100*g_mask:.0f}%  "
                 f"飞机={100*p_mask:.0f}%  天空={100*s_mask:.0f}%")
    if g_mask > 0.6 and p_mask < 0.4:
        lines.append(">>> ✓ mask 可用:草地大部分=realism,飞机大部分=fidelity。")
        lines.append(">>>   masked MSD 可以直接用这个 mask。")
        if s_mask > 0.5:
            lines.append(">>> ⚠ 但天空也被判成 realism → MSD 会污染天空。")
            lines.append(">>>   需要额外把天空排除(见脚本末注释)。")
    else:
        lines.append(">>> ✗ mask 分不开草地和飞机/天空。saliency 不适合直接当 mask。")
        lines.append(">>>   考虑: 调 --p-min / --sigma-max, 或换 mask 来源。")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.save_path, "report.txt"), "w") as f:
        f.write(report + "\n")

    print(f"\n结果在 {args.save_path}/  看 saliency_overlay.png 和 report.txt")


if __name__ == "__main__":
    main(sys.argv[1:])

# ============================================================
# 如果 saliency 分不开草地/天空(两者都是非显著→都被判 realism),
# masked MSD 的 mask 可以改用「saliency 低 + 非天空」的组合,例如:
#
#   # 天空通常在图像上方 + 高亮度低饱和,可以加一个简单排除:
#   is_sky = (x.mean(1, keepdim=True) > 0.7)      # 高亮度
#   realism_mask = (sigma > thr).float() * (~is_sky).float()
#
# 或者干脆用「绿色通道优势」当草地检测:
#   g, r, b = x[:,1:2], x[:,0:1], x[:,2:3]
#   is_grass = ((g > r) & (g > b)).float()
#   realism_mask = is_grass * (sigma > thr).float()
#
# 但优先试 saliency,它是语义的、比颜色/位置先验稳。
# ============================================================