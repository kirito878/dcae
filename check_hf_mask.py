"""
检查 high-frequency mask 分区
------------------------------
目的：验证用「局部高频能量」当 mask 能不能正确区分：
  - 草地(高频纹理) → mask=1 (MSD 该作用)
  - 天空(低频平滑) → mask=0 (MSD 该避开)
  - 飞机(结构高频) → ??? (这是关键隐患:飞机边缘也是高频,可能被误圈)

用法：
    python check_hf_mask.py --image /path/plane_grass.png --save_path hf_check

产出：
    hf_mask_overlay.png : 输入 / 高频能量图 / mask / mask叠加(草地染红) 四联图
    report.txt          : 草/天/飞机三区的高频能量 + mask 覆盖率 + 判读

判读：
    草地 mask≈1、天空 mask≈0 → 高低频能分开纹理/平滑 ✓
    飞机 mask 也≈1 → 飞机结构高频被误圈,MSD 会污染飞机,需加 saliency 排除
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


def high_freq_energy(target, blur_ks=15):
    """局部高频能量:原图 - 局部低通,取绝对值,再局部平滑。"""
    gray = target.mean(1, keepdim=True)
    low = F.avg_pool2d(gray, blur_ks, stride=1, padding=blur_ks // 2)
    high = (gray - low).abs()
    high = F.avg_pool2d(high, blur_ks, stride=1, padding=blur_ks // 2)
    return high  # (B,1,H,W)


def make_mask(high, thresh_percentile=0.5):
    B = high.shape[0]
    thr = high.flatten(1).quantile(thresh_percentile, dim=1).view(B, 1, 1, 1)
    return (high > thr).float(), thr


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--save_path", default="hf_check")
    ap.add_argument("--blur-ks", type=int, default=15,
                    help="局部高频的窗口大小,大=看更大尺度的纹理")
    ap.add_argument("--percentile", type=float, default=0.5,
                    help="阈值分位数,0.5=中位数以上算纹理区")
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args(argv)

    device = "cuda:0" if args.cuda and torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_path, exist_ok=True)

    img = Image.open(args.image).convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)

    high = high_freq_energy(x, blur_ks=args.blur_ks)
    mask, thr = make_mask(high, args.percentile)

    x_np = x[0].permute(1, 2, 0).cpu().numpy()
    high_np = high[0, 0].cpu().numpy()
    mask_np = mask[0, 0].cpu().numpy()
    H, W = mask_np.shape

    # ---- 四联图 ----
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))
    ax[0].imshow(x_np); ax[0].set_title("input"); ax[0].axis("off")
    im1 = ax[1].imshow(high_np, cmap="inferno")
    ax[1].set_title("high-freq energy (亮=纹理)"); ax[1].axis("off")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)
    ax[2].imshow(mask_np, cmap="gray")
    ax[2].set_title(f"mask (白=纹理区, thr={thr.item():.4f})"); ax[2].axis("off")
    overlay = x_np.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + 0.4 * mask_np, 0, 1)
    ax[3].imshow(overlay); ax[3].set_title("overlay (红=会被MSD作用)")
    ax[3].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_path, "hf_mask_overlay.png"), dpi=120)
    plt.close()
    print(f"saved hf_mask_overlay.png")

    # ---- 三区数值 ----
    regions = {
        "grass(下方)": (int(H*0.80), H,           int(W*0.20), int(W*0.60)),
        "plane(中间)": (int(H*0.40), int(H*0.60), int(W*0.30), int(W*0.65)),
        "sky(右上)":   (int(H*0.05), int(H*0.25), int(W*0.55), int(W*0.95)),
    }
    lines = ["=== high-freq mask 分区检查 ===",
             f"blur_ks={args.blur_ks}  percentile={args.percentile}  "
             f"thr={thr.item():.4f}",
             f"mask 全图覆盖率={100*mask_np.mean():.1f}%", ""]
    cover = {}
    for name, (y1, y2, x1, x2) in regions.items():
        e = high_np[y1:y2, x1:x2].mean()
        m = mask_np[y1:y2, x1:x2].mean()
        cover[name.split("(")[0]] = m
        lines.append(f"[{name:12s}] hf_energy={e:.4f}  mask覆盖={100*m:.0f}%")

    lines.append("")
    g, p, s = cover["grass"], cover["plane"], cover["sky"]
    lines.append(f"草地={100*g:.0f}%  飞机={100*p:.0f}%  天空={100*s:.0f}%")
    if g > 0.6 and s < 0.3:
        lines.append(">>> ✓ 草地=纹理区、天空=平滑区,高低频成功分开这两者。")
        if p > 0.5:
            lines.append(">>> ⚠ 但飞机也被判纹理区(结构高频)→ MSD 会污染飞机。")
            lines.append(">>>   解法:mask = high_freq AND (not saliency),用saliency排除飞机。")
        else:
            lines.append(">>> ✓ 飞机大部分未进 mask,high-freq mask 单独就够用。")
    else:
        lines.append(">>> ✗ 草地/天空没分开。调 --blur-ks 或 --percentile 重试。")
        lines.append(">>>   草地覆盖低→降 percentile; 天空覆盖高→升 percentile 或增 blur-ks。")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.save_path, "report.txt"), "w") as f:
        f.write(report + "\n")
    print(f"\n结果在 {args.save_path}/  看 hf_mask_overlay.png 和 report.txt")


if __name__ == "__main__":
    main(sys.argv[1:])