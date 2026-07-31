"""
检查组合 mask (方案 A): high-freq AND not-saliency
---------------------------------------------------
草地 mask = 有高频纹理(排除天空) AND 不显著(排除飞机)

  - high-freq  排除天空(天空低频 → hf=0)
  - not-saliency 排除飞机(飞机显著 → not-sal=0)
  - 交集 = 草地(高频 且 不显著)

用法：
    python check_maskA.py \
        --image /path/plane_grass.png \
        --emlnet-imagenet-path /path/res_imagenet.pth \
        --emlnet-places-path   /path/res_places.pth \
        --emlnet-decoder-path  /path/res_decoder.pth \
        --cuda --save_path maskA_check

产出：
    maskA_overlay.png : 6 联图
      [输入] [高频能量] [hf_mask] [saliency] [not_sal_mask] [组合mask叠加]
    report.txt        : 草/天/飞机三区在各 mask 的覆盖率 + 判读

判读理想：
    组合 mask → 草地≈1、飞机≈0、天空≈0
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

import emlnet.decoder as eml_decoder
import emlnet.resnet as eml_resnet


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


def high_freq_energy(target, blur_ks=15):
    gray = target.mean(1, keepdim=True)
    low = F.avg_pool2d(gray, blur_ks, stride=1, padding=blur_ks // 2)
    high = (gray - low).abs()
    high = F.avg_pool2d(high, blur_ks, stride=1, padding=blur_ks // 2)
    return high


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--emlnet-imagenet-path", required=True)
    ap.add_argument("--emlnet-places-path", required=True)
    ap.add_argument("--emlnet-decoder-path", required=True)
    ap.add_argument("--save_path", default="maskA_check")
    ap.add_argument("--blur-ks", type=int, default=15)
    ap.add_argument("--hf-percentile", type=float, default=0.5,
                    help="高频阈值:高于此分位数算纹理(排除天空)")
    ap.add_argument("--sal-percentile", type=float, default=0.5,
                    help="saliency阈值:低于此分位数算非显著(排除飞机)")
    ap.add_argument("--sal-blur", action="store_true", default=True,
                    help="对saliency做高斯模糊(和训练一致)")
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args(argv)

    device = "cuda:0" if args.cuda and torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_path, exist_ok=True)

    img = Image.open(args.image).convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    B, _, H, W = x.shape

    # --- 高频 mask (排除天空) ---
    high = high_freq_energy(x, blur_ks=args.blur_ks)
    hf_thr = high.flatten(1).quantile(args.hf_percentile, dim=1).view(B, 1, 1, 1)
    hf_mask = (high > hf_thr).float()

    # --- saliency (排除飞机) ---
    sal_model = EMLNetSaliency(
        args.emlnet_imagenet_path, args.emlnet_places_path,
        args.emlnet_decoder_path).to(device)
    saliency = sal_model(x)
    if args.sal_blur:
        saliency = TF.gaussian_blur(saliency, kernel_size=[31, 31], sigma=[5.0, 5.0])
    sal_thr = saliency.flatten(1).quantile(args.sal_percentile, dim=1).view(B, 1, 1, 1)
    not_sal_mask = (saliency < sal_thr).float()      # 不显著 = 非飞机

    # --- 组合 ---
    grass_mask = hf_mask * not_sal_mask              # 高频 且 非显著 = 草地

    # numpy
    x_np = x[0].permute(1, 2, 0).cpu().numpy()
    high_np = high[0, 0].cpu().numpy()
    hf_np = hf_mask[0, 0].cpu().numpy()
    sal_np = saliency[0, 0].cpu().numpy()
    nsal_np = not_sal_mask[0, 0].cpu().numpy()
    grass_np = grass_mask[0, 0].cpu().numpy()

    # ---- 6 联图 ----
    fig, ax = plt.subplots(1, 6, figsize=(36, 6))
    ax[0].imshow(x_np); ax[0].set_title("input"); ax[0].axis("off")
    im1 = ax[1].imshow(high_np, cmap="inferno"); ax[1].set_title("high-freq energy")
    ax[1].axis("off"); plt.colorbar(im1, ax=ax[1], fraction=0.046)
    ax[2].imshow(hf_np, cmap="gray"); ax[2].set_title("hf_mask (排天空)"); ax[2].axis("off")
    im3 = ax[3].imshow(sal_np, cmap="jet"); ax[3].set_title("saliency (亮=飞机)")
    ax[3].axis("off"); plt.colorbar(im3, ax=ax[3], fraction=0.046)
    ax[4].imshow(nsal_np, cmap="gray"); ax[4].set_title("not_sal (排飞机)"); ax[4].axis("off")
    overlay = x_np.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + 0.5 * grass_np, 0, 1)
    ax[5].imshow(overlay); ax[5].set_title("组合mask (红=草地=MSD作用)"); ax[5].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_path, "maskA_overlay.png"), dpi=110)
    plt.close()
    print("saved maskA_overlay.png")

    # ---- 三区数值 ----
    regions = {
        "grass": (int(H*0.80), H,           int(W*0.20), int(W*0.60)),
        "plane": (int(H*0.40), int(H*0.60), int(W*0.30), int(W*0.65)),
        "sky":   (int(H*0.05), int(H*0.25), int(W*0.55), int(W*0.95)),
    }
    lines = ["=== 组合 mask (high-freq AND not-saliency) ===",
             f"blur_ks={args.blur_ks} hf_pct={args.hf_percentile} "
             f"sal_pct={args.sal_percentile}",
             f"组合mask全图覆盖={100*grass_np.mean():.1f}%", ""]
    cov = {}
    for name, (y1, y2, x1, x2) in regions.items():
        hf_c = hf_np[y1:y2, x1:x2].mean()
        ns_c = nsal_np[y1:y2, x1:x2].mean()
        gm_c = grass_np[y1:y2, x1:x2].mean()
        cov[name] = gm_c
        lines.append(f"[{name:6s}] hf={100*hf_c:3.0f}%  not_sal={100*ns_c:3.0f}%  "
                     f"组合={100*gm_c:3.0f}%")

    lines.append("")
    g, p, s = cov["grass"], cov["plane"], cov["sky"]
    lines.append(f"组合mask覆盖:  草地={100*g:.0f}%  飞机={100*p:.0f}%  天空={100*s:.0f}%")
    if g > 0.5 and p < 0.3 and s < 0.3:
        lines.append(">>> ✓ 组合成功:草地被圈,飞机和天空都排除。可接进 masked MSD。")
    else:
        lines.append(">>> 需调整:")
        if g <= 0.5:
            lines.append(f">>>   草地覆盖低({100*g:.0f}%): 降 hf-percentile 或 升 sal-percentile")
        if p >= 0.3:
            lines.append(f">>>   飞机仍被圈({100*p:.0f}%): 降 sal-percentile(更严格排除显著区)")
        if s >= 0.3:
            lines.append(f">>>   天空仍被圈({100*s:.0f}%): 升 hf-percentile(更严格要求高频)")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.save_path, "report.txt"), "w") as f:
        f.write(report + "\n")
    print(f"\n结果在 {args.save_path}/  看 maskA_overlay.png 和 report.txt")


if __name__ == "__main__":
    main(sys.argv[1:])