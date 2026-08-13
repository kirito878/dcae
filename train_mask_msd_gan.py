import argparse
import math
import os
import random
import sys
import time
from datetime import datetime
from comet_ml import Experiment

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from compressai.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from pytorch_msssim import ms_ssim
import lpips
import vision_aided_loss

from my_utils.Meter import AverageMeterTEST, AverageMeterTRAIN
from models.dcae_gate import DCAE

import emlnet.decoder as eml_decoder
import emlnet.resnet as eml_resnet
from mask_utils import grass_mask_fn
from torchvision.utils import save_image
from PIL import Image


def gate_bce_loss(gates, mask, pos_weight=None, soften=0.0):
    """
    用 grass_mask 逐像素監督 gate(蒸餾)。
    training 時用 mask 教,推論時 gate 自己會了 —— decoder 不需要 mask。

    gates : (B,K,h,w) 所有注入點的 gate,已過 sigmoid,值域 (0,1)
    mask  : (B,1,H,W) grass_mask,硬 0/1
    soften: >0 時做 label smoothing,target 落在 [soften, 1-soften]。
            硬 0/1 target 會把 sigmoid 推向飽和(logit -> ±inf),
            之後 gate 就再也動不了。給 0.05 保留一點梯度。
    """
    m = F.interpolate(mask, size=gates.shape[-2:],
                      mode='bilinear', align_corners=False).clamp(0, 1)
    if soften > 0:
        m = m * (1.0 - 2 * soften) + soften
    m = m.expand_as(gates)

    g = gates.clamp(1e-6, 1 - 1e-6)      # 防 log(0)

    if pos_weight is None:
        loss = F.binary_cross_entropy(g, m)
    else:
        # mask cover 只有 ~18%,正樣本稀少;加權避免 gate 直接全關拿低 loss
        w = 1.0 + (pos_weight - 1.0) * m
        loss = (F.binary_cross_entropy(g, m, reduction='none') * w).mean()

    # 監控用:mask 內外的平均開度
    m_bin = (m > 0.5).float()
    cov_in = ((gates * m_bin).sum() / (m_bin.sum() + 1e-6)).item()
    cov_out = ((gates * (1 - m_bin)).sum() / ((1 - m_bin).sum() + 1e-6)).item()
    return loss, cov_in, cov_out


def calc_tv_loss(x):
    """計算 Total Variation Loss"""
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w


# ══════════════════════════════════════════════════════
# GAN 工具函式
# ══════════════════════════════════════════════════════
def to_disc_input(x):
    """
    [0,1] -> [-1,1]。
    dataset 只有 ToTensor(沒有 Normalize),影像在 [0,1];
    但 vision_aided_loss 的 CLIP/DINO backbone 期望 [-1,1]。
    忘了轉的話 D 一開始就在看錯誤的分佈,G 會被亂推。
    """
    return (x * 2.0 - 1.0).clamp(-1.0, 1.0)


def set_disc_requires_grad(net_disc, flag):
    """
    G step 時把 D 的參數關掉 requires_grad,避免 G 的 backward
    在 D 上累積無用梯度(梯度仍然會穿過 D 傳回 x_hat,這是要的)。
    cv_ensemble 是 frozen backbone,永遠不訓練。
    """
    if net_disc is None:
        return
    for p in net_disc.parameters():
        p.requires_grad_(flag)
    net_disc.cv_ensemble.requires_grad_(False)


def _to_comet_img(t):
    """(C,H,W) tensor in [0,1] -> (H,W,C) uint8 numpy for comet."""
    t = t.detach().cpu().clamp(0, 1)
    if t.dim() == 3 and t.size(0) == 1:      # 單通道 gate -> (H,W)
        arr = (t[0].numpy() * 255).astype('uint8')      # (H,W) 灰階
    else:                                    # 三通道 input -> (H,W,3)
        arr = (t.permute(1, 2, 0).numpy() * 255).astype('uint8')
    return arr


# ══════════════════════════════════════════════════════
# EML-net saliency wrapper + WD sigma factory
# ══════════════════════════════════════════════════════
class EMLNetSaliency(nn.Module):
    """Frozen EML-net saliency model."""

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


def get_wd_sigma_fn(mode="static", saliency_model=None, sigma_max=16.0,
                    p_min=0.5, apply_norm=False, apply_blur=False):
    """Return a fn that maps target_images (B,3,H,W) -> sigma map (B,1,H,W)."""
    def compute_wd_sigma(target_images):
        N, _, H, W = target_images.shape
        if mode == "static" or saliency_model is None:
            return torch.full((N, 1, H, W), sigma_max,
                              device=target_images.device,
                              dtype=target_images.dtype)
        elif mode == "saliency":
            with torch.no_grad():
                saliency = saliency_model(target_images)
            if apply_blur:
                saliency = TF.gaussian_blur(
                    saliency, kernel_size=[31, 31], sigma=[5.0, 5.0])
            if apply_norm:
                s_min = saliency.amin(dim=(-1, -2), keepdim=True)
                s_max = saliency.amax(dim=(-1, -2), keepdim=True)
                saliency = (saliency - s_min) / (s_max - s_min + 1e-8)
            s = saliency.clamp(min=1e-6)
            s_bar = s.mean(dim=(-1, -2), keepdim=True)
            p = p_min + (1.0 - p_min) * (s / s_bar)
            sigma = sigma_max * p_min / p
            return sigma.clamp(min=1e-6)
        else:
            raise ValueError(f"Unknown wd sigma mode: {mode}")
    return compute_wd_sigma


def log_metrics(experiment, metrics: dict, step: int):
    if experiment is not None:
        experiment.log_metrics(metrics, step=step)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ══════════════════════════════════════════════════════
# WD loss 載入
# ══════════════════════════════════════════════════════
wd_fn = None
try:
    from wa_wd import VGG16WaveletWassersteinDistortion
    wd_fn = VGG16WaveletWassersteinDistortion(
        num_levels=4,                          # 6 → 4
        dwt_levels=1,
        learnable_weights=False,
        sigma_offsets=(-0.5, 0.0, 0.0, 0.5),   # 收窄，配合 clamp
        ll_weight_boost=0.3,
    ).to(DEVICE).eval()
    for p in wd_fn.parameters():
        p.requires_grad_(False)
    print("WD loaded.")
except ImportError:
    print("WD not available.")


# ══════════════════════════════════════════════════════
# Rate-Distortion Perceptual Loss
# ══════════════════════════════════════════════════════
class RateDistortionPerceptualLoss(nn.Module):
    """
    Loss = mse_weight * MSE + wd_weight * WD + lpips_weight * LPIPS
           + lmbda * bpp + tv_weight * TV
    (GAN loss 在訓練迴圈外加,因為需要 D 的 forward)
    """

    def __init__(self, lmbda=1e-2, wd_weight=0.0, mse_weight=1.0,
                 lpips_weight=0.0, wd_sigma_fn=None, wd_fn=None,
                 lpips_fn=None, tv_weight=0.000):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.lmbda = lmbda
        self.wd_weight = wd_weight
        self.mse_weight = mse_weight
        self.lpips_weight = lpips_weight
        self.wd_sigma_fn = wd_sigma_fn
        self.wd_fn = wd_fn
        self.lpips_fn = lpips_fn
        self.tv_weight = tv_weight

    def set_weights(self, mse_weight=None, wd_weight=None, lpips_weight=None):
        if mse_weight is not None:
            self.mse_weight = mse_weight
        if wd_weight is not None:
            self.wd_weight = wd_weight
        if lpips_weight is not None:
            self.lpips_weight = lpips_weight

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["ssim"] = 1 - ms_ssim(output["x_hat"], target, data_range=1.)

        # WD loss
        if self.wd_sigma_fn is not None and self.wd_fn is not None and self.wd_weight > 0:
            wd_sigma = self.wd_sigma_fn(target)
            log2_sigma = torch.log2(wd_sigma)
            wd_loss = self.wd_fn(
                output["x_hat"], target, log2_sigma, num_scales=2).mean()
            out["wd_loss"] = wd_loss
        else:
            out["wd_loss"] = torch.tensor(0.0, device=target.device)

        # LPIPS loss
        if self.lpips_fn is not None and self.lpips_weight > 0:
            x_hat_norm = 2.0 * output["x_hat"] - 1.0
            target_norm = 2.0 * target - 1.0
            out["lpips_loss"] = self.lpips_fn(x_hat_norm, target_norm).mean()
        else:
            out["lpips_loss"] = torch.tensor(0.0, device=target.device)

        tv_loss = calc_tv_loss(output["x_hat"])
        out["loss"] = (
            self.mse_weight * out["mse_loss"]
            + self.wd_weight * out["wd_loss"]
            + self.lpips_weight * out["lpips_loss"]
            + self.lmbda * out["bpp_loss"]
            + tv_loss * self.tv_weight
        )
        return out


# ══════════════════════════════════════════════════════
# WD Warmup Scheduler
# ══════════════════════════════════════════════════════
class WDWarmupScheduler:
    def __init__(self, warmup_iters=0, transition_iters=15000,
                 initial_mse_weight=1.0, final_mse_weight=0.0,
                 final_wd_weight=0.05):
        self.warmup_iters = warmup_iters
        self.transition_iters = transition_iters
        self.initial_mse_weight = initial_mse_weight
        self.final_mse_weight = final_mse_weight
        self.final_wd_weight = final_wd_weight

    def get_weights(self, iteration):
        if iteration < self.warmup_iters:
            return self.initial_mse_weight, 0.0
        elif iteration < self.warmup_iters + self.transition_iters:
            t = (iteration - self.warmup_iters) / self.transition_iters
            mse_w = self.initial_mse_weight * \
                (1 - t) + self.final_mse_weight * t
            wd_w = self.final_wd_weight * t
            return mse_w, wd_w
        else:
            return self.final_mse_weight, self.final_wd_weight


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    trainable = {
        n for n, p in net.named_parameters()
        if p.requires_grad and not n.endswith(".quantiles")
    }
    aux_parameters = {
        n for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    params_dict = dict(net.named_parameters())

    # gate 參數單獨分組:參數量極少(每個 gate conv 只輸出 1 channel),
    # Adam 的自適應 lr 會讓它們在幾十步內劇烈移動 -> gate 崩塌。
    # 用較低 lr 讓它跟著網路慢慢走,而不是被 WD/LPIPS 一腳踩死。
    gate_names = sorted(n for n in trainable if ".gate_" in n)
    other_names = sorted(trainable - set(gate_names))

    gate_lr_scale = hasattr(args, "gate_lr_scale") and args.gate_lr_scale or 1
    gate_lr = args.learning_rate * gate_lr_scale
    param_groups = [
        {"params": [params_dict[n] for n in other_names],
         "lr": args.learning_rate},
    ]
    if gate_names:
        param_groups.append({
            "params": [params_dict[n] for n in gate_names],
            "lr": gate_lr,
        })

    print(f"[Optimizer] Trainable params: {len(trainable)} "
          f"(gate: {len(gate_names)} @ lr={gate_lr:.2e}, "
          f"other: {len(other_names)} @ lr={args.learning_rate:.2e})")

    optimizer = optim.AdamW(param_groups)

    aux_optimizer = None
    if aux_parameters:
        aux_optimizer = optim.AdamW(
            (params_dict[n] for n in sorted(aux_parameters)),
            lr=args.aux_learning_rate,
        )
    return optimizer, aux_optimizer


def _is_finite_tensor(value):
    return torch.isfinite(value).all().item()


def _criterion_is_finite(out_criterion):
    keys = ["loss", "bpp_loss", "mse_loss", "wd_loss", "lpips_loss"]
    return all(_is_finite_tensor(out_criterion[k]) for k in keys)


# ══════════════════════════════════════════════════════
# 載入預訓練權重
# ══════════════════════════════════════════════════════
def load_pretrained_state_dict(model, ckpt_path):
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model_state = model.state_dict()
    new_state = {}
    skipped = []

    for k, v in checkpoint.items():
        key = k[len("module."):] if k.startswith("module.") else k
        if key in model_state:
            new_state[key] = v
        else:
            skipped.append(key)

    missing_keys = [k for k in model_state.keys() if k not in new_state]

    msg = model.load_state_dict(new_state, strict=False)
    if msg is not None and hasattr(msg, "missing_keys"):
        missing_keys = list(msg.missing_keys)

    print(f"Loaded {len(new_state)} keys from checkpoint.")
    print(f"Missing keys: {len(missing_keys)}")
    if len(missing_keys) < 20:
        for missing_key in missing_keys:
            print(f"  Missing: {missing_key}")
    if skipped:
        print(f"Skipped {len(skipped)} keys not in model")
        if len(skipped) < 20:
            for s in skipped:
                print(f"  Skipped: {s}")


def test_epoch(epoch, test_dataloader, model, criterion, experiment=None):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeterTEST()
    bpp_loss = AverageMeterTEST()
    mse_loss = AverageMeterTEST()
    ssim_loss = AverageMeterTEST()
    wd_loss = AverageMeterTEST()
    lpips_loss = AverageMeterTEST()
    aux_loss = AverageMeterTEST()
    bpp_z_loss = AverageMeterTEST()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)
            N, _, H, W = d.size()
            num_pixels = N * H * W

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])

            if 'z' in out_net["likelihoods"]:
                bpp_z = (torch.log(out_net["likelihoods"]['z']).sum()
                         / (-math.log(2) * num_pixels))
            else:
                bpp_z = torch.tensor(0.0, device=device)

            loss.update(out_criterion["loss"])
            mse_losss = out_criterion["mse_loss"]
            ssim_losss = -10 * math.log10(out_criterion['ssim'])

            psnr = 10 * (torch.log(1 / mse_losss) / np.log(10))
            mse_loss.update(psnr)
            bpp_z_loss.update(bpp_z)
            ssim_loss.update(ssim_losss)
            wd_loss.update(out_criterion["wd_loss"].item())
            lpips_loss.update(out_criterion["lpips_loss"].item())

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tPSNR: {mse_loss.avg:.3f} |"
        f"\tSSIM: {ssim_loss.avg:.4f} |"
        f"\tWD loss: {wd_loss.avg:.4f} |"
        f"\tLPIPS loss: {lpips_loss.avg:.4f} |"
        f"\tBpp loss: {bpp_loss.avg:.4f} |"
        f"\tBpp z loss: {bpp_z_loss.avg:.4f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )
    log_metrics(experiment, {
        "val/bpp": bpp_loss.avg,
        "val/psnr": mse_loss.avg,
        "val/wd": wd_loss.avg,
        "val/lpips": lpips_loss.avg,
    }, step=epoch)
    return loss.avg


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="DCAE training script with WD + LPIPS + TV + GAN + grad accumulation.")
    parser.add_argument("-d", "--dataset", type=str,
                        required=True, help="Training dataset")
    parser.add_argument("-e", "--epochs", default=100, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float)
    parser.add_argument("-n", "--num-workers", type=int, default=16)
    parser.add_argument("--lambda", dest="lmbda", type=float, default=1e-2,
                        help="Rate weight in R-D loss")
    parser.add_argument("--wd-weight", type=float, default=1.0,
                        help="Final WD weight")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-batch-size", type=int, default=1)
    parser.add_argument("--aux-learning-rate", default=1e-3, type=float)
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--save_path", type=str, default="ckpt/model.pth.tar")
    parser.add_argument("--seed", type=float)
    parser.add_argument("--clip_max_norm", default=1.0, type=float)
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--comet", action="store_true")
    parser.add_argument("--comet-name", type=str, default="")
    parser.add_argument("--skip-loss-threshold", type=float, default=4.0,
                        help="Skip update when total loss exceeds this value")
    parser.add_argument("--val-interval", type=int, default=1000,
                        help="每 N 次『參數更新』驗證一次(不是 micro-batch)")

    # ---- WD σ-map 設定 ----
    parser.add_argument("--wd-sigma-mode", type=str, default="static",
                        choices=["static", "saliency"],
                        help="Sigma map mode for WD")
    parser.add_argument("--wd-sigma-max", type=float, default=16.0,
                        help="Max sigma value for WD")
    parser.add_argument("--wd-sigma-p-min", type=float, default=0.5,
                        help="Min p value for WD sigma map")
    parser.add_argument("--emlnet-imagenet-path", type=str,
                        default="/home/at9529/ycw.cs14/Michael/massive_activation/ICLR2024-FTIC/res_imagenet.pth")
    parser.add_argument("--emlnet-places-path", type=str,
                        default="/home/at9529/ycw.cs14/Michael/massive_activation/ICLR2024-FTIC/res_places.pth")
    parser.add_argument("--emlnet-decoder-path", type=str,
                        default="/home/at9529/ycw.cs14/Michael/massive_activation/ICLR2024-FTIC/res_decoder.pth")
    parser.add_argument("--emlnet-norm", action="store_true",
                        help="Normalize EMLNet saliency map")
    parser.add_argument("--emlnet-blur", action="store_true",
                        help="Apply Gaussian blur to EMLNet saliency map")

    # ---- WD warmup schedule ----
    parser.add_argument("--mse-weight", type=float, default=1.0,
                        help="Initial weight for MSE loss (during warmup)")
    parser.add_argument("--final-mse-weight", type=float, default=0.0,
                        help="Final weight for MSE loss (after transition)")
    parser.add_argument("--warmup-iters", type=int, default=0,
                        help="Iterations for pure MSE warmup phase")
    parser.add_argument("--transition-iters", type=int, default=15000,
                        help="Iterations for MSE→WD linear transition phase")
    parser.add_argument("--skip-warmup", action="store_true",
                        help="Skip warmup, use final weights from start")
    parser.add_argument("--msd-weight", type=float, default=0.0,
                        help="Mode-seeking diversity loss weight")

    # ---- MSD grass mask (路線 A 保留,路線 B 下面用 gate) ----
    parser.add_argument("--msd-mask-blur-ks", type=int, default=15,
                        help="局部高频窗口大小(草地纹理检测)")
    parser.add_argument("--msd-mask-hf-pct", type=float, default=0.5,
                        help="高频阈值分位数(排天空,越高越严格)")
    parser.add_argument("--msd-mask-sal-pct", type=float, default=0.5,
                        help="saliency阈值分位数(排飞机,越低越严格)")
    parser.add_argument("--msd-target-div", type=float, default=0.02,
                        help="MSD 目標多樣性,delta_x 達標即不再推大")
    parser.add_argument("--msd-mask-aniso-ks", type=int, default=7,
                        help="anisotropy 局部窗口大小")
    parser.add_argument("--msd-mask-aniso-pct", type=float, default=0.5,
                        help="anisotropy 阈值分位数(排柵欄/牆,越低越严格)")
    parser.add_argument("--msd-mask-use-aniso", action="store_true", default=True,
                        help="启用 anisotropy 条件(路线A)")

    # ---- 路線 B: gate-weighted MSD ----
    parser.add_argument("--msd-use-gate", action="store_true", default=False,
                        help="路線B:用 decoder gate 加權 MSD(取代硬 mask),"
                             "gate 由 WD/LPIPS 決定,MSD 端 detach 不動 gate")
    parser.add_argument("--msd-gate-min-sum", type=float, default=1.0,
                        help="gate 總和低於此值時跳過 MSD(gate 幾乎全關,避免除零)")

    parser.add_argument("--lpips-weight", type=float, default=0.0,
                        help="LPIPS perceptual loss weight")
    parser.add_argument("--lpips-net", type=str, default="vgg",
                        choices=["vgg", "alex"],
                        help="LPIPS backbone network")
    parser.add_argument("--tv-weight", type=float, default=0.0,
                        help="Total Variation loss weight")

    # ---- 對比式 MSD:非 mask 區的 diversity 要壓小(防 leak) ----
    parser.add_argument("--msd-contrast-weight", type=float, default=1.0,
                        help="對比項權重 lambda_out:非草地(或非gate)區的 diversity 懲罰。"
                             "0=關閉對比,退回純 div_in。越大越嚴格壓機身/天空 leak。")
    parser.add_argument("--msd-mask-blur-sigma", type=float, default=2.0,
                        help="grass_mask 柔化用高斯 sigma。硬 0/1 mask 邊界會造成 diversity 接縫,"
                             "柔化後草地/機身交界平滑過渡。0=不柔化。")
    parser.add_argument("--msd-out-eps", type=float, default=0.0,
                        help="非 mask 區允許的 diversity 容忍值。div_out 超過此值才罰,"
                             "0=直接壓到最小。給小正值(如0.002)可留一點自然變異。")

    # ---- gate BCE 監督(用 grass_mask 蒸餾進 gate)----
    parser.add_argument("--gate-weight", type=float, default=0.0,
                        help="gate BCE 監督權重。>0 才啟用。")
    parser.add_argument("--gate-pos-weight", type=float, default=4.0,
                        help="正樣本(mask 內)加權。mask cover ~18%,"
                             "不加權的話 gate 全關就能拿到低 BCE。")
    parser.add_argument("--gate-soften", type=float, default=0.05,
                        help="label smoothing,target 落在 [s, 1-s]。"
                             "防止 sigmoid 飽和後 gate 死掉。0=硬 0/1")
    parser.add_argument("--gate-lr-scale", type=float, default=0.1,
                        help="gate 參數 lr 倍率(相對 --learning-rate)")

    # ══════════════════════════════════════════════════
    # GAN loss (vision-aided discriminator)
    # ══════════════════════════════════════════════════
    parser.add_argument("--gan-weight", type=float, default=0.0,
                        help="Generator adversarial loss 權重。>0 才啟用 GAN。"
                             "建議 0.05 起手,最多不超過 0.5。")
    parser.add_argument("--disc-lr", type=float, default=2e-5,
                        help="Discriminator 的 lr(通常比 G 低)")
    parser.add_argument("--disc-cv-type", type=str, default="clip",
                        help="vision_aided_loss backbone: clip / dino / dinov3 / "
                             "'clip+dino'(組合,更吃 VRAM)")
    parser.add_argument("--disc-loss-type", type=str,
                        default="multilevel_sigmoid_s",
                        help="multilevel_sigmoid_s(最穩) / multilevel_sigmoid / multilevel(hinge)")
    parser.add_argument("--disc-start-iter", type=int, default=0,
                        help="第幾個 micro-batch 開始『訓練 D』。D 可以先自己學,G 還感覺不到。")
    parser.add_argument("--gan-start-iter", type=int, default=1500,
                        help="第幾個 micro-batch 開始把 adv loss 加進 G。"
                             "應該 > --disc-start-iter,讓 D 先學會分辨。"
                             "注意:用 --grad-accum N 時這個值要乘 N。")
    parser.add_argument("--gan-warmup-iters", type=int, default=2000,
                        help="gan_weight 從 0 線性爬到目標值的 micro-batch 數。0=直接全開。")
    parser.add_argument("--disc-every", type=int, default=1,
                        help="每 N 步更新一次 D。grad_accum>1 時會被強制設為 1。")
    parser.add_argument("--disc-clip-norm", type=float, default=1.0,
                        help="D 的 grad clip norm")
    parser.add_argument("--disc-checkpoint", type=str, default=None,
                        help="接續訓練時載入 D 的權重")

    # ══════════════════════════════════════════════════
    # 梯度累積
    # ══════════════════════════════════════════════════
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="梯度累積步數。effective batch = batch_size * grad_accum。"
                             "GAN 階段 D 需要大 batch 才有穩定的判別訊號"
                             "(論文 Stage 2 用 32)。")

    args = parser.parse_args(argv)
    return args


elapsed, data_times, losses, psnrs, bpps, bpp_ys, bpp_zs, mse_losses, wd_losses, aux_losses, msd_losses = \
    [AverageMeterTRAIN(2000) for _ in range(11)]
lpips_losses = AverageMeterTRAIN(2000)
adv_losses = AverageMeterTRAIN(2000)      # G 的 adversarial loss
disc_losses = AverageMeterTRAIN(2000)     # D 的 real+fake loss


def main(argv):
    args = parse_args(argv)
    print(args)

    # ══════════════════════════════════════════════════════
    # 1. Saliency model
    # ══════════════════════════════════════════════════════
    saliency_model = None
    if args.wd_sigma_mode == "saliency":
        if not (args.emlnet_imagenet_path and args.emlnet_places_path
                and args.emlnet_decoder_path):
            raise ValueError("wd_sigma_mode=saliency requires EMLNet paths.")
        saliency_model = EMLNetSaliency(
            args.emlnet_imagenet_path,
            args.emlnet_places_path,
            args.emlnet_decoder_path,
        ).to(DEVICE)
        print(f"EMLNetSaliency loaded "
              f"(norm={args.emlnet_norm}, blur={args.emlnet_blur}).")
    else:
        print("EMLNetSaliency: disabled (wd_sigma_mode=static).")

    get_wd_sigma = get_wd_sigma_fn(
        mode=args.wd_sigma_mode,
        saliency_model=saliency_model,
        sigma_max=args.wd_sigma_max,
        p_min=args.wd_sigma_p_min,
        apply_norm=args.emlnet_norm,
        apply_blur=args.emlnet_blur,
    )
    print(f"WD σ-map: mode={args.wd_sigma_mode}, σ_max={args.wd_sigma_max}, "
          f"p_min={args.wd_sigma_p_min}")

    # ══════════════════════════════════════════════════════
    # 1.2 MSD 专用 saliency model(路線 A 用;路線 B 用 gate 時可不需要)
    # ══════════════════════════════════════════════════════
    msd_saliency_model = None
    need_grass_mask = (args.msd_weight > 0 and not args.msd_use_gate) \
        or args.gate_weight > 0
    if need_grass_mask:
        if saliency_model is not None:
            msd_saliency_model = saliency_model
            print("[MSD] reuse WD's saliency model for grass mask.")
        else:
            msd_saliency_model = EMLNetSaliency(
                args.emlnet_imagenet_path,
                args.emlnet_places_path,
                args.emlnet_decoder_path,
            ).to(DEVICE)
            msd_saliency_model.eval()
            for p in msd_saliency_model.parameters():
                p.requires_grad = False
            print("[MSD] independent saliency model loaded for grass mask.")
    elif args.msd_weight > 0 and args.msd_use_gate:
        print("[MSD] route B: gate-weighted MSD, no grass mask needed.")

    # ══════════════════════════════════════════════════════
    # 1.5 LPIPS model
    # ══════════════════════════════════════════════════════
    if args.lpips_weight > 0:
        lpips_fn = lpips.LPIPS(net=args.lpips_net).to(DEVICE)
        for p in lpips_fn.parameters():
            p.requires_grad = False
        lpips_fn.eval()
        print(f"LPIPS loaded (net={args.lpips_net}, weight={args.lpips_weight}).")
    else:
        lpips_fn = None
        print("LPIPS: disabled.")

    # ══════════════════════════════════════════════════════
    # 2. Comet
    # ══════════════════════════════════════════════════════
    if args.comet:
        experiment = Experiment(
            api_key="1Ib940gnzS84GYrUu7LPpHvNi",
            project_name="DCAE_finetune",
            workspace="kirito878"
        )
        if args.comet_name:
            experiment.set_name(args.comet_name)
        experiment.log_parameters(vars(args))
        experiment.log_code(
            file_name="/home/at9529/ycw.cs14/Michael/massive_activation/DCAE/models/dcae_gate.py"
        )
        key = experiment.get_key()
        print(f"[Comet] key={experiment.get_key()}  url={experiment.url}")
    else:
        experiment = None

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # ══════════════════════════════════════════════════════
    # 3. Dataset
    # ══════════════════════════════════════════════════════
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )
    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    train_dataset = ImageFolder(
        args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(
        "/home/at9529/ycw.cs14/dataset/TestImage", split="Kodak", transform=test_transforms)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, pin_memory=(device == "cuda"),
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers,
        shuffle=False, pin_memory=(device == "cuda"),
    )

    # ══════════════════════════════════════════════════════
    # 4. Model
    # ══════════════════════════════════════════════════════
    net = DCAE()
    net = net.to(device)
    print(f"{count_params(net):.2f}M parameters")

    # ══════════════════════════════════════════════════════
    # 4.5 Discriminator (vision-aided GAN)
    # ══════════════════════════════════════════════════════
    net_disc = None
    optimizer_disc = None
    if args.gan_weight > 0:
        net_disc = vision_aided_loss.Discriminator(
            cv_type=args.disc_cv_type,
            output_type='conv_multi_level',
            loss_type=args.disc_loss_type,
            device=device,
        )
        net_disc = net_disc.to(device)
        net_disc.cv_ensemble.requires_grad_(False)   # frozen pretrained backbone
        net_disc.train()
        # timm 新版的 fused attention 在某些環境會炸,跟參考實作一樣關掉
        for _n, _m in net_disc.named_modules():
            if "attn" in _n and hasattr(_m, "fused_attn"):
                _m.fused_attn = False

        if args.disc_checkpoint and os.path.exists(args.disc_checkpoint):
            dck = torch.load(args.disc_checkpoint, map_location="cpu")
            dsd = dck.get("disc_state_dict", dck)
            net_disc.load_state_dict(dsd, strict=False)
            print(f"[GAN] Discriminator resumed from {args.disc_checkpoint}")

        disc_params = [p for p in net_disc.parameters() if p.requires_grad]
        optimizer_disc = optim.AdamW(disc_params, lr=args.disc_lr)
        print(f"[GAN] D loaded: cv_type={args.disc_cv_type}, "
              f"loss={args.disc_loss_type}, "
              f"trainable={sum(p.numel() for p in disc_params)/1e6:.2f}M, "
              f"lr={args.disc_lr}")
        print(f"[GAN] weight={args.gan_weight}, disc_start={args.disc_start_iter}, "
              f"gan_start={args.gan_start_iter}, warmup={args.gan_warmup_iters}, "
              f"disc_every={args.disc_every}")
    else:
        print("[GAN] disabled (gan_weight=0).")

    # ══════════════════════════════════════════════════════
    # 5. Loss + Warmup Scheduler
    # ══════════════════════════════════════════════════════
    criterion = RateDistortionPerceptualLoss(
        lmbda=args.lmbda,
        wd_weight=args.wd_weight,
        mse_weight=args.mse_weight,
        lpips_weight=args.lpips_weight,
        wd_sigma_fn=get_wd_sigma,
        wd_fn=wd_fn,
        lpips_fn=lpips_fn,
        tv_weight=args.tv_weight,
    ).to(device)

    wd_scheduler = WDWarmupScheduler(
        warmup_iters=args.warmup_iters,
        transition_iters=args.transition_iters,
        initial_mse_weight=args.mse_weight,
        final_mse_weight=args.final_mse_weight,
        final_wd_weight=args.wd_weight,
    )

    last_epoch = 0
    iterations = -1
    updates = 0                 # 真正的參數更新次數(≠ iterations)
    best_loss = float("inf")

    # ══════════════════════════════════════════════════════
    # 6. Checkpoint 載入
    # ══════════════════════════════════════════════════════
    if args.checkpoint:
        load_pretrained_state_dict(net, args.checkpoint)
        try:
            ckpt_raw = torch.load(args.checkpoint, map_location="cpu")
            if isinstance(ckpt_raw, dict) and "iterations" in ckpt_raw:
                iterations = int(ckpt_raw["iterations"])
                print(f"[Resume] iterations set to {iterations}")
                iterations = 0
        except Exception as e:
            print(f"[Resume] could not read iterations: {e}")

    ckpt_name = args.comet_name if args.comet_name else "default"
    exp_key = experiment.get_key() if experiment is not None else "nocomet"
    ckpt_epoch_dir = os.path.join("ckpt_epoch", ckpt_name, exp_key)
    os.makedirs(ckpt_epoch_dir, exist_ok=True)
    print(f"[Ckpt] Per-val checkpoints -> {ckpt_epoch_dir}")

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.3, patience=16)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    if args.test:
        test_epoch(0, test_dataloader, net, criterion, experiment=experiment)
        exit(-1)

    if args.skip_warmup:
        print("[Warmup] Skipping, using final weights from start.")
        criterion.set_weights(
            mse_weight=args.final_mse_weight,
            wd_weight=args.wd_weight,
        )

    print(f"[Training] λ={args.lmbda}, wd_weight={args.wd_weight}, "
          f"mse_weight={args.mse_weight} → {args.final_mse_weight}")
    print(f"[Training] warmup={args.warmup_iters}, transition={args.transition_iters}")
    if args.msd_weight > 0:
        mode = "gate-weighted (route B)" if args.msd_use_gate else "grass-mask (route A)"
        print(f"[MSD] mode={mode}, target_div={args.msd_target_div}, weight={args.msd_weight}")

    # ══════════════════════════════════════════════════════
    # gate 可視化用的固定測試圖 (Kodak 19, 20)
    # ══════════════════════════════════════════════════════
    gate_vis_imgs = None
    if args.msd_weight > 0:
        kodak_dir = "/home/at9529/ycw.cs14/dataset/TestImage/Kodak"
        vis_paths = [
            os.path.join(kodak_dir, "19.png"),
            os.path.join(kodak_dir, "20.png"),
        ]
        vis_list = []
        for p in vis_paths:
            if os.path.exists(p):
                img = Image.open(p).convert("RGB")
                vis_list.append(transforms.ToTensor()(img))
            else:
                print(f"[GateVis] WARNING: {p} not found, skip.")
        if vis_list:
            try:
                gate_vis_imgs = torch.stack(vis_list).to(device)  # (2,3,H,W)
            except RuntimeError:
                # 尺寸不同(19 直的 20 橫的),各自 unsqueeze 分開存
                gate_vis_imgs = [v.unsqueeze(0).to(device) for v in vis_list]
            print(f"[GateVis] loaded {len(vis_list)} fixed images for gate viz.")

    # ══════════════════════════════════════════════════════
    # 6.5 梯度累積設定
    # ══════════════════════════════════════════════════════
    accum = max(1, args.grad_accum)
    if accum > 1 and args.disc_every != 1:
        print(f"[Warn] grad_accum={accum} 時 disc_every 強制設為 1"
              f"(否則 D 的累積窗口會與 G 錯開)")
        args.disc_every = 1

    accum_count = 0      # 目前窗口已累積幾個 micro-batch
    optimizer.zero_grad(set_to_none=True)
    if aux_optimizer is not None:
        aux_optimizer.zero_grad(set_to_none=True)
    if optimizer_disc is not None:
        optimizer_disc.zero_grad(set_to_none=True)

    print(f"[Accum] micro_batch={args.batch_size} x accum={accum} "
          f"-> effective batch={args.batch_size * accum}")
    print(f"[Accum] 驗證頻率: 每 {args.val_interval} 次參數更新 "
          f"(= {args.val_interval * accum} 個 micro-batch)")

    # ══════════════════════════════════════════════════════
    # 7. Training loop
    # ══════════════════════════════════════════════════════
    for epoch in range(last_epoch, args.epochs):
        print(f"Epoch {epoch}: lr={optimizer.param_groups[0]['lr']}")
        net.train()
        device = next(net.parameters()).device

        for i, d in enumerate(train_dataloader):
            start_time = time.time()
            d = d.to(device)
            iterations += 1

            if not args.skip_warmup:
                mse_w, wd_w = wd_scheduler.get_weights(iterations)
                criterion.set_weights(mse_weight=mse_w, wd_weight=wd_w)

            # ─── Forward ───
            out_net = net(d)
            out_criterion = criterion(out_net, d)

            # ─── grass_mask(MSD 與 gate loss 共用,只算一次)───
            grass_mask = None
            need_mask = (args.msd_weight > 0 and not args.msd_use_gate) \
                or args.gate_weight > 0
            if need_mask:
                with torch.no_grad():
                    grass_mask = grass_mask_fn(
                        d, msd_saliency_model,
                        blur_ks=args.msd_mask_blur_ks,
                        hf_pct=args.msd_mask_hf_pct,
                        sal_pct=args.msd_mask_sal_pct,
                        aniso_ks=args.msd_mask_aniso_ks,
                        aniso_pct=args.msd_mask_aniso_pct,
                        use_aniso=args.msd_mask_use_aniso,
                    )
                    if args.msd_mask_blur_sigma > 0:
                        ks = int(2 * round(3 * args.msd_mask_blur_sigma) + 1)
                        grass_mask = TF.gaussian_blur(
                            grass_mask, kernel_size=[ks, ks],
                            sigma=[args.msd_mask_blur_sigma] * 2)
                    grass_mask = grass_mask.clamp(0, 1)

            # ─── MSD loss(對比式:div_in 達標 + div_out 壓小防 leak)───
            msd_loss = torch.tensor(0.0, device=device)
            delta_x = torch.tensor(0.0, device=device)     # mask 內 diversity(監控用)
            div_out = torch.tensor(0.0, device=device)     # mask 外 diversity(監控用)
            gcov = 0.0
            if args.msd_weight > 0:
                out_net_alt = net(d)
                delta = (out_net["x_hat"] - out_net_alt["x_hat"]
                         ).abs().mean(1, keepdim=True)   # (B,1,H,W)

                if args.msd_use_gate:
                    # ─── 路線 B: gate-weighted(對稱對比) ───
                    # gate 由 WD/LPIPS 決定,MSD 端 detach,只推 delta 不動 gate
                    gate = out_net.get("gate", None)
                    if gate is None:
                        msd_loss = torch.tensor(0.0, device=device)
                    else:
                        gate_w = gate.detach()
                        if gate_w.shape[-2:] != delta.shape[-2:]:
                            gate_w = F.interpolate(
                                gate_w, size=delta.shape[-2:],
                                mode='bilinear', align_corners=False)
                        gate_w = gate_w.clamp(0, 1)
                        inv_w = 1.0 - gate_w                    # 非注入區
                        gate_sum = gate_w.sum()
                        inv_sum = inv_w.sum()
                        gcov = gate_w.mean().item()

                        if gate_sum < args.msd_gate_min_sum:
                            # gate 幾乎全關,跳過(避免除零 + 沒必要推)
                            msd_loss = torch.tensor(0.0, device=device)
                            delta_x = torch.tensor(0.0, device=device)
                        else:
                            # div_in:注入區要達標
                            delta_x = (delta * gate_w).sum() / (gate_sum + 1e-6)
                            if args.msd_target_div > 0:
                                loss_in = (args.msd_target_div - delta_x).clamp(min=0)
                            else:
                                loss_in = -torch.log(delta_x + 1e-6)

                            # div_out:非注入區壓小(對比項)
                            if args.msd_contrast_weight > 0 and inv_sum > 1.0:
                                div_out = (delta * inv_w).sum() / (inv_sum + 1e-6)
                                loss_out = (div_out - args.msd_out_eps).clamp(min=0)
                            else:
                                loss_out = torch.tensor(0.0, device=device)

                            msd_loss = loss_in + args.msd_contrast_weight * loss_out

                else:
                    # ─── 路線 A: grass-mask(對比式) ───
                    # grass_mask 上面已經算好了,這裡直接用
                    inv_mask = 1.0 - grass_mask                 # 非草地(機身/天空)
                    mask_sum = grass_mask.sum()
                    inv_sum = inv_mask.sum()
                    gcov = grass_mask.mean().item()

                    # div_in:草地要達標
                    delta_x = (delta * grass_mask).sum() / (mask_sum + 1e-6)
                    if args.msd_target_div > 0:
                        loss_in = (args.msd_target_div - delta_x).clamp(min=0)
                    else:
                        loss_in = -torch.log(delta_x + 1e-6)

                    # div_out:非草地壓小(對比項,防 leak)
                    if args.msd_contrast_weight > 0 and inv_sum > 1.0:
                        div_out = (delta * inv_mask).sum() / (inv_sum + 1e-6)
                        loss_out = (div_out - args.msd_out_eps).clamp(min=0)
                    else:
                        loss_out = torch.tensor(0.0, device=device)

                    msd_loss = loss_in + args.msd_contrast_weight * loss_out

            # ─── gate coverage loss ───
            gate_loss = torch.tensor(0.0, device=device)
            g_cov_in = g_cov_out = 0.0
            if args.gate_weight > 0 and grass_mask is not None:
                gates = out_net.get("gates", None)
                if gates is not None:
                    gate_loss, g_cov_in, g_cov_out = gate_bce_loss(
                        gates, grass_mask,
                        pos_weight=args.gate_pos_weight,
                        soften=args.gate_soften)

            # ═══════════════════════════════════════════════
            # GAN: generator adversarial loss
            # ═══════════════════════════════════════════════
            adv_loss = torch.tensor(0.0, device=device)
            gan_w = 0.0
            if net_disc is not None and iterations >= args.gan_start_iter:
                if args.gan_warmup_iters > 0:
                    prog = (iterations - args.gan_start_iter) / args.gan_warmup_iters
                    gan_w = args.gan_weight * min(1.0, max(0.0, prog))
                else:
                    gan_w = args.gan_weight

                if gan_w > 0:
                    # G step:凍住 D 的參數,梯度只穿過 D 回到 x_hat
                    set_disc_requires_grad(net_disc, False)
                    adv_loss = net_disc(
                        to_disc_input(out_net["x_hat"]), for_G=True).mean()

            # total_loss 維持「未除以 accum」的值,
            # 這樣 skip 判斷與 log 都跟沒有累積時同一個尺度。
            total_loss = (out_criterion["loss"]
                          + args.msd_weight * msd_loss
                          + args.gate_weight * gate_loss
                          + gan_w * adv_loss)

            # 兩個 skip 檢查都在 backward 之前,
            # 所以 continue 不會留下半份髒梯度,窗口只是晚一步湊滿。
            if not _criterion_is_finite(out_criterion) or not _is_finite_tensor(total_loss):
                print(f"[Skip] Non-finite loss at epoch {epoch}, "
                      f"iter {iterations} (loss={total_loss.item()}).")
                continue
            if total_loss.item() > args.skip_loss_threshold and epoch > 0:
                print(f"[Skip] Loss>{args.skip_loss_threshold} at epoch {epoch}, "
                      f"iter {iterations} (loss={total_loss.item():.4f}).")
                continue

            # ─── 除以 accum 才 backward,梯度自動累加到 .grad ───
            (total_loss / accum).backward()

            # ─── aux 也一起累積(獨立的圖,每個 micro-batch 重建一次)───
            if aux_optimizer is not None:
                aux_loss = net.aux_loss()
                if _is_finite_tensor(aux_loss):
                    (aux_loss / accum).backward()
                else:
                    print(f"[Skip] Non-finite aux loss at epoch {epoch}, "
                          f"iter {iterations} (aux_loss={aux_loss.item()}).")
            else:
                aux_loss = torch.tensor(0.0, device=device)

            # ═══════════════════════════════════════════════
            # GAN: discriminator 也用同一個窗口累積
            # D 從 --disc-start-iter 就開始學,
            # 但 G 要到 --gan-start-iter 才會感覺到它。
            # ═══════════════════════════════════════════════
            d_loss_val = 0.0
            d_real_val = d_fake_val = 0.0
            if net_disc is not None and iterations >= args.disc_start_iter:
                set_disc_requires_grad(net_disc, True)

                # real / fake 分開 backward,省一份 activation 記憶體
                real_in = to_disc_input(d.detach())
                loss_real = net_disc(real_in, for_real=True).mean()
                if _is_finite_tensor(loss_real):
                    (loss_real / accum).backward()
                    d_real_val = loss_real.item()

                fake_in = to_disc_input(out_net["x_hat"].detach())
                loss_fake = net_disc(fake_in, for_real=False).mean()
                if _is_finite_tensor(loss_fake):
                    (loss_fake / accum).backward()
                    d_fake_val = loss_fake.item()

                d_loss_val = d_real_val + d_fake_val

            # ═══════════════════════════════════════════════
            # 累積滿了才真正 step
            # ═══════════════════════════════════════════════
            accum_count += 1
            d_gnorm = 0.0        # 先初始化,不然沒進 D 區塊時 print 會 NameError
            did_step = False
            if accum_count >= accum:
                if args.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        net.parameters(), args.clip_max_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if aux_optimizer is not None:
                    aux_optimizer.step()
                    aux_optimizer.zero_grad(set_to_none=True)

                if net_disc is not None and iterations >= args.disc_start_iter:
                    if args.disc_clip_norm > 0:
                        d_gnorm = float(torch.nn.utils.clip_grad_norm_(
                            net_disc.parameters(), args.disc_clip_norm))
                    optimizer_disc.step()
                    optimizer_disc.zero_grad(set_to_none=True)

                accum_count = 0
                updates += 1
                did_step = True

            if i % 2 == 0:
                mse_loss = out_criterion['mse_loss']
                if mse_loss.item() > 0:
                    psnr = 10 * (torch.log(1 / mse_loss) / np.log(10))
                    psnrs.update(psnr.item())
                else:
                    psnrs.update(100)
                elapsed.update(time.time() - start_time)
                losses.update(total_loss.item())
                bpps.update(out_criterion['bpp_loss'].item())
                mse_losses.update(mse_loss.item())
                wd_losses.update(out_criterion['wd_loss'].item())
                lpips_losses.update(out_criterion['lpips_loss'].item())
                aux_losses.update(aux_loss.item())
                msd_losses.update(msd_loss.item())
                adv_losses.update(adv_loss.item())
                disc_losses.update(d_loss_val)

            if i % 10 == 0:
                current_time = datetime.now()
                print(' | '.join([
                    f"{current_time}",
                    f'Epoch {epoch}',
                    f'Iter {iterations}',
                    f'Upd {updates}',
                    f"{i * len(d)}/{len(train_dataloader.dataset)}",
                    f'T {elapsed.val:.3f}({elapsed.avg:.3f})',
                    f'Loss {losses.val:.3f}({losses.avg:.3f})',
                    f'PSNR {psnrs.val:.3f}({psnrs.avg:.3f})',
                    f'Bpp {bpps.val:.5f}({bpps.avg:.5f})',
                    f'MSE {mse_losses.val:.5f}',
                    f'WD {wd_losses.val:.4f}',
                    f'LPIPS {lpips_losses.val:.4f}',
                    f'MSD {msd_losses.val:.4f}',
                    f'w_mse {criterion.mse_weight:.3f} w_wd {criterion.wd_weight:.3f}',
                ]))
                if args.msd_weight > 0:
                    ratio = (args.msd_weight * msd_loss /
                             out_criterion["loss"]).abs()
                    tag = "gate_cover" if args.msd_use_gate else "grass_mask_cover"
                    print(f"delta_x={delta_x.item():.5f}  div_out={div_out.item():.5f}  "
                          f"msd_loss={msd_loss.item():.4f}  "
                          f"ratio={ratio.item():.2%}  {tag}={gcov:.2%}")

                if args.gate_weight > 0:
                    print(f"gate_in={g_cov_in:.4f} gate_out={g_cov_out:.4f} "
                          f"sep={g_cov_in - g_cov_out:+.4f} "
                          f"gate_bce={gate_loss.item():.4f}")

                if net_disc is not None:
                    # D_gnorm 只有在真的 step 的那個 micro-batch 才有意義
                    gnorm_str = f"{d_gnorm:.3f}" if did_step else "n/a"
                    print(f"[GAN] w={gan_w:.4f} adv_G={adv_loss.item():.4f} "
                          f"D_real={d_real_val:.4f} D_fake={d_fake_val:.4f} "
                          f"D_total={d_loss_val:.4f} D_gnorm={gnorm_str}")

                start_time = time.time()

                log_dict = {
                    "train/loss": losses.val,
                    "train/bpp": bpps.val,
                    "train/psnr": psnrs.val,
                    "train/mse": mse_losses.val,
                    "train/lpips": lpips_losses.val,
                    "train/wd": wd_losses.val,
                    "train/msd": msd_losses.val,
                    "train/aux_loss": aux_losses.val,
                    "train/msd_div_out": div_out.item(),
                    "train/updates": updates,
                    "lr": optimizer.param_groups[0]['lr'],
                    "weights/mse": criterion.mse_weight,
                    "weights/wd": criterion.wd_weight,
                }
                log_dict.update({
                    "gate/cov_in": g_cov_in,
                    "gate/cov_out": g_cov_out,
                    "gate/loss": gate_loss.item(),
                })
                if net_disc is not None:
                    log_dict.update({
                        "gan/adv_G": adv_loss.item(),
                        "gan/D_real": d_real_val,
                        "gan/D_fake": d_fake_val,
                        "gan/D_total": d_loss_val,
                        "weights/gan": gan_w,
                    })
                    if did_step:
                        log_dict["gan/D_gnorm"] = d_gnorm
                log_metrics(experiment, log_dict, step=iterations)

            # ═══════════════════════════════════════════════
            # 驗證:以「參數更新次數」計,且只在累積窗口邊界做
            # ═══════════════════════════════════════════════
            if did_step and updates % args.val_interval == 0 and updates > 0:
                print(f"[VAL] updates={updates} (iterations={iterations})")
                net.eval()

                # ─── gate 可視化 (固定 Kodak 19/20, 存 comet + 本地) ───
                if args.msd_weight > 0 and gate_vis_imgs is not None:
                    gate_dir = os.path.join("gate_vis", ckpt_name)
                    os.makedirs(gate_dir, exist_ok=True)
                    with torch.no_grad():
                        # 支援 stack (同尺寸) 或 list (不同尺寸) 兩種情況
                        vis_batches = ([gate_vis_imgs]
                                       if torch.is_tensor(gate_vis_imgs)
                                       else gate_vis_imgs)
                        idx = 19  # Kodak 編號起點,用於命名
                        for vb in vis_batches:
                            vis_out = net(vb)
                            gate = vis_out.get("gate", None)
                            if gate is None:
                                continue
                            gate_up = F.interpolate(
                                gate, size=vb.shape[-2:],
                                mode='bilinear', align_corners=False)
                            for b in range(vb.size(0)):
                                name = f"kodim{idx}"
                                g = gate_up[b]
                                g_norm = (g - g.min()) / (g.max() - g.min() + 1e-8)
                                save_image(g_norm, os.path.join(
                                    gate_dir, f"gate_{name}_norm_upd{updates}.png"))
                                save_image(gate_up[b], os.path.join(
                                    gate_dir, f"gate_{name}_upd{updates}.png"))
                                if experiment is not None:
                                    experiment.log_image(_to_comet_img(gate_up[b]),
                                                         name=f"gate_{name}_upd{updates}",
                                                         step=iterations)
                                    if updates == args.val_interval:
                                        experiment.log_image(_to_comet_img(vb[b]),
                                                             name=f"input_{name}",
                                                             step=iterations)
                                print(f"[GateVis] {name} gate_cover="
                                      f"{gate_up[b].mean().item():.2%}")
                                idx += 1

                # ─── 原本的驗證 ───
                loss = test_epoch(iterations, test_dataloader,
                                  net, criterion, experiment=experiment)
                log_metrics(experiment, {"val/loss": loss}, step=iterations)
                lr_scheduler.step(loss)
                net.train()
                if net_disc is not None:
                    net_disc.train()

                is_best = loss < best_loss
                best_loss = min(loss, best_loss)

                if args.save:
                    save_obj = {
                        "state_dict": (net.module.state_dict()
                                       if isinstance(net, CustomDataParallel)
                                       else net.state_dict()),
                        "iterations": iterations,
                        "updates": updates,
                        "epoch": epoch,
                        "best_loss": best_loss,
                    }
                    if net_disc is not None:
                        # D 一起存,不然中斷後 D 要從頭再學一次
                        save_obj["disc_state_dict"] = net_disc.state_dict()
                        save_obj["disc_optimizer"] = optimizer_disc.state_dict()

                    epoch_ckpt_path = os.path.join(
                        ckpt_epoch_dir, f"upd_{updates}.pth.tar")
                    torch.save(save_obj, epoch_ckpt_path)
                    print(f"[Save] Per-val ckpt -> {epoch_ckpt_path}")

                    if is_best or updates % 20000 == 0:
                        torch.save(save_obj, f"{args.save_path}")
                        print(f"[Save] Saved to {args.save_path}")


if __name__ == "__main__":
    main(sys.argv[1:])