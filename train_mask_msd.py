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

from utils.Meter import AverageMeterTEST, AverageMeterTRAIN
from models.dcae_gate import DCAE

import emlnet.decoder as eml_decoder
import emlnet.resnet as eml_resnet
from mask_utils import stochastic_mask_fn
from torchvision.utils import save_image
from PIL import Image

def calc_tv_loss(x):
    """計算 Total Variation Loss"""
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w

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
        num_levels=5, dwt_levels=1,
        learnable_weights=False,
        sigma_offsets=(-0.5, 0.0, 0.0, 0.0),
        ll_weight_boost=0.3,
    ).to(DEVICE)
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
                output["x_hat"], target, log2_sigma, num_scales=3).mean()
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
    print(f"[Optimizer] Trainable params: {len(trainable)}")

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(trainable)),
        lr=args.learning_rate,
    )
    aux_optimizer = None
    if aux_parameters:
        aux_optimizer = optim.Adam(
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
        description="DCAE training script with WD + LPIPS + TV support.")
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
    args = parser.parse_args(argv)
    return args


elapsed, data_times, losses, psnrs, bpps, bpp_ys, bpp_zs, mse_losses, wd_losses, aux_losses, msd_losses = \
    [AverageMeterTRAIN(2000) for _ in range(11)]
lpips_losses = AverageMeterTRAIN(2000)


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
    if args.msd_weight > 0 and not args.msd_use_gate:
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
    ckpt_epoch_dir = os.path.join("ckpt_epoch", ckpt_name)
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
    if args.msd_weight > 0 :
        kodak_dir = "/home/at9529/ycw.cs14/dataset/TestImage/Kodak"  # ← 依你實際路徑改
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
            # 兩張尺寸一致才能 stack(Kodak 都是 768x512 或 512x768,若不同則分開處理)
            try:
                gate_vis_imgs = torch.stack(vis_list).to(device)  # (2,3,H,W)
            except RuntimeError:
                # 尺寸不同(19 直的 20 橫的),各自 unsqueeze 分開存
                gate_vis_imgs = [v.unsqueeze(0).to(device) for v in vis_list]
            print(f"[GateVis] loaded {len(vis_list)} fixed images for gate viz.")
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

            # ─── MSD loss ───
            msd_loss = torch.tensor(0.0, device=device)
            delta_x = torch.tensor(0.0, device=device)
            gcov = 0.0
            if args.msd_weight > 0:
                out_net_alt = net(d)
                delta = (out_net["x_hat"] - out_net_alt["x_hat"]
                         ).abs().mean(1, keepdim=True)   # (B,1,H,W)

                if args.msd_use_gate:
                    # ─── 路線 B: gate-weighted ───
                    # gate 由 WD/LPIPS 決定,MSD 端 detach,只推 delta 不動 gate
                    gate = out_net.get("gate", None)
                    if gate is None:
                        # 模型沒吐 gate(理論上不會),退回不加 MSD
                        msd_loss = torch.tensor(0.0, device=device)
                    else:
                        gate_w = gate.detach()
                        # gate 尺寸可能是 x_hat 的一半,upsample 對齊 delta
                        if gate_w.shape[-2:] != delta.shape[-2:]:
                            gate_w = F.interpolate(
                                gate_w, size=delta.shape[-2:],
                                mode='bilinear', align_corners=False)
                        gate_sum = gate_w.sum()
                        gcov = gate_w.mean().item()
                        if gate_sum < args.msd_gate_min_sum:
                            # gate 幾乎全關,跳過(避免除零 + 沒必要推)
                            msd_loss = torch.tensor(0.0, device=device)
                            delta_x = torch.tensor(0.0, device=device)
                        else:
                            delta_x = (delta * gate_w).sum() / (gate_sum + 1e-6)
                            if args.msd_target_div > 0:
                                msd_loss = (args.msd_target_div - delta_x).clamp(min=0)
                            else:
                                msd_loss = -torch.log(delta_x + 1e-6)

                else:
                    # ─── 路線 A: grass-mask ───
                    with torch.no_grad():
                        grass_mask = stochastic_mask_fn(
                            d, msd_saliency_model,
                            blur_ks=args.msd_mask_blur_ks,
                            hf_pct=args.msd_mask_hf_pct,
                            sal_pct=args.msd_mask_sal_pct,
                            aniso_ks=args.msd_mask_aniso_ks,
                            aniso_pct=args.msd_mask_aniso_pct,
                            use_aniso=args.msd_mask_use_aniso,
                        )
                        
                    delta_x = (delta * grass_mask).sum() / (grass_mask.sum() + 1e-6)
                    if args.msd_target_div > 0:
                        msd_loss = (args.msd_target_div - delta_x).clamp(min=0)
                    else:
                        msd_loss = -torch.log(delta_x + 1e-6)
                    gcov = grass_mask.mean().item()

            # ─── Total loss ───
            total_loss = out_criterion["loss"] + args.msd_weight * msd_loss

            if not _criterion_is_finite(out_criterion) or not _is_finite_tensor(total_loss):
                print(f"[Skip] Non-finite loss at epoch {epoch}, "
                      f"iter {iterations} (loss={total_loss.item()}).")
                continue
            if total_loss.item() > args.skip_loss_threshold and epoch > 0:
                print(f"[Skip] Loss>{args.skip_loss_threshold} at epoch {epoch}, "
                      f"iter {iterations} (loss={total_loss.item():.4f}).")
                continue

            optimizer.zero_grad()
            if aux_optimizer is not None:
                aux_optimizer.zero_grad()

            total_loss.backward()

            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_max_norm)
            optimizer.step()

            if aux_optimizer is not None:
                aux_loss = net.aux_loss()
                if not _is_finite_tensor(aux_loss):
                    print(f"[Skip] Non-finite aux loss at epoch {epoch}, "
                          f"iter {iterations} (aux_loss={aux_loss.item()}).")
                    continue
                aux_loss.backward()
                aux_optimizer.step()
            else:
                aux_loss = torch.tensor(0.0, device=device)

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

            if i % 10 == 0:
                current_time = datetime.now()
                print(' | '.join([
                    f"{current_time}",
                    f'Epoch {epoch}',
                    f'Iter {iterations}',
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
                    print(f"delta_x={delta_x.item():.5f}  msd_loss={msd_loss.item():.4f}  "
                          f"ratio={ratio.item():.2%}  {tag}={gcov:.2%}")
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
                    "lr": optimizer.param_groups[0]['lr'],
                    "weights/mse": criterion.mse_weight,
                    "weights/wd": criterion.wd_weight,
                }
                log_metrics(experiment, log_dict, step=iterations)

            # 驗證
            if (iterations % 500 == 0 and iterations != 0):
                print(f"[VAL] iterations={iterations}")
                net.eval()

                # ─── gate 可視化 (固定 Kodak 19/20, 存 comet + 本地) ───
                if args.msd_weight > 0  and gate_vis_imgs is not None:
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
                            # 逐張存(vb 可能含 1 或 2 張)
                            for b in range(vb.size(0)):
                                name = f"kodim{idx}"
                                # 
                                g = gate_up[b]
                                g_norm = (g - g.min()) / (g.max() - g.min() + 1e-8)   # 拉到 [0,1] 看相對
                                save_image(g_norm, os.path.join(gate_dir, f"gate_{name}_norm_iter{iterations}.png"))
                                save_image(gate_up[b], os.path.join(
                                    gate_dir, f"gate_{name}_iter{iterations}.png"))
                                # save_image(vb[b], os.path.join(
                                #     gate_dir, f"input_{name}_iter{iterations}.png"))
                                # comet
                                if experiment is not None:
                                    experiment.log_image(_to_comet_img(gate_up[b]),
                                                        name=f"gate_{name}_iter{iterations}",
                                                        step=iterations)
                                    if iterations == 500:
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

                is_best = loss < best_loss
                best_loss = min(loss, best_loss)

                if args.save:
                    save_obj = {
                        "state_dict": (net.module.state_dict()
                                       if isinstance(net, CustomDataParallel)
                                       else net.state_dict()),
                        "iterations": iterations,
                        "epoch": epoch,
                        "best_loss": best_loss,
                    }

                    epoch_ckpt_path = os.path.join(
                        ckpt_epoch_dir, f"iter_{iterations}.pth.tar")
                    torch.save(save_obj, epoch_ckpt_path)
                    print(f"[Save] Per-val ckpt -> {epoch_ckpt_path}")

                    if is_best or iterations % 20000 == 0:
                        torch.save(save_obj, f"{args.save_path}")
                        print(f"[Save] Saved to {args.save_path}")


if __name__ == "__main__":
    main(sys.argv[1:])