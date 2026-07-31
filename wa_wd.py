# Copyright 2025 Yueyu Hu, Jona Ballé.  (original Wasserstein Distortion implementation)
# Wavelet-Wasserstein Distortion (WA-WD) extension.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the specific language governing
# permissions and limitations under the License.
# ========================================================================================
"""Wavelet-Wasserstein Distortion (WA-WD) in PyTorch.

WA-WD = WD computed independently inside each orthonormal wavelet subband of the
VGG feature maps, then aggregated with (optionally learnable) per-subband weights.

Because the DWT is orthonormal, the metric properties of WD are preserved, and the
diagonal-Gaussian approximation underlying WD becomes tighter (wavelet coefficients
of natural images are approximately decorrelated).

The WD kernel itself (`WassersteinDistortionFeature`) is UNCHANGED from the original
implementation -- WA-WD only swaps the feature basis it operates on.
"""

from typing import override
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models as tv

Tensor = torch.Tensor


# ========================================================================================
# Unchanged building blocks from the original WD implementation
# ========================================================================================


class LowpassFilter2D(nn.Module):
    kernel: Tensor

    def __init__(self):
        super().__init__()
        kernel_1d = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float32)
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        self.register_buffer("kernel", kernel_2d[None, None, :, :])

    @override
    def forward(self, x, stride=1):
        kernel = self.kernel.expand((x.shape[1], 1, -1, -1))
        x = F.conv2d(x, kernel, stride=stride, padding=1, groups=x.shape[1])  # pylint: disable=not-callable
        return x


class MultiLevelStats(nn.Module):
    def __init__(self, num_levels=4):
        super().__init__()
        self.num_levels = num_levels
        self.lowpass = LowpassFilter2D()

    @override
    def forward(self, x):
        squared = x**2
        means = []
        variances = []
        for _ in range(self.num_levels):
            m = self.lowpass(x, stride=1)
            p = self.lowpass(squared, stride=1)
            means.append(m)
            variances.append(p - m**2)
            x = m[..., ::2, ::2]
            squared = p[..., ::2, ::2]
        return means, variances


class WassersteinDistortionFeature(nn.Module):
    """Calculates the Wasserstein distortion between two feature maps.

    NOTE: identical to the original implementation. WA-WD reuses it verbatim; the
    only difference is that it is fed wavelet subbands instead of raw VGG features.
    """

    def __init__(self, num_levels: int = 5):
        super().__init__()
        self.multi_level_stats = MultiLevelStats(num_levels)
        self.num_levels = num_levels
        self.lowpass = LowpassFilter2D()

    @override
    def forward(
        self,
        features_a: Tensor,
        features_b: Tensor,
        log2_sigma: Tensor,
    ) -> Tensor:
        """Calculates the Wasserstein distortion between two feature maps."""
        mean_pyr_a, var_pyr_a = self.multi_level_stats(features_a)
        mean_pyr_b, var_pyr_b = self.multi_level_stats(features_b)
        wd_maps = [torch.square(features_a - features_b)]
        for i in range(self.num_levels):
            std_pyr_a_i = torch.sqrt(torch.clamp(var_pyr_a[i], min=1e-8))
            std_pyr_b_i = torch.sqrt(torch.clamp(var_pyr_b[i], min=1e-8))
            square_mu = torch.square(mean_pyr_a[i] - mean_pyr_b[i])
            square_scale = torch.square(std_pyr_a_i - std_pyr_b_i)
            wd_maps.append(square_mu + square_scale)

        wasserstein_dist = 0
        for i, wd_map in enumerate(wd_maps):
            weights_i = F.relu(1 - torch.abs(log2_sigma - i))
            if i > 0:
                log2_sigma = self.lowpass(log2_sigma, stride=2)
            wasserstein_dist += (weights_i * wd_map).mean()
        assert isinstance(wasserstein_dist, Tensor)
        return wasserstein_dist


# pyright: reportIndexIssue=false
class MultiscaleTruncatedVGG16(nn.Module):
    """
    A VGG module that supports executing only the first few blocks
    (i.e. truncated) for computation saving. It supports multiscale
    feature extraction, where the input image is downsampled to
    different resolutions and processed through the VGG network.
    """

    mean: Tensor
    std: Tensor

    def __init__(
        self,
        requires_grad=False,
        pretrained=True,
        truncate_slice=5,
        replace_with_avg_pooling=True,
    ):
        """Initialize the MultiscaleTruncatedVGG module.
        The JAX version replaces the max pooling layers with average pooling, so
        this option is available here as well.
        """
        super().__init__()
        vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.num_slices = 5
        self.truncate_slice = truncate_slice
        if not 1 <= truncate_slice <= self.num_slices:
            raise ValueError(
                f"truncate_slice must be between 1 and {self.num_slices}, inclusive, "
                f"but is {truncate_slice}."
            )

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        if self.truncate_slice >= 2:
            for x in range(4, 9):
                if replace_with_avg_pooling and isinstance(
                    vgg_pretrained_features[x], nn.MaxPool2d
                ):
                    self.slice2.add_module(str(x), nn.AvgPool2d(kernel_size=2, stride=2))
                else:
                    self.slice2.add_module(str(x), vgg_pretrained_features[x])
        if self.truncate_slice >= 3:
            for x in range(9, 16):
                if replace_with_avg_pooling and isinstance(
                    vgg_pretrained_features[x], nn.MaxPool2d
                ):
                    self.slice3.add_module(str(x), nn.AvgPool2d(kernel_size=2, stride=2))
                else:
                    self.slice3.add_module(str(x), vgg_pretrained_features[x])
        if self.truncate_slice >= 4:
            for x in range(16, 23):
                if replace_with_avg_pooling and isinstance(
                    vgg_pretrained_features[x], nn.MaxPool2d
                ):
                    self.slice4.add_module(str(x), nn.AvgPool2d(kernel_size=2, stride=2))
                else:
                    self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if self.truncate_slice >= 5:
            for x in range(23, 30):
                if replace_with_avg_pooling and isinstance(
                    vgg_pretrained_features[x], nn.MaxPool2d
                ):
                    self.slice5.add_module(str(x), nn.AvgPool2d(kernel_size=2, stride=2))
                else:
                    self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.slice_names = ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        self.valid_slices = self.slice_names[: self.truncate_slice]

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.lowpass = LowpassFilter2D()

    @override
    def forward(self, x: Tensor, num_scales: int = 3) -> list[Tensor]:
        """
        Forward pass through the truncated VGG network.
        Args:
            X (Tensor): Input image tensor of shape (N, 3, H, W). Assumed
            to be RGB and normalized to [0, 1].
        Returns:
            list[Tensor]: feature maps from every (scale, slice) combination,
            plus the input image itself at the front.
        """
        x = (x - self.mean) / self.std
        features = [x]
        for _ in range(num_scales):
            h = self.slice1(x)
            h_relu1_2 = h
            output_slices = [h_relu1_2]
            if self.truncate_slice >= 2:
                h = self.slice2(h)
                h_relu2_2 = h
                output_slices.append(h_relu2_2)
            if self.truncate_slice >= 3:
                h = self.slice3(h)
                h_relu3_3 = h
                output_slices.append(h_relu3_3)
            if self.truncate_slice >= 4:
                h = self.slice4(h)
                h_relu4_3 = h
                output_slices.append(h_relu4_3)
            if self.truncate_slice >= 5:
                h = self.slice5(h)
                h_relu5_3 = h
                output_slices.append(h_relu5_3)
            features += output_slices
            x = self.lowpass(x, stride=2)

        return features


class VGG16WassersteinDistortion(nn.Module):
    """Calculates the VGG-16 Wasserstein Distortion between two images."""

    def __init__(
        self,
        feature_net: str = "vgg16",
        num_levels: int = 5,
        grayscale: bool = False,
        normalize_center_to_zero: bool = False,
    ):
        super().__init__()
        self.wasserstein_distortion_feature = WassersteinDistortionFeature(num_levels)
        self.grayscale = grayscale
        self.normalize_center_to_zero = normalize_center_to_zero
        if feature_net == "vgg16":
            truncate_slice = 5
            self.feature_backbone = MultiscaleTruncatedVGG16(
                requires_grad=False, pretrained=True, truncate_slice=truncate_slice
            )
            self.truncate_slice = truncate_slice
        else:
            raise ValueError(f"Unsupported feature network: {feature_net}.")

    def _preprocess(self, pred: Tensor, gt: Tensor) -> tuple[Tensor, Tensor]:
        if self.grayscale:
            pred = pred.expand(-1, 3, -1, -1)
            gt = gt.expand(-1, 3, -1, -1)
        if self.normalize_center_to_zero:
            pred = pred * 2 - 1
            gt = gt * 2 - 1
        if pred.shape != gt.shape:
            raise ValueError(
                f"Predicted and ground truth images must have the same shape, "
                f"but got {pred.shape} and {gt.shape}."
            )
        return pred, gt

    @staticmethod
    def _align_sigma(log2_sigma: Tensor, ref: Tensor, offset: float = 0.0) -> Tensor:
        """Resample and rescale the sigma field to a feature (or subband) array.

        Since a low-resolution array covers a larger portion of the image per
        element, sigma must shrink correspondingly. In log space that is a
        subtraction of the log size ratio, capped at zero. `offset` allows a
        per-subband shift of the pooling width (used by WA-WD).
        """
        ls = F.interpolate(
            log2_sigma, size=ref.shape[-2:], mode="bilinear", antialias=False
        )
        log_ratio_h = np.log2(log2_sigma.shape[-2] / ref.shape[-2])
        log_ratio_w = np.log2(log2_sigma.shape[-1] / ref.shape[-1])
        mean_log_ratio = (log_ratio_h + log_ratio_w) / 2
        return F.relu(ls - mean_log_ratio + offset)

    @override
    def forward(
        self,
        pred: Tensor,
        gt: Tensor,
        log2_sigma: Tensor,
        num_scales: int = 3,
    ) -> Tensor:
        pred, gt = self._preprocess(pred, gt)
        feats_pred = self.feature_backbone(pred, num_scales=num_scales)
        feats_gt = self.feature_backbone(gt, num_scales=num_scales)

        wasserstein_dist = 0
        assert len(feats_pred) == len(feats_gt)
        for fp, fgt in zip(feats_pred, feats_gt):
            ls = self._align_sigma(log2_sigma, fgt)
            wasserstein_dist += self.wasserstein_distortion_feature(fp, fgt, ls)
        assert isinstance(wasserstein_dist, Tensor)
        return wasserstein_dist


# ========================================================================================
# WA-WD: new components
# ========================================================================================


class HaarDWT2D(nn.Module):
    """Single-level orthonormal 2D Haar DWT.

    Returns (LL, LH, HL, HH), each at half spatial resolution. The 1/2 scaling is
    what makes the transform orthonormal -- Theorem 3.1 (metric preservation) and
    the Parseval identity used in the σ→0 reduction to MSE both depend on it.

    Odd spatial dimensions are replicate-padded. This matters in practice: deep VGG
    slices at multiple scales routinely produce odd-sized feature maps.
    """

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if x.shape[-2] % 2:
            x = F.pad(x, (0, 0, 0, 1), mode="replicate")
        if x.shape[-1] % 2:
            x = F.pad(x, (0, 1, 0, 0), mode="replicate")
        a = x[..., 0::2, 0::2]
        b = x[..., 0::2, 1::2]
        c = x[..., 1::2, 0::2]
        d = x[..., 1::2, 1::2]
        ll = 0.5 * (a + b + c + d)  # structure / fidelity
        lh = 0.5 * (a + b - c - d)  # horizontal edges
        hl = 0.5 * (a - b + c - d)  # vertical edges
        hh = 0.5 * (a - b - c + d)  # diagonal detail
        return ll, lh, hl, hh


BAND_NAMES = ("LL", "LH", "HL", "HH")


def multilevel_haar(
    x: Tensor, dwt: HaarDWT2D, levels: int
) -> list[tuple[str, int, Tensor]]:
    """Recursive Haar decomposition of the LL band.

    Returns a list of (band_name, level, tensor) with 3 * levels + 1 entries,
    ordered deterministically so two calls can be zipped safely.

    The paper uses levels=1: deeper decompositions decorrelate further (Fig. 15)
    but shrink already-small VGG features past the point where the local statistics
    are estimable (5 VGG stages x 2-3 DWT levels reduces resolution 128-256x).
    """
    bands: list[tuple[str, int, Tensor]] = []
    cur = x
    for lvl in range(1, levels + 1):
        ll, lh, hl, hh = dwt(cur)
        bands.append(("LH", lvl, lh))
        bands.append(("HL", lvl, hl))
        bands.append(("HH", lvl, hh))
        cur = ll
    bands.append(("LL", levels, cur))
    return bands


class VGG16WaveletWassersteinDistortion(VGG16WassersteinDistortion):
    """VGG-16 Wavelet-Wasserstein Distortion (WA-WD).

    Drop-in replacement for `VGG16WassersteinDistortion`: identical call signature,
    identical return type. Two knobs control the fidelity/realism balance:

      * `band_logits`   -- per-subband weights (softmax-normalised, sum to 1).
                           Learnable by default; zeros give the paper's equal-weight
                           default of 1/4 per band at levels=1.
      * `sigma_offsets` -- per-band-type shift of log2(sigma). Negative narrows the
                           pooling window (more fidelity), positive widens it (more
                           tolerance to texture resampling). LL should be the most
                           negative; HH the most positive.

    Args:
        dwt_levels: number of Haar decomposition levels. 1 in the paper.
        learnable_weights: if False, subband weights stay fixed at their init.
        sigma_offsets: (LL, LH, HL, HH) offsets in log2 units. Defaults to zeros,
            i.e. plain equal-treatment WA-WD. A reasonable tuned starting point is
            (-0.5, 0.5, 0.5, 1.0).
        num_levels: WD pyramid depth. Note the default here is 6, not 5: each DWT
            level halves the subband resolution, so one extra pyramid level is
            needed to reach the same absolute maximum pooling width as plain WD.
    """

    def __init__(
        self,
        feature_net: str = "vgg16",
        num_levels: int = 6,
        grayscale: bool = False,
        normalize_center_to_zero: bool = False,
        dwt_levels: int = 1,
        learnable_weights: bool = True,
        sigma_offsets: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        ll_weight_boost: float = 0.0
    ):
        super().__init__(
            feature_net=feature_net,
            num_levels=num_levels,
            grayscale=grayscale,
            normalize_center_to_zero=normalize_center_to_zero,
        )
        if dwt_levels < 1:
            raise ValueError(f"dwt_levels must be >= 1, but is {dwt_levels}.")
        self.dwt = HaarDWT2D()
        self.dwt_levels = dwt_levels
        self.num_bands = 3 * dwt_levels + 1

        # self.band_logits = nn.Parameter(
        #     torch.zeros(self.num_bands), requires_grad=learnable_weights
        # )
        init = torch.zeros(self.num_bands)
        if ll_weight_boost != 0.0:
            init[-1] = ll_weight_boost          # LL 在最後一個位置
        self.band_logits = nn.Parameter(init, requires_grad=learnable_weights)
        self._offset_lookup = dict(zip(BAND_NAMES, sigma_offsets))

    def band_weights(self) -> Tensor:
        """Normalised subband weights, in the order produced by `multilevel_haar`."""
        return torch.softmax(self.band_logits, dim=0)

    def named_band_weights(self) -> dict[str, float]:
        """Human-readable view of the current weights, for logging."""
        w = self.band_weights().detach().cpu()
        names = [f"{n}{lvl}" for n, lvl, _ in multilevel_haar(
            torch.zeros(1, 1, 2**self.dwt_levels, 2**self.dwt_levels),
            self.dwt,
            self.dwt_levels,
        )]
        return dict(zip(names, w.tolist()))

    @override
    def forward(
        self,
        pred: Tensor,
        gt: Tensor,
        log2_sigma: Tensor,
        num_scales: int = 3,
    ) -> Tensor:
        pred, gt = self._preprocess(pred, gt)
        feats_pred = self.feature_backbone(pred, num_scales=num_scales)
        feats_gt = self.feature_backbone(gt, num_scales=num_scales)
        assert len(feats_pred) == len(feats_gt)

        weights = self.band_weights()
        wasserstein_dist = 0
        for fp, fgt in zip(feats_pred, feats_gt):
            bands_p = multilevel_haar(fp, self.dwt, self.dwt_levels)
            bands_g = multilevel_haar(fgt, self.dwt, self.dwt_levels)
            for k, ((name, _, bp), (_, _, bg)) in enumerate(zip(bands_p, bands_g)):
                # The resolution ratio inside _align_sigma automatically accounts
                # for the -1 log2 shift per DWT level; only the per-band tuning
                # offset needs to be supplied explicitly.
                ls = self._align_sigma(log2_sigma, bg, self._offset_lookup[name])
                wasserstein_dist += weights[k] * self.wasserstein_distortion_feature(
                    bp, bg, ls
                )
        assert isinstance(wasserstein_dist, Tensor)
        return wasserstein_dist


# ========================================================================================
# Self-checks
# ========================================================================================


def _check_orthonormality(atol: float = 1e-5) -> None:
    """Parseval: an orthonormal DWT preserves total energy.

    Catches a wrong Haar normalisation constant (e.g. 1/sqrt(2) or 1/4 instead of 1/2).
    """
    dwt = HaarDWT2D()
    x = torch.randn(2, 7, 32, 32)
    bands = multilevel_haar(x, dwt, levels=2)
    energy_in = x.pow(2).sum()
    energy_out = sum(b.pow(2).sum() for _, _, b in bands)
    err = (energy_in - energy_out).abs().item()
    assert err < atol, f"Parseval violated: {energy_in:.6f} vs {energy_out:.6f}"
    print(f"[ok] orthonormality (Parseval)   energy error = {err:.2e}")


def _check_reduces_to_wd(atol: float = 1e-5) -> None:
    """At sigma = 1 (log2_sigma = 0) both WD and WA-WD collapse to the pointwise term.

    The pyramid weight is relu(1 - |log2_sigma - i|), so log2_sigma = 0 keeps only
    level i=0 -- i.e. mean squared error on the features. Do NOT use a large negative
    log2_sigma to emulate sigma -> 0: it drives every weight to zero and the loss
    becomes identically 0, which passes any comparison vacuously.

    Each subband holds 1/4 of the elements, so a per-band `.mean()` is 4x larger than
    the full-resolution mean; with 4 bands and softmax weights of 1/4 the factors
    cancel and WA-WD equals WD exactly. Use even spatial dimensions so that no
    replicate padding perturbs the identity.
    """
    torch.manual_seed(0)
    wd = VGG16WassersteinDistortion(num_levels=5)
    wa = VGG16WaveletWassersteinDistortion(num_levels=5, dwt_levels=1)
    wd.eval()
    wa.eval()

    x = torch.rand(1, 3, 64, 64)
    y = torch.rand(1, 3, 64, 64)
    s = torch.zeros(1, 1, 64, 64)  # sigma = 1 -> pointwise regime

    with torch.no_grad():
        a = wd(x, y, s, num_scales=1)
        b = wa(x, y, s, num_scales=1)
    err = (a - b).abs().item()
    assert err < atol, f"WA-WD != WD in the pointwise regime: {a:.6f} vs {b:.6f}"
    print(f"[ok] pointwise reduction to WD   |WD - WA-WD| = {err:.2e}")


def _check_gradients() -> None:
    """Subband weights should receive gradient when learnable."""
    wa = VGG16WaveletWassersteinDistortion(dwt_levels=1, learnable_weights=True)
    x = torch.rand(1, 3, 64, 64, requires_grad=True)
    y = torch.rand(1, 3, 64, 64)
    s = torch.zeros(1, 1, 64, 64)
    wa(x, y, s, num_scales=1).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert wa.band_logits.grad is not None
    print(f"[ok] gradients flow             weights = {wa.named_band_weights()}")


if __name__ == "__main__":
    _check_orthonormality()
    _check_reduces_to_wd()
    _check_gradients()