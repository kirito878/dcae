"""
tau_field.py

τ (tau) —— 「可交換性場」(exchangeability field)。

定義
----
τ(x) 大  ⟺  該位置存在豐富且可互換的高頻細節  → 該 realism (σ 大 / 注入 noise)
τ(x) 小  ⟺  高頻能量低(天空) 或 高頻高度結構化/語義顯著(飛機) → 該 fidelity

原理
----
無參數,確定性地從 VGG-wavelet 子帶統計導出(WD 前向已算出這些量):
  項1 淺層高頻能量  (relu1_2/2_2 的 LH/HL/HH):
        回答「有沒有可換的高頻細節」→ 排天空(高頻≈0)
  項2 深層語義門控  (relu4_3/5_3 的通道能量):
        回答「是不是該保真的語義物體」→ 排飛機(語義響應強)

  τ = norm(shallow_hf_energy) × (1 - norm(deep_semantic))

因為 τ 無可學參數,無法「為了降低 WD 而把 σ 調大作弊」——它是輸入影像的
固定函數(如同 BatchNorm 統計量),從根上消除 σ 退化捷徑。

用途
----
1. σ-map: τ 歸一化 → log2_sigma 範圍,取代 static / saliency-σ
2. gate 監督: 你現有 NoiseInjectedResBlock 的可學 gate 頭改為擬合 τ
                (BCE(gate, stopgrad(τ)))  → 解 gate 塌陷 + 保留端到端彈性
3. 頻譜 mask 退居離線 sanity check + 消融 baseline

注意
----
τ 在「淺層 DWT 一級」的分辨率(約原圖 1/2),是局部統計可估性與空間精度的
甜蜜點。進 WD 時交給 WD 自己的 _align_sigma 逐子帶重採樣,不用在這裡對齊。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 基礎算子
# ---------------------------------------------------------------------------
def _box(t, k):
    p = k // 2
    t = F.pad(t, (p, p, p, p), mode='reflect')
    return F.avg_pool2d(t, k, 1, 0)


def _haar1(x):
    """單級正交 Haar DWT。回傳 LL,LH,HL,HH,各半分辨率。與 wa_wd.HaarDWT2D 一致。"""
    if x.shape[-2] % 2:
        x = F.pad(x, (0, 0, 0, 1), mode='replicate')
    if x.shape[-1] % 2:
        x = F.pad(x, (0, 1, 0, 0), mode='replicate')
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    ll = 0.5 * (a + b + c + d)
    lh = 0.5 * (a + b - c - d)
    hl = 0.5 * (a - b + c - d)
    hh = 0.5 * (a - b - c + d)
    return ll, lh, hl, hh


def _robust_norm(x, lo=0.02, hi=0.98, eps=1e-6):
    """
    per-image 穩健歸一化到 [0,1]:用分位數而非 min/max,抗離群值。
    x: [B,1,H,W]
    """
    B = x.shape[0]
    flat = x.flatten(1)
    xlo = flat.quantile(lo, 1).view(B, 1, 1, 1)
    xhi = flat.quantile(hi, 1).view(B, 1, 1, 1)
    return ((x - xlo) / (xhi - xlo + eps)).clamp(0, 1)


# ---------------------------------------------------------------------------
# τ 主函式
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_tau(feats,
                saliency=None,
                shallow_idx=(0, 1),
                deep_idx=(3, 4),
                box_ks=5,
                sal_strength=1.0,
                use_deep=False,
                deep_gate_strength=1.0,
                gamma=1.0,
                return_parts=False):
    """
    從 VGG 多尺度特徵計算 τ 場。

    參數
    ----
    feats : list[Tensor]
        MultiscaleTruncatedVGG16.forward 的輸出。注意其結構是:
        [input_image, (scale0: s1,s2,s3,s4,s5), (scale1: ...), ...]
        亦即 index 0 是原圖,之後每 num_slices(=truncate_slice) 個為一個 scale。
        本函式只用 scale0 的 slices,即 feats[1 : 1+truncate_slice]。
        若你直接傳入「單一 scale 的 5 個 slice list」,設 feats 為那個 list 且
        令 shallow_idx/deep_idx 直接索引它。見下方 _select_scale0。
    saliency : [B,1,H,W] or None
        EMLNet saliency map(飛機顯著、草地/天空不顯著)。
        用作「壓飛機」門控:τ *= (1 - sal_strength * saliency_norm)。
        這是主要的飛機門控 —— 深層語義(use_deep)實測會連草地一起壓,
        故預設關閉,只保留 saliency。saliency=None 則退回純浅層HF(不壓飛機)。
    shallow_idx : 淺層 slice 索引(relu1_2, relu2_2) → 高頻能量,分天空
    deep_idx    : 深層 slice 索引 → 語義門控(預設不用;實測草地>飛機,有害)
    box_ks      : 局部統計窗口
    sal_strength: saliency 門控強度(1.0=全量)
    use_deep    : 是否啟用深層語義門控(預設 False;僅供對照實驗)
    gamma       : >1 銳化 τ 對比
    return_parts: 回傳中間量供視覺化

    設計
    ----
    τ = 浅層HF能量(高,分天空) × (1 - saliency)(非顯著,壓飛機)
      天空: HF≈0            → τ=0
      飛機: HF高 但 sal高    → (1-sal)≈0 → τ=0
      草地: HF高 且 sal低    → 兩項都過   → τ大
    與最初 grass_mask_fn 同結構(hf × not_sal),但 HF 算在 VGG 浅層子帶
    (天空壓得更乾淨),且 saliency 為連續軟值(無手調 quantile 硬閾)。

    回傳
    ----
    tau : [B,1,h,w]  in [0,1],分辨率 = 淺層 DWT 一級(約原圖 1/2)
    """
    # ---- 項1: 淺層高頻能量 ----
    hf_energy = None
    ref_shape = None
    for si in shallow_idx:
        f = feats[si]                              # [B,C,H,W]
        # 逐通道 Haar,取 LH/HL/HH 能量,跨通道平均
        _, lh, hl, hh = _haar1(f)
        e = (lh ** 2 + hl ** 2 + hh ** 2).mean(1, keepdim=True)  # [B,1,H/2,W/2]
        if hf_energy is None:
            hf_energy = e
            ref_shape = e.shape[-2:]
        else:
            # 不同淺層 slice 分辨率可能不同,對齊到第一個
            if e.shape[-2:] != ref_shape:
                e = F.interpolate(e, size=ref_shape, mode='bilinear',
                                  align_corners=False)
            hf_energy = hf_energy + e
    hf_energy = _box(hf_energy, box_ks)
    hf_norm = _robust_norm(hf_energy)

    # ---- 項2(主): saliency 門控 → 壓飛機 ----
    if saliency is not None:
        if saliency.shape[-2:] != ref_shape:
            sal = F.interpolate(saliency, size=ref_shape, mode='bilinear',
                                align_corners=False)
        else:
            sal = saliency
        sal = _box(sal, box_ks)
        sal_norm = _robust_norm(sal)
    else:
        sal_norm = torch.zeros_like(hf_norm)
    sal_gate = (1.0 - sal_strength * sal_norm).clamp(0, 1)

    # ---- 項3(可選,預設關): 深層語義門控 —— 僅供對照 ----
    # 實測:relu4_3/5_3 對草地紋理也強響應,草地 sem > 飛機 sem,
    # 用它壓飛機會連草地一起壓(草地/飛機 → 1.1x)。故預設不用。
    if use_deep:
        sem = None
        for di in deep_idx:
            if di >= len(feats):
                continue
            e = (feats[di] ** 2).mean(1, keepdim=True)
            e = F.interpolate(e, size=ref_shape, mode='bilinear',
                              align_corners=False)
            sem = e if sem is None else sem + e
        if sem is not None:
            sem_norm = _robust_norm(_box(sem, box_ks))
        else:
            sem_norm = torch.zeros_like(hf_norm)
        deep_gate = (1.0 - deep_gate_strength * sem_norm).clamp(0, 1)
    else:
        sem_norm = torch.zeros_like(hf_norm)
        deep_gate = torch.ones_like(hf_norm)

    # ---- 組合: 高頻能量(高) × 非顯著(壓飛機) × [可選]非語義 ----
    tau = hf_norm * sal_gate * deep_gate
    if gamma != 1.0:
        tau = tau.clamp(min=0) ** gamma

    if return_parts:
        return tau, {
            "hf_energy": hf_energy, "hf_norm": hf_norm,
            "sal_norm": sal_norm, "sal_gate": sal_gate,
            "sem_norm": sem_norm, "deep_gate": deep_gate,
            "tau": tau,
        }
    return tau


def _select_scale0(feats, truncate_slice=5):
    """
    從 MultiscaleTruncatedVGG16 的完整輸出中取出 scale0 的 slices。
    feats = [image, s1_0..s5_0, s1_1..s5_1, ...]  → 回傳 [s1_0..s5_0]
    """
    return feats[1: 1 + truncate_slice]


# ---------------------------------------------------------------------------
# τ → log2_sigma 映射(給 WD 用)
# ---------------------------------------------------------------------------
def tau_to_log2_sigma(tau, sigma_min=1.0, sigma_max=16.0):
    """
    把 τ∈[0,1] 映射到 log2_sigma。
      τ=0 (該保真) → log2(sigma_min)   (=0 若 sigma_min=1,純逐點)
      τ=1 (該realism) → log2(sigma_max)
    回傳與 τ 同分辨率的 log2_sigma,WD 內部 _align_sigma 會再逐子帶重採樣。
    """
    import math
    lo = math.log2(sigma_min)
    hi = math.log2(sigma_max)
    return lo + (hi - lo) * tau