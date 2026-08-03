"""
mask_utils.py

軟「隨機紋理」mask (路線 A++, 頻譜判準版)。

與舊的 grass_mask_fn 的差別:
  - 目標從「草地(語意)」改成「隨機紋理(頻譜性質)」:精確像素排列在感知上不重要的區域。
  - 判準 = HF能量(高) × 非週期(低 periodicity) × 非取向(低 coherence) × 非顯著(選用)。
  - 輸出 [0,1] 連續軟值,直接當 noise_scale 的空間調變係數(不二值化,保留過渡帶)。
  - 訓練時載入、推論不用;一次性掃訓練集存下即可,無 gate 崩塌問題。

保留舊的 grass_mask_fn 以便並排比對 / 回退。
"""
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# 基礎算子
# ---------------------------------------------------------------------------
def _box(t, k):
    """reflect-pad 的 box filter (等價局部均值)。"""
    p = k // 2
    t = F.pad(t, (p, p, p, p), mode='reflect')
    return F.avg_pool2d(t, k, 1, 0)


def _shift(r, dy, dx):
    """把 r 平移 (dy,dx),reflect pad 保形。rs[y,x] = r[y+dy, x+dx]。"""
    B, C, H, W = r.shape
    py, px = abs(dy), abs(dx)
    rp = F.pad(r, (px, px, py, py), mode='reflect')
    return rp[:, :, py + dy:py + dy + H, px + dx:px + dx + W]


def _local_periodicity(gray, res_ks=7, win=15, shifts=None):
    """
    局部正規化自相關的最大離心峰值。
      隨機紋理(草/礫石/樹葉) → 任何非零位移的相關都掉到 ~0 → 值低
      週期結構(柵欄/磚牆/欄杆/面板線) → 在週期位移處相關 ≈1 → 值高
    回傳 [B,1,H,W],高 = 結構。
    """
    if shifts is None:
        # 最小位移從 4px 起:草葉等細密隨機紋理的自相關在 4px 外已衰減到 ~0,
        # 而柵欄/鉚釘線等真週期(6-8px)在這些位移處仍有峰值 → 兩者分開。
        # (原本含 2px 位移會被草地的短程相關污染,誤報高 periodicity。)
        shifts = [(0, 4), (4, 0), (4, 4), (4, -4),
                  (0, 6), (6, 0), (6, 6), (6, -6),
                  (0, 8), (8, 0), (5, 8), (8, 5)]
    r = gray - _box(gray, res_ks)                 # 高通殘差
    mu = _box(r, win)
    var = (_box(r * r, win) - mu * mu).clamp(min=1e-6)
    best = torch.zeros_like(gray)
    for dy, dx in shifts:
        rs = _shift(r, dy, dx)
        mus = _box(rs, win)
        vs = (_box(rs * rs, win) - mus * mus).clamp(min=1e-6)
        cov = _box(r * rs, win) - mu * mus
        corr = cov / (var.sqrt() * vs.sqrt())
        best = torch.maximum(best, corr.clamp(min=0))
    return best


def _structure_anisotropy(gray, ks=7, smooth_ks=15):
    """
    structure tensor coherence:有明確取向的結構(柵欄/欄杆/牆線)→ 高;各向同性紋理 → 低。
    """
    gx = F.pad(gray[..., :, 1:] - gray[..., :, :-1], (0, 1, 0, 0))
    gy = F.pad(gray[..., 1:, :] - gray[..., :-1, :], (0, 0, 0, 1))
    Jxx = _box(gx * gx, smooth_ks)
    Jyy = _box(gy * gy, smooth_ks)
    Jxy = _box(gx * gy, smooth_ks)
    tmp = ((Jxx - Jyy) ** 2 + 4 * Jxy ** 2).clamp(min=0).sqrt()
    l1 = 0.5 * (Jxx + Jyy + tmp)
    l2 = 0.5 * (Jxx + Jyy - tmp)
    coh = (l1 - l2) / (l1 + l2 + 1e-6)
    return _box(coh, smooth_ks)


def _soft_norm(x, pct=0.5, temp=0.2):
    """
    per-image 軟正規化:以分位數為中心、IQR 為尺度的 sigmoid,取代硬 quantile 門檻。
    回傳 [0,1]。temp 越小越接近硬門檻。
    """
    B = x.shape[0]
    flat = x.flatten(1)
    c = flat.quantile(pct, 1).view(B, 1, 1, 1)
    q1 = flat.quantile(0.25, 1).view(B, 1, 1, 1)
    q3 = flat.quantile(0.75, 1).view(B, 1, 1, 1)
    spread = (q3 - q1).clamp(min=1e-6)
    return torch.sigmoid((x - c) / (temp * spread))


# ---------------------------------------------------------------------------
# 主函式:軟隨機紋理 mask
# ---------------------------------------------------------------------------
def stochastic_mask_fn(target, saliency_model=None,
                       blur_ks=15, hf_pct=0.5, hf_temp=0.2,
                       peri_win=15, peri_pct=0.5, peri_temp=0.2,
                       aniso_ks=7, aniso_pct=0.5, aniso_temp=0.2, use_aniso=True,
                       sal_pct=0.5, sal_temp=0.2,
                       smooth_ks=9, gamma=1.0, return_parts=False):
    """
    軟隨機紋理 mask = HF能量(高) × 非週期(低 periodicity) × 非取向(低 coherence) × 非顯著(選用)

    參數
    ----
    target        : [B,3,H,W] in [0,1]
    saliency_model: 選用。給了就多乘一項 not-saliency;None 則跳過(不依賴 pretrained 詞表)。
    blur_ks       : HF 能量的局部尺度。
    *_pct         : 各分量軟門檻的中心分位數(0.5=中位數)。
    *_temp        : sigmoid 溫度,越小越硬。
    peri_win      : 自相關窗口。
    use_aniso     : 是否加 coherence 這一項(建議 True,與 periodicity 互補)。
    smooth_ks     : 最終軟平滑(取代二值 erode/dilate,保留過渡帶)。
    gamma         : >1 銳化對比,=1 不變。
    return_parts  : True 則回傳中間量供除錯 / 視覺化。

    回傳
    ----
    score [B,1,H,W] in [0,1]:直接當 noise_scale 的空間調變係數。
    """
    B, _, H, W = target.shape
    gray = target.mean(1, keepdim=True)

    # 1) HF 能量:要高 → 排天空 / 平滑機身
    low = _box(gray, blur_ks)
    high = _box((gray - low).abs(), blur_ks)
    hf_soft = _soft_norm(high, hf_pct, hf_temp)

    # 2) 週期性:要低 → 排柵欄 / 磚牆 / 欄杆 / 面板線
    peri = _local_periodicity(gray, win=peri_win)
    notperi = 1.0 - _soft_norm(peri, peri_pct, peri_temp)

    score = hf_soft * notperi

    # 3) 取向性(coherence):要低 → 補強有方向的規律結構
    if use_aniso:
        aniso = _structure_anisotropy(gray, aniso_ks, aniso_ks * 2 + 1)
        notaniso = 1.0 - _soft_norm(aniso, aniso_pct, aniso_temp)
        score = score * notaniso
    else:
        aniso = torch.zeros_like(gray)
        notaniso = torch.ones_like(gray)

    # 4) 顯著性:選用保險。HF+結構通常已能排平滑機身+面板線。
    if saliency_model is not None:
        sal = saliency_model(target)
        sal = TF.gaussian_blur(sal, kernel_size=[31, 31], sigma=[5.0, 5.0])
        notsal = 1.0 - _soft_norm(sal, sal_pct, sal_temp)
        score = score * notsal
    else:
        sal = torch.zeros_like(gray)
        notsal = torch.ones_like(gray)

    # 軟平滑(保留過渡帶),可選 gamma 銳化
    score = _box(score, smooth_ks)
    if gamma != 1.0:
        score = score.clamp(min=0) ** gamma

    # 邊框清零(避免邊界殘影)
    m = max(blur_ks // 2, 8)
    score[:, :, :m, :] = 0
    score[:, :, -m:, :] = 0
    score[:, :, :, :m] = 0
    score[:, :, :, -m:] = 0

    if return_parts:
        return score, {
            "high": high, "hf_soft": hf_soft,
            "peri": peri, "notperi": notperi,
            "aniso": aniso, "notaniso": notaniso,
            "sal": sal, "notsal": notsal,
            "score": score, "border_m": m,
        }
    return score


# ---------------------------------------------------------------------------
# 舊版:草地(語意/啟發式) mask —— 保留供並排比對 / 回退
# ---------------------------------------------------------------------------
def _clean_mask(m, ks=5):
    p = ks // 2
    er = -F.max_pool2d(-m, ks, 1, p)
    di = F.max_pool2d(er, ks, 1, p)
    return di


def grass_mask_fn(target, saliency_model, blur_ks=15, hf_pct=0.5, sal_pct=0.5,
                  aniso_ks=7, aniso_pct=0.5, use_aniso=True, return_parts=False):
    """
    (舊) 硬草地 mask = high-freq(排天空) AND not-saliency(排飛機) AND isotropic(排柵欄/牆)。
    保留以便與 stochastic_mask_fn 並排比對。
    """
    B, _, H, W = target.shape
    pad = blur_ks // 2

    def _reflect_avgpool(x):
        x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        return F.avg_pool2d(x, blur_ks, 1, 0)

    gray = target.mean(1, keepdim=True)

    low = _reflect_avgpool(gray)
    high = _reflect_avgpool((gray - low).abs())
    hf_thr = high.flatten(1).quantile(hf_pct, 1).view(B, 1, 1, 1)
    hf_mask = (high > hf_thr).float()

    sal = saliency_model(target)
    sal = TF.gaussian_blur(sal, kernel_size=[31, 31], sigma=[5.0, 5.0])
    sal_thr = sal.flatten(1).quantile(sal_pct, 1).view(B, 1, 1, 1)
    not_sal = (sal < sal_thr).float()

    if use_aniso:
        aniso = _structure_anisotropy(gray, ks=aniso_ks, smooth_ks=aniso_ks * 2 + 1)
        aniso_thr = aniso.flatten(1).quantile(aniso_pct, 1).view(B, 1, 1, 1)
        iso_mask = _clean_mask((aniso < aniso_thr).float())
    else:
        aniso = torch.zeros_like(hf_mask)
        iso_mask = torch.ones_like(hf_mask)

    mask = hf_mask * not_sal * iso_mask

    m = max(pad, 8)
    mask[:, :, :m, :] = 0
    mask[:, :, -m:, :] = 0
    mask[:, :, :, :m] = 0
    mask[:, :, :, -m:] = 0

    if return_parts:
        return mask, {
            "high": high, "hf_mask": hf_mask,
            "sal": sal, "not_sal": not_sal,
            "aniso": aniso, "iso_mask": iso_mask,
            "border_m": m,
        }
    return mask