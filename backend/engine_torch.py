"""
GPU implementation of the render path (PyTorch: MPS on Apple Silicon, CUDA on
NVIDIA, CPU fallback).

It mirrors engine.process step by step. Two things deliberately stay in the
numpy engine:

  * statistics (veiling light, WB gains, levels, Sea-thru fits): computed on a
    640 px image, cheap, and full of small fits/percentiles that gain nothing
    from a GPU;
  * parameter-independent maps that use OpenCV morphology with elliptical
    kernels (transmission, haze range, depth upsampling, veiling map): computed
    once per image and size, then uploaded. Same code = identical results.

Everything that runs on every slider move, the guided-filter heavy finishing
stages, full-resolution export and the video colour-transfer fit run here.
engine.py stays the reference; tests compare both.
"""
from __future__ import annotations

import math
import threading

import numpy as np
import torch
import torch.nn.functional as F

import engine as E

_dev: torch.device | None = None
GPU_LOCK = threading.Lock()   # MPS does not like concurrent use from several threads


def device() -> torch.device:
    global _dev
    if _dev is None:
        if torch.cuda.is_available():
            _dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            _dev = torch.device("mps")
        else:
            _dev = torch.device("cpu")
    return _dev


def available() -> bool:
    return device().type in ("mps", "cuda")


def sync():
    d = device()
    if d.type == "mps":
        torch.mps.synchronize()
    elif d.type == "cuda":
        torch.cuda.synchronize()


# ----------------------------------------------------------------------------
# tensors <-> numpy (images are 1xCxHxW float32)
# ----------------------------------------------------------------------------
def T(a: np.ndarray) -> torch.Tensor:
    a = np.ascontiguousarray(a, dtype=np.float32)
    t = torch.from_numpy(a)
    t = t[None, None] if t.ndim == 2 else t.permute(2, 0, 1)[None]
    return t.to(device(), non_blocking=True)


def N(t: torch.Tensor) -> np.ndarray:
    t = t[0]
    a = t.permute(1, 2, 0) if t.shape[0] > 1 else t[0]
    return a.float().cpu().numpy()


_LUMA = None


def lum(x: torch.Tensor) -> torch.Tensor:
    global _LUMA
    if _LUMA is None or _LUMA.device != x.device:
        _LUMA = torch.tensor(E.LUMA, device=x.device).view(1, 3, 1, 1)
    return (x * _LUMA).sum(1, keepdim=True)


def vec(v, x: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(np.asarray(v, np.float32), device=x.device).view(1, -1, 1, 1)


# ----------------------------------------------------------------------------
# filters
# ----------------------------------------------------------------------------
def _pad(x: torch.Tensor, l: int, r: int, t: int, b: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    mode = "reflect" if max(l, r) < w and max(t, b) < h else "replicate"
    return F.pad(x, (l, r, t, b), mode=mode)


def box(x: torch.Tensor, r: int) -> torch.Tensor:
    """Normalised (2r+1)^2 box filter, separable."""
    k = 2 * r + 1
    x = F.avg_pool2d(_pad(x, r, r, 0, 0), (1, k), stride=1)
    return F.avg_pool2d(_pad(x, 0, 0, r, r), (k, 1), stride=1)


_gk: dict = {}


def gauss(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable gaussian, kernel radius ~4 sigma like cv2.GaussianBlur((0,0), sigma)."""
    r = max(1, int(round(sigma * 4)))
    key = (r, round(sigma, 4), x.device)
    if key not in _gk:
        g = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=torch.float32) / sigma) ** 2)
        _gk[key] = (g / g.sum()).to(x.device)
    g = _gk[key]
    c = x.shape[1]
    x = F.conv2d(_pad(x, r, r, 0, 0), g.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    return F.conv2d(_pad(x, 0, 0, r, r), g.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def guided(I: torch.Tensor, P: torch.Tensor, r: int, eps: float) -> torch.Tensor:
    """He et al. guided filter with a 1-channel guide, any number of channels in P."""
    mI = box(I, r)
    vI = box(I * I, r) - mI * mI
    mP = box(P, r)
    cov = box(I * P, r) - mI * mP
    a = cov / (vI + eps)
    b = mP - a * mI
    return box(a, r) * I + box(b, r)


# ----------------------------------------------------------------------------
# colour
# ----------------------------------------------------------------------------
def srgb2lin(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin2srgb(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(1e-12) ** (1 / 2.4) - 0.055)


# same constants as OpenCV's RGB<->Lab (D65)
_M = torch.tensor([[0.412453, 0.357580, 0.180423],
                   [0.212671, 0.715160, 0.072169],
                   [0.019334, 0.119193, 0.950227]])
_WN = torch.tensor([0.950456, 1.0, 1.088754])
_MI = torch.linalg.inv(_M)


_mk: dict = {}


def _mat(x, m):
    # 1x1 convolution: much faster than einsum on MPS
    key = (id(m), x.device)
    if key not in _mk:
        _mk[key] = m.to(x.device).view(3, 3, 1, 1).contiguous()
    return F.conv2d(x, _mk[key])


def rgb2lab(srgb):
    xyz = _mat(srgb2lin(srgb), _M) / _WN.to(srgb.device).view(1, 3, 1, 1)
    d = 0.008856
    f = torch.where(xyz > d, xyz.clamp_min(1e-12) ** (1 / 3), 7.787 * xyz + 16 / 116)
    Y = xyz[:, 1:2]
    L = torch.where(Y > d, 116 * f[:, 1:2] - 16, 903.3 * Y)
    a = 500 * (f[:, 0:1] - f[:, 1:2])
    b = 200 * (f[:, 1:2] - f[:, 2:3])
    return torch.cat([L, a, b], 1)


def lab2rgb(lab):
    L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    d = 6 / 29

    def finv(f):
        return torch.where(f > d, f ** 3, (f - 16 / 116) / 7.787)
    Y = torch.where(L > 903.3 * 0.008856, fy ** 3, L / 903.3)
    xyz = torch.cat([finv(fx), Y, finv(fz)], 1) * _WN.to(lab.device).view(1, 3, 1, 1)
    return lin2srgb(_mat(xyz, _MI))


# ----------------------------------------------------------------------------
# parameter-independent maps (numpy engine, once per image and size)
# ----------------------------------------------------------------------------
def _np_lin(img: np.ndarray, cache: dict) -> np.ndarray:
    if "lin" not in cache:
        cache["lin"] = E.u8_to_linear(img) if img.dtype == np.uint8 else E.srgb_to_linear(img)
    return cache["lin"]


def _cached_t(key: str, cache: dict, make):
    tk = "T:" + key
    if tk not in cache:
        cache[tk] = make()
    return cache[tk]


def prepare(img: np.ndarray, p: E.Params, stats: dict, cache: dict):
    """CPU-side, parameter-independent maps. Safe to run in parallel threads,
    outside GPU_LOCK, so the GPU never waits on OpenCV."""
    lin_np = _np_lin(img, cache)
    if p.method == "depth" and ("disp" in cache or "disp_src" in cache):
        import seathru
        from depth import upsample_to
        if "disp" not in cache:
            cache["disp"] = upsample_to(cache["disp_src"], img if img.dtype == np.uint8 else E.linear_to_srgb(lin_np))
        if "bmap" not in cache:
            cache["bmap"] = seathru.veiling_map(lin_np, cache["disp"])
        if cache["bmap"] is not None and "z_haze" not in cache and "bs" in stats:
            cache["z_haze"] = seathru._haze_range(lin_np, cache["bmap"], float(stats["bs"][1][1]))
    elif (p.dehaze > 0 or p.water_protect > 0) and "t" not in cache:
        cache["t"] = E.refine_transmission(lin_np, np.array(stats["A"], np.float32))


# ----------------------------------------------------------------------------
# render
# ----------------------------------------------------------------------------
def process(img: np.ndarray, p: E.Params, stats: dict, cache: dict | None = None,
            want_debug: bool = False, return_tensor: bool = False):
    """GPU render with precomputed `stats` (from engine.Session.stats_for)."""
    if cache is None:
        cache = {}
    with torch.inference_mode():
        lin0 = _cached_t("lin", cache, lambda: T(_np_lin(img, cache)))
        h, w = lin0.shape[-2:]
        short = min(h, w)
        A = np.array(stats["A"], np.float32)

        use_depth = p.method == "depth" and ("disp" in cache or "disp_src" in cache)
        if use_depth:
            lin, t = _front_depth(img, _np_lin(img, cache), lin0, p, stats, cache)
        else:
            lin, t = _front_classic(img, lin0, A, p, stats, cache, short)

        # 5. levels + exposure
        black = vec(stats["levels"][0], lin) * p.auto_levels
        white = 1.0 + (stats["levels"][1] - 1.0) * p.auto_levels
        bmax = float(np.max(np.array(stats["levels"][0]) * p.auto_levels))
        lin = ((lin - black) / max(white - bmax, 1e-3) * (2.0 ** p.exposure)).clamp_min(0)
        knee = 0.8
        x = (lin - knee).clamp_min(0)
        lin = torch.where(lin > knee, knee + (1 - knee) * (x / (x + (1 - knee))), lin)

        # 6. tone
        srgb = lin2srgb(lin)
        if p.highlights != 0 or p.shadows != 0:
            L = gauss(lum(srgb), max(1.0, short * 0.01))
            sh = (1 - L * 2).clamp(0, 1) ** 2
            hl = (L * 2 - 1).clamp(0, 1) ** 2
            srgb = srgb + p.shadows * 0.35 * sh * (1 - srgb) + p.highlights * 0.35 * hl * srgb
        if p.contrast > 0:
            x = srgb.clamp(0, 1)
            srgb = x + p.contrast * 0.6 * (x * x * (3 - 2 * x) - x)
        srgb = srgb.clamp(0, 1)

        # 7-9. Lab
        lab = rgb2lab(srgb)
        L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
        if p.local_contrast > 0 and short >= 64:
            Ln = L / 100.0
            base = guided(Ln, Ln, max(2, int(short * 0.015)), 4e-3)
            detail = torch.tanh((Ln - base) * 6.0) / 6.0
            if t is not None:
                detail = detail * ((t - 0.15) * 2.0).clamp(0, 1)
            L = L + 100.0 * p.local_contrast * 1.6 * detail

        if p.denoise > 0 and short >= 64:
            r = max(1, int(round(short * 0.004 * (0.5 + p.denoise))))
            gL = L / 100.0
            ab = guided(gL, torch.cat([a, b], 1), r, 1e-3 * (0.2 + p.denoise))
            a = a + p.denoise * (ab[:, 0:1] - a)
            b = b + p.denoise * (ab[:, 1:2] - b)
            Ls = guided(gL, gL, max(1, int(round(short * 0.003))), 2e-4 + 1e-3 * p.denoise) * 100.0
            k = 0.5 * p.denoise
            if t is not None:
                k = k + (1.0 - k) * p.denoise * ((0.55 - t) * 2.5).clamp(0, 1)
            L = L + k * (Ls - L)

        if p.vibrance != 0 or p.saturation != 0:
            C = torch.sqrt(a * a + b * b)
            scale = (1.0 + p.vibrance * (1.0 - C / 60.0).clamp(0, 1)) * (1.0 + p.saturation)
            a, b = a * scale, b * scale

        if p.sharpen > 0 and short >= 64:
            blur = gauss(L, max(0.6, short / 2400.0))
            d = L - blur
            d = torch.sign(d) * (d.abs() - 0.4).clamp_min(0)
            L = L + p.sharpen * 1.2 * d

        out = lab2rgb(torch.cat([L.clamp(0, 100), a, b], 1)).clamp(0, 1)
        if p.intensity < 1.0:
            orig = lin2srgb(lin0)
            out = orig + p.intensity * (out - orig)

        if want_debug and t is not None:
            stats["_t"] = N(t)
        if return_tensor:
            return out, stats
        return N(out), stats


def _front_classic(img, lin0, A, p, stats, cache, short):
    t = None
    if p.dehaze > 0 or p.water_protect > 0:
        if "T:t" not in cache and "t" not in cache:
            cache["t"] = E.refine_transmission(_np_lin(img, cache), A)
        t = _cached_t("t", cache, lambda: T(cache["t"]))

    if p.dehaze > 0:
        t_eff = (1.0 - p.dehaze * (1.0 - t)).clamp_min(0.25)
        base = _cached_t("base", cache, lambda: guided(lum(lin0), lin0, max(1, int(short * 0.003)), 4e-4))
        detail = _cached_t("detail", cache, lambda: lin0 - base)
        Av = vec(A, lin0)
        lin = ((base - Av * (1.0 - t_eff)) / t_eff + detail / torch.sqrt(t_eff)).clamp_min(0)
    else:
        lin = lin0.clone()

    mr, mg, mb = stats["means"]
    R, G, B = lin[:, 0:1], lin[:, 1:2], lin[:, 2:3]
    if p.red_comp > 0:
        R = R + p.red_comp * max(mg - mr, 0) * (1.0 - R) * G
    if p.blue_comp > 0:
        B = B + p.blue_comp * max(mg - mb, 0) * (1.0 - B) * G
    lin = torch.cat([R, G, B], 1)

    gains = np.array(stats["gains"], np.float32) * _temp_tint(p)
    if t is not None and p.water_protect > 0:
        if "A_comp" not in stats:
            stats["A_comp"] = E._compensate(A[None, None, :].copy(), p, stats).reshape(3).tolist()
        A_c = np.maximum(np.array(stats["A_comp"], np.float32), 1e-4)
        target = E._hue_to_linear(p.water_hue)
        gains_w = target * float(E.LUMA @ (A_c * gains)) / float(E.LUMA @ target) / A_c
        wgt = p.water_protect * ((1.0 - t) * 1.6 - 0.1).clamp(0, 1)
        g, gw = vec(gains, lin), vec(gains_w, lin)
        lin = lin * (g + (gw - g) * wgt)
    else:
        lin = lin * vec(gains, lin)
    return lin, t


def _temp_tint(p):
    return np.array([2 ** (0.5 * p.temperature), 2 ** (-0.35 * p.tint), 2 ** (-0.5 * p.temperature)], np.float32)


def _front_depth(img, lin_np, lin0, p, stats, cache):
    import seathru
    from depth import upsample_to
    if "disp" not in cache:
        cache["disp"] = upsample_to(cache["disp_src"], img if img.dtype == np.uint8 else E.linear_to_srgb(lin_np))
    disp_np = cache["disp"]
    if "bmap" not in cache:
        cache["bmap"] = seathru.veiling_map(lin_np, disp_np)
    bmap_np = cache["bmap"]
    prm = np.array(stats["bs"], np.float32)
    ratio = bool(stats.get("bs_ratio")) and bmap_np is not None
    if bmap_np is not None and "z_haze" not in cache:
        cache["z_haze"] = seathru._haze_range(lin_np, bmap_np, float(prm[1, 1]))

    disp = _cached_t("disp", cache, lambda: T(disp_np))
    zn, zf = max(0.1, p.z_near), max(p.z_near + 0.5, p.z_far)
    z = 1.0 / (1.0 / zf + disp * (1.0 / zn - 1.0 / zf))

    if bmap_np is not None:
        bmap = _cached_t("bmap", cache, lambda: T(bmap_np))
        z_haze = _cached_t("z_haze", cache, lambda: T(cache["z_haze"]))
        far_w = ((0.5 - disp) / 0.3).clamp(0, 1)
        z = z + (torch.maximum(z, z_haze) - z) * far_w

    pr = torch.as_tensor(prm, device=z.device)
    B = torch.cat([pr[c, 0] * (1 - torch.exp(-pr[c, 1] * z)) + pr[c, 2] * torch.exp(-pr[c, 3] * z)
                   for c in range(3)], 1)
    if ratio:
        B = B * bmap
    t = torch.exp(-pr[1, 1] * z)

    if bmap_np is not None:
        def _wl():
            lr = torch.log(lin0.clamp_min(1e-4) / bmap.clamp_min(1e-4))
            dist = torch.sqrt((lr ** 2).mean(1, keepdim=True))
            wl = torch.exp(-(dist / 0.12) ** 2)
            return gauss(wl, max(1.0, min(z.shape[-2:]) * 0.004))
        wl = _cached_t("wl", cache, _wl)
        B = B * (1 - wl) + bmap * wl
        t = t * (1 - wl)
    else:
        wl = torch.zeros_like(t)

    strength = p.dehaze + (1.0 - p.dehaze) * wl
    D = (lin0 - strength * B).clamp_min(0)

    il = stats["illum"]
    coef = np.array(il["coef"], np.float64)
    zc = z.clamp(il["z_lo"], il["z_hi"])
    gs = []
    for c in range(3):
        cc = coef[c]
        logE = torch.zeros_like(zc)
        for q in cc:                       # Horner, same as np.polyval
            logE = logE * zc + float(q)
        ref = float(np.polyval(cc, il["z_lo"]))
        gs.append(torch.exp((ref - logE).clamp(0, math.log(6.0))))
    G = 1.0 + (torch.cat(gs, 1) - 1.0) * (1.0 - wl)
    J = D * G ** float(np.clip(p.red_comp, 0, 2))

    gains = np.array(stats["gains"], np.float32) * _temp_tint(p)
    J = J * vec(gains, J)

    if p.water_protect > 0:
        if ratio:
            if "bmap_med" not in cache:
                cache["bmap_med"] = np.median(bmap_np.reshape(-1, 3), axis=0)
            b_inf = cache["bmap_med"] * prm[:, 0]
            veil = bmap * vec(prm[:, 0], J)
        else:
            b_inf = prm[:, 0]
            veil = vec(b_inf, J)
        target = E._hue_to_linear(p.water_hue)
        gains_w = target * float(E.LUMA @ (b_inf * gains)) / float(E.LUMA @ target) / np.maximum(b_inf, 1e-4)
        water = veil * vec(gains_w, J)
        haze = ((1.0 - t) * (0.5 + p.water_protect)).clamp(0, 1) * strength
        J = J * (1 - haze) + water * haze
    return J.clamp_min(0), t


# ----------------------------------------------------------------------------
# video: local colour transform (guided upsampling of the correction)
# ----------------------------------------------------------------------------
def fit_local_affine(src: torch.Tensor, dst: torch.Tensor, r: int, eps: float) -> torch.Tensor:
    """3x4 per-pixel colour transform dst ~= M [src, 1]; returns 1x12xHxW."""
    mI = box(src, r)
    mP = box(dst, r)
    I = [src[:, i:i + 1] for i in range(3)]
    m = [mI[:, i:i + 1] for i in range(3)]
    c = {}
    for i in range(3):
        for j in range(i, 3):
            c[i, j] = box(I[i] * I[j], r) - m[i] * m[j]
    a00, a01, a02 = c[0, 0] + eps, c[0, 1], c[0, 2]
    a11, a12, a22 = c[1, 1] + eps, c[1, 2], c[2, 2] + eps
    # closed-form inverse of the symmetric 3x3 covariance
    i00 = a11 * a22 - a12 * a12
    i01 = a02 * a12 - a01 * a22
    i02 = a01 * a12 - a02 * a11
    i11 = a00 * a22 - a02 * a02
    i12 = a01 * a02 - a00 * a12
    i22 = a00 * a11 - a01 * a01
    det = a00 * i00 + a01 * i01 + a02 * i02
    inv = [[i00, i01, i02], [i01, i11, i12], [i02, i12, i22]]
    rows = []
    for ch in range(3):
        P = dst[:, ch:ch + 1]
        mp = mP[:, ch:ch + 1]
        x = [box(P * I[k], r) - mp * m[k] for k in range(3)]
        a = [(x[0] * inv[0][k] + x[1] * inv[1][k] + x[2] * inv[2][k]) / det for k in range(3)]
        b = mp - a[0] * m[0] - a[1] * m[1] - a[2] * m[2]
        rows += a + [b]
    return box(torch.cat(rows, 1), r)


def apply_local_affine(img: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
    h, w = img.shape[-2:]
    up = F.interpolate(coef, size=(h, w), mode="bilinear", align_corners=False)
    out = []
    for c in range(3):
        o = c * 4
        out.append(up[:, o:o + 1] * img[:, 0:1] + up[:, o + 1:o + 2] * img[:, 1:2]
                   + up[:, o + 2:o + 3] * img[:, 2:3] + up[:, o + 3:o + 4])
    return torch.cat(out, 1).clamp(0, 1)


# ----------------------------------------------------------------------------
# all-GPU helpers for video frames (no OpenCV round trips)
# ----------------------------------------------------------------------------
def _erode(x, k):
    r = k // 2
    x = -F.max_pool2d(_pad(-x, r, r, 0, 0), (1, k), stride=1)
    return -F.max_pool2d(_pad(-x, 0, 0, r, r), (k, 1), stride=1)


def _dilate(x, k):
    r = k // 2
    x = F.max_pool2d(_pad(x, r, r, 0, 0), (1, k), stride=1)
    return F.max_pool2d(_pad(x, 0, 0, r, r), (k, 1), stride=1)


def refine_transmission(lin: torch.Tensor, A) -> torch.Tensor:
    """GPU version of engine.refine_transmission. The elliptical opening is
    replaced by a square of equal area (separable max-pooling)."""
    h, w = lin.shape[-2:]
    short = min(h, w)
    k = max(3, int(short * 0.012) | 1)
    k = max(3, int(round(k * math.sqrt(math.pi) / 2)) | 1)
    norm = torch.minimum(lin[:, 1:2] / float(A[1]), lin[:, 2:3] / float(A[2]))
    dark = torch.minimum(_dilate(_erode(norm, k), k), norm)
    t = (1.0 - 0.95 * dark).clamp(0, 1)
    pix = 1.0 - norm
    g = 0.6 * pix.clamp(-1, 1) + 0.4 * lin2srgb(lum(lin))
    t1 = guided(g, t, max(2, int(short * 0.012)), 1e-3)
    t2 = guided(g, t1, max(2, int(short * 0.004)), 1e-4)
    t_s = gauss(t1, short * 0.03)
    wgt = ((0.6 - t_s) / 0.4).clamp(0, 1)
    return (t2 + (t_s - t2) * wgt).clamp(0, 1)


def upload_rgb48(raw: bytes, h: int, w: int) -> torch.Tensor:
    """rgb48le bytes -> 1x3xHxW float in [0,1], converted on the GPU."""
    a = np.frombuffer(raw, np.int16).reshape(h, w, 3)          # uint16 bits viewed as int16
    t = torch.from_numpy(a).to(device()).permute(2, 0, 1)[None].to(torch.int32)
    return ((t & 0xFFFF).float() * (1.0 / 65535.0)).contiguous()


def download_rgb48(x: torch.Tensor) -> bytes:
    """1x3xHxW float in [0,1] -> rgb48le bytes, quantised on the GPU."""
    v = (x[0].permute(1, 2, 0) * 65535.0 + 0.5).clamp(0, 65535).to(torch.int32)
    v = torch.where(v > 32767, v - 65536, v).to(torch.int16)  # same bits as uint16
    return v.contiguous().cpu().numpy().tobytes()


def render_video_frame(full: torch.Tensor, work_hw: tuple, p: E.Params, stats: dict,
                       disp_src: np.ndarray | None = None) -> torch.Tensor:
    """Whole frame on the GPU: area-downscale, render at work size, fit the
    local colour transform and apply it at full size."""
    small = F.interpolate(full, size=work_hw, mode="area") if tuple(full.shape[-2:]) != tuple(work_hw) else full
    cache: dict = {}
    if p.method == "depth" and disp_src is not None:
        cache["disp_src"] = disp_src
        img_np = N(small)
        prepare(img_np, p, stats, cache)
        out_s, _ = process(img_np, p, stats, cache=cache, return_tensor=True)
    else:
        lin = srgb2lin(small)
        cache["T:lin"] = lin
        if p.dehaze > 0 or p.water_protect > 0:
            cache["T:t"] = refine_transmission(lin, stats["A"])
        out_s, _ = process(None, p, stats, cache=cache, return_tensor=True)
    if small is full:
        return out_s
    coef = fit_local_affine(small, out_s, 2, 1e-4)
    return apply_local_affine(full, coef)
