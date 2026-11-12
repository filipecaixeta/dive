"""
Underwater color correction engine.

Pipeline (all heavy math in float32, physically-motivated steps in linear light):

  1. Veiling light estimation   quad-tree search for the smooth, blue/green water region
  2. Backscatter removal         underwater dark channel (G,B, morphological opening) + guided refinement
  3. Channel compensation        Ancuti-style red (and optional blue) compensation from green
  4. White balance               auto (foreground-weighted shades-of-gray) or picked white point
  5. Levels + exposure           per-channel black point, common white point, soft highlight shoulder
  6. Tone                        sRGB encode + S-curve contrast
  7. Local contrast              edge-aware clarity on L* (guided-filter base/detail)
  8. Color                       vibrance / saturation on chroma
  9. Detail                      chroma denoise (guided by L*), luminance unsharp mask
 10. Global intensity            blend with the original

Statistics (veiling light, means, WB gains, levels) are computed once on a small
analysis image and reused when rendering any resolution, so the preview and the
full-resolution export look the same.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
import numpy as np
import cv2

ANALYSIS_SIZE = 1024   # veiling light + raw transmission
STATS_SIZE = 640       # slider-dependent statistics (WB, levels), recomputed on every change


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
@dataclass
class Params:
    intensity: float = 1.0        # 0..1   global blend with the original
    dehaze: float = 0.7           # 0..1   backscatter removal
    water_protect: float = 0.5    # 0..1   how much the open water follows water_hue instead of the WB
    water_hue: float = 198.0      # deg    target hue for open water (170 teal .. 230 deep blue)
    red_comp: float = 1.0         # 0..2   red channel compensation
    blue_comp: float = 0.0        # 0..2   blue channel compensation (green water)
    wb_mode: str = "auto"         # auto | pick | none
    wb_x: float = 0.5             # picked white point (normalized)
    wb_y: float = 0.5
    temperature: float = 0.0      # -1..1
    tint: float = 0.0             # -1..1
    auto_levels: float = 1.0      # 0..1
    exposure: float = 0.0         # EV, -2..2
    contrast: float = 0.25        # 0..1
    highlights: float = 0.0       # -1..1   (negative recovers highlights)
    shadows: float = 0.0          # -1..1   (positive lifts shadows)
    local_contrast: float = 0.35  # 0..1
    vibrance: float = 0.3         # -1..1
    saturation: float = 0.0       # -1..1
    denoise: float = 0.3          # 0..1
    sharpen: float = 0.3          # 0..1
    skin_strength: float = 0.8    # 0..1   diver skin-tone correction (only where divers were clicked)
    skin_feather: float = 0.3     # 0..1   softness of the diver mask edge
    skin_body: float = 0.5        # 0..1   0 = only the skin, 1 = the whole diver evenly
    method: str = "classic"       # classic (dark channel) | depth (Sea-thru with monocular depth)
    z_near: float = 0.6           # m, nearest visible distance (depth method)
    z_far: float = 12.0           # m, farthest visible distance (depth method)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Params":
        d = d or {}
        names = {f.name: f.type for f in fields(cls)}
        kw = {}
        for k, v in d.items():
            if k not in names or v is None:
                continue
            kw[k] = str(v) if k in ("wb_mode", "method") else float(v)
        return cls(**kw)

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------
# Color helpers
# ----------------------------------------------------------------------------
_SRGB_TO_LIN_U8 = np.array(
    [((i / 255) / 12.92) if i / 255 <= 0.04045 else (((i / 255) + 0.055) / 1.055) ** 2.4 for i in range(256)],
    dtype=np.float32,
)
_LIN_TO_SRGB_N = 4096
_LIN_TO_SRGB_LUT = np.array(
    [12.92 * x if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055
     for x in np.linspace(0, 1, _LIN_TO_SRGB_N)],
    dtype=np.float32,
)
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def u8_to_linear(img_u8: np.ndarray) -> np.ndarray:
    return _SRGB_TO_LIN_U8[img_u8]


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    f = x * (_LIN_TO_SRGB_N - 1)
    i0 = f.astype(np.int32)
    i1 = np.minimum(i0 + 1, _LIN_TO_SRGB_N - 1)
    w = f - i0
    return _LIN_TO_SRGB_LUT[i0] * (1 - w) + _LIN_TO_SRGB_LUT[i1] * w


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def luminance(x: np.ndarray) -> np.ndarray:
    return x @ LUMA


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He et al. guided filter, gray guide, 1- or multi-channel src."""
    ksize = (2 * r + 1, 2 * r + 1)

    def box(a):
        return cv2.boxFilter(a, -1, ksize, borderType=cv2.BORDER_REFLECT)

    mean_I = box(guide)
    var_I = box(guide * guide) - mean_I * mean_I
    if src.ndim == 3:
        out = np.empty_like(src)
        for c in range(src.shape[2]):
            out[..., c] = guided_filter(guide, np.ascontiguousarray(src[..., c]), r, eps)
        return out
    mean_p = box(src)
    cov_Ip = box(guide * src) - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return box(a) * guide + box(b)


def resize_long(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = size / max(h, w)
    if s >= 1:
        return img.copy()
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------
# Analysis pieces
# ----------------------------------------------------------------------------
def estimate_veiling_light(lin: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Quad-tree search for the region that looks most like open water:
    bright in G/B, little red, low texture. Returns (A rgb linear, bbox)."""
    gb = np.maximum(lin[..., 1], lin[..., 2])
    score_map = gb - lin[..., 0]               # water absorbs red
    lum = luminance(lin)
    y0, x0, y1, x1 = 0, 0, lin.shape[0], lin.shape[1]
    while (y1 - y0) * (x1 - x0) > 32 * 32:
        hy, hx = (y0 + y1) // 2, (x0 + x1) // 2
        best, best_q = -1e9, None
        for q in ((y0, x0, hy, hx), (y0, hx, hy, x1), (hy, x0, y1, hx), (hy, hx, y1, x1)):
            sy0, sx0, sy1, sx1 = q
            s = score_map[sy0:sy1, sx0:sx1]
            l = lum[sy0:sy1, sx0:sx1]
            g = gb[sy0:sy1, sx0:sx1]
            val = s.mean() + 0.5 * g.mean() - 3.0 * l.std()
            if val > best:
                best, best_q = val, q
        y0, x0, y1, x1 = best_q
    region = lin[y0:y1, x0:x1].reshape(-1, 3)
    A = np.median(region, axis=0).astype(np.float32)
    return np.clip(A, 1e-3, 1.0), (y0, x0, y1, x1)


def raw_transmission(lin: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Underwater dark channel prior on G and B (red is useless underwater).

    Uses a morphological *opening* (min then max) instead of the classic plain
    min-filter: the min-filter grows every thin dark branch by the patch size,
    which shows up as a bright halo of "near" water around it after dehazing.
    The opening keeps the bright-object rescue of the dark channel without
    spreading dark values beyond the object."""
    short = min(lin.shape[:2])
    k = max(3, int(short * 0.012) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    norm = np.minimum(lin[..., 1] / A[1], lin[..., 2] / A[2]).astype(np.float32)
    dark = cv2.dilate(cv2.erode(norm, se), se)
    dark = np.minimum(dark, norm)  # never claim more haze than the pixel itself shows
    return np.clip(1.0 - 0.95 * dark, 0.0, 1.0).astype(np.float32)


def refine_transmission(guide_lin: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Transmission at this resolution: raw opening-based estimate, then
    edge-aware smoothing guided by the image itself."""
    h, w = guide_lin.shape[:2]
    short = min(h, w)
    t = raw_transmission(guide_lin, A)
    # guide = per-pixel "how unlike the water" + luminance; luminance alone
    # can't separate a hydroid from the water behind it
    pix = 1.0 - np.minimum(guide_lin[..., 1] / A[1], guide_lin[..., 2] / A[2])
    lum = linear_to_srgb(luminance(guide_lin))
    g = (0.6 * np.clip(pix, -1, 1) + 0.4 * lum).astype(np.float32)
    t1 = guided_filter(g, t, max(2, int(short * 0.012)), 1e-3)
    t2 = guided_filter(g, t1, max(2, int(short * 0.004)), 1e-4)
    # in open water there is nothing to snap to, and any wobble in t becomes
    # blotches after inversion: use a very smooth map there
    t_s = cv2.GaussianBlur(t1, (0, 0), short * 0.03)
    wgt = np.clip((0.6 - t_s) / 0.4, 0, 1)
    t = t2 + (t_s - t2) * wgt
    return np.clip(t, 0.0, 1.0)


# ----------------------------------------------------------------------------
# The pipeline
# ----------------------------------------------------------------------------
def process(img_u8: np.ndarray, p: Params, stats: dict | None = None,
            spatial: bool = True, cache: dict | None = None,
            want_debug: bool = False, stats_only: bool = False) -> tuple[np.ndarray | None, dict]:
    """Run the pipeline. If `stats` is empty it is filled from this image
    (call that on the analysis image first, then reuse for other sizes).

    `cache` holds parameter-independent intermediates for *this* image size
    (linear image, transmission map, base/detail split) so that moving a
    slider only re-runs the cheap parts.

    Returns (srgb float32 HxWx3 in [0,1], stats)."""
    if stats is None:
        stats = {}
    if cache is None:
        cache = {}
    if "lin" not in cache:
        cache["lin"] = u8_to_linear(img_u8) if img_u8.dtype == np.uint8 else srgb_to_linear(img_u8)
    lin = cache["lin"]
    orig_lin = lin
    h, w = lin.shape[:2]

    # 1-2. Backscatter removal ------------------------------------------------
    if "A" not in stats:
        A, bbox = estimate_veiling_light(lin)
        stats["A"] = A.tolist()
        stats["A_bbox"] = [bbox[0] / h, bbox[1] / w, bbox[2] / h, bbox[3] / w]
    A = np.array(stats["A"], dtype=np.float32)

    use_depth = p.method == "depth" and spatial and ("disp" in cache or "disp_src" in cache)
    if use_depth:
        # Sea-thru style front end: needs a disparity map for this image size
        import seathru
        if "disp" not in cache:
            from depth import upsample_to
            cache["disp"] = upsample_to(cache["disp_src"], img_u8 if img_u8.dtype == np.uint8 else linear_to_srgb(lin))
        z = _range_map(cache["disp"], p)
        lin, t = seathru.front(lin.copy(), z, cache["disp"], p, stats, cache, _wb_gains, _hue_to_linear)
    else:
        t = None
        if spatial and (p.dehaze > 0 or p.water_protect > 0):
            if "t" not in cache:
                cache["t"] = refine_transmission(lin, A)
            t = cache["t"]

        if spatial and p.dehaze > 0:
            t_eff = 1.0 - p.dehaze * (1.0 - t)
            t_eff = np.maximum(t_eff, 0.25)[..., None]
            # Base/detail split: the haze model is inverted on an edge-preserving
            # base layer, while fine detail (sensor noise, compression) is boosted
            # only by sqrt(1/t). Keeps open water clean instead of amplifying noise 3-4x.
            if "base" not in cache:
                g = luminance(lin).astype(np.float32)
                r = max(1, int(min(h, w) * 0.003))
                cache["base"] = guided_filter(g, lin, r, 4e-4)
                cache["detail"] = lin - cache["base"]
            base, detail = cache["base"], cache["detail"]
            lin = (base - A * (1.0 - t_eff)) / t_eff + detail / np.sqrt(t_eff)
            lin = np.clip(lin, 0.0, None)
        else:
            lin = lin.copy()  # later stages work in place; never touch the cache

        # 3. Channel compensation (Ancuti et al. 2018) ----------------------------
        if "means" not in stats:
            stats["means"] = lin.reshape(-1, 3).mean(axis=0).tolist()
        lin = _compensate(lin, p, stats)

        # 4. White balance --------------------------------------------------------
        if "gains" not in stats:
            stats["gains"] = _wb_gains(lin, p, stats, t).tolist()
        gains = np.array(stats["gains"], dtype=np.float32)
        gains = gains * np.array([2 ** (0.5 * p.temperature), 2 ** (-0.35 * p.tint), 2 ** (-0.5 * p.temperature)],
                                 dtype=np.float32)
        if t is not None and p.water_protect > 0:
            # Open water (low t) is pulled toward a chosen blue instead of being
            # neutralized by the WB (which turns it gray/violet).
            if "A_comp" not in stats:
                stats["A_comp"] = _compensate(A[None, None, :].copy(), p, stats).reshape(3).tolist()
            A_c = np.maximum(np.array(stats["A_comp"], np.float32), 1e-4)
            target = _hue_to_linear(p.water_hue)
            gains_w = target * float(LUMA @ (A_c * gains)) / float(LUMA @ target) / A_c
            wgt = (p.water_protect * np.clip((1.0 - t) * 1.6 - 0.1, 0, 1))[..., None]
            lin = lin * (gains + (gains_w - gains) * wgt)
        else:
            lin = lin * gains


    # 5. Levels + exposure ----------------------------------------------------
    if "levels" not in stats:
        flat = lin.reshape(-1, 3)
        sub = flat[:: max(1, flat.shape[0] // 200_000)]
        black = np.percentile(sub, 0.15, axis=0)
        white = np.percentile(sub.max(axis=1), 99.7)
        stats["levels"] = [black.tolist(), float(white)]
    if stats_only:
        return None, stats
    black = np.array(stats["levels"][0], dtype=np.float32) * p.auto_levels
    white = 1.0 + (stats["levels"][1] - 1.0) * p.auto_levels
    lin = (lin - black) / max(white - black.max(), 1e-3)
    lin = lin * (2.0 ** p.exposure)
    lin = np.clip(lin, 0.0, None)

    # highlight shoulder: smoothly compress anything above the knee
    knee = 0.8
    over = lin > knee
    if over.any():
        x = lin[over] - knee
        lin[over] = knee + (1 - knee) * (x / (x + (1 - knee)))

    # 6. Tone -----------------------------------------------------------------
    srgb = linear_to_srgb(lin)
    if p.highlights != 0 or p.shadows != 0:
        L = luminance(srgb)[..., None]
        if spatial:
            L = cv2.GaussianBlur(L[..., 0], (0, 0), max(1.0, min(h, w) * 0.01))[..., None]
        sh = np.clip(1 - L * 2, 0, 1) ** 2
        hl = np.clip(L * 2 - 1, 0, 1) ** 2
        srgb = srgb + p.shadows * 0.35 * sh * (1 - srgb) + p.highlights * 0.35 * hl * srgb
    if p.contrast > 0:
        x = np.clip(srgb, 0, 1)
        s_curve = x * x * (3 - 2 * x)
        srgb = x + p.contrast * 0.6 * (s_curve - x)
    srgb = np.clip(srgb, 0.0, 1.0).astype(np.float32)

    # 7-9. Lab: local contrast, chroma, detail -------------------------------
    lab = cv2.cvtColor(srgb, cv2.COLOR_RGB2Lab)  # L 0..100, a/b ~ -128..127
    L = lab[..., 0]

    if spatial and p.local_contrast > 0 and min(h, w) >= 64:
        # Clarity: boost mid-frequency detail around an edge-preserving base.
        # (CLAHE was tried first: it re-maps tones per tile and puts bright
        # halos around every speck floating in the water.)
        Ln = (L / 100.0).astype(np.float32)
        short = min(h, w)
        base = guided_filter(Ln, Ln, max(2, int(short * 0.015)), 4e-3)
        detail = Ln - base
        # protect smooth water: little detail there, so the gain is small anyway,
        # and soft-limit large detail to avoid crunchy edges
        detail = np.tanh(detail * 6.0) / 6.0
        if t is not None:
            # open water has no detail to gain, only noise
            detail = detail * np.clip((t - 0.15) * 2.0, 0.0, 1.0)
        L = L + 100.0 * p.local_contrast * 1.6 * detail

    a, b = lab[..., 1], lab[..., 2]
    if spatial and p.denoise > 0 and min(h, w) >= 64:
        r = max(1, int(round(min(h, w) * 0.004 * (0.5 + p.denoise))))
        gL = (L / 100.0).astype(np.float32)
        ab = np.dstack([a, b]).astype(np.float32)
        ab_f = guided_filter(gL, ab, r, 1e-3 * (0.2 + p.denoise))
        a = a + p.denoise * (ab_f[..., 0] - a)
        b = b + p.denoise * (ab_f[..., 1] - b)
        # luminance: edge-preserving smoothing, much stronger in open water
        # (low t) where dehazing amplified the sensor noise
        Ls = guided_filter(gL, gL, max(1, int(round(min(h, w) * 0.003))), 2e-4 + 1e-3 * p.denoise) * 100.0
        k = 0.5 * p.denoise
        if t is not None:
            k = k + (1.0 - k) * p.denoise * np.clip((0.55 - t) * 2.5, 0.0, 1.0)
        L = L + k * (Ls - L)

    C = np.sqrt(a * a + b * b)
    if p.vibrance != 0 or p.saturation != 0:
        vib = 1.0 + p.vibrance * np.clip(1.0 - C / 60.0, 0, 1)
        scale = vib * (1.0 + p.saturation)
        a, b = a * scale, b * scale

    if spatial and p.sharpen > 0 and min(h, w) >= 64:
        sigma = max(0.6, min(h, w) / 2400.0)
        blur = cv2.GaussianBlur(L, (0, 0), sigma)
        detail = L - blur
        # threshold small detail so we don't sharpen noise
        detail = np.sign(detail) * np.maximum(np.abs(detail) - 0.4, 0)
        L = L + p.sharpen * 1.2 * detail

    lab = np.dstack([np.clip(L, 0, 100), a, b]).astype(np.float32)
    out = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    out = np.clip(out, 0.0, 1.0)

    # 10. Global intensity ----------------------------------------------------
    if p.intensity < 1.0:
        orig = linear_to_srgb(orig_lin)
        out = orig + p.intensity * (out - orig)

    if want_debug and t is not None:
        stats["_t"] = t
    return out.astype(np.float32), stats


def _range_map(disp: np.ndarray, p: Params) -> np.ndarray:
    from depth import to_range
    return to_range(disp, max(0.1, p.z_near), max(p.z_near + 0.5, p.z_far))


def _compensate(lin: np.ndarray, p: Params, stats: dict) -> np.ndarray:
    mr, mg, mb = stats["means"]
    G = lin[..., 1]
    if p.red_comp > 0:
        R = lin[..., 0]
        lin[..., 0] = R + p.red_comp * max(mg - mr, 0) * (1.0 - R) * G
    if p.blue_comp > 0:
        B = lin[..., 2]
        lin[..., 2] = B + p.blue_comp * max(mg - mb, 0) * (1.0 - B) * G
    return lin


def _hue_to_linear(hue: float, sat: float = 0.8) -> np.ndarray:
    hsv = np.array([[[hue / 2.0, sat * 255, 255]]], dtype=np.uint8)  # OpenCV hue is 0..180
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(3)
    return np.maximum(u8_to_linear(rgb), 1e-3)


def _wb_gains(lin: np.ndarray, p: Params, stats: dict, t: np.ndarray | None) -> np.ndarray:
    if p.wb_mode == "none":
        return np.ones(3, np.float32)
    h, w = lin.shape[:2]
    if p.wb_mode == "pick":
        cy, cx = int(p.wb_y * (h - 1)), int(p.wb_x * (w - 1))
        r = max(2, int(min(h, w) * 0.008))
        patch = lin[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1].reshape(-1, 3)
        ref = np.median(patch, axis=0)
        stats["wb_ref"] = ref.tolist()
    else:
        # foreground-weighted shades of gray (Minkowski p=6), ignoring clipped pixels
        x = lin.reshape(-1, 3)
        wgt = np.ones(x.shape[0], np.float32) if t is None else (t.reshape(-1) ** 2)
        valid = (x.max(axis=1) < 0.98) & (x.max(axis=1) > 0.005)
        x, wgt = x[valid], wgt[valid]
        pw = 6.0
        ref = (np.sum(wgt[:, None] * x ** pw, axis=0) / (wgt.sum() + 1e-6)) ** (1 / pw)
    ref = np.maximum(ref, 1e-4)
    gains = ref.max() / ref
    gains = np.minimum(gains, 8.0)
    # keep luminance roughly where it was
    gains = gains / float(LUMA @ (gains * ref) / (LUMA @ ref))
    return gains.astype(np.float32)


# ----------------------------------------------------------------------------
# Public helpers
# ----------------------------------------------------------------------------
PREVIEW_SIZES = {"fast": 760, "preview": 1800}


def use_gpu() -> bool:
    """Render on the GPU (PyTorch) when one is available; DIVE_GPU=0 forces numpy."""
    import os
    if os.environ.get("DIVE_GPU", "1") == "0":
        return False
    try:
        import engine_torch
        return engine_torch.available()
    except Exception:
        return False


class Session:
    """Holds an uploaded image at several resolutions, each with its own cache
    of parameter-independent intermediates."""

    def __init__(self, img_u8: np.ndarray):
        self.images = {"full": img_u8, "analysis": resize_long(img_u8, ANALYSIS_SIZE),
                       "stats": resize_long(img_u8, STATS_SIZE)}
        for name, size in PREVIEW_SIZES.items():
            self.images[name] = resize_long(img_u8, size)
        self.caches: dict[str, dict] = {k: {} for k in self.images}
        # veiling light + raw transmission never depend on the sliders
        self.base_stats: dict = {}
        process(self.images["analysis"], Params(), self.base_stats,
                cache=self.caches["analysis"], stats_only=True)
        self.base_stats = {k: self.base_stats[k] for k in ("A", "A_bbox")}

    @property
    def full(self):
        return self.images["full"]

    @property
    def preview(self):
        return self.images["preview"]

    @property
    def analysis(self):
        return self.images["analysis"]

    # sliders that change WB/levels statistics; the rest act after them
    _STATS_KEYS = ("dehaze", "water_protect", "water_hue", "red_comp", "blue_comp",
                   "wb_mode", "wb_x", "wb_y", "temperature", "tint", "method", "z_near", "z_far")

    def ensure_depth(self) -> np.ndarray:
        """Run the depth model once per image and hand the result to every size."""
        if getattr(self, "disp", None) is None:
            from depth import predict_disparity
            a = self.images["analysis"]
            self.disp = predict_disparity(a if a.dtype == np.uint8 else to_u8(a, dither=False))
            for c in self.caches.values():
                c["disp_src"] = self.disp
                c.pop("disp", None)
        return self.disp

    def stats_for(self, p: Params) -> dict:
        key = tuple(getattr(p, k) for k in self._STATS_KEYS)
        if p.method == "depth":
            self.ensure_depth()
        if getattr(self, "_stats_key", None) != key:
            stats = dict(self.base_stats)
            process(self.images["stats"], p, stats, cache=self.caches["stats"], stats_only=True)
            self._stats_key, self._stats = key, stats
        return dict(self._stats)

    def warm(self, which: str):
        """Precompute the cached intermediates for one size."""
        p = Params()
        if use_gpu():
            import engine_torch as G
            st = self.stats_for(p)
            G.prepare(self.images[which], p, st, self.caches[which])
            with G.GPU_LOCK:
                G.process(self.images[which], p, st, cache=self.caches[which])
            return
        process(self.images[which], p, self.stats_for(p), cache=self.caches[which], stats_only=True)

    def render(self, p: Params, which: str = "preview", debug: bool = False, skin_mask_view: bool = False):
        out, stats = self._render(p, which, debug)
        if getattr(self, "skin_divers", None):
            import skin
            out, stats["skin"] = skin.apply(out, self.skin_divers, p.skin_strength * p.intensity,
                                            p.skin_feather, p.skin_body, mask_view=skin_mask_view)
        return out, stats

    def add_skin(self, x: float, y: float, p: Params) -> dict | None:
        """Outline the diver under the click (normalised coords) on the current look."""
        import skin
        out, _ = self._render(p, "preview")
        img = to_u8(out, dither=False)
        if getattr(self, "_skin_dets", None) is None:
            self._skin_dets = skin.detect(img)   # once per photo: boxes don't depend on the sliders
        d = skin.outline_diver(img, x, y, self._skin_dets)
        if d is not None:
            self.skin_divers = getattr(self, "skin_divers", []) + [d]
        return d

    def clear_skin(self):
        self.skin_divers = []

    def _render(self, p: Params, which: str = "preview", debug: bool = False):
        stats = self.stats_for(p)
        if use_gpu():
            import engine_torch as G
            if p.method == "depth":
                self.ensure_depth()
            cache = self.caches[which] if which != "full" else {"disp_src": getattr(self, "disp", None)}
            if which == "fast":
                p = Params(**{**p.to_dict(), "denoise": 0.0, "sharpen": 0.0})
            G.prepare(self.images[which], p, stats, cache)
            with G.GPU_LOCK:
                return G.process(self.images[which], p, stats, cache=cache, want_debug=debug)
        # the full-res cache is huge (several GB of float32 for 13 MP): don't keep it
        if p.method == "depth":
            self.ensure_depth()
        cache = self.caches[which] if which != "full" else {"disp_src": getattr(self, "disp", None)}
        if which == "fast":
            # draft while dragging: detail filters are invisible at this size
            p = Params(**{**p.to_dict(), "denoise": 0.0, "sharpen": 0.0})
        out, stats = process(self.images[which], p, stats, cache=cache, want_debug=debug)
        return out, stats


def to_u8(srgb: np.ndarray, dither: bool = True) -> np.ndarray:
    x = srgb * 255.0
    if dither:
        # triangular dither breaks up banding in smooth water gradients
        rng = np.random.default_rng(0)
        x = x + (rng.random(x.shape, dtype=np.float32) - rng.random(x.shape, dtype=np.float32))
    return np.clip(x + 0.5, 0, 255).astype(np.uint8)


def to_u16(srgb: np.ndarray) -> np.ndarray:
    return np.clip(srgb * 65535.0 + 0.5, 0, 65535).astype(np.uint16)


def build_cube_lut(session: Session, p: Params, size: int = 33) -> str:
    """Bake the *global* part of the correction into a .cube 3D LUT
    (for video in Resolve/Premiere/FCP). Spatial steps are disabled."""
    q = Params(**{**p.to_dict(), "dehaze": 0.0, "water_protect": 0.0,
                  "local_contrast": 0.0, "denoise": 0.0, "sharpen": 0.0})
    stats = dict(session.base_stats)
    process(session.images["stats"], q, stats, cache=session.caches["stats"], stats_only=True)
    grid = np.linspace(0, 1, size, dtype=np.float32)
    b, g, r = np.meshgrid(grid, grid, grid, indexing="ij")  # R varies fastest
    lattice = np.stack([r, g, b], axis=-1).reshape(1, -1, 3)
    out, _ = process(lattice, q, stats, spatial=False)
    out = out.reshape(-1, 3)
    lines = ["TITLE \"Underwater correction\"", f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0 0 0", "DOMAIN_MAX 1 1 1", ""]
    lines += [f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in out]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# Guided upsampling of the correction (used for video)
# ----------------------------------------------------------------------------
def fit_local_affine(src: np.ndarray, dst: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Per-pixel 3x4 colour transform such that dst ~= M @ [src, 1], fitted in
    a (2r+1)^2 window (colour guided filter, He et al.). Both images are
    display-encoded float32 HxWx3 in [0,1]. Returns HxWx3x4, already box-averaged
    so it can be upsampled and applied directly."""
    h, w = src.shape[:2]
    ksize = (2 * r + 1, 2 * r + 1)

    def box(a):
        return cv2.boxFilter(a, -1, ksize, borderType=cv2.BORDER_REFLECT)

    mI = box(src)                                           # HxWx3
    mP = box(dst)
    # covariance of the guide (3x3) and cross-covariance guide/target (3x3)
    cov = np.empty((h, w, 3, 3), np.float32)
    xc = np.empty((h, w, 3, 3), np.float32)                  # [..., out_c, in_k]
    for i in range(3):
        for j in range(i, 3):
            v = box(src[..., i] * src[..., j]) - mI[..., i] * mI[..., j]
            cov[..., i, j] = v
            cov[..., j, i] = v
        for c in range(3):
            xc[..., c, i] = box(dst[..., c] * src[..., i]) - mP[..., c] * mI[..., i]
    cov += np.eye(3, dtype=np.float32) * eps
    a = xc @ np.linalg.inv(cov)                              # HxWx3x3
    b = mP - np.einsum("hwck,hwk->hwc", a, mI)
    coef = np.concatenate([a, b[..., None]], axis=-1)        # HxWx3x4
    flat = coef.reshape(h, w, 12)
    return np.dstack([box(np.ascontiguousarray(flat[..., i])) for i in range(12)]).reshape(h, w, 3, 4)


def apply_local_affine(img: np.ndarray, coef_small: np.ndarray) -> np.ndarray:
    """Upsample the 3x4 coefficients to img's size and apply them."""
    h, w = img.shape[:2]
    hs, ws = coef_small.shape[:2]
    flat = coef_small.reshape(hs, ws, 12)
    up = [cv2.resize(np.ascontiguousarray(flat[..., i]), (w, h), interpolation=cv2.INTER_LINEAR) for i in range(12)]
    out = np.empty_like(img)
    for c in range(3):
        o = c * 4
        out[..., c] = up[o] * img[..., 0] + up[o + 1] * img[..., 1] + up[o + 2] * img[..., 2] + up[o + 3]
    return np.clip(out, 0.0, 1.0, out=out)
