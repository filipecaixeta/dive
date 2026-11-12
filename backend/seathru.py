"""
Depth-aware underwater correction, after Sea-thru (Akkaynak & Treibitz, CVPR 2019).

Image formation, per channel c, at range z:

    I_c = J_c * exp(-beta_D_c(z) * z)  +  B_inf_c * (1 - exp(-beta_B_c * z))
          ^ direct signal (attenuated)     ^ backscatter (veiling light)

  1. Backscatter: in each range bin the darkest pixels are ~pure backscatter.
     Fit B_c(z) = B_inf (1 - e^{-bB z}) + J' e^{-bD' z} to them and subtract.
  2. Attenuation: estimate the local illuminant of the direct signal with a
     depth-aware local average (neighbours at similar range only), then fit a
     smooth curve of it against range. Dividing by that curve undoes the
     wavelength- and range-dependent colour loss: each distance gets its own
     gains instead of one white balance for the whole frame.
  3. Re-add a stylised haze (the chosen water blue) in proportion to range so
     the open water reads as water and the scene keeps its depth cue.

Depth comes from a monocular model (see depth.py). Its scale is approximate;
the fitted coefficients absorb the error.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


# ----------------------------------------------------------------------------
# 1. Backscatter
# ----------------------------------------------------------------------------
def _bs_model(q, z):
    b_inf, b_b, j, b_d = q
    return b_inf * (1 - np.exp(-b_b * z)) + j * np.exp(-b_d * z)


def _smooth_fill(lin: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Very smooth colour field from the masked pixels, extended over the frame."""
    h, w = mask.shape
    s = 64 / max(h, w)
    sw, sh = max(8, round(w * s)), max(8, round(h * s))
    m = cv2.resize(mask.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    l = cv2.resize(lin, (sw, sh), interpolation=cv2.INTER_AREA)
    out, cover = None, None
    for sig in (2.0, 6.0, 18.0):   # successively wider, to reach areas far from any water
        num = cv2.GaussianBlur(l * m[..., None], (0, 0), sig)
        den = cv2.GaussianBlur(m, (0, 0), sig)[..., None]
        est = num / np.maximum(den, 1e-6)
        wgt = np.clip(den / 0.15, 0, 1)
        if out is None:
            out, cover = est, wgt
        else:
            out = out * cover + est * (1 - cover)   # keep the tighter estimate where it is reliable
            cover = np.maximum(cover, wgt)
    glob = np.median(lin[mask], axis=0)
    out = out * cover + glob * (1 - cover)
    return cv2.resize(out.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)


def water_likeness(lin: np.ndarray, bmap: np.ndarray, tol: float) -> np.ndarray:
    lr = np.log(np.maximum(lin, 1e-4) / np.maximum(bmap, 1e-4))
    return np.sqrt((lr ** 2).mean(axis=2))


def veiling_map(lin: np.ndarray, disp: np.ndarray, water_disp: float = 0.06) -> np.ndarray | None:
    """Spatially varying veiling light B_inf(x), measured on the open-water
    pixels and smoothly extended over the frame. Captures lens vignetting and
    the vertical light gradient (brighter towards the surface), which a single
    global B_inf (classic Sea-thru) gets badly wrong on wide-angle footage.

    Pass 1 takes the farthest pixels per the depth model; pass 2 adds pixels
    whose colour matches that first estimate, because the model is unreliable
    at the frame edges and in featureless water."""
    water = disp < water_disp
    if water.mean() < 0.03:
        return None
    bmap = _smooth_fill(lin, water)
    for _ in range(2):
        water = (water_likeness(lin, bmap, 0) < 0.2) & (disp < 0.3)
        if water.mean() < 0.03:
            break
        bmap = _smooth_fill(lin, water)
    return bmap


def fit_backscatter(lin: np.ndarray, z: np.ndarray, bmap: np.ndarray | None = None,
                    n_bins: int = 10, frac: float = 0.01) -> np.ndarray:
    """Returns 3x4 parameters [B_inf, beta_B, J', beta_D'] per channel. With a
    veiling map the fit is on I / B_inf(x), so B_inf is ~1 (a scale factor)."""
    ratio = lin / np.maximum(bmap, 1e-4) if bmap is not None else lin
    zf = z.reshape(-1)
    px = ratio.reshape(-1, 3)
    s = px.sum(axis=1)
    lo, hi = np.percentile(zf, [1, 99.8])
    edges = np.linspace(lo, hi + 1e-6, n_bins + 1)
    zs, ps = [], []
    for i in range(n_bins):
        idx = np.flatnonzero((zf >= edges[i]) & (zf < edges[i + 1]))
        if idx.size < 50:
            continue
        k = max(10, int(idx.size * frac))
        sel = idx[np.argpartition(s[idx], k)[:k]]
        zs.append(zf[sel])
        ps.append(px[sel])
    Z = np.concatenate(zs).astype(np.float64)
    P = np.concatenate(ps).astype(np.float64)
    lb = [0.8, 0, 0, 0] if bmap is not None else [0, 0, 0, 0]
    ub = [1.1, 5, 1, 5] if bmap is not None else [1, 5, 1, 5]
    out = np.zeros((3, 4), np.float32)
    for c in range(3):
        y = P[:, c]
        best = None
        b0 = 1.0 if bmap is not None else y.max()
        for init in ((b0, 0.3, 0.02, 1.0), (b0, 1.0, 0.0, 0.3), (b0, 0.1, 0.05, 2.0)):
            init = tuple(np.clip(init, lb, ub))
            try:
                r = least_squares(lambda q: _bs_model(q, Z) - y, init,
                                  bounds=(lb, ub), loss="soft_l1", f_scale=0.01)
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
        out[c] = best.x if best is not None else (b0, 0.5, 0, 1)
    return out


def backscatter(z: np.ndarray, prm: np.ndarray) -> np.ndarray:
    return np.dstack([_bs_model(prm[c], z) for c in range(3)]).astype(np.float32)


def transmission(z: np.ndarray, prm: np.ndarray) -> np.ndarray:
    """exp(-beta_B z) on green: how much of the scene survives at that range."""
    return np.exp(-prm[1, 1] * z).astype(np.float32)


# ----------------------------------------------------------------------------
# 2. Range-dependent attenuation
# ----------------------------------------------------------------------------
def fit_illuminant(D: np.ndarray, z: np.ndarray, valid: np.ndarray,
                   n_layers: int = 10, n_bins: int = 14) -> dict:
    """Depth-aware local space average colour of the direct signal, reduced to
    a smooth per-channel curve log E_c(z) = poly2(z)."""
    h, w = z.shape
    sig = 0.06 * min(h, w)
    zv = z[valid]
    lo, hi = np.percentile(zv, [1, 99])
    hi = max(hi, lo + 1e-3)
    centers = np.linspace(lo, hi, n_layers)
    sz = (hi - lo) / n_layers + 1e-6
    acc = np.zeros_like(D)
    wsum = np.zeros((h, w), np.float32)
    vf = valid.astype(np.float32)
    for c in centers:
        wk = np.exp(-0.5 * ((z - c) / sz) ** 2).astype(np.float32)
        wv = wk * vf
        num = cv2.GaussianBlur(D * wv[..., None], (0, 0), sig)
        den = cv2.GaussianBlur(wv, (0, 0), sig)[..., None]
        acc += wk[..., None] * (num / (den + 1e-6))
        wsum += wk
    E = acc / (wsum[..., None] + 1e-6)

    # robust curve vs range
    edges = np.linspace(lo, hi + 1e-6, n_bins + 1)
    zz, ll, ww = [], [], []
    for i in range(n_bins):
        m = valid & (z >= edges[i]) & (z < edges[i + 1])
        n = int(m.sum())
        if n < 30:
            continue
        zz.append(0.5 * (edges[i] + edges[i + 1]))
        ll.append(np.median(np.log(np.maximum(E[m], 1e-5)), axis=0))
        ww.append(np.sqrt(n))
    zz, ll, ww = np.array(zz), np.array(ll), np.array(ww)
    deg = 2 if len(zz) >= 5 else 1
    coef = np.stack([np.polyfit(zz, ll[:, c], deg, w=ww) for c in range(3)]) if len(zz) >= 2 else \
        np.stack([[0.0] * deg + [float(np.log(max(np.median(D[..., c][valid]), 1e-4)))] for c in range(3)])
    return {"coef": coef.tolist(), "z_lo": float(lo), "z_hi": float(hi)}


def gain_curves(z: np.ndarray, illum: dict, max_gain: float = 12.0) -> np.ndarray:
    coef = np.array(illum["coef"])
    zc = np.clip(z, illum["z_lo"], illum["z_hi"])   # never extrapolate the fit
    logE = np.dstack([np.polyval(coef[c], zc) for c in range(3)])
    ref = np.array([np.polyval(coef[c], illum["z_lo"]) for c in range(3)])
    # Each channel relative to *itself* at the nearest range: this only undoes
    # the extra colour loss with distance. The absolute white point is left to
    # the robust WB (or the eyedropper); neutralising the local average at every
    # range (gray world) turned sand magenta next to green algae.
    g = np.exp(np.clip(ref - logE, 0.0, np.log(max_gain)))
    return g.astype(np.float32)


def _haze_range(lin: np.ndarray, bmap: np.ndarray, beta_g: float) -> np.ndarray:
    from engine import guided_filter, linear_to_srgb, luminance
    h, w = lin.shape[:2]
    short = min(h, w)
    k = max(3, int(short * 0.012) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    norm = np.minimum(lin[..., 1] / np.maximum(bmap[..., 1], 1e-4), lin[..., 2] / np.maximum(bmap[..., 2], 1e-4))
    dark = np.minimum(cv2.dilate(cv2.erode(norm.astype(np.float32), se), se), norm)
    t = np.clip(1.0 - 0.95 * dark, 0.02, 1.0).astype(np.float32)
    g = linear_to_srgb(luminance(lin)).astype(np.float32)
    t = np.clip(guided_filter(g, t, max(2, int(short * 0.01)), 1e-3), 0.02, 1.0)
    # anything brighter than the veiling light there (sand, white bellies) is
    # a bright object, not haze: no evidence from it
    ratio = luminance(lin) / np.maximum(luminance(bmap), 1e-4)
    trust = np.clip((1.0 - ratio) / 0.25, 0.0, 1.0)
    trust = cv2.GaussianBlur(trust.astype(np.float32), (0, 0), max(1.0, short * 0.006))
    t = 1.0 - (1.0 - t) * trust
    return (-np.log(np.clip(t, 0.02, 1.0)) / max(beta_g, 1e-3)).astype(np.float32)


# ----------------------------------------------------------------------------
# Front end used by engine.process
# ----------------------------------------------------------------------------
def front(lin: np.ndarray, z: np.ndarray, disp: np.ndarray, p, stats: dict, cache: dict, wb_fn, hue_fn):
    """Replaces dehaze + compensation + WB of the classic path.
    Returns (lin after correction, t map for later stages)."""
    if "bmap" not in cache:
        cache["bmap"] = veiling_map(lin, disp)
    bmap = cache["bmap"]
    if "bs" not in stats:
        stats["bs"] = fit_backscatter(lin, z, bmap).tolist()
        stats["bs_ratio"] = bmap is not None
    prm = np.array(stats["bs"], np.float32)

    if bmap is not None:
        # The depth model underestimates the range of distant, hazy objects
        # (they come out as "mid-range"). The pixel's own haze is evidence of a
        # minimum range: dark-channel transmission against the veiling map,
        # converted to range with the fitted green backscatter coefficient.
        # Only trusted away from the camera: near sand is bright and would fool it.
        if "z_haze" not in cache:
            cache["z_haze"] = _haze_range(lin, bmap, float(prm[1, 1]))
        far_w = np.clip((0.5 - disp) / 0.3, 0, 1)
        z = z + (np.maximum(z, cache["z_haze"]) - z) * far_w

    B = backscatter(z, prm)
    if stats.get("bs_ratio") and bmap is not None:
        B = B * bmap
    t = transmission(z, prm)

    if bmap is not None:
        # Pixels whose colour matches the veiling light at that spot are open
        # water whatever the depth model says (it is unreliable at the frame
        # edges and in featureless water): treat them as fully hazed.
        dist = water_likeness(lin, bmap, 0)
        wl = np.exp(-(dist / 0.12) ** 2).astype(np.float32)
        wl = cv2.GaussianBlur(wl, (0, 0), max(1.0, min(z.shape) * 0.004))
        B = B * (1 - wl[..., None]) + bmap * wl[..., None]
        t = t * (1 - wl)
    else:
        wl = np.zeros(z.shape, np.float32)

    # in water-like pixels the backscatter is removed fully: whatever is left
    # there would only be amplified by the range gains into grey/red mush
    strength = p.dehaze + (1.0 - p.dehaze) * wl[..., None]
    D = np.maximum(lin - strength * B, 0.0)

    if "illum" not in stats:
        valid = (t > 0.15) & (D.sum(axis=2) > 0.01)
        if valid.mean() < 0.02:
            valid = D.sum(axis=2) > 0.005
        stats["illum"] = fit_illuminant(D, z, valid)
    # translucent / mixed pixels (hydroid fuzz against water) must not get the
    # full far-range gain or they glow
    G = 1.0 + (gain_curves(z, stats["illum"], max_gain=6.0) - 1.0) * (1.0 - wl[..., None])
    # red_comp scales how much of the range-dependent colour shift we undo
    # (1 = full Sea-thru); keeps the classic slider meaningful
    k = np.clip(p.red_comp, 0, 2)
    J = D * (G ** k)

    if "gains" not in stats:
        stats["gains"] = wb_fn(J, p, stats, t).tolist()
    gains = np.array(stats["gains"], np.float32)
    gains = gains * np.array([2 ** (0.5 * p.temperature), 2 ** (-0.35 * p.tint), 2 ** (-0.5 * p.temperature)],
                             np.float32)
    J = J * gains

    # stylised haze: the veiling light recoloured to the chosen water hue (keeps
    # its natural gradient), as much as the physics says there was
    if p.water_protect > 0:
        if stats.get("bs_ratio") and bmap is not None:
            veil = bmap * prm[:, 0]
            b_inf = np.median(veil.reshape(-1, 3), axis=0)
        else:
            b_inf = prm[:, 0]
            veil = b_inf
        target = hue_fn(p.water_hue)
        gains_w = target * float(LUMA @ (b_inf * gains)) / float(LUMA @ target) / np.maximum(b_inf, 1e-4)
        water = veil * gains_w
        haze = np.clip((1.0 - t) * (0.5 + p.water_protect), 0, 1)[..., None] * strength
        J = J * (1 - haze) + water * haze
    return np.maximum(J, 0.0).astype(np.float32), t
