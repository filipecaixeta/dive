"""
Diver skin-tone correction.

Click on a diver -> the open-vocabulary detector (Grounding DINO) finds the
diver and their exposed parts (face, arms, legs, hands) -> SAM 2 outlines the
diver and the skin -> at render time the skin colour is measured and the
diver gets a correction that puts the skin's *hue* on the skin-tone line
(vectorscope ~123 deg, the same angle for every ethnicity) while keeping its
lightness and saturation, so darker skin is never lightened.

The correction is applied to the whole diver (same distance from the camera ->
same colour loss), full strength on skin, half on the rest, almost none on
near-black gear (a strong red gain turns black neoprene maroon).

Detection alone is not trusted (it called a stingray a "scuba diver" in tests):
the user's click decides which diver, the detector only helps outline it.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np

DETECTOR = "IDEA-Research/grounding-dino-base"
SEGMENTER = "facebook/sam2.1-hiera-large"
SKIN_LINE_DEG = 123.0
PROMPT = "a scuba diver. a human arm. a human leg. a human hand. a human face."

_lock = threading.Lock()
_models: dict = {}


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load():
    if not _models:
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                                  Sam2Model, Sam2Processor)
        _models["dp"] = AutoProcessor.from_pretrained(DETECTOR)
        _models["dm"] = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR).eval()   # CPU: reliable
        _models["sp"] = Sam2Processor.from_pretrained(SEGMENTER)
        _models["sm"] = Sam2Model.from_pretrained(SEGMENTER).to(_device()).eval()
    return _models


def detect(img_u8: np.ndarray) -> list:
    """[(label, score, [x0,y0,x1,y1])] on this image (pixel coords)."""
    import torch
    from PIL import Image
    m = _load()
    pil = Image.fromarray(img_u8)
    inp = m["dp"](images=pil, text=PROMPT, return_tensors="pt")
    with torch.no_grad():
        out = m["dm"](**inp)
    r = m["dp"].post_process_grounded_object_detection(
        out, inp.input_ids, threshold=0.3, text_threshold=0.25, target_sizes=[pil.size[::-1]])[0]
    labels = r.get("text_labels", r["labels"])
    return [(str(l), float(s), [float(v) for v in b]) for l, s, b in zip(labels, r["scores"], r["boxes"])]


def _sam(crop: np.ndarray, boxes=None, point=None) -> list[np.ndarray]:
    import torch
    from PIL import Image
    m = _load()
    kw = {}
    if boxes is not None:
        kw["input_boxes"] = [boxes]
    else:
        kw["input_points"] = [[[point]]]
        kw["input_labels"] = [[[1]]]
    i = m["sp"](images=Image.fromarray(crop), return_tensors="pt", **kw).to(_device())
    with torch.no_grad():
        o = m["sm"](**i, multimask_output=False)
    masks = m["sp"].post_process_masks(o.pred_masks.cpu(), i["original_sizes"])[0]
    return [mk[0].numpy() > 0 for mk in masks]


def skin_pixels(rgb: np.ndarray) -> np.ndarray:
    """Plausible skin on a corrected image: not black gear, not specular, not water-blue."""
    y = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return (y > 0.18) & (y < 0.92) & (rgb[..., 0] > 0.6 * rgb[..., 2])


def outline_diver(img_u8: np.ndarray, x: float, y: float, dets: list) -> dict | None:
    """Masks for the diver under the click (normalised x, y), at img size."""
    with _lock:
        H, W = img_u8.shape[:2]
        px, py = x * W, y * H
        divers = [d for d in dets if "diver" in d[0]]
        hit = [d for d in divers if d[2][0] <= px <= d[2][2] and d[2][1] <= py <= d[2][3]]
        if hit:
            box = min(hit, key=lambda d: (d[2][2] - d[2][0]) * (d[2][3] - d[2][1]))[2]
        else:
            # no detection there: let SAM 2 outline whatever was clicked, from the point
            r = 0.12 * max(W, H)
            box = [max(0, px - r), max(0, py - r), min(W, px + r), min(H, py + r)]
        bw, bh = box[2] - box[0], box[3] - box[1]
        cx0, cy0 = int(max(0, box[0] - 0.15 * bw)), int(max(0, box[1] - 0.15 * bh))
        cx1, cy1 = int(min(W, box[2] + 0.15 * bw)), int(min(H, box[3] + 0.15 * bh))
        crop = img_u8[cy0:cy1, cx0:cx1]
        rel = lambda b: [b[0] - cx0, b[1] - cy0, b[2] - cx0, b[3] - cy0]

        if hit:
            inside = [d for d in dets if "diver" not in d[0]
                      and box[0] - 20 <= (d[2][0] + d[2][2]) / 2 <= box[2] + 20
                      and box[1] - 20 <= (d[2][1] + d[2][3]) / 2 <= box[3] + 20]
            masks = _sam(crop, boxes=[rel(box)] + [rel(d[2]) for d in inside])
            diver, parts = masks[0], masks[1:]
            skin = np.zeros_like(diver)
            for mk in parts:
                skin |= mk
            skin &= diver
        else:
            diver = _sam(crop, point=[px - cx0, py - cy0])[0]
            skin = diver.copy()
        skin &= skin_pixels(crop.astype(np.float32) / 255.0)
        if diver.sum() < 50:
            return None
        full_d = np.zeros((H, W), bool)
        full_s = np.zeros((H, W), bool)
        full_d[cy0:cy1, cx0:cx1] = diver
        full_s[cy0:cy1, cx0:cx1] = skin
        ys, xs = np.nonzero(full_d)
        return {
            "diver": full_d, "skin": full_s,
            "bbox": [xs.min() / W, ys.min() / H, (xs.max() + 1) / W, (ys.max() + 1) / H],
            "click": [x, y], "detected": bool(hit),
        }


# ----------------------------------------------------------------------------
# correction at render time
# ----------------------------------------------------------------------------
def _scope(rgb):
    y = rgb @ np.array([0.2126, 0.7152, 0.0722])
    cb = (rgb[..., 2] - y) / 1.8556
    cr = (rgb[..., 0] - y) / 1.5748
    return (np.degrees(np.arctan2(cr, cb)) + 360) % 360, np.hypot(cb, cr), y


def skin_gains(skin_rgb: np.ndarray) -> np.ndarray:
    """Linear-light gains that move this (display-encoded) skin colour onto the
    skin-tone line at the same luminance; saturation kept, gently floored."""
    from engine import LUMA, srgb_to_linear
    _, mag, y = _scope(skin_rgb)
    mag_t = float(np.clip(max(mag, 0.05), 0, mag * 2.5 + 1e-3))
    cb = mag_t * np.cos(np.radians(SKIN_LINE_DEG))
    cr = mag_t * np.sin(np.radians(SKIN_LINE_DEG))
    R = y + 1.5748 * cr
    B = y + 1.8556 * cb
    G = (y - 0.2126 * R - 0.0722 * B) / 0.7152
    target = np.clip(np.array([R, G, B]), 0, 1)
    s_lin = np.maximum(srgb_to_linear(skin_rgb), 1e-4)
    g = srgb_to_linear(target) / s_lin
    g /= float(LUMA @ (g * s_lin) / (LUMA @ s_lin))
    return np.clip(g, 0.3, 4.0).astype(np.float32)


def apply(out: np.ndarray, divers: list, strength: float, feather: float = 0.3, body: float = 0.5,
          mask_view: bool = False) -> tuple[np.ndarray, list]:
    """Correct each stored diver on a rendered image (float sRGB, any size).
    mask_view: instead of the result, show the mask strength in pink over the image."""
    from engine import linear_to_srgb, srgb_to_linear
    if strength <= 0 or not divers:
        return out, []
    H, W = out.shape[:2]
    out = out.copy()
    report = []
    for d in divers:
        x0, y0, x1, y1 = d["bbox"]
        pad = 0.04
        cx0, cy0 = int(max(0, (x0 - pad) * W)), int(max(0, (y0 - pad) * H))
        cx1, cy1 = int(min(W, (x1 + pad) * W)), int(min(H, (y1 + pad) * H))
        if cx1 - cx0 < 4 or cy1 - cy0 < 4:
            continue
        mh, mw = d["diver"].shape
        sx0, sy0 = int(cx0 * mw / W), int(cy0 * mh / H)
        sx1, sy1 = max(sx0 + 1, int(np.ceil(cx1 * mw / W))), max(sy0 + 1, int(np.ceil(cy1 * mh / H)))
        size = (cx1 - cx0, cy1 - cy0)
        dm = cv2.resize(d["diver"][sy0:sy1, sx0:sx1].astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
        sm = cv2.resize(d["skin"][sy0:sy1, sx0:sx1].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        c = out[cy0:cy1, cx0:cx1]
        sk = measurable_skin(c, sm)
        if sk.sum() < 30:
            continue
        before = np.median(c[sk], axis=0)
        gains = skin_gains(before)
        if mask_view:
            w = mask_weights(c, dm, sk.astype(np.float32), 1.0, min(W, H), feather, body)[..., None]
            out[cy0:cy1, cx0:cx1] = c * (1 - 0.75 * w) + np.array([1.0, 0.25, 0.75], np.float32) * 0.75 * w
            continue
        fixed = correct_region(c, dm, sk.astype(np.float32), gains, strength, min(W, H), feather, body)
        out[cy0:cy1, cx0:cx1] = fixed
        after = np.median(fixed[sk], axis=0)
        report.append({"before": _hex(before), "after": _hex(after),
                       "angle_before": round(float(_scope(before)[0])), "angle_after": round(float(_scope(after)[0]))})
    return out, report


def measurable_skin(rgb: np.ndarray, skin_mask: np.ndarray) -> np.ndarray:
    """SAM skin pixels minus black/specular ones, on this render."""
    yy = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return (skin_mask > 0.5) & (yy > 0.12) & (yy < 0.95)


def mask_weights(c: np.ndarray, diver_f: np.ndarray, skin_f: np.ndarray, strength: float, ref_size: int,
                 feather: float = 0.3, body: float = 0.5) -> np.ndarray:
    """Per-pixel strength of the correction: the diver mask with a soft edge.
    feather 0..1 = edge softness; body 0..1 = 0 only the skin, 1 the whole diver evenly.
    Near-black gear is always protected (a strong red gain turns black neoprene maroon)."""
    sig = max(1.0, (0.002 + 0.016 * feather) * ref_size)
    m = cv2.GaussianBlur(diver_f.astype(np.float32), (0, 0), sig)
    sf = cv2.GaussianBlur(skin_f.astype(np.float32), (0, 0), sig * 1.5)
    sf = np.clip(sf / max(float(sf.max()), 1e-6) * 2.5, 0, 1)
    y = c @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    dark = np.clip((y - 0.06) / 0.22, 0, 1)
    return strength * m * (sf + (1 - sf) * body) * (0.25 + 0.75 * dark)


def correct_region(c: np.ndarray, diver_f: np.ndarray, skin_f: np.ndarray, gains: np.ndarray,
                   strength: float, ref_size: int, feather: float = 0.3, body: float = 0.5) -> np.ndarray:
    """Apply the skin gains inside the feathered diver mask."""
    from engine import linear_to_srgb, srgb_to_linear
    w = mask_weights(c, diver_f, skin_f, strength, ref_size, feather, body)
    return linear_to_srgb(srgb_to_linear(c) * (1 + (gains - 1) * w[..., None]))


def _hex(rgb) -> str:
    r, g, b = (np.clip(rgb, 0, 1) * 255).round().astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"
