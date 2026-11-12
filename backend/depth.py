"""
Monocular depth (Depth Anything V2) for the depth-aware correction.

The model returns relative inverse depth (disparity) with unknown scale.
`to_range` maps it to an approximate range in metres using two user-tunable
anchors (nearest / farthest visible distance). The Sea-thru fit downstream
absorbs whatever scale error remains, since it estimates the attenuation
coefficients from the image itself.
"""
from __future__ import annotations

import os
import threading

import cv2
import numpy as np

MODEL_SIZE = os.environ.get("DIVE_DEPTH_MODEL", "Base")   # Small | Base | Large
INPUT_LONG = int(os.environ.get("DIVE_DEPTH_INPUT", "756"))  # multiple of 14

_lock = threading.Lock()
_model = None
_proc = None
_device = None


def device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load():
    global _model, _proc, _device
    if _model is None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        mid = f"depth-anything/Depth-Anything-V2-{MODEL_SIZE}-hf"
        _device = device()
        _proc = AutoImageProcessor.from_pretrained(mid)
        _model = AutoModelForDepthEstimation.from_pretrained(mid).to(_device).eval()
        if _device == "cuda":
            _model = _model.half()
    return _model, _proc


def predict_disparity(rgb_u8: np.ndarray) -> np.ndarray:
    """Relative inverse depth in [0,1] (1 = nearest), at a reduced resolution
    with the image's aspect ratio."""
    import torch
    with _lock:
        model, proc = _load()
        h, w = rgb_u8.shape[:2]
        s = INPUT_LONG / max(h, w)
        nh, nw = max(14, round(h * s / 14) * 14), max(14, round(w * s / 14) * 14)
        small = cv2.resize(rgb_u8, (nw, nh), interpolation=cv2.INTER_AREA)
        inp = proc(images=small, return_tensors="pt", do_resize=False).to(_device)
        if _device == "cuda":
            inp = {k: v.half() for k, v in inp.items()}
        with torch.no_grad():
            d = model(**inp).predicted_depth[0].float().cpu().numpy()
    lo, hi = np.percentile(d, [0.5, 99.5])
    return np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def to_range(disp: np.ndarray, z_near: float, z_far: float) -> np.ndarray:
    """Disparity -> approximate range in metres (inverse-linear, as disparity is)."""
    inv_near, inv_far = 1.0 / z_near, 1.0 / z_far
    return (1.0 / (inv_far + disp * (inv_near - inv_far))).astype(np.float32)


def upsample_to(disp: np.ndarray, guide_rgb: np.ndarray) -> np.ndarray:
    """Resize the disparity to the guide's size and snap edges to it."""
    from engine import guided_filter
    h, w = guide_rgb.shape[:2]
    d = cv2.resize(disp, (w, h), interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(guide_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0 if guide_rgb.dtype == np.uint8 \
        else (guide_rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)).astype(np.float32)
    r = max(2, int(min(h, w) * 0.004))
    return np.clip(guided_filter(g, d, r, 1e-4), 0, 1).astype(np.float32)
