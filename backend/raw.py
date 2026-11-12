"""
RAW photo input (LibRaw via rawpy).

The sensor data is demosaiced to 16-bit *linear* RGB with a fixed daylight
white balance, no auto-brightness and no camera tone curve: the underwater
pipeline does all of that itself, and it gets 12-14 bits per channel instead of
the 8 of a JPEG (the red channel underwater often uses < 70 of 256 JPEG levels).

GoPro .GPR: the rawpy wheels are built without GoPro's VC-5 codec. If LibRaw
can't open a .GPR, GoPro's open-source converter is tried
(https://github.com/gopro/gpr, binary `gpr_tools`), which turns it into a DNG.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

RAW_EXT = {".dng", ".gpr", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2", ".raf", ".pef", ".srw"}


def is_raw(name: str) -> bool:
    return Path(name).suffix.lower() in RAW_EXT


def _decode(path: Path) -> np.ndarray:
    import rawpy
    with rawpy.imread(str(path)) as r:
        rgb = r.postprocess(
            use_camera_wb=False,
            use_auto_wb=False,
            user_wb=list(r.daylight_whitebalance) if r.daylight_whitebalance and r.daylight_whitebalance[0] > 0 else None,
            output_color=rawpy.ColorSpace.sRGB,
            gamma=(1, 1),             # linear
            no_auto_bright=True,
            output_bps=16,
            highlight_mode=rawpy.HighlightMode.Blend,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
        )
    lin = rgb.astype(np.float32) / 65535.0
    # expose so the brightest non-specular content sits near white; the pipeline's
    # levels refine it, this just keeps the preview sane
    p = float(np.percentile(lin.max(axis=2), 99.5))
    if p > 1e-4:
        lin = np.clip(lin * (0.92 / p), 0, 1)
    return lin


def load_raw(data: bytes, name: str) -> np.ndarray:
    """RAW bytes -> float32 HxWx3 sRGB-*encoded* in [0,1] (the engine's float input)."""
    from engine import linear_to_srgb
    suffix = Path(name).suffix.lower()
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / f"in{suffix}"
        src.write_bytes(data)
        try:
            lin = _decode(src)
        except Exception as first:
            if suffix != ".gpr":
                raise
            local = Path(__file__).resolve().parent.parent / "tools" / "bin" / "gpr_tools"
            tool = str(local) if local.exists() else shutil.which("gpr_tools")
            if not tool:
                raise RuntimeError(
                    "This .GPR file needs the GoPro converter (gpr_tools), which is not installed. "
                    "Alternative: convert it to DNG with Adobe DNG Converter.") from first
            dng = Path(td) / "out.dng"
            subprocess.run([tool, "-i", str(src), "-o", str(dng)], check=True, capture_output=True)
            lin = _decode(dng)
    return linear_to_srgb(lin).astype(np.float32)
