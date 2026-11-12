"""
3D LUT (.cube) loading and application.

Used as an *input transform* for log footage (e.g. GoPro GP-Log): the
manufacturer's log->Rec.709 LUT is applied to every frame before the
underwater correction, so the pipeline sees normal display-referred colour
while the extra latitude of the log recording is kept until that point.
"""
from __future__ import annotations

import numpy as np


class Lut3D:
    def __init__(self, table: np.ndarray, dmin, dmax, name: str = ""):
        self.table = table.astype(np.float32)       # [b, g, r, 3], r varies fastest in the file
        self.n = table.shape[0]
        self.dmin = np.asarray(dmin, np.float32)
        self.dmax = np.asarray(dmax, np.float32)
        self.name = name
        self._t = None

    # -- numpy (CPU) --------------------------------------------------------
    def apply(self, img: np.ndarray) -> np.ndarray:
        x = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)
        x = np.clip((x - self.dmin) / (self.dmax - self.dmin), 0, 1) * (self.n - 1)
        i0 = np.minimum(x.astype(np.int32), self.n - 2)
        f = x - i0
        r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
        fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
        T = self.table
        c000, c100 = T[b0, g0, r0], T[b0, g0, r0 + 1]
        c010, c110 = T[b0, g0 + 1, r0], T[b0, g0 + 1, r0 + 1]
        c001, c101 = T[b0 + 1, g0, r0], T[b0 + 1, g0, r0 + 1]
        c011, c111 = T[b0 + 1, g0 + 1, r0], T[b0 + 1, g0 + 1, r0 + 1]
        c00 = c000 + (c100 - c000) * fr
        c10 = c010 + (c110 - c010) * fr
        c01 = c001 + (c101 - c001) * fr
        c11 = c011 + (c111 - c011) * fr
        c0 = c00 + (c10 - c00) * fg
        c1 = c01 + (c11 - c01) * fg
        return np.clip(c0 + (c1 - c0) * fb, 0, 1).astype(np.float32)

    # -- torch (GPU) ---------------------------------------------------------
    def apply_t(self, x):
        """x: 1x3xHxW float tensor in [0,1]. Trilinear via 3D grid_sample
        (~10x faster on MPS than explicit gathers, identical result)."""
        import torch
        import torch.nn.functional as F
        if self._t is None or self._t.device != x.device:
            self._t = torch.from_numpy(self.table).permute(3, 0, 1, 2)[None].contiguous().to(x.device)  # 1x3xBxGxR
        dmin = torch.as_tensor(self.dmin, device=x.device).view(1, 3, 1, 1)
        dmax = torch.as_tensor(self.dmax, device=x.device).view(1, 3, 1, 1)
        v = ((x - dmin) / (dmax - dmin)).clamp(0, 1) * 2 - 1
        grid = v.permute(0, 2, 3, 1)[:, None]          # 1x1xHxWx3, (x, y, z) = (r, g, b)
        out = F.grid_sample(self._t, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return out[:, :, 0].clamp(0, 1)


def parse_cube(text: str, name: str = "") -> Lut3D:
    size, dmin, dmax, rows = None, [0, 0, 0], [1, 1, 1], []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        key = s.split()[0].upper()
        if key == "LUT_3D_SIZE":
            size = int(s.split()[1])
        elif key == "LUT_1D_SIZE":
            raise ValueError("LUT 1D não suportado; use um LUT 3D (.cube)")
        elif key == "DOMAIN_MIN":
            dmin = [float(v) for v in s.split()[1:4]]
        elif key == "DOMAIN_MAX":
            dmax = [float(v) for v in s.split()[1:4]]
        elif key in ("TITLE", "LUT_3D_INPUT_RANGE"):
            continue
        else:
            try:
                rows.append([float(v) for v in s.split()[:3]])
            except ValueError:
                continue
    if not size or len(rows) != size ** 3:
        raise ValueError(f"LUT inválido: esperava {size}³ linhas, encontrei {len(rows)}")
    table = np.array(rows, np.float32).reshape(size, size, size, 3)
    return Lut3D(table, dmin, dmax, name)
