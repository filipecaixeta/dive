"""
Regression checks between the numpy reference engine and the GPU engine.

    .venv/bin/python backend/tests/test_engines.py imagem.jpg [imagem ...]

Fails if the GPU render drifts from the reference (PSNR below threshold).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import engine as E          # noqa: E402
import engine_torch as G    # noqa: E402

MIN_PSNR = 45.0
ROOT = Path(__file__).resolve().parent.parent.parent


def psnr(a, b):
    return 10 * np.log10(1 / max(float(np.mean((a - b) ** 2)), 1e-12))


def main(paths):
    ok = True
    for path in paths:
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        s = E.Session(img)
        for method in ("classic", "depth"):
            for params in ({}, {"local_contrast": 0.8, "denoise": 0.7, "shadows": 0.4, "vibrance": -0.3}):
                p = E.Params(method=method, **params)
                if method == "depth":
                    s.ensure_depth()
                st = s.stats_for(p)
                im = s.images["preview"]
                ref, _ = E.process(im, p, dict(st), cache=dict(s.caches["preview"]))
                out, _ = G.process(im, p, dict(st), cache=dict(s.caches["preview"]))
                v = psnr(ref, out)
                flag = "ok " if v >= MIN_PSNR else "FALHOU"
                ok &= v >= MIN_PSNR
                print(f"{flag} {Path(path).name:16s} {method:8s} {str(params)[:40]:40s} {v:5.1f} dB")
    return ok


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted((ROOT / "media").glob("*.jpg"))
    if not paths:
        sys.exit("Passe uma ou mais fotos: test_engines.py foto.jpg [...]")
    sys.exit(0 if main(paths) else 1)
