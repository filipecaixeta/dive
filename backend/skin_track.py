"""
Follow clicked divers through a video (SAM 2 video tracking) for the
skin-tone correction on export.

The divers clicked on one frame are the seed (their diver + skin masks).
SAM 2 propagates both masks forward and backward from that frame, sampling
every STEP_SECONDS, streaming frames so the whole video never sits in memory.
Each direction stops on its own once the diver has been gone for a while.

At every sample the skin colour is measured on the corrected render; the
resulting gains are smoothed over time so the correction does not flicker.
At render time masks and gains are interpolated between samples.
"""
from __future__ import annotations

import os
import subprocess
import threading

import cv2
import numpy as np

TRACK_MODEL = os.environ.get("DIVE_TRACK_MODEL", "facebook/sam2.1-hiera-small")
TRACK_LONG = 1024          # tracking resolution (SAM 2 works at 1024 internally)
STORE_LONG = 512           # stored mask resolution
STEP_SECONDS = 0.2         # sampling interval
GONE_SECONDS = 2.5         # stop a direction after the diver is absent this long

_lock = threading.Lock()
_m: dict = {}


def _load():
    if not _m:
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        _m["dev"] = dev
        _m["proc"] = Sam2VideoProcessor.from_pretrained(TRACK_MODEL)
        _m["model"] = Sam2VideoModel.from_pretrained(TRACK_MODEL).to(dev).eval()
    return _m


class DiverTrack:
    """Per-sample masks (small) + smoothed gains for one diver."""

    def __init__(self, times, diver, skin, gains, present):
        order = np.argsort(times)
        self.times = np.asarray(times, np.float64)[order]
        self.diver = [diver[i] for i in order]
        self.skin = [skin[i] for i in order]
        self.gains = np.asarray(gains, np.float32)[order]
        self.present = np.asarray(present, bool)[order]

    def at(self, t: float, w: int, h: int):
        """(diver float mask, skin float mask, gains) at time t and size w x h, or None."""
        if len(self.times) == 0 or t < self.times[0] - STEP_SECONDS or t > self.times[-1] + STEP_SECONDS:
            return None
        j = int(np.searchsorted(self.times, t))
        i0, i1 = max(0, j - 1), min(len(self.times) - 1, j)
        t0, t1 = self.times[i0], self.times[i1]
        a = 0.0 if t1 == t0 else float(np.clip((t - t0) / (t1 - t0), 0, 1))
        if not (self.present[i0] or self.present[i1]):
            return None
        def mix(lst):
            m0 = cv2.resize(lst[i0].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            if i1 == i0 or a == 0:
                return m0
            m1 = cv2.resize(lst[i1].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            return m0 * (1 - a) + m1 * a
        g = self.gains[i0] * (1 - a) + self.gains[i1] * a
        return mix(self.diver), mix(self.skin), g


def _even(x):
    return max(2, int(round(x / 2)) * 2)


def _decode(path, info, t_from: float, t_to: float | None, w: int, h: int, fps: float, input_lut):
    """Yield (t, rgb uint8) sampled at `fps` between t_from and t_to."""
    from video import FFMPEG, _scale_filter
    cmd = [FFMPEG, "-v", "error", "-hwaccel", "videotoolbox", "-ss", f"{t_from:.3f}", "-i", str(path)]
    if t_to is not None:
        cmd += ["-t", f"{max(0.0, t_to - t_from):.3f}"]
    cmd += ["-map", "0:v:0", "-vf", f"fps={fps}," + _scale_filter(info, w, h) + ",format=rgb24",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = 0
    try:
        while True:
            raw = p.stdout.read(w * h * 3)
            if not raw or len(raw) < w * h * 3:
                break
            f = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
            if input_lut is not None:
                f = (input_lut.apply(f) * 255 + 0.5).astype(np.uint8)
            yield t_from + n / fps, f
            n += 1
    finally:
        p.kill()


def track(path, info: dict, seed_t: float, seeds: list, measure, input_lut=None,
          progress=None, cancel: threading.Event | None = None, on_sample=None) -> list[DiverTrack]:
    """seeds: [{"diver": bool mask, "skin": bool mask}] at any size (seed frame).
    measure(t, frame_u8, diver_mask, skin_mask) -> skin gains (3,) or None."""
    import torch
    m = _load()
    proc, model, dev = m["proc"], m["model"], m["dev"]
    W, H = info["width"], info["height"]
    s = TRACK_LONG / max(W, H)
    w, h = _even(W * s), _even(H * s)
    ss = STORE_LONG / max(w, h)
    sw, sh = max(8, round(w * ss)), max(8, round(h * ss))
    fps = 1.0 / STEP_SECONDS
    n_obj = len(seeds)
    obj_ids = list(range(1, 2 * n_obj + 1))      # (diver, skin) per seed
    seed_masks = []
    for sd in seeds:
        for k in ("diver", "skin"):
            seed_masks.append(cv2.resize(sd[k].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0)

    results = [dict(times=[], diver=[], skin=[], gains=[], present=[]) for _ in range(n_obj)]
    total = max(1.0, info["duration"] / STEP_SECONDS)
    done = [0]

    def run(frames_iter):
        with _lock, torch.inference_mode():
            session = proc.init_video_session(inference_device=dev, video_storage_device=dev, dtype=torch.float32)
            gone = [0.0] * n_obj
            for k, (t, f) in enumerate(frames_iter):
                if cancel is not None and cancel.is_set():
                    return
                inp = proc(images=f, device=dev, return_tensors="pt")
                if k == 0:
                    proc.add_inputs_to_inference_session(session, frame_idx=0, obj_ids=list(obj_ids),  # the library mutates it
                                                         input_masks=[mk.astype(np.float32) for mk in seed_masks],
                                                         original_size=inp.original_sizes[0])
                # explicit frame_idx: we evict old frames below, and the library would
                # otherwise derive the index from how many frames it still holds
                out = model(inference_session=session, frame_idx=k, frame=inp.pixel_values[0])
                masks = proc.post_process_masks([out.pred_masks], original_sizes=[[h, w]], binarize=True)[0]
                # drop the pixels of frames already processed: only the memory features are reused
                for old in list(session.processed_frames.keys()):
                    if old != k:
                        del session.processed_frames[old]
                any_present = False
                for i in range(n_obj):
                    dmask = masks[2 * i, 0].cpu().numpy()
                    smask = masks[2 * i + 1, 0].cpu().numpy() & dmask
                    present = dmask.sum() > 0.0005 * w * h
                    g = measure(t, f, dmask, smask) if present else None
                    r = results[i]
                    r["times"].append(t)
                    r["diver"].append(cv2.resize(dmask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_AREA) > 0)
                    r["skin"].append(cv2.resize(smask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_AREA) > 0)
                    r["gains"].append(g if g is not None else np.full(3, np.nan, np.float32))
                    r["present"].append(bool(present and g is not None))
                    gone[i] = 0.0 if present else gone[i] + STEP_SECONDS
                    any_present |= gone[i] < GONE_SECONDS
                if on_sample is not None:
                    on_sample(t, f, [masks[2 * i, 0].cpu().numpy() for i in range(n_obj)])
                done[0] += 1
                if progress:
                    progress(min(1.0, done[0] / total))
                if k > 0 and not any_present:
                    return

    # forward from the seed...
    seed_frame = {}

    def forward():
        for i, (t, f) in enumerate(_decode(path, info, seed_t, None, w, h, fps, input_lut)):
            if i == 0:
                seed_frame["f"] = f
            yield t, f
    run(forward())
    # ...then backward: decode the stretch before the seed and feed it in reverse,
    # starting again from the seed frame with the same masks
    if seed_t > STEP_SECONDS / 2 and "f" in seed_frame and not (cancel is not None and cancel.is_set()):
        back = list(_decode(path, info, max(0.0, seed_t - 60.0), seed_t, w, h, fps, input_lut))
        run(iter([(seed_t, seed_frame["f"])] + list(reversed(back))))

    tracks = []
    for r in results:
        g = np.array(r["gains"], np.float32)
        tracks.append(DiverTrack(r["times"], r["diver"], r["skin"], _smooth_gains(np.array(r["times"]), g), r["present"]))
    return tracks


def _smooth_gains(times: np.ndarray, g: np.ndarray, sigma_s: float = 0.8) -> np.ndarray:
    """Gaussian smoothing over time, ignoring samples where the skin wasn't measured."""
    if len(g) == 0:
        return g
    order = np.argsort(times)
    t, gg = times[order], g[order]
    valid = np.isfinite(gg).all(axis=1)
    out = np.ones_like(gg)
    if valid.any():
        tv, gv = t[valid], gg[valid]
        for k in range(len(t)):
            wgt = np.exp(-0.5 * ((tv - t[k]) / sigma_s) ** 2)
            out[k] = (wgt[:, None] * gv).sum(0) / max(wgt.sum(), 1e-6)
    res = np.empty_like(out)
    res[order] = out
    return res
