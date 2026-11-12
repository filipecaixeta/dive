"""
Video support: probing, frame grabs, thumbnails and the export job.

Export runs in two passes:
  1. Analysis   decode ~2 frames/s at 640 px and compute the correction
                statistics (veiling light, WB gains, levels) for each sample,
                then smooth them over time so the colour never flickers.
  2. Render     decode every frame (16-bit RGB), run the full pipeline at a
                working resolution, fit a local 3x4 colour transform between the
                working-res input and output, and apply it to the full-res frame.
                That keeps all the original detail while costing a fraction of a
                full-res render. Frames go out as 10-bit HEVC with the audio copied.
"""
from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from depth import predict_disparity
from engine import Params, Session, apply_local_affine, fit_local_affine, process, resize_long, use_gpu

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"
WORK_SIZE = 1280        # long side of the working resolution for video frames
ANALYSIS_FPS = 2.0
ANALYSIS_SIZE = 640
SMOOTH_SECONDS = 1.0    # temporal smoothing (gaussian sigma) of the statistics
WORKERS = 6             # frames in parallel on CPU (numpy/OpenCV release the GIL)
WORKERS_GPU = 4         # with a GPU the GPU is the bottleneck; more threads only add contention


# ----------------------------------------------------------------------------
# Probing / decoding helpers
# ----------------------------------------------------------------------------
def probe(path: Path) -> dict:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    v = next(s for s in data["streams"] if s["codec_type"] == "video")
    num, den = (int(x) for x in v.get("r_frame_rate", "30/1").split("/"))
    duration = float(v.get("duration") or data["format"].get("duration") or 0)
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"]) % 360
    if rot in (90, 270):
        w, h = h, w
    return {
        "width": w,
        "height": h,
        "fps_num": num,
        "fps_den": den,
        "fps": num / den if den else 30.0,
        "duration": duration,
        "frames": int(v.get("nb_frames") or round(duration * num / max(den, 1))),
        "pix_fmt": v.get("pix_fmt", ""),
        "bit_rate": int(v.get("bit_rate") or data["format"].get("bit_rate") or 0),
        "color_range": v.get("color_range", "tv"),
        "color_space": v.get("color_space", "bt709"),
        "codec": v.get("codec_name", ""),
        "has_audio": any(s["codec_type"] == "audio" for s in data["streams"]),
    }


def _scale_filter(info: dict, w: int | None = None, h: int | None = None) -> str:
    """YUV->RGB with the *declared* range and matrix (ffmpeg's default is BT.601,
    which shifts hues on HD/4K footage)."""
    rng = "full" if info["color_range"] in ("pc", "jpeg") else "limited"
    mtx = {"bt709": "bt709", "smpte170m": "smpte170m", "bt470bg": "bt470bg",
           "bt2020nc": "bt2020", "bt2020c": "bt2020"}.get(info["color_space"], "bt709")
    resize = w is not None and (w, h) != (info["width"], info["height"])
    size = f"w={w}:h={h}:" if resize else ""
    # NB: "area+accurate_rnd+full_chroma_int" without a resize and with rgb48
    # output trips a swscale bug (garbage colours), so the flags differ per case.
    flags = "area+accurate_rnd" if resize else "accurate_rnd+full_chroma_int"
    return f"scale={size}in_range={rng}:in_color_matrix={mtx}:flags={flags}"


_ENCODERS: set | None = None


def encoders() -> set:
    global _ENCODERS
    if _ENCODERS is None:
        out = subprocess.run([FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        _ENCODERS = {ln.split()[1] for ln in out.splitlines() if ln.startswith(" V")}
    return _ENCODERS


def video_codec_args(codec: str, bitrate: int) -> tuple[list, str]:
    """ffmpeg args for the chosen output, picking the best encoder available
    (hardware on Mac/NVIDIA, software elsewhere). Returns (args, extension)."""
    enc = encoders()
    if codec == "prores":
        if "prores_videotoolbox" in enc:
            v = ["-c:v", "prores_videotoolbox", "-profile:v", "hq"]
        else:
            v = ["-c:v", "prores_ks", "-profile:v", "3", "-vendor", "apl0"]
        return v + ["-pix_fmt", "yuv422p10le"], ".mov"
    if "hevc_videotoolbox" in enc:
        v = ["-c:v", "hevc_videotoolbox", "-profile:v", "main10", "-b:v", str(bitrate), "-pix_fmt", "p010le"]
    elif "hevc_nvenc" in enc:
        v = ["-c:v", "hevc_nvenc", "-profile:v", "main10", "-preset", "p6", "-b:v", str(bitrate), "-pix_fmt", "p010le"]
    else:
        v = ["-c:v", "libx265", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p10le"]
    return v + ["-tag:v", "hvc1"], ".mp4"


def _pix_fmt(args: list) -> str:
    return args[args.index("-pix_fmt") + 1]


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def grab_frame(path: Path, info: dict, t: float, long_side: int | None = None) -> np.ndarray:
    """One frame as uint8 RGB (optionally downscaled)."""
    w, h = info["width"], info["height"]
    if long_side and max(w, h) > long_side:
        s = long_side / max(w, h)
        w, h = _even(w * s), _even(h * s)
    t = max(0.0, min(t, max(info["duration"] - 1.0 / info["fps"], 0)))
    cmd = [FFMPEG, "-v", "error", "-hwaccel", "videotoolbox", "-ss", f"{t:.3f}", "-i", str(path),
           "-map", "0:v:0", "-frames:v", "1", "-vf", _scale_filter(info, w, h) + ",format=rgb24",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    if len(raw) < w * h * 3:
        raise RuntimeError("Não consegui ler o frame nesse ponto do vídeo")
    return np.frombuffer(raw[: w * h * 3], np.uint8).reshape(h, w, 3).copy()


def thumbnail_strip(path: Path, info: dict, n: int = 14, height: int = 64) -> np.ndarray:
    w = _even(info["width"] * height / info["height"])
    times = [(i + 0.5) * info["duration"] / n for i in range(n)]
    with ThreadPoolExecutor(4) as ex:
        frames = list(ex.map(lambda t: grab_frame(path, info, t, max(w, height)), times))
    frames = [cv2.resize(f, (w, height), interpolation=cv2.INTER_AREA) for f in frames]
    return np.hstack(frames)


# ----------------------------------------------------------------------------
# Temporal statistics
# ----------------------------------------------------------------------------
def _leaves(st: dict, keys: list) -> list:
    """Numeric leaves of the stats dict in a fixed key order (lists/dicts walked)."""
    out = []

    def walk(v):
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, (int, float)):
            out.append(float(v))
        elif isinstance(v, dict):
            for k in sorted(v):
                walk(v[k])
        else:
            for x in v:
                walk(x)
    for k in keys:
        walk(st[k])
    return out


def _rebuild(template: dict, keys: list, vals: np.ndarray) -> dict:
    it = iter(vals.tolist())

    def build(v):
        if isinstance(v, bool) or v is None:
            return v
        if isinstance(v, (int, float)):
            return next(it)
        if isinstance(v, dict):
            return {k: build(v[k]) for k in sorted(v)}
        return [build(x) for x in v]
    out = {k: build(template[k]) for k in keys}
    for k, v in template.items():   # non-numeric extras (flags) as in the template
        out.setdefault(k, v)
    return out


_SMOOTH_KEYS = ("A", "means", "gains", "A_comp", "levels", "bs", "illum")


def _smooth(series: np.ndarray, sigma: float) -> np.ndarray:
    """Median (kills single-sample outliers, e.g. the water detector jumping)
    followed by a gaussian, both edge-padded, along axis 0."""
    n = series.shape[0]
    if n < 3:
        return series
    pad = np.pad(series, ((2, 2), (0, 0)), mode="edge")
    med = np.median(np.stack([pad[i:i + n] for i in range(5)]), axis=0)
    r = max(1, int(round(sigma * 3)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / max(sigma, 1e-3)) ** 2)
    k /= k.sum()
    pad = np.pad(med, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(series.shape[1])], axis=1)


class StatsTimeline:
    def __init__(self, times: np.ndarray, values: np.ndarray, template: dict, keys: list):
        self.times, self.values, self.template, self.keys = times, values, template, keys

    def at(self, t: float) -> dict:
        if len(self.times) == 1:
            v = self.values[0]
        else:
            v = np.array([np.interp(t, self.times, self.values[:, j]) for j in range(self.values.shape[1])])
        return _rebuild(self.template, self.keys, v)


# ----------------------------------------------------------------------------
# Export job
# ----------------------------------------------------------------------------
@dataclass
class VideoJob:
    id: str
    src: Path
    out: Path
    info: dict
    params: Params
    scale: float = 1.0
    mode: str = "adaptive"              # adaptive | fixed
    codec: str = "hevc"                 # hevc (10-bit, delivery) | prores (422 HQ, for editing)
    input_lut: object = None            # lut.Lut3D applied to every decoded frame (log footage)
    skin_seeds: list = field(default_factory=list)   # divers clicked on the seed frame (masks)
    skin_seed_t: float = 0.0
    preview_jpeg: bytes | None = None   # latest frame for the live preview in the UI
    preview_t: float = 0.0
    preview_seq: int = 0
    preview_kind: str = ""              # tracking | rendering
    _preview_clock: float = 0.0
    fixed_stats: dict | None = None     # stats of the preview frame (fixed mode / picked WB)
    status: str = "queued"              # queued | analyzing | rendering | done | error | cancelled
    progress: float = 0.0
    frame: int = 0
    fps: float = 0.0
    eta: float | None = None
    error: str | None = None
    started: float = field(default_factory=time.time)
    _cancel: threading.Event = field(default_factory=threading.Event)
    _procs: list = field(default_factory=list)

    def public(self) -> dict:
        return {
            "id": self.id, "status": self.status, "progress": round(self.progress, 4),
            "frame": self.frame, "frames": self.info["frames"], "fps": round(self.fps, 2),
            "eta": None if self.eta is None else round(self.eta), "error": self.error,
            "file": self.out.name, "path": str(self.out),
            "elapsed": round(time.time() - self.started),
            "preview_seq": self.preview_seq, "preview_t": round(self.preview_t, 2),
            "preview_kind": self.preview_kind,
        }

    PREVIEW_EVERY = 0.5    # seconds of wall time between preview frames
    PREVIEW_LONG = 960

    def _preview_due(self) -> bool:
        now = time.time()
        if now - self._preview_clock < self.PREVIEW_EVERY:
            return False
        self._preview_clock = now
        return True

    def _set_preview(self, rgb_u8: np.ndarray, t: float, kind: str):
        h, w = rgb_u8.shape[:2]
        s = self.PREVIEW_LONG / max(h, w)
        if s < 1:
            rgb_u8 = cv2.resize(rgb_u8, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            self.preview_jpeg, self.preview_t, self.preview_kind = buf.tobytes(), t, kind
            self.preview_seq += 1

    def cancel(self):
        self._cancel.set()
        for p in self._procs:
            try:
                p.kill()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def run(self):
        try:
            timeline = self._analyze()
            if self._cancel.is_set():
                raise _Cancelled()
            self._tracks = self._track_skin(timeline) if self.skin_seeds else []
            if self._cancel.is_set():
                raise _Cancelled()
            self._render(timeline)
            self.status, self.progress, self.eta = "done", 1.0, 0
        except _Cancelled:
            self.status = "cancelled"
            self.out.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            if self._cancel.is_set():
                self.status = "cancelled"
            else:
                self.status, self.error = "error", str(e)
            self.out.unlink(missing_ok=True)
        finally:
            for p in self._procs:
                try:
                    p.kill()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    def _analyze(self) -> StatsTimeline | None:
        if self.mode == "fixed":
            return None
        self.status = "analyzing"
        info = self.info
        s = ANALYSIS_SIZE / max(info["width"], info["height"])
        w, h = _even(info["width"] * s), _even(info["height"] * s)
        cmd = [FFMPEG, "-v", "error", "-hwaccel", "videotoolbox", "-i", str(self.src), "-map", "0:v:0",
               "-vf", f"fps={ANALYSIS_FPS}," + _scale_filter(info, w, h) + ",format=rgb24",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._procs.append(proc)
        expected = max(1, int(info["duration"] * ANALYSIS_FPS))
        locked = self.fixed_stats.get("gains") if (self.params.wb_mode == "pick" and self.fixed_stats) else None
        times, rows, template, keys = [], [], None, None
        n = 0
        use_depth = self.params.method == "depth"
        while True:
            raw = proc.stdout.read(w * h * 3)
            if not raw or len(raw) < w * h * 3:
                break
            if self._cancel.is_set():
                raise _Cancelled()
            frame = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
            if self.input_lut is not None:
                frame = (self.input_lut.apply(frame) * 255 + 0.5).astype(np.uint8)
            t_sample = n / ANALYSIS_FPS
            n += 1
            self.progress = 0.08 * min(1.0, n / expected)
            st: dict = {}
            if locked is not None:
                st["gains"] = list(locked)
            cache = {"disp_src": predict_disparity(frame)} if use_depth else {}
            process(frame, self.params, st, cache=cache, stats_only=True)
            if template is None:
                template = st
                keys = [k for k in _SMOOTH_KEYS if k in st]
            row = _leaves(st, keys)
            if rows and len(row) != len(rows[0]):
                continue   # structure changed (e.g. no water in this frame): skip the sample
            rows.append(np.array(row))
            times.append(t_sample)
        proc.wait()
        if not rows:
            raise RuntimeError("Não consegui decodificar o vídeo")
        values = _smooth(np.stack(rows), SMOOTH_SECONDS * ANALYSIS_FPS)
        return StatsTimeline(np.array(times), values, template, keys)

    # ------------------------------------------------------------------
    def _track_skin(self, timeline):
        """Follow the clicked divers through the video (SAM 2) and measure their skin."""
        import skin
        import skin_track
        self.status = "tracking"
        p = self.params
        gpu = use_gpu()

        def measure(t, f, dmask, smask):
            st = dict(self.fixed_stats) if timeline is None else timeline.at(t)
            if gpu:
                import engine_torch as G
                with G.GPU_LOCK:
                    out, _ = G.process(f, p, st, cache={})
            else:
                out, _ = process(f, p, st, cache={})
            sk = skin.measurable_skin(out, smask)
            return skin.skin_gains(np.median(out[sk], axis=0)) if sk.sum() >= 30 else None

        def on_sample(t, f, masks):
            # live preview of the tracking: the frame with each diver outlined in pink
            if not self._preview_due():
                return
            v = f.copy()
            for m in masks:
                if m.any():
                    v[m] = (0.55 * v[m] + np.array([115, 0, 90])).astype(np.uint8)
                    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(v, cnts, -1, (255, 90, 200), 3)
            self._set_preview(v, t, "tracking")

        return skin_track.track(self.src, self.info, self.skin_seed_t, self.skin_seeds, measure,
                                input_lut=self.input_lut, cancel=self._cancel, on_sample=on_sample,
                                progress=lambda f: setattr(self, "progress", 0.08 + 0.22 * f))

    def _apply_skin(self, out_np: np.ndarray, t: float) -> np.ndarray:
        import skin
        h, w = out_np.shape[:2]
        for tr in getattr(self, "_tracks", []):
            r = tr.at(t, w, h)
            if r is None:
                continue
            diver_f, skin_f, gains = r
            ys, xs = np.nonzero(diver_f > 0.05)
            if len(xs) == 0:
                continue
            pad = int(0.03 * max(w, h))
            x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
            y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
            pr = self.params
            out_np[y0:y1, x0:x1] = skin.correct_region(out_np[y0:y1, x0:x1], diver_f[y0:y1, x0:x1],
                                                       skin_f[y0:y1, x0:x1], gains, pr.skin_strength * pr.intensity,
                                                       min(w, h), pr.skin_feather, pr.skin_body)
        return out_np

    # ------------------------------------------------------------------
    def _render(self, timeline: StatsTimeline | None):
        self.status = "rendering"
        p0 = 0.30 if getattr(self, "_tracks", None) else 0.08
        info = self.info
        W, H = info["width"], info["height"]
        if self.scale < 1:
            W, H = _even(W * self.scale), _even(H * self.scale)
        frame_bytes = W * H * 3 * 2
        fps_arg = f"{info['fps_num']}/{info['fps_den']}"

        dec_cmd = [FFMPEG, "-v", "error", "-hwaccel", "videotoolbox", "-i", str(self.src), "-map", "0:v:0",
                   "-vf", _scale_filter(info, W, H) + ",format=rgb48le",
                   "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"]
        src_bps = info["bit_rate"] or 80_000_000
        ratio = (W * H) / (info["width"] * info["height"])
        bitrate = int(np.clip(src_bps * ratio, 20_000_000, 150_000_000))
        codec_args = video_codec_args(self.codec, bitrate)[0]
        enc_cmd = [FFMPEG, "-v", "error", "-y",
                   "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{W}x{H}", "-framerate", fps_arg, "-i", "-",
                   "-i", str(self.src),
                   "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy", "-map_metadata", "1",
                   # the RGB->YUV conversion must happen in *this* filter (BT.709);
                   # an auto-inserted one would use ffmpeg's BT.601 default
                   "-vf", "scale=out_color_matrix=bt709:out_range=limited:flags=accurate_rnd+full_chroma_int,"
                          f"format={_pix_fmt(codec_args)}",
                   *codec_args,
                   "-colorspace", "bt709", "-color_primaries", "bt709",
                   "-color_trc", "bt709", "-color_range", "tv", "-movflags", "+faststart",
                   str(self.out)]
        dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_bytes)
        enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._procs += [dec, enc]

        p = self.params
        fixed = self.fixed_stats
        ws = WORK_SIZE / max(W, H)
        wW, wH = (max(2, round(W * ws)), max(2, round(H * ws))) if ws < 1 else (W, H)

        gpu = use_gpu()
        self._has_tracks = bool(getattr(self, "_tracks", None))

        lut_in = self.input_lut

        def work(idx: int, raw: bytes) -> bytes:
            if gpu and lut_in is not None:
                return _work_gpu_lut(idx, raw)
            full = np.frombuffer(raw, np.uint16).reshape(H, W, 3).astype(np.float32) * (1.0 / 65535.0)
            if lut_in is not None:
                full = lut_in.apply(full)
            small = cv2.resize(full, (wW, wH), interpolation=cv2.INTER_AREA) if (wW, wH) != (W, H) else full
            st = dict(fixed) if timeline is None else timeline.at(idx / info["fps"])
            cache = {}
            if p.method == "depth":
                cache["disp_src"] = predict_disparity((small * 255 + 0.5).astype(np.uint8))
            if gpu:
                return _work_gpu(full, small, st, cache, idx)
            out_s, _ = process(small, p, st, cache=cache)
            if self._has_tracks:
                out_s = self._apply_skin(out_s, idx / info["fps"])
            if small is full:
                out = out_s
            else:
                coef = fit_local_affine(small, out_s, 2, 1e-4)
                out = apply_local_affine(full, coef)
            return (out * 65535.0 + 0.5).astype(np.uint16).tobytes()

        def _work_gpu(full, small, st, cache, idx=0) -> bytes:
            # Hybrid: byte conversions and the OpenCV maps run on CPU worker
            # threads in parallel; only the heavy maths holds the GPU. On an
            # M1 Pro this beats an all-GPU frame (integer ops and transfers are
            # slow there and nothing would overlap).
            import torch
            import engine_torch as G
            G.prepare(small, p, st, cache)
            with G.GPU_LOCK, torch.inference_mode():
                out_s, _ = G.process(small, p, st, cache=cache, return_tensor=True)
                if self._has_tracks:
                    out_s = G.T(self._apply_skin(G.N(out_s), idx / info["fps"]))
                if small is full:
                    out = out_s
                else:
                    coef = G.fit_local_affine(G.T(small), out_s, 2, 1e-4)
                    out = G.apply_local_affine(G.T(full), coef)
                out = out[0].permute(1, 2, 0).contiguous().cpu()
            return (out.numpy() * 65535.0 + 0.5).astype(np.uint16).tobytes()

        def _work_gpu_lut(idx: int, raw: bytes) -> bytes:
            # log footage: the input LUT runs on the GPU at full size, then the
            # usual hybrid path continues from the transformed frame
            import torch
            import engine_torch as G
            st = dict(fixed) if timeline is None else timeline.at(idx / info["fps"])
            full_np = np.frombuffer(raw, np.uint16).reshape(H, W, 3).astype(np.float32) * (1.0 / 65535.0)
            with G.GPU_LOCK, torch.inference_mode():
                full_t = lut_in.apply_t(G.T(full_np))
                small_t = torch.nn.functional.interpolate(full_t, size=(wH, wW), mode="area")
                small = G.N(small_t)
            cache = {}
            if p.method == "depth":
                cache["disp_src"] = predict_disparity((small * 255 + 0.5).astype(np.uint8))
            G.prepare(small, p, st, cache)
            with G.GPU_LOCK, torch.inference_mode():
                out_s, _ = G.process(small, p, st, cache=cache, return_tensor=True)
                if self._has_tracks:
                    out_s = G.T(self._apply_skin(G.N(out_s), idx / info["fps"]))
                coef = G.fit_local_affine(small_t, out_s, 2, 1e-4)
                out = G.apply_local_affine(full_t, coef)[0].permute(1, 2, 0).contiguous().cpu()
            return (out.numpy() * 65535.0 + 0.5).astype(np.uint16).tobytes()

        workers = WORKERS_GPU if gpu else WORKERS
        pending: "queue.Queue" = queue.Queue(maxsize=workers + 2)
        err: list = []

        def writer():
            try:
                n = 0
                while True:
                    fut = pending.get()
                    if fut is None:
                        break
                    data = fut.result()
                    enc.stdin.write(data)
                    if self._preview_due():
                        # strided subsample of the 16-bit frame: a few ms, no full conversion
                        step = max(1, int(np.ceil(max(W, H) / self.PREVIEW_LONG)))
                        a = np.frombuffer(data, np.uint16).reshape(H, W, 3)[::step, ::step]
                        self._set_preview((a >> 8).astype(np.uint8), n / info["fps"], "rendering")
                    n += 1
            except Exception as e:  # noqa: BLE001
                err.append(e)
                self.cancel()

        wt = threading.Thread(target=writer, daemon=True)
        wt.start()
        total = max(1, info["frames"])
        t0 = time.time()
        idx = 0
        with ThreadPoolExecutor(workers) as ex:
            while not self._cancel.is_set():
                raw = dec.stdout.read(frame_bytes)
                if not raw or len(raw) < frame_bytes:
                    break
                pending.put(ex.submit(work, idx, raw))
                idx += 1
                self.frame = idx
                el = time.time() - t0
                self.fps = idx / el if el > 0 else 0
                self.progress = p0 + (1.0 - p0) * min(1.0, idx / total)
                self.eta = (total - idx) / self.fps if self.fps > 0 else None
            pending.put(None)
            wt.join()
        if self._cancel.is_set():
            raise _Cancelled()
        if err:
            raise err[0]
        enc.stdin.close()
        rc = enc.wait()
        dec.wait()
        if rc != 0:
            msg = enc.stderr.read().decode(errors="ignore")[-600:]
            raise RuntimeError("Falha no encoder: " + msg)


class _Cancelled(Exception):
    pass
