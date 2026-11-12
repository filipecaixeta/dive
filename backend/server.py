"""
FastAPI server for the underwater photo corrector.

  POST /api/images                      upload -> {id, width, height, ...}
  GET  /api/images/{id}/original        preview-size original (JPEG)
  POST /api/images/{id}/render          params -> preview-size result (JPEG), stats in X-Stats header
  POST /api/images/{id}/export          params + format -> full-resolution file
  POST /api/images/{id}/lut             params -> .cube 3D LUT (global part of the correction)

  POST /api/videos                      upload video -> {video, image}; image = editable frame session
  POST /api/videos/{id}/frame           {t} -> new frame session at that time
  GET  /api/videos/{id}/thumbs          JPEG strip for the timeline
  POST /api/videos/{id}/export          params -> background job
  GET  /api/jobs/{id}                   job progress
  POST /api/jobs/{id}/cancel
  GET  /api/jobs/{id}/download          finished MP4
  GET  /                                frontend
"""
from __future__ import annotations

import io
import json
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

from fastapi.concurrency import run_in_threadpool

from engine import Params, Session, build_cube_lut, to_u8, to_u16
from raw import is_raw, load_raw
import video as vid

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
UPLOADS = ROOT / "uploads"
OUTPUT = ROOT / "output"
MAX_SESSIONS = 6
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".mts"}

app = FastAPI(title="Dive Color")


class Entry:
    def __init__(self, session: Session, name: str, exif: bytes | None, icc: bytes | None):
        self.session = session
        self.name = name
        self.exif = exif
        self.icc = icc
        self.lock = threading.Lock()
        self.latest_seq = -1   # newest render request seen; older ones are dropped


_sessions: "OrderedDict[str, Entry]" = OrderedDict()
_sessions_lock = threading.Lock()


def _get(image_id: str) -> Entry:
    with _sessions_lock:
        e = _sessions.get(image_id)
        if e is None:
            raise HTTPException(404, "Image not found (the server may have restarted). Please upload it again.")
        _sessions.move_to_end(image_id)
        return e


def _encode_jpeg(rgb_u8: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_JPEG_PROGRESSIVE, 1])
    if not ok:
        raise HTTPException(500, "Failed to encode JPEG")
    return buf.tobytes()


class RenderReq(BaseModel):
    params: dict = {}
    view: str = "result"     # result | haze
    quality: str = "preview" # fast (while dragging) | preview (on release)
    seq: int = 0


class ExportReq(BaseModel):
    params: dict = {}
    format: str = "jpeg"   # jpeg | tiff16
    quality: int = 95


@app.post("/api/images")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    name = file.filename or "photo.jpg"
    if is_raw(name):
        try:
            rgb = await run_in_threadpool(load_raw, data, name)
        except Exception as e:
            raise HTTPException(400, f"Could not open the RAW file: {e}")
        return await run_in_threadpool(_register_image, rgb, name)
    try:
        im = Image.open(io.BytesIO(data))
        exif = im.info.get("exif")
        icc = im.info.get("icc_profile")
        im = ImageOps.exif_transpose(im).convert("RGB")
    except Exception:
        raise HTTPException(400, "Unsupported image format")
    rgb = np.asarray(im, dtype=np.uint8).copy()

    if exif:
        # orientation was baked into the pixels: reset the tag
        try:
            ex = Image.Exif()
            ex.load(exif)
            ex[0x0112] = 1
            exif = ex.tobytes()
        except Exception:
            exif = None

    return _register_image(rgb, file.filename or "photo.jpg", exif, icc)


def _register_image(rgb: np.ndarray, name: str, exif: bytes | None = None, icc: bytes | None = None) -> dict:
    session = Session(rgb)
    session.warm("fast")
    entry = Entry(session, name, exif, icc)

    def _warm_preview():
        with entry.lock:
            session.warm("preview")
    threading.Thread(target=_warm_preview, daemon=True).start()
    image_id = uuid.uuid4().hex[:12]
    with _sessions_lock:
        _sessions[image_id] = entry
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)

    ph, pw = entry.session.preview.shape[:2]
    return {
        "id": image_id,
        "name": entry.name,
        "width": rgb.shape[1],
        "height": rgb.shape[0],
        "preview_width": pw,
        "preview_height": ph,
        "defaults": Params().to_dict(),
    }


@app.get("/api/images/{image_id}/original")
def original(image_id: str):
    e = _get(image_id)
    prev = e.session.preview
    if prev.dtype != np.uint8:
        prev = to_u8(prev)
    return Response(_encode_jpeg(prev, 92), media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=3600"})


@app.post("/api/images/{image_id}/render")
def render(image_id: str, req: RenderReq):
    e = _get(image_id)
    p = Params.from_dict(req.params)
    which = "fast" if req.quality == "fast" else "preview"
    e.latest_seq = max(e.latest_seq, req.seq)
    with e.lock:
        if req.seq < e.latest_seq:
            return Response(status_code=204)  # superseded while waiting
        t0 = time.perf_counter()
        out, stats = e.session.render(p, which, debug=req.view == "haze", skin_mask_view=req.view == "skinmask")
    if req.view == "haze" and "_t" in stats:
        t = stats["_t"]
        vis = cv2.applyColorMap((np.clip(1 - t, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        img = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    else:
        img = to_u8(out)
    ms = (time.perf_counter() - t0) * 1000
    info = {
        "ms": round(ms),
        "seq": req.seq,
        "quality": which,
        "veiling_light": _lin_to_hex(stats["A"]),
        "veiling_bbox": stats["A_bbox"],
        "gains": [round(g, 3) for g in stats["gains"]],
        "wb_ref": _lin_to_hex(stats["wb_ref"]) if "wb_ref" in stats else None,
        "skin": stats.get("skin", []),
    }
    return Response(_encode_jpeg(img, 90), media_type="image/jpeg",
                    headers={"X-Stats": json.dumps(info), "Cache-Control": "no-store"})


class SkinReq(BaseModel):
    x: float
    y: float
    params: dict = {}


def _skin_list(e: Entry) -> list:
    return [{"bbox": d["bbox"], "click": d["click"], "detected": d["detected"]}
            for d in getattr(e.session, "skin_divers", [])]


@app.post("/api/images/{image_id}/skin")
def add_skin(image_id: str, req: SkinReq):
    """Click on a diver: outline them and their skin for the skin-tone correction."""
    e = _get(image_id)
    p = Params.from_dict(req.params)
    with e.lock:
        d = e.session.add_skin(req.x, req.y, p)
    if d is None:
        raise HTTPException(422, "No diver found at that point. Try clicking on their body or skin.")
    return {"divers": _skin_list(e), "detected": d["detected"], "skin_px": int(d["skin"].sum())}


@app.delete("/api/images/{image_id}/skin")
def clear_skin(image_id: str):
    e = _get(image_id)
    with e.lock:
        e.session.clear_skin()
    return {"divers": []}


@app.post("/api/images/{image_id}/export")
def export(image_id: str, req: ExportReq):
    e = _get(image_id)
    p = Params.from_dict(req.params)
    with e.lock:
        out, _ = e.session.render(p, "full")
    stem = Path(e.name).stem + "_corrected"

    if req.format == "tiff16":
        ok, buf = cv2.imencode(".tif", cv2.cvtColor(to_u16(out), cv2.COLOR_RGB2BGR))
        if not ok:
            raise HTTPException(500, "Failed to write TIFF")
        return Response(buf.tobytes(), media_type="image/tiff",
                        headers={"Content-Disposition": f'attachment; filename="{stem}.tif"'})

    im = Image.fromarray(to_u8(out))
    bio = io.BytesIO()
    kw = {"quality": int(np.clip(req.quality, 70, 100)), "subsampling": 0, "optimize": True}
    if e.exif:
        kw["exif"] = e.exif
    if e.icc:
        kw["icc_profile"] = e.icc
    im.save(bio, "JPEG", **kw)
    return Response(bio.getvalue(), media_type="image/jpeg",
                    headers={"Content-Disposition": f'attachment; filename="{stem}.jpg"'})


@app.post("/api/images/{image_id}/lut")
def lut(image_id: str, req: RenderReq):
    e = _get(image_id)
    p = Params.from_dict(req.params)
    with e.lock:
        text = build_cube_lut(e.session, p)
    stem = Path(e.name).stem
    return Response(text, media_type="text/plain",
                    headers={"Content-Disposition": f'attachment; filename="{stem}.cube"'})


# ----------------------------------------------------------------------------
# Video
# ----------------------------------------------------------------------------
class VideoEntry:
    def __init__(self, path: Path, name: str, info: dict):
        self.path, self.name, self.info = path, name, info
        self.thumbs: bytes | None = None
        self.input_lut = None          # lut.Lut3D for log footage (GP-Log etc.)
        self.lock = threading.Lock()


_videos: dict[str, VideoEntry] = {}
_jobs: dict[str, vid.VideoJob] = {}


def _get_video(video_id: str) -> VideoEntry:
    v = _videos.get(video_id)
    if v is None:
        raise HTTPException(404, "Video not found (the server may have restarted). Please upload it again.")
    return v


def _video_frame_image(v: VideoEntry, t: float) -> dict:
    rgb = vid.grab_frame(v.path, v.info, t)
    if v.input_lut is not None:
        rgb = v.input_lut.apply(rgb)          # float32, keeps the LUT's precision
    resp = _register_image(rgb, f"{Path(v.name).stem}_{t:06.2f}s.jpg")
    resp["t"] = t
    _sessions[resp["id"]].video_t = t         # seed time for the skin tracking
    return resp


class FrameReq(BaseModel):
    t: float = 0.0


class VideoExportReq(BaseModel):
    params: dict = {}
    image_id: str            # the frame session the user is looking at
    scale: float = 1.0       # 1 = original resolution, 0.5 = half
    mode: str = "adaptive"   # adaptive | fixed
    codec: str = "hevc"      # hevc | prores


@app.post("/api/videos")
def upload_video(file: UploadFile = File(...)):
    suffix = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    if suffix not in VIDEO_EXT:
        raise HTTPException(400, "Unsupported video format")
    UPLOADS.mkdir(exist_ok=True)
    video_id = uuid.uuid4().hex[:12]
    path = UPLOADS / f"{video_id}{suffix}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f, 16 * 1024 * 1024)
    try:
        info = vid.probe(path)
    except Exception:
        path.unlink(missing_ok=True)
        raise HTTPException(400, "Could not read this video")
    v = VideoEntry(path, file.filename or path.name, info)
    _videos[video_id] = v
    t0 = round(info["duration"] * 0.3, 2)
    return {"video": {"id": video_id, "name": v.name, **info}, "image": _video_frame_image(v, t0)}


@app.post("/api/videos/{video_id}/input_lut")
def video_input_lut(video_id: str, t: float = 0.0, file: UploadFile | None = File(None)):
    """Set (or clear, without a file) the input LUT and return the frame at t re-read through it."""
    from lut import parse_cube
    v = _get_video(video_id)
    if file is None:
        v.input_lut = None
    else:
        try:
            v.input_lut = parse_cube(file.file.read().decode("utf-8", "ignore"), file.filename or "lut.cube")
        except Exception as e:
            raise HTTPException(400, f"Invalid LUT: {e}")
    resp = _video_frame_image(v, t)
    resp["input_lut"] = v.input_lut.name if v.input_lut else None
    return resp


@app.post("/api/videos/{video_id}/frame")
def video_frame(video_id: str, req: FrameReq):
    return _video_frame_image(_get_video(video_id), req.t)


@app.get("/api/videos/{video_id}/thumbs")
def video_thumbs(video_id: str):
    v = _get_video(video_id)
    with v.lock:
        if v.thumbs is None:
            v.thumbs = _encode_jpeg(vid.thumbnail_strip(v.path, v.info), 80)
    return Response(v.thumbs, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.post("/api/videos/{video_id}/export")
def video_export(video_id: str, req: VideoExportReq):
    v = _get_video(video_id)
    e = _get(req.image_id)
    p = Params.from_dict(req.params)
    with e.lock:
        fixed = e.session.stats_for(p)   # stats of the frame on screen
    OUTPUT.mkdir(exist_ok=True)
    codec = "prores" if req.codec == "prores" else "hevc"
    suffix = "_corrected" if req.scale >= 1 else "_corrected_half"
    ext = vid.video_codec_args(codec, 0)[1]
    out = OUTPUT / f"{Path(v.name).stem}{suffix}{'_prores' if codec == 'prores' else ''}{ext}"
    job_id = uuid.uuid4().hex[:12]
    seeds = [{"diver": d["diver"], "skin": d["skin"]} for d in getattr(e.session, "skin_divers", [])]
    job = vid.VideoJob(id=job_id, src=v.path, out=out, info=v.info, params=p,
                       scale=min(1.0, max(0.25, req.scale)), mode=req.mode, fixed_stats=fixed, codec=codec,
                       input_lut=v.input_lut, skin_seeds=seeds, skin_seed_t=getattr(e, "video_t", 0.0))
    _jobs[job_id] = job
    threading.Thread(target=job.run, daemon=True).start()
    return job.public()


def _get_job(job_id: str) -> vid.VideoJob:
    j = _jobs.get(job_id)
    if j is None:
        raise HTTPException(404, "Export not found")
    return j


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return _get_job(job_id).public()


@app.get("/api/jobs/{job_id}/preview")
def job_preview(job_id: str):
    j = _get_job(job_id)
    if j.preview_jpeg is None:
        raise HTTPException(404, "No preview yet")
    return Response(j.preview_jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    j = _get_job(job_id)
    j.cancel()
    return j.public()


@app.get("/api/jobs/{job_id}/download")
def job_download(job_id: str):
    j = _get_job(job_id)
    if j.status != "done" or not j.out.exists():
        raise HTTPException(409, "The video is not ready yet")
    media = "video/quicktime" if j.out.suffix == ".mov" else "video/mp4"
    return FileResponse(j.out, media_type=media, filename=j.out.name)


def _lin_to_hex(rgb) -> str:
    from engine import linear_to_srgb
    s = linear_to_srgb(np.array(rgb, np.float32))
    r, g, b = (np.clip(s, 0, 1) * 255).round().astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
