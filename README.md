# Dive Color

Color correction for underwater photos and videos, built mainly for GoPro footage. HTML/JS frontend, Python processing (FastAPI + OpenCV + PyTorch + ffmpeg).

![Dive Color: before and after](docs/screenshot.jpg)

## Installation

Requirements:

- macOS, Linux or Windows (WSL) with **Python 3.12** (3.13 also works)
- **ffmpeg** for video
- Optional: **cmake** and **git**, to build the GoPro RAW converter (`.GPR`)
- Optional: a GPU (Apple Silicon via MPS or NVIDIA via CUDA). Without one it runs on the CPU, more slowly.

macOS (Homebrew):

```bash
brew install python@3.12 ffmpeg cmake
```

Ubuntu/Debian:

```bash
sudo apt install python3.12 python3.12-venv ffmpeg cmake git
```

Clone and run:

```bash
git clone git@github.com:filipecaixeta/dive.git
cd dive
./run.sh
```

On the first run, `run.sh` creates `.venv`, installs the dependencies from `requirements.txt` and builds `gpr_tools` (if `cmake` is available). The AI models (Depth Anything V2, Grounding DINO, SAM 2) are downloaded from Hugging Face the first time each feature is used.

Open http://localhost:8765, drop in a photo or video and adjust the controls. The preview updates while you drag. To use another port: `PORT=9000 ./run.sh`.

## Layout

```
backend/     FastAPI server and correction engine (numpy, PyTorch, video, skin, RAW)
frontend/    user interface (HTML/CSS/JS, no build step)
docs/        README image
run.sh       sets up the environment and starts the server
```

Folders created at runtime and kept out of git: `uploads/` (uploaded videos), `output/` (exported videos), `tools/` (gpr_tools), `media/` (local test photos and videos).

## Input

| Type | Formats | Notes |
|---|---|---|
| Photo | JPEG, PNG, TIFF | 8 or 16 bits |
| RAW photo | **.GPR (GoPro)**, DNG, CR2/CR3, NEF, ARW, ORF, RW2, RAF | 16-bit linear, without the camera's white balance. GPR goes through GoPro's `gpr_tools` (LibRaw cannot decode the VC-5 codec) |
| Video | MP4, MOV (HEVC/H.264, 8 or 10-bit) | GP-Log: load the official GoPro LUT under "Input LUT" |

## Output

- Photo: JPEG 95 (4:4:4, EXIF preserved), 16-bit TIFF, `.cube` LUT.
- Video: 10-bit HEVC MP4 (delivery) or **ProRes 422 HQ MOV** (editing). During export, the progress card shows a live preview (diver tracking, then the final frame), refreshed every ~0.5 s. The encoder is picked from what is available: VideoToolbox on Mac, NVENC on NVIDIA GPUs, libx265/prores_ks in software.

## Two correction methods

- **Classic** (default): estimates haze from color (underwater dark channel) and applies a per-scene white balance. Fast, robust, no AI.
- **Depth (beta)**: estimates the distance of each point with Depth Anything V2 and applies the Sea-thru physical model (Akkaynak & Treibitz, 2019): haze and color loss computed per distance. Adaptations for wide-angle GoPro footage: position-dependent haze map (vignetting and light gradient), water detection by color, and haze evidence for distant objects. More faithful neutral colors and cleaner mid-ground; it still struggles with translucent structures and can crush shadows.
  Variables: `DIVE_DEPTH_MODEL=Small|Base|Large` (default Base), `DIVE_DEPTH_INPUT` (input resolution, default 756).

## Diver skin tones (backend/skin.py, backend/skin_track.py)

- **"Skin: click on the diver"** button (White balance section). The click picks the diver; Grounding DINO finds the body and the skin areas (face, arms, legs, hands) and SAM 2 cuts out the outline. If the detector does not recognize anything, the app cuts out what was clicked and warns you.
- The correction moves the skin **hue** to the vectorscope skin line (~123°, the same for every ethnicity), keeping brightness and saturation: it does not lighten dark skin. It is applied to the whole diver (same distance = same color loss): full on skin, half on the rest, almost nothing on black gear. Control: "Skin correction".
- Each diver's mask has a feathered edge and can be viewed and adjusted: **"Mask"** mode in the view bar (pink = where and how strongly the correction applies), **"Mask edge softness"** and **"Skin only ↔ whole diver"** (default in the middle: at the "whole" end, black gear tends to turn red).
- Several divers per image (one click each). The first click takes ~15 s (loads the models), then a few seconds.
- **Video:** divers marked on the current frame are tracked through the video (streaming SAM 2, 5 samples/s, forward and backward, stopping when the diver disappears). Skin color is measured at each sample and smoothed over time. On an M1 Pro, ~10 s of tracking per second of video in which the diver appears.
- Limits: subtle correction when almost no skin is visible; a diver who leaves and comes back much later may not be found again; only divers marked on the current frame are included in the export.

## GPU engine (backend/engine_torch.py)

- PyTorch: MPS on Mac, CUDA on NVIDIA, CPU as a last resort. `DIVE_GPU=0` forces the numpy engine.
- Statistics (white balance, levels, Sea-thru fits) stay in numpy at 640 px. So do control-independent maps that use OpenCV elliptical morphology, once per image.
- The numpy engine (`engine.py`) is the reference. `backend/tests/test_engines.py` compares the two (60 dB today; fails below 45 dB): `.venv/bin/python backend/tests/test_engines.py photo.jpg`.

| On an M1 Pro | numpy | GPU |
|---|---|---|
| Draft while dragging (760 px) | ~110 ms | ~20-40 ms |
| Preview (1800 px) | ~450 ms | ~90-160 ms |
| Preview, depth method | ~1500 ms | ~190 ms |
| Export 4K video (46 s) | ~9 min (2.7 fps) | ~4 min (5.7 fps) |

For video, the limit becomes the M1 Pro GPU itself (~170 ms of compute per 4K frame). The hybrid path (conversions and maps on the CPU in parallel, heavy math on the GPU) was the fastest; running everything on the GPU was slower because integer operations and transfers are expensive on MPS and nothing overlapped.

## How the real-time preview works

- When a photo is opened, the server keeps versions of it at 640, 760, 1024 and 1800 px, plus the original.
- Anything that does not depend on the controls (backscatter light, haze map, base/detail split) is computed once and cached.
- While you drag a control, the frontend requests a 760 px draft (~110 ms). At most one request is in flight at a time, carrying the latest parameters.
- When you release (or pause for ~0.3 s), it requests the 1800 px version (~0.5 s).
- Export processes the original resolution with the same statistics, so the final file matches the preview.

## Pipeline (backend/engine.py)

1. **Backscatter light**: quad-tree search for the open-water region (smooth, blue/green, no red).
2. **Haze removal**: underwater dark channel (G and B only) with morphological opening, to avoid halos on thin branches, refined with a guided filter. The inversion is done only on the base layer, so water noise is not amplified.
3. **Recover red/blue**: channel compensation from green (Ancuti et al.).
4. **White balance**: automatic (foreground-weighted shades-of-gray) or eyedropper.
5. **Water color**: open water goes to a chosen blue hue instead of turning gray.
6. **Levels, exposure, soft highlight shoulder, S-curve**.
7. **Local contrast** with an edge-aware filter, reduced in open water.
8. **Vibrance/saturation**, noise reduction (stronger in the water), thresholded sharpening.
9. **Overall strength**: blend with the original.

## Video (backend/video.py)

- When a video is opened, it goes to `uploads/` and one frame becomes a regular editing session. The timeline changes the frame used for adjusting.
- **Adaptive color** (default): on export, it analyzes ~2 frames per second, computes the correction statistics (backscatter light, white balance, levels) for each and smooths them over time (median + ~1 s Gaussian). The correction follows scene changes without flicker.
- **Fixed to this frame**: uses the correction of the frame on screen for the whole video.
- Render: each frame is read as 16-bit RGB, the full pipeline runs at 1280 px, and the correction is transferred to the original resolution as a local 3x4 color transform (color guided filter). It keeps all the original detail, lands ~35 dB from direct full-resolution processing and is ~4x faster. 6 frames in parallel.
- Output: 10-bit HEVC (VideoToolbox), BT.709, audio copied, bitrate equivalent to the original. Goes to `output/`.
- On an M1 Pro, the sample video (46 s, 3840x3360, 30 fps) takes about 8 to 9 minutes.
- The YUV→RGB conversion declares the file's range and matrix. ffmpeg's default (BT.601) shifts the hue.

## Export

- JPEG quality 95, 4:4:4, with EXIF and color profile preserved.
- 16-bit TIFF.
- 33³ `.cube` LUT for video (DaVinci, Premiere, Final Cut). It contains only the global part of the correction. Haze removal and local contrast depend on pixel position and do not fit in a LUT.
