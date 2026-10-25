# Mac test plan: LTX-2.5 and MiniMax H3 (32 GB Apple Silicon)

**Status:** Done (unreleased, 2026-10-25). The plan ran on an M5 with 32 GiB, then the failed steps ran again on the fix branch `fix/mac-video` (all passed; fixes in the commits "download only the files macOS LTX loads, at pinned revisions", "run MiniMax H3 image modes on macOS in stages" and "extend with the clip's own model when its downloads are present", and in the `test(video)` commits that made the test suite pass on macOS). See [Results](#results-2026-10-25) and [Follow-ups](#follow-ups). The steps below are kept as the record of what was run; the H-c and W2 bars are the corrected ones.

## Summary

1. **What's new:** LTX-2.5 (`ltx25-4`, "LTX-2.5 4-bit") and MiniMax H3 (`h3`, "MiniMax H3") now run on macOS. Neither has run on Apple hardware. This plan checks that they work, fit in 32 GB and land near their estimates, and that LTX-2.3 (`ltx-4`) still runs.
2. **Time:** about 3 hours (the table below), of which about 2 hours 15 minutes is unattended generation. The 138 GB of downloads overlap setup and T0; on a slow connection L1 and H-a wait for them, and that wait is not in the 3 hours.
3. **Disk:** about 170 GB of new files, so keep **at least 180 GB free**: LTX-2.5 30.5 GB, H3 107.2 GB (INT8 transformer 37.8, FL2VA components 68.0, turbo LoRA 1.4), the one-time `h3@mlx-te4` text-encoder copy about 15 GB, `klein4b` 16 GB if you don't have it, the two test LoRAs 1.1 GB, outputs about 2 GB.
4. **Order:** S1-S2 (facts and helpers, in a worktree of `main`), B0 in a second checkout at `0c48aae` (the last commit before the video work), S3-S5 (install, start both downloads with LTX-2.5 first, make the test inputs), T0 while the downloads run, then L1-L5 (LTX), H-a to H-f (H3 on the command line), W1-W5 (Web UI), and V (all the looking, at the end).
5. **Not run on purpose:** H3 `--upscale` and `--review` (a Mac refuses both, and unit tests cover that), `xl` sizes, the 24 GB low-memory streaming path, and anything on Windows/Linux.

| Step | What | Min |
|---|---|---|
| S1-S5 | Setup, branch, both downloads started, two test images, one LoRA | 20 |
| B0 | Optional: `ltx-4` on the old branch, saved for L5 | 6 |
| T0 | `make check` and both MLX contract tests (during the downloads, so not added to the total) | 10 |
| L1-L5 | `ltx25-4` text to video, image to video, Extend, upscale with review; `ltx-4` against B0 | 33 |
| H-a to H-f | H3: first use, 4 steps, image to video, Extend, Extend with a LoRA, refused LoRA | 67 |
| W1-W5 | Workspace: H3 controls, first and last frame, Extend panel, cancel, `ltx25-4` | 36 |
| V | One visual batch | 15 |
| Total | 20 + 6 + 33 + 67 + 36 + 15 | about 3 h (177 min) |

## Results (2026-10-25)

Header: Apple M5, 32 GiB, macOS 26.4, 195 GB free at the start. First run on commit `4d74ff4` (a worktree of `main`), mlx 0.32.3, mlx-vlm 0.6.10, GPU budget 25.0 GiB (the plan assumed about 21: the M5 recommends more). Swap Δ was about zero in every generation step. Sound was judged by level only (`ffmpeg volumedetect`); nobody listened.

### First run

| Area | Outcome |
|---|---|
| Downloads (S4) | LTX-2.5: 1 part, 30.50 GB, badge 15.1 GB. H3: 3 parts, 107.19 GB, badge 15.0 GB. |
| LTX-2.5 (L1-L5) | Text to video 193 s at a 17.9 GiB peak (estimate 15; flagged, 2.8 GiB over the badge). Image to video frame 0 at 27.8 dB (bar 28, a marginal miss; the clip has little motion). Extend 138 s at 23.8 GiB, close to the 25 GiB budget; seam jump 1.57, seam PSNR 35.2 dB, no audio dip. Upscale with review: 11 steps, 384×224 preview, 768×448 result. `ltx-4` frames equal B0's bit for bit. |
| H3 text to video (H-a, H-b, H-f) | 15.6 min total at a 12.5 GiB peak, generation 0.66 times the estimate; the 4-step clip took 3.0 min at 11.2 GiB; the LoRA refusal message appeared and nothing was written. Sound present but quiet, real stereo. |
| H3 image to video, first and last frame, Extend (H-c, H-d, H-e, W2) | Failed. |
| Workspace (W1, W3-W5) | Passed: step snapping, the three time lines, no Upscale for H3 or LTX on a Mac, a cancel that ended 210 s after the click with no output, and an image job after an H3 job within 0.6 % of its baseline. |
| `make check` (T0) | 7536 passed, 76 failed, 20 errors, 40 skipped: tests that did not skip on macOS (CUDA H3 backend, bitsandbytes, torchvision) or expected the Linux preset; one MLX contract test was wrong. |

Bugs found, all since fixed on `fix/mac-video`:

1. H3 image to video, first and last frame and Extend failed on macOS: the Qwen3-VL image processor needs torchvision, which is not installed there.
2. After that, the vision tower's patch-embed weight was in PyTorch layout, and the text-encoder fork's `encode` could not scatter the vision rows into the prompt.
3. CLI Extend of an H3 clip without `-m` fell back to the default video model, because the Mac model check looked for a full folder.
4. Loading `ltx-4` fetched the whole `dgrauet/ltx-2.3-mlx-q4` repo (about 49 GB more than it uses), and again every time a file was missing; two runs filled the disk.
5. The test suite was not macOS-clean.

A suspected bug was withdrawn: LTX-2.5 with `--upscale 2` gave near-silent sound only because seed 42 at 384×224 does so with one stage as well.

A prototype of the three H3 image fixes (runtime patches, no repo edits) made H-c work at 5.0 min, but at a 28.9 GiB peak against the budget of 25: the image modes loaded the text encoder, vision tower and the 10.4 GB float32 video VAE together (25.6 GiB). That is why the fix runs them in stages.

### Re-run on the fix branch

Worktree on the fix branch after the Extend fix, then with the image-mode estimate. `make check` exit 0 (7830 passed, 39 skipped), `packages/` unchanged.

| Step | Result |
|---|---|
| H-c (`xs`, 73 frames, image to video) | 275 s, peak **13.98 GiB** (bar at most 15; the prototype peaked at 28.9). Frames identical to the prototype (73 of 73 `framemd5` lines). Frame 0 27.7 dB / 0.866. |
| H-c at `s` (768×448, 73 frames) | 595 s (9.9 min), peak 14.19 GiB, frame 0 27.2 dB. |
| H-d (Extend, no `-m`) | `Loading H3 video model: MiniMaxAI/MiniMax-H3@42ed227...`, no fallback warning; 124 frames at 13.98 GiB; seam jump 0.79, seam PSNR 25.7 dB (source p10 20.9), no audio dip of 6 dB or more in any window. |
| H-e (Extend with a user LoRA at 0.5) | 124 frames at 15.48 GiB (the LoRA costs about 1.5 GiB); seam jump 0.82, no cut to another shot. |
| W2 (Web UI, both frames) | Time line `About 9 min (estimated)`; the job card counted denoise steps 0 to 7 of 8; frame 72 against the end image 36.9 dB / 0.953 (bar 21 / 0.70); frame 0 27.2 dB. |
| Extra: CLI first and last frame | Frame 0 27.0 dB, frame 72 against the end image 37.0 dB / 0.956. |
| Contact sheets (V6-V8) | Frame 0 is the start image, then a coherent clip; the Extend clips continue the source. No garbage frames, no tiling. |
| Phase peaks (MLX, GiB) | Keyframes 14.1, vision 1.2-1.5, prompt 13.4-13.7, denoise 2.6-3.7, video decode 11.3-11.8, audio decode 2.6-3.1. |
| LTX-2.5 after deleting and downloading it from the Models page | Dialog "28.8 GB", 1 part, 16 files (the listed ones, nothing else). Text to video, image to video and upscale frames identical to the first run. Extend: the first try gained +1.07 GiB swap with the Web UI server running; on a quiet machine the same command gained none (peak 23.8 GiB, close to the 25 GiB budget). |
| `ltx-4` after deleting it and Gemma 3 | Dialog "28.5 GB in 2 parts" (20.5 + 8.1 GB, 10 files plus Gemma 3). Text to video and upscale frames identical to the first run (B0 and the first `--upscale 2` run). Image to video completes (frame 0 28.3 dB). Extend keeps `ltx-4`, no fallback. |
| `ltx-8` (download, one text-to-video render, delete) | 29.75 GB pack in 10 files, about 9.5 min; peak 22.67 GiB; swap Δ 0. |

### What changed in the code and the figures

- `utils/h3_memory.py`: the image modes count the 14.2 GiB staged load phase measured above. The time estimate for a 768×448 image-to-video clip is about 14 min against 9.9 measured (see Follow-ups).
- The docs and CHANGELOG no longer say H3 and LTX-2.5 are untested on a Mac.

## How to use this plan

- Work top to bottom. Each step has a command or UI steps, **Pass** (what to look for) and **Record** (what to write in the results table).
- Run commands from the repo root in a terminal that has run `source "$ZIV_TEST_DIR/env.sh"`. That file sets up:
  - `run <ID> ziv-video|ziv-image …` runs the command under `/usr/bin/time -l`, logs it to `$ZIV_TEST_DIR/logs/<ID>.log` with start and end times and the swap before and after, and prints the wall time and peak memory. Never put `uv run` in front of `/usr/bin/time -l`: that measures `uv`.
  - `peak <ID>` prints the peak in GiB (the app's "GB" are GiB too). `latest <dir>` is the newest `.mp4` in a folder, `ui_latest` the newest one the Web UI made. `probe <file>` shows size, frames, fps and audio. `img_start` and `img_end` are the two test images.
  - `ana frame <clip> <index> <image>`, `ana seam <source> <extended>` and `ana audio <source> <extended>` print frame, seam and audio figures, with the same formulas as the Windows/Linux runs.
  - `$FOX`, `$EXT`: the named prompts. Use them wherever a step says "the fox prompt", in the CLI and typed into the Workspace, so runs compare like for like.
- The estimates come from the code: derived from file sizes (LTX) or from someone else's M4 Pro (H3). They are not targets. **Flag**, don't fail, a peak more than 2 GiB over its estimate or a time more than 1.5 times its estimate.
- Every H3 CLI run prints `MiniMax H3 at <W>x<H>, <N> frames: estimated <M> min (excludes the one-time weight preparation).` before loading. The Workspace shows the same estimate as `About <M> min (estimated).` under Duration and under the Extend length. A memory note only shows when a clip is tight or too large.

## Setup

### S1. Machine facts
```bash
sysctl -n machdep.cpu.brand_string; echo "$(( $(sysctl -n hw.memsize) / 1073741824 )) GiB"; sw_vers -productVersion; df -h ~ | tail -1; ffmpeg -version | head -1
```
- **Pass:** Apple M-series, 32 GiB, at least 180 GB free, ffmpeg present (`brew install ffmpeg` if not).
- **Record:** chip, memory, macOS, free disk.

### S2. Scratch folder and helpers (once; `source` it in every new terminal)
The work is on `main`. Make a worktree for the test and run this from it, so `ZIV_REPO` (whose `.venv` the `run` helper uses) points there:
```bash
git worktree add -b test/mac-video ../ziv-test main && cd ../ziv-test
```
```bash
export ZIV_TEST_DIR=/path/to/an/empty/scratch/folder   # outside the repo
mkdir -p "$ZIV_TEST_DIR/logs"
cat > "$ZIV_TEST_DIR/env.sh" <<'EOF'
export T="$ZIV_TEST_DIR"
export PYTHONUNBUFFERED=1
FOX="A red fox trots through fresh snow at dawn, soft golden light, its breath visible"
EXT="The fox stops and looks at the camera"
UI_OUT=~/.ziv/outputs   # the Web UI's output folder; change it if you moved it
run() {
  local id=$1; shift; mkdir -p "$T/logs"
  sysctl -n vm.swapusage > "$T/logs/$id.swap-before"
  { echo "start $(date +%T)"; /usr/bin/time -l "$ZIV_REPO/.venv/bin/$@" 2>&1; echo "end $(date +%T)"; } | tee "$T/logs/$id.log"
  sysctl -n vm.swapusage > "$T/logs/$id.swap-after"
  echo "--- $id: $(grep -E ' real ' "$T/logs/$id.log" | tail -1 | awk '{print $1}') s wall, peak $(peak "$id")"
  echo "swap before: $(cat "$T/logs/$id.swap-before")"; echo "swap after:  $(cat "$T/logs/$id.swap-after")"
}
peak() { awk '/peak memory footprint/ {b=$1} END {printf "%.1f GiB\n", b / 1073741824}' "$T/logs/$1.log"; }
latest() { ls -t "$1"/*.mp4 2>/dev/null | head -1; }
ui_latest() { find "$UI_OUT" -name '*.mp4' -print0 | xargs -0 ls -t | head -1; }
probe() { ffprobe -v error -count_frames -show_entries stream=codec_type,width,height,nb_read_frames,r_frame_rate,channels,sample_rate -of compact "$1"; }
img_start() { ls -t "$T"/img/start/*.png | head -1; }
img_end() { ls -t "$T"/img/end/*.png | head -1; }
ana() { "$ZIV_REPO/.venv/bin/python" -I "$T/ana.py" "$@"; }
EOF
echo "export ZIV_REPO=\"$PWD\"" >> "$ZIV_TEST_DIR/env.sh"
cat > "$ZIV_TEST_DIR/ana.py" <<'PY'
"""Frame, seam and audio checks for the H3 steps; CPU only, nothing is written."""

import json
import subprocess
import sys

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter


def frames(path):
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", path], capture_output=True, text=True, check=True).stdout)["streams"][0]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, info["height"], info["width"], 3)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def ssim(a, b):
    a, b = (x.astype(np.float64) @ [0.299, 0.587, 0.114] for x in (a, b))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ma, mb = gaussian_filter(a, 1.5), gaussian_filter(b, 1.5)
    va, vb, cov = gaussian_filter(a * a, 1.5) - ma**2, gaussian_filter(b * b, 1.5) - mb**2, gaussian_filter(a * b, 1.5) - ma * mb
    return float((((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma**2 + mb**2 + c1) * (va + vb + c2))).mean())


def frame(video, index, image):
    f = frames(video)[int(index)]
    with Image.open(image) as im:
        ref = np.asarray(ImageOps.fit(im.convert("RGB"), (f.shape[1], f.shape[0]), method=Image.Resampling.BILINEAR))
    print(f"frame {index}: PSNR {psnr(f, ref):.1f} dB, SSIM {ssim(f, ref):.3f}")


def seam(source, extended):
    n, ext = len(frames(source)), frames(extended)
    d = np.array([np.mean(np.abs(ext[i + 1].astype(np.int16) - ext[i].astype(np.int16))) for i in range(len(ext) - 1)])
    p = np.array([psnr(ext[i], ext[i + 1]) for i in range(len(ext) - 1)])
    print(f"jump ratio {d[n - 1] / np.median(d[: n - 1]):.2f} (seam {d[n - 1]:.2f}, source median {np.median(d[: n - 1]):.2f})")
    print(f"seam PSNR {p[n - 1]:.1f} dB; source frame-to-frame p10 {np.percentile(p[: n - 1], 10):.1f} dB, median {np.median(p[: n - 1]):.1f} dB")


def audio(source, extended, fps=24):
    n = len(frames(source))
    rate = int(json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate", "-of", "json", extended], capture_output=True, text=True, check=True).stdout)["streams"][0]["sample_rate"])
    x = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", extended, "-map", "0:a:0", "-ac", "1", "-f", "f32le", "-"], capture_output=True, check=True).stdout, np.float32)
    w, at = rate // 50, round(n / fps * rate)
    db = [20 * np.log10(max(float(np.sqrt(np.mean(x[i : i + w].astype(np.float64) ** 2))), 1e-9)) for i in range(0, len(x) - w + 1, w)]
    k = at // w
    before, after = float(np.median(db[max(0, k - 50) : k])), db[k : k + 15]
    print(f"audio: {before:.1f} dB in the second before the seam; lowest 20 ms in the 300 ms after it {min(after):.1f} dB (dip {before - min(after):.1f} dB); {sum(a <= before - 6 for a in after)} of 15 windows 6 dB or more below")


if __name__ == "__main__":
    {"frame": frame, "seam": seam, "audio": audio}[sys.argv[1]](*sys.argv[2:])
PY
source "$ZIV_TEST_DIR/env.sh"
```
- **Pass:** `type run` and `type ana` show the functions. `ana.py` only reads clips and images. `frame` gives RGB PSNR and luma SSIM of one frame against the image (centre-fitted to the clip); `seam` gives the jump ratio (the seam's mean frame change over the source's median) and the seam's PSNR against the source's own 10th percentile and median; `audio` gives the level in the second before the seam and the lowest 20 ms in the 300 ms after it.

### B0. (Optional, 6 min) The LTX-2.3 baseline before the update
Run it in a second checkout at `0c48aae`, the last commit before the video work, so L5 can compare against the code before the update. It needs `ltx-4` downloaded. Both commits lock mlx 0.32.3 and the vendored LTX packages are unchanged, but the Mac LTX backend changed, so identical frames are expected.
```bash
git -C "$ZIV_REPO" worktree add --detach ../ziv-baseline 0c48aae
B0_REPO="$(cd "$ZIV_REPO/../ziv-baseline" && pwd)"
(cd "$B0_REPO" && uv sync && uv run python -c "import mlx.core as mx; print('mlx', mx.__version__)")
ZIV_REPO="$B0_REPO" run B0 ziv-video -m ltx-4 --size s --frames 121 --prompt "$FOX" --seed 42 -o $T/b0
ffmpeg -v error -i "$(latest $T/b0)" -map 0:v -f framemd5 - | grep -v '^#' > $T/logs/b0.framemd5
```
- **Pass:** the clip is 121 frames. **Record:** wall time, `peak B0`, the mlx version. The old and new badge figures are not compared: the old badge was not recorded.

### S3. Install in the worktree
```bash
cd "$ZIV_REPO" && git log --oneline -1
make install
uv run python -c "import mlx.core as mx, mlx_vlm; print('mlx', mx.__version__, 'mlx-vlm', mlx_vlm.__version__)"
uv run python -c "from zvisiongenerator.backends import accelerator_memory_budget_or_none as b; x = b(); print(round(x.gpu_bytes / 2**30, 1), 'GiB budget; fits up to', round(x.gpu_bytes * 1.1 / 2**30, 1), 'GiB, tight up to', round(x.gpu_bytes * 1.5 / 2**30, 1), 'GiB')"
```
- **Pass:** install finishes; mlx 0.32.3 and mlx-vlm 0.6.10 (the lock); the budget is about 21 GiB.
- **Record:** the commit hash, the mlx versions, the budget line.

### S4. Start the Web UI and both downloads
In a second terminal, and leave it running all session:
```bash
cd "$ZIV_REPO" && uv run ziv ui
```
(`--host local` would make it reachable from the whole network; the test does not need that.)
Models page: click **Download** on **LTX-2.5 4-bit** first, then on **MiniMax H3**. Confirm **FLUX.2 Klein 4B** is downloaded, or add its dialog size to the disk budget (16 GB).
- **Pass:** the LTX-2.5 dialog lists **one part**, `MarceloFSO/ltx-2.5-mlx-q4-fast`, about 30.5 GB, with no Gemma 3 part. The H3 dialog lists **three parts**: `water1234/MiniMax-H3-MLX-Argus-Calibrated-INT8` about 37.8 GB, `MiniMaxAI/MiniMax-H3` (FL2VA components) about 68.0 GB, `lightx2v/Minimax-h3-Turbo` about 1.4 GB; about 107.2 GB in all, no text-encoder or bf16 part. A download interrupted by sleep or network continues, it does not restart.
- When each finishes its row shows downloaded with a memory badge: **LTX-2.5 about 15 GB**, **MiniMax H3 about 15.0 GB** ("Estimated for a 5 s clip at 768x448 from the text. The Workspace checks the size and length you choose.").
- **Record:** the dialog sizes, each download's time, both badge figures, any interruption.

### S5. Test inputs (made by the app; never use your own images)
```bash
source "$ZIV_TEST_DIR/env.sh" && cd "$ZIV_REPO"
uv run ziv-image -m klein4b --prompt "A lighthouse on a rocky coast at sunset, photo" --ratio 16:9 --seed 7 -o $T/img/start
uv run ziv-image -m klein4b --prompt "The same lighthouse at night, beam lit, photo" --ratio 16:9 --seed 8 -o $T/img/end
img_start; img_end
# two MiniMax H3 LoRAs in $T/loras: h3-accepted.safetensors (transformer blocks only, about 0.3 GB)
# and h3-refused.safetensors (one that changes the timestep embedding, about 0.8 GB)
uv run ziv model lora -i $T/loras/h3-accepted.safetensors
```
- **Pass:** two `.png` paths print; both LoRA files exist; the import prints `LoRA imported: …`.
- The accepted LoRA is now available as `h3-accepted` (H-e, W1). The refused LoRA is only for H-f, which expects it to be refused; it stays in the scratch folder.

### T0. Tests (while the downloads run)
```bash
uv run pytest tests/test_ltx_mlx_contract.py tests/test_h3_mlx_contract.py -q -rs 2>&1 | tee $T/logs/T0.log
make check 2>&1 | tee $T/logs/T0-check.log
```
- **Pass:** `make check` passes, and the two contract tests have **no skips**: on a Mac their MLX import checks must run.
- **Record:** pass, skip and fail counts, and every skip reason.

## LTX: `ltx25-4`, and `ltx-4` against B0

The H3 download is likely still running during these steps. Its disk and CPU load can slow them, so note for each step whether it was running, and don't flag a slow time from a step that overlapped it.

### L1. Text to video with the defaults
```bash
run L1 ziv-video -m ltx25-4 --prompt "$FOX" --seed 42 -o $T/l1
probe "$(latest $T/l1)"
```
- **Pass:** 768×448, 121 frames, `24/1` fps, an audio stream, a progress line of 8 steps, and no Gemma 3 download line.
- **Record:** wall time, s/step, `peak L1` against the S4 badge (about 15 GiB), swap.

### L2. Image to video
```bash
run L2 ziv-video -m ltx25-4 --image "$(img_start)" --prompt "Waves crash against the rocks as the light turns" --seed 42 -o $T/l2
probe "$(latest $T/l2)"
ana frame "$(latest $T/l2)" 0 "$(img_start)"
```
- **Pass:** 121 frames with audio; frame-0 PSNR **at least 28 dB** (`ana frame` centre-fits the image the way the clip was made).
- **Record:** PSNR and SSIM, wall time, `peak L2`.

### L3. Extend
```bash
run L3 ziv-video --extend "$(latest $T/l1)" --prompt "The fox stops and looks straight at the camera." --extend-seconds 2 -o $T/l3
probe "$(latest $T/l3)"
ana seam "$(latest $T/l1)" "$(latest $T/l3)"
```
- **Pass:** the log has `Loading LTX video model: MarceloFSO/ltx-2.5-mlx-q4-fast@` (the CLI prints the resolved reference, not the alias); 121 + 48 = **169 frames** with audio. Windows/Linux measured a seam PSNR of 35.8 dB. Flag a jump ratio over 2, or a seam PSNR more than 5 dB under the source's 10th percentile.
- **Record:** the `ana seam` line, wall time, `peak L3`.

### L4. Upscale with review
```bash
run L4 ziv-video -m ltx25-4 --prompt "$FOX" --upscale 2 --review --seed 42 -o $T/l4   # when asked, press y
probe "$(latest $T/l4)"
```
- The command line does not open the preview: it prints `Small render: <path>`. Open that file from another terminal (`open <path>`), then press `y`.
- **Pass:** the half-size preview is 384×224 and progress is one run of **11 steps** (8 + 3); after `y` the clip is 768×448, 121 frames.
- **Record:** wall time, `peak L4`, whether the preview appeared without a memory spike.

### L5. `ltx-4` after the update
Run B0's command on the branch and compare. Skip the compare if you skipped B0.
```bash
run L5 ziv-video -m ltx-4 --size s --frames 121 --prompt "$FOX" --seed 42 -o $T/l5
ffmpeg -v error -i "$(latest $T/l5)" -map 0:v -f framemd5 - | grep -v '^#' > $T/logs/l5.framemd5
diff -q $T/logs/b0.framemd5 $T/logs/l5.framemd5 && echo IDENTICAL
```
- **Pass:** it completes with 121 frames. Identical frames to B0 is the ideal; a mismatch is a flag only when the mlx version printed in B0 and S3 is the same.
- **Record:** identical or not, wall time, `peak L5` against B0.

## H3 on the command line

### H-a. Text to video, full size, first use
Wait until the Models page shows MiniMax H3 as downloaded.
```bash
run Ha ziv-video -m h3 --ratio 16:9 --size s --seed 7 --prompt "$FOX" -o $T/ha
probe "$(latest $T/ha)"
TE4=$(uv run python -c 'from zvisiongenerator.utils.paths import get_ziv_data_dir as d; print(d() / "models" / "h3@mlx-te4")')
du -sh "$TE4"; stat -f '%Sm %N' "$TE4"/* | sort | tail -1
uv run ziv model list | grep -c "mlx-te4"
```
- **Pass:** before loading it prints `MiniMax H3 at 768x448, 124 frames: estimated 23 min (excludes the one-time weight preparation).` The log then shows the one-time text-encoder conversion to `h3@mlx-te4`, then the port's own progress. The output is 768×448, **124 frames**, `24/1` fps, audio of **2 channels at 32000 Hz**. No crash, no heavy swapping. The last command prints `0`: the copy is not a listed model.
- **Record:** total wall time; conversion time (newest file time in `$TE4` minus the log's `start`); generation time against **23 min** (the estimate leaves out the conversion); `peak Ha` against **15.0 GiB**; `h3@mlx-te4` size (about 15 GB); swap.

### H-b. 4 steps, short
```bash
run Hb ziv-video -m h3 --size xs --frames 73 --steps 4 --seed 7 --prompt "$FOX" -o $T/hb
probe "$(latest $T/hb)"
```
- **Pass:** `MiniMax H3 at 576x320, 73 frames: estimated 6 min …`; output 576×320, 73 frames, with audio.
- **Record:** wall time (estimate 6 min), `peak Hb` (estimate 13.7 GiB), swap.

### H-c. Image to video
```bash
run Hc ziv-video -m h3 --size xs --frames 73 --image "$(img_start)" --seed 7 --prompt "$FOX" -o $T/hc
probe "$(latest $T/hc)"
ana frame "$(latest $T/hc)" 0 "$(img_start)"
```
- **Pass:** `estimated 9 min` (image to video holds a keyframe); the run completes (a crash right after the text encode would be an `mx.eval` error); 576×320, 73 frames, audio. Frame 0 has PSNR **at least 21 dB** and SSIM **at least 0.70**: H3 re-renders frame 0 from a slightly noised keyframe, so it is not pixel-exact (Windows/Linux measured 23.0 dB / 0.77 at 864×480).
- Image to video runs the staged path (keyframes, vision tower, prompt, denoise, decode), so the peak bar is **at most 15 GiB** (the 4-bit text encoder stays resident at about 13.2 GiB), not the 17.7 GiB of the load phase.
- **Record:** PSNR and SSIM, wall time (estimate 9 min), `peak Hc` against **15 GiB**, swap.

### H-d. Extend
```bash
run Hd ziv-video --extend "$(latest $T/hb)" --extend-seconds 2 --seed 7 --prompt "$EXT" -o $T/hd
probe "$(latest $T/hd)"
ana seam "$(latest $T/hb)" "$(latest $T/hd)"
ana audio "$(latest $T/hb)" "$(latest $T/hd)"
```
- **Pass:** the log has `Loading H3 video model: MiniMaxAI/MiniMax-H3@`. It warns `--extend-seconds 2 is 48 frames at 24 fps; using 51 frames (2.12 s).` It prints **`MiniMax H3 at 576x320, 56 frames: estimated 8 min …`**: on a Mac the window is the 51 new frames plus 1 carried frame, moved up to the next valid length (56), because the clip is rendered as image to video from the source's last frame. The output is 73 + 51 = **124 frames** at 576×320 with audio.
- There is no bar on the seam: an image-to-video join is expected to be less smooth than the Windows/Linux Extend. For comparison, Windows/Linux measured a median jump ratio of 1.12 (clips at 1.12, 1.91, 1.11), the seam PSNR at or near each clip's own 10th percentile, and audio dips of 5 to 10 dB with 0 to 4 of 15 windows 6 dB or more down.
- **Record:** the `ana seam` and `ana audio` lines, wall time (estimate 8 min), `peak Hd` (estimate 17.7 GiB), swap.

### H-e. Extend with a LoRA
On Windows/Linux this LoRA broke the Extend seam: the new part cut to a different shot. Same source, prompt and seed as H-d, so the two compare directly.
```bash
run He ziv-video --extend "$(latest $T/hb)" --extend-seconds 2 --seed 7 --lora h3-accepted:0.5 --prompt "$EXT" -o $T/he
probe "$(latest $T/he)"
ana seam "$(latest $T/hb)" "$(latest $T/he)"
ana audio "$(latest $T/hb)" "$(latest $T/he)"
```
- **Pass:** the same `56 frames: estimated 8 min` line as H-d (the Mac estimate ignores LoRAs); 124 frames at 576×320 with audio. No bar on the seam.
- **Record:** the seam and audio figures next to H-d's, wall time, `peak He` against 17.7 GiB (the gap is what a LoRA costs that the estimate leaves out). A seam PSNR far under the source's 10th percentile with a jump ratio many times H-d's means a cut; V judges it by eye.

### H-f. A LoRA the Mac port can't take
```bash
run Hf ziv-video -m h3 --size xs --frames 73 --lora $T/loras/h3-refused.safetensors:1.0 --seed 7 --prompt "$FOX" -o $T/hf
```
- **Pass:** after the `estimated 9 min` line it exits within about a minute, before the pipeline loads, with `MiniMax H3 on macOS takes LoRAs for transformer blocks and the token refiner only.` No file appears in `hf`.
- **Record:** the error text and how long it took.

## Web UI

Use the UI from S4. Its outputs go to the app's output folder, not the scratch folder; read times from the job card.

### W1. H3 controls (no generation)
Text to Video, model **MiniMax H3**, 16:9.
- **Pass:**
  - **Steps** offers only 4 and 8, default 8. **Duration (s)** defaults to 5.13 (124 frames). There are **no Upscale or Review controls** and no **Low memory** toggle; audio and LoRAs are offered.
  - The time line under Duration follows size, length and steps, and no memory note shows:

    | Setting | Time line |
    |---|---|
    | `s`, default length, Steps 8 | `About 23 min (estimated).` |
    | `s`, default length, Steps 4 | `About 14 min (estimated).` |
    | `xs`, Duration 3 (73 frames), Steps 8 | `About 9 min (estimated).` |

  - Add the accepted LoRA at the `xs`, 3 s setting: the line stays `About 9 min (estimated).` (the Mac estimate ignores LoRAs).
  - Switch to `ltx25-4`: **Duration (s)** shows **5** (121 frames) at every size.
- **Record:** any line or note that differs, and a screenshot of the H3 settings pane.

### W2. First and last frame
Image to Video, MiniMax H3, `xs`, Duration 3, Steps 8. **First frame** is the start image and **Last frame** (the slot under it; the **Last** chip on a phone) the end image, from `$T/img`. Type the fox prompt (`echo $FOX`): it doesn't match the lighthouse on purpose, so the run compares with H-c. Generate.
```bash
probe "$(ui_latest)"
ana frame "$(ui_latest)" 72 "$(img_end)"
```
- **Pass:** the time line reads `About 9 min (estimated).`; the clip is 576×320, 73 frames, with audio; the job card shows the **denoise steps** (0 to 7 of 8, at about 13, 25, 38 % and on), because the staged path reports each step.
- **Record:** job time, and the last-frame PSNR and SSIM.

### W3. Extend panel and the gallery
Gallery, open W2's clip, **Extend**. Don't generate.
- **Pass:**
  - The default new length is 51 frames, the total reads 5.17 s · 124 frames, and the time line reads `About 8 min (estimated).` with no memory note.
  - Set **Steps** to 4 in the settings and the line becomes `About 6 min (estimated).`
  - In the viewer there is **no Upscale button**, pressing `X` then `2` does nothing, and the `⋯` menu has no 2×. Check at desktop width and at phone width (a narrow window).
  - Repeat the viewer check on an LTX clip (L1 or W5): video 2× (SeedVR2) is Windows/Linux only, and an H3 clip alone cannot show that, since H3 has no upscale on a Mac either.
- **Record:** the three time lines, and anything that differs.

### W4. Cancel, and an image job afterwards
1. Text to Image, FLUX.2 Klein 4B, 16:9, default size, seed 7, prompt `A lighthouse on a rocky coast at sunset, photo`. Run it **twice**; the second run is the baseline.
2. Text to Video, MiniMax H3, `xs`, 3 s, Generate. After about 1 minute click **Cancel job** and confirm. Watch Activity Monitor, Memory, "Memory Used".
3. Run the image job again.
- **Pass:**
  - The H3 clip cannot stop mid-run on a Mac. It ends without error when the clip finishes (about 9 min at `xs`), and **nothing is saved**. Memory used falls back to about where it was before the job.
  - The image job after H3 has an s/step within about 15% of the baseline and shows no memory error. A much slower job means the MLX cache or memory limit was not restored.
  - Open the klein4b image in the gallery: its Upscale menu still offers 2× and 4× (don't run one: the first use downloads the upscale model).
- **Record:** time from Cancel to the job ending, memory before, during and after, and the baseline and after-H3 s/step and total time.

### W5. `ltx25-4` in the Workspace
Text to Video, LTX-2.5 4-bit, default size and length, the fox prompt, Generate.
- **Pass:** it completes with 121 frames; the job card shows progress phases; no memory note appears.
- **Record:** job time against L1.

## V. Visual checks (all at once, at the end)

Open each clip with sound on (headphones for the H3 stereo): the scratch ones with `open <file>`, the Web UI ones in the gallery. A contact sheet of the CLI clips:
```bash
mkdir -p $T/sheets
for f in $T/*/*.mp4; do n=$(basename "$(dirname "$f")"); ffmpeg -v error -y -i "$f" -vf "select=not(mod(n\,12)),scale=384:-1,tile=4x3" -frames:v 1 "$T/sheets/$n.png"; done
open $T/sheets
```

| ID | Clip | Look for |
|---|---|---|
| V1 | L1 | Coherent motion with no garbage frames (garbage would mean the wrong text encoder), plausible sound |
| V2 | L2 | Frame 0 is the lighthouse image, plausible motion |
| V3 | L4 against L1 | Sharper with no tiling seams; the review preview was a half-size version |
| V4 | L3 | No jump at the seam (5.0 s), audio continues without a click |
| V5 | Ha | Coherent motion, sound that fits the scene, real stereo |
| V6 | Hc, W2 | Frame 0 is clearly the start image: same composition and light, some fine detail made anew |
| V7 | W2 | The last frame resembles the night image |
| V8 | Hd, He | The fox continues across the seam (about 3.0 s) and the sound continues; for He, whether the new part continues the shot or cuts to a different one |
| V9 | L5 against B0 | LTX-2.3 looks as it did before |

## Results template

Header: chip / memory / macOS / free disk / commit and mlx versions (S3) / budget line (S3).

Swap Δ is the "used" value after minus before; output is from `probe`.

| ID | Result (pass / fail / flag) | Wall time | Peak (GiB) | Estimate (GiB / min) | Swap Δ | Output | Notes |
|---|---|---|---|---|---|---|---|
| S4 | | | — | 15 / 15.0 badges | — | — | dialog sizes, download times |
| T0 | | — | — | — | — | — | pass / skip / fail counts |
| B0 | | | | — | | | mlx version |
| L1 | | | | about 15 | | | s/step |
| L2 | | | | | | | PSNR (at least 28) |
| L3 | | | | | | | seam line |
| L4 | | | | | | | 11 steps? |
| L5 | | | | | | | identical to B0? |
| H-a | | | | 15.0 / 23 | | | conversion time, te4 size |
| H-b | | | | 13.7 / 6 | | | |
| H-c | | | | 17.7 / 9 | | | PSNR (at least 21), SSIM (at least 0.70) |
| H-d | | | | 17.7 / 8 | | | seam and audio figures |
| H-e | | | | 17.7 / 8 | | | seam cut? peak over H-d |
| H-f | | | — | — | — | — | message, time |
| W1 | | — | — | 23 / 14 / 9 | — | — | lines that differ |
| W2 | | | | / 9 | | | last-frame PSNR |
| W3 | | — | — | 8 / 6 | — | — | no video 2× |
| W4 | | | | — | — | — | cancel time, memory, s/step |
| W5 | | | | | | | |
| V1-V9 | | — | — | — | — | — | one line each |

## What the results feed

- The "hasn't been run on a Mac" lines in `docs/concepts/models.md` and `docs/concepts/memory.md` and the "not yet tested on a Mac" line in the CHANGELOG `[Unreleased]` entry: reword them once H-a to H-e pass.
- The `_MLX_*` constants in `utils/h3_memory.py`: refit them if a peak is more than 2 GiB over its estimate or a time is more than 1.5 times it.
- Whether the Mac H3 estimate should count LoRAs: H-e's peak against H-d's.

## If something fails: what to send back

For each failed or flagged step, send:
- the step ID and the exact command, or the UI steps;
- the log with your home folder masked: `sed "s#$HOME#~#g" $T/logs/<ID>.log` (the error text and traceback, the `/usr/bin/time -l` block with `peak memory footprint`, the start and end times);
- the swap before and after (`$T/logs/<ID>.swap-before` and `.swap-after`);
- `probe` of the output, if there is one;
- for Web UI steps, the server terminal's output around the failure (masked the same way) and a screenshot;
- the results-template header.

Before sending anything, check it holds nothing personal: no other folders, file names, host names or account details.

## Follow-ups

- **LTX-2.5 Extend peaks at 23.8 GiB**, about 1.2 GiB under the 25 GiB budget, and gained swap once with the Web UI server running. Watch it on 24 GB Macs.
- **H3 Mac time estimates run high.** An image-to-video clip at 768×448 estimates 14 min and measured 9.9; text to video estimates 23 and measured 15.1. Refit `_MLX_*` once more measurements exist.
- **The denoise-phase memory formula still assumes the resident 15 GiB.**
- **The keyframe phase peaks at 14.1 GiB**, near MLX's 16e9-byte limit; the headroom is about 0.8 GiB at `s`.
- **No Linux CI job.** CI runs `make check` on macOS only, so the Windows/Linux paths are checked on the reference card alone.
