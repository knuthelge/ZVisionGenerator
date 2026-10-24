# Mac test plan: LTX-2.5 and MiniMax H3 (32 GB Apple Silicon)

## Summary

1. **What's new:** LTX-2.5 (`ltx25-4`, "LTX-2.5 4-bit") and MiniMax H3 (`h3`, "MiniMax H3") now run on macOS. Neither has run on Apple hardware. This plan checks that they work, fit in 32 GB and land near their estimates, and that LTX-2.3 (`ltx-4`) still runs.
2. **Time:** about 3 hours (the table below), of which about 2 hours 15 minutes is unattended generation. The 138 GB of downloads overlap setup and T0; on a slow connection L1 and H-a wait for them, and that wait is not in the 3 hours.
3. **Disk:** about 170 GB of new files, so keep **at least 180 GB free**: LTX-2.5 30.5 GB, H3 107.2 GB (INT8 transformer 37.8, FL2VA components 68.0, turbo LoRA 1.4), the one-time `h3@mlx-te4` text-encoder copy about 15 GB, `klein4b` 16 GB if you don't have it, the two test LoRAs 1.1 GB, outputs about 2 GB.
4. **Order:** S1-S2 (facts and helpers), B0 on the branch you have now, S3-S5 (switch to the new branch, start both downloads with LTX-2.5 first, make the test inputs), T0 while the downloads run, then L1-L5 (LTX), H-a to H-f (H3 on the command line), W1-W5 (Web UI), and V (all the looking, at the end).
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

## How to use this plan

- Work top to bottom. Each step has a command or UI steps, **Pass** (what to look for) and **Record** (what to write in the results table).
- Run commands from the repo root in a terminal that has run `source ~/ziv-mac-test/env.sh`. That file sets up:
  - `run <ID> ziv-video|ziv-image …` runs the command under `/usr/bin/time -l`, logs it to `~/ziv-mac-test/logs/<ID>.log` with start and end times and the swap before and after, and prints the wall time and peak memory. Never put `uv run` in front of `/usr/bin/time -l`: that measures `uv`.
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
Run this from your checkout, still on the branch you normally use.
```bash
mkdir -p ~/ziv-mac-test/logs
cat > ~/ziv-mac-test/env.sh <<'EOF'
export T=~/ziv-mac-test
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
echo "export ZIV_REPO=\"$PWD\"" >> ~/ziv-mac-test/env.sh
cat > ~/ziv-mac-test/ana.py <<'PY'
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
source ~/ziv-mac-test/env.sh
```
- **Pass:** `type run` and `type ana` show the functions. `ana.py` only reads clips and images. `frame` gives RGB PSNR and luma SSIM of one frame against the image (centre-fitted to the clip); `seam` gives the jump ratio (the seam's mean frame change over the source's median) and the seam's PSNR against the source's own 10th percentile and median; `audio` gives the level in the second before the seam and the lowest 20 ms in the 300 ms after it.

### B0. (Optional, 6 min) The LTX-2.3 baseline on the old branch
Do this **before** S3, on the branch you have now, so L5 can compare against the code before the update. It needs `ltx-4` downloaded.
```bash
cd "$ZIV_REPO" && uv run python -c "import mlx.core as mx; print('mlx', mx.__version__)"
run B0 ziv-video -m ltx-4 --size s --frames 121 --prompt "$FOX" --seed 42 -o $T/b0
ffmpeg -v error -i "$(latest $T/b0)" -map 0:v -f framemd5 - | grep -v '^#' > $T/logs/b0.framemd5
```
- **Pass:** the clip is 121 frames. **Record:** wall time, `peak B0`, the mlx version. The old and new badge figures are not compared: the old badge was not recorded.

### S3. Get the branch onto the Mac and install
The branch is not pushed, so there is no `git fetch origin`. Bundle it on the machine that has it, after its history is final:
```bash
# on the machine with the branch
git bundle create ziv-feat.bundle feat/ltx-2.5      # the whole branch, about 13 MB; no prerequisites, so it fetches into any checkout
# copy ziv-feat.bundle to the Mac (AirDrop, scp, a USB stick), then on the Mac:
cd "$ZIV_REPO" && git bundle verify ~/ziv-feat.bundle
git fetch ~/ziv-feat.bundle feat/ltx-2.5:feat/ltx-2.5 && git switch feat/ltx-2.5
git log --oneline -1
make install
uv run python -c "import mlx.core as mx, mlx_vlm; print('mlx', mx.__version__, 'mlx-vlm', mlx_vlm.__version__)"
uv run python -c "from zvisiongenerator.backends import accelerator_memory_budget_or_none as b; x = b(); print(round(x.gpu_bytes / 2**30, 1), 'GiB budget; fits up to', round(x.gpu_bytes * 1.1 / 2**30, 1), 'GiB, tight up to', round(x.gpu_bytes * 1.5 / 2**30, 1), 'GiB')"
```
- Once the branch is pushed, `git fetch origin && git switch feat/ltx-2.5` does the same. A shorter bundle (`main..feat/ltx-2.5`) fetches only if your Mac's `main` already holds the branch's base commit.
- **Pass:** install finishes; mlx 0.32.3 and mlx-vlm 0.6.10 (the lock); the budget is about 21 GiB.
- **Record:** the commit hash, the mlx versions, the budget line.

### S4. Start the Web UI and both downloads
In a second terminal, and leave it running all session:
```bash
cd "$ZIV_REPO" && uv run ziv ui --host local
```
Models page: click **Download** on **LTX-2.5 4-bit** first, then on **MiniMax H3**. Confirm **FLUX.2 Klein 4B** is downloaded, or add its dialog size to the disk budget (16 GB).
- **Pass:** the LTX-2.5 dialog lists **one part**, `MarceloFSO/ltx-2.5-mlx-q4-fast`, about 30.5 GB, with no Gemma 3 part. The H3 dialog lists **three parts**: `water1234/MiniMax-H3-MLX-Argus-Calibrated-INT8` about 37.8 GB, `MiniMaxAI/MiniMax-H3` (FL2VA components) about 68.0 GB, `lightx2v/Minimax-h3-Turbo` about 1.4 GB; about 107.2 GB in all, no text-encoder or bf16 part. A download interrupted by sleep or network continues, it does not restart.
- When each finishes its row shows downloaded with a memory badge: **LTX-2.5 about 15 GB**, **MiniMax H3 about 15.0 GB** ("Estimated for a 5 s clip at 768x448 from the text. The Workspace checks the size and length you choose.").
- **Record:** the dialog sizes, each download's time, both badge figures, any interruption.

### S5. Test inputs (made by the app; never use your own images)
```bash
source ~/ziv-mac-test/env.sh && cd "$ZIV_REPO"
uv run ziv-image -m klein4b --prompt "A lighthouse on a rocky coast at sunset, photo" --ratio 16:9 --seed 7 -o $T/img/start
uv run ziv-image -m klein4b --prompt "The same lighthouse at night, beam lit, photo" --ratio 16:9 --seed 8 -o $T/img/end
img_start; img_end
uv run hf download Imagineer13/H3-Anti-Burn-LoRA h3_antiburn_universal_v1.safetensors --revision b98eb1d3f5574d1b8f57d6b0e720d2f047cfadb3 --local-dir $T/loras
uv run hf download larryvrh/MiniMax-H3-Turbo-Lora minimax_h3_turbo_v4_step600_ema.safetensors --revision 43a74557ac3f6539db8e0f2a959d03feb7a81480 --local-dir $T/loras
uv run ziv model lora -i $T/loras/h3_antiburn_universal_v1.safetensors
```
- **Pass:** two `.png` paths print; both LoRA files exist (0.31 GB and 0.78 GB); the import prints `LoRA imported: …`.
- The anti-burn LoRA is now available as `h3_antiburn_universal_v1` (H-e, W1). The turbo v4 LoRA is only for H-f, which expects it to be refused; it stays in the scratch folder.

### T0. Tests (while the downloads run)
```bash
uv run pytest tests/test_ltx_mlx_contract.py tests/test_h3_mlx_contract.py -q -rs 2>&1 | tee $T/logs/T0.log
make check 2>&1 | tee $T/logs/T0-check.log
```
- **Pass:** `make check` passes, and the two contract tests have **no skips**: on a Mac their MLX import checks must run.
- **Record:** pass, skip and fail counts, and every skip reason.

## LTX: `ltx25-4`, and `ltx-4` against B0

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
- **Pass:** the half-size preview opens and progress is one run of **11 steps** (8 + 3); after `y` the clip is 768×448, 121 frames.
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
- **Record:** PSNR and SSIM, wall time (estimate 9 min), `peak Hc` against **17.7 GiB** (the load phase), swap.

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
On Windows/Linux the anti-burn LoRA broke the Extend seam: the new part cut to a different shot. Same source, prompt and seed as H-d, so the two compare directly.
```bash
run He ziv-video --extend "$(latest $T/hb)" --extend-seconds 2 --seed 7 --lora h3_antiburn_universal_v1:0.5 --prompt "$EXT" -o $T/he
probe "$(latest $T/he)"
ana seam "$(latest $T/hb)" "$(latest $T/he)"
ana audio "$(latest $T/hb)" "$(latest $T/he)"
```
- **Pass:** the same `56 frames: estimated 8 min` line as H-d (the Mac estimate ignores LoRAs); 124 frames at 576×320 with audio. No bar on the seam.
- **Record:** the seam and audio figures next to H-d's, wall time, `peak He` against 17.7 GiB (the gap is what a LoRA costs that the estimate leaves out). A seam PSNR far under the source's 10th percentile with a jump ratio many times H-d's means a cut; V judges it by eye.

### H-f. A LoRA the Mac port can't take
```bash
run Hf ziv-video -m h3 --size xs --frames 73 --lora $T/loras/minimax_h3_turbo_v4_step600_ema.safetensors:1.0 --seed 7 --prompt "$FOX" -o $T/hf
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

  - Add the anti-burn LoRA at the `xs`, 3 s setting: the line stays `About 9 min (estimated).` (the Mac estimate ignores LoRAs).
  - Switch to `ltx25-4`: **Duration (s)** shows **5** (121 frames) at every size.
- **Record:** any line or note that differs, and a screenshot of the H3 settings pane.

### W2. First and last frame
Image to Video, MiniMax H3, `xs`, Duration 3, Steps 8. **First frame** is the start image and **Last frame** (the slot under it; the **Last** chip on a phone) the end image, from `~/ziv-mac-test/img`. Type the fox prompt (`echo $FOX`): it doesn't match the lighthouse on purpose, so the run compares with H-c. Generate.
```bash
probe "$(ui_latest)"
ana frame "$(ui_latest)" 72 "$(img_end)"
```
- **Pass:** the time line reads `About 9 min (estimated).`; the clip is 576×320, 73 frames, with audio; the job card shows a **single step** (0 to done), because the Mac port reports only its start and end, which is not a hang.
- **Record:** job time, and the last-frame PSNR and SSIM.

### W3. Extend panel and the gallery
Gallery, open W2's clip, **Extend**. Don't generate.
- **Pass:**
  - The default new length is 51 frames, the total reads 5.17 s · 124 frames, and the time line reads `About 8 min (estimated).` with no memory note.
  - Set **Steps** to 4 in the settings and the line becomes `About 6 min (estimated).`
  - In the viewer there is **no Upscale button**, pressing `X` then `2` does nothing, and the `⋯` menu has no 2×. Check at desktop width and at phone width (a narrow window).
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

- The "not run on a Mac" lines in `docs/guides/video.md` and the "not yet tested on a Mac" line in the CHANGELOG `[Unreleased]` entry: reword them once H-a to H-e pass.
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
