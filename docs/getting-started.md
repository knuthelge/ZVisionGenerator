# Getting Started

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager (**required** — pip is not supported)
- **macOS:** Apple Silicon (M1+) — mflux requires MLX
- **Windows:** NVIDIA GPU, RTX 20-series (Turing) or newer including RTX 50-series, with NVIDIA driver 580 or newer
- **Linux:** NVIDIA GPU, RTX 20-series (Turing) or newer including RTX 50-series, with NVIDIA driver 580 or newer
- **Video:** ffmpeg

## Installation

```bash
# Install globally — provides `ziv`, `ziv-ui`, `ziv-image`, `ziv-video`, and `ziv-model`
uv tool install z-vision-generator

# Install globally from repository
uv tool install -e git+https://github.com/knuthelge/ZVisionGenerator.git

# Or, for development
git clone https://github.com/knuthelge/ZVisionGenerator && cd ZVisionGenerator
uv sync
```

> **uv is required.** This package cannot be installed with pip — some dependencies require uv-specific resolution that pip does not support.

> **Note:** Video generation requires ffmpeg. On Windows and Linux, image and video generation use diffusers/CUDA and fail fast when PyTorch cannot see an NVIDIA CUDA device. If the error mentions the driver, update the NVIDIA driver to 580 or newer.

## Quick Start

### Image Generation

```bash
# Local model (bare name resolved from ~/.ziv/models/)
ziv-image -m my-model --prompt "a beautiful sunset"

# HuggingFace model (downloaded automatically)
ziv-image -m Tongyi-MAI/Z-Image-Turbo --prompt "a cat"

# From a prompts file with multiple runs
ziv-image -m my-model -p prompts.yaml -r 3
```

### Video Generation

Use `MODEL` below as `ltx-4` or `ltx-8` (4-bit or 8-bit) on every platform.

```bash
# Text-to-video with LTX (4-bit)
ziv-video -m ltx-4 --prompt "A cat walking through a garden"

# Image-to-video with your platform alias
ziv-video -m MODEL --image photo.jpg --prompt "Camera slowly zooms in"

# Batch from prompts file with your platform alias
ziv-video -m MODEL -p prompts.yaml -r 3
```

### Choosing a Video Model

`ltx-4` needs the least memory; `ltx-8` is closer to the full model. On Windows and Linux, `ltx-4` needs about 20 GB of system memory and `ltx-8` about 27 GB, and the first run of each takes a while to download and prepare the model. See the [Video Guide](guides/video.md#ltx-constraints).

### Model & LoRA Management

```bash
# Convert a Z-Image checkpoint
ziv-model model -i checkpoint.safetensors --name my-model

# Import a local LoRA
ziv-model lora -i /path/to/style.safetensors --name my-style

# List installed assets
ziv-model list
```

### Web UI

```bash
# Start the Web UI
ziv ui

# From a repository checkout
uv run ziv ui --no-browser
```

By default, the Web UI listens on `http://127.0.0.1:8080/` and opens your browser automatically. If port `8080` is busy, Z-Vision Generator chooses the next available local port and prints the final address in the terminal. If a local environment is missing required Web UI packages, `ziv ui` and `ziv-ui` print repair guidance for the base install instead of failing with an import traceback.

While a generation is running, the Web UI keeps the active job attached to the current browser tab. Refreshing the tab reconnects to the running job when the server still has it, and stale completed or missing jobs are cleared automatically. Job controls are shown only when the running job supports them.

You can keep working while a job runs: change the settings and press **Add to queue** (the Generate button while jobs are active, or `⌘/Ctrl+↵`). Queued jobs run one after another, in order, and use the settings they had when you added them. They are listed under the job card as **Up next**, where each one can be removed or its settings copied back into the form with **Load settings**; **Clear queue** removes them all (after asking) and never stops the running job. If a job fails, the next one starts anyway. The queue is shared by every open tab and is kept in the server's memory, so stopping the server clears it. While jobs are running or queued, the manual **Enhance prompt** button is unavailable, because the enhancer would need the same memory; auto enhancement still runs inside each job.

The **Prompts** page builds prompt files visually: snippets, sets of prompts, and per-prompt enhancement. See [Prompt Builder](guides/prompts.md#prompt-builder-web-ui).

Workspace settings are remembered in the browser between visits and restarts, as long as the Web UI starts on the same port. Choosing a different model applies that model's defaults. LoRAs are remembered per model family: switching to a model of another family brings back the LoRAs you last used with that family (or none), and switching between models of the same family keeps them. **Reuse settings**, **Load settings** on a queued job and **Extend** use exactly the LoRAs the source was made with, so a source made without LoRAs clears them.

If the server can't start a job with the current settings, for example image-to-image without a reference image, the reason shows above the Generate button and in a notice. It goes away when you change a setting.

To run the same job from a terminal, press the terminal button in the *Settings* header. It copies a `ziv image` or `ziv video` command with the current model, prompt, LoRAs and settings, writing to the Web UI's output folder. The CLI has no inline negative prompt and runs every prompt in a prompts file rather than only the selected ones, so a notice says when the copied command differs from what Generate would run; an uploaded reference image is named by its file name, so replace it with the file's path.

Fields that take a path on the server's machine (prompt files, the output folder, checkpoints, LoRAs, reference images and source videos) have a folder button that opens a file browser inside the Web UI, so it works the same on every system and from a phone or another computer. It opens at the field's current path or at the matching folder (the output folder, `~/.ziv/models` or `~/.ziv/loras`), lists only the files that field accepts (**Show all** lists the rest), and keeps the last five choices under **Recent**. Click the path bar or press `/` to type a path; **Hidden files** shows names starting with a dot. Typing a path in the field itself suggests matching folders and files: `Tab` completes the first suggestion, the arrow keys and `Enter` pick another, and `↓` in an empty field suggests recent paths and the shortcut folders. The reference image also keeps an upload button for a picture on the device you're browsing from.

To give the preview more room, collapse the Compose and Settings sidebar with the **«** button next to *Compose*. It folds into a narrow strip with an expand button and a Generate button, and the choice is remembered in the browser. On phone-width screens the sidebar always shows above the preview.

While a job runs, its card shows the progress, the time left and a live preview of the current image. **Pause** and **Cancel** are at the top of the card; **Skip** and **Repeat** are next to the current image. Pause and Repeat wait until the current image is finished, while Skip and Cancel stop it at its next step. A video job offers Cancel only, plus Approve and Discard while it waits for a review (macOS); while its model loads, Cancel takes effect once loading finishes. If a job fails, the card shows why. Finished images appear in the history strip, newest first; scroll it to the end and older ones load in.

The workspace model picker and the Models page show whether each model is already downloaded; models that are not yet downloaded download on first use. They also show whether a downloaded model fits this machine's memory. On macOS this is compared with the GPU memory Apple recommends for the machine:

- **Fits**: the estimate is at most 1.1 times the recommendation.
- **Tight**: the estimate is at most 1.5 times the recommendation, which is MLX's default memory limit. The model runs, but macOS compresses or swaps other memory to make room, so the Mac can slow down while generating.
- **Too large**: the estimate is above MLX's default memory limit. Expect heavy swapping that can make the whole Mac unresponsive, or an out-of-memory error.

On Windows and Linux the badges compare an image model with the GPU's memory and the machine's system memory (see [Quantization](guides/image.md#quantization)):

- **Fits**: the model fits comfortably.
- **Tight**: the model runs, but slowly, because it does not fit in system memory or the GPU is nearly full.
- **Too large**: the model needs more GPU memory than the GPU has. Try a lower quant.

The workspace badge follows the selected quantize level, so you can see whether `q8` or `q4` brings a large model within reach. Hover the badge for the estimate. After each job the Web UI returns the model's memory to the system.

Ideogram 4 (`ideo`) is selectable in the workspace model dropdown. When it is selected, the workspace offers a structured-JSON-caption input and a first-step-sigma control, while reference-image (img2img) and upscale are unavailable, and width and height are constrained to the 256–2048 range in multiples of 16.

#### Keyboard shortcuts

Press `?` anywhere outside a text field to see every shortcut. `⌘` on macOS is `Ctrl` elsewhere.

Anywhere:

| Key | Action |
|-----|--------|
| `?` | Show all keyboard shortcuts |
| `G` then `W` / `P` / `G` / `M` / `C` | Go to Workspace / Prompts / Gallery / Models / Config |

Workspace:

| Key | Action |
|-----|--------|
| `⌘↵` | Generate |
| `⌘⇧↵` | Generate with a new seed (re-rolls a locked seed) |
| `⌘E` | Enhance the prompt |
| `Alt+1` / `Alt+2` | Focus the prompt / the settings (expands the sidebar) |
| `Alt+3` | Focus the newest history tile (expands the strip; `←` / `→` move between tiles) |
| `Alt+⇧+1` / `Alt+⇧+2` | Show or hide the sidebar (focuses the prompt / the settings when shown) |
| `Alt+⇧+3` | Show or hide the history strip (focuses the newest tile when shown) |
| `Esc` `Esc` | Stop the running job (press twice) |
| `P` | Pause or resume the running job |
| `N` | Skip to the next image; during a review, discard the small render |
| `R` | Repeat the current image |
| `Y` | Approve the small render for upscaling (during a review) |

Prompts:

| Key | Action |
|-----|--------|
| `⌘S` | Save the prompt file |
| `⌘Z` | Undo (outside text fields) |
| `↵` | Edit the focused prompt text or random choice |
| `$` | Suggest snippets while editing a prompt |
| `⌥↵` | Edit the `{a|b}` choice at the caret, make one from the selected words, or start a new one (while editing) |
| `Esc` / `⌘↵` | Finish editing a prompt |

Gallery:

| Key | Action |
|-----|--------|
| Arrow keys | Move between assets |
| `↵` | Open the viewer |
| `Space` / `X` | Select or deselect the focused asset |
| `⌘A` | Select all loaded assets |
| `Delete` / `Backspace` | Delete the selection, or the focused asset (asks first) |
| `Esc` | Clear the selection |

Full-screen asset viewer:

| Key | Action |
|-----|--------|
| `←` / `→` | Previous / next asset |
| `Home` / `End` | First / last asset |
| `R` | Reuse settings |
| `E` | Use as reference: load an image as the reference image, together with its prompt. On a video, Extend: open it in the Extend Video tab |
| `D` | Download |
| `C` | Copy the prompt |
| `Delete` / `Backspace` | Delete (asks for confirmation: `Enter` confirms, `Esc` cancels) |
| `Shift+Delete` | Delete without asking |
| `I` | Toggle the details panel |
| `?` | Show the viewer's shortcuts |
| `Esc` | Close the shortcut list, then the viewer |

## Model Store (`~/.ziv/`)

Z-Vision Generator uses a central data directory for models and LoRAs:

```
~/.ziv/
├── models/    # Diffusers-format model directories
└── loras/     # LoRA .safetensors files
```

**To add a model**, copy (or symlink) the model directory into `~/.ziv/models/`:

```bash
cp -r /path/to/my-model ~/.ziv/models/my-model
```

Then reference it by bare name:

```bash
ziv-image -m my-model --prompt "hello world"
```

### Resolution Order

When you pass `-m <name>` (both `ziv-image` and `ziv-video`):

1. **Path with `/` or `\`** → used as-is (local path)
2. **Bare name** → checks `~/.ziv/models/<name>/` → uses it if found
3. **Alias** → checks built-in and custom aliases (see [Image Guide](guides/image.md#model-aliases))
4. **Otherwise** → treated as a HuggingFace repo ID (downloaded on first use)

> **Note:** A local model directory at `~/.ziv/models/<name>/` takes priority over an alias with the same name.

LoRA generation inputs are local paths or bare names from `~/.ziv/loras/`. To use a HuggingFace LoRA, import it first with `ziv-model lora --hf ...`, then reference the local LoRA name or path during generation.

### Download speed

Model downloads from HuggingFace use Xet. Set `HF_XET_HIGH_PERFORMANCE=1` to let downloads try to saturate the network bandwidth and use all CPU cores. The variable is read when the program starts, so set it before launching `ziv`:

```bash
export HF_XET_HIGH_PERFORMANCE=1
ziv-image -m my-model --prompt "a landscape"
```

### `ZIV_DATA_DIR` Override

Set the `ZIV_DATA_DIR` environment variable to use a custom location instead of `~/.ziv/`:

```bash
export ZIV_DATA_DIR=/mnt/fast-ssd/ziv
ziv-image -m my-model --prompt "a landscape"
```

### Error Log

Each command records warnings and errors, with full tracebacks, in `~/.ziv/logs/<command>.log` (`ui.log`, `image.log`, `video.log` or `model.log`). The file is cleared every time the command starts, so it only covers the latest run, and stays empty when nothing goes wrong. If the logs folder cannot be written, the log goes to the system temp folder instead; `ziv ui` prints the file's location at startup. Set `ZIV_LOG=off` to turn the log off.

## Troubleshooting

### Linux: the terminal or editor closes while a model loads

On Linux distributions that run `systemd-oomd`, such as Ubuntu, loading a large model can close the terminal or editor that started the job, and the job with it, even when there is enough memory for the load to finish. `systemd-oomd` reacts to memory pressure (the time spent waiting for memory to be reclaimed), not to running out of memory. Loading 15–35 GB of weights on a machine with about 32 GB of RAM fills the page cache and can keep pressure above Ubuntu's limit of 50% for 20 seconds. oomd then stops the whole terminal or editor the job runs in. `journalctl -u systemd-oomd` lists each time it did.

To relax the limit for your user session, create `/etc/systemd/system/user@.service.d/99-oomd-relax.conf` with sudo:

```ini
[Service]
ManagedOOMMemoryPressureLimit=80%
ManagedOOMMemoryPressureDurationSec=60s
```

- **Apply:** run `sudo systemctl daemon-reload`, then log out and back in, or reboot.
- **Verify:** `systemctl show user@$(id -u).service -p ManagedOOMMemoryPressureLimit` prints `80%`, and `oomctl` lists the new limit.
- **Revert:** delete the file and repeat the reload and re-login.

80% for 60 seconds still stops runaway memory use, but lets typical model loads finish. The largest loads, such as Krea 2 without quantization, can still go over it; use `-q 8` or `-q 4` on these machines. ZVisionGenerator never changes this setting itself.

To protect your editor when oomd does act, start heavy jobs in their own unit, so oomd stops the job rather than the editor:

```bash
systemd-run --user --collect --same-dir -p OOMScoreAdjust=500 ziv-image -m my-model --prompt "a landscape"
```
