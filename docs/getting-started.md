# Getting Started

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager (**required** — pip is not supported)
- **macOS:** Apple Silicon (M1+) — mflux requires MLX
- **Windows:** NVIDIA GPU with CUDA support
- **Linux:** NVIDIA GPU with CUDA support
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

> **Note:** Video generation requires ffmpeg. On Windows and Linux, image and video generation use diffusers/CUDA and fail fast when PyTorch cannot see an NVIDIA CUDA device.

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

Use `MODEL` below as follows: `ltx-4` or `ltx-8` on macOS, `ltx-2.3` on Windows and Linux.

```bash
# Text-to-video with LTX on macOS
ziv-video -m ltx-4 --prompt "A cat walking through a garden"

# Text-to-video with LTX on Windows or Linux
ziv-video -m ltx-2.3 --prompt "A cat walking through a garden"

# Image-to-video with your platform alias
ziv-video -m MODEL --image photo.jpg --prompt "Camera slowly zooms in"

# Batch from prompts file with your platform alias
ziv-video -m MODEL -p prompts.yaml -r 3
```

### Video Model Defaults And Overrides

- macOS exposes `ltx-4` and `ltx-8`, which resolve to the MLX LTX repositories and are the shipped Q4/Q8 presets.
- Windows and Linux expose `ltx-2.3`, which resolves to the configurable diffusers repository in `video_model_presets.ltx.diffusers.default_repo`.
- The packaged default is `dg845/LTX-2.3-Diffusers`, a diffusers-converted repository with the layout required by the Windows/Linux diffusers video backend. It is a configurable diffusers default, not a shipped Q4/Q8 alias.

To override the Windows/Linux default video repository, add a user config file at `~/.ziv/config.yaml`:

```yaml
video_model_presets:
  ltx:
    diffusers:
      default_repo: your-org/your-ltx-diffusers-repo
```

After the override, `ziv-video -m ltx-2.3 ...` and the Web UI both resolve to the new repository.

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

Workspace settings are remembered in the browser between visits and restarts, as long as the Web UI starts on the same port. Choosing a different model applies that model's defaults.

The workspace model picker and the Models page show whether each model is already downloaded; models that are not yet downloaded download on first use. On macOS they also show whether a downloaded model fits this Mac's memory, compared with the GPU memory Apple recommends for the machine:

- **Fits**: the estimate is at most 1.1 times the recommendation.
- **Tight**: the estimate is at most 1.5 times the recommendation, which is MLX's default memory limit. The model runs, but macOS compresses or swaps other memory to make room, so the Mac can slow down while generating.
- **Too large**: the estimate is above MLX's default memory limit. Expect heavy swapping that can make the whole Mac unresponsive, or an out-of-memory error.

The workspace badge follows the selected quantize level, so you can see whether `q8` or `q4` brings a large model within reach. Hover the badge for the estimate. After each job the Web UI returns the model's memory to the system.

Ideogram 4 (`ideo`) is selectable in the workspace model dropdown. When it is selected, the workspace offers a structured-JSON-caption input and a first-step-sigma control, while reference-image (img2img) and upscale are unavailable, and width and height are constrained to the 256–2048 range in multiples of 16.

#### Keyboard shortcuts

Press `?` anywhere outside a text field to see every shortcut. `⌘` on macOS is `Ctrl` elsewhere.

Anywhere:

| Key | Action |
|-----|--------|
| `?` | Show all keyboard shortcuts |
| `G` then `W` / `G` / `M` / `C` | Go to Workspace / Gallery / Models / Config |

Workspace:

| Key | Action |
|-----|--------|
| `⌘↵` | Generate |
| `⌘⇧↵` | Generate with a new seed (re-rolls a locked seed) |
| `⌘E` | Enhance the prompt |
| `Alt+1` / `Alt+2` | Focus the prompt / the settings |
| `Alt+3` | Focus the newest history tile (expands the strip; `←` / `→` move between tiles) |
| `Esc` | Stop the running job |
| `P` | Pause or resume the running job |
| `N` | Skip to the next image |
| `R` | Repeat the current image |

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
| `E` | Use as reference |
| `D` | Download |
| `C` | Copy the prompt |
| `Delete` / `Backspace` | Delete (asks for confirmation) |
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

### `ZIV_DATA_DIR` Override

Set the `ZIV_DATA_DIR` environment variable to use a custom location instead of `~/.ziv/`:

```bash
export ZIV_DATA_DIR=/mnt/fast-ssd/ziv
ziv-image -m my-model --prompt "a landscape"
```

### Error Log

Each command records warnings and errors, with full tracebacks, in `~/.ziv/logs/<command>.log` (`ui.log`, `image.log`, `video.log` or `model.log`). The file is cleared every time the command starts, so it only covers the latest run, and stays empty when nothing goes wrong. If the logs folder cannot be written, the log goes to the system temp folder instead; `ziv ui` prints the file's location at startup. Set `ZIV_LOG=off` to turn the log off.
