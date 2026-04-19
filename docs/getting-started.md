# Getting Started

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager (**required** — pip is not supported)
- **macOS:** Apple Silicon (M1+) — mflux requires MLX
- **Windows:** NVIDIA GPU with CUDA support
- **Video:** [ffmpeg](https://ffmpeg.org/) — install via Homebrew on macOS, winget (`winget install Gyan.FFmpeg`), Chocolatey, or Scoop on Windows

## Installation

```bash
# Install globally — provides `ziv` (unified), `ziv-image`, `ziv-video`, and `ziv-model` commands
uv tool install z-vision-generator

# Install globally from repository
uv tool install -e git+https://github.com/knuthelge/ZVisionGenerator.git

# Or, for development
git clone https://github.com/knuthelge/ZVisionGenerator && cd ZVisionGenerator
uv sync
```

> **uv is required.** This package cannot be installed with pip — some dependencies require uv-specific resolution that pip does not support.

> **Note:** Video generation requires ffmpeg. On macOS, `ziv-video` will offer to install it via Homebrew on first run. On Windows, install ffmpeg manually (e.g. via [winget](https://learn.microsoft.com/en-us/windows/package-manager/winget/) (`winget install Gyan.FFmpeg`), [Chocolatey](https://chocolatey.org/), or [Scoop](https://scoop.sh/)) and ensure it is on your `PATH`.

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

Video generation works on both macOS (MLX) and Windows (CUDA). The `ltx-8` alias automatically resolves to the correct model for your platform. The `ltx-4` alias is currently macOS-only (4-bit will come to Windows when Lightricks releases the distilled checkpoint).

```bash
# Text-to-video
ziv-video -m ltx-8 --prompt "A cat walking through a garden"

# Image-to-video
ziv-video -m ltx-8 --image photo.jpg --prompt "Camera slowly zooms in"

# Batch from prompts file
ziv-video -m ltx-8 -p prompts.yaml -r 3
```

### Model & LoRA Management

```bash
# Convert a Z-Image checkpoint
ziv-model model -i checkpoint.safetensors --name my-model

# Import a local LoRA
ziv-model lora -i /path/to/style.safetensors --name my-style

# List installed assets
ziv-model list
```

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

### `ZIV_DATA_DIR` Override

Set the `ZIV_DATA_DIR` environment variable to use a custom location instead of `~/.ziv/`:

```bash
export ZIV_DATA_DIR=/mnt/fast-ssd/ziv
ziv-image -m my-model --prompt "a landscape"
```
