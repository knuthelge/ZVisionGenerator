# Getting Started

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager
- **macOS:** Apple Silicon (M1+) — mflux requires MLX
- **Windows:** NVIDIA GPU with CUDA support
- **Video:** ffmpeg (auto-offered on first run via Homebrew)

## Installation

```bash
# Install globally as `ziv` command from PyPI
uv tool install z-vision-generator

# Install globally from repository
uv tool install -e git+https://github.com/knuthelge/ZVisionGenerator.git

# Or, for development
git clone https://github.com/knuthelge/ZVisionGenerator && cd ZVisionGenerator
uv sync
```

> **Note:** Video generation requires ffmpeg. If missing, `ziv-video` will offer to install it via Homebrew on first run.

## Quick Start

### Image Generation

```bash
# Local model (bare name resolved from ~/.ziv/models/)
ziv -m my-model --prompt "a beautiful sunset"

# HuggingFace model (downloaded automatically)
ziv -m Tongyi-MAI/Z-Image-Turbo --prompt "a cat"

# From a prompts file with multiple runs
ziv -m my-model -p prompts.yaml -r 3
```

### Video Generation

```bash
# Text-to-video with LTX
ziv-video -m dgrauet/ltx-2.3-mlx-q4 --prompt "A cat walking through a garden"

# Image-to-video with LTX
ziv-video -m dgrauet/ltx-2.3-mlx-q4 --image photo.jpg --prompt "Camera slowly zooms in"

# Batch from prompts file
ziv-video -m dgrauet/ltx-2.3-mlx-q4 -p prompts.yaml -r 3
```

### Model & LoRA Management

```bash
# Convert a Z-Image checkpoint
ziv-convert model -i checkpoint.safetensors --name my-model

# Import a local LoRA
ziv-convert lora -i /path/to/style.safetensors --name my-style

# List installed assets
ziv-convert list
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
ziv -m my-model --prompt "hello world"
```

### Resolution Order

When you pass `-m <name>` (both `ziv` and `ziv-video`):

1. **Path with `/` or `\`** → used as-is (local path)
2. **Bare name** → checks `~/.ziv/models/<name>/` → uses it if found
3. **Alias** → checks built-in and custom aliases (see [Image Guide](guides/image.md#model-aliases))
4. **Otherwise** → treated as a HuggingFace repo ID (downloaded on first use)

> **Note:** A local model directory at `~/.ziv/models/<name>/` takes priority over an alias with the same name.

### `ZIV_DATA_DIR` Override

Set the `ZIV_DATA_DIR` environment variable to use a custom location instead of `~/.ziv/`:

```bash
export ZIV_DATA_DIR=/mnt/fast-ssd/ziv
ziv -m my-model --prompt "a landscape"
```
