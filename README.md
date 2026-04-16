# Z-Vision Generator

[![CI](https://github.com/knuthelge/ZVisionGenerator/actions/workflows/ci.yml/badge.svg)](https://github.com/knuthelge/ZVisionGenerator/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/z-vision-generator)](https://pypi.org/project/z-vision-generator/)
[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue)](LICENSE)

<p align="center">
    <img src="https://raw.githubusercontent.com/knuthelge/ZVisionGenerator/main/docs/assets/zvision-duo.png" alt="Z-Vision Generator Logo" width="500"/>
</p>

Local AI image and video generation — hassle-free and fun. No tangled node graphs, no cloud dependencies, just prompts and results. Runs on macOS (Apple Silicon / MLX) and Windows (NVIDIA / CUDA), tuned for an M-series Mac with 32 GB unified memory and an NVIDIA RTX 3080.

## Features

- **Image generation** — text-to-image with Z-Image and FLUX.2 Klein (4B/9B) model families
- **Video generation** — text-to-video and image-to-video with LTX-2.3 (macOS)
- **Cross-platform** — automatic backend selection: MLX on macOS, CUDA on Windows
- **Prompt system** — YAML prompt files with variables, structured prompts, snippets, and batch runs
- **Model store** — central `~/.ziv/` directory with bare-name resolution and HuggingFace fallback
- **LoRA support** — single or stacked, configurable weights, bare-name resolution
- **Image upscale** — generate small → Lanczos → img2img refine → CAS sharpen
- **Video upscale** — distilled-only two-stage 2× spatial upscaling
- **Reference images** — img2img steering from any starting image
- **Quantization** — 4-bit and 8-bit on both platforms
- **Post-processing** — contrast, saturation, and CAS sharpening (image only)
- **Interactive controls** — skip, quit, pause, and repeat during batch runs (image only)

## Platform Support

| Platform | Image Generation | Video Generation |
|----------|------------------|------------------|
| macOS (Apple Silicon) | ✅ Z-Image / FLUX via mflux/MLX | ✅ LTX-2.3 via MLX |
| Windows (NVIDIA GPU) | ✅ Z-Image / FLUX via diffusers/CUDA | ❌ Not supported |

## Installation

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

> **uv is required.** This package cannot be installed with pip — some dependencies require uv-specific resolution that pip does not support. All commands below use uv.

```bash
# Install globally from PyPI
uv tool install z-vision-generator

# Install globally from repository
uv tool install -e git+https://github.com/knuthelge/ZVisionGenerator.git

# Development setup
git clone https://github.com/knuthelge/ZVisionGenerator && cd ZVisionGenerator
uv sync
```

> Video generation requires [ffmpeg](https://ffmpeg.org/). On macOS, `ziv-video` offers to install it via Homebrew on first run.

## Quick Start

```bash
# Generate an image (bare name from ~/.ziv/models/)
ziv-image -m my-model --prompt "a beautiful sunset"

# Generate from a HuggingFace model
ziv-image -m Tongyi-MAI/Z-Image-Turbo --prompt "a cat in a garden"

# Batch run from a prompts file
ziv-image -m my-model -p prompts.yaml -r 3

# Generate a video
ziv-video -m dgrauet/ltx-2.3-mlx-q4 --prompt "A cat walking through a garden"

# Image-to-video
ziv-video -m dgrauet/ltx-2.3-mlx-q4 --image photo.jpg --prompt "Camera zooms in slowly"
```

> **Tip:** `ziv image`, `ziv video`, and `ziv model` are also available as subcommands of the unified `ziv` parent command.

## Documentation

Full documentation is available at **[knuthelge.github.io/ZVisionGenerator](https://knuthelge.github.io/ZVisionGenerator/)**.

- [Getting Started](https://knuthelge.github.io/ZVisionGenerator/getting-started/) — installation, model store, quick start
- [Image Guide](https://knuthelge.github.io/ZVisionGenerator/guides/image/) — aliases, sizes, reference images, LoRA, upscaling, quantization
- [Video Guide](https://knuthelge.github.io/ZVisionGenerator/guides/video/) — T2V, I2V, upscale, audio, LoRA, constraints
- [Prompts Guide](https://knuthelge.github.io/ZVisionGenerator/guides/prompts/) — prompt files, variables, structured prompts, snippets
- [Model & LoRA Guide](https://knuthelge.github.io/ZVisionGenerator/guides/model/) — checkpoint conversion, LoRA import, asset listing
- [CLI Reference](https://knuthelge.github.io/ZVisionGenerator/reference/cli/) — full argument tables for all commands
- [Development](https://knuthelge.github.io/ZVisionGenerator/development/) — setup, testing, architecture

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the [GNU Affero General Public License v3.0 or later](LICENSE).
