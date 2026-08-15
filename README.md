&nbsp;
<p align="center">
    <img src="https://raw.githubusercontent.com/knuthelge/ZVisionGenerator/main/docs/assets/zvision-duo.png" alt="Z-Vision Generator Logo" width="300"/>
</p>
&nbsp;

# Z-Vision Generator

[![CI](https://github.com/knuthelge/ZVisionGenerator/actions/workflows/ci.yml/badge.svg)](https://github.com/knuthelge/ZVisionGenerator/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/z-vision-generator)](https://pypi.org/project/z-vision-generator/)
[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue)](LICENSE)

Local AI image and video generation — hassle-free and fun. No tangled node graphs, no cloud dependencies, just prompts and results. Runs on macOS (Apple Silicon / MLX) and on Windows and Linux with NVIDIA CUDA through diffusers.

## Features

- **Image generation** — text-to-image with Z-Image and FLUX.2 Klein (4B/9B) model families, plus Ideogram 4 (FP8) on macOS/MLX
- **Video generation** — text-to-video and image-to-video with platform-specific LTX aliases across macOS, Windows, and Linux
- **Cross-platform** — automatic backend selection: MLX on macOS, diffusers/CUDA on Windows and Linux for images, and the shared diffusers/CUDA LTX backend on Windows and Linux for video
- **Prompt system** — YAML prompt files with variables, structured prompts, snippets, and batch runs
- **Model store** — central `~/.ziv/` directory with bare-name resolution and HuggingFace fallback
- **LoRA support** — single or stacked, configurable weights, bare-name resolution
- **Image upscale** — generate small → Lanczos → img2img refine → CAS sharpen
- **Video upscale** — 2× spatial upscaling through the platform LTX backend when supported by the selected runtime
- **Reference images** — img2img steering from any starting image
- **Model variants** — image quantization across supported image backends, plus macOS MLX video Q4/Q8 aliases
- **Post-processing** — contrast, saturation, and CAS sharpening (image only)
- **Interactive controls** — skip, quit, pause, and repeat during batch runs (image only)

## Platform Support

| Platform | Image Generation | Video Generation |
|----------|------------------|------------------|
| macOS (Apple Silicon) | ✅ Z-Image / FLUX / Ideogram 4 via mflux/MLX | ✅ LTX via MLX aliases (`ltx-4`, `ltx-8`) |
| Windows (NVIDIA GPU) | ✅ Z-Image / FLUX via diffusers/CUDA | ✅ LTX via diffusers/CUDA alias (`ltx-2.3`) |
| Linux (NVIDIA GPU) | ✅ Z-Image / FLUX via diffusers/CUDA | ✅ LTX via diffusers/CUDA alias (`ltx-2.3`) |

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

> Video generation requires [ffmpeg](https://ffmpeg.org/). On Windows and Linux, image and video generation require an NVIDIA GPU with CUDA available to PyTorch.

> The packaged Windows/Linux `ltx-2.3` alias defaults to the configurable diffusers-converted repository `dg845/LTX-2.3-Diffusers`. This is the diffusers-compatible layout required by the Windows/Linux video backend, not an official Lightricks alias. Override it in `~/.ziv/config.yaml` if you want to point `ltx-2.3` at a different compatible diffusers repository.

> The macOS video aliases `ltx-4` and `ltx-8` are the shipped MLX Q4/Q8 presets. Windows and Linux use the diffusers-backed `ltx-2.3` alias instead, so the Q4/Q8 naming does not carry across platforms.

## Quick Start

```bash
# Generate an image (bare name from ~/.ziv/models/)
ziv-image -m my-model --prompt "a beautiful sunset"

# Generate from a HuggingFace model
ziv-image -m Tongyi-MAI/Z-Image-Turbo --prompt "a cat in a garden"

# Batch run from a prompts file
ziv-image -m my-model -p prompts.yaml -r 3

# Generate a video on macOS
ziv-video -m ltx-4 --prompt "A cat walking through a garden"

# Generate a video on Windows or Linux
ziv-video -m ltx-2.3 --prompt "A cat walking through a garden"

# Image-to-video on macOS
ziv-video -m ltx-4 --image photo.jpg --prompt "Camera zooms in slowly"

# Image-to-video on Windows or Linux
ziv-video -m ltx-2.3 --image photo.jpg --prompt "Camera zooms in slowly"

# Launch the Web UI
ziv ui

# Show command help and available subcommands
ziv
```

The Web UI is an explicit local launcher: use `ziv ui` or `ziv-ui`. Running bare `ziv` prints command-discovery help without starting a browser or server. If a local environment is missing required Web UI packages, the launcher reports how to repair the base install instead of failing with an import traceback.

The packaged Web UI is served from local static files and does not fetch cloud fonts or other font assets at runtime. It uses local system font fallbacks when Inter or JetBrains Mono are not installed.

> **Tip:** `ziv image`, `ziv video`, and `ziv model` are also available as subcommands of the unified `ziv` parent command. Use `ziv -h` or `ziv --help` to print terminal help.

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
