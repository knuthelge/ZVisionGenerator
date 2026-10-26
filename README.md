&nbsp;
<p align="center">
    <img src="https://raw.githubusercontent.com/knuthelge/ZVisionGenerator/main/docs/assets/ziv-logo.png" alt="Z-Vision Generator logo with the Blob mascot" width="360"/>
</p>
&nbsp;

# Z-Vision Generator

[![CI](https://github.com/knuthelge/ZVisionGenerator/actions/workflows/ci.yml/badge.svg)](https://github.com/knuthelge/ZVisionGenerator/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/z-vision-generator)](https://pypi.org/project/z-vision-generator/)
[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue)](LICENSE)

Local AI image and video generation — hassle-free and fun. No tangled node graphs, no cloud dependencies, just prompts and results. Runs on macOS (Apple Silicon / MLX) and on Windows and Linux with NVIDIA CUDA through diffusers.

## Features

- **Images** from a prompt, with Z-Image, FLUX.2 Klein, Krea 2 Turbo, Ideogram 4 Instant and, on Windows and Linux, MOSS-Image.
- **Videos with sound** from a prompt or a starting image, with LTX-2.3, LTX-2.5 and MiniMax H3. End on a chosen last frame with H3, and extend a clip to make it longer.
- **A prompt enhancer** that runs on your machine and turns a short idea into a detailed prompt.
- **Upscaling** of new and existing images, of video as it's made, and of existing videos on Windows and Linux.
- **Prompt files** with random choices, reusable snippets and batch runs.
- **LoRAs**, stacked and weighted, for images and video.
- **A Web UI** for your computer, phone or tablet, and a **CLI** for scripts and batches, sharing the same models, prompt files and outputs.

| Platform | Images | Video |
|---|---|---|
| macOS (Apple Silicon) | ✅ via MLX | ✅ via MLX |
| Windows (NVIDIA GPU) | ✅ via diffusers/CUDA | ✅ via diffusers/CUDA |
| Linux (NVIDIA GPU) | ✅ via diffusers/CUDA | ✅ via diffusers/CUDA |

## Install

Requires Python 3.14+, [uv](https://docs.astral.sh/uv/), and [ffmpeg](https://ffmpeg.org/) for video. On Windows and Linux: an RTX 20-series or newer NVIDIA GPU with driver 580 or newer.

```bash
uv tool install z-vision-generator
ziv ui
```

On Windows, add PyTorch's CUDA index to the install command; PyPI's PyTorch for Windows has no CUDA. See [Install](https://knuthelge.github.io/ZVisionGenerator/install/) for the command and the details.

The Web UI opens in your browser and offers to download an image model picked for your machine.

## Quick start

```bash
# Start the Web UI
ziv ui

# An image from the command line
ziv-image -m klein4b --prompt "A red fox sitting in fresh snow, early morning light"

# A batch from a prompt file, three runs each
ziv-image -m klein9b -p prompts.yaml -r 3

# A video with sound, and one from an image
ziv-video -m ltx-4 --prompt "A cat walking through a garden, birds singing"
ziv-video -m ltx-4 --image photo.jpg --prompt "The camera slowly zooms in"
```

## Documentation

**[knuthelge.github.io/ZVisionGenerator](https://knuthelge.github.io/ZVisionGenerator/)**

- [Web UI](https://knuthelge.github.io/ZVisionGenerator/web-ui/): your first image and video, and every page of the app
- [CLI](https://knuthelge.github.io/ZVisionGenerator/cli/): how-tos and the full command reference
- [Concepts](https://knuthelge.github.io/ZVisionGenerator/concepts/): models, prompts, the enhancer, upscaling, memory and configuration
- [Developer](https://knuthelge.github.io/ZVisionGenerator/developer/): setting up, conventions and architecture

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) and the [developer docs](https://knuthelge.github.io/ZVisionGenerator/developer/).

## License

This project is licensed under the [GNU Affero General Public License v3.0 or later](LICENSE).
