&nbsp;
<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/knuthelge/ZVisionGenerator/main/docs/assets/ziv-logo-dark.png"/>
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/knuthelge/ZVisionGenerator/main/docs/assets/ziv-logo-light.png"/>
        <img src="https://raw.githubusercontent.com/knuthelge/ZVisionGenerator/main/docs/assets/ziv-logo-light.png" alt="Z-Vision Generator logo with the Blob mascot" width="360"/>
    </picture>
</p>
&nbsp;

# Z-Vision Generator

[![CI](https://github.com/knuthelge/ZVisionGenerator/actions/workflows/ci.yml/badge.svg)](https://github.com/knuthelge/ZVisionGenerator/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/z-vision-generator)](https://pypi.org/project/z-vision-generator/)
[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue)](LICENSE)

Your own studio for AI images, video and audio. Write a prompt, press Generate, and the picture, clip or track is made on your own computer. No cloud, no node graphs.

## Install

Runs on a Mac with Apple silicon (macOS 14 or newer), or on a Windows or Linux PC with an NVIDIA RTX 20-series or newer card. Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv tool install z-vision-generator
ziv ui
```

On Windows, add PyTorch's CUDA index to the install command, because PyPI's PyTorch for Windows has no CUDA:

```bash
uv tool install z-vision-generator --index https://download.pytorch.org/whl/cu132 --index-strategy unsafe-best-match
```

Video and audio need [ffmpeg](https://ffmpeg.org/). The Web UI opens in your browser and offers to download an image model picked for your machine. Steps for each system: [Install](https://knuthelge.github.io/ZVisionGenerator/get-started/install/).

## Features

- **Images** from a prompt, with Z-Image, FLUX.2 Klein, Krea 2 Turbo, Ideogram 4 Instant and MOSS-Image.
- **Videos with sound** from a prompt or a starting image, with LTX-2.3, LTX-2.5 and MiniMax H3.
- **Audio**: songs with YuE2, speech with Qwen3-TTS, and sound effects with Stable Audio 3.
- **A prompt enhancer** that runs on your machine.
- **Upscaling**, **LoRAs** and **prompt files** with random choices and batch runs.
- **A Web UI** for your computer, phone or tablet, and a **CLI** for scripts and batches, sharing the same models, prompt files and outputs.

## Quick start

```bash
# Start the Web UI
ziv ui

# An image from the command line
ziv-image -m klein4b --prompt "A red fox sitting in fresh snow, early morning light"
```

## Documentation

**[knuthelge.github.io/ZVisionGenerator](https://knuthelge.github.io/ZVisionGenerator/)**: your first image, video and audio, guides for each task, every page of the app, and the command reference.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) and the [developer docs](https://knuthelge.github.io/ZVisionGenerator/developer/).

## License

This project is licensed under the [GNU Affero General Public License v3.0 or later](LICENSE).
