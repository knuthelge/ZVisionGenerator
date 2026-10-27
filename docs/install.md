# Install

## What you need

=== "macOS"

    - A Mac with Apple Silicon (M1 or newer) and macOS 14 Sonoma or newer.
    - 16 GB of memory runs Z-Image Turbo and FLUX.2 Klein quantized (`q8` or `q4`). 32 GB or more runs FLUX.2 Klein 9B and video at `ltx-8`. See [Memory](concepts/memory.md).
    - [ffmpeg](https://ffmpeg.org/) for video. `ziv-video` offers to install it with Homebrew or MacPorts if it's missing.

=== "Windows and Linux"

    - An NVIDIA GPU, RTX 20-series (Turing) or newer, including the RTX 50-series.
    - NVIDIA driver 580 or newer.
    - About 20 GB of system memory for video with `ltx-4`, 21 GB with `ltx25-4`, 27 GB with `ltx-8`, and 18 GB free for MiniMax H3. See [Memory](concepts/memory.md).
    - [ffmpeg](https://ffmpeg.org/) for video. `ziv-video` offers to install it with `winget` (Windows) or `apt`, `dnf` or `pacman` (Linux) if it's missing.

    There is no CPU fallback for generation: PyTorch must see a CUDA device.

On every platform:

- Python 3.14 or newer.
- [uv](https://docs.astral.sh/uv/). Every command here uses it, and a checkout needs it: uv's resolver is what picks the CUDA build of PyTorch there.
- Disk space for models: an image model is a 16–36 GB download, and a video model up to about 107 GB (MiniMax H3 on macOS). The [Models](concepts/models.md) page lists each one's size.

## Install

```bash
uv tool install z-vision-generator
```

This puts the commands on your path: `ziv`, `ziv-ui`, `ziv-image`, `ziv-video` and `ziv-model`.

!!! warning "Windows: add PyTorch's CUDA index"

    PyPI's PyTorch for Windows has no CUDA, so an install from PyPI alone ends with "CUDA is not available" even on an NVIDIA GPU. Point uv at PyTorch's CUDA index as well:

    ```bash
    uv tool install z-vision-generator --index https://download.pytorch.org/whl/cu132 --index-strategy unsafe-best-match
    ```

    The index strategy lets uv take PyTorch from that index and everything else from PyPI at its newest version.

    Linux gets a CUDA build from PyPI and needs no extra index. macOS has no CUDA and needs none.

??? note "Other ways to install"

    The latest code from the repository:

    ```bash
    uv tool install git+https://github.com/knuthelge/ZVisionGenerator.git
    ```

    A checkout for working on the code: see [Developer](developer/index.md).

    ```bash
    git clone https://github.com/knuthelge/ZVisionGenerator && cd ZVisionGenerator
    uv sync
    uv run ziv ui
    ```

## Start

```bash
ziv ui
```

The Web UI opens in your browser. The first time, it offers to download an image model picked for your machine. Continue with [Your first image](web-ui/first-image.md).

Prefer the terminal? Go to the [CLI](cli/index.md).

## Update

```bash
uv tool upgrade z-vision-generator
```

Your models, LoRAs, settings and outputs live in the [data folder](concepts/data-folder.md), not in the install, so they survive an update.
