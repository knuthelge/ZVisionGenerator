<p align="center">
    <img src="https://raw.githubusercontent.com/knuthelge/ZVisionGenerator/main/docs/assets/zvision-duo.png" alt="Z-Vision Generator Logo" width="300"/>
</p>
&nbsp;

# Z-Vision Generator

Local AI image and video generation — hassle-free and fun. No tangled node graphs, no cloud dependencies, just prompts and results. Runs on macOS (Apple Silicon / MLX) and Windows (NVIDIA / CUDA), tuned for an M-series Mac with 32 GB unified memory and an NVIDIA RTX 3080.

Z-Vision Generator gives you a unified CLI for both image and video generation, with batch runs, prompt variables, LoRA support, upscaling, and more — all while abstracting away platform-specific details.

## Features

- **Image generation** — text-to-image with Z-Image and FLUX.2 Klein (4B/9B) model families
- **Video generation** — text-to-video and image-to-video with LTX-2.3 (macOS), audio included by default
- **Cross-platform** — automatic backend selection: MLX on macOS, CUDA on Windows
- **Prompt system** — YAML prompt files with variables (`{red|blue|green}`), structured prompts, snippets, and batch runs
- **Model store** — central `~/.ziv/` directory with bare-name resolution and HuggingFace fallback
- **LoRA support** — single or stacked, configurable weights, bare-name resolution (image and video)
- **Image upscale** — generate small → Lanczos → img2img refine → CAS sharpen
- **Video upscale** — distilled-only two-stage 2× spatial upscaling
- **Reference images** — img2img steering from any starting image
- **Quantization** — 4-bit and 8-bit on both platforms
- **Post-processing** — contrast, saturation, and CAS sharpening (image only)
- **Interactive controls** — skip, quit, pause, and repeat during batch runs (image only)

## Platform Support

| Platform | Image Generation | Video Generation |
|----------|------------------|------------------|
| macOS (Apple Silicon) | ✅ Z-Image / FLUX models via mflux/MLX | ✅ LTX-2.3 via MLX |
| Windows (NVIDIA GPU) | ✅ Z-Image / FLUX models via diffusers/CUDA | ❌ Not supported |

## Quick Links

- [Getting Started](getting-started.md) — installation, prerequisites, and first steps
- [Image Generation Guide](guides/image.md) — model aliases, sizes, LoRA, upscaling, post-processing
- [Video Generation Guide](guides/video.md) — text-to-video, image-to-video, upscale, audio
- [Prompts Guide](guides/prompts.md) — prompt files, variables, structured prompts, snippets
- [Model & LoRA Management](guides/model.md) — converting checkpoints, importing LoRAs
- [CLI Reference](reference/cli.md) — full argument tables for all commands
- [Development](development.md) — setup, testing, architecture
