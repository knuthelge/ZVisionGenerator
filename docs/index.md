<p align="center">
    <img src="https://raw.githubusercontent.com/knuthelge/ZVisionGenerator/main/docs/assets/zvision-duo.png" alt="Z-Vision Generator Logo" width="300"/>
</p>
&nbsp;

# Z-Vision Generator

Local AI image and video generation — hassle-free and fun. No tangled node graphs, no cloud dependencies, just prompts and results. Runs on macOS (Apple Silicon / MLX) and on Windows and Linux with NVIDIA CUDA through diffusers.

Z-Vision Generator gives you a unified CLI for both image and video generation, with batch runs, prompt variables, LoRA support, upscaling, and more — all while abstracting away platform-specific details.

## Features

- **Image generation** — text-to-image with Z-Image and FLUX.2 Klein (4B/9B) model families, plus Ideogram 4 (FP8) on macOS/MLX
- **Video generation** — text-to-video and image-to-video with platform-specific LTX aliases, audio included by default
- **Cross-platform** — automatic backend selection: MLX on macOS, diffusers/CUDA on Windows and Linux for images, and the shared diffusers/CUDA LTX backend on Windows and Linux for video
- **Prompt system** — YAML prompt files with variables (`{red|blue|green}`), structured prompts, snippets, and batch runs
- **Model store** — central `~/.ziv/` directory with bare-name resolution, aliases, local paths, and supported HuggingFace model references
- **LoRA support** — single or stacked, configurable weights, bare-name resolution (image and video)
- **Image upscale** — generate small → Lanczos → img2img refine → CAS sharpen
- **Video upscale** — macOS MLX uses distilled two-stage 2× spatial upscaling; Windows/Linux diffusers uses latent upscaling when the model supports it and fails explicitly when it does not
- **Reference images** — img2img steering from any starting image
- **Model variants** — image quantization across supported image backends, plus macOS MLX video Q4/Q8 aliases
- **Post-processing** — contrast, saturation, and CAS sharpening (image only)
- **Interactive controls** — skip, quit, pause, and repeat during batch runs (image only)

## Platform Support

| Platform | Image Generation | Video Generation |
|----------|------------------|------------------|
| macOS (Apple Silicon) | ✅ Z-Image / FLUX / Ideogram 4 via mflux/MLX | ✅ LTX via MLX aliases (`ltx-4`, `ltx-8`) |
| Windows (NVIDIA GPU) | ✅ Z-Image / FLUX models via diffusers/CUDA | ✅ LTX via diffusers/CUDA alias (`ltx-2.3`) |
| Linux (NVIDIA GPU) | ✅ Z-Image / FLUX models via diffusers/CUDA | ✅ LTX via diffusers/CUDA alias (`ltx-2.3`) |

## Quick Links

- [Getting Started](getting-started.md) — installation, prerequisites, and first steps
- [Image Generation Guide](guides/image.md) — model aliases, sizes, LoRA, upscaling, post-processing
- [Video Generation Guide](guides/video.md) — text-to-video, image-to-video, upscale, audio
- [Prompts Guide](guides/prompts.md) — prompt files, variables, structured prompts, snippets
- [Model & LoRA Management](guides/model.md) — converting checkpoints, importing LoRAs
- [CLI Reference](reference/cli.md) — full argument tables for all commands
- [Development](development.md) — setup, testing, architecture
