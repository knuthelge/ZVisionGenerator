# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Packaged local Web UI with workspace, gallery, config, and model views
- FastAPI-backed local job APIs, status streaming, and background web runner support for browser-submitted image and video jobs
- Frontend build, test, and packaged static asset verification integrated into the standard Makefile and CI workflows

### Changed
- Base installs now include the local Web UI runtime dependencies and packaged SPA assets
- Web configuration and model resolution now expose backend-aware defaults to the local UI without duplicating platform selection logic
- Model aliases may now resolve to different repositories depending on your operating system
- If a model alias is unavailable on your platform, a guidance message now explains why and suggests alternatives
- `ziv model list` now shows platform-specific availability for model aliases
- Bare `ziv` now shows command discovery/help, and the local Web UI is launched explicitly with `ziv ui` or `ziv-ui`

## [0.6.5] - 2026-04

### Added
- New logo asset set for project and documentation branding

### Changed
- README branding updated to use the new logo treatment
- Documentation home page and MkDocs site branding updated to use the new logo assets

### Removed
- Older root-level logo assets replaced by the new documentation asset set

## [0.6.4] - 2026-04

### Added
- Bundled `ltx_core_mlx` and `ltx_pipelines_mlx` directly into the repository so the package installs correctly from PyPI without relying on an external distribution of `ltx-2-mlx`
- Dynamic versioning via `hatch-vcs`: package version is now derived automatically from git tags

### Changed
- `ziv` is now a unified parent command with subcommands: `ziv image`, `ziv video`, `ziv model`
- `ziv-image` is now the primary standalone image generation command (previously `ziv`)
- `ziv-convert` renamed to `ziv-model`

### Removed
- `ziv` no longer acts as the image generation command directly — use `ziv-image` or `ziv image`
- `ziv-convert` entry point removed — use `ziv-model` or `ziv model`

## [0.6.0] - 2026-04

### Added

- Video upscale: distilled-only two-stage 2× spatial pipeline
- Documentation site (MkDocs + GitHub Pages)
- CI/CD workflows: lint/test, automated PyPI release, docs deployment
- Community files: license, code of conduct, contributing guide, security policy
- Issue and PR templates

## [0.5.0] - 2026-04

### Added

- Image-to-video generation (`--image` flag in `ziv-video`)
- Video LoRA support (single, stacked, configurable weights)
- Audio included by default in video generation (`--no-audio` to strip)

## [0.4.0] - 2026-04

### Added

- `ziv-video` entry point for video generation
- Text-to-video with LTX-2.3 on macOS (MLX backend)
- Video size presets (s/m/l/xl × 3 aspect ratios)
- Video model auto-detection
- Low-memory mode for video generation
- Batch video generation from prompt files

## [0.3.0] - 2026-04

### Added

- Reference image steering (`--image` and `--image-strength` flags)
- Image-to-image generation on both macOS and Windows platforms

## [0.2.0] - 2026-04

### Added

- `ziv-convert` entry point for model and LoRA management
- Checkpoint conversion (Z-Image, FLUX.2 Klein 4B/9B formats)
- LoRA import from local files and HuggingFace Hub
- `ziv-convert list` to display installed models and LoRAs

## [0.1.0] - 2026-04

### Added

- Text-to-image generation (macOS via mflux/MLX, Windows via diffusers/CUDA)
- Model store at `~/.ziv/` with bare-name resolution and HuggingFace fallback
- Built-in model aliases with custom alias support
- YAML prompt files with multiple prompt sets and `active` filtering
- Prompt variable syntax (`{option1|option2}`) with nesting
- Structured prompts (dict/list flattening) and reusable snippets
- LoRA support (single and stacked, configurable weights)
- Upscale pipeline (generate small → Lanczos → img2img refine → CAS sharpen)
- 4-bit and 8-bit quantization on both platforms
- Post-processing: contrast, saturation, CAS sharpening
- Interactive keyboard controls during batch runs (skip, quit, pause, repeat)
- `ziv` and `ziv-image` entry points
