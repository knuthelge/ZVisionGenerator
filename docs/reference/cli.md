# CLI Reference

## Entry Points

| Command | Description |
|---|---|
| `ziv` | Unified parent command with subcommands: `image`, `video`, `model`, `ui` |
| `ziv-ui` | Standalone Web UI launcher |
| `ziv-image` | Image generation (standalone entry point) |
| `ziv-video` | Video generation (standalone entry point) |
| `ziv-model` | Model/LoRA management (standalone entry point) |

## `ziv` — Unified Command

`ziv` is the parent command that dispatches to subcommands:

```bash
ziv image [options]    # Same as ziv-image
ziv video [options]    # Same as ziv-video
ziv model [options]    # Same as ziv-model
ziv ui [options]       # Same as ziv-ui
```

Running `ziv` with no subcommand prints command-discovery help and exits without starting a server. Use `ziv ui` or `ziv-ui` when you want to launch the Web UI.

Run `ziv --help` for available subcommands, or `ziv <command> --help` for command-specific options.

## `ziv ui` / `ziv-ui` — Web UI Launcher

The base package install includes the Web UI runtime, so `ziv ui` and `ziv-ui` work after a normal `uv tool install z-vision-generator` or `uv sync`. If a local environment is missing required Web UI packages, the launcher exits with repair guidance for the base install instead of pointing users at a second install step.

| Argument | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Host interface to bind the local Web UI server to |
| `--port` | `8080` | Preferred local port; the app chooses the next available port if this one is busy |
| `--no-browser` | off | Start the server without opening a browser tab |

## `ziv-image` — Image Generation

| Argument | Default | Description |
|---|---|---|
| `-m`, `--model` | *(required)* | Model name, local path, alias, or supported HuggingFace repo ID |
| `-q`, `--quantize` | `None` | Quantization level: `4` or `8` |
| `--scheduler` | model-aware | Scheduler name (e.g., `beta`) |
| `--lora` | `None` | LoRA specifier: `name:weight,name2:weight2` (weight optional, default 1.0) |
| `--prompt` | `None` | Inline prompt; overrides `--prompts-file` |
| `-p`, `--prompts-file` | `prompts.yaml` | YAML prompt file path |
| `-r`, `--runs` | `1` | Number of generation runs |
| `-s`, `--size` | `m` | Size preset: `xs`, `s`, `m`, `l`, `xl` |
| `--ratio` | `2:3` | Aspect ratio: `1:1`, `16:9`, `9:16`, `3:2`, `2:3` |
| `-W`, `--width` | from preset | Override image width |
| `-H`, `--height` | from preset | Override image height |
| `--steps` | model-aware | Number of diffusion steps |
| `--guidance` | model-aware | Guidance scale |
| `--seed` | `None` (random) | Seed for reproducible image generation |
| `--upscale` | disabled | Upscale factor: `2` or `4` |
| `--upscale-denoise` | `0.3` (2×) / `0.4` (4×) | Denoising strength for upscale pass |
| `--upscale-steps` | `steps / 2` | Refinement steps for upscale |
| `--upscale-guidance` | same as `--guidance` | Override guidance scale for the upscale refine pass only |
| `--upscale-sharpen` | `True` | CAS sharpening step before upscale refinement (`--no-upscale-sharpen` to disable) |
| `--upscale-save-pre` | `False` | Save pre-upscale image alongside final |
| `--sharpen [AMOUNT]` / `--no-sharpen` | enabled | Apply CAS sharpening. Optional amount overrides config (0.0–1.0) |
| `--contrast [AMOUNT]` / `--no-contrast` | disabled | Apply contrast adjustment. Optional amount (1.0 = no change) |
| `--saturation [AMOUNT]` / `--no-saturation` | disabled | Apply saturation adjustment. Optional amount (1.0 = no change) |
| `--image` | disabled | Path to reference image for img2img steering |
| `--image-strength` | `0.5` | Denoising strength for reference (0.0–1.0) |
| `-o`, `--output` | `.` | Output directory for generated images |

Run `ziv-image --help` for the full list.

## `ziv-video` — Video Generation

Platform-aware aliases:

- macOS: `ltx-4`, `ltx-8`
- Windows and Linux: `ltx-2.3`
- The Windows/Linux alias resolves through `video_model_presets.ltx.diffusers.default_repo`, which defaults to `dg845/LTX-2.3-Diffusers` and can be overridden in `~/.ziv/config.yaml`.
- `ltx-4` and `ltx-8` are the shipped macOS MLX Q4/Q8 aliases. `ltx-2.3` is the Windows/Linux diffusers alias and does not imply a packaged Q4/Q8 tier.

| Argument | Default | Description |
|---|---|---|
| `-m`, `--model` | *(required)* | Model alias, local path, or supported/configured LTX HuggingFace repo ID |
| `--prompt` | `None` | Inline prompt; overrides `--prompts-file` |
| `-p`, `--prompts-file` | `prompts.yaml` | YAML prompt file path |
| `--image` | `None` | Input image for image-to-video |
| `-W`, `--width` | model-aware | Video width (LTX: auto-aligned to 32px, 64px with `--upscale`) |
| `-H`, `--height` | model-aware | Video height (LTX: auto-aligned to 32px, 64px with `--upscale`) |
| `--frames` | model-aware | Number of frames (LTX: auto-aligned to 8k+1) |
| `--ratio` | `16:9` | Aspect ratio preset (16:9, 9:16, 1:1) |
| `-s`, `--size` | `m` | Size preset (`s`, `m`, `l`, `xl`) |
| `--steps` | model-aware | Inference steps |
| `--seed` | `None` (random) | Seed for reproducible video generation |
| `--low-memory` / `--no-low-memory` | enabled | Low-memory mode for LTX |
| `--upscale` | disabled | Upscale factor: `2` only. Fails explicitly when the active backend cannot provide the LTX latent upscaler. |
| `--audio` / `--no-audio` | enabled | Include or strip audio in video output |
| `--lora` | `None` | Comma-separated LoRAs with optional weights: `name:0.8,name2:0.5`. Windows/Linux require diffusers-compatible LTX LoRA adapters. |
| `-r`, `--runs` | `1` | Number of batch runs |
| `-o`, `--output` | `.` | Output directory |
| `--format` | `mp4` | Output format |

Operational notes:

- Video generation requires `ffmpeg` on every platform.
- Windows and Linux video generation are CUDA-only.
- Alias mismatches are reported directly, for example `ltx-4` on Windows or `ltx-2.3` on macOS.

## `ziv-model model` — Checkpoint Conversion

| Flag | Default | Description |
|---|---|---|
| `-i`, `--input` | *(required)* | Path to  `.safetensors` checkpoint |
| `--name` | input filename | Custom model folder name |
| `--model-type` | `zimage` | Model type: `zimage`, `flux2-klein-4b`, `flux2-klein-9b` |
| `--base-model` | `Tongyi-MAI/Z-Image-Turbo` | Base HF repo (only for zimage type) |
| `--copy` | off | Copy files instead of symlinking |

## `ziv-model lora` — LoRA Import

| Flag | Default | Description |
|---|---|---|
| `-i`, `--input` | — | Path to local `.safetensors` file (mutually exclusive with `--hf`) |
| `--hf` | — | HuggingFace repo ID (mutually exclusive with `-i`) |
| `--file` | auto-detect | Specific `.safetensors` file in the HF repo |
| `--name` | filename stem | Custom LoRA name |

## `ziv-model list` — Asset Listing

| Flag | Default | Description |
|---|---|---|
| `--models` | off | Show only models |
| `--loras` | off | Show only LoRAs |

## Image Size Presets

Default ratio is `2:3`. Dimensions vary by `--ratio`.

| Preset | 1:1 | 16:9 | 9:16 | 3:2 | 2:3 |
|--------|-----|------|------|-----|-----|
| `xs` | 512×512 | 672×384 | 384×672 | 608×400 | 400×608 |
| `s` | 704×704 | 944×528 | 528×944 | 864×576 | 576×864 |
| `m` | 1024×1024 | 1344×768 | 768×1344 | 1216×832 | 832×1216 |
| `l` | 1440×1440 | 1888×1056 | 1056×1888 | 1728×1152 | 1152×1728 |
| `xl` | 1600×1600 | 2112×1184 | 1184×2112 | 1936×1296 | 1296×1936 |

## Video Size Presets

Default ratio is `16:9`. Dimensions vary by `--ratio`.

| Preset | 16:9 | 9:16 | 1:1 |
|--------|------|------|-----|
| `s` | 512×256 (49f) | 256×512 (49f) | 384×384 (49f) |
| `m` | 704×448 (49f) | 448×704 (49f) | 512×512 (33f) |
| `l` | 960×512 (33f) | 512×960 (33f) | 768×768 (33f) |
| `xl` | 1408×896 (25f) | 896×1408 (25f) | 1024×1024 (25f) |
