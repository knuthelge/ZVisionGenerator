# CLI Reference

## Entry Points

| Command | Description |
|---|---|
| `ziv` | Unified parent command with subcommands: `image`, `video`, `model` |
| `ziv-image` | Image generation (standalone entry point) |
| `ziv-video` | Video generation (standalone entry point) |
| `ziv-model` | Model/LoRA management (standalone entry point) |

## `ziv` — Unified Command

`ziv` is the parent command that dispatches to subcommands:

```bash
ziv image [options]    # Same as ziv-image
ziv video [options]    # Same as ziv-video
ziv model [options]    # Same as ziv-model
```

Run `ziv --help` for available subcommands, or `ziv <command> --help` for command-specific options.

## `ziv-image` — Image Generation

| Argument | Default | Description |
|---|---|---|
| `-m`, `--model` | *(required)* | Model name, path, or HuggingFace repo ID |
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

| Argument | Default | Description |
|---|---|---|
| `-m`, `--model` | *(required)* | Model path or HuggingFace repo ID |
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
| `--upscale` | disabled | Upscale factor: `2` (two-stage pipeline, 2× spatial resolution) |
| `--audio` / `--no-audio` | enabled | Include or strip audio in video output |
| `--lora` | `None` | Comma-separated LoRAs with optional weights: `name:0.8,name2:0.5` |
| `-r`, `--runs` | `1` | Number of batch runs |
| `-o`, `--output` | `.` | Output directory |
| `--format` | `mp4` | Output format |

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
