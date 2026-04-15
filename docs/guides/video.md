# Video Generation Guide

Generate videos from text prompts or images using `ziv-video`. macOS (Apple Silicon) only.

## Supported Models

| Model | Backend | Path / HF Repo ID | Quantization | I2V |
|---|---|---|---|---|
| LTX-2.3 | ltx-pipelines-mlx | `dgrauet/ltx-2.3-mlx-q4` | Pre-quantized Q4 | ✅ |
| LTX-2.3 | ltx-pipelines-mlx | `dgrauet/ltx-2.3-mlx-q8` | Pre-quantized Q8 | ✅ |

## Model Aliases

| Alias | Expands To |
|-------|------------|
| `ltx-4` | `dgrauet/ltx-2.3-mlx-q4` |
| `ltx-8` | `dgrauet/ltx-2.3-mlx-q8` |

See [Image Guide → Model Aliases](image.md#model-aliases) for the full alias list.

## Quick Start

```bash
# Text-to-video with LTX
ziv-video -m ltx-4 --prompt "A cat walking through a garden"

# Image-to-video with LTX
ziv-video -m ltx-4 --image photo.jpg --prompt "Camera slowly zooms in"

# Batch from prompts file
ziv-video -m ltx-4 -p prompts.yaml -r 3

# Square aspect, small size
ziv-video -m ltx-4 --ratio 1:1 --size s --prompt "Abstract art"
```

## Video Upscale & Audio

```bash
# Generate upscaled video (2x resolution)
ziv-video -m MODEL --upscale 2 --prompt "..."

# Upscaled video with custom step count
ziv-video -m MODEL --upscale 2 --steps 6 --prompt "..."

# Upscaled image-to-video
ziv-video -m MODEL --upscale 2 --image photo.jpg --prompt "Camera slowly zooms in"

# Strip audio from output
ziv-video -m MODEL --no-audio --prompt "..."

# Upscale + no audio
ziv-video -m MODEL --upscale 2 --no-audio --prompt "..."
```

Both Q4 and full models include all required weights for upscaling — no extra downloads needed. The upscaler uses a distilled-only two-stage pipeline (8 denoising steps at half resolution, then 3 refinement steps at full resolution).

> **Memory:** 32 GB unified memory is sufficient for upscaled generation with Q4 model (~13 GB peak).

## Video LoRA

```bash
# Single LoRA
ziv-video -m ltx-4 --prompt "A sunset" --lora /path/to/style.safetensors

# LoRA with custom weight
ziv-video -m ltx-4 --prompt "A sunset" --lora /path/to/style.safetensors:0.8

# Multiple LoRAs
ziv-video -m ltx-4 --prompt "A dance" --lora style.safetensors:0.5,motion.safetensors:0.8
```

## Video Sizes

Default ratio is `16:9`. Dimensions vary by `--ratio`.

| Preset | 16:9 | 9:16 | 1:1 |
|--------|------|------|-----|
| `s` | 512×256 (49f) | 256×512 (49f) | 384×384 (49f) |
| `m` | 704×448 (49f) | 448×704 (49f) | 512×512 (33f) |
| `l` | 960×512 (33f) | 512×960 (33f) | 768×768 (33f) |
| `xl` | 1408×896 (25f) | 896×1408 (25f) | 1024×1024 (25f) |

Use `-W` / `-H` to override with exact pixel dimensions.

## LTX Constraints

LTX-2.3 has specific alignment requirements that are auto-corrected with a warning:

- **Resolution**: width and height must be divisible by 32 (64 when using `--upscale`)
- **Frames**: must follow 8k+1 pattern (9, 17, 25, 33, 41, 49, ..., 97, 121)
- **32GB safe**: Q4 pre-quantized, 704×448, ≤49 frames with `--low-memory`

## Model Detection

`ziv-video` auto-detects the model family from the path:

- **HF repo IDs** — prefix matching: `dgrauet/ltx*` → LTX
- **Local paths** — substring matching: paths containing "ltx" → LTX

## Related Guides

- See [Prompts Guide](prompts.md) for prompt syntax, variables, structured prompts, and snippets.
