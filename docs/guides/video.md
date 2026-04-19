# Video Generation Guide

Generate videos from text prompts or images using `ziv-video`. Works on macOS (Apple Silicon / MLX) and Windows (NVIDIA GPU / CUDA).

## Supported Models

| Model | Platform | Path / HF Repo ID | Quantization | I2V |
|---|---|---|---|---|
| LTX-2.3 | macOS | `dgrauet/ltx-2.3-mlx-q4` | Pre-quantized Q4 | ✅ |
| LTX-2.3 | macOS | `dgrauet/ltx-2.3-mlx-q8` | Pre-quantized Q8 | ✅ |
| LTX-2.3 | Windows | `Lightricks/LTX-2.3-fp8` | Pre-quantized FP8 | ✅ |

## Model Aliases

Aliases are platform-aware — they resolve to the correct model for your platform automatically.

| Alias | macOS | Windows |
|-------|-------|----------|
| `ltx-4` | `dgrauet/ltx-2.3-mlx-q4` | — (macOS-only; coming to Windows when Lightricks releases the distilled checkpoint) |
| `ltx-8` | `dgrauet/ltx-2.3-mlx-q8` | `Lightricks/LTX-2.3-fp8` |

See [Image Guide → Model Aliases](image.md#model-aliases) for the full alias list.

## Quick Start

```bash
# Text-to-video with LTX
ziv-video -m ltx-8 --prompt "A cat walking through a garden"

# Image-to-video with LTX
ziv-video -m ltx-8 --image photo.jpg --prompt "Camera slowly zooms in"

# Batch from prompts file
ziv-video -m ltx-8 -p prompts.yaml -r 3

# Square aspect, small size
ziv-video -m ltx-8 --ratio 1:1 --size s --prompt "Abstract art"
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

Both Q4 and full models include all required weights for upscaling — no extra downloads needed. The `--upscale 2` flag produces 2× resolution output with automatic refinement for sharper detail.

> **Memory:** 32 GB unified memory is sufficient for upscaled generation on macOS (~13 GB peak). On Windows, `ltx-8` uses a pre-quantized FP8 checkpoint for reduced VRAM usage.

## Video LoRA

```bash
# Single LoRA
ziv-video -m ltx-8 --prompt "A sunset" --lora /path/to/style.safetensors

# LoRA with custom weight
ziv-video -m ltx-8 --prompt "A sunset" --lora /path/to/style.safetensors:0.8

# Multiple LoRAs
ziv-video -m ltx-8 --prompt "A dance" --lora style.safetensors:0.5,motion.safetensors:0.8
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

- **HF repo IDs** — prefix matching: `dgrauet/ltx*` or `Lightricks/LTX*` → LTX
- **Local paths** — substring matching: paths containing "ltx" → LTX

## Related Guides

- See [Prompts Guide](prompts.md) for prompt syntax, variables, structured prompts, and snippets.
