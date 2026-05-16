# Video Generation Guide

Generate videos from text prompts or images using `ziv-video`. macOS uses the MLX LTX backend, while Windows and Linux use the shared diffusers/CUDA LTX backend.

## Supported Models

| Platform | Alias | Backend | Default Target | I2V |
|---|---|---|---|---|
| macOS | `ltx-4` | MLX (`ltx-pipelines-mlx`) | `dgrauet/ltx-2.3-mlx-q4` | ✅ |
| macOS | `ltx-8` | MLX (`ltx-pipelines-mlx`) | `dgrauet/ltx-2.3-mlx-q8` | ✅ |
| Windows | `ltx-2.3` | diffusers/CUDA | `dg845/LTX-2.3-Diffusers` by default | ✅ |
| Linux | `ltx-2.3` | diffusers/CUDA | `dg845/LTX-2.3-Diffusers` by default | ✅ |

## Model Aliases

| Alias | Expands To |
|-------|------------|
| `ltx-4` | `dgrauet/ltx-2.3-mlx-q4` |
| `ltx-8` | `dgrauet/ltx-2.3-mlx-q8` |
| `ltx-2.3` | `video_model_presets.ltx.diffusers.default_repo` on Windows and Linux |

`ltx-4` and `ltx-8` are macOS-only aliases. They are the shipped MLX Q4/Q8 presets. `ltx-2.3` is a Windows/Linux-only alias for the configurable diffusers default and does not select a packaged Q4 or Q8 tier. The mismatch errors are intentional and point to the platform-appropriate alias.

See [Image Guide → Model Aliases](image.md#model-aliases) for the full alias list.

## Quick Start

Use `MODEL` below as follows: `ltx-4` or `ltx-8` on macOS, `ltx-2.3` on Windows and Linux.

```bash
# Text-to-video with LTX on macOS
ziv-video -m ltx-4 --prompt "A cat walking through a garden"

# Text-to-video with LTX on Windows or Linux
ziv-video -m ltx-2.3 --prompt "A cat walking through a garden"

# Image-to-video with your platform alias
ziv-video -m MODEL --image photo.jpg --prompt "Camera slowly zooms in"

# Batch from prompts file with your platform alias
ziv-video -m MODEL -p prompts.yaml -r 3

# Square aspect, small size
ziv-video -m MODEL --ratio 1:1 --size s --prompt "Abstract art"
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

`--upscale 2` is the only supported video upscale mode. On macOS it uses the existing MLX distilled two-stage path. On Windows and Linux it uses the diffusers latent upscaler when that API is present in the installed diffusers build. If the selected runtime or model layout does not expose the latent upscaler, `ziv-video` fails explicitly instead of silently ignoring the request.

Windows and Linux video generation require CUDA. CPU fallback is not available for the diffusers video backend.

## Video LoRA

```bash
# Single LoRA
ziv-video -m MODEL --prompt "A sunset" --lora /path/to/style.safetensors

# LoRA with custom weight
ziv-video -m MODEL --prompt "A sunset" --lora /path/to/style.safetensors:0.8

# Multiple LoRAs
ziv-video -m MODEL --prompt "A dance" --lora style.safetensors:0.5,motion.safetensors:0.8
```

macOS keeps the existing MLX LoRA behavior. Windows and Linux accept LoRAs only when the selected asset and installed diffusers runtime support the LTX adapter APIs in a diffusers-compatible format. Unsupported adapter APIs, unsupported file formats, and unsupported explicit scales fail before generation with a descriptive error.

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
- **macOS MLX Q4 baseline**: `ltx-4`, 704×448, ≤49 frames with `--low-memory`

Windows and Linux use the configurable diffusers repository behind `ltx-2.3` instead of shipped Q4/Q8 aliases.

## Model Detection

`ziv-video` auto-detects the model family from supported model values:

- **Aliases** — `ltx-4` and `ltx-8` on macOS, `ltx-2.3` on Windows and Linux
- **Supported/configured repo IDs** — known LTX prefixes such as `dgrauet/ltx*` and `Lightricks/LTX-Video*`, plus the configured `video_model_presets.ltx.diffusers.default_repo`
- **Local paths** — local paths containing `ltx` are treated as LTX candidates, including Windows-style path strings on any host

For Windows/Linux direct repo swaps, point `ltx-2.3` at a compatible diffusers repository in `~/.ziv/config.yaml`. Arbitrary unconfigured HuggingFace repositories are not accepted solely because their names contain `ltx`.

Runtime `--lora` values must be local LoRA files or bare LoRA names that resolve from `~/.ziv/loras/`. HuggingFace-shaped LoRA references such as `org/lora:0.8` are import inputs for `ziv-model lora --hf`, not direct generation inputs.

## Common Platform Errors

- `Alias 'ltx-4' is macOS-only...` or `Alias 'ltx-8' is macOS-only...`: use `ltx-2.3` on Windows or Linux.
- `Alias 'ltx-2.3' is available on Windows and Linux only...`: use `ltx-4` or `ltx-8` on macOS.
- `CUDA is not available...`: the diffusers video backend requires an NVIDIA CUDA device on Windows and Linux.
- `Could not detect video model family...`: use the platform alias (`ltx-4`/`ltx-8` on macOS or `ltx-2.3` on Windows/Linux), a known/configured LTX repo ID, or a local path containing `ltx`.

## Related Guides

- See [Prompts Guide](prompts.md) for prompt syntax, variables, structured prompts, and snippets.
