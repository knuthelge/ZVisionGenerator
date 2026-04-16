# Image Generation Guide

Generate images from text prompts using `ziv-image`. Supports Z-Image / FLUX models on macOS (Apple Silicon via mflux/MLX) and Windows (NVIDIA GPU via diffusers/CUDA).

## Model Aliases

Built-in shorthands for common image models.

| Alias | Expands To |
|-------|------------|
| `zit` | `Tongyi-MAI/Z-Image-Turbo` |
| `klein4b` | `black-forest-labs/FLUX.2-klein-4B` |
| `klein9b` | `black-forest-labs/FLUX.2-klein-9B` |

For video aliases, see [Video Guide → Model Aliases](video.md#model-aliases).

```bash
ziv-image -m zit --prompt "a beautiful sunset"
ziv-image -m klein4b --prompt "a portrait"
```

### Custom Aliases

Add your own aliases in `~/.ziv/config.yaml`:

```yaml
model_aliases:
  mymodel: "my-org/my-model"
```

> **Note:** A local model directory at `~/.ziv/models/<alias>/` overrides the alias.

View all aliases with:

```bash
ziv-model list
```

## Image Sizes

Default ratio is `2:3`. Dimensions vary by `--ratio`.

| Preset | 1:1 | 16:9 | 9:16 | 3:2 | 2:3 |
|--------|-----|------|------|-----|-----|
| `xs` | 512×512 | 672×384 | 384×672 | 608×400 | 400×608 |
| `s` | 704×704 | 944×528 | 528×944 | 864×576 | 576×864 |
| `m` | 1024×1024 | 1344×768 | 768×1344 | 1216×832 | 832×1216 |
| `l` | 1440×1440 | 1888×1056 | 1056×1888 | 1728×1152 | 1152×1728 |
| `xl` | 1600×1600 | 2112×1184 | 1184×2112 | 1936×1296 | 1296×1936 |

Use `-W` / `-H` to override with exact pixel dimensions:

```bash
ziv-image -m my-model --prompt "a portrait" -W 1024 -H 1024
```

## Reference Image Steering

Use any image as a starting point — the model denoises it guided by your prompt.

```bash
ziv-image -m my-model --prompt "A woman in a red dress" --image photo.jpg --image-strength 0.4
ziv-image -m my-model --prompt "Cyberpunk cityscape" --image sketch.png --image-strength 0.8
```

The reference image is automatically resized to match target dimensions. Works on both macOS and Windows.

## LoRA Support

Both platforms support LoRA weights. Place `.safetensors` files in `~/.ziv/loras/`:

```bash
# Single LoRA at default weight (1.0)
ziv-image -m my-model --lora myStyle

# Single LoRA with explicit weight
ziv-image -m my-model --lora myStyle:0.8

# Two LoRAs stacked
ziv-image -m my-model --lora style1:0.8,detail:0.5
```

Bare names are resolved from `~/.ziv/loras/`. Full paths also work.

## Upscaling

The built-in upscale pipeline generates at a reduced size, then refines to target resolution:

1. Generate at reduced size (target ÷ upscale factor)
2. Lanczos upscale to target dimensions
3. CAS pre-sharpening → img2img refinement → CAS post-sharpening

```bash
ziv-image -m my-model --prompt "a landscape" --upscale 2
ziv-image -m my-model --prompt "a landscape" --upscale 4 --upscale-denoise 0.3 --upscale-steps 8

# Use a different guidance for the upscale refine pass
ziv-image -m my-model --prompt "a landscape" --upscale 2 --upscale-guidance 0.8

# Skip the pre-sharpening CAS step before upscale refinement
ziv-image -m my-model --prompt "a landscape" --upscale 2 --no-upscale-sharpen
```

### Upscale Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--upscale` | disabled | Upscale factor: `2` or `4` |
| `--upscale-denoise` | `0.3` (2×) / `0.4` (4×) | Denoising strength for upscale pass |
| `--upscale-steps` | `steps / 2` | Refinement steps for upscale |
| `--upscale-guidance` | same as `--guidance` | Override guidance scale for the upscale refine pass only |
| `--upscale-sharpen` | `True` | CAS sharpening step before upscale refinement (`--no-upscale-sharpen` to disable) |
| `--upscale-save-pre` | `False` | Save pre-upscale image alongside final |

## Quantization

Reduces memory usage and speeds up generation at the cost of some quality.

| Platform | Levels | Method |
|----------|--------|--------|
| macOS | 4-bit, 8-bit | mflux quantization |
| Windows | 4-bit (NF4), 8-bit (INT8) | bitsandbytes |

```bash
ziv-image -m my-model -q 4    # 4-bit quantization
ziv-image -m my-model -q 8    # 8-bit quantization
```

## Post-Processing

### Contrast

```bash
ziv-image -m my-model --prompt "a sunset" --contrast 1.2        # boost contrast (1.0 = no change)
ziv-image -m my-model --prompt "a sunset" --no-contrast         # disable entirely
```

### Saturation

```bash
ziv-image -m my-model --prompt "a sunset" --saturation 1.3       # boost saturation (1.0 = no change)
ziv-image -m my-model --prompt "a sunset" --no-saturation        # disable entirely
```

### Sharpening

```bash
ziv-image -m my-model --prompt "a sunset" --sharpen              # enabled by default
ziv-image -m my-model --prompt "a sunset" --sharpen 0.6          # custom amount (0.0–1.0)
ziv-image -m my-model --prompt "a sunset" --no-sharpen           # disable
```

## Keyboard Shortcuts

During batch generation:

| Key | Action | Description |
|-----|--------|-------------|
| `n` | **Skip** | Stop current image, move to next prompt |
| `q` | **Quit** | Stop current image, exit batch |
| `p` | **Pause** | Finish current image, pause until keypress |
| `r` | **Repeat** | Finish current image, re-run same prompt with new seed |

## Related Guides

- [Prompts Guide](prompts.md) — prompt files, variables, structured prompts, and snippets
