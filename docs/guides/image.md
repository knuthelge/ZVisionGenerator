# Image Generation Guide

Generate images from text prompts using `ziv-image`. Supports Z-Image / FLUX models on macOS (Apple Silicon via mflux/MLX) and on Windows and Linux with NVIDIA GPUs via diffusers/CUDA, plus Ideogram 4 on macOS.

On Windows and Linux, image generation requires CUDA to be visible to PyTorch. CPU fallback is not available for the diffusers image backend.

## Model Aliases

Built-in shorthands for common image models.

| Alias | Expands To |
|-------|------------|
| `zit` | `Tongyi-MAI/Z-Image-Turbo` |
| `klein4b` | `black-forest-labs/FLUX.2-klein-4B` |
| `klein9b` | `black-forest-labs/FLUX.2-klein-9B` |
| `ideo` | `ideogram-ai/ideogram-4-fp8` |

For video aliases, see [Video Guide → Model Aliases](video.md#model-aliases).

```bash
ziv-image -m zit --prompt "a beautiful sunset"
ziv-image -m klein4b --prompt "a portrait"
ziv-image -m ideo --prompt "a portrait"
```

### Ideogram 4

Ideogram 4 runs on macOS (Apple Silicon via mflux/MLX) only; it is unavailable on Windows and Linux. It ships as a single FP8 model, so quantization tiers (`-q 4` / `-q 8`) do not apply. Width and height must be in the 256–2048 range and multiples of 16; size presets that exceed this range (for example `--size xl` with `--ratio 16:9`) are rejected before the model loads.

```bash
ziv-image -m ideo --prompt "a portrait"
```

Ideogram 4 does not support negative prompts or reference-image (img2img) steering; a negative prompt is dropped with a warning, and both `--image` and `--upscale` are rejected (each requires img2img, which Ideogram 4 does not support). LoRA weights work at parity with other mflux models via `--lora`.

Plain-text prompts are automatically wrapped into Ideogram 4's structured JSON caption format, so they generate without a plain-text caption warning while preserving the original wording. To supply a full structured JSON caption instead, pass it as the value of `--json-prompt` (mutually exclusive with `--prompt`): this skips random-choice `{a|b|c}` expansion and sends the caption verbatim. Without `--json-prompt`, `{...}` in a prompt is treated as random-choice syntax and corrupts a JSON caption. The `--json-prompt` value must be a valid JSON object, or generation is rejected before the model loads:

```bash
ziv-image -m ideo --json-prompt '{"high_level_description": "a portrait"}'
```

When neither `--steps` nor `--guidance` is given, Ideogram 4 uses its built-in tuned quality schedule. Supplying `--steps` and/or `--guidance` overrides that schedule with the explicit values.

Ideogram 4 applies an automatic first-step adjustment to its denoising schedule that reduces spurious "Image blocked by safety filter" grey results at no change to the prompt, seed, or resolution. This mitigation is best-effort and not guaranteed to recover every refused generation. The adjustment defaults to a first-step sigma of `1.004` and can be overridden per run with `--first-sigma` (for example `--first-sigma 1.005` or `--first-sigma 1.006`) when a benign prompt is still blocked.

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

The reference image is automatically resized to match target dimensions. Works on macOS, Windows, and Linux.

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
| Windows / Linux | 4-bit (NF4), 8-bit (INT8) | bitsandbytes |

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

In the Web UI, image job controls appear only while the backend reports that the running job supports them.

## Related Guides

- [Prompts Guide](prompts.md) — prompt files, variables, structured prompts, and snippets
