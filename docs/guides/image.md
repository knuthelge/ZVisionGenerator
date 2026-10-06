# Image Generation Guide

Generate images from text prompts using `ziv-image`. Supports Z-Image / FLUX / Krea 2 models on macOS (Apple Silicon via mflux/MLX) and on Windows and Linux with NVIDIA GPUs via diffusers/CUDA, plus Ideogram 4 on macOS.

On Windows and Linux, image generation requires CUDA to be visible to PyTorch. CPU fallback is not available for the diffusers image backend.

## Model Aliases

Built-in shorthands for common image models.

| Alias | Expands To |
|-------|------------|
| `zit` | `Tongyi-MAI/Z-Image-Turbo` |
| `klein4b` | `black-forest-labs/FLUX.2-klein-4B` |
| `klein9b` | `black-forest-labs/FLUX.2-klein-9B` |
| `ideo` | `ideogram-ai/ideogram-4-fp8` |
| `krea2` | `krea/Krea-2-Turbo` |

For video aliases, see [Video Guide → Model Aliases](video.md#model-aliases).

```bash
ziv-image -m zit --prompt "a beautiful sunset"
ziv-image -m klein4b --prompt "a portrait"
ziv-image -m ideo --prompt "a portrait"
ziv-image -m krea2 -q 4 --prompt "a portrait"
```

### Ideogram 4

Ideogram 4 runs on macOS (Apple Silicon via mflux/MLX) only; it is unavailable on Windows and Linux. It ships as a single FP8 model, so quantization tiers do not apply: `ziv-image` rejects `-q 4` / `-q 8` for it, as the Web UI does. Width and height must be in the 256–2048 range and multiples of 16; size presets that exceed this range (for example `--size xl` with `--ratio 16:9`) are rejected before the model loads.

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

#### In the Web UI

Ideogram 4 is selectable in the workspace model dropdown. When it is selected, the workspace exposes a structured-JSON-caption input and a first-step-sigma control that mirror `--json-prompt` and `--first-sigma`. Reference-image (img2img) and upscale are unavailable for it, and width and height are constrained to the 256–2048 range in multiples of 16.

### Krea 2

`krea2` is Krea 2 Turbo, a 12B text-to-image model distilled to 8 steps. Its Hugging Face repo is gated: accept the Krea 2 Community License on the [model page](https://huggingface.co/krea/Krea-2-Turbo) and set a Hugging Face token before the first run. It defaults to 8 steps and guidance 1.0, which turns classifier-free guidance off; a negative prompt only has an effect with guidance above 1.0, and each step then takes about twice as long. Guidance uses the same scale on every platform.

Krea 2 samples with its own schedule, so `--scheduler` is rejected for it and the Web UI hides the Scheduler control. Krea 2 Raw (`krea/Krea-2-Raw`, or a fine-tune converted with `--model-type krea2-raw`, see [Model & LoRA Guide](model.md)) defaults to 28 steps and guidance 5.5.

The weights are large (about 26 GB for the transformer plus a 9 GB text encoder), so run it quantized on smaller machines:

- **macOS:** `-q 8` or `-q 4`. The text encoder is not quantized, so it adds about 9 GB at any level. The first quantized run saves a stored quant (`krea2@q8`, `krea2@q4`).
- **Windows and Linux:** `-q 4` is the level for 10–12 GB GPUs; its text encoder and transformer take turns on the GPU. Without quantization all 35 GB of weights are held in system memory. Reference images (`--image`) and `--upscale` are not available here, because diffusers has no Krea 2 image-to-image pipeline: `ziv-image` rejects them and the Web UI hides them. On macOS both work.

LoRAs trained for Krea 2 load with `--lora` on both platforms.

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

### Upscaling an existing image (Web UI)

To upscale an image you already have, choose **Upscale → 2×** or **4×** in the asset viewer (Gallery or Workspace) or in the **⋯** menu on its thumbnail. In the viewer you can also press **X** then **2** or **4**. Each option shows the output size, e.g. `2× → 1664×2432`. The job runs in the Workspace job panel, or joins the queue when another job is running; starting it from the Gallery takes you there.

The image is resized, refined with img2img and sharpened, using the settings recorded in the file: prompt, model, seed, steps, guidance, scheduler, LoRAs, negative prompt and quantization. When the recorded model is not configured here, the default image model refines it with its own defaults. Images without recorded settings (imported or older files) are refined by the default image model with an empty prompt; the job panel notes this. The image is already sharpened, so by default the CAS pass before refinement is skipped (`sharpening.existing_pre_upscale: 0`); the final pass uses `sharpening.existing_upscaled`. Contrast and saturation are not applied again.

The result is saved next to the original as `<name>_2x.png` or `<name>_4x.png`, with a counter (`_2`) if that name exists. An option is disabled, with the reason in its tooltip, when the model cannot upscale or refine images (e.g. Ideogram 4), or when the output would exceed the model's maximum dimension or the megapixel limit. These settings live in the `upscale` section of `~/.ziv/config.yaml`:

```yaml
upscale:
  existing_denoise_small: 0.4      # Refinement denoise for outputs up to existing_large_megapixels
  existing_denoise_large: 0.2      # Larger outputs: high denoise at large sizes invents detail and seams
  existing_large_megapixels: 2
  max_megapixels: 20               # Largest output a viewer upscale may produce
```

The denoise depends on the output size, not the factor, so upscaling an upscale (2× then 2×) refines as lightly as a direct 4× to the same size.

`default_denoise_2x` and `default_denoise_4x` apply to `--upscale` and the Workspace upscale settings only.

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

On macOS, memory is shared between the CPU and GPU, so an unquantized model larger than the machine's recommended GPU memory makes macOS compress or swap other memory, and well beyond it the whole system swaps heavily. On a 16 GB Mac, for example, Z-Image Turbo and FLUX.2 Klein need `-q 8` or `-q 4`. The Web UI's memory badge shows which levels fit.

The [prompt enhancer](prompts.md#enhancing-prompts) is unloaded before the model loads, so **Enhance each image** does not add to the model's memory.

### Stored quants (macOS)

The first time a model runs at `-q 8` or `-q 4` (or **Quant: q8/q4** in the Web UI), its quantized weights are saved in `~/.ziv/models/` as `<name>@q8` or `<name>@q4`, where `<name>` is the installed model or the alias you picked (for example `zit@q8`). Later jobs load that copy instead of quantizing again, so they start faster and do not briefly need the unquantized model's memory. The first job shows *Saving a q8 copy…* and takes about 20–30 seconds longer; the copy needs roughly half (q8) or a quarter (q4) of the model's disk space.

- The copy holds the base model only. LoRAs are applied on top of it when a job loads, with the same result as quantizing at load.
- It is replaced automatically when the source model's weights change (including a new download of an alias's Hugging Face repo) or mflux is updated.
- It is listed on the Models page as *q8 copy of &lt;name&gt;*, where you can delete it to free disk space. It does not appear in the model picker: choose the base model and a quantize level.
- Deleting a model on the Models page, or an alias's Hugging Face download, also deletes its copies.
- Stopping a job while the copy is being saved takes effect once the current write finishes (up to about 30 seconds); the copy is then discarded and the next job tries again.
- An alias whose model is not downloaded yet is downloaded first; the copy is saved once the download is complete.
- A model given as a raw Hugging Face repo id or a folder outside `~/.ziv/models/` (`-m org/repo`, `-m /path/to/model`) is quantized at load each time. Use an alias, or install the model, to get a copy.

You can also save a quantized copy when converting a checkpoint (`ziv-model model … --quantize 8`, or **Quantized Copy** on the Models page).

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
ziv-image -m my-model --prompt "a sunset" --sharpen 0.6          # custom amount (0.0–1.5)
ziv-image -m my-model --prompt "a sunset" --no-sharpen           # disable
```

Without an amount, the final pass uses the config: `sharpening.normal` for plain images and `sharpening.upscaled` for upscaled ones (`sharpening.pre_upscale` is the pass before upscale refinement). In the Web UI, leave the **Sharpen** amount empty (**auto**) to use these, or type an amount to override them. Amounts above 1.5 are rejected: the filter breaks down beyond that.

```yaml
sharpening:
  normal: 1.0
  upscaled: 1.2
  pre_upscale: 0.8
  existing_upscaled: 1.2     # final pass when upscaling an existing image (viewer)
  existing_pre_upscale: 0    # before refining an existing image; 0 = off
```

The asset viewer's **Details** show the amounts an image was made with, including the pre-sharpen of an upscale.

## Output Files

Images are saved as `<set name>_<YYYY-MM-DD_HH-MM-SS>.png` (for example `portrait_2026-10-04_14-03-22.png`), with `_2`, `_3`… added when that name is already taken, so no file is overwritten. Videos follow the same pattern with their format's extension.

The settings each file was made with are embedded in it: prompt, negative prompt, model and family, seed, steps, guidance, scheduler, size, LoRAs, reference image and strength, quantization, generation time, upscale factor/denoise/steps, and the sharpen, contrast and saturation amounts. The Web UI asset viewer shows them under **Details** (**I**), and **Reuse settings** loads them into the Workspace.

## Keyboard Shortcuts

During batch generation:

| Key | Action | Description |
|-----|--------|-------------|
| `n` | **Skip** | Stop current image, move to next prompt |
| `q` | **Quit** | Stop current image, exit batch |
| `p` | **Pause** | Finish current image, pause until keypress |
| `r` | **Repeat** | Finish current image, re-run the same prompt (same random choices and enhanced prompt) with a new random seed |

The keys also work while prompts are enhanced and while the model loads. During enhancement, `n` skips the current rewrite, `q` stops before the model loads, `p` pauses, and `r` has no effect. See [While Prompts Are Enhanced](prompts.md#while-prompts-are-enhanced).

In the Web UI, image job controls appear only while the backend reports that the running job supports them.

## Related Guides

- [Prompts Guide](prompts.md) — prompt files, variables, structured prompts, and snippets
