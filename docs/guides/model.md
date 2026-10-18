# Model & LoRA Management

The `ziv-model` command converts checkpoints, imports LoRAs, and lists installed assets.

## Overview

`ziv-model` has three subcommands:

- **`model`** — convert safetensors checkpoints to diffusers format
- **`lora`** — import LoRA files from local paths or HuggingFace
- **`list`** — show installed models, video models, LoRAs, and aliases

## Converting Models (`model` subcommand)

Convert safetensors checkpoints to diffusers format, stored in `~/.ziv/models/`.

```bash
# Convert a Z-Image checkpoint (output: ~/.ziv/models/<name>/)
ziv-model model -i checkpoint.safetensors --name my-model

# Convert a FLUX.2 Klein 4B checkpoint
ziv-model model -i klein4b.safetensors --name klein4b --model-type flux2-klein-4b

# Copy base model files instead of symlinking
ziv-model model -i checkpoint.safetensors --name my-model --copy

# Also save a q8 copy (~/.ziv/models/my-model@q8/), used whenever my-model runs at -q 8 (macOS)
ziv-model model -i checkpoint.safetensors --name my-model --quantize 8
```

### `model` Flags

| Flag | Default | Description |
|---|---|---|
| `-i`, `--input` | *(required)* | Path to `.safetensors` checkpoint |
| `--name` | input filename | Custom model folder name |
| `--model-type` | `zimage` | Model type: `zimage`, `flux2-klein-4b`, `flux2-klein-9b`, `krea2-turbo` |
| `--base-model` | `Tongyi-MAI/Z-Image-Turbo` | Base HF repo (only for zimage type) |
| `--copy` | off | Copy files instead of symlinking |
| `--quantize` | off | Also save a quantized copy as `<name>@q4` or `<name>@q8` (`4` or `8`); see [Stored quants](image.md#stored-quants) |

Checkpoints can be bfloat16, float16, float32, or FP8. Scaled FP8 checkpoints (for example ComfyUI FP8 exports, which store a `weight_scale` next to each weight) are dequantized to bfloat16 during conversion. Block-wise or other quantized formats (such as NVFP4) are not supported.

`krea2-turbo` converts a fine-tune of Krea 2 Turbo. Native Krea 2 and ComfyUI checkpoints are both accepted, and a bundled text encoder or VAE is ignored; the text encoder and VAE come from unsloth's ungated mirror `unsloth/Krea-2-Turbo`.

> **Ideogram 4** is used via the `ideo` alias or a local FP8 (macOS) or NF4 (Windows and Linux) model directory and is distributed ready-to-use. It has no checkpoint-convert path, so it is not a `--model-type` option.

## Importing LoRAs (`lora` subcommand)

Import LoRA `.safetensors` files into `~/.ziv/loras/` from local paths or HuggingFace.

```bash
# Import a local LoRA file
ziv-model lora -i /path/to/style.safetensors --name my-style

# Download a LoRA from HuggingFace
ziv-model lora --hf user/lora-repo --name my-lora

# Download a specific file from a multi-file HF repo
ziv-model lora --hf user/lora-repo --file model.safetensors
```

### `lora` Flags

| Flag | Default | Description |
|---|---|---|
| `-i`, `--input` | — | Path to local `.safetensors` file (mutually exclusive with `--hf`) |
| `--hf` | — | HuggingFace repo ID (mutually exclusive with `-i`) |
| `--file` | auto-detect | Specific `.safetensors` file in the HF repo |
| `--name` | filename stem | Custom LoRA name |

## Listing Assets (`list` subcommand)

Show installed models, video models, LoRAs, and model aliases.

```bash
# List everything
ziv-model list

# Show only models
ziv-model list --models

# Show only LoRAs
ziv-model list --loras
```

### `list` Flags

| Flag | Default | Description |
|---|---|---|
| `--models` | off | Show only models |
| `--loras` | off | Show only LoRAs |

## Managing Models in the Web UI

The Web UI's Models page lists the same models and LoRAs and can convert checkpoints and import LoRAs. The **Size** column shows the disk each model uses: an alias's HuggingFace download (with the extra repos LTX downloads on Windows/Linux), or a converted model's own folder without the base-model files it links to. Quantized copies count with their model when they don't have a row of their own. Two aliases that load the same download, such as `ltx-4` and `ltx-8` on Windows/Linux, both show it. Each row also has a delete button, which asks for confirmation first and is unavailable while generation jobs are running or queued:

- **Converted models** (in `~/.ziv/models/`): deletes the model folder. Base-model files it links to in the HuggingFace cache are kept.
- **HuggingFace models** (aliases such as `klein9b`): deletes the model's download from the HuggingFace cache. The alias stays, and the model downloads again the next time you use it. If converted models link to files in that download, the confirmation lists them, because they stop working until it is downloaded again. The button appears once the model's files are fully downloaded.
- **LoRAs** (in `~/.ziv/loras/`): deletes the `.safetensors` file.
- **Prompt enhancer models**: the **Prompt enhancer** table has a row for **Fast**, **Smart** and, when a custom enhancer model is set on the Config page, **Custom**, with its Hugging Face repository and its disk use (or the size of the first-use download). Delete removes that model's download from the HuggingFace cache, including a partly finished one; it downloads again the next time you enhance a prompt with it. A custom model in a local folder is never deleted. Delete is unavailable while a prompt enhancement is running.
- **Other downloads**: models in the HuggingFace cache that no alias, enhancer model or converted model uses, for example a model an alias pointed at before an update, or a file a detection check fetched. The size is what deleting it frees: files it shares with another download stay. The HuggingFace cache is shared with other apps on your computer, so another app may download a deleted repo again. **Unused cache files** appears when the cache holds files no download links to any more, which deletes outside the app (or before this version) leave behind; deleting them frees that space without touching any download.
- **SeedVR2 upscale model** (macOS): the image upscale model has a row of its own, `seedvr2-7b@q8` or `seedvr2-3b@q8` depending on your Mac's memory, with its download size. Once the first plain upscale has stored it, the row shows its disk use and a delete button that removes that stored copy. Nothing else is affected, and the next plain upscale downloads the model again. See [Upscaling an existing image](image.md#upscaling-an-existing-image-web-ui).
