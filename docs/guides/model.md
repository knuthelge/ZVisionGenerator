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
| `--model-type` | `zimage` | Model type: `zimage`, `flux2-klein-4b`, `flux2-klein-9b` |
| `--base-model` | `Tongyi-MAI/Z-Image-Turbo` | Base HF repo (only for zimage type) |
| `--copy` | off | Copy files instead of symlinking |
| `--quantize` | off | Also save a quantized copy as `<name>@q4` or `<name>@q8` (`4` or `8`, macOS); see [Stored quants](image.md#stored-quants-macos) |

Checkpoints can be bfloat16, float16, float32, or FP8. Scaled FP8 checkpoints (for example ComfyUI FP8 exports, which store a `weight_scale` next to each weight) are dequantized to bfloat16 during conversion. Block-wise or other quantized formats (such as NVFP4) are not supported.

> **Ideogram 4** is used via the `ideo` alias or a local FP8 model directory and is distributed ready-to-use. It has no checkpoint-convert path, so it is not a `--model-type` option.

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

The Web UI's Models page lists the same models and LoRAs and can convert checkpoints and import LoRAs. Each row also has a delete button, which asks for confirmation first and is unavailable while a generation is running:

- **Converted models** (in `~/.ziv/models/`): deletes the model folder. Base-model files it links to in the HuggingFace cache are kept.
- **HuggingFace models** (aliases such as `klein9b`): deletes the model's download from the HuggingFace cache. The alias stays, and the model downloads again the next time you use it. If converted models link to files in that download, the confirmation lists them, because they stop working until it is downloaded again. The button appears once the model's files are fully downloaded.
- **LoRAs** (in `~/.ziv/loras/`): deletes the `.safetensors` file.
