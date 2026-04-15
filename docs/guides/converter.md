# Model & LoRA Management

The `ziv-convert` command converts checkpoints, imports LoRAs, and lists installed assets.

## Overview

`ziv-convert` has three subcommands:

- **`model`** — convert safetensors checkpoints to diffusers format
- **`lora`** — import LoRA files from local paths or HuggingFace
- **`list`** — show installed models, video models, LoRAs, and aliases

## Converting Models (`model` subcommand)

Convert safetensors checkpoints to diffusers format, stored in `~/.ziv/models/`.

```bash
# Convert a Z-Image checkpoint (output: ~/.ziv/models/<name>/)
ziv-convert model -i checkpoint.safetensors --name my-model

# Convert a FLUX.2 Klein 4B checkpoint
ziv-convert model -i klein4b.safetensors --name klein4b --model-type flux2-klein-4b

# Copy base model files instead of symlinking
ziv-convert model -i checkpoint.safetensors --name my-model --copy
```

### `model` Flags

| Flag | Default | Description |
|---|---|---|
| `-i`, `--input` | *(required)* | Path to `.safetensors` checkpoint |
| `--name` | input filename | Custom model folder name |
| `--model-type` | `zimage` | Model type: `zimage`, `flux2-klein-4b`, `flux2-klein-9b` |
| `--base-model` | `Tongyi-MAI/Z-Image-Turbo` | Base HF repo (only for zimage type) |
| `--copy` | off | Copy files instead of symlinking |

## Importing LoRAs (`lora` subcommand)

Import LoRA `.safetensors` files into `~/.ziv/loras/` from local paths or HuggingFace.

```bash
# Import a local LoRA file
ziv-convert lora -i /path/to/style.safetensors --name my-style

# Download a LoRA from HuggingFace
ziv-convert lora --hf user/lora-repo --name my-lora

# Download a specific file from a multi-file HF repo
ziv-convert lora --hf user/lora-repo --file model.safetensors
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
ziv-convert list

# Show only models
ziv-convert list --models

# Show only LoRAs
ziv-convert list --loras
```

### `list` Flags

| Flag | Default | Description |
|---|---|---|
| `--models` | off | Show only models |
| `--loras` | off | Show only LoRAs |
