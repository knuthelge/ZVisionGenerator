# Models and LoRAs

`ziv-model` converts a fine-tuned checkpoint into a model the app can run, imports LoRAs, and lists the models, LoRAs and model names you can use. The guide [Download, delete and convert models](../guides/models.md) says when to convert a fine-tune and what the options mean.

A built-in model downloads the first time you use it with `-m`, so there is nothing to download first. Errors of `ziv-model` print `Error: …` on standard error and exit with status `1`.

## Convert a fine-tune { #convert }

### Convert a Z-Image fine-tune

```bash
ziv-model model -i checkpoint.safetensors --name my-model
```

You get `my-model` in `models/` in the data folder. Run it with `-m my-model`. `--model-type` is `zimage` when left out, and `--base-model` names the base repo of a Z-Image model (default `Tongyi-MAI/Z-Image-Turbo`). The model's folder links to the base model's files; `--copy` copies them in instead. `--name` defaults to the file's name, and a built-in model name such as `ltx-4` is refused.

### Convert a FLUX.2 Klein fine-tune with a q8 copy

```bash
ziv-model model -i klein-finetune.safetensors --name my-klein --model-type flux2-klein-4b --quantize 8
```

`--quantize 8` also stores a quantized copy beside the model, as `my-klein@q8`. That copy is used only when you run with `-q 8`; without `-q`, `-m my-klein` runs the full model. `--quantize 4` stores a `@q4` copy for `-q 4`. The other image types are `flux2-klein-9b` and `krea2-turbo`.

```bash
ziv-image -m my-klein -q 8 --prompt "A greenhouse full of orchids, soft morning light"
```

### Convert an LTX video fine-tune

```bash
ziv-model model -i ltx-finetune.safetensors --name my-video-model --model-type ltx-2.3 --quantize 4 --distill
```

For `ltx-2.3` and `ltx-2.5`, `--quantize 4` or `8` is required: it is the only level stored, and no extra copy is made. `--distill` merges Lightricks' distilled LoRA into a plain-dev checkpoint, so it runs at 8 steps. It is for a plain-dev checkpoint. Run it with `ziv-video -m my-video-model`.

### Convert a MiniMax H3 fine-tune

```bash
ziv-model model -i h3-turbo-merge.safetensors --name my-h3-turbo --model-type h3 --turbo
```

`--turbo` says the checkpoint is already step-distilled, so the turbo LoRA is not merged into it. `h3` takes no `--quantize`: the level is fixed.

### Replace a video model

```bash
ziv-model model -i ltx-finetune-v2.safetensors --name my-video-model --model-type ltx-2.3 --quantize 4 --force
```

An existing video model of the same name is refused. `--force` replaces it once the new one is complete. See [Video fine-tunes](../guides/models.md#video-fine-tunes) for what each type needs.

## Import a LoRA { #import-a-lora }

### Import a LoRA from a file

```bash
ziv-model lora -i style.safetensors --name my-style
```

The file is copied into `loras/` in the data folder as `my-style`. Use it with `--lora my-style` or `--lora my-style:0.7`. Without `--name`, the file keeps its own name.

### Download a LoRA from Hugging Face

```bash
ziv-model lora --hf user/lora-repo --name my-lora
ziv-model lora --hf user/lora-repo --file model.safetensors --name my-lora
```

`--file` names the file to take when the repo holds several `.safetensors` files. `--hf` and `-i` can't be combined. A repo can't be named in `--lora` directly: import it first.

## List what you can use

```bash
ziv-model list
ziv-model list --loras
```

The list has up to four parts:

- **Models:** the image models in `models/` in the data folder: the ones you converted, and the quantized copies (`my-klein@q8`) the app stores there.
- **Video Models:** the video models there that you converted.
- **LoRAs:** the files in `loras/`, with their sizes.
- **Model Aliases:** the built-in model names such as `klein4b`, with what each stands for on each platform. They are listed whether you have downloaded them or not, so the list does not tell you what is on disk. The Web UI's Models page does.

`--models` shows only the models and `--loras` only the LoRAs; the aliases are listed either way.

## Flags

Every flag, with its default and limits, is in the [command reference](reference.md#ziv-model-model).

| Flag | Does | In the Web UI |
|---|---|---|
| `model -i` | The checkpoint to convert | **Checkpoint file** |
| `--name` | The name of the model or LoRA | **Name** |
| `--model-type` | `zimage` (default), `flux2-klein-4b`, `flux2-klein-9b`, `krea2-turbo`, `ltx-2.3`, `ltx-2.5` or `h3` | **Model type** |
| `--base-model` | The base repo, for `zimage` | **Base model (optional)** |
| `--copy` | Copy the base model's files instead of linking them | **Copy instead of moving** |
| `--quantize 4`, `--quantize 8` | Also store a quantized copy; required for LTX | **Quantized copy**, **Quantize level** |
| `--distill` | `ltx-2.3`, `ltx-2.5`: merge Lightricks' distilled LoRA | **Distillation** |
| `--turbo` | `h3`: the checkpoint is already step-distilled | **Already step-distilled** |
| `--force` | Replace an existing video model | |
| `lora -i` | A local LoRA file | **Import a local LoRA** |
| `lora --hf`, `--file` | A LoRA repo on Hugging Face, and the file in it | **Download a LoRA from Hugging Face** |
| `list --models`, `list --loras` | Show only models, or only LoRAs | |
