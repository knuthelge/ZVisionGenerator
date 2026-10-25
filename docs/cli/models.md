# Manage models and LoRAs

`ziv-model` has three subcommands:

| Subcommand | Does |
|---|---|
| `model` | Converts a single-file checkpoint into a model you can pick by name |
| `lora` | Imports a LoRA from a file or from Hugging Face |
| `list` | Lists installed models, LoRAs and aliases |

The Web UI's [Models page](../web-ui/models.md) does the same, and also downloads and deletes models.

## Convert a checkpoint

Community fine-tunes often come as one `.safetensors` file. Converting puts it in `~/.ziv/models/<name>/`, where `-m <name>` finds it.

```bash
# A Z-Image fine-tune
ziv-model model -i checkpoint.safetensors --name my-model

# A FLUX.2 Klein 4B fine-tune
ziv-model model -i klein-finetune.safetensors --name my-klein --model-type flux2-klein-4b

# Also save a q8 copy, used whenever my-model runs at -q 8
ziv-model model -i checkpoint.safetensors --name my-model --quantize 8

# An LTX-2.3 video fine-tune at q4, merging Lightricks' distilled LoRA
ziv-model model -i ltx-finetune.safetensors --name my-video-model --model-type ltx-2.3 --quantize 4 --distill

# An LTX-2.5 video fine-tune at q4
ziv-model model -i ltx25-finetune.safetensors --name my-ltx25 --model-type ltx-2.5 --quantize 4

# A MiniMax H3 fine-tune (the turbo LoRA is merged in), or one that is already step-distilled
ziv-model model -i h3-finetune.safetensors --name my-h3 --model-type h3
ziv-model model -i h3-turbo-merge.safetensors --name my-h3-turbo --model-type h3 --turbo
```

`--model-type` is `zimage` (the default), `flux2-klein-4b`, `flux2-klein-9b`, `krea2-turbo`, `ltx-2.3`, `ltx-2.5` or `h3`.

An image model's folder holds the fine-tuned weights and links to the base model's other files, so it takes little extra space. `--copy` copies them instead.

### What converts

- Checkpoints in bfloat16, float16, float32 or FP8. Scaled FP8 checkpoints, such as ComfyUI FP8 exports, are converted to bfloat16.
- **Image types:** block-wise and other quantized formats, such as NVFP4, are not supported.
- **Video types** (`ltx-2.3`, `ltx-2.5`, `h3`) also read ComfyUI's other quantized formats: INT8 (with or without ConvRot), INT4 W4A4 and W4A8, INT6 W6A8, NVFP4 and MXFP8. They are decoded back before converting. GGUF, SVDQuant and torchao files are refused.
- `krea2-turbo` takes native Krea 2 and ComfyUI checkpoints of a Krea 2 Turbo fine-tune. A bundled text encoder or VAE is ignored; they come from `unsloth/Krea-2-Turbo`.
- Ideogram 4 isn't a model type: the `ideo` alias builds it on first use.

### Video fine-tunes, in general

A video model is converted for the machine it is converted on, so convert the checkpoint again on each machine.

- **Names.** An existing name is refused unless you add `--force`, which replaces that model once the new one is complete. Alias names such as `ltx-4` are refused. The name doesn't need to contain `ltx`.
- **Interrupted?** An interrupted conversion leaves nothing behind.
- **A Hugging Face cache file** can be given directly; its name and sibling shards are kept.

### LTX-2.3 and LTX-2.5 video fine-tunes

`ltx-2.3` and `ltx-2.5` convert a single-file LTX checkpoint, a community fine-tune or merge, into a video model you pick by name like `ltx-4` and `ltx25-4`. The version is read from the checkpoint, and a checkpoint of the other version is refused with the type to pick. A converted LTX-2.5 model runs with LTX-2.5's settings.

- **One quantize level.** `--quantize 4` or `--quantize 8` is required, and that level is the model: no unquantized copy is kept. q4 needs the least memory, like `ltx-4`; q8 needs more, like `ltx-8`. To have both, convert twice under two names.
- **Dev or distilled.** The app runs LTX at 8 steps, which needs a step-distilled model. Many community merges are dev and expect Lightricks' distilled LoRA beside them; converted as they are, they come out ghosted and noisy. The converter checks this and says what it found ("can't tell" for a quantized checkpoint). For a plain-dev checkpoint, add `--distill` to merge the distilled LoRA of that version (7.6 GB for LTX-2.3, 8.9 GB for LTX-2.5, downloaded once) into the weights.
- **Inputs.** Lightricks' full checkpoint layout and transformer-only exports are accepted, in bfloat16 or any of the quantized formats above. Checkpoints in diffusers' layout, LTX-2.0 checkpoints, LoRAs and torchao repacks are refused. A checkpoint whose text connectors differ from stock's converts with them on macOS and is refused on Windows and Linux.
- **The checkpoint** can be moved or deleted afterwards.

Converting takes a few minutes. On Windows and Linux it needs about 5 GB of system memory. On macOS, the parts it links from the stock model are downloaded first if you haven't used that level yet (about 10 GB, 15 GB for LTX-2.5).

### MiniMax H3 video fine-tunes

`h3` converts a MiniMax H3 fine-tune into a video model you pick by name. It runs like stock `h3`: 4 or 8 steps, LoRAs, image to video, first and last frame, Extend and, on Windows and Linux, upscale.

- **Inputs.** ComfyUI single files, MiniMax's own sharded transformer (give any shard), and diffusers-layout files, in bfloat16 or any of the quantized formats above. Pruned checkpoints, such as ComfyUI's `*_pruned_*` files, convert too. LoRAs (import them instead), GGUF, PDD and Ref2VA checkpoints are refused.
- **Turbo.** H3 runs at 4 or 8 steps through a turbo LoRA, which the converter merges in. Add `--turbo` (the Web UI's **Already step-distilled**) for a checkpoint that already runs at 4 or 8 steps on its own, such as a turbo or lightning merge. Nothing in a file tells this reliably, so the choice is yours. A wrong choice renders blurry or burnt clips: convert again with the other choice.
- **Storage.** Each converted model takes about as much disk as stock's copy: about 12 GB on Windows and Linux, 38 GB on macOS. It needs stock `h3`'s other parts to run, which the first use downloads.
- **On Windows and Linux, keep the checkpoint.** The folder is a 4-bit copy built for the current H3 settings; if an update changes them, loading it asks you to convert again.

Converting holds one tensor at a time (about 6 GB of system memory on Windows and Linux) and takes about as long as building `h3@q4`.

## Import a LoRA

```bash
# A local file
ziv-model lora -i /path/to/style.safetensors --name my-style

# From Hugging Face
ziv-model lora --hf user/lora-repo --name my-lora

# A specific file from a repository with several
ziv-model lora --hf user/lora-repo --file model.safetensors
```

The LoRA lands in `~/.ziv/loras/`, and `--lora my-style` finds it. A Hugging Face LoRA must be imported before you can use it. See [LoRAs](../concepts/loras.md).

## List what's installed

```bash
ziv-model list            # everything, with aliases
ziv-model list --models   # models only
ziv-model list --loras    # LoRAs only
```
