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

# Convert an LTX-2.3 video fine-tune at q4, merging Lightricks' distilled LoRA
ziv-model model -i ltx-finetune.safetensors --name my-video-model --model-type ltx-2.3 --quantize 4 --distill

# Convert an LTX-2.5 video fine-tune at q4
ziv-model model -i ltx25-finetune.safetensors --name my-ltx25 --model-type ltx-2.5 --quantize 4

# Convert a MiniMax H3 fine-tune (the turbo LoRA is merged in), or one that is already step-distilled
ziv-model model -i h3-finetune.safetensors --name my-h3 --model-type h3
ziv-model model -i h3-turbo-merge.safetensors --name my-h3-turbo --model-type h3 --turbo
```

### `model` Flags

| Flag | Default | Description |
|---|---|---|
| `-i`, `--input` | *(required)* | Path to `.safetensors` checkpoint |
| `--name` | input filename | Custom model folder name |
| `--model-type` | `zimage` | Model type: `zimage`, `flux2-klein-4b`, `flux2-klein-9b`, `krea2-turbo`, `ltx-2.3`, `ltx-2.5`, `h3` |
| `--base-model` | `Tongyi-MAI/Z-Image-Turbo` | Base HF repo (only for zimage type) |
| `--copy` | off | Copy files instead of symlinking |
| `--quantize` | off | Also save a quantized copy as `<name>@q4` or `<name>@q8` (`4` or `8`); see [Stored quants](image.md#stored-quants). For `ltx-2.3` and `ltx-2.5` it is required and is the only level stored; `h3` takes none |
| `--distill` | off | `ltx-2.3` and `ltx-2.5` only: merge Lightricks' distilled LoRA into a plain-dev checkpoint |
| `--turbo` | off | `h3` only: the checkpoint is already step-distilled, so the turbo LoRA is not merged |
| `--force` | off | Video types only: replace an existing model of that name, once the new one is complete |

Checkpoints can be bfloat16, float16, float32, or FP8. Scaled FP8 checkpoints (for example ComfyUI FP8 exports, which store a `weight_scale` next to each weight) are dequantized to bfloat16 during conversion. For the image types, block-wise or other quantized formats (such as NVFP4) are not supported. The video types (`ltx-2.3`, `ltx-2.5`, `h3`) also read ComfyUI's other quantized formats: INT8 (with or without ConvRot), INT4 W4A4 and W4A8, INT6 W6A8, NVFP4 and MXFP8. They are decoded back to bfloat16, rotation included, before the model's own quantization; GGUF, SVDQuant and torchao files are refused.

`krea2-turbo` converts a fine-tune of Krea 2 Turbo. Native Krea 2 and ComfyUI checkpoints are both accepted, and a bundled text encoder or VAE is ignored; the text encoder and VAE come from unsloth's ungated mirror `unsloth/Krea-2-Turbo`.

### LTX-2.3 and LTX-2.5 video fine-tunes

`ltx-2.3` and `ltx-2.5` convert a single-file LTX checkpoint (a community fine-tune or merge) into a video model you pick by its name, like `ltx-4`, `ltx-8` and `ltx25-4`. They run on macOS (Apple Silicon) and on Windows and Linux with an NVIDIA GPU. The model is converted for the platform it is converted on, so convert the checkpoint again on each machine. The version is read from the checkpoint (LTX-2.5's video feed-forward has no bias), and a checkpoint of the other version is refused with the type to pick. A converted LTX-2.5 model runs with LTX-2.5's settings, as `ltx25-4` does.

- **One quantize level.** `--quantize 4` or `--quantize 8` is required, and that level is the model: no unquantized copy is kept. q4 needs the least memory (as `ltx-4`); q8 needs more (as `ltx-8`: about 7 GB more system memory on Windows/Linux, and a Mac with 32 GB or more). To have both, convert twice under two names.
- **Inputs.** Lightricks' full checkpoint layout (with the bundled VAEs and vocoder, which are ignored) and transformer-only exports are accepted, in bfloat16 or in any of the quantized formats above. Checkpoints in diffusers' layout, LTX-2.0 checkpoints, LoRAs and torchao repacks are refused. A checkpoint whose text connectors differ from stock's is converted with them on macOS, and refused on Windows and Linux. Quantized connectors, or ones cast to FP8 or INT8, can't be compared with stock's: macOS keeps them, and Windows and Linux use stock's, as most fine-tunes don't train them. On macOS a connector group the checkpoint lacks (transformer-only exports often leave out `text_embedding_projection`) is taken from stock's.
- **Dev or distilled.** The app runs LTX at 8 steps, which needs a step-distilled model. Many community merges are dev and expect Lightricks' distilled LoRA beside them; converted as they are, they render ghosted and noisy. The converter checks this offline and says what it found (a checkpoint whose checked tensors are quantized reads as "can't tell"). For a plain-dev checkpoint, add `--distill`: it merges Lightricks' distilled LoRA of that version (7.6 GB for LTX-2.3, 8.9 GB for LTX-2.5, downloaded once into the Hugging Face cache) into the weights before they are quantized. It is never merged without `--distill`. The LTX-2.5 LoRA comes from an ungated copy whose file is byte-identical to Lightricks' gated original.
- **Names.** A name that already exists is refused (`--force` replaces that model once the new one is complete), and so are alias names such as `ltx-4`. The name doesn't need to contain `ltx`.
- **Self-contained.** On macOS the model folder holds its own transformer (and text connectors, when they differ from stock's), and links the other components from the MLX folder of its version: the VAEs, vocoder and upsamplers from `dgrauet/ltx-2.3-mlx-q4` or `-q8` for LTX-2.3, and those plus the Gemma 4 text encoder and duration head from the `ltx25-4` folder for LTX-2.5. On Windows and Linux it holds its own transformer and text encoder, and links the other components from the version's diffusers repository. Either way the checkpoint can be moved or deleted after converting. An interrupted conversion leaves nothing behind under the model's name.

Converting takes a few minutes and holds one tensor at a time. On Windows and Linux it needs about 5 GB of system memory, and the text encoder is shared with the stock models of that version when one of them has been used. On macOS the components linked from the MLX folder are downloaded first if the stock model hasn't been used at that level (about 10 GB, 15 GB for LTX-2.5). On the Web UI's Models page, pick **LTX-2.3 (video)** or **LTX-2.5 (video)** as the model type: the form reads the checkpoint, shows its version and storage, preselects the level that fits this machine (q4 for LTX-2.5, whose q8 headroom isn't measured yet) and, for a plain-dev checkpoint, **Distill**, and shows the conversion's progress. The model's row then shows the level it was converted at.

### MiniMax H3 video fine-tunes

`h3` converts a MiniMax H3 fine-tune into a video model you pick by its name, like `h3`. It runs like stock H3, at 4 or 8 steps, with LoRAs, image to video, first and last frame, Extend and (on Windows and Linux) upscale. It runs on macOS and on Windows and Linux with an NVIDIA GPU, converted for the machine it is converted on.

- **Inputs.** ComfyUI single files (bare or with a `model.diffusion_model.` prefix), MiniMax's own sharded transformer (pick any shard: the set its index lists is read), and diffusers-layout files, in bfloat16 or any of the quantized formats above. **Pruned** checkpoints, whose AdaLN reads a small timestep curve instead of the time embedding (as ComfyUI's `*_pruned_*` files do), are converted too: the curve is mapped back onto stock's time embedding, whose 60 MB are read from the stock repository while converting. Refused: LoRAs (import them on the LoRAs tab), GGUF, PDD and Ref2VA checkpoints, and anything that is not stock-sized FL2VA.
- **Turbo.** H3 runs at 4 or 8 steps through lightx2v's turbo LoRA, which stock gets merged in. The converter merges it too, unless you pass `--turbo` (the Web UI's **Already step-distilled**): use it for a checkpoint that already runs at 4 or 8 steps on its own, such as a turbo, lightning or fused-turbo merge. Nothing in a file tells this reliably, so the choice is yours; the form shows what the file's name or metadata hints, and never ticks the box. A wrong choice renders blurry or burnt clips: convert again with the other choice. Every converted model runs at the configured shift and step counts.
- **Storage.** On Windows and Linux the folder is a 4-bit copy in the layout of stock's `h3@q4`, with the AdaLN folded into tables for the configured schedules. If the schedules or the turbo strength in the config change, or an update changes bitsandbytes' 4-bit format, loading it says so: convert again. It needs stock `h3`'s components but not the turbo LoRA. On macOS it is an 8-bit transformer in the layout the MLX port loads (plain rounding, without the shipped transformer's calibration), with the stock components linked beside it. It reads its text encoder from the stock download and shares stock's text-encoder copy (`h3@mlx-te4`), so switching between it and `h3` prepares nothing again.
- **Names, self-containment and memory** are as for LTX: a name that exists or is an alias is refused (`--force` replaces), the checkpoint can be removed afterwards on macOS, and a converted model needs the same memory as stock `h3`. Converting holds one tensor at a time (about 6 GB of system memory on Windows and Linux) and takes about as long as building `h3@q4`. Each converted model takes as much disk as stock's copy: about 12 GB on Windows and Linux, 38 GB on macOS. On Windows and Linux, keep the checkpoint: the folder holds a 4-bit copy built for the current H3 settings (schedules, turbo strength, 4-bit format), and an update that changes them asks you to convert again.

On the Web UI's Models page, pick **MiniMax H3 (video)**: the form reads the checkpoint, shows its layout and storage, and has the **Already step-distilled** box.

> **Ideogram 4** is used via the `ideo` alias: the first job with it builds Ideogram 4 Instant in `~/.ziv/models/ideo` (see [Ideogram 4](image.md#ideogram-4)). It is not a `--model-type` option.

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

The Web UI Workspace lists the LoRAs made for the selected model first and folds all others, including ones whose model can't be detected, under **Other models**; it reads which model a LoRA was made for from the file's header the first time it sees the file.

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

The Web UI's Models page lists the same models and LoRAs and can convert checkpoints and import LoRAs. Each alias row shows the model's name (for example FLUX.2 Klein 9B) with the alias under it. The **Size** column shows the disk each model uses: an alias's HuggingFace download (with the extra repos LTX and MiniMax H3 download), or a converted model's own folder without the base-model files it links to. Quantized copies count with their model when they don't have a row of their own. Two aliases that load the same download, such as `ltx-4` and `ltx-8` on Windows/Linux, both show it.

Rows whose files are not fully downloaded have a **Download** button, so a model is ready before the first job that uses it (without it, the first job downloads the model and shows the download on its card). The prompt enhancer rows and the SeedVR2 row have one too.

- **Confirm.** The button first shows the download's size, split into parts for models that download several (such as LTX video and MiniMax H3), and the free space on the drive it downloads to. Both are estimates: the free-space check covers only the main download drive and can count files that other downloads share twice. **Download anyway** replaces the button when the space looks short.
- **One at a time, beside your jobs.** Downloads run one after another and don't take a place in the job queue, so generation jobs start while a download runs. Later downloads wait as **Queued**; **Cancel** on a row stops a running download or removes a queued one. Cancelling keeps the finished files and removes the half-written one, so the next Download continues with the rest.
- **Jobs share the download.** A job that needs a model being downloaded waits for it and shows its progress on the job card. Cancelling that job leaves the download running. The Workspace model picker and its status badge show a running download's progress (for example "Downloading 42%") or "Download queued" instead of "not downloaded", and the picker updates when the download finishes.
- **Ideogram 4 Instant and SeedVR2.** The **Download** button for `ideo` fetches its source files; the first job then prepares the model folder from them (see [Ideogram 4 Instant](image.md#ideogram-4)). On a Mac, the SeedVR2 row downloads the model, and the first plain upscale makes its 8-bit copy and removes the download.
- **No connection.** If Hugging Face can't be reached, the size can't be read and the download can't start. A gated or missing repository shows one line saying what to do: the model's page to accept its license on, and `hf auth login` when no [Hugging Face token](../getting-started.md#hugging-face-token) is found.

Each row also has a delete button, which asks for confirmation first and is unavailable while generation jobs are running or queued, or while the row's model is downloading (wait for the download or cancel it):

- **Converted models** (in `~/.ziv/models/`): deletes the model folder. Base-model files it links to in the HuggingFace cache are kept.
- **HuggingFace models** (aliases such as `klein9b`): deletes the model's download from the HuggingFace cache. The alias stays, and the model downloads again the next time you use it. If converted models link to files in that download, the confirmation lists them, because they stop working until it is downloaded again. The button appears as soon as the cache holds any of the model's files, so an interrupted or partial download can be cleared too.
- **MiniMax H3** (`h3`): deletes its download together with the repositories it downloads beside it (the text encoder and turbo LoRA on Windows and Linux; the INT8 transformer and turbo LoRA on a Mac), and its prepared copy: `h3@q4` on Windows and Linux, `h3@mlx-te4` on a Mac. The next use downloads and prepares them again. A converted H3 model is a folder of its own, removed with that model; deleting `h3` leaves it alone, though it needs `h3`'s components (and on a Mac its text-encoder copy) to run, which the next use downloads again. The button appears as soon as the cache holds any of the files.
- **LoRAs** (in `~/.ziv/loras/`): deletes the `.safetensors` file.
- **Prompt enhancer models**: the **Prompt enhancer** table has a row for **Fast**, **Smart** and, when a custom enhancer model is set on the Config page, **Custom**, with its Hugging Face repository and its disk use (or the size of the first-use download). Delete removes that model's download from the HuggingFace cache, including a partly finished one; it downloads again the next time you enhance a prompt with it. A custom model in a local folder is never deleted. Delete is unavailable while a prompt enhancement is running.
- **Other downloads**: models in the HuggingFace cache that no alias, enhancer model or converted model uses, for example a model an alias pointed at before an update (such as the old Klein 9B download), or a file a detection check fetched. The size is what deleting it frees: files it shares with another download stay. The HuggingFace cache is shared with other apps on your computer, so another app may download a deleted repo again. **Unused cache files** appears when the cache holds files no download links to any more, which deletes outside the app (or before this version) leave behind; deleting them frees that space without touching any download.
- **SeedVR2 upscale model**: the upscale model, used for images and, on Windows and Linux, for [videos](video.md#upscale-an-existing-video-web-ui), has a row of its own with its download size: `seedvr2-7b@q8` or `seedvr2-3b@q8` on a Mac, depending on its memory, and `seedvr2-7b@fp8` or `seedvr2-3b@fp8` on Windows and Linux, depending on system memory. Once the first plain upscale has downloaded it, the row shows its disk use and a delete button that removes its folder in the models folder: the stored 8-bit copy on a Mac, the downloaded 8-bit files on Windows and Linux. Nothing else is affected, and the next plain upscale downloads the model again. See [Upscaling an existing image](image.md#upscaling-an-existing-image-web-ui).
