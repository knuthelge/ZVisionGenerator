# Download, delete and convert models

Models are large, so it helps to know where the disk space goes and how to manage it. The [Models page](../web-ui/models.md) (++g++ ++m++) does the downloading, deleting, converting and importing.

Where the space goes:

- **Models the app downloads** (the built-in ones, the prompt enhancer, SeedVR2, background removal and the audio models) are kept in the Hugging Face cache, not the data folder: `~/.cache/huggingface` by default, shared with other tools. This is the biggest part. Set `HF_HOME` to put it on a larger drive; see [Data folder](../reference/data-folder.md#models-and-the-hugging-face-cache).
- **Models you convert**, stored quantized copies and prepared copies are in `models/` in the data folder (`~/.ziv` by default).
- **LoRAs** are `.safetensors` files in `loras/` in the data folder. Your pictures, videos and audio are in `outputs/`.

The **Size** column of each table on the Models page shows the disk a downloaded row uses. Download sizes are in [Models and licences](../reference/models.md#download-size-and-memory).

## Download a model ahead of time

The first job that needs a model asks to download it and shows the progress on its card. To have it ready before you need it, use the Models page.

1. Press **Download** on a row that isn't fully downloaded.
2. Keep working. Downloads run one at a time, beside your jobs, and don't take a place in the job queue, so you can generate while one runs.
3. **Cancel** stops a download. Finished files are kept, so the next **Download** continues with the rest.

A job that needs a model that is downloading waits for it and shows its progress. Cancelling that job leaves the download running.

Some rows take a second step:

- **Ideogram 4 Instant** (`ideo`): **Download** fetches its source files, and the first job builds the model folder from them.
- **MOSS-Image Turbo** (`moss`) and **MOSS-Image SFT** (`moss-sft`): **Download** fetches the files, and the first image builds the model. The two share the download.
- **SeedVR2**, the model behind the plain upscale: on a Mac, the first plain upscale makes an 8-bit copy and removes the download.

If a download can't start, see [Downloads](../reference/troubleshooting.md#downloads).

## Delete models and free space

Each row has a delete button once any of its files are on disk, so an interrupted download can be cleared too. Its dialog says what goes and what stays: a converted model keeps the base-model files it links to, and a model you delete downloads again the next time you use it. Takes you made stay in the output folder.

**Other downloads** and **Unused cache files** at the bottom of the page hold what no model uses. The cache is shared with other apps, so another app may download an item again.

!!! warning "Delete from the Models page, not by hand"
    Don't delete folders in the Hugging Face cache yourself. Downloads are shared by several models, and converted models link to their base model's files, so removing a folder by hand can break a model that still shows as installed. If you did, **Unused cache files** clears what it left behind.

## Use your own models { #your-own-models }

Besides the built-in models, you can use:

- **A converted fine-tune.** A `.safetensors` checkpoint of Z-Image, FLUX.2 Klein, Krea 2, LTX-2.3, LTX-2.5 or MiniMax H3 becomes a model in `models/`. See [Convert a fine-tune](#convert-a-checkpoint).
- **A model folder.** A model in diffusers format can be copied straight into `models/<name>/` in the data folder.
- **A Hugging Face repository.** `-m Tongyi-MAI/Z-Image-Turbo` downloads it on first use.
- **An alias.** Give a repository a short name under `model_aliases` in [`config.yaml`](../reference/configuration.md); a name the app already uses is taken.

A converted model appears in the Workspace's model picker and in the Models page, and `-m <name>` finds it. How a name is resolved is in [Models and licences](../reference/models.md#how-a-name-is-found).

## Convert a fine-tune { #convert-a-checkpoint }

Community fine-tunes, such as those on Civitai or Hugging Face, often come as one `.safetensors` file. Converting puts it in `models/<name>/`.

1. On the Models page, find **Convert a checkpoint**.
2. Choose the **Checkpoint file**, and give it a **Name**.
3. Pick the **Model type**: `zimage`, `flux2-klein-4b`, `flux2-klein-9b`, `krea2-turbo`, **LTX-2.3 (video)**, **LTX-2.5 (video)** or **MiniMax H3 (video)**.
4. For an image type, you can also save a **Quantized copy** at `q8` or `q4`, used whenever the model runs at that level.
5. Press **Convert checkpoint**. Progress shows while it runs.

For the video types the form reads the checkpoint and preselects what fits, so check its choices against the notes below.

An image model's folder holds the fine-tuned weights and links to the base model's other files, so it takes little extra space. **Copy instead of moving** copies those base files into the folder instead; your checkpoint file stays where it is either way. **Base model (optional)** is for Z-Image only and defaults to `Tongyi-MAI/Z-Image-Turbo`.

An existing name is refused for the video types unless you convert from the command line with `--force`. Alias names such as `ltx-4` are refused. A refused checkpoint says why; see [Models, LoRAs and conversion](../reference/troubleshooting.md#models-loras-and-conversion).

??? note "What converts"

    - Checkpoints in bfloat16, float16, float32 or FP8. Scaled FP8 checkpoints, such as ComfyUI FP8 exports, are converted to bfloat16.
    - **Image types:** block-wise and other quantized formats, such as NVFP4, are not supported.
    - **Video types** (`ltx-2.3`, `ltx-2.5`, `h3`) also read ComfyUI's other quantized formats: INT8 (with or without ConvRot), INT4 W4A4 and W4A8, INT6 W6A8, NVFP4 and MXFP8. They are decoded back before converting. GGUF, SVDQuant and torchao files are refused.
    - **Krea 2 Turbo** takes native Krea 2 and ComfyUI checkpoints of a Krea 2 Turbo fine-tune. A bundled text encoder or VAE is ignored.
    - **Ideogram 4** isn't a model type: the `ideo` alias builds it on first use.

### Video fine-tunes

A video model is converted for the machine it is converted on, so convert the checkpoint again on each machine. An interrupted conversion leaves nothing behind. You can give a checkpoint from the Hugging Face cache directly; its name and sibling shards are kept. Three things to get right:

- **LTX: one quantize level.** `--quantize 4` or `--quantize 8` is required, and that level is the model: no unquantized copy is kept. The Web UI preselects the level for this machine once you pick a checkpoint: `4` needs less memory, `8` is closer to the full model (see [Switch to the smaller model](../concepts/memory.md#switch-to-the-smaller-model)). To have both, convert twice under two names.
- **LTX: dev or distilled.** The app runs LTX at 8 steps, which needs a step-distilled model. A plain-dev checkpoint converted as it is comes out ghosted and noisy, so turn on **Distillation** (`--distill`) to merge Lightricks' distilled LoRA into it. The converter checks and says what it found.
- **H3, Windows and Linux: keep the checkpoint.** The converted folder is a 4-bit copy built for the current H3 settings; if an update changes them, loading it asks you to convert again.

??? note "LTX-2.3 and LTX-2.5"

    ### LTX-2.3 and LTX-2.5 video fine-tunes

    `ltx-2.3` and `ltx-2.5` convert a single-file LTX checkpoint, a community fine-tune or merge, into a video model you pick by name, like `ltx-4` and `ltx25-4`. The version is read from the checkpoint, and a checkpoint of the other version is refused with the type to pick. A converted LTX-2.5 model runs with LTX-2.5's settings.

    - **Inputs.** Lightricks' full checkpoint layout and transformer-only exports are accepted, in bfloat16 or any of the quantized formats above. Checkpoints in diffusers' layout, LTX-2.0 checkpoints, LoRAs and torchao repacks are refused. A checkpoint whose text connectors differ from stock's converts with them on macOS and is refused on Windows and Linux.
    - **The checkpoint** can be moved or deleted afterwards.

??? note "MiniMax H3"

    ### MiniMax H3 video fine-tunes

    `h3` converts a MiniMax H3 fine-tune into a video model you pick by name. It runs like stock `h3`: 4 or 8 steps, LoRAs, image to video, first and last frame, Extend and, on Windows and Linux, upscale. It needs stock `h3`'s other parts to run, which the first use downloads.

    - **Inputs.** ComfyUI single files, MiniMax's own sharded transformer (give any shard), and diffusers-layout files, in bfloat16 or any of the quantized formats above. Pruned checkpoints, such as ComfyUI's `*_pruned_*` files, convert too. LoRAs (import them instead), GGUF, PDD and Ref2VA checkpoints are refused.
    - **Turbo.** H3 runs at 4 or 8 steps through a turbo LoRA, which the converter merges in. Turn on **Already step-distilled** (`--turbo`) for a checkpoint that already runs at 4 or 8 steps on its own, such as a turbo or lightning merge. Nothing in a file tells this reliably, so the choice is yours. A wrong choice renders blurry or burnt clips: convert again with the other choice.
    - An H3 model converted on another system, or for other H3 settings, asks you to convert again on this machine.

## Import a LoRA

A [LoRA](../concepts/loras.md) must be in `loras/` in the data folder before you can use it.

On the Models page:

- **Import a local LoRA** copies a `.safetensors` file into `loras/`. Use it for a LoRA file you downloaded from a site such as Civitai.
- **Download a LoRA from Hugging Face** fetches one from a repository `owner/repository`. Name the **File** when the repository has several. The download shows as a row of the LoRAs table with **Cancel**, like a model download. Without a **Name**, the file keeps its own name.

**Discover LoRAs in the Workspace** finds LoRAs for your model and downloads them for you. See [Discover LoRAs](../concepts/loras.md#discover-loras). The LoRA is then in the Workspace's LoRA picker. A Hugging Face repository can't be named in `--lora` directly: import it first.

## From the command line

`ziv-model` converts checkpoints, imports LoRAs and lists what is installed:

```bash
ziv-model model -i klein-finetune.safetensors --name my-klein --model-type flux2-klein-4b --quantize 8
ziv-model list
```

More recipes and every flag are in [Models and LoRAs](../cli/models.md).
