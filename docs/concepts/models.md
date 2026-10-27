# Models

A model is what turns a prompt into an image or video. The app has built-in models that download on first use, and you can add your own.

## Image models

| Model | Alias | Download | Steps | Negative prompt | License |
|---|---|---|---|---|---|
| FLUX.2 Klein 4B | `klein4b` | 16 GB | 4 | No | See its [model page](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) |
| FLUX.2 Klein 9B | `klein9b` | 35 GB | 4 | No | See its [model page](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) |
| Z-Image Turbo | `zit` | 33 GB | 10 | Yes | See its [model page](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) |
| Krea 2 Turbo | `krea2` | 36 GB | 8 | No | [Krea 2 Community License](https://huggingface.co/krea/Krea-2-Turbo) |
| Ideogram 4 Instant | `ideo` | 28 GB (macOS), 24 GB (Windows, Linux) | 8 | No | Ideogram's non-commercial license |
| MOSS-Image Turbo | `moss` | 28 GB (Windows, Linux) | 4 | No | Apache-2.0 |
| MOSS-Image SFT | `moss-sft` | 27 GB (Windows, Linux) | 50 | Yes | Apache-2.0 |

Klein 4B is the lightest and the first-run pick on machines with less than 32 GB of memory; Klein 9B is the pick from 32 GB.

The download is the one-time size. Memory needs depend on your machine and the [quant](quantization.md); the Web UI's badges show what fits. See [Memory](memory.md).

All built-in models download without a Hugging Face account. Where the original repository needs a login, the alias points to an ungated mirror whose weights match the original.

### FLUX.2 Klein

`klein4b` and `klein9b` are distilled to 4 steps. They don't use a negative prompt.

### Z-Image Turbo

`zit` uses the `beta` scheduler on every platform, so a prompt and seed look alike on macOS, Windows and Linux. `--scheduler` or the Web UI's **Scheduler** picks another.

### Krea 2 Turbo

`krea2` defaults to 8 steps and guidance 1.0, which turns classifier-free guidance off. Guidance above 1.0 turns it on, and each step then takes about twice as long. It samples with its own schedule, so it has no scheduler setting, and it doesn't use a negative prompt. Only Krea 2 Turbo and its fine-tunes are supported, not Krea 2 Raw.

The weights are large, about 26 GB for the transformer plus a 9 GB text encoder, so run it [quantized](quantization.md) on smaller machines:

=== "macOS"

    Use `q8` or `q4`. The text encoder isn't quantized, so it adds about 9 GB at any level.

=== "Windows and Linux"

    `q8` needs about 18 GB of system memory, and `q4` fits 10–12 GB GPUs. Unquantized, Krea 2 needs about 35 GB of system memory and is very slow on machines with less.

### Ideogram 4 Instant

`ideo` is Ideogram 4 Instant, fal's distilled Ideogram 4. It makes an image in 8 steps, and its guidance is built in, so a guidance setting is ignored.

- **Sizes** must be 256–2048 pixels, in steps of 16. Presets beyond that, such as `xl` at `16:9`, are refused before the model loads.
- **No reference image or upscale while generating**, and no negative prompt. The [plain upscale](upscaling.md#plain-upscale) of a finished image works.
- **LoRAs** work. Ones made for Ideogram 4 load onto Instant, but they were trained on the base model, so check the results. On Windows and Linux a LoRA adds about 1 GB of GPU memory.

**Captions.** Ideogram 4 reads structured JSON captions. A plain prompt is wrapped into one for you, but such a sparse caption often comes out as a flat cut-out on a checkerboard background, especially for posters and product shots. For a fuller caption, let the [prompt enhancer](prompt-enhancer.md#json-captions-ideogram-4) write one (Smart writes better ones than Fast), or write your own with `--json-prompt` or the Web UI's JSON mode.

**The grey frame.** Ideogram 4 sometimes returns a grey "Image blocked by safety filter" frame for harmless prompts. The app nudges the first denoising step slightly (a first-step sigma of `1.004`), which makes this rarer without changing your prompt, seed or size. If a harmless prompt is still blocked, try **First sigma** `1.005` or `1.006`. This is not a way around the filter: prompts it actually objects to are still blocked.

**The first job builds the model** in `~/.ziv/models/ideo`. It downloads Instant's 18.6 GB transformer and converts it for your platform (FP8 on macOS, BF16 on Windows and Linux), takes the text encoder, VAE and tokenizer from an ungated mirror of the base model, and then deletes the download. An interrupted download resumes on the next run. The Web UI shows the downloads on the job card, then **Build model**.

??? info "Memory and speed"

    `ideo` takes `q4` (NF4 on Windows and Linux) and `q8` (FP8), saved once as `ideo@q4` and `ideo@q8`. On macOS the unquantized model is the FP8 build.

    - **macOS:** at 1024×1024, unquantized and `q8` peak at about 33 GB and `q4` at about 25 GB, mostly briefly while the image is decoded at the end. On an M5 all three take about 15 s per step at that size; `q4` is faster at smaller sizes.
    - **Windows and Linux:** the transformer streams to the GPU block by block. On a 10 GB GPU with 32 GB of system memory, `q4` and `q8` both take about 3.5 s per step at 1024×1024, about a minute per image (`q8` peaks at about 17 GB of system memory, `q4` at 14 GB). Unquantized it takes about 15 s per step and peaks at about 26 GB, so pick `q4` or `q8`.

    Every source is pinned to a checked revision in the `ideogram4_instant` section of the [configuration](configuration.md), and the folder is rebuilt when a pin changes. The model is under Ideogram's non-commercial license, which applies as soon as you download or use it: see `LICENSE.md` in the folder.

### MOSS-Image

`moss` and `moss-sft` are MOSS-Image from OpenMOSS, an image model with a 5-billion-parameter transformer and a Qwen3.5-based text encoder, released under the Apache-2.0 license. It renders text well, in English and in Chinese, and follows long, detailed prompts.

=== "Windows and Linux"

    Two models share one set of weights:

    | | `moss` (Turbo) | `moss-sft` |
    |---|---|---|
    | What it is | MOSS-Image SFT with a community Turbo LoRA distilled from it | MOSS-Image SFT as released |
    | Steps | 4 by default | 50 |
    | Guidance | Built in; a guidance setting is ignored with a warning | 3.0 |
    | Negative prompt | No | Yes |
    | Speed | About 13 s per image at 1024×1024 on a 10 GB GPU, quantized | About 2 minutes at 50 steps |

    Start with `moss`. Pick `moss-sft` when you want a negative prompt or the model as released. Both run on a 10 GB card at every level; run them [quantized](quantization.md), as unquantized Turbo takes about a minute per image. The figures are in [Memory](memory.md#moss-image).

=== "macOS"

    MOSS-Image isn't available on macOS yet.

- **Sizes** are multiples of 16, from about 0.25 to 4 megapixels. The model is made for about 1–2 megapixels; above that, Turbo's 4 steps look softer, so use 6–8 steps.
- **Text.** Put the words you want in the image in quotes in the prompt.
- **Reference image, upscale, LoRAs and live previews** work. LoRAs must be MOSS-Image LoRAs: see [LoRAs](loras.md). The Turbo LoRA stays on, and a LoRA of your own is added on top.
- **No JSON captions.** MOSS-Image takes plain prompts, so `--json-prompt` and the enhancer's JSON format don't apply.

**The first job builds the model** in `~/.ziv/models/moss` or `~/.ziv/models/moss-sft` from the shared download, which takes a moment. The community Turbo LoRA comes from `akhaliq/MOSS-Image-SFT-4-steps`; its pin is in the `moss_image` section of the [configuration](configuration.md).

## Video models

| Model | Alias | Download | Needs | Can do |
|---|---|---|---|---|
| LTX-2.3 4-bit | `ltx-4` | 28.6 GB (macOS) | The least memory. About 20 GB of system memory on Windows and Linux. | Text or image to video, Extend, upscale |
| LTX-2.3 8-bit | `ltx-8` | 37.8 GB (macOS) | Closer to the full model. About 27 GB of system memory on Windows and Linux; a Mac with 32 GB or more. | The same |
| LTX-2.5 4-bit | `ltx25-4` | 71 GB (Windows, Linux), 28.8 GB (macOS) | The newer LTX, with the memory need of `ltx-4`: about 21 GB of system memory on Windows and Linux. | The same |
| MiniMax H3 | `h3` | 98 GB (Windows, Linux), 107 GB (macOS) | Much larger and slower than LTX. About 18 GB of free system memory on Windows and Linux; a Mac with 32 GB. | 4 or 8 steps. No upscale on macOS |

All of them make video with sound, from a prompt or a starting image (with an optional [last frame](video.md#first-and-last-frame)), and can [extend](video.md#extending-a-video) a clip. The first run of each downloads it and, on Windows and Linux, prepares it for your GPU, which takes a while. See [Video](video.md).

You can also convert an [LTX-2.3 or LTX-2.5 fine-tune](../cli/models.md#ltx-23-and-ltx-25-video-fine-tunes) or a [MiniMax H3 fine-tune](../cli/models.md#minimax-h3-video-fine-tunes) into a model you pick by name.

On macOS, `ltx-4` and `ltx-8` download only the files the app loads, at fixed revisions, from ungated repositories, so no Hugging Face account is needed. Each figure includes the 8.1 GB Gemma 3 text encoder, which the two share, and nothing is prepared afterwards.

### LTX-2.5

`ltx25-4` is LTX-2.5, the newer LTX model, at 4 bits. It fits the same machines as `ltx-4`: a Mac with 32 GB of memory, or a 10 GB GPU with 30 GB of system memory. It downloads without a Hugging Face account.

=== "macOS"

    The download is about 28.8 GB and holds the model with its own Gemma 4 text encoder; nothing is prepared afterwards. A 5 s clip at `s` with **Low memory** is estimated at about 15 GiB.

=== "Windows and Linux"

    The download is about 71 GB. The first run prepares a 4-bit transformer and an FP8 text encoder from it, about 22 GiB more on disk, and later runs reuse them. As a typical figure, a 5 s clip with sound at 1280×704, upscaled 2×, took about 200 s on an RTX 3080 10 GB with 30 GB of system memory, the same as `ltx-4` on that machine.

- **Sampler.** Text and image to video use LTX-2.5's own distilled sampler, which adds fresh noise at every step; a repeated seed still gives the same video. **Extend** uses the plain deterministic sampler, as `ltx-4` does.
- **No 8-bit level** for `ltx25-4` itself. LTX-2.5 fine-tunes convert at q4 or q8 and run with LTX-2.5's settings.

### MiniMax H3

`h3` is MiniMax H3, a video model that is stronger on dynamic scenes than LTX and makes stereo sound with the picture. It is far larger, so it is slower and its download is bigger. It downloads without a Hugging Face account.

- **Steps:** 4 or 8, with 8 the default. Four steps take about a quarter less time and give the same scene, slightly softer; fast motion can blur.
- **Length:** clips are 17n+5 frames long (22, 39 … 124, 141 …), and a count in between moves up to the next. The default is 5 s (124 frames at 24 fps). The model was trained on 4 to 15 s clips; outside that range quality isn't guaranteed.
- **Sizes:** the same presets as LTX, in multiples of 32 (64 with upscaling). `xl` is above the size the model was trained on.
- **Extend** works from 24 fps clips whose size is a multiple of 32. See [Extending a video](video.md#extending-a-video).
- **LoRAs** must be MiniMax H3 LoRAs. See [LoRAs](loras.md).
- **Some H3 checkpoints don't follow the start image.** A few community checkpoints ignore the start image in image to video, first and last frame and Extend. If a converted model does, use stock `h3` for those modes.

=== "macOS"

    MiniMax H3 on macOS does everything above except upscale. It is built on a port to MLX. Image to video, first and last frame and Extend run in stages, with one component in memory at a time, and their figures were measured on a Mac with 32 GB of memory; the other figures are estimates.

    - **Download and first use:** about 107 GB. The first run converts the text encoder once into a 4-bit copy of about 15 GB (`h3@mlx-te4`).
    - **Memory and time:** a 5 s clip at `s` is estimated at about 15 GiB and 23 minutes with 8 steps, 14 minutes with 4. At `l` it is about 18 GiB and 77 minutes. The image modes need about the same as text to video: a 3 s image-to-video clip at `s` with 8 steps took about 10 minutes and peaked at about 14 GiB with no swapping, and is estimated at about 14 minutes. First and last frame and Extend peak at about the same, and a LoRA adds about 1.5 GiB. A Mac with 32 GB is the target.
    - **Extend** continues from the source's last frame alone, so the seam is weaker than on Windows and Linux.
    - **No upscale or review.** Image to video, first and last frame and Extend show a step counter and stop at a step boundary; text to video shows its phases but no step counter, and **Next** and **Quit** take effect when the clip finishes ([Controls while a video runs](video.md#controls-while-a-video-runs)).
    - **A local folder** given to `-m` can carry its own INT8 transformer in the MLX port's layout. A folder with a bf16 transformer is refused; [convert](../cli/models.md#minimax-h3-video-fine-tunes) a fine-tune instead.

=== "Windows and Linux"

    - **Download and first use:** about 98 GB. The first run builds a 4-bit copy of the transformer, about 12 GB, as `h3@q4`; it took about 4 minutes, and later runs reuse it. A fine-tune gets a copy of its own, named after it.
    - **Memory:** a 5 s clip at 864×480 uses about 6.3 GiB of GPU memory and about 17 GiB of system memory; plan for about 18 GB free. 1280×736 clips and 14 s clips at 864×480 fit a 10 GB GPU.
    - **Upscale:** a 2× upscale renders at half size, enlarges the result in latent space with H3's own upscaler (0.69 GB, downloaded on first use) and refines it at full size in 2 more steps. The clip keeps the sound of the half-size render and looks like a native full-size render.
    - **Time:** the CLI prints an estimate when a job starts, and the Workspace shows one under **Duration (s)**, such as *About 5 min (estimated)*. On an RTX 3080 10 GB, a 5 s clip with sound:

    | Size (16:9) | 8 steps | 4 steps |
    |---|---|---|
    | `s` 768×448 | about 4 min | about 3 min |
    | `m` 1024×576 | about 6 min | about 4 min |
    | `l` 1280×704 | about 9 min | about 6 min |

    Longer clips take longer: 10 s at 864×480 took about 9 minutes, 14 s about 13. An upscaled `l` clip is estimated at about 5½ minutes. The estimate leaves out the one-time preparation.

MiniMax H3 is released under the MiniMax H3 Community License, published with the model on Hugging Face (`MiniMaxAI/MiniMax-H3`). It restricts use in some regions and above an annual revenue limit, asks commercial products that show H3 to name it in their interface, and counts LoRAs as derivatives of the model. Read it before you use the model or its output.

## Your own models

- **Convert a checkpoint.** A `.safetensors` fine-tune of Z-Image, FLUX.2 Klein, Krea 2, LTX-2.3, LTX-2.5 or MiniMax H3 becomes a model in `~/.ziv/models/`. See [Manage models](../cli/models.md#convert-a-checkpoint) or the [Models page](../web-ui/models.md#convert-a-checkpoint).
- **Copy a model folder.** A model in diffusers format can go straight into `~/.ziv/models/<name>/`.
- **Use a Hugging Face repo.** `-m Tongyi-MAI/Z-Image-Turbo` downloads it on first use.
- **Add an alias.** Give a repo a short name in your [configuration](configuration.md):

    ```yaml title="~/.ziv/config.yaml"
    model_aliases:
      mymodel: "my-org/my-model"
    ```

## How a name is found

When you pass `-m <name>`, or pick a model in the Web UI, the app looks in this order:

1. **A local path**, such as `./my-model` or `~/models/my-model`, or **a Hugging Face repo**, such as `Tongyi-MAI/Z-Image-Turbo`, is used as is. A repo downloads on first use.
2. **A folder** `~/.ziv/models/<name>/` is used if it exists.
3. **An alias**, built in or your own.
4. **Otherwise** the name is used as given, and the job stops if no model is found under it.

A folder in `~/.ziv/models/` wins over an alias of the same name. `ziv-model list` shows every model and alias.

For video, a name is accepted when it is `ltx-4`, `ltx-8`, `ltx25-4` or `h3`, a converted LTX or MiniMax H3 model, a known LTX repo (`dgrauet/ltx*`, `Lightricks/LTX-Video*`, `unsloth/LTX-2.5*`) or MiniMax H3 repo (`MiniMaxAI/MiniMax-H3*`, `unsloth/MiniMax-H3*`), a folder laid out like a MiniMax H3 repository, or a local path containing `ltx`.

## Names in the Web UI

The Web UI shows a model by its name, such as FLUX.2 Klein 9B, and the Models page lists the alias under it. The alias is what you use with `-m`, in prompt files and in the configuration.
