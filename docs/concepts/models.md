# Models

A model is what turns a prompt into an image, a video or a sound. The app has built-in models that download on first use, and you can add your own.

## Image models

| Model | Alias | Download | Steps | Negative prompt | License |
|---|---|---|---|---|---|
| FLUX.2 Klein 4B | `klein4b` | 16 GB | 4 | No | Apache-2.0 |
| FLUX.2 Klein 9B | `klein9b` | 35 GB | 4 | No | FLUX Non-Commercial License v2.1 |
| Z-Image Turbo | `zit` | 33 GB | 10 | Yes | Apache-2.0 |
| Krea 2 Turbo | `krea2` | 36 GB | 8 | No | Krea 2 Community License Agreement |
| Ideogram 4 Instant | `ideo` | 28 GB (macOS), 24 GB (Windows, Linux) | 8 | No | Ideogram Non-Commercial Model Agreement |
| MOSS-Image Turbo | `moss` | 28 GB | 4 | No | Apache-2.0 |
| MOSS-Image SFT | `moss-sft` | 27 GB | 50 | Yes | Apache-2.0 |

Klein 4B is the lightest and the first-run pick on machines with less than 32 GB of memory; Klein 9B is the pick from 32 GB.

The download is the one-time size. Memory needs depend on your machine and the [quant](quantization.md); the Web UI's badges show what fits. See [Memory](memory.md).

The License column names each model's licence; [Licences](#licences) has the terms and links. All built-in models download without a Hugging Face account. Where the original repository needs a login, the alias points to an ungated mirror whose weights match the original.

### FLUX.2 Klein

`klein4b` and `klein9b` are distilled to 4 steps. They don't use a negative prompt. They also take up to four [references](images.md#references): images of a person, object, place or style the result should keep.

### Z-Image Turbo

`zit` uses the `beta` scheduler on every platform, so a prompt and seed look alike on macOS, Windows and Linux. `--scheduler` or the Web UI's **Scheduler** picks another.

### Krea 2 Turbo

`krea2` defaults to 8 steps and guidance 1.0, which turns classifier-free guidance off. Guidance above 1.0 turns it on, and each step then takes about twice as long. It doesn't use a negative prompt. Only Krea 2 Turbo and its fine-tunes are supported, not Krea 2 Raw.

The weights are large, about 26 GB for the transformer plus a 9 GB text encoder, so run it [quantized](quantization.md) on smaller machines:

=== "macOS"

    Use `q8` or `q4`. The text encoder isn't quantized, so it adds about 9 GB at any level.

=== "Windows and Linux"

    `q8` needs about 18 GB of system memory, and `q4` fits 10–12 GB GPUs. Unquantized, Krea 2 needs about 35 GB of system memory and is very slow on machines with less.

### Ideogram 4 Instant

`ideo` is Ideogram 4 Instant, fal's distilled Ideogram 4. It makes an image in 8 steps, and its guidance is built in, so a guidance setting is ignored.

- **Sizes** must be 256–2048 pixels, in steps of 16. Presets beyond that, such as `xl` at `16:9`, are refused before the model loads.
- **No start image, references or upscale while generating**, and no negative prompt. The [plain upscale](upscaling.md#plain-upscale) of a finished image works.
- **LoRAs** work. Ones made for Ideogram 4 load onto Instant, but they were trained on the base model, so check the results. On Windows and Linux a LoRA adds about 1 GB of GPU memory.

**Captions.** Ideogram 4 reads structured JSON captions. A plain prompt is wrapped into one for you, but such a sparse caption often comes out as a flat cut-out on a checkerboard background, especially for posters and product shots. For a fuller caption, let the [prompt enhancer](prompt-enhancer.md#json-captions-ideogram-4) write one (Smart writes better ones than Fast), or write your own with `--json-prompt` or the Web UI's JSON mode.

**The grey frame.** Ideogram 4 sometimes returns a grey "Image blocked by safety filter" frame for harmless prompts. The app nudges the first denoising step slightly (a first-step sigma of `1.004`), which makes this rarer without changing your prompt, seed or size. If a harmless prompt is still blocked, try **First sigma** `1.005` or `1.006`. This is not a way around the filter: prompts it actually objects to are still blocked.

**The first job builds the model** in `~/.ziv/models/ideo`. It downloads Instant's 18.6 GB transformer and converts it for your platform (FP8 on macOS, BF16 on Windows and Linux), takes the text encoder, VAE and tokenizer from an ungated mirror of the base model, and then deletes the download. An interrupted download resumes on the next run. The Web UI shows the downloads on the job card, then **Build model**.

??? info "Memory and speed"

    `ideo` takes `q4` (NF4 on Windows and Linux) and `q8` (FP8), saved once as `ideo@q4` and `ideo@q8`. On macOS the unquantized model is the FP8 build.

    - **macOS:** at 1024×1024, unquantized and `q8` peak at about 33 GB and `q4` at about 25 GB, mostly briefly while the image is decoded at the end. On an M5 all three take about 15 s per step at that size; `q4` is faster at smaller sizes.
    - **Windows and Linux:** the transformer streams to the GPU block by block. On a 10 GB GPU with 32 GB of system memory, `q4` and `q8` both take about 3.5 s per step at 1024×1024, about a minute per image (`q8` peaks at about 17 GB of system memory, `q4` at 14 GB). Unquantized it takes about 15 s per step and peaks at about 26 GB, so pick `q4` or `q8`.

    Every source is pinned to a checked revision in the `ideogram4_instant` section of the [configuration](configuration.md), and the folder is rebuilt when a pin changes. The model is under the Ideogram Non-Commercial Model Agreement, which applies as soon as you download or use it: see `LICENSE.md` in the folder.

### MOSS-Image

`moss` and `moss-sft` are MOSS-Image from OpenMOSS, an image model with a 5-billion-parameter transformer and a Qwen3.5-based text encoder, released under the Apache-2.0 license. It renders text well, in English and in Chinese, and follows long, detailed prompts.

Two models share one set of weights:

| | `moss` (Turbo) | `moss-sft` |
|---|---|---|
| What it is | MOSS-Image SFT with a community Turbo LoRA distilled from it | MOSS-Image SFT as released |
| Steps | 4 by default | 50 |
| Guidance | Built in; a guidance setting is ignored with a warning | 3.0 |
| Negative prompt | No | Yes |

Start with `moss`. Pick `moss-sft` when you want a negative prompt or the model as released.

=== "Windows and Linux"

    `moss` takes about 13 s per image at 1024×1024 on a 10 GB GPU, quantized; `moss-sft` about 2 minutes at 50 steps. Both run on a 10 GB card at every level; run them [quantized](quantization.md), as unquantized Turbo takes about a minute per image. The figures are in [Memory](memory.md#moss-image).

=== "macOS"

    `moss` takes about 30 s per image at 1024×1024 on a 32 GB Mac, quantized; `moss-sft` about 15 minutes at 50 steps. Both fit a 32 GB Mac at every level.

- **Sizes** are multiples of 16, from about 0.25 to 4 megapixels. The model is made for about 1–2 megapixels; above that, Turbo's 4 steps look softer, so use 6–8 steps.
- **Text.** Put the words you want in the image in quotes in the prompt.
- **Start image, upscale, LoRAs and live previews** work. LoRAs must be MOSS-Image LoRAs: see [LoRAs](loras.md). The Turbo LoRA stays on, and a LoRA of your own is added on top.
- **No JSON captions.** MOSS-Image takes plain prompts, so `--json-prompt` and the enhancer's JSON format don't apply.

**The first job builds the model** in `~/.ziv/models/moss` or `~/.ziv/models/moss-sft` from the shared download, which takes a moment. The community Turbo LoRA comes from `akhaliq/MOSS-Image-SFT-4-steps`; its pin is in the `moss_image` section of the [configuration](configuration.md).

## Video models

| Model | Alias | Download | Needs | Can do | License |
|---|---|---|---|---|---|
| LTX-2.3 4-bit | `ltx-4` | 28.6 GB (macOS) | The least memory. About 20 GB of system memory on Windows and Linux. | Text or image to video, Extend, upscale | LTX-2 Community License Agreement and Gemma Terms of Use |
| LTX-2.3 8-bit | `ltx-8` | 37.8 GB (macOS) | Closer to the full model. About 27 GB of system memory on Windows and Linux; a Mac with 32 GB or more. | The same | LTX-2 Community License Agreement and Gemma Terms of Use |
| LTX-2.5 4-bit | `ltx25-4` | 71 GB (Windows, Linux), 28.8 GB (macOS) | The newer LTX, with the memory need of `ltx-4`: about 21 GB of system memory on Windows and Linux. | The same | LTX-2.x Community License Agreement |
| MiniMax H3 | `h3` | 98 GB (Windows, Linux), 107 GB (macOS) | Much larger and slower than LTX. About 18 GB of free system memory on Windows and Linux; a Mac with 32 GB. | 4 or 8 steps. No upscale on macOS | MiniMax H3 Community License Agreement |

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

MiniMax H3 is released under the MiniMax H3 Community License Agreement, published with the model on Hugging Face (`MiniMaxAI/MiniMax-H3`). It restricts use in some regions and above an annual revenue limit, requires commercial products to show "MiniMax H3" in their interface, and counts LoRAs as derivatives of the model. Read it before you use the model or its output; see [Licences](#licences).

## Audio models

| Model | Alias | Makes | Download | License |
|---|---|---|---|---|
| YuE2 3B | `yue2` | Songs and instrumentals up to 6:00 | about 8 GB | CC BY-NC 4.0 with additional creator permission |
| Higgs TTS 3 4B | `higgs3` | Speech of up to about 10 minutes, in a voice the model picks or one that follows a recording | about 10 GB | Boson Higgs TTS 3 Research and Non-Commercial License and Boson Higgs Audio 2 Community License Agreement |
| Stable Audio 3 Medium | `sa3` | Ambience and sound effects of 1 to 380 s | about 10 GB | Stability AI Community License and Gemma Terms of Use |

The alias is what you pass to `ziv audio -m`. What each kind does, and how long a take can be, is in [Audio](audio.md); what they need is in [Memory](memory.md#audio). All three are restricted in some way, so read their terms in [Licences](#licences) before you use the output.

## Licences

Every built-in model comes with terms. The Web UI names them in the download dialog, the model picker and the Models page, and the CLI prints them before a download starts. Downloading a model means you agree to its terms. The links lead to the model's original repository, where a gated model's page shows its terms and the form that grants access; the app downloads from a mirror that needs no access, so you need no Hugging Face account. Where a licence asks for a credit line, it shows next to that model in the download dialog and in the **ⓘ** popover. Every model has an entry, the free ones too, so the **ⓘ** shows for all of them.

| Licence | Supplier | Covers | Terms | Links |
|---|---|---|---|---|
| FLUX Non-Commercial License v2.1 | Black Forest Labs | `klein9b` | Non-commercial use of the model; images may be used commercially | [Model page](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) |
| Krea 2 Community License Agreement | Krea | `krea2` | Free under $1M revenue, images included; you must filter harmful content | [Licence](https://cdn.jsdelivr.net/gh/krea-ai/krea-2@db3984fbc6e13b34c0064990fc2d95ac64d00058/assets/hf_samples/LICENSE.pdf) |
| Ideogram Non-Commercial Model Agreement | Ideogram | `ideo` | Non-commercial, images included: no ads, client work or paid products | [Model page](https://huggingface.co/ideogram-ai/ideogram-4-fp8) |
| LTX-2 Community License Agreement | Lightricks | `ltx-4`, `ltx-8` | Free under $10M revenue | [Licence](https://github.com/Lightricks/LTX-2/blob/a95ab856bf29407b6b066ede0abe1846050db56c/LICENSE-2) |
| LTX-2.x Community License Agreement | Lightricks | `ltx25-4` | Free under $10M revenue | [Model page](https://huggingface.co/Lightricks/LTX-2.5) |
| MiniMax H3 Community License Agreement | MiniMax | `h3` | Not licensed in the EU, UK, South Korea or the USA; written permission needed over $20M revenue; commercial products show "MiniMax H3" | [Licence](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE) |
| CC BY-NC 4.0 with additional creator permission | M-A-P | `yue2` | Non-commercial; individual creators may monetise their outputs, companies need a licence | [Licence](https://github.com/multimodal-art-projection/YuE/blob/3acdec8cf1b97603bb696e4c6c4560348ffbb0b7/MODEL_LICENSE) |
| Boson Higgs TTS 3 Research and Non-Commercial License | Boson AI | `higgs3` | Non-commercial; creators may publish and monetise their own content if they credit Boson AI's Higgs Audio; apps and demos show "Built with Higgs TTS 3 licensed from Boson AI USA, Inc." | [Licence](https://huggingface.co/bosonai/higgs-tts-3-4b/blob/239f63fb7b02b1aa085f98d9efae5e35cc5523e8/LICENSE) |
| Boson Higgs Audio 2 Community License Agreement | Boson AI | `higgs3` | Free up to 100,000 annual active users; products show "Built with Higgs Materials licensed from Boson AI USA, Inc., Copyright Boson AI USA, Inc., All Rights Reserved and Meta Llama 3 licensed under the Meta Llama 3 Community License, Copyright Meta Platforms, Inc., All Right Reserved" | [Licence](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer/blob/815a03f1b57dcb5ae918db415f22cd0c8a6f0f54/LICENSE) |
| Stability AI Community License | Stability AI | `sa3` | Free under $1M revenue; commercial use needs a free registration | [Model page](https://huggingface.co/stabilityai/stable-audio-3-medium) |
| Gemma Terms of Use | Google | `ltx-4`, `ltx-8`, `sa3` | Free to use, also commercially; Gemma's prohibited use policy applies | [Model page](https://huggingface.co/google/gemma-3-12b-it) |
| Apache License 2.0 | Tongyi-MAI | `zit` | Free to use, also commercially | [Model page](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) |
| Apache License 2.0 | Black Forest Labs | `klein4b` | Free to use, also commercially | [Licence](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/e7b7dc27f91deacad38e78976d1f2b499d76a294/LICENSE.md) |
| Apache License 2.0 | OpenMOSS | `moss`, `moss-sft` | Free to use, also commercially | [Licence](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/blob/f972bd4b4b9e3852133cc5da954d87bc32cd8fb9/LICENSE) |
| Apache License 2.0 | Qwen | The Fast and Smart [prompt enhancer](prompt-enhancer.md) models | Free to use, also commercially | [Licence](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/LICENSE) |
| Apache License 2.0 | ByteDance Seed | The SeedVR2 upscale models | Free to use, also commercially | [Model page](https://huggingface.co/ByteDance-Seed/SeedVR2-7B) |
| Research-only training data | BiRefNet and ViTMatte authors | Background removal | Non-commercial: ViTMatte was trained on Adobe's Deep Image Matting dataset, whose agreement allows models trained on it only for non-commercial use, and BiRefNet (MIT) partly on DIS5K, which is research-only | [Dataset terms](https://github.com/xuebinqin/DIS/blob/e6455bae3892b3a7d8cf59ca91011a220d1ed606/DIS5K-Dataset-Terms-of-Use.pdf) |

LTX-2.3 and Stable Audio 3 Medium each come with two licences, their own and Gemma's, because a Gemma model is their text encoder. Higgs TTS 3 comes with two as well, one for the model and one for the audio codec it uses. LTX-2.5 uses Gemma 4, which is Apache-2.0, as are the text encoder and the latent upscaler of MiniMax H3. The background removal models of [Transparent background](images.md#transparent-background) are licensed MIT and Apache-2.0, but ViTMatte was trained on Adobe's Deep Image Matting dataset, whose agreement allows models trained on it only for non-commercial use, and BiRefNet partly on DIS5K, a research-only dataset (the link is DIS5K's terms; Adobe's agreement is not public). Neither licence clears commercial use of the cut-outs, so treat the result as non-commercial.

**FLUX.2 Klein 9B.** The licence limits the model and anything built from it to non-commercial use, but the images it makes may be used for any purpose, commercial ones included (section 2.d). They may not be used to train a model that competes with FLUX.

**MiniMax H3.** The licence requires commercial products to prominently show "MiniMax H3" in their user interface. A line such as "Powered by MiniMax H3" is encouraged, not required. Read the full licence for the regions and the revenue limit.

**Higgs TTS 3.** Audio you publish with Higgs TTS 3 must acknowledge it, in the audio itself or in the text shown with it, for example "This audio was created with Boson AI's Higgs Audio". An application or demo that other people can use shows "Built with Higgs TTS 3 licensed from Boson AI USA, Inc.". This app's Higgs TTS 3 code is "Derived from Higgs TTS 3, licensed from Boson AI USA, Inc."; the weights it loads stay under Boson's licence. The Higgs TTS 3 licence also forbids cloning a voice without the speaker's consent, and commercial use beyond creators publishing their own content needs a licence from Boson. The audio codec comes under Boson's Higgs Audio 2 Community License Agreement, which builds on the Meta Llama 3 Community License and includes Meta's Acceptable Use Policy. It asks products that use the codec to prominently show "Built with Higgs Materials licensed from Boson AI USA, Inc., Copyright Boson AI USA, Inc., All Rights Reserved and Meta Llama 3 licensed under the Meta Llama 3 Community License, Copyright Meta Platforms, Inc., All Right Reserved" on a related website, user interface, blogpost, about page or product documentation, and it needs an expanded licence from Boson above 100,000 annual active users.

**Stable Audio 3 Medium.** Using it commercially needs a free registration with Stability AI, and the licence ends once you or your affiliates earn more than $1M a year. Anyone who makes the model, or a product or service built on it, available to others must show "Powered by Stability AI" on a related website, interface or page, and pass on the licence with the notice "This Stability AI Model is licensed under the Stability AI Community License, Copyright © Stability AI Ltd. All Rights Reserved". Personal and research use needs none of this.

**YuE2.** Individual creators may publish and sell the songs they make. A company that uses the model commercially needs a licence from the model's authors.

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
