# Memory and quantization

Memory is what usually limits which models you can run. The app judges what fits before a job starts, tells you in one tag, and lets you trade a little quality for a lot of memory with quantization.

## What fits

The app holds one generation model at a time. The [prompt enhancer](#the-prompt-enhancer) and [cut-out models](#background-removal) have their own rules, below.

How much a model needs depends on where it lives:

=== "macOS"

    The CPU and GPU share one pool of memory. macOS recommends how much of it the GPU should use; a model above that makes macOS compress or swap other memory, and well beyond it the whole system swaps heavily.

    On a 16 GB Mac, Z-Image Turbo and FLUX.2 Klein 4B need `q8` or `q4`, and Klein 9B needs `q4`.

=== "Windows"

    A model can live in the GPU's memory or in system memory. Unquantized and at `q8`, the model stays in system memory and moves to the GPU piece by piece, so it needs only about 4 GB of GPU memory, but enough system memory to hold it. At `q4` the whole model sits on the GPU.

=== "Linux"

    A model can live in the GPU's memory or in system memory. Unquantized and at `q8`, the model stays in system memory and moves to the GPU piece by piece, so it needs only about 4 GB of GPU memory, but enough system memory to hold it. At `q4` the whole model sits on the GPU.

## The memory tag { #the-memory-tag }

A tag beside **Model** and **Quant** on the Image, Video and Audio tabs, on each model in the picker and on the Models page, says whether the job as you set it up fits this computer. It never blocks **Generate**.

- **Fits:** the job runs comfortably.
- **Tight:** it runs, but slowly or with little left over. On a Mac, macOS compresses or swaps other memory. On Windows and Linux it may run slowly or stop with an out-of-memory error.
- **Too large:** expect heavy swapping or an out-of-memory error. On Windows and Linux, LTX and MiniMax H3 can also be [refused](#the-memory-check).

Hover, focus or tap the tag for the details: what was judged, what the status means for this computer, and the job at every **Quant** (or **Low memory** on and off) with the memory each needs. In the Workspace, click a row to choose it. A model that isn't downloaded is judged too. In the picker and on the Models page the tag judges a typical job: a 1024×1024 image, or a 5 s clip at 768×448.

When it says **Tight** or **Too large**, pick a row that **Fits**, make the picture smaller or the clip shorter, use fewer reference images, or [switch to the smaller model](#switch-to-the-smaller-model). On Windows and Linux, `q4` can read **Tight** where `q8` fits: `q8` [streams from system memory](#quantization), so it needs less GPU memory and more system memory.

## The memory check { #the-memory-check }

Before a job loads, the app compares what it needs with the machine.

- **macOS:** nothing is refused. A job that doesn't fit runs slowly while macOS swaps, or stops with an out-of-memory error. See [the Mac slows down or beachballs](../reference/troubleshooting.md#macos-the-mac-slows-down-or-beachballs).
- **Windows and Linux:** LTX and MiniMax H3 are refused when the job is **Too large** for the machine's total system memory. A **Tight** job loads.
    - **LTX** (`ltx-4`, `ltx-8`, `ltx25-4`, converted models): refused before the model loads, when even loading its parts one after the other needs more than total system memory: [*LTX at … needs about … GB of system memory; this machine has … GB*](../reference/troubleshooting.md#ltx-needs-more-system-memory).
    - **MiniMax H3** (`h3`): refused before the model loads, when the weights and your LoRAs need more than total system memory: [*MiniMax H3 needs about … GB of system memory; this machine has … GB.*](../reference/troubleshooting.md#h3-needs-more-system-memory)

  A job the check lets through can still crawl when other programs hold memory. See [the PC crawls or freezes during a job](../reference/troubleshooting.md#windows-the-pc-crawls-or-freezes-during-a-job) on Windows. On Linux distributions that run `systemd-oomd`, such as Ubuntu, it can close the terminal or editor that started it: see [the terminal or editor closes while a model loads](../reference/troubleshooting.md#linux-the-terminal-or-editor-closes-while-a-model-loads).

`ziv-image`, `ziv-video` and `ziv-audio` print the tag's sentence once, before anything downloads, when a job is **Tight** or **Too large**, and carry on: see [*Memory for this …*](../reference/troubleshooting.md#memory-for-this). Image models and audio are never refused.

The estimates don't count memory used by other apps, or by an image model that is still loaded, so a job the check allows can still be slow on a busy machine.

## Memory while a job runs { #memory-while-a-job-runs }

While an image, video or audio job runs in the Web UI, the foot of its card shows the memory in use on the whole machine, other programs included. It carries no status.

- **macOS:** `Memory 18.2 / 32 GB`.
- **Windows and Linux:** `GPU 7.1 / 9.6 GB · RAM 21.4 / 32 GB`. The GPU is measured against the card's usable memory, about 9.6 GB on a 10 GB card. A job that streams its weights can read close to full, and the GPU reads `—` where the driver's reading isn't available.

Close what you don't need before a large job. The command-line tools show no live figure.

## Switch to the smaller model { #switch-to-the-smaller-model }

The Video tab starts on **LTX-2.3 8-bit** (`ltx-8`), the model that comes closest to the full LTX-2.3. **LTX-2.3 4-bit** (`ltx-4`) is the same model stored at 4 bits instead of 8, so it needs less memory and is the one to switch to when the clip doesn't fit.

Pick it under **Model**, or pass `-m ltx-4` to `ziv-video`. Switch when the tag says **Tight** or **Too large**, or when a job is refused for lack of system memory. `ltx-4` needs less system memory on Windows and Linux (about 20 GB against 24 GB for a 5 s clip) and less memory on a Mac with 32 GB. It saves nothing on a Mac with 24 GB or less, or on a card's GPU memory, where a shorter or smaller clip with **Low memory** on helps. **LTX-2.5 4-bit** (`ltx25-4`) needs about as much as `ltx-4`. What each model needs is in [Models](../reference/models.md#video-models).

For images, the same step is a smaller model or a lower [quant](#quantization): FLUX.2 Klein 4B, or `q8` and `q4`.

## Quantization { #quantization }

Quantization stores a model's weights in fewer bits. It cuts the memory a model needs, and often speeds it up, at the cost of some quality.

| Level | Web UI | Command line | Memory |
|---|---|---|---|
| None | **Quant: None** | | Full size |
| 8-bit | **Quant: q8** | `-q 8` | About half |
| 4-bit | **Quant: q4** | `-q 4` | About a quarter |

On macOS both levels use mflux's quantization. On Windows and Linux, `q8` is FP8 and `q4` is NF4:

- **Unquantized and `q8`** keep the model in system memory and move it to the GPU piece by piece. `q8` halves the system memory a model needs, which makes large models much faster on machines that can't hold them unquantized.
- **`q4`** keeps the whole model on the GPU at 4 bits, or, when the card is short, lets the text encoder and transformer take turns there.
- **MOSS-Image** keeps its text encoder in 4 bits (NF4) at every level, because it is larger than its transformer. The level you pick sets the transformer: `q4` on the GPU, `q8` and unquantized streamed from system memory.

Video uses the same idea through its models: `ltx-4` and `ltx-8` are LTX-2.3 at 4 and 8 bits, and `ltx25-4` is LTX-2.5 at 4 bits. On Windows and Linux, each LTX model and MiniMax H3 prepare a stored copy of their weights on the first run; see [Stored copies](#stored-quants).

### Stored copies { #stored-quants }

The first time a model runs at `q8` or `q4`, its quantized weights are saved in `models/` in the [data folder](../reference/data-folder.md) (`~/.ziv` by default) as `<name>@q8` or `<name>@q4`, for example `zit@q8`. Later jobs load that copy instead of quantizing again, so they start faster. On Windows and Linux they also never briefly need the unquantized model's memory.

The first job shows *Saving a q8 copy…* and takes longer. The copy needs roughly half (`q8`) or a quarter (`q4`) of the model's disk space.

- The copy holds the base model only. LoRAs are applied on top when a job loads. That includes the [MOSS-Image](../reference/models.md#moss-image) Turbo LoRA, so `moss@q4` and `moss-sft@q4` share one set of weights.
- It is replaced automatically when the model's weights change, or when an update changes how copies are made. The next job rebuilds it once.
- The Models page lists it as *q8 copy of &lt;name&gt;*, where you can delete it. It doesn't appear in the model picker: choose the model and a quant level.
- Deleting a model also deletes its copies.
- A model given as a raw Hugging Face repo, or a folder outside `models/` in the data folder, is quantized at load every time. Use an alias, or install the model, to get a copy.

You can also save a copy when you convert a checkpoint (`--quantize 8`, or **Quantized copy** on the Models page).

## Audio { #audio }

Speech memory depends on the part being read, not the whole text, because a long text is read one part at a time. The figures per model are in [Models](../reference/models.md#audio-memory).

## The prompt enhancer { #the-prompt-enhancer }

The enhancer is never loaded with the image or video model: it is unloaded, and its memory freed, before the model loads, so peak memory is the larger of the two, not their sum. Enhancing adds nothing during generation. In the Web UI, **Enhance** loads it, keeps it for two idle minutes and frees it sooner when a job starts. It is greyed while jobs run or wait, and so is deleting a model, because they need the memory to themselves.

## Transparent backgrounds { #background-removal }

[Transparent background](../guides/images.md#transparent-background) loads its cut-out models beside the image model, and they stay loaded for the rest of the job. The estimate doesn't count them. On a small GPU (or a Mac with little free unified memory), the app may release the image model before the cut-out and load it again before the next image of a batch, at the cost of a model load for every image.
