# Memory and quantization

Memory is what usually limits which models you can run. The app estimates what fits before a job starts, shows it as badges and notes, and lets you trade a little quality for a lot of memory with quantization.

## What fits

The app holds one generation model at a time. The [prompt enhancer](#the-prompt-enhancer) and [cut-out models](#background-removal) have their own rules, below.

How much a model needs depends on where it lives:

=== "macOS"

    The CPU and GPU share one pool of memory. macOS recommends how much of it the GPU should use; a model above that makes macOS compress or swap other memory, and well beyond it the whole system swaps heavily.

    On a 16 GB Mac, Z-Image Turbo and FLUX.2 Klein need `q8` or `q4`.

=== "Windows"

    A model can live in the GPU's memory or in system memory. Unquantized and at `q8`, the model stays in system memory and moves to the GPU piece by piece, so it needs only about 4 GB of GPU memory, but enough system memory to hold it. At `q4` the whole model sits on the GPU.

=== "Linux"

    A model can live in the GPU's memory or in system memory. Unquantized and at `q8`, the model stays in system memory and moves to the GPU piece by piece, so it needs only about 4 GB of GPU memory, but enough system memory to hold it. At `q4` the whole model sits on the GPU.

## The memory badges { #the-memory-badges }

The model picker and the Models page show whether a downloaded model fits this machine. The badge follows the quant you pick, so you can see whether `q8` or `q4` brings a model within reach. Hover it for the estimate. The Web UI gives a model's memory back to the system after each job.

- **Fits:** the model runs comfortably.
- **Tight:** it runs, but slowly or with little left for other apps. On a Mac, macOS compresses or swaps other memory. On Windows and Linux, an image model streams part of itself from disk, and LTX and MiniMax H3 run with a [warning](#the-memory-check).
- **Too large:** expect heavy swapping that can make a Mac unresponsive, or an out-of-memory error. On Windows and Linux, LTX and MiniMax H3 can also be [refused](#the-memory-check). Try a lower quant.

## The Duration note { #video }

For video, a note under **Duration (s)** says **Tight** or **Too large** and which memory it is about. It checks the size and length of a new clip, or the window an extension generates (about the last second plus the new frames, whatever the source's length), and stays hidden while the clip fits. The badge on a video model is the estimate for a 5 s clip at 768×448.

The note only warns; it never blocks **Generate**. To fit a clip, make it shorter, use a smaller size and keep **Low memory** on, or [switch to the smaller model](#switch-to-the-smaller-model). `ziv-video` prints the same estimate as one warning and carries on.

## The memory check { #the-memory-check }

Before and while a job loads, the app compares what it needs with the machine.

- **macOS:** nothing is refused. A job that doesn't fit runs slowly while macOS swaps, or stops with an out-of-memory error. See [the Mac slows down or beachballs](../reference/troubleshooting.md#macos-the-mac-slows-down-or-beachballs).
- **Windows and Linux:** LTX and MiniMax H3 are refused when their parts need more than the machine's total system memory, and warn above about 80% of it.
    - **LTX** (`ltx-4`, `ltx-8`, `ltx25-4`, converted models): refused before the model loads, when even loading its parts one after the other needs more than total system memory: [*LTX at … needs about … GB of system memory; this machine has … GB*](../reference/troubleshooting.md#ltx-needs-more-system-memory).
    - **MiniMax H3** (`h3`): refused before the model loads, when the weights and your LoRAs need more than total system memory: [*MiniMax H3 needs about … GB of system memory; this machine has … GB.*](../reference/troubleshooting.md#h3-needs-more-system-memory)

  A job the check lets through can still crawl when other programs hold memory. See [the PC crawls or freezes during a job](../reference/troubleshooting.md#windows-the-pc-crawls-or-freezes-during-a-job) on Windows. On Linux distributions that run `systemd-oomd`, such as Ubuntu, it can close the terminal or editor that started it: see [the terminal or editor closes while a model loads](../reference/troubleshooting.md#linux-the-terminal-or-editor-closes-while-a-model-loads).

On every platform, the video clip check is separate and only warns: a `ziv-video` warning and the [Duration note](#video). **Image models** have no check. **Audio** warns before it starts when the model is tight or too large, and an out-of-memory error names the model and says what to close.

The estimates don't count memory used by other apps, or by an image model that is still loaded, so a job the check allows can still be slow on a busy machine.

## Switch to the smaller model { #switch-to-the-smaller-model }

The Video tab starts on **LTX-2.3 8-bit** (`ltx-8`), the model that comes closest to the full LTX-2.3. **LTX-2.3 4-bit** (`ltx-4`) is the same model stored at 4 bits instead of 8, so it needs less memory and is the one to switch to when the clip doesn't fit.

Pick it under **Model**, or pass `-m ltx-4` to `ziv-video`. Switch when the model's badge or the note under **Duration (s)** says **Tight** or **Too large**, or when a job is refused for lack of system memory. `ltx-8` is for a Mac with 32 GB or more; on a Mac with 16 GB, the estimate points to `ltx-4`. On Windows and Linux with 32 GB of system memory or less, use `ltx-4` or **LTX-2.5 4-bit** (`ltx25-4`), and on a card with 8 to 12 GB start with `ltx-4`. What each model needs is in [Models](../reference/models.md#video-models).

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
- **`q4`** keeps the whole model on the GPU, so the GPU must hold it at 4 bits.
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
