# Memory

Image and video models are large, and memory is usually what limits what you can run. The app estimates what fits before it starts, and only ever has one model loaded at a time, apart from the [cut-out models](#background-removal) of a transparent background.

## How memory works on each platform

=== "macOS"

    The CPU and GPU share one pool of memory. macOS recommends how much of it the GPU should use; a model above that makes macOS compress or swap other memory, and well beyond it the whole system swaps heavily.

    On a 16 GB Mac, for example, Z-Image Turbo and FLUX.2 Klein need `q8` or `q4`.

=== "Windows and Linux"

    A model can live in the GPU's memory or in system memory. Unquantized and at `q8`, the model stays in system memory and moves to the GPU piece by piece, so it needs only about 4 GB of GPU memory, but enough system memory to hold it. At `q4` the whole model sits on the GPU. See [Quantization](quantization.md).

## The memory badges

The Web UI's model picker and Models page show whether a downloaded model fits this machine.

=== "macOS"

    The estimate is compared with the GPU memory Apple recommends for this Mac:

    | Badge | Means |
    |---|---|
    | **Fits** | The estimate is at most 1.1 times the recommendation. |
    | **Tight** | At most 1.5 times, MLX's default limit. It runs, but macOS compresses or swaps other memory, so the Mac may slow down. |
    | **Too large** | Above MLX's limit. Expect heavy swapping that can make the Mac unresponsive, or an out-of-memory error. |

=== "Windows and Linux"

    The estimate is compared with the GPU's memory and the system memory:

    | Badge | Means |
    |---|---|
    | **Fits** | The model fits comfortably. |
    | **Tight** | It runs, but slowly, because it doesn't fit in system memory or the GPU is nearly full. |
    | **Too large** | It needs more GPU memory than the GPU has. Try a lower quant. |

The badge follows the quant you pick, so you can see whether `q8` or `q4` brings a model within reach. Hover it for the estimate.

After each job the Web UI gives the model's memory back to the system.

## MOSS-Image

MOSS-Image (`moss` and `moss-sft`, see [Models](models.md#moss-image)) fits a 10 GB card at every level on Windows and Linux, and a 32 GB Mac at every level.

| Level | GPU memory | System memory |
|---|---|---|
| `q4` | about 9 GB | about 15 GB |
| `q8` | about 9 GB | about 17 GB |
| None | about 7 GB | about 17 GB |

The figures are peaks for Turbo images from 1024×1024 up to 4 megapixels, on a 10 GB GPU with 32 GB of system memory. Unquantized is about five times slower than `q4` or `q8` (about a minute per image instead of 13 s), so pick `q4` or `q8` unless you want the unquantized weights.

**Upscale 4× creative.** The viewer's **4× creative** on a MOSS-Image picture fits a 10 GB GPU up to about 8 megapixels (a 576×864 picture enlarged to 2304×3456 takes about a minute at `q8`); a 16 megapixel result runs out of memory. For a larger one, use **2× creative** or the plain **4×** upscale. See [Upscaling](upscaling.md#upscaling-an-existing-image).

## Video

Video, with or without a first frame, checks the clip's size and length against memory before generating, and an extension checks the extension. The check never blocks a job; it warns when the clip may run slowly or fail.

- **Web UI:** a note under **Duration (s)** says **Tight** or **Too large** and which memory it's about. It follows the model, size, duration, **Low memory** and **Upscale** (for MiniMax H3 on Windows and Linux, also the LoRAs and **Review**), and stays hidden while the clip fits. For MiniMax H3 it also shows the expected time. The badge on a video model is the estimate for a 5 s clip at 768×448.
- **CLI:** `ziv-video` prints one warning and carries on.

To fit a clip: make it shorter, use a smaller size, and keep **Low memory** on.

=== "macOS"

    The estimate was measured with `ltx-4` on a 32 GB Mac, for clips up to about 10 s at 704×448, and is extrapolated beyond that; `ltx25-4` is estimated from its file sizes on the same basis. Macs with 24 GB or less stream the transformer from disk, so the estimate is on the high side there. The MiniMax H3 estimate ignores LoRAs. For text to video it comes from the MLX port's measurements on an M4 Pro; for image to video, first and last frame and Extend, from runs on a Mac with 32 GB of memory.

=== "Windows and Linux"

    `ltx-4` needs about 20 GB of system memory, `ltx-8` about 27 GB, `ltx25-4` about 21 GB, and MiniMax H3 about 18 GB free. On a 10 GB GPU with `ltx-4`, clips up to 20 s at `m`, 15 s at `l` and 10 s at `xl` fit, with or without upscaling. The estimate was measured on one 10 GB GPU with 32 GB of system memory; on other cards the limits are extrapolated. For MiniMax H3 it also covers the mode (image to video and Extend need more), LoRAs and upscale, and every estimate was at or above the measured peak.

The estimate doesn't count memory used by other apps, or by an image model that is still loaded.

## Audio

The [audio models](audio.md) fit a 10 GB GPU on Windows and Linux, and a Mac with 32 GB of unified memory. Speech memory depends on the part, not the whole text, because a long text is read one part at a time.

=== "macOS"

    The audio models run on the Mac's GPU. The figures are peak GPU memory measured on a 32 GB Mac:

    | Model | GPU memory | Measured at |
    |---|---|---|
    | YuE2 3B (`yue2`) | about 10.5 GiB | A take of the full 6:00 |
    | Qwen3-TTS 1.7B (`qwen3_tts`) | about 12.0 GiB | A text read in parts, in a default voice or a recording |
    | Stable Audio 3 Medium (`sa3`) | about 6.1 GiB | A sound of the full 380 s |

=== "Windows and Linux"

    The figures are peak GPU memory measured on an otherwise empty 10 GB card:

    | Model | GPU memory | Measured at |
    |---|---|---|
    | YuE2 3B (`yue2`) | about 6.7 GiB | A take of the full 6:00 |
    | Qwen3-TTS 1.7B (`qwen3_tts`) | about 6.0 GiB | A text read in parts, in a default voice or a recording |
    | Stable Audio 3 Medium (`sa3`) | about 6.1 GiB | A sound of the full 380 s |

    Designing a voice loads its own checkpoint and needs no more than the Qwen3-TTS row.

    System memory matters too: YuE2 peaks at about 11 GiB while it loads or runs, which leaves a computer with 16 GB of RAM little to spare. YuE2 keeps its weights in system memory and moves the part it needs to the GPU, which is how it fits a 10 GB card. This adds a few seconds when the model loads and does not change the result.

## One model at a time

The [prompt enhancer](prompt-enhancer.md#memory) and the image or video model are never loaded together, so peak memory is the larger of the two, not their sum. Actions that need the model memory to themselves, such as enhancing a prompt by hand or deleting a model, wait until no job is running or queued.

## Background removal

[Transparent background](images.md#transparent-background) loads its cut-out models beside the image model, and they stay loaded for the rest of the job. The estimate doesn't count them.

When too little GPU memory (unified memory on a Mac) is free after a render, the app releases the image model before the cut-out and loads it again before the next image of a batch. "Too little" is 2 GB plus room for the cut-out at the image's final size: about 2.25 GB for a 1024×1024 image, and about 6 GB after a 4× upscale to 4096×4096. This keeps a small GPU from running out, at the cost of a model load for every image. When the free memory can't be read, the image model stays loaded.

## Linux: the terminal closes while a model loads

On Linux distributions that run `systemd-oomd`, such as Ubuntu, loading a large model can close the terminal or editor that started it. See [Troubleshooting](troubleshooting.md#linux-the-terminal-or-editor-closes-while-a-model-loads).
