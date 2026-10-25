# Memory

Image and video models are large, and memory is usually what limits what you can run. The app estimates what fits before it starts, and only ever has one model loaded at a time.

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

## Video

Text to Video and Image to Video check the clip's size and length against memory before generating, and Extend Video checks the extension. The check never blocks a job; it warns when the clip may run slowly or fail.

- **Web UI:** a note under **Duration (s)** says **Tight** or **Too large** and which memory it's about. It follows the model, size, duration, **Low memory** and **Upscale** (for MiniMax H3 on Windows and Linux, also the LoRAs and **Review**), and stays hidden while the clip fits. For MiniMax H3 it also shows the expected time. The badge on a video model is the estimate for a 5 s clip at 768×448.
- **CLI:** `ziv-video` prints one warning and carries on.

To fit a clip: make it shorter, use a smaller size, and keep **Low memory** on.

=== "macOS"

    The estimate was measured with `ltx-4` on a 32 GB Mac, for clips up to about 10 s at 704×448, and is extrapolated beyond that; `ltx25-4` is estimated from its file sizes on the same basis. Macs with 24 GB or less stream the transformer from disk, so the estimate is on the high side there. The MiniMax H3 estimate ignores LoRAs. For text to video it comes from the MLX port's measurements on an M4 Pro; for image to video, first and last frame and Extend, from runs on a Mac with 32 GB of memory.

=== "Windows and Linux"

    `ltx-4` needs about 20 GB of system memory, `ltx-8` about 27 GB, `ltx25-4` about 21 GB, and MiniMax H3 about 18 GB free. On a 10 GB GPU with `ltx-4`, clips up to 20 s at `m`, 15 s at `l` and 10 s at `xl` fit, with or without upscaling. The estimate was measured on one 10 GB GPU with 32 GB of system memory; on other cards the limits are extrapolated. For MiniMax H3 it also covers the mode (image to video and Extend need more), LoRAs and upscale, and every estimate was at or above the measured peak.

The estimate doesn't count memory used by other apps, or by an image model that is still loaded.

## One model at a time

The [prompt enhancer](prompt-enhancer.md#memory) and the image or video model are never loaded together, so peak memory is the larger of the two, not their sum. Actions that need the model memory to themselves, such as enhancing a prompt by hand or deleting a model, wait until no job is running or queued.

## Linux: the terminal closes while a model loads

On Linux distributions that run `systemd-oomd`, such as Ubuntu, loading a large model can close the terminal or editor that started it. See [Troubleshooting](troubleshooting.md#linux-the-terminal-or-editor-closes-while-a-model-loads).
