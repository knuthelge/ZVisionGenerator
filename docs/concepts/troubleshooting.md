# Troubleshooting

## Installing and starting

### Installed with pip

The docs and the error messages assume [uv](https://docs.astral.sh/uv/): `uv tool install z-vision-generator`. A pip install from PyPI resolves the same packages, but a checkout needs uv, whose resolver picks the CUDA build of PyTorch.

### `ziv ui` says Web UI packages are missing

The launcher prints how to repair the install. Usually reinstalling fixes it: `uv tool install --reinstall z-vision-generator`, or `uv sync` in a checkout.

### The browser opens on a different port

Port 8080 was busy, so the next free port was used. The terminal prints the address.

### Another device can't reach the Web UI

Start it with `ziv ui --host local` and use the network address it prints. To reach it under a host name that isn't this computer's name, list the name in `ZIV_UI_ALLOWED_HOSTS`. See [Other devices](../web-ui/devices.md).

## Windows and Linux: CUDA

### "CUDA is not available"

Generation on Windows and Linux needs an NVIDIA GPU that PyTorch can see; there is no CPU fallback. Check that `nvidia-smi` lists your GPU. When the message mentions the driver, install NVIDIA driver 580 or newer.

When the message says the installed PyTorch has no CUDA, the install came from PyPI alone, whose Windows PyTorch is CPU-only. The message shows the install command with PyTorch's CUDA index, as does [the Install page](../install.md).

The prompt enhancer does run on the CPU without CUDA, but slowly.

## Downloads

### A model download is refused

The model is gated: its page asks you to accept a license. The message names the model and links to its page.

1. Accept the license on the model's page, signed in to Hugging Face.
2. Run `hf auth login`, or set `HF_TOKEN`.

"Hugging Face rejected your token" or "your Hugging Face token has no access" means a token is set, but its account hasn't accepted the license, or the token is no longer valid. See [Hugging Face token](data-folder.md#hugging-face-token).

### "Could not reach Hugging Face"

The download needs the network. Models already downloaded keep working offline.

### A download was interrupted

Start it again. Finished files are kept and only the rest is downloaded.

### Downloads are slow

Set `HF_XET_HIGH_PERFORMANCE=1` before starting the app. See [Download speed](data-folder.md#download-speed).

## Memory

### The computer slows down, or the job stops with an out-of-memory error

The model or clip is too large for this machine.

- Pick a lower [quant](quantization.md): `q8`, or `q4`.
- Make the image smaller, or generate small and [upscale](upscaling.md).
- For video, use `ltx-4`, a shorter clip, a smaller size, and keep **Low memory** on.
- Close other apps that use a lot of memory.

The Web UI's [memory badges](memory.md#the-memory-badges) show what fits before you start.

### "LTX at q8 needs about 27 GB of system memory; this machine has 16 GB"

On Windows and Linux, video models check system memory before loading. Use the model the message names; `ltx-4` needs the least.

### Linux: the terminal or editor closes while a model loads

On Linux distributions that run `systemd-oomd`, such as Ubuntu, loading a large model can close the terminal or editor that started the job, and the job with it, even when there is enough memory. `systemd-oomd` reacts to memory *pressure*, not to running out of memory: loading 15–35 GB of weights on a machine with about 32 GB of RAM fills the page cache and can keep pressure above Ubuntu's limit of 50% for 20 seconds. `journalctl -u systemd-oomd` lists each time it acted.

To relax the limit for your user session, create `/etc/systemd/system/user@.service.d/99-oomd-relax.conf` with sudo:

```ini
[Service]
ManagedOOMMemoryPressureLimit=80%
ManagedOOMMemoryPressureDurationSec=60s
```

- **Apply:** run `sudo systemctl daemon-reload`, then log out and back in, or reboot.
- **Check:** `systemctl show user@$(id -u).service -p ManagedOOMMemoryPressureLimit` prints `80%`.
- **Undo:** delete the file, reload and log in again.

80% for 60 seconds still stops runaway memory use, but lets typical model loads finish. The largest loads, such as Krea 2 unquantized, can still go over it; use `q8` or `q4` on these machines. The app never changes this setting itself.

To protect your editor when oomd does act, start heavy jobs in their own unit, so oomd stops the job rather than the editor:

```bash
systemd-run --user --collect --same-dir -p OOMScoreAdjust=500 ziv-image -m klein9b --prompt "a landscape"
```

## Images

### Ideogram 4 returns a grey "blocked by safety filter" frame

This sometimes happens for harmless prompts. Set **First sigma** (`--first-sigma`) to `1.005` or `1.006`. See [Ideogram 4 Instant](models.md#ideogram-4-instant).

### Ideogram 4 makes a flat cut-out on a checkerboard

A plain prompt becomes a sparse caption. Let the enhancer write a full one with **Write as: JSON** (`--enhance-format json`). See [JSON captions](prompt-enhancer.md#json-captions-ideogram-4).

### A negative prompt is ignored

Only some models use one. See [Negative prompts](prompts.md#negative-prompts).

### A LoRA has no effect, or a warning says parts were skipped

Check that the LoRA was made for this model family. Only standard LoRAs and LyCORIS LoKr files are supported. See [LoRAs](loras.md#what-works).

## Video

### ffmpeg is missing

`ziv-video` offers to install it and asks first. On Windows, open a new terminal afterwards. On Fedora, `dnf install ffmpeg` needs RPM Fusion.

### "Could not detect video model family"

Use `ltx-4`, `ltx-8`, `ltx25-4` or `h3`, a converted LTX model, a known LTX or MiniMax H3 repo, or a local path containing `ltx`.

### "LTX on Windows/Linux runs 4-bit or 8-bit"

On Windows and Linux, LTX only loads at a quant level. Use `ltx-4`, `ltx-8` or `ltx25-4` instead of a repository id.

### A video can't be extended

The check lists every problem. Common ones: the clip was upscaled (extend the original), it has a variable frame rate, or its size isn't a multiple of 32. See [What can be extended](video.md#what-can-be-extended).

### "MiniMax H3 ran out of GPU memory at …"

The clip didn't fit the GPU. Make it shorter or smaller and run it again; the next job isn't affected.

### "MiniMax H3 extends 24 fps clips only" or "… clips whose size is a multiple of 32"

MiniMax H3 can only extend a 24 fps clip whose width and height are multiples of 32. Extend it with an LTX model instead.

### A video can't be upscaled

The plain video upscale refuses HDR video, a variable frame rate, interlaced video and non-square pixels. "Not enough GPU memory to upscale to …" means the size didn't fit even one frame at a time; pick a smaller video. See [Upscaling an existing video](upscaling.md#upscaling-an-existing-video).

### A converted LTX fine-tune comes out ghosted and noisy

It is a dev checkpoint, which needs Lightricks' distilled LoRA. Convert it again with `--distill`. See [LTX-2.3 and LTX-2.5 video fine-tunes](../cli/models.md#ltx-23-and-ltx-25-video-fine-tunes).

### A converted MiniMax H3 fine-tune comes out blurry or burnt

The turbo choice was wrong. Convert it again with the other choice: with or without `--turbo` (**Already step-distilled**). See [MiniMax H3 video fine-tunes](../cli/models.md#minimax-h3-video-fine-tunes).

## Report a problem

Open a [bug report](https://github.com/knuthelge/ZVisionGenerator/issues/new?template=bug_report.yml) with:

- what you did, what you expected and what happened;
- your platform, GPU or Mac model and memory;
- the [error log](data-folder.md#error-log) of the command, `~/.ziv/logs/<command>.log`.

Check the log for anything private, such as file paths, before you attach it.
