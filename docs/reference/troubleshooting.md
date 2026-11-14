# Troubleshooting

What an error message or a slowdown means, and what to do. Find your problem by the words of the message, or by what you see: slow, stuck, frozen, nothing happens.

A message that names a file, a setting or a model, or that says "use X instead", means exactly that. Do what it says first. The data folder is `~/.ziv` by default, and `ZIV_DATA_DIR` moves it.

## Where to look { #where-to-look }

- [Installing and starting](#installing-and-starting): `ziv ui` won't start, no NVIDIA GPU found, a package or ffmpeg missing, the browser opens on another port, another device can't connect, and the launcher's own messages: this computer isn't supported, the install failed and the app didn't start.
- [Downloads](#downloads): a model won't download, Hugging Face refuses or can't be reached, the download is slow or stops, files missing from the cache.
- [Stuck or slow](#stuck-or-slow): the first job seems stuck, Generate is greyed out, a job failed.
- [Memory and slowdowns](#memory-and-slowdowns): the computer is slow, crawls, freezes or beachballs, out of memory, the terminal closes while a model loads, a model is refused for lack of memory.
- [Images](#images): the size is refused, a grey frame, a flat cut-out, a negative prompt is ignored.
- [Video](#video): a clip can't be extended or upscaled, a first frame is needed, ffmpeg failed on a clip.
- [Audio](#audio): a bracketed word is refused, the model made no sound, the peak was scaled.
- [Prompts and prompt files](#prompts-and-prompt-files): the prompt file isn't found, can't be read or names an undefined snippet, a set is skipped.
- [Prompt enhancer](#prompt-enhancer): the rewrite failed or the original prompt was used, Smart changed to Fast, the enhancer is slow or won't load.
- [Models, LoRAs and conversion](#models-loras-and-conversion): a model isn't found or can't be detected, a LoRA does nothing, a converted model looks wrong.
- [Web UI messages](#web-ui-messages): a picture is missing, the Workspace won't load, the live connection was lost, a file can't be played, an upscale isn't offered.
- [Configuration file](#configuration-file): every command fails with a message about `config.yaml`.
- [Report a problem](#report-a-problem): what to send when nothing here helps.

## Installing and starting { #installing-and-starting }

### "CUDA is not available"

Windows and Linux. PyTorch has CUDA, but it can't use a GPU: the NVIDIA driver is too old, or the PC has no NVIDIA GPU. Generation needs an NVIDIA GPU that PyTorch can see; there is no CPU fallback. Check that `nvidia-smi` lists your GPU. When the message mentions the driver, install NVIDIA driver 580 or newer. A PC without an NVIDIA GPU can't generate. See [Install](../get-started/install.md#check-your-machine). A PyTorch without CUDA gives [a different message](#pytorch-has-no-cuda).

### "… cannot run: the installed PyTorch has no CUDA. Install again with PyTorch's CUDA build: …" { #pytorch-has-no-cuda }

Windows. The install came from PyPI alone, and PyPI's Windows PyTorch is CPU-only, so no driver or GPU change helps. Reinstall with PyTorch's CUDA build; the message prints the command, and the install page has it under the Windows tab. See [Repair the install](../get-started/install.md#repair-the-install).

### A package is missing or too old { #a-package-is-missing-or-too-old }

A package the app needs is missing or out of date, usually after a partial install. Reinstall: see [Repair the install](../get-started/install.md#repair-the-install). On Windows, use the command on that page, not the one a message prints: it reinstalls the package and keeps PyTorch's CUDA build. With the launcher, choose **Repair** in its window instead.

#### "diffusers>=… is required for Windows/Linux LTX video support."

The installed `diffusers` is older than LTX video needs; the message names the version it needs.

#### "Installed diffusers is missing the LTX-2 video pipelines (…)."

The installed `diffusers` lacks the LTX-2 pipelines the message lists, so it is too old or incomplete.

#### "Installed diffusers is missing the LTX-2 distilled sigma tables, the latent upsampler model or the video condition."

The installed `diffusers` lacks parts that LTX video needs, so it is too old or incomplete.

#### "Video diffusers dependencies are unavailable. …"

The video packages for Windows and Linux aren't installed.

#### macOS: "Video dependency ltx-pipelines-mlx not found. Reinstall with: uv sync"

The package that runs LTX video on a Mac isn't installed.

#### "… could not start because this installation is missing required Web UI dependencies: …"

The Web UI's packages, which the message lists, aren't installed.

### ffmpeg is missing { #ffmpeg-is-missing }

Video and audio need ffmpeg. `ziv-video` and `ziv-audio` offer to install it and ask first. The Web UI reports it when you press **Generate**: install ffmpeg, make sure it is on your `PATH`, and try again. On Windows, open a new terminal afterwards. On Fedora, `dnf install ffmpeg` needs RPM Fusion. The message reads "ffmpeg is required but was not found on PATH." See [Install](../get-started/install.md#ffmpeg). With the launcher, choose **Repair** in its window: it downloads ffmpeg again, which also helps when antivirus software has removed it.

### The browser opens on a different port { #the-browser-opens-on-a-different-port }

Port 8080 was busy, so the next free port was used. The terminal prints the address. To keep the address, stop the other program that holds port 8080. See [Install](../get-started/install.md#start-it).

### "No available port found starting at …"

`ziv ui` tried the port it was asked for and the ones after it, and none was free. Close the program that holds them, or start with a free port, for example `ziv ui --port 8090`. See the [command reference](../cli/reference.md#ziv-ui-ziv-ui).

### "Invalid Host header. Add the hostname to …"

Another device reached the Web UI under a name the app doesn't know. List the name in `ZIV_UI_ALLOWED_HOSTS` before starting the app; IP addresses and `localhost` are always allowed. See [Phones, tablets and other computers](../web-ui/devices.md).

### Another device can't reach the Web UI { #another-device-cant-reach-the-web-ui }

The app answers only on this computer unless you ask it to share. Start it with `ziv ui --host local` and open the network address it prints from the other device. Allow it when Windows Firewall or macOS asks about incoming connections. If the browser then shows "Invalid Host header", use the entry above. See [Phones, tablets and other computers](../web-ui/devices.md#open-it-from-another-device).

### The launcher says this computer isn't supported { #launcher-unsupported }

The launcher checked the machine before downloading anything. The window names what it found and what is needed; when it is something you can fix, fix it and choose **Check again**. A driver that is installed but isn't working usually needs a restart after a driver update, or, on Linux, a driver key that Secure Boot accepts. Without an NVIDIA RTX 20-series or newer card the app can't run; see [Install](../get-started/install.md#check-your-machine).

### The launcher's install failed { #launcher-install-failed }

The window says why, and **Retry** carries on with what was already downloaded. A file that doesn't match usually comes from a cut-off download or a hotel or office Wi-Fi login page; retry, or try another network. If the window says a newer launcher fixes it, download it from [Install](../get-started/install.md#install-with-the-launcher). If **Retry** keeps failing, copy **Show the log** into a [bug report](#report-a-problem).

### The launcher says Didn't start { #launcher-didnt-start }

**Open Z-Vision Generator** tries again, and **Repair** in the **⋯** menu reinstalls. If it keeps failing, copy **Show the log** into a [bug report](#report-a-problem).

### Installed with pip

The docs and the error messages assume [uv](https://docs.astral.sh/uv/): `uv tool install z-vision-generator`. A pip install from PyPI resolves the same packages, but a checkout needs uv, whose resolver picks the CUDA build of PyTorch. See [Install](../get-started/install.md#get-uv).

## Downloads { #downloads }

### A model download is refused

The model is gated: its page on Hugging Face asks you to accept a license. The message names the model and links to its page.

1. Accept the license on the model's page, signed in to Hugging Face.
2. Run `hf auth login`, or set `HF_TOKEN`.

In the Web UI the message appears as an error when you press **Generate** or **Download**, and no download dialog opens. Each message below says which step is missing. See [Hugging Face token](data-folder.md#hugging-face-token).

#### "… is gated: …"

A token is set, but its account hasn't accepted the model's license. Accept it signed in as the account the token belongs to.

#### "… needs a Hugging Face token: …"

No token is set. Do both steps above.

#### "Hugging Face rejected your token: …"

Hugging Face no longer accepts the token. Run `hf auth login` again, or unset `HF_TOKEN` to download without one.

#### "Your Hugging Face token has no access to …"

The token is valid, but its account hasn't accepted this model's license. Accept it signed in as that account.

#### "… was not found, or …"

The name is wrong, or the model needs a token. Check the name first; if it is right, do both steps above.

### Hugging Face can't be reached { #hugging-face-cant-be-reached }

The download needs the network. Check the internet connection and try again; a model already on disk needs no connection. In the Web UI the download dialog reads the model's size first. If that fails the Models page shows an error with **Retry**, and the dialog warns and offers **Download anyway**; press it to try, or **Cancel** and come back online. See [Models page](../web-ui/models.md#download-and-cancel).

#### "Could not reach Hugging Face to download …"

The job needed a download and Hugging Face didn't answer. Go online and start it again.

#### "Could not reach Hugging Face to check the download of …"

The Models page couldn't read the model's size. Press **Retry** once you're online.

#### "Couldn't check the size: Hugging Face can't be reached. …"

The download dialog couldn't read the size. **Download anyway** tries; if the download can't run, the job fails.

### A download stopped { #a-download-stopped }

The message ends with the reason, usually a lost network or a refusal. Fix that, then start the job again; finished files are kept and only the rest is downloaded. See [Download speed](data-folder.md#download-speed).

#### "Failed to download the …"

A download of the named model or file failed; the end of the message says why.

#### "Download of … failed after … of …: …"

A download stopped part-way, after the amount shown; the end of the message says why.

### Downloads are slow { #downloads-are-slow }

Set `HF_XET_HIGH_PERFORMANCE=1` before starting the app so downloads use all the bandwidth and CPU cores. A model is many gigabytes, so on a home connection a large one takes tens of minutes. See [Download speed](data-folder.md#download-speed).

### "… is not downloaded: some of its files are missing from the cache"

Audio. Some of the model's files were deleted from the Hugging Face cache, or the download stopped early. Delete the model on the Models page and download it again. See [Models page](../web-ui/models.md#delete-and-free-space).

### macOS: "… is an LTX-2.5 pack but has no text_encoder.safetensors and text_encoder_config.json at its root; …"

The downloaded LTX-2.5 files are incomplete. Delete the model on the Models page and download it again. See [Models page](../web-ui/models.md#delete-and-free-space).

## Stuck or slow { #stuck-or-slow }

### The first job seems stuck { #the-first-job-seems-stuck }

A first job that moves slowly is usually a download or a one-time preparation, not a hang: the job card names what it is doing. A large model takes tens of minutes to download on a home connection, and on Windows and Linux a video model is also prepared for your graphics card once, which can take several minutes. Later jobs start much faster. If the download speed stays at zero, see [Downloads are slow](#downloads-are-slow). See [Your first image](../get-started/first-image.md#what-happens-and-how-long).

### Generate is greyed out { #generate-is-greyed-out }

The reason is written above the button; do what it says. A missing picture or audio track: [choose it again or remove it](#a-picture-or-audio-track-is-missing). A last frame without a first frame: [add a first frame](#add-a-first-frame). A LoRA still downloading: the button turns on when it ends, and **Retry** above it handles a failure. Text being written ("Enhancing… Generate when it's done."): wait. See [Generate](../web-ui/workspace.md#generate).

### A job failed

The job card says why, and the [Job log](../web-ui/job-log.md) keeps the reason after the card is gone. Look for the message on this page. If it isn't here, the full error is in the command's error log, `logs/<command>.log` in the data folder. The log is emptied when the command starts again, so copy it before you restart. See [Report a problem](#report-a-problem).

## Memory and slowdowns { #memory-and-slowdowns }

The app estimates memory before a job, and what it does about it depends on the platform:

- **Windows and Linux:** LTX and MiniMax H3 are refused before they load when the model needs more system memory than the machine has (the model's **Too large** badge). They warn when it needs more than 80% of it (**Tight**), and run.
- **macOS:** nothing is refused. A video clip that may not fit gets a warning and the job continues.
- **Everywhere:** the clip check (`ziv-video`'s warning and the note under **Duration (s)**) never blocks a job.

A job the check lets through can still be slow, or run out of memory. Image models have no refusal: on Windows and Linux a **Tight** image model runs slowly because part of it streams from disk, and one that needs more GPU memory than the card has stops with an out-of-memory error. Audio models warn before they start when the card is short.

### The computer slows down, or the job stops with an out-of-memory error

The model or clip is too large for this machine.

- Pick a lower [quant](../concepts/memory.md#quantization): `q8`, or `q4`.
- Make the image smaller, or generate small and [upscale](../concepts/upscaling.md).
- For video, use **LTX-2.3 4-bit** (`ltx-4`), a shorter clip, a smaller size, and keep **Low memory** on.
- Close other apps that use a lot of memory.

On Windows and Linux, PyTorch's own message begins with "CUDA out of memory". See [Switch to the smaller model](../concepts/memory.md#switch-to-the-smaller-model).

### Windows: the PC crawls or freezes during a job

The memory check can pass and the PC can still crawl. The check estimates the model and the clip. It doesn't count what other programs hold, such as a browser with many tabs, or an image model that is still loaded. When system memory runs short, Windows moves memory to the disk and everything slows down, not only the app.

- Close other programs, a browser first, then run the job again.
- Use the smaller choice: **LTX-2.3 4-bit** (`ltx-4`) for video, `q8` or `q4` for an image model, a shorter or smaller clip, and **Low memory** on.
- Run one job at a time.
- A job that sits close to the limit shows **Tight** on the model, or a note under **Duration (s)**. Treat that as a reason to switch before you start.

The first run of a video model prepares it for your graphics card and reads tens of gigabytes; the PC can feel slow until that ends. See [Switch to the smaller model](../concepts/memory.md#switch-to-the-smaller-model).

### macOS: the Mac slows down or beachballs

The CPU and GPU share one pool of memory. A model above what macOS recommends for the GPU makes macOS compress or swap other memory, so the Mac slows down, the spinning beachball appears and the apps freeze. The memory check doesn't stop it: on a Mac it only warns. **Tight** on a model means the Mac may slow down, and **Too large** means heavy swapping that can make it unresponsive.

- Quit other apps first, a browser first.
- Use the smaller choice: **LTX-2.3 4-bit** (`ltx-4`) for video, `q8` or `q4` for an image model, a shorter or smaller clip, and **Low memory** on.
- On a Mac with 16 GB, start with **LTX-2.3 4-bit** (`ltx-4`).

See [The memory badges](../concepts/memory.md#the-memory-badges).

### Linux: the terminal or editor closes while a model loads

On distributions that run `systemd-oomd`, such as Ubuntu, loading a large model can close the terminal or editor that started the job, and the job with it, even when there is enough memory. `systemd-oomd` reacts to memory *pressure*, and loading 15–35 GB of weights fills the page cache and can hold pressure above Ubuntu's 50% limit for 20 seconds. `journalctl -u systemd-oomd` lists each time it acted.

To relax the limit for your user session, create `/etc/systemd/system/user@.service.d/99-oomd-relax.conf` with sudo:

```ini
[Service]
ManagedOOMMemoryPressureLimit=80%
ManagedOOMMemoryPressureDurationSec=60s
```

Run `sudo systemctl daemon-reload`, then log out and back in. To undo, delete the file and do the same. The largest loads, such as Krea 2 unquantized, can still go over 80%; use `q8` or `q4`. The app never changes this setting itself.

To protect your editor when oomd does act, start a heavy job in its own unit, so oomd stops the job rather than the editor:

```bash
systemd-run --user --collect --same-dir -p OOMScoreAdjust=500 ziv-image -m klein9b --prompt "a landscape"
```

### "LTX at … needs about … GB of system memory; this machine has … GB" { #ltx-needs-more-system-memory }

Windows and Linux. The job is refused before the model loads: even loading its parts one after the other needs more system memory than the machine has. Use the model the message names; `ltx-4` needs the least. For a converted model, convert the checkpoint again with `--quantize 4` under a new name. See [Switch to the smaller model](../concepts/memory.md#switch-to-the-smaller-model).

### "LTX at … needs about … GB of this machine's … GB of system memory, leaving little for other apps."

Windows and Linux. A warning, not a refusal: the job runs, but the model needs more than 80% of system memory and the machine may slow down. For `q8`, switch to **LTX-2.3 4-bit** (`ltx-4`) if it is stopped for lack of memory. See [Windows: the PC crawls or freezes during a job](#windows-the-pc-crawls-or-freezes-during-a-job).

### "MiniMax H3 needs about … GB of system memory; this machine has … GB." { #h3-needs-more-system-memory }

Windows and Linux. The job is refused before the model loads: the weights and your LoRAs need more system memory than the machine has. H3 needs about 18 GB before any LoRAs, so remove LoRAs first. A machine with less system memory than that can't run H3, and an LTX model needs more, not less: **LTX-2.3 4-bit** (`ltx-4`) needs about 20 GB. See [Models and licences](models.md#video-models).

### "MiniMax H3 needs about … GB of this machine's … GB of system memory, leaving little for other apps."

Windows and Linux. A warning: the job runs, but the weights need more than 80% of system memory, so close other programs first. The decoded frames add to this for each clip. See [Windows: the PC crawls or freezes during a job](#windows-the-pc-crawls-or-freezes-during-a-job).

### "The model needs about … GB of system memory; this machine has … GB (…). Continuing; it may be slow or fail"

Windows and Linux. `ziv-video` warns before the model loads. The weights, not the clip, fill system memory, so a shorter or smaller clip doesn't help; switch model instead. With `tight` in the brackets the job carries on and may be slow; with `too large` the weights need more system memory than the machine has, and the job is then refused with [*LTX at … needs about … GB of system memory*](#ltx-needs-more-system-memory) or [*MiniMax H3 needs about … GB of system memory*](#h3-needs-more-system-memory). See [Switch to the smaller model](../concepts/memory.md#switch-to-the-smaller-model).

### "At …x…, … frames this clip"

`ziv-video` warns and carries on: the clip may be slow or fail. Use fewer `--frames` or a smaller `--size`, and keep `--low-memory` on. See [Switch to the smaller model](../concepts/memory.md#switch-to-the-smaller-model).

### "The …-frame extension window"

`ziv-video` warns that the window an extension makes may not fit, and carries on. Use `--extend-seconds` or `--extend-frames` for a shorter extension. See [Switch to the smaller model](../concepts/memory.md#switch-to-the-smaller-model).

### "MiniMax H3 ran out of GPU memory at …"

Windows and Linux. The clip didn't fit the GPU. Make it shorter or smaller and run it again; the next job isn't affected. See [Switch to the smaller model](../concepts/memory.md#switch-to-the-smaller-model).

### "… ran out of GPU memory"

Audio. The model needed more GPU memory than the card has. Close other apps that use the GPU and try again; for music, a shorter lyrics text also helps. When the model's memory estimate is known, the message adds what it needs and what the GPU has. See [Memory for audio](../concepts/memory.md#audio).

## Images { #images }

### "--width must be a multiple of 16 (got …)"

Every image model needs a width and height that are multiples of 16. `--height` gives the same message with its own name. Pick a **Ratio** and **Size** instead, or round the number to a multiple of 16. See [Make images](../guides/images.md#pick-a-size).

### "… is not compatible with --upscale …"

With `--upscale`, an explicit `--width` or `--height` is first divided by the upscale factor, and the result must be a multiple of 16. The message gives the size it would end at. Choose a value whose quotient is already a multiple of 16, or use a **Ratio** and **Size**. See [Upscale while generating](../guides/images.md#upscale-while-generating).

### Ideogram 4 returns a grey "blocked by safety filter" frame

This sometimes happens for harmless prompts. Set **First sigma** (`--first-sigma`) to `1.005` or `1.006`. See [Ideogram 4 Instant](models.md#ideogram-4-instant).

### Ideogram 4 makes a flat cut-out on a checkerboard

A plain prompt becomes a sparse caption. Let the enhancer write a full one with **Write as: JSON** (`--enhance-format json`). See [JSON captions](../concepts/prompt-enhancer.md#json-captions-ideogram-4).

### A negative prompt is ignored { #a-negative-prompt-is-ignored }

The picked model, or a variant of it, doesn't use a negative prompt, so it was left out and the job went on. The message may add a hint on what to use instead. See [Negative prompts](../concepts/prompt-syntax.md#negative-prompts).

#### "… models do not support negative prompts"

The model family has no negative prompt, so yours was left out.

#### "… does not support negative prompts; it was ignored."

This model has no negative prompt, so yours was left out; the message may name what to use instead.

## Video { #video }

### "Add a first frame. A last frame needs one." { #add-a-first-frame }

A last frame is set but the first frame is not, so **Generate** stays off. Add a first frame, or remove the last one. See [First and last frame](../guides/video.md#first-and-last-frame).

### "… cannot be extended: …" { #a-video-cant-be-extended }

A clip can't be extended. The message names the file and lists every problem. Common ones: the clip was upscaled (extend the original), it has a variable frame rate, or its size isn't a multiple of 32; every video model needs that, so resize or crop the clip first. Clips made by the app always fit. MiniMax H3 also extends 24 fps clips only; use an LTX model for another rate. See [What can be extended](../guides/video.md#what-can-be-extended).

#### "MiniMax H3 extends 24 fps clips only; the source runs at … fps."

Extend this clip with an LTX model, or use a 24 fps clip.

#### "MiniMax H3 extends clips whose size is a multiple of …; the source is …x…."

Resize or crop the clip to the multiple the message names, or extend it with an LTX model.

### "--model is required: …"

`ziv-video --extend` normally uses the model the source video was made with. Here the video records no model that is available on this computer, and no default video model is set. Pass `-m` with a video model, such as `-m ltx-4`, or set `default_models.video` under `ui` in `config.yaml` (the Config page's **Default video model**). See [Extending a video](../guides/video.md#extend-a-clip).

### "… can't be upscaled: …"

The plain video upscale refuses HDR video, a variable frame rate, interlaced video and non-square pixels. The message names the file and lists each problem. See [Upscaling an existing video](../concepts/upscaling.md#upscaling-an-existing-video).

### "Not enough GPU memory to upscale to …"

Windows and Linux. The size didn't fit even one frame at a time. Free GPU memory, or upscale a smaller video. See [Upscaling an existing video](../concepts/upscaling.md#upscaling-an-existing-video).

### "Can't read the audio track"

ffmpeg couldn't decode the audio file you chose, so the job was refused before the model loads. Choose another file. See [Audio track](../guides/video.md#audio-track).

### ffmpeg failed on a clip { #ffmpeg-failed-on-a-clip }

The message ends with the reason from ffmpeg. The job is marked failed, and a clip may be left in the output folder without the change. Fix the reason, then run the job again. See [Audio track](../guides/video.md#audio-track), [Extending a video](../guides/video.md#extend-a-clip) and [Sound](../guides/video.md#sound).

#### "Adding the audio track failed"

ffmpeg couldn't put the audio track on the clip.

#### "Joining the clips failed"

ffmpeg couldn't join the new part to the source.

#### "Removing the audio track failed"

ffmpeg couldn't take the sound off the clip.

#### "ffmpeg strip-audio failed"

The warning that goes with "Removing the audio track failed".

### "ffmpeg metadata embed failed"

A warning only: the clip was saved, but without the settings recorded in it, so **Reuse settings** has nothing to load from that file. See [Reuse settings](../web-ui/gallery.md#reuse-settings).

## Audio { #audio }

### "… isn't a Delivery mark."

A bracketed word such as `[sadd]` isn't one of the Delivery marks. Speech refuses it before generating, in the Web UI and in `ziv-audio`, because a model would read it aloud. Use one of the marks, or remove the brackets; brackets around other text, such as `[1]`, are read as written. For a pause or a laugh, the Insert menu adds the right text. See [Delivery](../guides/speech-and-sound.md#delivery).

### The model made no sound { #the-model-made-no-sound }

The model returned silence. Change the seed or the text, or for a designed voice the description, and try again; a silent candidate of a voice is skipped and the others are kept. See [Make speech and sound](../guides/speech-and-sound.md).

#### "The model made no sound for this text. Try another seed."

A speech part came out silent. Change the seed or the text.

#### "The model made no sound; skipped."

A designed-voice candidate came out silent and was skipped; the others are kept.

#### "… made no sound. Try a different description."

A sound came out silent. Describe it differently.

### "Peak above full scale; scaled to … dBFS."

Information only: the take was louder than the file format holds and was turned down. Nothing to fix.

## Prompts and prompt files { #prompts-and-prompt-files }

### "Prompts file not found: …. Provide --prompt or a valid --prompts-file."

`ziv-image` and `ziv-video` read `prompts.yaml` in the current folder when you give no `--prompt`, and there is none. Add `--prompt "…"`, or point `--prompts-file` at a prompt file. See [Prompt files](../concepts/prompt-syntax.md#prompt-files).

### "No active prompt sets found. Exiting."

Every set or entry in the file is switched off, so there is nothing to run. Turn one on. See [Turn prompts on and off](../guides/batches.md#turn-prompts-on-and-off).

### A prompt file or snippet is refused { #a-prompt-file-or-snippet-is-refused }

The message names the file and the set or entry; fix that spot. A prompt file maps set names to lists of `- prompt: "…"` entries, with an optional `snippets` block of `name: text` lines. Only a duplicate set name is a warning: the later set replaces the first. See [Prompt files](../concepts/prompt-syntax.md#prompt-files) and [Snippets](../concepts/prompt-syntax.md#snippets).

#### "Failed to parse prompts file …"

The file isn't valid YAML. The message says where the reader stopped; fix the indentation or quote there.

#### "Prompt file is empty or malformed: …"

The file has no sets, or its top level isn't a mapping of set names. Start each set with its name and a colon.

#### "Prompt set name in … is empty. Give the set a name."

A set has a blank name. Give it one.

#### "Prompt set '…' in … must be a list of mappings, got …"

The set isn't a list of entries. Write each entry as a `- prompt: "…"` line.

#### "Entry … in prompt set '…' in … must be a mapping with at least a 'prompt' key, got …"

An entry (counted from 0) is plain text, not a mapping. Write it as `- prompt: "…"`.

#### "Entry … in prompt set '…' in … is missing the required 'prompt' key."

An entry has keys but no `prompt:`. Add one.

#### "Entry in prompt set '…' in … has an empty prompt after snippet resolution."

An entry's prompt is blank once its `$snippets` are filled in. Write a prompt, or define the snippet with text.

#### "'snippets' block in … must be a mapping, got …"

`snippets:` must be `name: text` lines, not a list or a single value.

#### "Duplicate prompt set name … only the last one is used"

Two sets share a name (a quoted `"1"` and an unquoted `1` count as the same), and the last one replaces the first. Rename one.

#### "Undefined snippet …"

A prompt uses a `$name` with no entry under `snippets:`. Define it, or fix the spelling.

#### "Circular snippet reference …"

A snippet uses itself, directly or through other snippets. Break the loop.

### "Invalid 'enhance' in …"

An entry's `enhance:` key must be `true`, `false`, or a mapping of enhance options. In a video run the entry is not enhanced and the run goes on. See [The enhance key](../concepts/prompt-syntax.md#the-enhance-key).

## Prompt enhancer { #prompt-enhancer }

### "Prompt enhancement failed (…); using the original prompt."

The enhancer failed on one prompt, so the job used your own prompt as written and carried on. The text in the brackets says why; if the enhancer was stopped on purpose, no message appears. See [Prompt enhancer](../concepts/prompt-enhancer.md).

### "The enhancer returned no usable rewrite …"

The enhancer's answer was empty, the same as your prompt, or a refusal. The JSON-caption form of the message reads "The enhancer returned no usable JSON caption." Try again, change the options, write as text, or write the prompt yourself. See [Options](../concepts/prompt-enhancer.md#options).

### "The Smart enhancer is unavailable: … Using Fast."

Smart was picked on a machine without enough memory, so the job used Fast instead. The message says what Smart needs and what this machine has. See [Fast and Smart](../concepts/prompt-enhancer.md#fast-and-smart).

### "The prompt enhancer runs on the CPU. …"

Windows and Linux. No usable NVIDIA GPU was found, so the enhancer runs on the CPU. It works, but each prompt can take a few minutes. When the message mentions the driver, install NVIDIA driver 580 or newer. See ["CUDA is not available"](#cuda-is-not-available).

### "Invalid enhancer model '…'. Use REPO or REPO@REVISION."

The model you named for the enhancer has an `@` with nothing on one side of it, such as `owner/model@` or `@abc123`. Name a Hugging Face repository as `owner/model`, or pin it as `owner/model@revision`. The value comes from `--enhance-model` or `prompt_enhancer.user_model` in `config.yaml`; a folder on disk is used as it is, whatever its name holds. See [Fast and Smart](../concepts/prompt-enhancer.md#fast-and-smart).

### "Could not load prompt enhancer model …"

The enhancer's files couldn't be loaded. The message ends with the reason. A model you set yourself with `--enhance-model` or `prompt_enhancer.user_model` may be misnamed or incomplete; remove it to use Fast or Smart. See [Fast and Smart](../concepts/prompt-enhancer.md#fast-and-smart).

### "Enhancer model … is not downloaded and Hugging Face is offline"

The enhancer's model isn't on this computer and the network is down. Connect and run it once to download, or switch to a model you already have. See [Fast and Smart](../concepts/prompt-enhancer.md#fast-and-smart).

## Models, LoRAs and conversion { #models-loras-and-conversion }

### "Could not detect model type: …"

`-m` named something the app can't read as an image model. The message ends with the reason. Use an alias from `ziv-model list`, a Hugging Face repository, or a local folder. A local folder is recognised only in the diffusers layout: it needs a `model_index.json` at its top, and that file needs a `_class_name` field. A single `.safetensors` checkpoint is not a folder it can read; [convert it first](../guides/models.md#convert-a-checkpoint). See [Use your own models](../guides/models.md#your-own-models).

### "Local model directory not found …"

The folder you gave doesn't exist. Check the path. A name that starts with `./`, `../`, `~/` or a drive letter is always read as a folder. See [Models and licences](models.md#how-a-name-is-found).

### "Could not find model '…'. The '…/' directory exists locally"

The name looks like a Hugging Face repository, but a folder with that first part exists here. Put `./` in front if you meant the folder. See [Models and licences](models.md#how-a-name-is-found).

### "model_index.json not found …"

The folder isn't a Hugging Face (diffusers) model folder. Point `-m` at the folder that holds `model_index.json`. See [Your own models](models.md#your-own-models).

### "FLUX.2-dev (32B) is not supported. Use FLUX.2 Klein variants (4B/9B)."

The 32B FLUX.2-dev is too large for this app. Use FLUX.2 Klein 4B or 9B. See [FLUX.2 Klein](models.md#flux2-klein).

### "… family does not support quantization; drop --quantize"

`-q` was used on a model that can't be quantized. Remove `-q`, or pick a model that can be. See [Quantization](../concepts/memory.md#quantization).

### "Failed to load model: …"

The image model couldn't be loaded. The message ends with the reason, often a Hugging Face refusal (see [A model download is refused](#a-model-download-is-refused)) or too little memory (see [Memory and slowdowns](#memory-and-slowdowns)). See [Models and licences](models.md#image-models).

### "Failed to prepare …: …"

Ideogram 4 Instant and MOSS-Image are built from their parts the first time you use them, and that failed. The message ends with the reason. Fix it, then run again; the build restarts. See [Models and licences](models.md#image-models).

### "Cannot build '…': the folder … already exists and was not built by ziv"

A folder you made has the name of the model the app wants to build, so it left it alone. Rename or move the folder, then run again. See [Models and licences](models.md#image-models).

### "Could not detect video model family …"

`-m` named something that isn't a video model the app knows. Use `ltx-4`, `ltx-8`, `ltx25-4` or `h3`, a converted LTX model, a known LTX or MiniMax H3 repository, or a local path containing `ltx`. See [Models and licences](models.md#video-models).

### "LTX on Windows/Linux runs 4-bit or 8-bit: use the model 'ltx-4', 'ltx-8' or 'ltx25-4' …"

On Windows and Linux, LTX only loads at a quant level. Use `ltx-4`, `ltx-8` or `ltx25-4` instead of a repository id. See [Models and licences](models.md#video-models).

### "Remote HuggingFace LoRA references are not supported …"

`--lora` takes a file you already have, not a repository. Import it first with `ziv-model lora --hf`, then use its name. See [Import a LoRA](../guides/models.md#import-a-lora).

### "Empty LoRA entry in --lora value"

The `--lora` value has an empty part, such as a trailing comma. Remove it. See [Use one](../concepts/loras.md#use-one).

### A LoRA has no effect, or a warning says parts were skipped

Check these in turn:

- **Another model family.** A LoRA does nothing on a model it wasn't made for. Only standard LoRAs and LyCORIS LoKr files are supported; other kinds are skipped. See [What works](../concepts/loras.md#what-works).
- **A missing trigger word.** Some LoRAs only act when their trigger word is in the prompt, and the app never adds it for you. Pick it from the **Triggers** menu. See [Trigger words](../concepts/loras.md#trigger-words).
- **A weight that is too low.** A low weight makes a LoRA subtle, so it can look as if it does nothing. Raise its weight toward 1.0. See [Use one](../concepts/loras.md#use-one).

### A LoRA download fails

On the Workspace, the LoRA chip turns red and **Retry** appears above **Generate**. Retry, or remove the LoRA from the chip's menu. A gated LoRA needs the same license and token as a model. See [Discover LoRAs](../concepts/loras.md#discover-loras).

### A converted LTX fine-tune comes out ghosted and noisy

It is a dev checkpoint, which needs Lightricks' distilled LoRA. Convert it again with `--distill`. See [LTX-2.3 and LTX-2.5 video fine-tunes](../guides/models.md#ltx-23-and-ltx-25-video-fine-tunes).

### A converted MiniMax H3 fine-tune comes out blurry or burnt

The turbo choice was wrong. Convert it again with the other choice: with or without `--turbo` (**Already step-distilled**). See [MiniMax H3 video fine-tunes](../guides/models.md#minimax-h3-video-fine-tunes).

## Web UI messages { #web-ui-messages }

### A picture or audio track is missing { #a-picture-or-audio-track-is-missing }

**Generate** stays off, and the reason above it names the picture or track. Choose it again, or for a picture remove it. The messages differ by cause:

- A file chosen from the server was moved or deleted: "This image has moved or been deleted. Choose it again or remove it."
- A file uploaded from your device is no longer in the browser's own storage, where the app keeps uploads so that they survive a reload and a visit to another page. They are gone after the browser's site data is cleared, in a private window, in a different browser or profile, or where the browser blocks storage: "Choose this image again." or "Choose the audio track again."

See [Images](../web-ui/workspace.md#images) and [Audio track](../guides/video.md#audio-track).

#### "The start image is missing."

The start image of an image edit is gone. Choose it again.

#### "The first frame is missing."

The video's first frame is gone. Choose it again.

#### "The last frame is missing."

The video's last frame is gone. Choose it again or remove it.

#### "This image has moved or been deleted. …"

The server file was moved or deleted. Choose it again or remove it.

#### "Choose this image again."

An uploaded image is no longer in the browser's storage. Choose it again.

#### "Choose the audio track again."

An uploaded audio track is no longer in the browser's storage. Choose it again.

### "The Workspace could not load"

The page couldn't fetch its settings from the app. Use **Retry**. If it keeps failing, check that `ziv ui` is still running and look in its terminal for an error. See [Start it](../web-ui/index.md#start-it).

### "Lost the live connection. …"

The page lost its link to the app, for example after the computer slept or `ziv ui` stopped. If it doesn't reconnect, check that `ziv ui` is still running, then reload the page. The job keeps running. See [While a job runs](../web-ui/workspace.md#while-a-job-runs).

### "This file can’t be played."

The browser can't play the file; for a video the cause is often a codec it doesn't support. Open the file in a media player. See [The player](../web-ui/gallery.md#the-player).

### An upscale isn't available { #an-upscale-isnt-available }

The Web UI offers only the upscales this machine can run, so these messages mostly appear to scripts that call the app directly. A creative upscale or an LTX upscale may still work. See [Upscaling](../concepts/upscaling.md).

#### "SeedVR2 upscaling is not available on this machine"

SeedVR2 isn't set up on this computer. Pick another upscale in the Web UI.

#### "Video upscaling is not available on this machine"

The video upscale asked for isn't the one this computer uses: SeedVR2 on Windows and Linux, LTX on macOS.

### "Upscaling a video on this computer needs an LTX-2.3 video model …"

A Mac upscales a video with LTX, and no LTX-2.3 video model is set. Download **LTX-2.3 4-bit** or **LTX-2.3 8-bit** on the Models page. See [macOS: LTX](../concepts/upscaling.md#macos-ltx).

### "No image model is configured to refine the upscale"

A creative upscale needs an image model to redraw detail. Choose a default image model on the Config page. See [Creative upscale](../concepts/upscaling.md#creative-upscale).

### A job control is refused { #a-job-control-is-refused }

Pause, skip or stop can't be used in this state. Approve or discard a render that waits for review, or resume or quit a paused job, then send the control again. See [Review before upscaling](../concepts/upscaling.md#review-before-upscaling) and [The queue](../web-ui/workspace.md#the-queue).

#### "A render is waiting for review: …"

Approve or discard the render, or quit, then send the control again.

#### "Resume or quit the paused job before sending another control"

Resume the paused job, or quit it, first.

### "… written by a newer version of ziv; update ziv to read it"

A file in the data folder was saved by a newer version of the app, so this older one won't read it. Update the app; see [Install](../get-started/install.md#update).

### "The app database … is damaged …"

The file that holds the app's own records, such as the job log, is unreadable. The app moves it aside and starts a new one, so the job log may start empty. Nothing in your outputs is touched. See [App database](data-folder.md#app-database).

## Configuration file { #configuration-file }

### "Failed to parse config file: …" { #failed-to-parse-config-file }

Your `config.yaml` in the data folder isn't valid YAML, so every command stops. The message says where the reader stopped. Fix the indentation or the quote at that line, or move the file away to use the defaults. See [Configuration](configuration.md).

### A setting in config.yaml has the wrong shape { #a-setting-in-config-yaml-has-the-wrong-shape }

Put the settings of a section under its name, indented, and give each key the kind of value the message names: a number, a string or a mapping. See [Configuration](configuration.md#which-value-wins).

#### "Config section '…' must be a mapping …"

A top-level section of `config.yaml`, such as `generation:`, has a single value instead of indented keys under it. Indent its keys beneath it.

#### "config '…' must be …"

The named key has the wrong kind of value, such as text where a number is needed. Change it to the kind the message names.

## Report a problem { #report-a-problem }

Open a [bug report](https://github.com/knuthelge/ZVisionGenerator/issues/new?template=bug_report.yml) with:

- what you did, what you expected and what happened;
- your platform, GPU or Mac model and memory;
- the [error log](data-folder.md#error-logs) of the command, `logs/<command>.log` in the data folder: `ui`, `image`, `video`, `audio` or `model`. Copy it before you start the command again, which empties it.

Check the log for anything private, such as file paths, before you attach it.
