# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Upgrading from 0.13
- **Windows: install again with the new command in [Install](https://knuthelge.github.io/ZVisionGenerator/install/)** to move to PyTorch's CUDA 13.2 build. An upgrade keeps the CUDA 13.0 index, which gets no PyTorch updates after 2.14
- **Stored q4 and q8 image copies are made again once**, on the first job with each model, after the library updates
- **Windows/Linux: the first `ltx-4` or `ltx-8` job needs a connection once**, to link the files already on disk to the pinned LTX-2.3 revision. Nothing downloads again
- **macOS: the same seed can give a slightly different image**, and a LoRA in diffusers format can apply at a different strength, because mflux now reads its alpha

### Added

#### Web UI
- **Recent prompts.** A history button in the prompt box lists the prompts you generated in the current mode (image prompts, or video scenes and actions), newest first, with a filter that searches all of that mode's prompts. Pick one to bring it back: the prompt, a video's scene and the negative prompt, or a JSON caption into the `{ }` box on Ideogram 4. Press Alt+↑ in an empty prompt to open it. **Clear recents** removes the current mode's. They are kept on the server, so they are the same on every device
- **The last five rewrites.** In the Enhanced tab, a `‹ 2 of 3 ›` stepper moves through the last five rewrites of the same text; Generate sends the one shown, with how it was written

#### Video
- **Next, Pause and Repeat for video jobs.** Video jobs in the Web UI and `ziv-video` (**n**, **p**, **r**, **q**, with or without `--review`) take the same controls as image jobs. Next stops the current video at its next step and moves on, Pause waits after the current video, and Repeat makes the same prompt again with a new seed. During a review, Next declines the upscale. A Next or Quit pressed on a video's final step saves the video first. On a Mac, MiniMax H3 text to video cannot stop mid-clip, so Next and Quit take effect when the clip finishes

### Changed

#### Web UI
- **The prompt box bars are reorganised.** The tabs only say which text you see. A status line under the box says what Generate sends, with **Re-enhance** and **Use mine** beside **Out of date**. **Use mine** replaces **Clear**

#### Video
- **macOS: LTX LoRAs apply at full strength.** On `ltx-4`, `ltx-8` and `ltx25-4` a LoRA lost much of its effect to the model's quantization; it now shows fully, at about the same speed. A weight you raised to make up for it may now be too high

#### Platforms and libraries
- **Pillow 13**, which fixes heap overflows in its JPEG and FlashPix decoders that a crafted image could trigger
- **Windows/Linux: diffusers 0.42**
- **macOS: mflux 0.24 and MLX 0.32.4.** MLX 0.32.4 fixes an attention crash on M1 and M2
- **macOS: LTX is updated to ltx-2-mlx 0.18.** On Macs that stream the model (24 GB or less), a LoRA's weights outside the transformer blocks now apply, and image to video needs about 1.4 GB less memory
- **macOS needs macOS 14 Sonoma or newer**, as MLX and PyTorch do

### Fixed
- **Converting a MiniMax H3 checkpoint refuses Ref2VA (reference-to-video) checkpoints,** which used to convert as if they were the normal model
- **Pause after a failed image pauses the job.** A Pause pressed while an image fails now pauses the job, instead of leaving the Pause button stuck on "Pausing"
- **The remaining-time estimate stays sensible.** An image skipped with Next no longer pulls the estimate down, and after Repeats it can no longer go negative; the final count includes the repeats
- **A repeated image can always be reviewed.** When you discard an image and Repeat it, Approve and Discard no longer stay disabled for the new render if the Web UI lost its connection meanwhile
- **Windows/Linux: the LTX-2.3 downloads are pinned to a fixed revision**, so a file Lightricks renames or deletes upstream no longer breaks `ltx-4` and `ltx-8`
- **Windows/Linux: the first Web UI page load is fast.** It no longer waits about 5 seconds when an LTX model is downloaded

## [0.14.0b1] - 2026-10-26

### Added

#### Video
- **A last frame for LTX.** `ltx-4`, `ltx-8` and `ltx25-4` take a **Last frame** in Image to Video, or `--end-image`, as MiniMax H3 does: the clip runs from the first image to the last. Use two images of the same scene, and give a big change a few seconds

#### Images
- **MOSS-Image on Windows and Linux: `moss` (Turbo, 4 steps) and `moss-sft` (50 steps, with a negative prompt).** OpenMOSS's Apache-2.0 text-to-image model, strong at rendering text in English and Chinese, and it runs on a 10 GB card at every quant level. Not available on macOS yet

### Changed

#### Web UI
- **The mascot lives in the logo.** He hops out to greet you in the first-run welcome, to the job card (on a phone, the job bar) while a job runs, waits beside Approve and Discard during a review, and hops back home once he has reacted to the result. His outline keeps his place in the logo while he is out, on every page. He no longer sits in a corner of the latest output
- **A job that ends while you're on another page** shows its "Generation complete" or failure toast there too
- **The live preview updates after every step** instead of at 25, 50 and 75%, so a 10-step Z-Image job shows 9 previews instead of 3

### Fixed
- **Ideogram 4 Instant downloads on the Models page.** Its source download no longer shows under **Other downloads** while `ideo` uses it, and deleting the `ideo` row also deletes that download and, if it isn't built yet, the transformer file waiting beside it, so the space is freed. The delete dialog lists the downloads it removes
- **Building Ideogram 4 Instant never replaces another model.** If a folder called `ideo` that the app didn't build is already in your models folder, the first job that uses `ideo` stops with a message to rename or move it, instead of downloading and then replacing it

## [0.13.0] - 2026-10-25

### Upgrading from 0.12
- **Windows/Linux needs NVIDIA driver 580+ and an RTX 20-series or newer GPU.** GTX 900/10-series, Titan V and V100 are no longer supported
- **Windows/Linux video: `ltx-2.3` is gone.** Use `ltx-4` or `ltx-8`, as on macOS. Only distilled LTX models are supported, so remove a `default_repo` that points at a non-distilled one such as `dg845/LTX-2.3-Diffusers`
- **Video sizes are new, and `m`, `l` and `xl` are larger than before.** Set `--size` or `video_generation.default_size` if you relied on the old sizes. Six image presets also changed slightly
- **Output files are named `<set name>_<YYYY-MM-DD_HH-MM-SS>.png`** (or the video's extension). Settings are no longer packed into the name, so scripts that parse the old names need updating; the Gallery still reads them
- **Some models download or build again once.** `klein9b` downloads again from an ungated mirror (about 35 GB); delete the old `black-forest-labs/FLUX.2-klein-9B` under Models → Other downloads. `ideo` is now Ideogram 4 Instant and is built on its first job. Stored q4/q8 copies, and copies built from local files, are made again once
- **The same seed can give a slightly different image or clip** after the library updates on macOS and on Windows/Linux, listed under Changed

### Added

#### Video
- **LTX-2.5 video: `ltx25-4`.** The newer LTX model, 4-bit, for `ziv-video -m ltx25-4` and the Web UI on macOS, Windows and Linux, with LTX-2.5's own distilled sampler. It fits the same machines as `ltx-4`: a 5 s 1280×704 clip with `--upscale 2` takes about 200 s on an RTX 3080 10 GB. The download is ungated, about 71 GB on Windows and Linux (plus a 22 GiB prepared copy) and 28.8 GB on macOS
- **MiniMax H3 video: `h3`.** MiniMax's open video model with stereo audio, for `ziv-video -m h3` and the Web UI on macOS, Windows and Linux: text or image to video, a last frame (`--end-image` or **Last frame**), Extend from a 24 fps clip, user LoRAs, 4 or 8 steps and the LTX sizes. On Windows and Linux `--upscale 2` enlarges the half-size render in latent space and refines it at full size, and a 5 s 864×480 clip takes about 5 minutes on an RTX 3080 10 GB. On macOS the image modes and Extend run in stages and peak at about 14 GiB (15.5 GiB with a LoRA); a 3 s image-to-video clip at 768×448 takes about 10 minutes on a Mac with 32 GB, and there is no Upscale. The download is about 98 GB (107 GB on macOS), and the first run builds a 4-bit copy. A user LoRA can break an Extend seam. Under the MiniMax H3 Community License
- **`ltx-4` and `ltx-8` on Windows and Linux**, as on macOS, using the same distilled 1.1 model. They need about 20 GB and 27 GB of system memory, and run quantized at 8 steps. With `ltx-4` a 10 GB GPU runs clips up to 20 s at 704×448, 15 s at 960×512 and 10 s at 1408×896, with or without `--upscale 2`. If a level can't fit, the job stops before loading and suggests `ltx-4`
- **Extend a video.** Continue a clip from where it ends, sound included. Open the **Extend Video** tab, choose **Extend** in the viewer or a video's menu (`E`), or run `ziv-video --extend SRC` with `--extend-seconds` or `--extend-frames`. The scene carries over, so you only describe what happens next. Your original clip is never changed, and its part of the new file keeps its quality. Sideways phone clips work, silent clips stay silent, and an extended clip can be extended again
- **Scene and action for video prompts.** A video prompt has an optional **Scene** (who and where, the look, style and camera) next to the **Action** (what happens). In the Web UI the Scene box sits above Action; on the command line use `--scene`, and in prompt files `scene:` (a `$snippet` can share one scene across entries). A scene on its own is enough to generate. **Use as reference** on an image fills Scene with its prompt and leaves Action for what happens
- **Video with sound on Windows and Linux.** Videos include the audio the model generates, also with `--upscale 2`. The **Audio** toggle, `--audio`/`--no-audio` and the prompt enhancer's **Sound** options work there too
- **Upscale an existing video 2× on Windows and Linux.** **Upscale** (or `X` then `2`) on any video in the output folder makes `<name>_2x.mp4` with SeedVR2, keeping the frames, frame rate and sound. The batch size follows your GPU's memory, and a size that only fits one frame at a time is marked **· very slow**. HDR, variable-frame-rate and interlaced clips are refused, and an upscaled video can't be extended
- **Review before upscaling.** See each small render first and approve or discard it before the full-size pass runs: **Review** under Upscale in the Web UI, or `--review` with `--upscale`. Works for images and video on every platform
- **Memory check for video.** Text to Video, Image to Video and Extend Video warn when the chosen size and length may not fit your memory (on Windows and Linux, the GPU or system memory): in the Web UI as a note under the length, on the command line as one warning when the run starts. The model badge estimates a 5 s clip. The job is never blocked
- **RTX 50-series (Blackwell) GPUs** are supported on Windows and Linux
- Windows/Linux, advanced: `video_model_presets.ltx.diffusers.transformer` and `video_model_presets.ltx.diffusers.latent_upsampler` choose which LTX model file and 2× upscaler file to download (`repo` and `filename`)

#### Images
- **Krea 2 Turbo** image model (`krea2`, from an ungated mirror, so no Hugging Face token is needed) on macOS and Windows/Linux: 8 steps, guidance off, LoRAs, q4/q8, live previews, reference images and upscale. On Windows and Linux, `-q 4` lets the text encoder and transformer take turns on the GPU so it runs on 10–12 GB cards. Config `supports_scheduler: false` hides the Scheduler control for models that sample with their own schedule, like this one
- **Ideogram 4 on Windows and Linux.** `ideo` now works with an NVIDIA GPU too (see Changed for the switch to Ideogram 4 Instant)
- **Stored quants.** The first job that runs a model at q8 or q4 saves a quantized copy in the models folder (`<name>@q8`, `<name>@q4`, e.g. `zit@q8`) that later jobs load instead of quantizing again; LoRAs apply on top. Copies are listed on the Models page and deleted with their model. `ziv-model model --quantize 4|8` and **Quantized Copy** on the Models page save one while converting a checkpoint. On macOS the first run, which builds the copy, peaks at about the same memory as later runs. On Windows and Linux q8 stores FP8 weights and streams them from system memory, so any model that runs unquantized also runs at q8, including Krea 2 on 10–12 GB GPUs
- **Upscale an existing image.** **Upscale** in the viewer and the **⋯** menu (or `X` then `2` / `4`) offers **2×** and **4×**, which enlarge the image faithfully with SeedVR2 and need no image model, and **2× creative** and **4× creative**, which let the image model refine the enlarged image with the settings recorded in it. Each option shows the output size and says why when it can't run; sizes that may not fit in memory are marked **high memory**. The result is saved next to the original as `<name>_2x.png` / `<name>_4x.png`. SeedVR2 downloads on first use (17 GB, or 7.3 GB on Macs with less memory; 9.0 GB for the 7B or 3.9 GB for the 3B on Windows and Linux, picked by system memory). Settings: the `seedvr2` section and `upscale.existing_*`, `upscale.max_megapixels` and `sharpening.existing_*`
- **Live previews.** The job card shows the image at 25%, 50% and 75% of the steps, so you can skip a generation early with Next. On macOS for every image model; on Windows and Linux for Z-Image, FLUX.2 Klein and FLUX.2
- **Does it fit?** The model picker and the Models page show whether each model is downloaded and whether it fits this machine's memory (**Fits**, **Tight**, **Too large**) at the selected quantize level
- Windows/Linux: LyCORIS **LoKr** LoRAs, as on macOS

#### Converting checkpoints
- **Convert video fine-tunes.** `ziv-model model -i <file> --model-type ltx-2.3`, `ltx-2.5` or `h3` (and **LTX-2.3**, **LTX-2.5** or **MiniMax H3 (video)** on the Models page, which runs in the background with progress) turns a community checkpoint into a video model you pick by name, on macOS, Windows and Linux. LTX checkpoints are checked offline for being step-distilled, and `--distill` merges the matching distilled LoRA first; for H3 the turbo LoRA is merged unless you say the checkpoint is already step-distilled (`--turbo`, **Already step-distilled**). ComfyUI single files, MiniMax's sharded transformer and diffusers files are read, pruned checkpoints included, as are scaled FP8 and ComfyUI's INT8 (with or without ConvRot), INT4 W4A4 and W4A8, INT6 W6A8, NVFP4 and MXFP8 formats. A checkpoint of another type is refused with the type to pick. A Hugging Face cache file can be picked directly
- `ziv-model model --model-type krea2-turbo` (and the Models page) converts native and ComfyUI Krea 2 Turbo checkpoints

#### Prompt enhancer
- **Enhance prompts with a local LLM.** ✨ **Enhance** in the Web UI rewrites the prompt into an **Enhanced** tab that is generated instead while it has text, or **Enhance each image when generating** rewrites every image's prompt (also from prompt files). On the command line use `--enhance [SPEC]`, `--no-enhance` and `--enhance-model`, and in prompt files a per-entry `enhance:` key. With no options picked it does a general rewrite that adds detail where the prompt is thin. Quoted text such as a sign's lettering comes back unchanged, and rewrites take the frame shape and, for video, the clip length into account
- **Fast or Smart.** **Fast** is a small uncensored Qwen3.5-4B (about 2.4 GB on macOS, 9 GB on Windows and Linux); **Smart** is a larger model that follows your options and wording more closely (Qwen3.5-9B on Windows and Linux, Qwen3.5-27B on macOS). `--enhance-model smart` on the command line, or a Hugging Face model or local folder of your own on the Config page
- **Enhancer options**, grouped as Look, Focus on, Motion, Sound and Output: style (from Natural photo, Candid, Analog film and Studio portrait to Watercolor, Pixel art, Ukiyo-e and Claymation), **Theme**, **Mood**, lighting, composition, color palette, expression, **Length**, and for video **Camera**, **Pace**, **Secondary motion** and **Sound**. A free-text **Direction** (`--enhance-note`, `note:`) steers it, and **Rewrite** can keep your scene and rewrite only the action, the default for Extend and for Image to Video with a scene
- **JSON captions for Ideogram 4.** With `ideo`, **Write as** can write a full Ideogram 4 JSON caption instead of a rewritten prompt (`--enhance-format json`)

#### Web UI
- **Pick a model on first run.** When no image model is downloaded yet, the Workspace opens a welcome dialog listing the image models with their download size, one preselected for your machine. **Download** starts it and makes it your default; **Skip** closes the dialog for good
- **Job queue.** While a job runs, Generate becomes **Add to queue**: change the settings and add more runs, which start one after another. The queue shows under the job card with **Load settings**, remove and **Clear queue**, is shared by every tab, and keeps going when a job fails. New `GET /api/jobs` and `DELETE /api/jobs/queue` endpoints
- **Presets.** Save every Workspace selection under a name (model, quantization, LoRAs, size, runs, steps, guidance, scheduler, post-processing, upscale and the video settings) and bring it back in one step from the **Preset** control at the start of the toolbar, with **Update**, **Save as…**, **Rename…** and **Delete**. Presets are files in the `presets` folder of the data directory, so they follow you across browsers
- **Job log.** The clock button in the top bar (or **G** then **J**) lists every finished job with its workflow, model, prompt, duration and, for a failure, the reason. Filter by Failed or Completed, **Load settings**, **Copy error**, **Delete** or **Clear log**. Entries are kept in the app database `ziv.sqlite3` in the data folder until you delete them
- **Prompts page.** Build prompt files visually: snippets, sets of entries with `$snippet` chips, editable `{a|b}` choices, named fields, an active switch, a negative prompt and full `enhance:` options, all drag-to-arrange, with a preview that shows each prompt as the model receives it. Saving keeps the file's comments and layout and asks before overwriting a file that changed on disk; `⌘Z` undoes. **Generate this one** queues a single prompt. **Save…** in the Workspace's prompt box appends the current prompt to a prompt file, with **Open in Prompts** in its confirmation
- **Model downloads.** Rows on the Models page that aren't fully downloaded have a **Download** button that first shows the size and the free space on the drive; an interrupted download can be deleted and started again. Downloads run one at a time beside your jobs, can be cancelled, and show their progress on the Models page, in the Workspace model picker, on the job card and in the header. A job whose model isn't downloaded yet downloads it as its own step
- **Free disk space from the Models page.** Delete models, LoRAs, converted models and the prompt enhancer's models; the **Size** column shows each one's disk use. **Other downloads** lists Hugging Face downloads nothing here uses, and cache files earlier deletes left behind, with the space each frees. Deletes are refused while a generation runs or a download is fetching that model
- **Phone and tablet.** On a phone the preview fills the screen, with setting chips under it and the prompt and Generate at the bottom; a chip opens one setting at thumb size and **All** the full inspector. Menus and dialogs rise as sheets you pull down to close, the viewer follows your finger (swipe between assets, pull down to close, swipe up for details, hold for the menu), and the Gallery has **Select**. Tablets get finger-sized controls, and an iPad held upright puts the preview on top. The app can be added to the home screen
- **Viewer and history.** ← and →, the arrows over the preview, a swipe or a two-finger trackpad swipe step through earlier outputs in place. The viewer opens out of the tile as a panel, with a **Details** panel (`I`) showing everything recorded in the file, and zooms with a pinch, a double tap or a click for real pixels
- **A file browser in the Web UI.** The folder button next to a path field, and the reference image's picker, open a file browser in the page instead of a system dialog on the server's screen, so it works from a phone or another computer. It has shortcuts to your home, output, model and LoRA folders, recent choices, a filter, an editable path bar and **New folder**. Typing a path suggests matching folders and files (`Tab` completes)
- **Keyboard shortcuts across the app**; press `?` for the list. Among them: `G` then a letter to switch pages, `P`, `N` and `R` to pause, skip and repeat, `E` to extend, `X` to upscale, and `Alt+1`–`3` and `Alt+⇧+1`–`3` to focus or show the Workspace panes
- **Copy CLI command** copies the `ziv image` / `ziv video` command for the current settings, and says when the CLI can't match the job exactly
- **Workspace conveniences.** A reset button beside Generate and a reset per changed setting; drag a setting's label to change it (↑/↓ nudge, Shift ×10); a seed lock; a prompt-file chooser with a filter and per-set All/None; batch size up to 100. The **Add LoRA** menu lists LoRAs made for the selected model first
- **Blob reacts to more.** It reads the prompt from a scroll while it is enhanced, holds a clapperboard for video, nods when a job joins the queue, blends between moods and follows the pointer while idle

#### Command line and setup
- **`ziv ui --background`** starts the Web UI and returns to the prompt; `ziv ui --background-stop` stops every running Web UI. **`ziv ui --host local`** makes it reachable on your network and prints the address
- **`ziv-video` offers to install a missing ffmpeg on Windows and Linux** with `winget`, `apt`, `dnf` or `pacman`, as macOS already did with Homebrew and MacPorts
- **Log files.** Every command writes warnings and errors, with full tracebacks, to `~/.ziv/logs/<command>.log`, cleared at each start. Set `ZIV_LOG=off` to disable it
- **Images and videos record more.** The negative prompt, scheduler, model family, quantization, reference strength, generation time, upscale and post-processing settings, and how the prompt was enhanced. **Reuse settings** brings all of it back. Paths inside your home folder are recorded as `~/…`

### Changed
- **New documentation site,** built with Zensical, in four sections: **Web UI**, **CLI**, **Concepts** and **Developer**
- **macOS: image models stay in memory for the whole job** instead of re-reading and re-quantizing their weights for every image. On a 32 GB Mac, FLUX.2 Klein 9B q8 batches went from 60–80 s to about 28 s per image. MLX's free-buffer cache is capped at 4 GB, and a finished job's memory is returned to macOS, which could freeze 16 GB machines before
- **macOS video.** Each step is about 15–25% faster, with LTX MLX 0.16.3 and the `transformer-distilled-1.1` weights; set `LTX2_COMPUTE_DTYPE=float32` for the old results. `--low-memory` (on by default) streams the transformer from disk on Macs with 24 GB or less. Memory estimates are more accurate. `ltx-4` and `ltx-8` download only the files they load (28.6 GB and 37.8 GB instead of 67.8 GB and 95.6 GB), pinned to fixed revisions
- **Windows/Linux video.** Each step is about three times faster, video no longer looks over-sharpened, and clips are saved at higher quality. Image to Video scales and crops the image to fill the frame instead of stretching it, and `--upscale 2` gives the size you asked for. Pinned host memory is no longer rounded up to a power of two
- **`ideo` is Ideogram 4 Instant, the distilled Ideogram 4.** It makes an image in 8 steps with one transformer instead of 20 steps with two, several times faster and with far less memory, on every platform. The first job builds it in your models folder from an ungated copy (18.6 GB plus the base text encoder). It supports `-q 4` and `-q 8`; on a 10 GB GPU a 1024×1024 image takes about 30 s. Guidance is built in, so `--guidance` is ignored. The model stays under Ideogram's non-commercial license
- **Video sizes and length.** The presets match their aspect ratio and common resolutions: `xs` 576×320 (new), `s` 768×448 (the default), `m` 1024×576, `l` 1280×704 and `xl` 1600×896, and the same turned upright for 9:16; 1:1 runs from 384×384 to 1216×1216. Clips default to 5 s at every size (121 frames, 124 for MiniMax H3), and the Web UI sets length in seconds without the old 256-frame limit
- **Image sizes.** Six presets changed slightly so images keep their exact size when upscaled: 1:1 `l` 1408×1408; 16:9 `s` 960×544, `l` 1920×1088, `xl` 2176×1216; 3:2 `xs` 576×384, `xl` 1920×1280; 9:16 and 2:3 the same turned upright
- **`klein9b` downloads from an ungated mirror** and needs no Hugging Face token or license step
- **Web UI: a redesigned Workspace.** Compose and Settings sit on the left (collapsible to a strip), the preview and live preview fill the middle, and the Gallery runs as a strip under it. The job card shows one progress bar for the whole job, the time spent queueing, enhancing and loading, and a separate bar for the upscale pass. The look is calmer, with fewer outlines, neutral grey surfaces, filled controls and one teal main action per section; dropdowns use the app's own list on every platform
- **Web UI: one Gallery.** The Workspace strip and the Gallery page show one list, so a finished job or a delete lands in both at once. What each file records is kept in `ziv.sqlite3`, so the Gallery opens fast after a restart, and the Gallery, Models and Config pages come back where you left them. `/api/gallery` replaces `/api/history`
- **Web UI: settings follow you.** Settings are kept between visits and restarts. Switching models keeps the settings you changed and takes the new model's defaults for the rest; LoRAs are remembered per model family. Models are shown by name, with the alias under it on the Models page
- **Web UI: the reason is shown.** A refused Generate, a refused settings save and a failed job show the server's reason (for example "CUDA out of memory") instead of a generic message. Notifications appear one at a time at the bottom centre, merge repeats and keep errors until dismissed
- **Web UI: jobs.** The form stays editable while a job runs. `Esc` stops the running job only when pressed twice, and Ctrl-C on the server cancels the running job. Pause, Skip and Cancel work while prompts are enhanced, and Pause while the model loads. **Sharpen** defaults to **auto**, like `ziv-image`
- **Repeat and automatic retries keep the image's `{a|b}` choices and enhanced prompt** and only change the seed
- **Gated models say what to do.** When Hugging Face refuses a model, the Web UI and `ziv-image` / `ziv-video` show one line that names the model, links to its page and says whether to accept its license, run `hf auth login` or sign in again
- **`ZIV_DATA_DIR` moves the outputs folder too,** unless you set an output folder yourself
- **Z-Image on Windows and Linux uses the `beta` scheduler by default**, as on macOS, and `--scheduler beta` follows the macOS schedule
- **macOS memory badges use wider limits:** **Fits** up to 1.1 times Apple's recommended GPU memory and **Tight** up to 1.5 times, so models that run with some swapping show **Tight**
- **Dependencies.** macOS: mflux 0.22 (pinned to a tested minor version) and mlx-lm 0.32; Z-Image runs in bfloat16. Windows/Linux: torch 2.14 with CUDA 13.0, diffusers 0.41, transformers 5.19, accelerate 1.15, bitsandbytes 0.50, peft 0.21. Web server: FastAPI 0.143, uvicorn 0.54. Images, LoRA images and clips can change slightly for the same seed. Building the frontend from source needs Node.js 22.22.2+, 24.15+ or 26+

### Removed
- Windows/Linux video: the `ltx-2.3` alias (use `ltx-4` or `ltx-8`) and non-distilled LTX models
- The base Ideogram 4 model (`ideo` is Ideogram 4 Instant)
- `/api/history` (use `/api/gallery`)
- Windows/Linux: the `hf-transfer` package; downloads use Xet, and `HF_HUB_ENABLE_HF_TRANSFER` has no effect. Set `HF_XET_HIGH_PERFORMANCE=1` for faster downloads
- The unsupported `HF_HUB_TOKEN` and `HUGGINGFACE_HUB_TOKEN` variables no longer count as a Hugging Face token
- Web UI: the YAML text dialog for prompt files; **Edit** opens the file on the Prompts page

### Fixed
- **A token saved with `hf auth login` is found.** The Config and Models pages showed "Not set" unless `HF_TOKEN` was set, although downloads used the saved token
- **Windows: a PyTorch without CUDA is named as the cause,** with the `uv tool install` command for PyTorch's CUDA index, instead of saying the GPU is missing
- **`ziv image` without a prompt says so at once,** before loading the model, instead of a bare "No such file or directory" after the wait
- **Reuse settings brings back exactly what made the image:** its quantization, its LoRAs (or none) by name, its negative prompt, scheduler and JSON caption
- **Windows/Linux: FLUX.2 Klein reference images and upscale work,** as on macOS
- **Windows/Linux: a finished job's model is freed before the next job loads,** a refine or upscale pass no longer switches the rest of the job to float32, and a LoRA that matches none of the model's layers is skipped with a warning
- **Windows: converting a checkpoint no longer fails on a non-Latin file or model name**
- **Converting a scaled FP8 checkpoint works,** and converted models no longer show as not downloaded
- **Image generation works with mflux 0.20 and later,** which failed with `unexpected keyword argument 'control_images'`
- **Deleting a Hugging Face model frees its disk space,** and downloading a LoRA from Hugging Face no longer keeps a second copy in the cache
- **Videos keep their recorded settings,** so the Gallery shows their model, size and seed, and **Reuse** and **Extend** load the same model; macOS clips have the requested number of frames
- **No half-saved video in the Gallery** while a video's settings are being written
- **Web UI: a running job's view goes live again right away when you return to the browser from another app,** instead of waiting a long time to reconnect, on a phone in particular
- **macOS: progress reports each step once it has finished,** so the bar no longer sits still and jumps two steps
- **Sharpen amounts above 1.5 are rejected,** as they broke the sharpening filter
- **A queued quit is no longer replaced** by a later skip, pause or repeat
- **The Add LoRA menu closes** on Escape, an outside click or a choice, and works with the arrow keys

## [0.12.0] - 2026-10-01

### Security
- The Web UI server now rejects requests addressed to unknown hostnames (DNS rebinding) and state-changing requests from other sites, so a page open in the same browser can no longer change settings, queue jobs or read files through the UI. IP addresses, `localhost`, the `--host` name and, for LAN binds, the machine's name are allowed; add others with `ZIV_UI_ALLOWED_HOSTS`
- `/media` now serves only gallery images and videos
- Fixed a command-injection issue in the macOS folder picker when the starting folder had a crafted name

### Added
- `ZIV_UI_ALLOWED_HOSTS` environment variable (comma-separated) for reaching the Web UI through extra hostnames

### Changed
- Gallery and page loads are much faster: model detection uses the local Hugging Face cache first, video models are no longer probed as image models, and thumbnails no longer rebuild the model list
- Native pickers, model conversion and LoRA imports no longer freeze the Web UI or its progress updates while they run, and only one picker can be open at a time
- On Linux and Windows the file picker now runs in a separate process
- Duplicate prompt set names now log a warning (the last one is used, as before)

### Fixed
- Stop and Pause clicked while the model is loading now take effect instead of being silently dropped
- The Workspace no longer stays stuck on "Generating…" when a job's progress stream ends; it recovers the job's real outcome, retries brief connection problems, and shows an "unknown" outcome only if the server restarted
- Finished jobs keep their final status after cleanup, so a page that reconnects later (e.g. after sleep) still sees the real result
- Restarting the Web UI quickly now keeps the same port instead of silently moving to the next one; `--host ::` works
- The Web UI now rejects remote or missing LoRAs like the CLI does, and uses the preset's default upscale steps like the CLI
- Prompt set names such as `2025:`, `true:` or `1.10:` are kept exactly as written instead of crashing the batch
- Settings saves are atomic and safe when made at the same time as other requests, and keep a symlinked `config.yaml` and its permissions
- Image models stored under a path containing "ltx" no longer disappear from the model list

## [0.11.0] - 2026-09-30

### Changed
- Refreshed the Web UI to match Blob's style: teal-tinted surfaces, rounder corners, pill-shaped tabs, chips, and toggles, and round icon badges on settings panels
- Headings, labels, tabs, and buttons now use the rounded Nunito typeface from Google Fonts, loaded without blocking render and falling back to rounded system fonts when unavailable
- Primary actions share one flat teal style with dark-teal text and a subtle lift on hover, including buttons on the Config and Models pages

## [0.10.0] - 2026-09-30

### Added
- Added Blob, an animated Workspace mascot that reacts to generation: thinking while queued or loading, painting while creating, cheering on completion, sad on failure, surprised when stopped, and sleeping while paused or idle
- Blob waves on arrival, watches the prompt while you type, and hops smoothly between spots in the preview as the Workspace view changes
- Latest-output previews now show a loading state until the image has painted instead of an empty stage

### Changed
- Replaced the app icon, favicon, and documentation logo with Blob-based artwork

## [0.9.0] - 2026-09-09

### Added
- Added multi-select prompt-file batches with select-all controls and preservation of valid selections after YAML edits
- Added live run and prompt counters with a compact visual sequence indicator during generation

### Changed
- Expanded prompt-file selection with concise two-line previews and optional full prompt and negative-prompt details
- Generation output previews now open in the shared Workspace lightbox with navigation and focus restoration
- LoRA weights in the Workspace now accept values from -10 through 10

## [0.8.1] - 2026-09-09

### Changed
- Refactored host-local path selection into one compact, accessible selector shared by model conversion, LoRA import, prompt files, and output configuration
- Reorganized the Models page for clearer responsive operation cards and reliable bottom spacing
- Added progressive batch output previews in both the Workspace preview area and its History pane

### Fixed
- Model conversion and local LoRA import now submit the selected host path reliably and preserve retry state after failures
- Configured output directories are now used consistently for generation, previews, Workspace history, and Gallery inventory
- Gallery filtering, pagination, selection, lightbox navigation, concurrent loading, and partial deletion remain synchronized under overlapping operations
- Required model and path controls now use native validation with accessible feedback
- Generation output grids remain scrollable and reachable for large batches and constrained window sizes

## [0.8.0] - 2026-09-08

### Changed
- Refined the Web UI with compact navigation, neutral surfaces, improved text contrast, and the original teal brand accent
- Tightened workspace controls with paired fields and collapsible post-processing, upscale, and video settings
- Expanded preview space with slimmer side panels and responsive workspace controls
- Redesigned generation cards with compact timing, readable stage labels, visual batch tracking, and a subtle progress sweep that respects reduced-motion preferences

### Fixed
- Generation controls now show pending, accepted, and failed requests and prevent duplicate clicks while a request is in flight
- Removed duplicate running-stage messages and conflicting pause/resume controls
- Kept the Generate bar clear of scrolling form controls

## [0.7.0] - 2026-09-08

### Added
- Packaged local Web UI with workspace, gallery, config, and model views
- FastAPI-backed local job APIs, status streaming, and background web runner support for browser-submitted image and video jobs
- Frontend build, test, and packaged static asset verification integrated into the standard Makefile and CI workflows
- Ideogram 4 image generation with structured JSON captions, configurable first-step sigma, and Web UI capability controls
- Windows and Linux video generation through the diffusers/CUDA LTX backend
- Generation provenance metadata for saved image and video outputs

### Changed
- Base installs now include the local Web UI runtime dependencies and packaged SPA assets
- Web configuration and model resolution now expose backend-aware defaults to the local UI without duplicating platform selection logic
- Model aliases may now resolve to different repositories depending on your operating system
- If a model alias is unavailable on your platform, a guidance message now explains why and suggests alternatives
- `ziv model list` now shows platform-specific availability for model aliases
- Bare `ziv` now shows command discovery/help, and the local Web UI is launched explicitly with `ziv ui` or `ziv-ui`
- Frontend dependency management now uses pnpm
- Dependency automation now groups major updates while ignoring minor and patch updates

### Fixed
- Local Web UI job recovery, status streaming, gallery updates, and runtime delivery are more resilient
- Image backend weight casting now handles both `bfloat16` and `float16` tensors correctly
- Model management frontend assets are included in packaged builds

## [0.6.5] - 2026-04

### Added
- New logo asset set for project and documentation branding

### Changed
- README branding updated to use the new logo treatment
- Documentation home page and MkDocs site branding updated to use the new logo assets

### Removed
- Older root-level logo assets replaced by the new documentation asset set

## [0.6.4] - 2026-04

### Added
- Bundled `ltx_core_mlx` and `ltx_pipelines_mlx` directly into the repository so the package installs correctly from PyPI without relying on an external distribution of `ltx-2-mlx`
- Dynamic versioning via `hatch-vcs`: package version is now derived automatically from git tags

### Changed
- `ziv` is now a unified parent command with subcommands: `ziv image`, `ziv video`, `ziv model`
- `ziv-image` is now the primary standalone image generation command (previously `ziv`)
- `ziv-convert` renamed to `ziv-model`

### Removed
- `ziv` no longer acts as the image generation command directly — use `ziv-image` or `ziv image`
- `ziv-convert` entry point removed — use `ziv-model` or `ziv model`

## [0.6.0] - 2026-04

### Added

- Video upscale: distilled-only two-stage 2× spatial pipeline
- Documentation site (MkDocs + GitHub Pages)
- CI/CD workflows: lint/test, automated PyPI release, docs deployment
- Community files: license, code of conduct, contributing guide, security policy
- Issue and PR templates

## [0.5.0] - 2026-04

### Added

- Image-to-video generation (`--image` flag in `ziv-video`)
- Video LoRA support (single, stacked, configurable weights)
- Audio included by default in video generation (`--no-audio` to strip)

## [0.4.0] - 2026-04

### Added

- `ziv-video` entry point for video generation
- Text-to-video with LTX-2.3 on macOS (MLX backend)
- Video size presets (s/m/l/xl × 3 aspect ratios)
- Video model auto-detection
- Low-memory mode for video generation
- Batch video generation from prompt files

## [0.3.0] - 2026-04

### Added

- Reference image steering (`--image` and `--image-strength` flags)
- Image-to-image generation on both macOS and Windows platforms

## [0.2.0] - 2026-04

### Added

- `ziv-convert` entry point for model and LoRA management
- Checkpoint conversion (Z-Image, FLUX.2 Klein 4B/9B formats)
- LoRA import from local files and HuggingFace Hub
- `ziv-convert list` to display installed models and LoRAs

## [0.1.0] - 2026-04

### Added

- Text-to-image generation (macOS via mflux/MLX, Windows via diffusers/CUDA)
- Model store at `~/.ziv/` with bare-name resolution and HuggingFace fallback
- Built-in model aliases with custom alias support
- YAML prompt files with multiple prompt sets and `active` filtering
- Prompt variable syntax (`{option1|option2}`) with nesting
- Structured prompts (dict/list flattening) and reusable snippets
- LoRA support (single and stacked, configurable weights)
- Upscale pipeline (generate small → Lanczos → img2img refine → CAS sharpen)
- 4-bit and 8-bit quantization on both platforms
- Post-processing: contrast, saturation, CAS sharpening
- Interactive keyboard controls during batch runs (skip, quit, pause, repeat)
- `ziv` and `ziv-image` entry points
