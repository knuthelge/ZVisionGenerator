# Workspace

The Workspace is where you make things. Pick a workflow in the top bar, write a prompt under *Compose*, adjust *Settings*, and press **Generate**.

![The Workspace: Compose and Settings on the left, the preview in the middle and the Gallery strip below](../assets/screenshots/workspace.webp)

## Model and LoRAs

The top of the sidebar picks the **Model** for the current workflow, its quantization (**Quant**) and its **LoRAs**.

Each model shows whether it is downloaded and a memory badge:

| Badge | Means |
|---|---|
| **Fits** | The model fits comfortably. |
| **Tight** | It runs, but your computer may slow down while it works. |
| **Too large** | Expect heavy swapping or an out-of-memory error. Try a lower quant. |

The badge follows the quant you pick, so you can see whether `q8` or `q4` brings a large model within reach. Hover it for the estimate. See [Memory](../concepts/memory.md) for how it's worked out.

A model that isn't downloaded yet shows *not downloaded*, and downloads by itself when a job first needs it. While it downloads from the [Models](models.md) page, the picker shows its progress, for example *Downloading 42%*.

The **Add LoRA** menu lists the LoRAs made for the selected model first, with the ones you used with it recently on top, and folds the rest under **Other models (n)**. LoRAs are remembered per model family, and per size for FLUX.2 Klein (4B and 9B keep separate LoRAs): switch to another family and you get the LoRAs you last used with it. See [LoRAs](../concepts/loras.md).

## Presets

A **preset** saves your whole setup under a name, so you can bring it back in one step. The **Preset** control is the first item of the Workspace toolbar, before the model.

![The Preset menu open in the Workspace toolbar](../assets/screenshots/preset-menu.webp){ width="540" }

- **What it keeps:** the model, quant, LoRAs and their weights, size, number of runs, steps, guidance, image strength, scheduler, post-processing and upscale settings including review. For video: the length, audio, low memory and upscale.
- **What it leaves alone:** the prompt, seed, start image, references and workflow.
- **The menu** has **Update** (write the current settings into the preset), **Save as…**, **Rename…** and **Delete**. A dot on the control shows when the form differs from the applied preset.
- Image and video have their own presets; only the ones for the current kind are listed.
- A model or LoRA that is no longer installed is skipped.
- Switching between image and video, **Reset**, starting an extend, and reusing a job's settings deselect the preset. Pick it again afterwards.

Presets are kept as files in `presets` in the [data folder](../concepts/data-folder.md), so they follow you to every browser. The CLI doesn't use them.

## Compose

### Prompt source

**Prompt source** chooses where the prompt comes from:

- **Inline**: the prompt you type in the box.
- **Prompt file**: prompts from a YAML file on the machine running the server.

With a prompt file, *Compose* summarizes the prompts you chose. Click **Prompts…** to open the chooser, where you filter prompts, select whole sets with **All** and **None**, and confirm with **Use N prompts**. **Edit** opens the file on the [Prompts](prompts.md) page. See [Prompts and prompt files](../concepts/prompts.md).

### The prompt box

The box has three bars, each with one job.

**The tabs** say which text you are looking at:

- **Prompt** (**Your text** for video) is what you wrote.
- **Enhanced** is the [prompt enhancer](../concepts/prompt-enhancer.md)'s rewrite. When it has text, Generate uses it.

In the **Enhanced** tab, `‹ 2 of 3 ›` at the right steps through the last five rewrites of the same text; Generate sends the one shown. See [Prompt enhancer](../concepts/prompt-enhancer.md#in-the-web-ui).

**The tools row** under the box holds what you can do with the text. With the JSON box on, **Enhance** and **Save…** are hidden:

- ✨ **Enhance**: rewrite the prompt now, or with **Enhance each image when generating**, rewrite every image's prompt as the job runs. See [Prompt enhancer](../concepts/prompt-enhancer.md#in-the-web-ui).
- **{ } JSON** (Ideogram 4): swap the prompt for a structured caption.
- 🕘 **Recent prompts**: bring back a prompt you generated before. See below.
- **Save…**: append the prompt to a prompt file. See [Save a Workspace prompt](prompts.md#save-a-workspace-prompt).

**The status line** under the box says what Generate sends and counts the words. It shows **Generate uses the Enhanced text** with **Use mine** to go back to your text; **Out of date** with **Re-enhance** and **Use mine**; or **Enhanced when the job runs** with auto-enhance on. Nothing shows while your own text is sent.

For video, *Compose* has a **Scene** box above the **Action** box. See [Scene and action](../concepts/video.md#scene-and-action).

### Recent prompts

The 🕘 button in the tools row lists the prompts you generated in the current mode, newest first: image prompts while you work on an image workflow, video prompts (a scene and an action together) on a video workflow. Each row shows up to two lines of the prompt (or of the scene), how long ago, and a **{ } JSON** badge for a JSON caption. Hover a row to read the whole prompt. A line below shows a video's scene, or an image prompt's **Negative:** prompt when it has one. Type in the filter to search all of that mode's prompts, not only the ones shown. The list shows up to 50 matches.

- **Pick a row** to bring the prompt back: the prompt, a video's scene and the negative prompt go into their boxes, and any Enhanced text is dropped. A JSON caption turns on **{ } JSON** and fills its box. JSON rows are greyed out, with the reason, on a model that takes no JSON captions. Settings, model and workflow don't change.
- **From the keyboard:** press ++alt+up++ in an empty prompt to open the list with the filter ready. ++up++ and ++down++ move, ++enter++ picks and ++esc++ closes and puts the cursor back in the prompt box (or on the button, if you opened the list with it).
- **Clear recents** at the bottom of the list removes the prompts of the mode you are in, after asking. The other mode's prompts stay.

![Recent prompts menu open over the prompt box](../assets/screenshots/placeholder.svg "recent-prompts")

Recents are kept in the app database on the server, not in the browser, so they are the same on every device you open the Web UI from. Prompts that come from a prompt file are not recorded, and neither are refused or blank submits.

### First and last frame

Image to Video shows two slots under **Frames**: **First frame** and an optional **Last frame**. Pick the last frame the same ways as the first: browse the server, upload from your device, drop an image on the slot, type a path, or choose **Use as** → **Last frame** on an image. See [First and last frame](../concepts/video.md#first-and-last-frame).

A **Negative prompt** box appears for models that use one. Ideogram 4 adds a **Prompt (JSON)** mode for a structured caption.

## Settings

*Settings* holds everything else, grouped:

| Group | Has |
|---|---|
| **Size** | **Ratio**, **Resolution**, or a custom **W × H**; **Duration (s)** for video |
| **Sampling** | **Steps**, **Guidance**, **Scheduler**, **Seed**, **Batch size** |
| **Start image** | The image for Image to Image, and **Img strength**. For video, **Frames**: a **First frame** and an optional **Last frame** |
| **References** | Images whose content the result keeps (FLUX.2 Klein, in Text to Image and Image to Image) |
| **Upscale** | The factor, **Denoise**, and **Review** before upscaling |
| **Post-processing** | **Sharpen**, **Contrast** and **Saturation** (images) |
| **Video** | **Audio**, **Low memory** |

Groups and rows only show when the model and workflow use them.

- **Change a number** by typing it, or by dragging its label sideways. On a touch screen, numbers have − and + buttons instead (hold to repeat).
- **Put one setting back** with the reset mark next to it. **Reset** in the *Settings* header puts all settings back to the model's defaults, and keeps the model, LoRAs, quant and prompt.
- **Seed** is random for every run unless you type one. The dice button picks a new one.

What each setting does is in [Images](../concepts/images.md), [Video](../concepts/video.md) and [Upscaling](../concepts/upscaling.md).

### References

When the model takes [references](../concepts/images.md#references), **References** lists up to four image rows, each with a thumbnail, its name, and buttons to browse the server or upload from your device. Below the filled rows is an empty **Add reference** row. You can also drop several images on **Add reference** at once, or drop one on a filled row to replace it. The settings summary shows how many are set. On a phone, the **References** chip opens the same rows; a row whose image is gone has **Replace**, which swaps the next image you choose into that row and keeps its number.

- **Number:** a row's position is its number in the prompt: the first row is "image 1". Removing a row renumbers the rest, and a line under the section reminds you to refer to them that way.
- **A model without references, or a video tab,** hides the section and sends none, but keeps the list for when you come back. A model that takes fewer shows and sends the first ones, and the rest come back with a model that takes more.
- **With a start image,** both apply. See [References with a start image](../concepts/images.md#references-with-a-start-image) for what to expect.
- **Reuse settings** brings the references back. A row whose file has moved or been deleted is marked, and **Generate** stays off until you choose the file again or remove the row. Images you uploaded from your device stay while you visit other pages. After you reload the page the browser can't keep them, so those rows say **Choose this image again** and block **Generate** the same way.

### Settings are remembered

Your settings stay in the browser between visits, as long as the Web UI starts on the same port.

When you choose another model, the settings you changed are kept and the rest take the new model's defaults. A size the new model doesn't offer becomes its largest preset, and a custom size is scaled to fit its limits, keeping its shape. Changing the quant changes nothing else.

**Reset**, **Reuse settings**, **Load settings** and **Extend** apply the model's defaults instead, and use exactly the LoRAs the source was made with.

### Copy as a command

The terminal button in the *Settings* header copies a `ziv image` or `ziv video` command with the current model, prompt, LoRAs and settings, writing to the Web UI's output folder. Paste it in a terminal to run the same job from the [CLI](../cli/index.md).

!!! note "When the command differs"
    The CLI has no inline negative prompt, and runs every prompt in a prompt file rather than only the ones you selected. A notice says when the copied command would do something different. An uploaded start image or reference is named by its file name: replace it with the file's path.

### Paths on the server

Fields that take a path on the server's machine (prompt files, the output folder, checkpoints, LoRAs, start images, references, source videos) have a folder button. It opens a file browser inside the Web UI, so it works the same on every system and from a phone.

- It opens at the field's current path or the matching folder, and lists only the files the field accepts. **Show all** lists the rest; **Hidden files** shows names that start with a dot.
- Your last five choices are under **Recent**.
- Click the path bar or press ++slash++ to type a path.

Typing in the field itself suggests matching folders and files: ++tab++ takes the first, the arrow keys and ++enter++ pick another, and ++down++ in an empty field suggests recent paths and the usual folders.

The start image, the last frame and each reference also have an upload button for a picture on the device you're browsing from. A picture you uploaded as the start image or last frame stays while you visit other pages; a reload clears it.

## Generate

Press **Generate**, or ++ctrl+enter++ (++cmd+enter++ on a Mac). ++ctrl+shift+enter++ generates with a new seed, even when the seed is locked.

If the job can't start with the current settings, for example Image to Image without a start image, the reason shows above the button. It goes away when you change a setting.

### While a job runs

![The job card at step 3 of 4, with Pause, Cancel and Repeat](../assets/screenshots/job-card.webp){ width="420" }

The **live preview** shows the current image taking shape after every step ([Live preview](../concepts/images.md#live-preview)), so you can skip a bad one early. It has the image's own shape, a custom size included. It is a preview only and never saved.

The **job card** beside it shows the progress, the time left and each phase with its timing. When a model isn't downloaded yet, the load step shows the download first, with its size, speed and time left.

| Control | Does | Key |
|---|---|---|
| **Pause** | Waits after the current image or video, until you resume | ++p++ |
| **Cancel** | Stops the job at its next step | ++esc++ ++esc++ |
| **Skip** | Stops the current image or video and moves to the next | ++n++ |
| **Repeat** | Makes the current image or video again with a new seed, after it finishes | ++r++ |

Video jobs have the same controls; see [Controls while a video runs](../concepts/video.md#controls-while-a-video-runs). With [review before upscaling](../concepts/upscaling.md#review-before-upscaling) on, the card also offers **Approve** (++y++) and **Discard** (++n++).

!!! note "Cancelling during a download"
    Cancel stops a download at once. Finished files are kept and the file in progress starts over next time. A job that only waits for a download started from the Models page is cancelled, and the download keeps going.

If a job fails, the card says why. The [Job log](job-log.md) keeps the reason.

### The queue

You can keep working while a job runs. Change the prompt or settings and press **Add to queue** (the Generate button while jobs are active).

- Queued jobs run one after another, in order, each with the settings it had when you added it.
- They are listed under the job card as **Up next**, each with its prompt, size and steps, such as `1600x896 · 8 steps`. Each can be removed, or its settings copied back into the form with **Load settings**.
- **Clear queue** removes them all after asking. It never stops the running job.
- If a job fails, the next one starts anyway.

The queue is shared by every open tab, and lives in the server's memory: stopping the server clears it.

While jobs are running or queued, the manual **Enhance prompt** button is unavailable, because the enhancer needs the same memory. Enhancing inside each job still works.

A job keeps running when you close or reload the tab. Reopen the Workspace and it reconnects.

## The preview and the Gallery strip

Finished images and videos appear in the Gallery strip, newest first. It is the same list as the [Gallery](gallery.md) page, so a new output or a delete shows in both. Scroll to the end and older ones load in. Click one to open the [viewer](gallery.md#the-viewer); **Show all** opens the Gallery page.

- Press ++left++ and ++right++ (or use the arrows over the preview, or swipe sideways with two fingers on a trackpad) to step through earlier outputs in the preview. **Latest** goes back to the newest.
- The **«** button next to *Compose* folds the sidebar into a narrow strip, giving the preview more room. The choice is remembered.

## Model-specific controls

Some models change what the Workspace offers:

- **Ideogram 4 Instant** adds a JSON caption mode and a **First sigma** setting. It has no start image or upscale, and its width and height are limited to 256–2048 in steps of 16. See [Ideogram 4](../concepts/models.md#ideogram-4-instant).
- **Krea 2** hides **Scheduler**, because it samples with its own schedule.
- **Video** adds **Duration (s)**, **Audio** and **Low memory**.
