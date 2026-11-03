# Workspace

The Workspace is where you make things. Pick **Image**, **Video** or [**Audio**](#the-audio-tab) in the top bar, write a prompt under *Compose*, adjust *Settings*, and press **Generate**. A run starts from an image when you add one in [Images](#images). **Image** and **Video** each keep their own prompt, so switching tabs never moves text from one to the other.

**Video** has two kinds, chosen with **Kind** at the start of the toolbar: **New** makes a clip from text or a first frame, and **Extend** continues a clip you already have (see [Extending a video](../concepts/video.md#extending-a-video)). Both kinds share the prompt, the model, the LoRAs and the steps and seed. Each keeps its own inputs, so switching back and forth loses nothing: **New** its size, duration, frames, upscale and audio track, **Extend** its source video, **Extend by** and audio track. Video reopens on the kind you left it on. **Kind** is hidden on a computer that can't extend. On a tablet held upright the toolbar shows only the controls, without the words *Kind* and *Model*. On a phone, a **Kind** chip leads the strip ([Video on a phone](devices.md#video-on-a-phone)).

![The Workspace: Compose and Settings on the left, the preview in the middle and the Gallery strip below](../assets/screenshots/workspace.webp)

## Model and LoRAs

The top of the sidebar picks the **Model** for the current tab, its quantization (**Quant**) and its **LoRAs**.

Each model shows whether it is downloaded and a memory badge:

| Badge | Means |
|---|---|
| **Fits** | The model fits comfortably. |
| **Tight** | It runs, but your computer may slow down while it works. |
| **Too large** | Expect heavy swapping or an out-of-memory error. Try a lower quant. |

The badge follows the quant you pick, so you can see whether `q8` or `q4` brings a large model within reach. Hover it for the estimate. See [Memory](../concepts/memory.md) for how it's worked out.

Every built-in model shows an **ⓘ** right after the model picker. Hover it, or press it, to read the licence and open its text. It follows the model you pick and is missing for a model of your own, such as a converted checkpoint. On a phone, the **Model** list has an **ⓘ** at the end of each built-in model's row, and **All** shows the one beside the picker.

A model that isn't downloaded yet shows *not downloaded*. When a job would download it, **Generate** first asks: the dialog lists each model the job needs to download, such as the model, the prompt enhancer and the background removal models, with its size and the free space on the drive. Each model shows its terms under its name, such as the non-commercial terms of FLUX.2 Klein 9B, with links to the licence, and the dialog says that downloading means you agree to them (see [Licences](../concepts/models.md#licences)). **Download** starts the job and the download; **Cancel** queues nothing. A model that is already downloading, for example from the [Models](models.md) page, isn't asked about again, and its picker shows its progress, for example *Downloading 42%*. When Hugging Face can't be reached to read the sizes, the dialog says it couldn't check the size and offers **Download anyway**.

The **Add LoRA** menu lists the LoRAs made for the selected model first, with the ones you used with it recently on top, and folds the rest under **Other models (n)**. Its last item, **Discover LoRAs…**, opens [Discover LoRAs](#discover-loras). LoRAs are remembered per model family, and per size for FLUX.2 Klein (4B and 9B keep separate LoRAs): switch to another family and you get the LoRAs you last used with it. See [LoRAs](../concepts/loras.md).

### Discover LoRAs

**Discover LoRAs…** is the last item of **Add LoRA** (on a phone, in the LoRAs menu under **All**), and **Add LoRA** opens it straight away when you have no LoRAs yet. What Curated and Hugging Face are is in [LoRAs](../concepts/loras.md#discover-loras).

- The **Installed**, **Curated** and **Hugging Face** tabs each have a search. The first time you open **Hugging Face**, you confirm that you want its results.
- Pick a LoRA to see its details. **Download** gets the file, and **Use in LoRAs** adds it to your chips, downloading it first if needed.

While a LoRA downloads, **Generate** waits for it. If the download fails, the chip turns red and **Retry** appears above Generate.

## Presets

A **preset** saves your whole setup under a name, so you can bring it back in one step. The **Preset** control is the first item of the Workspace toolbar, before the model.

![The Preset menu open in the Workspace toolbar](../assets/screenshots/preset-menu.webp){ width="540" }

- **What it keeps:** the model, quant, LoRAs and their weights, size, number of runs, steps, guidance, image strength, scheduler, post-processing and upscale settings including review. For video: the length, audio, low memory and upscale.
- **What it leaves alone:** the prompt, seed and images.
- **The menu** has **Update** (write the current settings into the preset), **Save as…**, **Rename…** and **Delete**. A dot on the control shows when the form differs from the applied preset.
- Image and video have their own presets; only the ones for the current kind are listed.
- A model or LoRA that isn't installed is skipped.
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

![The Enhance popover with Shot, Angle and Palette set under Framing and Color](../assets/screenshots/enhance-popover.webp){ width="420" }

**The status line** under the box says what Generate sends and counts the words. It shows **Generate uses the Enhanced text** with **Use mine** to go back to your text; **Out of date** with **Re-enhance** and **Use mine**; or **Enhanced when the job runs** with auto-enhance on. Nothing shows while your own text is sent.

For video, *Compose* has a **Scene** box above the **Action** box. See [Scene and action](../concepts/video.md#scene-and-action).

### Recent prompts

The 🕘 button in the tools row lists the prompts you generated in the current mode, newest first: image prompts in **Image**, video prompts (a scene and an action together) in **Video**, whichever kind. Each row shows up to two lines of the prompt (or of the scene), how long ago, and a **{ } JSON** badge for a JSON caption. Hover a row to read the whole prompt. A line below shows a video's scene, or an image prompt's **Negative:** prompt when it has one. Type in the filter to search all of that mode's prompts, not only the ones shown. The list shows up to 50 matches.

- **Pick a row** to bring the prompt back: the prompt, a video's scene and the negative prompt go into their boxes, and any Enhanced text is dropped. A JSON caption turns on **{ } JSON** and fills its box. JSON rows are greyed out, with the reason, on a model that takes no JSON captions. Settings, model and tab don't change.
- **From the keyboard:** press ++alt+up++ in an empty prompt to open the list with the filter ready. ++up++ and ++down++ move, ++enter++ picks and ++esc++ closes and puts the cursor back in the prompt box (or on the button, if you opened the list with it).
- **Clear recents** at the bottom of the list removes the prompts of the list you are in, after asking. Every other list stays.

![Recent prompts menu open over the prompt box](../assets/screenshots/placeholder.svg "recent-prompts")

Recents are kept in the app database on the server, not in the browser, so they are the same on every device you open the Web UI from. Prompts that come from a prompt file are not recorded, and neither are refused or blank submits.

A **Negative prompt** box appears for models that use one. Ideogram 4 adds a **Prompt (JSON)** mode for a structured caption.

## Settings

*Settings* holds everything else, grouped:

| Group | Has |
|---|---|
| **Size** | **Ratio**, **Resolution**, or a custom **W × H**; **Duration (s)** for video |
| **Sampling** | **Steps**, **Guidance**, **Scheduler**, **Seed**, **Batch size** |
| **Images** | In Image, the **Start image** with its **Strength**, and [references](#references) (FLUX.2 Klein). In Video, a **First frame** and an optional **Last frame**. See [Images](#images) |
| **Audio track** | In Video, either kind, an audio file the clip follows, with **Start at**. See [Audio track](#audio-track) |
| **Upscale** | The factor, **Denoise**, and **Review** before upscaling |
| **Post-processing** | **Sharpen**, **Contrast**, **Saturation** and **Transparent** (images) |
| **Video** | **Audio**, **Low memory** |

Groups and rows only show when the model and tab use them.

- **Change a number** by typing it, or by dragging its label sideways. On a touch screen, numbers have − and + buttons instead (hold to repeat).
- **Put one setting back** with the reset mark next to it. **Reset** in the *Settings* header puts the open tab [back to its defaults](#reset). The number beside it counts the settings that differ from their defaults, and a phone shows the same number on its *All* sheet.
- **Seed** is random for every run unless you type one. The dice button picks a new one.

What each setting does is in [Images](../concepts/images.md), [Video](../concepts/video.md) and [Upscaling](../concepts/upscaling.md).

### Transparent background

**Transparent** in **Post-processing** switches on [Transparent background](../concepts/images.md#transparent-background). A **Non-commercial** button with an **ⓘ** sits beside it; hover it, or press it, to read the cut-out models' terms and open the dataset terms. The job card shows **Downloading background removal** while the cut-out models download and **Removing background…** while the cut-out runs.

![The Transparent switch in Post-processing, with its Non-commercial button](../assets/screenshots/placeholder.svg "transparent-background")

On a phone, the switch is in the **Post-processing** section of **All**. While it is on, a **Transparent** chip joins the settings strip; its popover, **Transparent background**, holds the switch.

### Images

The **Images** section holds every image the run uses, in **Image** and in **Video**. What an image is for, and how it changes the result, is in [Images](../concepts/images.md) and [Video](../concepts/video.md#first-and-last-frame).

With no image, **Generate** makes the picture or clip from the prompt. With a start image or a first frame, it starts from that image. The [Gallery](gallery.md) and the [Job log](job-log.md) label each job by what ran, such as *Image to image* or *Text to video*.

Folded, the header's summary says what **Generate** will use: in Image *None*, *pond.png*, *pond.png · 2 references* or *2 references*; in Video *None*, *dawn.png* or *dawn.png → dusk.png*. It turns red when an image the run uses is missing.

**In Image**

- **Start image:** an *Image* row with a thumbnail, the name, buttons to browse the server and to upload from your device, and × while an image is set. The empty row reads *Browse or drop*: drop an image on it to set it.
- **Strength** is a row under the start image's, shown only while a start image is set.
- **[References](#references)** follow, for models that take them (FLUX.2 Klein only).

**In Video**

- **First frame:** an *Image* row, as for a start image.
- **Last frame:** the same row, marked *Optional*, to end the clip on a second image.
- **Last frame without a first frame:** if you remove the first frame, the last frame stays, greyed, with *Not used without a first frame*. It is used again when you add a first frame.

**Where images stay**

- An image stays until you press its × or **Reset**, including after you reload the page and including images uploaded from your device.
- **Image** and **Video** each keep their own images. Adding a first frame in Video doesn't put it into Image, and switching between **Image** and **Video**, between its kinds, or changing the model, clears none of them.
- **Reuse settings** and **Load settings** replace the images of the tab they open with the ones the job used. A job made from text leaves that tab's images empty.
- **Reset** in the *Settings* header clears the open tab's images along with its other settings; the other tab keeps its images. What it clears on each tab is in [Reset](#reset).

**Images a model can't use**

An image the model can't use is hidden: a start image on Ideogram 4, references on a model without them, a first frame on a video model that can't start from an image, a last frame on a video model without one. It is not shown, not sent and not counted in the summary, but it stays in the draft and comes back on a model that can use it. When the model can use no image at all, the **Images** section is hidden.

**Missing images**

An image that **Generate** would use but can't find shows a mark in its row and blocks **Generate**, with the reason above the button, such as *The start image is missing.* A file on the server that has moved or been deleted says *This image has moved or been deleted. Choose it again or remove it.* An upload whose file the browser can't read back says *Choose this image again.* Choose the file again or remove the image. A last frame blocks only while there is a first frame.

**On a phone,** one **Images** chip in the strip opens the same groups in a popover. See [Other devices](devices.md#images-on-a-phone).

#### References

When the model takes [references](../concepts/images.md#references), the **Images** section has a **References** group of up to four rows, each like the start image's row. The empty row after them, *Add reference*, adds the next one. You can also drop several images on it at once, or drop one on a filled row to replace it. On a phone, **Replace** on a row whose image is gone swaps the next image you choose into that row and keeps its number.

- **Number:** a row's position is its number in the prompt: *Reference 1* is "image 1". Removing a row renumbers the rest, and a line under the rows reminds you to refer to them that way.
- **Fewer references than you set:** the Video tab uses none, and a model that takes fewer shows and sends the first ones. The rest come back with a model that takes more.
- **With a start image,** both apply. See [References with a start image](../concepts/images.md#references-with-a-start-image) for what to expect.
- **Add one from an image you made:** **Use as** → **Reference** in the [Gallery or viewer](gallery.md#use-as) adds it as the next number and keeps the prompt.

### Audio track

The **Audio track** section comes after **Images** in **Video**, and after **Source video** in the **Extend** kind. Give it a file by path, **Browse**, upload or drop (WAV, MP3, M4A, FLAC or OGG), and set **Start at** in seconds. The track's length shows beside its name to help you choose **Start at**. What the track does is in [Audio track](../concepts/video.md#audio-track).

In the **Extend** kind, choosing a source made from a track fills in that track from where its sound ended; a track you chose yourself stays when you change the source. [Reset](#reset) clears the track, except one the source filled in.

A server file that can't be read is refused at **Generate** with *Can't read the audio track*. An upload lost on reload blocks **Generate** with *Choose the audio track again.* On a phone, the **Track** chip opens the same rows in a popover.

### Settings are remembered

Your settings stay in the browser between visits, as long as the Web UI starts on the same port.

When you choose another model, the settings you changed are kept and the rest take the new model's defaults. A size the new model doesn't offer becomes its largest preset, and a custom size is scaled to fit its limits, keeping its shape. Changing the quant changes nothing else.

**Reuse settings** and **Load settings** apply a take's settings exactly. What the take did not record takes its default instead of what the form held: a take made without LoRAs, a negative prompt or a seed comes back without them, and a size the model doesn't offer becomes its largest preset. They replace the open tab's settings, images, prompts, LoRAs, quant and *Enhance* options, and leave the other tabs, the prompts file and your enhancer choices as they are. **Extend** brings the LoRAs the source was made with, and **Use as** and a link change only what they name.

### Reset

**Reset** in the *Settings* header puts the open tab back to its defaults. It touches that tab only: nothing on another tab, and nothing in the other mode, changes.

| Tab | Reset puts back |
|---|---|
| Image | The settings at the model's defaults; the start image, the references and **Strength**; Image's *Enhance* options |
| Video, New | The settings at the model's defaults; the first and last frame and the audio track; Video's *Enhance* options |
| Video, Extend | The settings at the model's defaults; the length of the extension, to the source's default; the audio track, to the one the source filled in; Video's *Enhance* options |
| Audio | The length, the voice (the recording, what it says, *Any* or *From a recording*, and the tick), **Batch size** and **Seed** |

**Batch size** and **Seed** are shared by every tab, so each Reset puts them back. The applied preset is deselected.

Reset keeps the model, LoRAs and quant, the prompts and the prompt file (the Enhanced text included), the Extend source, the Video kind that is open, the *Enhance* choices of enhancer, format and auto-enhance, and the Audio tab's kind, models and text. Image and Video each have their own *Enhance* options, so a Reset on one leaves the other's.

### Copy as a command

The terminal button in the *Settings* header copies a `ziv image`, `ziv video` or `ziv audio` command with the current model, prompt, LoRAs and settings, writing to the Web UI's output folder. Paste it in a terminal to run the same job from the [CLI](../cli/index.md).

!!! note "When the command differs"
    The CLI has no inline negative prompt, and runs every prompt in a prompt file rather than only the ones you selected. A notice says when the copied command would do something different. An uploaded start image or reference is named by its file name: replace it with the file's path.

### Paths on the server

Fields that take a path on the server's machine (prompt files, the output folder, checkpoints, LoRAs, source videos) have a folder button. It opens a file browser inside the Web UI, so it works the same on every system and from a phone.

- It opens at the field's current path or the matching folder, and lists only the files the field accepts. **Show all** lists the rest; **Hidden files** shows names that start with a dot.
- Your last five choices are under **Recent**.
- Click the path bar or press ++slash++ to type a path.

Typing in a path field suggests matching folders and files: ++tab++ takes the first, the arrow keys and ++enter++ pick another, and ++down++ in an empty field suggests recent paths and the usual folders.

## The Audio tab

**Audio** is the last tab in the top bar. It makes music, speech and sound; what each kind is, the models and the limits are in [Audio](../concepts/audio.md). The tab is hidden on a machine that can't run an audio model.

![The Audio tab on Music, with Style and Lyrics under Compose and the status line saying Song](../assets/screenshots/placeholder.svg "audio-music")

The toolbar reads **Kind**, **Model**, the model's **ⓘ** licence mark and its memory badge. **Kind** is **Music**, **Speech** or **Sound**, and decides what *Compose* and *Settings* show. Presets, **Quant** and LoRAs aren't used. The tab opens on the kind you left it on and remembers the model you picked for each kind. A model that isn't downloaded yet is asked for when you press **Generate**, with its size and licence, as for [images](#model-and-loras).

### Compose

| Kind | Compose | Settings |
|---|---|---|
| **Music** | **Style** (required) above **Lyrics** (optional) | **Batch size**, **Seed** |
| **Speech** | **Text**, with a count against the 10,000 characters it can hold | **Voice**, then **Batch size** and **Seed** |
| **Sound** | **Prompt**, what you hear | **Length (s)**, from 1 to 380 and 10 to begin with, then **Batch size** and **Seed** |

- **Music:** a line under the boxes says what you will get: *Song · length follows the lyrics*, or *Instrumental · 8 sections · add or remove section tags to change its length*. Write words under `[verse]` and `[chorus]` tags for a song; leave **Lyrics** empty, or write only tags, for an instrumental; `[Instrumental]` and `[Solo]` tags keep vocals out more reliably than `[verse]` and `[chorus]`, and a take can still occasionally carry a voice, so generate again if it does. See [Music](../concepts/audio.md#music).
- **Speech:** the help under **Text** says that long text is read in parts and joined into one take. Expressive tags such as `<|sfx:laughter|>` are typed into the text; see [Expressive tags](../concepts/audio.md#expressive-tags).
- **Compose tools:** there is no enhancer, JSON mode or negative prompt, and no steps or guidance to set. The tools row has only **Recent prompts**: music, speech and sound each keep their own list, a song's recent prompt brings its lyrics back with its style, and **Clear recents** clears only the open kind's list.

### Voice

**Voice** is the first group in Speech's *Settings*. See [Voices](../concepts/audio.md#voices) for what each choice does.

1. Leave it on **Any** to let the model pick a voice. The help under it points to **Use as** → **Voice** in the [Gallery](gallery.md#use-as-an-audio-track-or-a-voice) for keeping one.
2. Or choose **From a recording**. Add a file of 3 to 30 seconds (Browse, upload or drop; its length shows beside its name), type **What it says**, and tick *This is my voice, or I have the speaker's permission to clone it.*

![The Voice section on From a recording, with the file, What it says and the consent tick](../assets/screenshots/placeholder.svg "audio-voice")

The tick is asked again after a new recording, **Reset**, a reload, or a **Reuse settings**, **Load settings** or **Use as** that replaces the voice. Reusing a song or a sound leaves the voice and its tick as they are. A take made with **Any** that you use as the voice needs no tick.

A clean, close-miked recording gives the best voice; see [Clone from a clean recording](../concepts/audio.md#voices).

**Reset** keeps the kind, the models and the text you wrote (Style, Lyrics, Text and Prompt), and puts back the length, **Batch size**, **Seed** and the voice. See [Reset](#reset).

### Why Generate is off

While something is missing, **Generate** is disabled and one reason shows above it, always the first that applies: the style, the text or prompt, the recording, its length, **What it says**, the tick, and then text over the limit (*Shorten the text: it's over 10,000 characters, about 10 minutes of speech.*).

### While a take is made

The job card counts **takes** (*Take 1 of 2*) and puts the text of the take under **Style** (music), **Text** (speech) or **Prompt** (sound). Its rows use the same words as the preview strip:

- **Music** has four rows: *Planning*, *Writing the song*, *Rendering* and *Finishing*. The row at work shows how much of the song is written so far (*Writing the song · 1:15 so far*) and then the steps of the render; rows still to come read *Next*.
- **Speech** has one row, *Reading part 2 of 4 · 0:16*, which reads *Finishing part 2 of 4* while that part is decoded. Text in one part reads *Reading · 0:16*.
- **Sound** has one step line as an image does, *Step 5 of 8*, and then *Finishing up…*.

The bar in the card's header is the whole take's progress, the same as the strip's: it keeps moving while a song is written, it never goes back between the parts of a speech, and a sound's reaches full as its last step ends. When a job is paused between takes, the card says which comes next (*Next: take 2 of 3*).

The preview strip shows slow, drifting ribbons of light over a faint starfield. The part of the take already made glows and the rest stays faint. Music is amber, teal and coral bands, speech a coral ribbon that swells with the words, and sound a wide teal mist that brightens with each step. The sky changes with each stage: a haze while the model loads, stars coming out while a song is planned, a glow along the bottom that rises as the take is written, a faint far layer of ribbons while a song is rendered, and a clear sky with one line down the middle while the take is decoded (speech does this for each part). When the take is ready, the ribbons settle into its waveform, captioned *Last finished take* with its length; the same waveform and caption stay between takes and while paused. A job that fails or is stopped keeps the waveform of the last take it saved. With reduced motion turned on in your system, the strip only changes when the job reports progress, and shows the waveform at once when the take is ready. **Pause**, **Skip**, **Repeat** and **Cancel** work as for [images](#while-a-job-runs); Cancel also stops a take that is in progress.

A job that runs out of GPU memory fails with a message that names the model and says to close other apps that use the GPU. For music it also says to shorten the lyrics.

### The result

![A finished take in the player, with its waveform and the browser's audio controls](../assets/screenshots/placeholder.svg "audio-take")

The preview shows the finished take as a **player**: its waveform with a playhead, over the browser's own audio controls. Click or drag the waveform to seek. Nothing plays by itself. **Reuse settings** and the **⋯** menu sit in the corner of the panel, and the caption gives the words, model, length and seed. A take that reached YuE2's limit carries **⚠ Cut at 6:00**.

Takes appear in the Gallery strip as audio tiles. See [Audio in the Gallery](gallery.md#audio).

### On other screens

On a tablet held upright, *Compose* and *Settings* sit side by side under the player. On a phone, **Audio** is the third choice in the top strip's workflow menu, and the strip's chips are **Kind**, **Model**, **Voice** (Speech) or **Length** (Sound), **Seed** and **Batch**, then **All**. See [Other devices](devices.md#audio-on-a-phone).

## Generate

Press **Generate**, or ++ctrl+enter++ (++cmd+enter++ on a Mac). ++ctrl+shift+enter++ generates with a new seed, even when the seed is locked.

If the job can't start with the current settings, for example because the start image is missing, the reason shows above the button. It goes away when you change a setting.

### While a job runs

![The job card at step 3 of 4, with Pause, Cancel and Repeat](../assets/screenshots/job-card.webp){ width="420" }

The **live preview** shows the current image taking shape after every step ([Live preview](../concepts/images.md#live-preview)), so you can skip a bad one early. It has the image's own shape, a custom size included. It is a preview only and never saved.

The **job card** beside it shows the progress, the time left and each phase with its timing. When a model isn't downloaded yet, the load step shows the download first, with its size, speed and time left. Jobs that are already in the queue don't ask; they download when their turn comes.

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

- **Ideogram 4 Instant** adds a JSON caption mode and a **First sigma** setting. It has no start image or upscale, so the **Images** section is hidden, and its width and height are limited to 256–2048 in steps of 16. See [Ideogram 4](../concepts/models.md#ideogram-4-instant).
- **Video** adds **Duration (s)**, **Audio**, **Low memory** and, on models that follow one, an [**Audio track**](#audio-track).
- **Audio** replaces the prompt box and settings with those of its [kind](#the-audio-tab).
