# Gallery and viewer

The **Gallery** (++g++ ++g++) shows every image, video and audio file in your output folder, including ones the CLI made there, or just those of the [project](#projects) you have picked. The Workspace's Gallery strip shows the same list, so a new output or a delete shows in both.

![The Gallery page with a grid of images](../assets/screenshots/gallery-page.webp)

## Browse

- **Filter** with **All media**, **Images only**, **Videos only** or **Audio only**, and sort **Newest first** or **Oldest first**. With **All files** picked, the filter also offers **Not in a project**.
- Scroll down and more load in.
- Click a tile, or press ++enter++ on it, to open the viewer.
- Each tile's **⋯** menu has the same actions as the viewer, led by **Reuse settings**. Right-click a tile with a mouse, or press and hold it on a touch screen, for the same menu.
- When you come back to the Gallery, it shows what it had, scrolled to the same place, and fetches anything new.
- Transparent images show over a checkerboard. It is only a display aid; the file holds real transparency.

## Projects

Pick a project with the picker in the top bar, as on every page. What a project is and what it shares with the rest of the app is in [Projects](../concepts/projects.md).

- With a project picked, the Gallery shows only its files, and the line under the title starts with its name. A project with nothing in it says *Nothing in Night Bus yet*.
- Changing the picked project clears the selection, closes the viewer and scrolls back to the top. The filter and sort stay as they are.
- With **All files**, each tile's line leads with its project, such as *Night Bus · 11/8/2026*.

## Audio

An audio take is a tile with an icon for its kind, the words of the take, a thin waveform and its length. A tile doesn't play; open it for that.

![The Gallery filtered to Audio only, with music, speech and sound tiles](../assets/screenshots/placeholder.svg "gallery-audio")

The viewer shows a player: the take's waveform over the browser's own audio controls. Click or drag the waveform to seek. **Details** lists the kind, model, voice, length, seed and format; the **Tempo · Key** of a song (*92 BPM · A minor*), the **Delivery** of a speech take in a built-in voice and the **Sound type** of a sound, each only when the take set it; how its text was written when it was [enhanced or filled in](../concepts/audio.md); and a **Note** when the model could not follow a setting of the take. **Reuse settings** opens the **Audio** tab on the take's kind with its settings; see [Takes and files](../concepts/audio.md#takes-and-files). **Download** and **Delete** work as for other files.

An audio file without this app's settings, such as one you copied into the output folder, plays but has no **Reuse settings**.

### Use as an audio track or a voice

**Use as** on an audio take offers:

- **Audio track** opens **Video** with the file as the clip's [audio track](workspace.md#audio-track). It is greyed, with the reason, when the video model can't follow a track.
- **Voice** (speech takes) opens **Audio** on Speech with the take as the recording and its text as **What it says**. It is greyed for a take longer than 30 seconds. This is how to keep one voice across separate takes; see [Voices](../concepts/audio.md#voices).

## Select, move and delete

- ++space++ or ++x++ selects the focused tile; ++ctrl+a++ selects everything loaded.
- **Move selected** moves the selection to a project; see [Move to project](#move-to-project). The button reads **Move** in a narrower window and in a phone's select bar.
- ++delete++ deletes the selection, or the focused tile, after asking.
- ++esc++ clears the selection.

Deleting removes the file from the output folder.

## Move to project

Move puts existing files into a project, or takes them out of one. The rules are in [Move a file to another project](../concepts/projects.md#move-a-file-to-another-project).

- **One file:** **Move to project…** in its **⋯** menu, which a Gallery tile, the Workspace strip, the preview and the viewer all have. On a computer or tablet the viewer's toolbar also has a **Move** button.
- **Several files:** select them, then **Move selected**.

Choose the target in the list. The place the file is in now is checked and greyed out; for several files from different places nothing is checked. **New project…** asks for a name, then **Create and move** makes the project and moves the files into it.

After the move a notice says where the files went. If some files couldn't be moved it says how many and why, and those stay selected. If none could be moved, the notice gives the reason.

In **All files** a moved tile stays in the list, under its new project. In a project, or in **Not in a project**, a file that no longer belongs leaves the list, and the viewer steps to a neighbouring file.

## The viewer

Clicking an image grows it out of its tile into the viewer. On a computer or iPad the viewer is a panel over the page, as wide as the window less a narrow margin, with the details beside the image; click outside it, or press ++esc++, to close it. On a phone it fills the screen; see [The viewer on a phone](devices.md#the-viewer-on-a-phone).

Step through the set with ++left++ and ++right++, or ++home++ and ++end++ for the first and last. Right-click the image for its menu.

![The viewer with the Details panel open, showing how the prompt was enhanced](../assets/screenshots/viewer-details.webp)

| Action | Does | Key |
|---|---|---|
| **Details** | Shows the settings the file was made with | ++i++ |
| **Reuse settings** | Loads those settings into the Workspace | ++r++ |
| **Use as** | Opens an image as the start image or a reference in Image, or the first or last frame in Video; opens audio as a video's audio track or a speech voice | ++e++ |
| **Extend** | Opens a video in Video, on its Extend kind | ++e++ |
| **Upscale** | Upscales an image 2× or 4×, or a video 2× | ++x++ then ++2++ or ++4++ |
| **Move** | Moves the file to a project. On a phone it is **Move to project…** in the **⋯** menu | |
| **Download** | Saves the file to this device | ++d++ |
| Copy prompt | Copies the prompt to the clipboard | ++c++ |
| Delete | Deletes the file, after asking (++shift+delete++ skips asking) | ++delete++ |

Zoom into an image to see its real pixels: click it on a computer, or pinch or double-tap on a touch screen.

### Details

**Details** lists what the file records, starting with its **Project** (*No project* for a file at the top of the output folder): prompt, negative prompt, model, seed, steps, guidance, scheduler, size, LoRAs, start image, audio track with the span of it the clip used (*song.mp3 · 1:30–1:34*), references, quant, upscale and post-processing. An image with a [transparent background](../concepts/images.md#transparent-background) shows *Transparent background: On*. See [Output files](../concepts/data-folder.md#output-files).

When the prompt was enhanced, Details shows how, as one line such as *Fast · Cinematic, Melancholic, Longer*, with **Your prompt** under it. An audio take shows the same for its Style, Text or Prompt, for example *Fast · Style · Folk*.

### Reuse settings

**Reuse settings** loads the file's settings into the Workspace: model, LoRAs, quant, size, seed, references and the rest.

- Post-processing (**Sharpen**, **Contrast**, **Saturation**, **Transparent**), the upscale settings and **Strength** come back too. For an image upscaled in the viewer, only **Transparent** does.

- For an enhanced prompt, **Prompt** gets what you typed, **Enhanced** gets the rewrite, and the enhance options come back too.
- For a video, Scene and Action are filled. A video that was itself an extension comes back with its source.
- For audio, the settings the take was made with come back: the **Tempo** and **Key** of a song, the length, the **Delivery** of a speech take in a built-in voice and the **Type** of a sound. A setting the take did not set, such as a song made without a Tempo, comes back on **Auto**.

### Use as

**Use as** opens an image in the **Images** section of the Workspace, in the slot you choose:

- **Start image** opens **Image** with it as the start image and puts the prompt in the Prompt box. The text of **Video** stays as it was.
- **Reference** adds it to **Image** as the next reference, and confirms with *Added as image 2*. The prompt and everything else stay as they are. The menu says which number it will get (*as image 2*), and the item is greyed, with the reason (*takes none*, *all 4 in use*), when the image model that **Image** opens on takes no references or has taken as many as it can.
- **First frame** opens **Video** with it as the first frame and puts the prompt in Scene, since it describes the first frame, and leaves Action empty for what happens. The prompt of **Image** stays as it was.
- **Last frame** opens **Video** with it as the last frame. The clip uses it once there is a first frame.

A target the model it opens on can't use is greyed, with the reason beside it, such as *can't use a start image* or *has no last frame*. The arrow keys still stop on a greyed item, so you can read why; ++enter++ on it does nothing.

### Upscale a video

**Upscale** on a video offers one item, such as **2× → 1536×896**, and makes `<name>_2x.mp4`, keeping the frames, frame rate and sound. Windows and Linux use SeedVR2, and mark a size that has to go one frame at a time **· very slow**. macOS uses LTX and offers the item for LTX clips. A size that may not fit memory is marked **· high memory**. See [Upscaling an existing video](../concepts/upscaling.md#upscaling-an-existing-video).

### Upscale an image

The **Upscale** menu offers:

- **2×** and **4×**: a plain upscale that enlarges faithfully and adds fine detail. Works for any image.
- **2× creative** and **4× creative**: the image model redraws the enlarged image with the image's own settings.

Each item shows the output size, for example `2× → 1664×2432`. The job runs in the Workspace, or joins the queue. The result is saved next to the original as `<name>_2x.png` or `<name>_4x.png`.

How the two kinds differ, and when an option is unavailable, is in [Upscaling an existing image](../concepts/upscaling.md#upscaling-an-existing-image).
