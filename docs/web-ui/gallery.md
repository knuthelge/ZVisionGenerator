# Gallery and viewer

The **Gallery** (++g++ ++g++) shows every image and video in your output folder, including ones the CLI made there. The Workspace's Gallery strip shows the same list, so a new output or a delete shows in both.

![The Gallery page with a grid of images](../assets/screenshots/gallery-page.webp)

## Browse

- **Filter** with **All media**, **Images only** or **Videos only**, and sort **Newest first** or **Oldest first**.
- Scroll down and more load in.
- Click a tile, or press ++enter++ on it, to open the viewer.
- Each tile's **⋯** menu has the same actions as the viewer, led by **Reuse settings**. Right-click a tile with a mouse, or press and hold it on a touch screen, for the same menu.
- When you come back to the Gallery, it shows what it had, scrolled to the same place, and fetches anything new.
- Transparent images show over a checkerboard. It is only a display aid; the file holds real transparency.

## Select and delete

- ++space++ or ++x++ selects the focused tile; ++ctrl+a++ selects everything loaded.
- ++delete++ deletes the selection, or the focused tile, after asking.
- ++esc++ clears the selection.

Deleting removes the file from the output folder.

## The viewer

Clicking an image grows it out of its tile into the viewer. On a computer or iPad the viewer is a panel over the page, as wide as the window less a narrow margin, with the details beside the image; click outside it, or press ++esc++, to close it. On a phone it fills the screen; see [The viewer on a phone](devices.md#the-viewer-on-a-phone).

Step through the set with ++left++ and ++right++, or ++home++ and ++end++ for the first and last. Right-click the image for its menu.

![The viewer with the Details panel open, showing how the prompt was enhanced](../assets/screenshots/viewer-details.webp)

| Action | Does | Key |
|---|---|---|
| **Details** | Shows the settings the file was made with | ++i++ |
| **Reuse settings** | Loads those settings into the Workspace | ++r++ |
| **Use as** | Opens an image as the start image or a reference in Image, or the first or last frame in Video | ++e++ |
| **Extend** | Opens a video in Extend Video | ++e++ |
| **Upscale** | Upscales an image 2× or 4×, or a video 2× (Windows and Linux) | ++x++ then ++2++ or ++4++ |
| **Download** | Saves the file to this device | ++d++ |
| Copy prompt | Copies the prompt to the clipboard | ++c++ |
| Delete | Deletes the file, after asking (++shift+delete++ skips asking) | ++delete++ |

Zoom into an image to see its real pixels: click it on a computer, or pinch or double-tap on a touch screen.

### Details

**Details** lists what the file records: prompt, negative prompt, model, seed, steps, guidance, scheduler, size, LoRAs, start image, audio track with the span of it the clip used (*song.mp3 · 1:30–1:34*), references, quant, upscale and post-processing. An image with a [transparent background](../concepts/images.md#transparent-background) shows *Transparent background: On*. See [Output files](../concepts/data-folder.md#output-files).

When the prompt was enhanced, Details shows how, as one line such as *Fast · Cinematic, Melancholic, Longer*, with **Your prompt** under it.

### Reuse settings

**Reuse settings** loads the file's settings into the Workspace: model, LoRAs, quant, size, seed, references and the rest.

- Post-processing (**Sharpen**, **Contrast**, **Saturation**, **Transparent**), the upscale settings and **Strength** come back too. For an image upscaled in the viewer, only **Transparent** does.

- For an enhanced prompt, **Your text** gets what you typed, **Enhanced** gets the rewrite, and the enhance options come back too.
- For a video, Scene and Action are filled. A video that was itself an extension comes back with its source.

### Use as

**Use as** opens an image in the **Images** section of the Workspace, in the slot you choose:

- **Start image** opens **Image** with it as the start image and puts the prompt in the Prompt box. The text of **Video** stays as it was.
- **Reference** adds it to **Image** as the next reference, and confirms with *Added as image 2*. The prompt and everything else stay as they are. The menu says which number it will get (*as image 2*), and the item is greyed, with the reason (*takes none*, *all 4 in use*), when the image model that **Image** opens on takes no references or has taken as many as it can.
- **First frame** opens **Video** with it as the first frame and puts the prompt in Scene, since it describes the first frame, and leaves Action empty for what happens. The prompt of **Image** stays as it was.
- **Last frame** opens **Video** with it as the last frame. The clip uses it once there is a first frame.

A target the model it opens on can't use is greyed, with the reason beside it, such as *can't use a start image* or *has no last frame*. The arrow keys still stop on a greyed item, so you can read why; ++enter++ on it does nothing.

### Upscale a video

On Windows and Linux, **Upscale** on a video offers one item, such as **2× → 1536×896**, and makes `<name>_2x.mp4` with SeedVR2, keeping the frames, frame rate and sound. A size that has to go one frame at a time is marked **· very slow**. See [Upscaling an existing video](../concepts/upscaling.md#upscaling-an-existing-video).

### Upscale an image

The **Upscale** menu offers:

- **2×** and **4×**: a plain upscale that enlarges faithfully and adds fine detail. Works for any image.
- **2× creative** and **4× creative**: the image model redraws the enlarged image with the image's own settings.

Each item shows the output size, for example `2× → 1664×2432`. The job runs in the Workspace, or joins the queue. The result is saved next to the original as `<name>_2x.png` or `<name>_4x.png`.

How the two kinds differ, and when an option is unavailable, is in [Upscaling an existing image](../concepts/upscaling.md#upscaling-an-existing-image).
