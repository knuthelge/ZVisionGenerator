# Make images

Set the size, steps and seed, start from an image you have, keep a person or style from other images, edit a picture you made, finish the picture and cut out its background. The controls are described in [Workspace](../web-ui/workspace.md); this page says what they do and how to use them.

Everything here is in the **Image** tab's settings. **On a phone,** tap a chip in the strip under the preview, or **All** for every setting; see [Phones, tablets and other computers](../web-ui/devices.md#on-a-phone).

## Pick a size { #pick-a-size }

Choose a **Ratio** and a **Resolution**. The default is `2:3` portrait at `m`.

| Preset | 1:1 | 16:9 | 9:16 | 3:2 | 2:3 |
|---|---|---|---|---|---|
| `xs` | 512×512 | 672×384 | 384×672 | 576×384 | 384×576 |
| `s` | 704×704 | 960×544 | 544×960 | 864×576 | 576×864 |
| `m` | 1024×1024 | 1344×768 | 768×1344 | 1216×832 | 832×1216 |
| `l` | 1408×1408 | 1920×1088 | 1088×1920 | 1728×1152 | 1152×1728 |
| `xl` | 1600×1600 | 2176×1216 | 1216×2176 | 1920×1280 | 1280×1920 |

For another shape, type an exact **W × H**. Every image model needs a width and height that are multiples of 16, and Ideogram 4 also limits each side to 256–2048 pixels.

Larger images take longer and need more memory. To get a large one quickly, make it smaller and [upscale it](#upscale-while-generating).

To change the size every new image starts with, set **Base resolution** on the [Config page](../web-ui/config.md).

## Set steps, guidance and seed

All three are in **Sampling**. The defaults suit each model, so change one only when you have a reason.

- **Steps** is how many denoising steps the model takes. More steps are not always better on the distilled models; each model's default is in [Models and licences](../reference/models.md#image-models).
- **Guidance** is how strictly the model follows the prompt. Ideogram 4 and MOSS-Image Turbo ignore it and warn if you set one.
- **Seed** picks the starting noise. The same model, prompt, settings and seed give the same image. Without a seed every image gets a random one, recorded in the file, so **Reuse settings** can bring it back.

??? note "Scheduler"

    **Scheduler** decides how the steps are spaced. Z-Image and Krea 2 default to `beta`, and `simple` spaces them evenly. Most other models have no choice.

## Start from an image { #start-from-an-image }

A start image is a starting point: the model starts from it and moves toward your prompt, keeping its layout and colours. Add one in the **Images** section of **Image**: browse the server, upload from your device or drop a file on the row. **On a phone,** the **Images** chip holds it.

A low **Strength** stays close to the image; a high one follows the prompt more.

```text
Start image: a pencil sketch of a harbour
Strength: 0.8
Prompt: A watercolour of the same harbour at dusk
```

The image is resized to the target size, a photo is turned upright by its orientation tag, and transparent areas are filled with white. Ideogram 4 doesn't take a start image.

The start image stays until you remove it. To start from a picture you already made, open it in the viewer and choose **Use as** → **Start image** (++e++), which also brings its prompt; see [Use as](../web-ui/gallery.md#use-as).

## Keep a person, object or style { #references }

A start image sets the layout. A **reference** is different: its *content* is reused, such as a person, an object, a place or a style. You can give up to four, alone or combined, for example a woman from one photo, a backpack from another and a pier from a third.

Only FLUX.2 Klein 4B and 9B (`klein4b`, `klein9b`) take references. Pick one of them and a **References** group appears in the **Images** section; on other models it is not shown. **On a phone,** references are in the **Images** chip.

1. Add each image to **References**. A row's position is its number: the first is "image 1", the second "image 2".
2. Name them in your prompt, in order, and say which image supplies what: "the woman from image 1 wearing the backpack from image 2, on a rainy street". The [prompt enhancer](../concepts/prompt-enhancer.md#what-the-enhancer-knows-about-the-job) keeps these phrases.

To add an image you made, use **Use as** → **Reference** in the viewer.

- **What works:** a person, an object, a place and a style, alone or together. A style works best next to one other reference; with four references it tends to be ignored.
- **Size:** each reference keeps its own aspect ratio and is scaled to at most about 1 megapixel, and is prepared as a start image is. The result has the size you chose.
- **Cost:** every reference makes each step slower and uses more memory, so use as few as the picture needs.
- **Files:** PNG, JPEG or WebP.
- **LoRAs** keep working.

??? note "Using references with a start image"

    ### References with a start image { #references-with-a-start-image }

    You can use both. The start image sets the layout and **Strength** how much of it stays. To put a referenced subject in a new scene, use references without a start image.

??? note "References with an upscale, and Reuse settings"

    ### Upscale and Reuse

    An upscale made while generating refines with the same references, so what they fix stays the same in the larger image. [Upscaling a finished image](../concepts/upscaling.md#upscaling-an-existing-image) later doesn't use them, but the result records them, so **Reuse settings** brings them back. A reference whose file has since moved or been deleted keeps its row, so the numbers in your prompt still match.

## Edit a picture { #edit }

Edit changes part of a picture you already have, or gives it more room, and saves the result as a new file. It works on FLUX.2 Klein only (`klein4b`, `klein9b`), which sees the picture while it draws, so what it adds fits what is around it.

| You give | The result |
|---|---|
| **Paint**: the area to change | That area is redrawn from your prompt |
| **Extend**: one side to grow, by 128 to 512 pixels | The picture grows on that side, and the new area is drawn from your prompt |
| **Neither** | The prompt changes the whole picture ("make it night") |

1. **Choose the image.** In **Image**, set **Kind** to **Edit**, then pick the picture in **Image to edit**: **Browse**, **Upload** or **Latest output**, or drop a file on it.
   To edit a picture you made, press **Edit** in the viewer (++m++) instead; it opens here with an empty prompt.
2. **Paint or extend.** Press **Open painter**, work in one mode and press **Done**. **Image to edit** then says what will be sent. The painter's tools are in [The painter](../web-ui/workspace.md#the-painter).
3. **Write the prompt and press Generate.** Write what should be there ("a glass vase of red tulips"), or, with nothing marked, what should change. The prompt is Edit's own and starts empty. It runs as an ordinary job.

Paint and Extend are one at a time. The rest of the picture stays as it was, and the new drawing fades into it over a few pixels around the paint.

- **Trying again:** press **Generate** again; the image and the paint stay. Choosing another image clears the paint, **Clear** in the painter does too, and **Reset** puts back the whole Edit kind, the image included.
- **Removing an object** can be unreliable: try again, or describe what should be there instead.
- **Extend** works best on backgrounds, textures, interiors and streets; on a close-up it may not complete the subject. On a large photo the new strip is drawn at about one megapixel, so it comes out softer than the photo beside it, most visibly on textured content such as foliage or stone. A line in perspective that runs into the new strip, such as a window frame, can break at the join.
- **A picture with transparent parts** (a cut-out) keeps its transparency, so anything drawn outside its outline disappears. It can't be extended, and needs paint: **Generate** waits until you paint.

An edit uses the prompt, the model, **Quant**, LoRAs, **Steps**, **Seed** and **Batch size**. It uses none of the size, **Strength**, **Guidance**, upscale, **Enhance** or post-processing options. The result keeps the picture's size, or grows by the Extend amount; larger pictures take longer.

An edit is saved as `<name>_edit.png`, with a counter when that name exists. In the Web UI it goes beside the original when that is a picture in the output folder, and into the chosen [project](../concepts/projects.md) (the output folder when none is chosen) when the picture was uploaded or lives elsewhere. The file records the picture it was made from, shown as **Made from** in the viewer, but not the paint. **Reuse settings** on an edit returns to **Edit** with the same image, prompt and settings, and asks you to paint again.

## Get a larger image { #upscale-while-generating }

Set **Upscale** to 2× or 4× and the image is made at a smaller size, enlarged and refined, so it keeps the size you picked. It is faster than making the large image directly. Turn on **Review** to see each small render first and approve or discard it before the enlarging.

How the refinement works, the other settings and upscaling a finished image are in [Upscaling](../concepts/upscaling.md). The presets in [the size table](#pick-a-size) keep their exact size under a 2× upscale, and `m`, `l` and `xl` do under 4×.

## Sharpen and adjust contrast and saturation

**Post-processing** has three optional passes that run after the image is made:

| Pass | Default | Range |
|---|---|---|
| **Sharpen** | On | 0–1.5. At **auto**, the [configured](../reference/configuration.md#sharpening) amount is used |
| **Contrast** | Off | 1.0 = no change |
| **Saturation** | Off | 1.0 = no change |

Sharpening is CAS (contrast-adaptive sharpening). The configured amounts are `sharpening.normal` (1.0) for plain images and `sharpening.upscaled` (1.2) for upscaled ones. Amounts above 1.5 are refused, because the filter breaks down beyond that. **On a phone,** the passes are in the **Post-processing** section of **All**. Post-processing is for images only, and an [edit](#edit) applies none of it.

## Make the background transparent { #transparent-background }

Switch on **Transparent** in **Post-processing** to cut the subject out of the finished image and save a PNG with a transparent background. It works with every image model. Use it for stickers, product shots, game sprites and anything you want to place on another picture.

!!! warning "Non-commercial use only"
    The cut-out models carry permissive licences, but they were trained on research-only datasets, and commercial use of the cut-outs is not cleared. Treat the result as for non-commercial use. The **Non-commercial** button beside the switch opens the cut-out models' terms.

The cut-out models download once, before the first image, and can be deleted from the [Models page](../web-ui/models.md).

It cuts out the main subject, so a scene without one (a landscape, a pattern) has nothing clear to keep. Smoke, fog, fine hair, glass and sheer fabric can come out rough or mostly opaque, and a cast shadow is removed with the background.

??? note "How the cut-out works"

    - **The prompt gets a background phrase.** The model receives your prompt with `, isolated on a plain light grey studio background` appended, so the subject sits on an even backdrop. The prompt saved in the file, shown in **Details** and loaded by **Reuse settings** is the one you wrote. A JSON caption gets no phrase.
    - **The cut-out runs last,** after upscaling, sharpen, contrast and saturation, at the final size. It finds the subject, refines its edge and removes the backdrop colour from the edge pixels, so the result shows no grey halo on another picture.

??? note "Transparent images with start images, references and upscales"

    - **Start images and references** are used as usual; see [Start from an image](#start-from-an-image).
    - **Upscaling a transparent image** later, from the viewer, gives a transparent result.
    - **`--upscale-save-pre`** saves the small image as an ordinary opaque image, before the cut-out.
    - The cut-out models load beside the image model for the rest of the job. See [Memory](../concepts/memory.md#background-removal).

## Use Ideogram 4's controls

Ideogram 4 Instant adds two things to the Workspace; see [Ideogram 4 Instant](../reference/models.md#ideogram-4-instant) for the model itself.

- **Prompt (JSON)** is a mode of the prompt box that takes a structured caption. The [enhancer](../concepts/prompt-enhancer.md#json-captions-ideogram-4) can write one for you.
- **First sigma** (`--first-sigma`) is best-effort help when a harmless prompt keeps being blocked by the safety filter. Try `1.005` or `1.006`; the default is `1.004`.

It has no start image or upscale, so the **Images** section is hidden for it.

## From the command line

`ziv-image` takes the settings above as flags and saves in the current folder, or in `-o`:

```bash
ziv-image -m klein4b --ratio 16:9 --size l --prompt "A mountain lake at dawn"
ziv-image -m klein9b --reference woman.jpg,backpack.jpg --prompt "the woman from image 1 wearing the backpack from image 2, on a rainy street"
ziv-image -m klein9b --edit photo.png --extend right=320 --prompt "more of the garden"
```

More recipes and every flag are in [Make images](../cli/images.md).
