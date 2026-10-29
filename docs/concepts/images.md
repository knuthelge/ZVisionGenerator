# Images

## Sizes

An image's size is a **ratio** and a **size preset**. The default is `2:3` portrait at `m`.

| Preset | 1:1 | 16:9 | 9:16 | 3:2 | 2:3 |
|---|---|---|---|---|---|
| `xs` | 512×512 | 672×384 | 384×672 | 576×384 | 384×576 |
| `s` | 704×704 | 960×544 | 544×960 | 864×576 | 576×864 |
| `m` | 1024×1024 | 1344×768 | 768×1344 | 1216×832 | 832×1216 |
| `l` | 1408×1408 | 1920×1088 | 1088×1920 | 1728×1152 | 1152×1728 |
| `xl` | 1600×1600 | 2176×1216 | 1216×2176 | 1920×1280 | 1280×1920 |

You can also set an exact width and height (**W × H** in the Web UI, `-W` and `-H` in the CLI). Some models limit the size: Ideogram 4 takes 256–2048 pixels in steps of 16, and MOSS-Image takes multiples of 16.

Larger images take longer and need more memory. To get a large image quickly, generate smaller and [upscale](upscaling.md): every preset keeps its exact size under a 2× upscale, and `m`, `l` and `xl` under 4×.

The **Base resolution** on the [Config page](../web-ui/config.md) changes the default preset.

## Steps, guidance and seed

- **Steps** is how many denoising steps the model takes. Each model has a default that suits it, such as 4 for FLUX.2 Klein and MOSS-Image Turbo, 8 for Krea 2 and Ideogram 4, and 50 for MOSS-Image SFT. More steps aren't always better with these distilled models.
- **Guidance** is how strictly the model follows the prompt. The defaults suit each model; Ideogram 4 and MOSS-Image Turbo ignore it, and for Krea 2 and MOSS-Image SFT anything above 1.0 makes each step take about twice as long.
- **Seed** picks the starting noise. The same model, prompt, settings and seed give the same image. Without a seed every image gets a random one, which is recorded in the file.
- **Scheduler** decides how the steps are spaced. Z-Image uses `beta` by default; most other models have no choice.

## Start image

A start image is a starting point: the model starts from it and moves toward your prompt, keeping its layout and colours. This is **Image to Image** in the Web UI and `--image` in the CLI.

**Img strength** (`--image-strength`, 0–1, default 0.5) sets how far the result may move: low values stay close to the image, high values follow the prompt more.

```bash
ziv-image -m klein9b --image photo.jpg --image-strength 0.4 --prompt "A woman in a red dress"
ziv-image -m klein9b --image sketch.png --image-strength 0.8 --prompt "Cyberpunk cityscape"
```

The image is resized to the target size. Photos are turned upright by their orientation tag, and transparent areas are filled with white. Ideogram 4 doesn't take a start image.

In the Web UI viewer, **Use as** → **Start image** (++e++) opens an image as the start image together with its prompt.

## References

A start image sets the layout. A **reference** is different: it is an image whose *content* the result reuses, such as a person, an object, a place or a style. You can give up to four, alone or combined, for example a woman from one photo, a backpack from another and a pier from a third.

References work with FLUX.2 Klein 4B and 9B (`klein4b`, `klein9b`). In the Web UI, the **References** section appears in **Text to Image** and **Image to Image** when the selected model takes them; other models don't show it. In the CLI, `--reference` takes a comma-separated list, and other models refuse it.

```bash
ziv-image -m klein9b --reference woman.jpg,backpack.jpg --prompt "the woman from image 1 wearing the backpack from image 2, on a rainy street"
```

**Refer to each reference by its number, in order:** the first is "image 1", the second "image 2". The model reads those words in your prompt, so say which image supplies what: "the woman from image 1 wearing the backpack from image 2". The [prompt enhancer](prompt-enhancer.md#what-the-enhancer-knows-about-the-job) keeps these phrases.

- **What works:** a person, an object, a place and a style, alone or combined. A person, an object and a place in one image works, and so does a person with a style.
- **A style works best next to one other reference.** With four references, a style one tends to be ignored.
- **Size:** each reference keeps its own aspect ratio and is scaled to at most about 1 megapixel. Photos are turned upright by their orientation tag, and transparent areas are filled with white. The result has the size you chose.
- **Cost:** every reference makes each step slower and uses more memory. Three references roughly double the time per step on FLUX.2 Klein 4B, and four make it about two and a half times as long at a large size. Use as few as the picture needs.
- **LoRAs** keep working.

### References with a start image

You can use both. The start image sets the layout and **Img strength** how much of it stays. On Klein's 4-step models, strength has few useful levels: where the layout stays, there is little room for new content, so a referenced person may not appear, and at a high strength the layout is lost. To put a referenced subject in a new scene, use references without a start image.

### Upscale and Reuse

An upscale made while generating refines with the same references, so what they fix stays the same in the larger image. [Upscaling a finished image](upscaling.md#upscaling-an-existing-image) later doesn't use them, but the upscaled image records them, so **Reuse settings** brings them back. In the Web UI, a reference whose file has since moved or been deleted keeps its row, so the numbers in your prompt still match; see [References](../web-ui/workspace.md#references).

## Post-processing

After an image is made, three optional passes adjust it:

| Pass | Default | Range |
|---|---|---|
| **Sharpen** | On | 0–1.5. Without an amount, the [configured](configuration.md#sharpening) amount is used |
| **Contrast** | Off | 1.0 = no change |
| **Saturation** | Off | 1.0 = no change |

Sharpening is CAS (contrast-adaptive sharpening). The configured amounts are `sharpening.normal` (1.0) for plain images and `sharpening.upscaled` (1.2) for upscaled ones. Amounts above 1.5 are refused, because the filter breaks down beyond that. In the Web UI, leave **Sharpen** at **auto** to use the configured amount.

Post-processing is for images only.

## Live preview

While an image is made, the Web UI shows it taking shape after every step, so you can skip a bad one early. The preview is never saved. It works for every built-in image model.

## Output files

See [Output files](data-folder.md#output-files) for how images are named and what they record.
