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

You can also set an exact width and height (**W × H** in the Web UI, `-W` and `-H` in the CLI). Some models limit the size: Ideogram 4 takes 256–2048 pixels in steps of 16.

Larger images take longer and need more memory. To get a large image quickly, generate smaller and [upscale](upscaling.md): every preset keeps its exact size under a 2× upscale, and `m`, `l` and `xl` under 4×.

The **Base resolution** on the [Config page](../web-ui/config.md) changes the default preset.

## Steps, guidance and seed

- **Steps** is how many denoising steps the model takes. Each model has a default that suits it, such as 4 for FLUX.2 Klein and 8 for Krea 2 and Ideogram 4. More steps aren't always better with these distilled models.
- **Guidance** is how strictly the model follows the prompt. The defaults suit each model; Ideogram 4 ignores it, and for Krea 2 anything above 1.0 makes each step take about twice as long.
- **Seed** picks the starting noise. The same model, prompt, settings and seed give the same image. Without a seed every image gets a random one, which is recorded in the file.
- **Scheduler** decides how the steps are spaced. Z-Image uses `beta` by default; most other models have no choice.

## Reference images

A reference image is a starting point: the model starts from it and moves toward your prompt. This is **Image to Image** in the Web UI and `--image` in the CLI.

**Img strength** (`--image-strength`, 0–1, default 0.5) sets how far the result may move: low values stay close to the image, high values follow the prompt more.

```bash
ziv-image -m klein9b --image photo.jpg --image-strength 0.4 --prompt "A woman in a red dress"
ziv-image -m klein9b --image sketch.png --image-strength 0.8 --prompt "Cyberpunk cityscape"
```

The image is resized to the target size. Ideogram 4 doesn't take a reference image.

In the Web UI viewer, **Use as reference** (++e++) opens an image as the reference together with its prompt.

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

While an image is made, the Web UI shows it taking shape at 25, 50 and 75% of the steps, so you can skip a bad one early. The preview is never saved. It works for every built-in image model.

## Output files

See [Output files](data-folder.md#output-files) for how images are named and what they record.
