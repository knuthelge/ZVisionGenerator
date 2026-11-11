# Generate images

`ziv-image` makes images. Every option is in the [command reference](../cli/reference.md#ziv-image); this page shows the common tasks.

## Pick a model

```bash
ziv-image -m zit --prompt "a beautiful sunset"            # Z-Image Turbo
ziv-image -m klein4b --prompt "a portrait"                 # FLUX.2 Klein 4B
ziv-image -m klein9b --prompt "a portrait"                 # FLUX.2 Klein 9B
ziv-image -m krea2 -q 4 --prompt "a portrait"              # Krea 2 Turbo, 4-bit
ziv-image -m ideo --prompt "a travel poster for Lisbon"    # Ideogram 4 Instant
ziv-image -m moss -q 4 --prompt 'a cafe sign that reads "Open"'    # MOSS-Image Turbo
```

`-m` also takes a model you converted, a folder path or a Hugging Face repo. See [Models](../concepts/models.md) for what each model is good at and how names are resolved.

## Size and shape

```bash
ziv-image -m klein9b --ratio 16:9 --size l --prompt "a mountain lake"
ziv-image -m klein9b -W 1024 -H 1024 --prompt "a mountain lake"
```

`--ratio` is `1:1`, `16:9`, `9:16`, `3:2` or `2:3` (the default), and `--size` is `xs` to `xl` (default `m`). `-W` and `-H` set an exact size. The sizes are listed in [Images](../concepts/images.md#sizes).

## Steps, guidance and seed

```bash
ziv-image -m klein9b --steps 6 --guidance 1.5 --seed 42 --prompt "a mountain lake"
```

Each model has its own default steps and guidance, which suit it best. `--seed` makes a run repeatable; without it every image gets a random seed.

`moss-sft` takes a negative prompt through a [prompt file](../concepts/prompts.md#negative-prompts); `moss` (Turbo) has none.

## Use less memory

```bash
ziv-image -m klein9b -q 8 --prompt "a mountain lake"
ziv-image -m krea2 -q 4 --prompt "a mountain lake"
```

`-q 8` and `-q 4` quantize the model. The first run saves a quantized copy, so later runs start faster. See [Quantization](../concepts/quantization.md).

## Start from an image

```bash
ziv-image -m klein9b --image photo.jpg --image-strength 0.4 --prompt "A woman in a red dress"
```

`--image-strength` (0–1, default 0.5) sets how far the result may move from the start image. See [Start image](../concepts/images.md#start-image).

## Keep what is in other images

`--reference` takes a comma-separated list of up to four images whose person, object, place or style the result keeps. Refer to them in the prompt by number, in order:

```bash
ziv-image -m klein9b --reference woman.jpg,backpack.jpg --prompt "the woman from image 1 wearing the backpack from image 2, on a rainy street"
```

Only FLUX.2 Klein takes references; other models refuse the flag. It also refuses more images than the model takes and a file that is missing or isn't a PNG, JPEG or WebP. See [References](../concepts/images.md#references) for what works and what it costs.

## Edit a picture

`--edit` changes a picture you already have instead of generating a new one. It needs `--prompt` and `-m`.

```bash
ziv-image -m klein9b --edit photo.png --mask mask.png --prompt "a glass vase of red tulips"
ziv-image -m zit --edit photo.png --extend right=320 --prompt "more of the garden"
ziv-image -m klein9b --edit photo.png --prompt "make it night"
```

- **`--mask`** redraws the painted area. The mask is a PNG of the picture's shape, as the picture looks upright (a photo's orientation tag is applied first). It may be another size of that shape, such as a downscaled copy, and is scaled to the picture: in a PNG with transparency, any pixel that isn't fully transparent is painted; in any other PNG, including one with an alpha channel that is fully opaque, any pixel that isn't black is. A mask of another shape, or one with nothing painted, is refused.
- **`--extend`** grows one side: `top`, `right`, `bottom` or `left`, then `=` and 128 to 512 pixels in steps of 16. It combines with `--mask`.
- **Neither** changes the whole picture, on FLUX.2 Klein and an opaque picture. Z-Image and Krea 2, and a picture with transparent parts, need `--mask` or `--extend`.

The result is saved as `<name>_edit.png` in the picture's folder, or in `-o`; a picture already named `..._edit` is saved with a counter instead (`photo_edit_2.png`). Every pixel outside the paint is the original's. An edit takes `-q`, `--lora`, `--steps`, `--seed` and `-r`; any other option is refused before a model loads, such as `--image`, `--reference`, `--upscale`, `--enhance`, `--transparent`, the size options, `--guidance` and the finishing options. What each case does, and how it treats a picture with transparent parts, is in [Edit a picture](../concepts/images.md#edit).

## Add LoRAs

```bash
ziv-image -m klein9b --lora myStyle --prompt "a portrait"
ziv-image -m klein9b --lora style1:0.8,detail:0.5 --prompt "a portrait"
```

Names come from `~/.ziv/loras/`; full paths work too. See [LoRAs](../concepts/loras.md).

## Upscale while generating

```bash
ziv-image -m klein9b --upscale 2 --prompt "a landscape"
ziv-image -m klein9b --upscale 4 --upscale-denoise 0.3 --upscale-steps 8 --prompt "a landscape"
ziv-image -m klein9b --upscale 2 --review --prompt "a landscape"
```

The image is made at a smaller size, then enlarged and refined, so it keeps the size you asked for. `--review` pauses after the small render: press ++y++ to upscale it, ++n++ to discard it or ++q++ to quit. See [Upscaling](../concepts/upscaling.md).

## Adjust the finish

```bash
ziv-image -m klein9b --sharpen 0.6 --prompt "a sunset"        # custom sharpening (0–1.5)
ziv-image -m klein9b --no-sharpen --prompt "a sunset"         # no sharpening
ziv-image -m klein9b --contrast 1.2 --saturation 1.3 --prompt "a sunset"
```

Sharpening is on by default; contrast and saturation are off. See [Post-processing](../concepts/images.md#post-processing).

## Ideogram 4 captions

Ideogram 4 reads structured JSON captions. Pass one with `--json-prompt`, or let the enhancer write one:

```bash
ziv-image -m ideo --json-prompt '{"high_level_description": "a portrait"}'
ziv-image -m ideo --prompt "a travel poster for Lisbon with the text 'LISBOA'" --enhance-format json
```

`--first-sigma 1.005` helps when a harmless prompt still comes back as a grey "blocked" frame. See [Ideogram 4 Instant](../concepts/models.md#ideogram-4-instant).

## Where images go

Images are saved in the current folder, or in `-o`:

```bash
ziv-image -m klein9b -o ~/Pictures/ziv --prompt "a sunset"
```

Name a folder under the Web UI's output folder, such as `-o ~/.ziv/outputs/"Night Bus"`, to put the images in a [project](../concepts/projects.md#on-disk-and-in-the-cli).

The file name is the prompt set's name and the time, for example `prompt_2026-10-04_14-03-22.png`. The settings are embedded in the file. See [Output files](../concepts/data-folder.md#output-files).
