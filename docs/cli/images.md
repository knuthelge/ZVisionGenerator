# Generate images

`ziv-image` makes images. Every option is in the [command reference](../cli/reference.md#ziv-image); this page shows the common tasks.

## Pick a model

```bash
ziv-image -m zit --prompt "a beautiful sunset"            # Z-Image Turbo
ziv-image -m klein4b --prompt "a portrait"                 # FLUX.2 Klein 4B
ziv-image -m klein9b --prompt "a portrait"                 # FLUX.2 Klein 9B
ziv-image -m krea2 -q 4 --prompt "a portrait"              # Krea 2 Turbo, 4-bit
ziv-image -m ideo --prompt "a travel poster for Lisbon"    # Ideogram 4 Instant
ziv-image -m moss -q 4 --prompt 'a cafe sign that reads "Open"'    # MOSS-Image Turbo (Windows, Linux)
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

`--image-strength` (0–1, default 0.5) sets how far the result may move from the image. See [Reference images](../concepts/images.md#reference-images).

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

The file name is the prompt set's name and the time, for example `prompt_2026-10-04_14-03-22.png`. The settings are embedded in the file. See [Output files](../concepts/data-folder.md#output-files).
