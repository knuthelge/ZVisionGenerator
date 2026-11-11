# Make images

`ziv-image` makes images from a prompt, a start image or reference images, edits a picture you already have, and can upscale and cut out the background on the way. The guide [Make images](../guides/images.md) says what each setting does and why.

```bash
ziv-image -m klein4b --prompt "A red fox sitting in fresh snow, early morning light"
```

You get one portrait image (`2:3`, size `m`) in the current folder, named `prompt_<date>_<time>.png`. `-o folder` saves it elsewhere; see [Where files go](index.md#where-files-go).

## Choose what to make

### Make a vertical or wide image

`--ratio` sets the shape and `--size` the resolution (`xs`, `s`, `m`, `l`, `xl`).

```bash
ziv-image -m klein4b --ratio 9:16 --size m --prompt "A neon-lit alley in the rain, a cat on a doorstep"
ziv-image -m klein4b --ratio 16:9 --size l --prompt "A mountain lake at dawn"
```

The first gives 768×1344 and the second 1920×1088. For an exact size, give `-W` and `-H`, each a multiple of 16: `-W 1280 -H 720`. The sizes of every preset are in [Pick a size](../guides/images.md#pick-a-size).

### Start from an image

```bash
ziv-image -m klein9b --image sketch.png --image-strength 0.8 --prompt "A watercolour of the same harbour at dusk"
```

You get a new image that starts from the layout and colours of `sketch.png`. A low `--image-strength` stays close to it and a high one follows the prompt more; the default is `0.5`. Ideogram 4 takes no start image.

### Use references

```bash
ziv-image -m klein9b --reference woman.jpg,backpack.jpg --prompt "the woman from image 1 wearing the backpack from image 2, on a rainy street"
```

You get a new scene that keeps what the references show: a person, an object, a place or a style. Name them in the prompt as "image 1", "image 2", in the order you list them. FLUX.2 Klein (`klein4b`, `klein9b`) takes up to four, and `--reference` can be combined with `--image`. See [Keep a person, object or style](../guides/images.md#references).

### Use a LoRA

```bash
ziv-image -m klein9b --lora my-style:0.7 --prompt "A portrait of an old fisherman"
ziv-image -m klein9b --lora my-style:0.7,detail:0.4 --prompt "A portrait of an old fisherman"
```

`name:weight` takes a LoRA from `loras/` in the data folder, and the weight is `1.0` when left out. A full path works too. To get a LoRA from Hugging Face first, see [Models and LoRAs](models.md#import-a-lora). More in [LoRAs](../concepts/loras.md).

### Fit a big model in less memory

```bash
ziv-image -m klein9b -q 8 --prompt "A greenhouse full of orchids, soft morning light"
```

`-q 8` or `-q 4` runs the model quantized: it needs less memory and is a little less exact. The first run at a level may save a quantized copy, so later loads are faster. See [Memory and quantization](../concepts/memory.md#quantization).

### Cut out the background

```bash
ziv-image -m klein4b --transparent --prompt "A red fox sticker, flat colours"
```

You get a PNG with a transparent background. The cut-out models download once, with their terms, before the first image. They are for non-commercial use only; see [Make the background transparent](../guides/images.md#transparent-background).

### Make an Ideogram 4 image from a JSON caption

```bash
ziv-image -m ideo --json-prompt '{"high_level_description": "A red fox sitting in fresh snow, early morning light"}'
ziv-image -m ideo --prompt "A poster for a night market, bold lettering" --enhance-format json --first-sigma 1.005
```

`--json-prompt` sends a structured caption as it is: it must be a JSON object, it skips `{a|b}` choices, and it replaces `--prompt`. On other models it is taken as a plain prompt, with a warning. For a fuller caption, `--enhance-format json` has the enhancer write one from your prompt. `--first-sigma` is for Ideogram 4 only: when a harmless prompt keeps coming out as a grey "blocked by safety filter" frame, raise it from the default `1.004` to `1.005` or `1.006`.

## Edit a picture

`--edit` changes a picture you already have instead of generating a new one. It needs `--prompt` and a FLUX.2 Klein model (`klein4b`, `klein9b`). What each case does, and its limits, are in [Edit a picture](../guides/images.md#edit).

The result is saved as `<name>_edit.png` in the picture's folder, or in `-o`, with a counter when that name exists. The original is never changed. An edit takes `-q`, `--lora`, `--steps`, `--seed` and `-r`.

### Paint a mask

```bash
ziv-image -m klein9b --edit photo.png --mask mask.png --prompt "a glass vase of red tulips"
```

You get the picture with the masked area redrawn from the prompt; the rest stays as it was. `--mask` is a PNG with the picture's shape (any size of it). Where the PNG has transparency, any pixel that isn't fully transparent is painted; otherwise any pixel that isn't black. A mask with nothing painted, or the wrong shape, is refused.

### Extend a side

```bash
ziv-image -m klein9b --edit photo.png --extend right=320 --prompt "more of the garden"
```

You get a wider picture, with the new area drawn from the prompt. `--extend` takes `top`, `right`, `bottom` or `left`, then `=` and 128 to 512 pixels in steps of 16. It grows one side, and can't be combined with `--mask`.

### Change the whole picture

```bash
ziv-image -m klein9b --edit photo.png --prompt "make it night"
```

With neither `--mask` nor `--extend`, the prompt changes the whole picture at its own size. A picture with transparent parts needs one of them, and can't be extended.

Options that generate a new picture are refused with `--edit`: `--image`, `--reference`, `--upscale`, `--review`, `--enhance`, `--transparent`, `--json-prompt`, `--width`, `--height`, `--ratio`, `--size`, `--guidance`, `--scheduler`, `--contrast`, `--saturation` and a `--sharpen` amount. An edit keeps the picture's own size and look.

## Finish the picture

### Upscale while generating

```bash
ziv-image -m klein4b --size s --upscale 2 --review --prompt "A lighthouse in a storm, long exposure"
```

The image is made at half the size, enlarged and refined, so you get the size you chose in less time. `--review` pauses after each small render: press ++y++ to upscale it, ++n++ to discard it or ++q++ to quit.

To tune the refining pass, and to keep the small image too:

```bash
ziv-image -m klein4b --size s --upscale 2 --upscale-denoise 0.3 --upscale-steps 6 --upscale-guidance 1.5 --upscale-save-pre --prompt "A lighthouse in a storm, long exposure"
```

`--upscale-denoise` is how much the pass may change the image (0 to 1), `--upscale-steps` and `--upscale-guidance` are its own steps and guidance, and `--upscale-save-pre` also saves the small image with a `_preupscale` suffix. How the pass works is in [Upscaling](../concepts/upscaling.md#while-generating-an-image). To enlarge an image you already have, use **Upscale** in the Web UI's viewer.

### Sharpen, contrast and saturation

```bash
ziv-image -m klein4b --sharpen 0.6 --contrast 1.2 --saturation 1.3 --prompt "A market stall of oranges, midday sun"
ziv-image -m klein4b --no-sharpen --prompt "A market stall of oranges, midday sun"
```

Sharpening is on by default; contrast and saturation are off until you give an amount (`1.0` is no change). See [Sharpen and adjust contrast and saturation](../guides/images.md#sharpen-and-adjust-contrast-and-saturation).

## Get better results { #get-better-results }

The habits in [Get better results](../guides/better-results.md) are flags too.

### Let the enhancer add detail

```bash
ziv-image -m klein4b --prompt "a fox in snow" --enhance style=photo,mood=serene
ziv-image -m klein4b --prompt "a fox in snow" --enhance-note "in a storm, at night"
```

`--enhance` rewrites the prompt into a fuller one before the model loads; `style=photo,mood=serene` steers it, and without a value you get a general rewrite. `--enhance-note` is your own direction in words. The values `--enhance` takes are in [the enhance spec](reference.md#the-spec).

### Make several different images

```bash
ziv-image -m klein4b -r 4 --prompt "A red fox {leaping|sleeping} in a {birch|pine} forest"
```

`-r 4` runs the prompt four times with one model load. Each run gets its own random seed and picks the `{a|b}` choices again, so you get four different images.

### Repeat one image

```bash
ziv-image -m klein4b --seed 42 --prompt "A red fox curled up in fresh snow, dim blue light at dusk"
```

The same model, prompt, settings and seed give the same image. A fixed `--seed` is used for every run, so `-r 4 --seed 42` makes the same image four times when the prompt has no `{a|b}` choices. Leave `--seed` out to explore, and use it to redo one image with a small change. Every image records its seed in **Details**.

### Run a prompt file

```bash
ziv-image -m klein9b -p prompts.yaml -r 3
```

See [Batches and prompt files](batches.md).

## Flags

Every flag, with its default and limits, is in the [command reference](reference.md#ziv-image).

| Flag | Does | In the Web UI |
|---|---|---|
| `-m` | The model, by name or alias | **Model** |
| `-q 4`, `-q 8` | Run the model quantized | **Quant** |
| `--prompt` | One prompt, typed in the command | **Prompt** (**Inline**) |
| `--json-prompt` | One structured JSON caption, sent as it is | **JSON caption** |
| `-p` | A prompt file | **Prompt source** → **Prompt file** |
| `-r` | How many times to run each prompt | **Batch size** |
| `--ratio`, `--size` | Shape and resolution | **Ratio**, **Resolution** |
| `-W`, `-H` | Exact width and height | **W × H** |
| `--steps`, `--guidance` | Steps and guidance | **Steps**, **Guidance** |
| `--scheduler` | `beta` or `simple` | **Scheduler** |
| `--seed` | The seed, used for every run | **Seed** |
| `--first-sigma` | Ideogram 4's first-step sigma | **First sigma** |
| `--lora` | LoRAs and their weights | **LoRAs** |
| `--image`, `--image-strength` | Start image and how far to move from it | **Start image**, **Strength** |
| `--reference` | Reference images | **References** |
| `--edit` | The picture to edit | **Image to edit** |
| `--mask` | The area of the `--edit` picture to redraw, as a PNG | **Paint** (the painter) |
| `--extend` | Grow one side of the `--edit` picture, such as `top=256` | **Extend**, **Extend by** (the painter) |
| `--upscale` | Upscale factor, 2 or 4 | **Upscale** |
| `--upscale-denoise`, `--upscale-steps`, `--upscale-guidance` | Settings of the refining pass | **Denoise** (the others have no control) |
| `--upscale-sharpen` | Sharpen before refining | |
| `--upscale-save-pre` | Also save the small image | |
| `--review` | Approve each small render before upscaling | **Review** |
| `--sharpen`, `--contrast`, `--saturation` | Finishing passes, with an optional amount | **Sharpen**, **Contrast**, **Saturation** |
| `--transparent` | Cut out the background | **Transparent** |
| `-o` | The output folder; with `--edit`, the picture's own folder when left out | **Output folder** (Config page) |
| `--enhance`, `--enhance-note`, `--no-enhance` | Rewrite the prompt first | ✨ **Enhance**, **Direction** |
| `--enhance-model`, `--enhance-format` | The enhancer model, and text or JSON | **Prompt enhancer model** (Config page), **Write as** |
