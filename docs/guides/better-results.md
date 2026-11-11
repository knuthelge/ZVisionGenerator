# Get better results

Seven habits that improve what you make, from the prompt you type to the setup you keep. The examples are images.

To keep experiments separate from finished work, pick a [project](../concepts/projects.md) before you generate: each project keeps its own files apart from the rest.

## Write a clearer prompt

Describe what the picture shows, in plain words: the subject, where it is, the light, how it is framed and the style. Say what you want to see, not the idea behind it.

| Instead of | Write |
|---|---|
| A sad fox | A red fox curled up in fresh snow, head down, dim blue light at dusk |
| A nice portrait | A portrait of an old fisherman, close-up, soft window light, shallow depth of field |
| Something cosy | A small kitchen with a wooden table and a steaming teapot, warm lamp light, wide shot |

1. Start with the subject and what it is doing.
2. Add where it is and the light: *early morning light*, *under neon signs*, *overcast*.
3. Add the framing: *close-up*, *wide shot*, *from above*.
4. Add a style only if you want one: *watercolour*, *35 mm film*.

Change one thing at a time between runs, so you can see what each word does. To try several wordings in one go, write `{a|b}` choices in the prompt; each run picks one. See [Prompt syntax](../concepts/prompt-syntax.md).

## Let the enhancer add detail

A short idea is a fine start. The [prompt enhancer](../concepts/prompt-enhancer.md) is a model on your machine that rewrites it with more detail, keeping your subject and everything you described.

1. Type your prompt, for example `a fox in snow`.
2. Click ✨ **Enhance** in the prompt box.
3. Optionally pick a **Style**, **Mood**, **Shot** or **Palette**, or write a **Direction** in your own words.
4. Click **Enhance prompt**. The rewrite appears in the **Enhanced** tab, and **Generate** sends it.

Read the rewrite and edit it: it is plain text. Enhance again for a different one; each press starts from your own text, never from the rewrite. **✕** after the **Enhanced** tab scratches the rewrite, so **Generate** sends your text again.

With a **Batch size** above 1, turn on **Enhance each image when generating**: every image gets its own rewrite as the job runs.

## Generate small, then upscale

A large image takes longer and needs more memory, and a small one is quick to judge. Find the picture first at a small size such as `s` or `m`, then make it bigger with **Upscale** in the viewer, or set **Upscale** before you generate.

How both work, and what the settings do, is in [Get a larger image](images.md#upscale-while-generating).

## Make variations with Reuse and a new seed

Every job uses a new random seed, so pressing **Generate** again gives a different image. To vary one you like:

1. Open the image in the Gallery or viewer and choose **Reuse settings** (++r++). Everything, the seed included, goes back into the Workspace.
2. Change a word or two of the prompt, or leave it as it is.
3. Press ++ctrl+shift+enter++ (++cmd+shift+enter++ on a Mac) to generate with a new seed. To get the same image again, keep the seed: a plain **Generate** with the same seed, prompt and settings gives the same image again.

For many variations at once, raise **Batch size**: each image gets its own random seed.

## Save a setup as a preset

When you find settings you like, save them as a **preset** and bring them back in one step.

1. Set up the model, LoRAs, size, steps, upscale and the rest.
2. Open the **Preset** control in the toolbar and choose **Save as…**. Give it a name.
3. Later, pick the preset in the same menu. **Update** writes your current settings into it.

A preset keeps the model, LoRAs, size, steps, upscale and post-processing settings, and for video the length and audio. It leaves the prompt, seed and images alone, so one preset serves many prompts. A model or LoRA that isn't installed is skipped when you apply one. Presets are files in the data folder, shared by every browser that uses the Web UI; the command line doesn't use them. See [Presets](../web-ui/workspace.md#presets).

**On a phone:** the preset is a chip in the strip under the preview. See [Phones, tablets and other computers](../web-ui/devices.md#on-a-phone).

## Use a LoRA for a style or subject

A [LoRA](../concepts/loras.md) is a small add-on file that teaches a model a style, a character or a subject. It must be made for the model family you use.

1. Get one: click **Discover LoRAs** in the Workspace, or import a file on the [Models page](models.md#import-a-lora).
2. Add it under **LoRAs** in the toolbar and set its weight: 1.0 is full strength, lower is subtler.
3. If the LoRA has a trigger word, pick it from the **Triggers** menu (the tag icon) in the prompt box.

If the effect is too strong, lower the weight. Several LoRAs stack, so lower each weight when you combine them.

## Keep the same character with references

To put the same person, object or place into new pictures, give FLUX.2 Klein up to four **references** and name them in the prompt by number, such as *the woman from image 1 wearing the backpack from image 2*.

How to add them, what works and what they cost is in [Keep a person, object or style](images.md#references).

## From the command line

Every setting above is also a flag, except presets:

```bash
ziv-image -m klein4b --prompt "a fox in snow" --enhance style=photo,mood=serene
ziv-image -m klein4b --size s --upscale 2 --review --prompt "A lighthouse in a storm, long exposure"
```

More recipes are in [Make images](../cli/images.md#get-better-results).
