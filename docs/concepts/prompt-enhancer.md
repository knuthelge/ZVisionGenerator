# Prompt enhancer

The prompt enhancer rewrites a prompt with more visual detail before an image or video is made. It works on images and video; the audio tab has its own helpers, described in [Make music](../guides/music.md#enhance-the-prompt) and [Make speech and sound](../guides/speech-and-sound.md#script-helper).

## What it does

The enhancer is a small language model that runs on your machine. Nothing is sent anywhere.

```text title="Your prompt"
a fox in snow
```

```text title="Enhanced, with Natural photo and Serene (an example)"
A red fox curled in fresh powder snow at the edge of a birch forest, soft overcast morning light, frost on its whiskers, shot on a 200mm lens at eye level, muted whites and warm orange fur.
```

## Fast and Smart { #fast-and-smart }

There are two built-in enhancer models. **Fast** is the default. **Smart** is larger and follows your options and wording more closely: it keeps counts, adds fewer things you didn't ask for, and writes better JSON captions. It needs more memory and a larger download.

| Platform | | Model |
|---|---|---|
| macOS | Fast | `McG-221/Qwen3.5-4B-heretic-mlx-4Bit` |
| macOS | Smart | `TheCluster/Qwen3.5-27B-Heretic-MLX-4bit` |
| Windows, Linux | Fast | `coder3101/Qwen3.5-4B-heretic` |
| Windows, Linux | Smart | `coder3101/Qwen3.5-9B-heretic` |

Both are uncensored variants of Qwen3.5, because small instruction-tuned models often refuse or tone down ordinary creative prompts, such as a battle scene, a horror shot or a villain's portrait.

- **When Smart doesn't fit,** the Web UI says why and Fast runs; the command line warns and uses Fast.
- **The download** happens the first time a model is needed (the Web UI asks first, with its size), or ahead of time from the [Models page](../web-ui/models.md).
- **Memory:** see [Memory](memory.md#the-prompt-enhancer).
- **Without a CUDA GPU,** Windows and Linux run only the enhancer on the CPU (image and video generation still need the GPU). It works, but each rewrite can take a few minutes and about 10 GB of memory.

??? note "Use your own enhancer model"

    Any chat model works. Set **Prompt enhancer model** on the [Config page](../web-ui/config.md) to a Hugging Face `owner/name` (optionally `@revision`) or a local folder, or pass `--enhance-model` for one run. Thinking-tuned or heavily merged models are slower and follow the length and style options less reliably.

## Options { #options }

**Nothing is picked by default.** With every choice at Keep, the enhancer still rewrites: it adds concrete visual detail where your prompt is thin. Options steer that rewrite.

Options are Style, Mood, Theme, Shot, Angle, Palette, Focus on and Length, plus Camera, Pace, Describe and Sound for video and a free-text Direction. Each is described in its menu, and the command line and prompt files take them as keys and short values: every one is in [the spec](../cli/reference.md#the-spec).

### Length

When you pick no Length, **Same** applies. Length is a target, not a guarantee: **Same** grows a very short prompt a little, **Shorter** keeps every element you named, so a prompt full of named things shrinks only a little, and results are capped. The numbers are the [`prompt_enhancer.length` settings](../reference/configuration.md#prompt_enhancer).

## What it keeps and knows

The enhancer keeps your subject and every detail you describe. It adds only things that can be seen, no smells or story, and never adds new people, animals or major objects.

- **Quoted text is copied exactly.** Text in quotes, such as a sign's `"OPEN 24 HOURS"` or a line of dialogue, stays word for word. If a rewrite changes it, the enhancer tries once more; if that also fails, the rewrite is used with a note (*A quoted text was changed in the rewrite*) so you can fix it.
- **Random choices are picked first.** A `{a|b}` in your prompt is resolved before the rewrite, so the rewrite has none left. Enhance again for a different pick. See [Prompt syntax](prompt-syntax.md#random-choices).

### Sound { #sound }

For video with audio on:

- Every sound and quoted line of speech you wrote is kept, word for word.
- With no Sound option picked, no new sounds are added.
- **Ambience** and **Sound effects** describe the setting's soundscape and sounds that match the action. **Music** describes music in the scene or a score.
- **Dialogue** keeps your quoted lines. If you wrote none, it writes short lines that fit the scene, spoken only by people already in it.

With audio off, and for images, no sounds are described. In a silent video, quoted speech may be dropped, but quoted lettering is kept.

### What it knows about the job { #what-the-enhancer-knows-about-the-job }

The enhancer knows the frame's shape, the clip length, whether there is a start or last image, and any [references](../guides/images.md#references), though it never sees the images. It keeps phrases such as "image 1" as you wrote them; if a rewrite drops one, it tries once more, then uses the rewrite and warns you.

## Video: scene and action { #video-scene-and-action }

For video, one rewrite returns two parts, a **scene** and an **action** (see [Make video](../guides/video.md#scene-and-action)).

Style, Mood, Theme, Shot, Angle, Palette and Focus on shape the scene; Camera, Pace, Describe and Sound shape the action; Direction and Length shape both. **Rewrite** chooses between **Scene and action** (`both`) and **Action only** (`action`).

The default depends on the workflow, and on whether a first frame is set. The choice is remembered separately for runs with and without a first frame:

- **Video without a first frame** rewrites both, splitting what you wrote: setting, look and camera setup go to the scene, events in order to the action.
- **Video with a first frame** rewrites the action only when you gave a scene (what the image shows), so it isn't restyled. With a blank scene it writes a short scene too.
- **Extend** rewrites the action only when the clip has a scene, so a chain of extensions keeps its look. It uses the previous clip's prompt as context, so it doesn't repeat its actions or lines.

Action only needs a scene: with a blank one, both parts are written, and the scene options have no effect with Action only. If the enhancer doesn't return both parts, its rewrite is used as the action and a warning says so.

## JSON captions (Ideogram 4) { #json-captions-ideogram-4 }

[Ideogram 4](../reference/models.md#ideogram-4-instant) reads structured JSON captions. With `ideo`, **Write as: JSON** (`--enhance-format json`) writes a full caption from your prompt: a summary, the look, lighting and colours, the background, one element per thing you named, and the lettering you asked for.

- The same options apply: Style decides photo or artwork, Mood and Theme shape the look, Focus on and Length decide how much is described.
- A palette's exact colours go into the caption's colour palette, and its colour names into the descriptions of the things in the picture. Shot and Angle are asked for in the camera line. Ideogram 4 often keeps eye level whatever the angle.
- Quoted text (`"OPEN"`, `'LISBOA'`, `「パン屋」`) and names after *called* or *named* become text elements, copied exactly. Signs you mention without their words may get short words that fit, in the language you name.
- The caption appears in the **Enhanced** tab, laid out for reading, and you can edit it there.
- If the enhancer returns no usable caption twice, the run reports it and keeps your prompt.

**Smart** writes noticeably better captions than Fast.

## Enhancing every prompt in a job or file { #enhance-every-prompt }

A rewrite can be made by hand, for one prompt, or as part of a job, for every prompt in it. A job rewrites each prompt after its random choices are picked, and works with inline prompts and prompt files. The [Workspace](../web-ui/workspace.md) describes the Enhance controls; the steps for a batch are in [Run batches from prompt files](../guides/batches.md#enhance-a-whole-batch).

- **In the Web UI,** the **Enhance each image when generating** setting rewrites each prompt as its image starts. The job card shows *Enhancing prompt N of M*, and a prompt that couldn't be enhanced is generated from the original.
- **On the command line,** `--enhance` rewrites every prompt of the run.
- **Before the model loads.** All rewrites happen first; see [Memory](memory.md#the-prompt-enhancer).
- **Repeat and automatic retries** keep the image's random choices and its enhanced prompt, and change only the seed.

### In prompt files { #in-prompt-files }

An entry with an [`enhance:` key](prompt-syntax.md#the-enhance-key) is enhanced on every run. On the command line:

- `--enhance` enhances every prompt and overrides the entries' settings;
- `--no-enhance` turns all of it off;
- `--enhance-format json` with a prompt file writes captions only for entries with their own `enhance:`, unless you add `--enhance`.

## What is recorded { #what-is-recorded }

The enhanced prompt is what's stored in the image or video, so the Gallery shows and reuses the prompt that was actually used. The file also keeps your own prompt and how it was enhanced: the options, the enhancer model and Text or JSON. **Reuse settings** brings both back.

The command line prints the rewrite while prompts are enhanced and again before each image or clip.

## Where you set it

- **Web UI:** the ✨ **Enhance** button in the prompt box and **Enhance each image when generating** (see [Workspace](../web-ui/workspace.md)); **Prompt enhancer model** on the [Config page](../web-ui/config.md).
- **Command line:** `--enhance [SPEC]`, `--enhance-note`, `--enhance-model` and `--enhance-format`, as in `ziv-image -m klein9b --prompt "a fox in snow" --enhance style=photo,mood=serene` (see [the spec](../cli/reference.md#the-spec)).
