# Prompt enhancer

The prompt enhancer is a small language model that runs on your machine. It rewrites a prompt with more visual detail before the image or video is made. Nothing is sent anywhere.

```text title="Your prompt"
a fox in snow
```

```text title="Enhanced, with Natural photo and Serene (an example)"
A red fox curled in fresh powder snow at the edge of a birch forest, soft overcast morning light, frost on its whiskers, shot on a 200mm lens at eye level, muted whites and warm orange fur.
```

The enhancer keeps your subject and every detail you describe. It adds only things that can be seen, no smells or story, and never adds new people, animals or major objects.

## Fast and Smart

There are two built-in enhancer models. **Fast** is the default. **Smart** is larger and follows your options and wording more closely: it keeps counts, adds fewer things you didn't ask for, and writes better JSON captions. It needs more memory and a larger download.

| Platform | | Model | First download | Needs |
|---|---|---|---|---|
| macOS | Fast | `McG-221/Qwen3.5-4B-heretic-mlx-4Bit` | 2.4 GB | |
| macOS | Smart | `TheCluster/Qwen3.5-27B-Heretic-MLX-4bit` | 16 GB | About 20 GB of GPU memory (a Mac with 32 GB) |
| Windows, Linux | Fast | `coder3101/Qwen3.5-4B-heretic` | 9 GB | |
| Windows, Linux | Smart | `coder3101/Qwen3.5-9B-heretic` | 19 GB | About 9 GB of GPU memory |

Both are uncensored variants of Qwen3.5, because small instruction-tuned models often refuse or tone down ordinary creative prompts, such as a battle scene, a horror shot or a villain's portrait.

Where Smart doesn't fit, the Web UI says why and Fast runs; the CLI warns and uses Fast. A model downloads the first time it's needed, or ahead of time from the [Models page](../web-ui/models.md).

Without a CUDA GPU, Windows and Linux run the enhancer on the CPU: it works, but each rewrite can take a few minutes and about 10 GB of memory.

!!! tip "Your own enhancer model"
    Any chat model works. Set **Prompt enhancer model** on the [Config page](../web-ui/config.md) to a Hugging Face `owner/name` (optionally `@revision`) or a local folder, or pass `--enhance-model` for one run. Thinking-tuned or heavily merged models are slower and follow the length and style options less reliably.

## Options

**Nothing is picked by default.** With every choice at Keep, the enhancer still rewrites: it adds concrete visual detail where your prompt is thin. Options steer that rewrite.

| Option | Choose | Values | Key |
|---|---|---|---|
| Style | one | Keep; **Photo:** Natural photo, Candid, Street, Analog film, Black & white, Studio portrait, Product, Cinematic; **Art:** Illustration, Anime, Comic, 3D render, Oil painting, Watercolor, Gouache, Sketch, Pixel art, Concept art, Graphic design, Isometric, Claymation, Ukiyo-e, Papercraft | `style` |
| Mood | one | Keep, Serene, Cozy, Joyful, Romantic, Whimsical, Nostalgic, Melancholic, Mysterious, Eerie, Dramatic, Epic | `mood` |
| Theme | one | Keep, Zen, Spiritual, Rustic, Sensual, Glamorous, Gothic, Noir, Horror, Dystopian, Fairytale, Mythic, Surreal, Retro, Futuristic, Cyberpunk, Steampunk | `theme` |
| Focus on | any | Lighting, Composition, Camera & viewpoint, Materials & textures, Color palette, Environment, Subject, Expression | `details` |
| Camera (video) | one | Keep, Auto, Static, Push in, Pull back, Pan, Tilt, Tracking, Orbit, Handheld, Aerial | `camera` |
| Pace (video) | one | Keep, Slow motion, Slow, Brisk, Fast | `pace` |
| Describe (video) | any | Action sequence, Secondary motion | `motion` |
| Sound (video with audio) | any | Ambience, Sound effects, Dialogue, Music | `sound` |
| Length | one | Shorter (≈50%), Same, Longer (≈200%), Extra long (≈300%) | `length` |
| Direction | free text | For example *focus on the dog, make her look surprised* (up to 300 characters) | `note` |

The CLI and prompt files use the key with a short value, such as `film` for Analog film or `ambient` for Ambience. Every value is listed in [the spec](../cli/enhance.md#the-spec).

- **Style** sets the medium. The **Photo** styles ask for a real photograph: a camera or lens, where the light comes from, and natural skin rather than airbrushed skin. They avoid words like "hyperrealistic" or "8k", which push models toward a glossy, rendered look.
- **Mood** sets the feeling, shown through light, colour, setting and expression rather than named.
- **Theme** sets the character the image carries, such as zen, noir or cyberpunk, through styling, materials, colour and small details. It never adds people, animals or objects. Time of day and weather aren't themes: write them in your prompt.
- **Focus on** decides where the rewrite spends its words. With Length Shorter, the picked aspects are kept while the rest is cut.
- **Camera** and **Pace** shape a video's action. **Auto** lets the enhancer pick a camera move. A faster pace fits more action into the clip. **Secondary motion** describes what moves along with the action: hair, cloth, water, smoke, leaves or dust.
- **Direction** is a short instruction the enhancer follows.

### Length

Length is a target, not a guarantee:

- **Same** keeps a prompt's length, but grows a short one to about 25 words.
- **Longer** aims for at least 40 words, **Extra long** for at least 80.
- **Shorter** never aims below 12 words, and keeps every element you named, so a prompt full of named things shrinks only a little.

Results are capped at 300 words.

### Quoted text

Text in quotes, such as a sign's `"OPEN 24 HOURS"` or a line of dialogue, is copied exactly. If a rewrite changes it, the enhancer tries once more; if that also fails, the rewrite is used with a note so you can fix it.

### Sound

For video with audio on:

- Every sound and quoted line of speech you wrote is kept, word for word.
- With no Sound option picked, no new sounds are added.
- **Ambience** and **Sound effects** describe the setting's soundscape and sounds that match the action. **Music** describes music in the scene or a score.
- **Dialogue** keeps your quoted lines. If you wrote none, it writes short lines that fit the scene, spoken only by people already in it.

With audio off, and for images, no sounds are described. In a silent video, quoted speech may be dropped, but quoted lettering is kept.

### What the enhancer knows about the job

- **The frame:** portrait, landscape or square, so the composition fits.
- **The clip length:** one continuous action up to 5 seconds, two or three beats up to 10 seconds.
- **A starting image:** for image to video, the rewrite describes motion and change rather than what the first frame already shows. For image to image, it mentions the subject only briefly. The enhancer doesn't see the image itself. With a last frame, it describes the change that leads from the first image to the last and doesn't invent a different ending.

## Video: scene and action

For video, one rewrite returns two parts, a **scene** and an **action** (see [Scene and action](video.md#scene-and-action)).

| Option | Shapes |
|---|---|
| Style, Mood, Theme, Focus on | Mostly the scene |
| Camera, Pace, Describe, Sound | The action |
| Direction, Length | Both |

**Rewrite** chooses what is written:

| Rewrite | `parts` | You get |
|---|---|---|
| Scene and action | `both` | A rewritten scene and action. With no scene, your text is split into both. |
| Action only | `action` | Your scene kept exactly as written, and an action written to fit it. |

The default depends on the workflow:

- **Text to Video** rewrites both, splitting what you wrote: setting, look and camera setup go to the scene, events in order to the action.
- **Image to Video** rewrites the action only when you gave a scene (what the image shows), so it isn't restyled. With a blank scene it writes a short scene too.
- **Extend Video** rewrites the action only when the clip has a scene, so a chain of extensions keeps its look. It uses the previous clip's prompt as context, so it doesn't repeat its actions or lines.

Action only needs a scene; with a blank one, both parts are written. Style, Mood, Theme and Focus on have no effect with Action only, since they shape the scene.

## JSON captions (Ideogram 4)

[Ideogram 4](models.md#ideogram-4-instant) reads structured JSON captions. With `ideo`, **Write as: JSON** (`--enhance-format json`) writes a full caption from your prompt: a summary, the look, lighting and colours, the background, one element per thing you named, and the lettering you asked for.

- The same options apply: Style decides photo or artwork, Mood and Theme shape the look, Focus on and Length decide how much is described.
- Quoted text (`"OPEN"`, `'LISBOA'`, `「パン屋」`) and names after *called* or *named* become text elements, copied exactly. Signs you mention without their words may get short words that fit, in the language you name.
- The caption appears in the **Enhanced** tab, laid out for reading, and you can edit it there.
- If the enhancer returns no usable caption twice, the run reports it and keeps your prompt.

**Smart** writes noticeably better captions than Fast.

## In the Web UI

- **Enhance now:** click ✨ **Enhance** in the prompt box, pick options, and click **Enhance prompt**. The rewrite appears in the **Enhanced** tab. Enhancing again always starts from your text; **Clear** empties the Enhanced text.
- **The popover** groups the options as **Look** (Style, Mood, Theme), **Focus on**, **Motion** and **Sound** for video, and **Output** (Rewrite or Write as, and Length), then Direction. **Reset** puts every option back. **Enhancer** (Fast or Smart) sits next to **Enhance prompt**. Images and video remember their own options.
- **What Generate sends:** for each part, the Enhanced text when it has any, otherwise your text. A part that uses its Enhanced text shows **Uses enhanced**, with **Use mine** to go back to yours.
- **Out of date:** if you change your text or switch workflows afterwards, the Enhanced text is marked **Out of date**. It is still used until you choose **Use mine**, **Re-enhance** or **Clear**.
- **Random choices** are picked before the rewrite, so the Enhanced prompt has none left. Enhance again for a different pick.
- **Enhance each image when generating** rewrites every image's prompt as part of the job, after its random choices are picked. It works with inline prompts and prompt files. The job card shows *Enhancing prompt N of M*; a prompt that couldn't be enhanced is generated from the original.

The Enhance button is unavailable while jobs are running or queued, because it needs the same memory.

## In prompt files

Add `enhance:` to an entry to enhance it on every run:

```yaml
woman:
  - prompt: 30yo woman in red dress walking down a city street, evening
    enhance:
      style: cinematic
      mood: mysterious
      theme: noir
      details: [lighting, camera]
      length: longer
      note: rain on the pavement
  - prompt: a quiet street at dawn
    enhance: true                  # a general rewrite
  - prompt: two friends talking at a cafe table
    enhance:
      sound: [ambient, dialogue]   # video with audio on
  - scene: A fisherman on a wooden pier at dawn, mist over the water.
    prompt: he casts his line
    enhance:
      parts: action                # video: keep the scene, rewrite the action
```

In an image run, an entry whose `enhance:` sets only video options (`camera`, `pace`, `motion`, `sound` or `parts`) is not enhanced, with a warning. Video options are ignored for images, so one file can serve both.

## While prompts are enhanced

Jobs rewrite every prompt before the model loads. During that time, image jobs respond to:

- **Skip** (++n++): skip the current rewrite; that image uses your prompt.
- **Quit** (++q++): stop before the model loads.
- **Pause** (++p++): wait until you resume.

**Repeat** and automatic retries keep the image's random choices and enhanced prompt, and only change the seed.

## Memory

The enhancer and the image or video model are never loaded together: the enhancer is unloaded, and its memory freed, before the model loads. Peak memory is the larger of the two, not their sum.

Clicking Enhance in the Web UI loads the enhancer (about 3 GB on macOS) and frees it after two idle minutes, or when a job starts.

## What is recorded

The enhanced prompt is what's stored in the image or video, so the Gallery shows and reuses the prompt that was actually used. The file also keeps your own prompt and how it was enhanced: the options, the enhancer model and Text or JSON.
