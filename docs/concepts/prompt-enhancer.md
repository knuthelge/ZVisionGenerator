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

Where Smart doesn't fit, the Web UI says why and Fast runs; the CLI warns and uses Fast. A model downloads the first time it's needed (the Web UI asks first, with its size) or ahead of time from the [Models page](../web-ui/models.md).

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
| Shot | one | Keep, Extreme close-up, Close-up, Medium, Full body, Wide, Extreme wide | `shot` |
| Angle | one | Keep, Eye level, High angle, Low angle, Overhead, Worm's-eye, Dutch angle, Over the shoulder, From behind | `angle` |
| Palette | one | Keep, Teal & orange, Pastel, Earth tones, Film, Neutrals, Forest, Cool blues, Warm gold, Neon, Jewel tones, Primary, Sunset, Retro, Custom (up to five colours you pick) | `palette`, `colors` |
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
- **Shot** and **Angle** set the framing: how much of the subject the frame holds, and where the camera sits. The rewrite opens with the framing, for example "A low-angle view of …", and describes what that view sees, such as sky behind a low-angle subject. Angles land more often on some models than others: Low angle and Worm's-eye vary most, and the other angles are dependable.
- **Palette** colours what your prompt already has, such as "a terracotta sofa on an olive green rug", and adds no new objects to carry the colours. It decides the colours over Mood and Theme: Pastel with Eerie stays pastel, and the mood still shows through light, setting and expressions. It differs from **Focus on → Color palette**, which only asks for more description of whatever colours the image has.
- **Custom** takes up to five colours you pick. The image models follow plain colour names better than codes, so each colour is matched to the nearest plain name, such as petrol blue or coral, and the rewrite uses the names.
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
- **A start image:** for image to video, the rewrite describes motion and change rather than what the first frame already shows. For image to image, it mentions the subject only briefly. The enhancer doesn't see the image itself. With a last frame, it describes the change that leads from the first image to the last and doesn't invent a different ending.
- **References:** the rewrite keeps "image 1", "image 2" as you wrote them and describes the new scene and composition rather than what the [references](images.md#references) show. If a rewrite drops one of those phrases, the enhancer tries once more, then uses the rewrite and warns you.

## Video: scene and action

For video, one rewrite returns two parts, a **scene** and an **action** (see [Scene and action](video.md#scene-and-action)).

| Option | Shapes |
|---|---|
| Style, Mood, Theme, Shot, Angle, Palette, Focus on | Mostly the scene |
| Camera, Pace, Describe, Sound | The action |
| Direction, Length | Both |

**Rewrite** chooses what is written:

| Rewrite | `parts` | You get |
|---|---|---|
| Scene and action | `both` | A rewritten scene and action. With no scene, your text is split into both. |
| Action only | `action` | Your scene kept exactly as written, and an action written to fit it. |

The default depends on the workflow, and on whether a first frame is set. The choice is remembered separately for runs with and without a first frame:

- **Video without a first frame** rewrites both, splitting what you wrote: setting, look and camera setup go to the scene, events in order to the action.
- **Video with a first frame** rewrites the action only when you gave a scene (what the image shows), so it isn't restyled. With a blank scene it writes a short scene too.
- **Extend** rewrites the action only when the clip has a scene, so a chain of extensions keeps its look. It uses the previous clip's prompt as context, so it doesn't repeat its actions or lines.

Action only needs a scene; with a blank one, both parts are written. Style, Mood, Theme, Shot, Angle, Palette and Focus on have no effect with Action only, since they shape the scene.

## JSON captions (Ideogram 4)

[Ideogram 4](models.md#ideogram-4-instant) reads structured JSON captions. With `ideo`, **Write as: JSON** (`--enhance-format json`) writes a full caption from your prompt: a summary, the look, lighting and colours, the background, one element per thing you named, and the lettering you asked for.

- The same options apply: Style decides photo or artwork, Mood and Theme shape the look, Focus on and Length decide how much is described.
- A palette's exact colours go into the caption's colour palette, and its colour names into the descriptions of the things in the picture. Shot and Angle are asked for in the camera line. Ideogram 4 often keeps eye level whatever the angle.
- Quoted text (`"OPEN"`, `'LISBOA'`, `「パン屋」`) and names after *called* or *named* become text elements, copied exactly. Signs you mention without their words may get short words that fit, in the language you name.
- The caption appears in the **Enhanced** tab, laid out for reading, and you can edit it there.
- If the enhancer returns no usable caption twice, the run reports it and keeps your prompt.

**Smart** writes noticeably better captions than Fast.

## In the Web UI

- **Enhance now:** click ✨ **Enhance** in the prompt box, pick options, and click **Enhance prompt**. The rewrite appears in the **Enhanced** tab. Enhancing always starts from your text, never from the Enhanced text.
- **The popover** groups the options as **Look** (Style, Mood, Theme), **Framing** (Shot, Angle), **Color** (Palette), **Focus on**, **Motion** and **Sound** for video, and **Output** (Rewrite or Write as, and Length), then Direction. With **Custom**, a row of colour dots appears under **Palette**: **+** adds a colour (up to five), tap a dot to change it, and **×** removes it. **Reset** puts every option back. **Enhancer** (Fast or Smart) sits next to **Enhance prompt**. Images and video remember their own options.
- **What Generate sends:** for each part, the Enhanced text when it has any, otherwise your text. A dot on the **Enhanced** tab means Generate sends the Enhanced text; no mark anywhere means it sends your text. In video one header and one stepper cover the scene and the action, because one Enhance writes both.
- **Out of date:** once your text changes after the rewrite was written, or you pick another model, the **Enhanced** tab shows ⚠ and the panel opens on **Enhance again**, with the reason beside it. The Enhanced text is still sent until you scratch it or enhance again. With **Enhance each image when generating** on, nothing is out of date: Generate sends your text and each image is enhanced as its job runs.
- **The last five versions:** each Enhance, or **Enhance again** after your text changed, adds a version and keeps the earlier ones, up to five. In the **Enhanced** tab, `‹ 2 of 3 ›` steps through them, and Generate sends the one shown. Typing in a shown version edits that version. The versions are kept in the browser with the rest of the form, not on the server.
- **Scratch:** **✕** after the open **Enhanced** tab empties the Enhanced text, so Generate sends your text again; ‹ brings the scratched version back.
- **Running:** a stop ring (or the panel's **Stop**) ends a run, and **Generate** waits while Enhance, a helper or Fill in is writing the text it sends.
- **Messages:** an error, or the note about the last result, shows as one line under the box. A run that fails shows your own text again.
- **Switching between Image and Video** opens the prompt box on your own tab; the Enhanced text is kept for its mode.
- **Random choices** are picked before the rewrite, so the Enhanced prompt has none left. Enhance again for a different pick.
- **Enhance each image when generating** rewrites every image's prompt as part of the job, after its random choices are picked. It works with inline prompts and prompt files. The job card shows *Enhancing prompt N of M*; a prompt that couldn't be enhanced is generated from the original.

The Enhance button is unavailable while jobs are running or queued, because it needs the same memory.

## Audio

On the Audio tab the enhancer rewrites the text of a take into the shape its model was trained on, so a short idea becomes a caption the model follows. It works on a **Sound** prompt and on a **Music** style. The enhancer never rewrites Music's Lyrics as part of a style: a separate [lyrics helper](#lyrics-helper) polishes them. Speech's text is read as written, with a separate [script helper](#script-helper) to mark it up or fit it to a time. A take is written from an idea in one place only, [Describe it](#describe-it-and-fill-in). `ziv audio --enhance` does the same for a sound's prompt and a song's style; see [Generate audio](../cli/audio.md#enhance-the-prompt). Every audio helper is unavailable while a take is being made, because the enhancer needs the same memory.

- **Sound** is rewritten into one short sentence for Stable Audio 3, at most 45 words, with no brackets, markdown, length or tempo. It describes only what is heard: the source, what it hits, the space, how it changes and how it moves. It adds no music or speech unless you ask for it.
- **The take's Type and Length** go along as context. An effect or a one-shot is written as one event, an ambience as a steady bed, and the rewrite describes only what fits in the take's length.
- **A Music style** takes the shape of the open model, so the same idea reads differently for each:

| Model | The style is written as | Example |
|---|---|---|
| YuE2 | One line of comma-separated descriptors: short noun phrases, no sentences, no period | *Jazz, expressive lead vocal, piano, tenor saxophone, upright bass, brushed drums, warm* |
| Stable Audio 3 | One sentence of at most 45 words, plain English, with no singing: instruments carry the melody | *A warm jazz tune led by tenor saxophone and piano over upright bass and brushed drums.* |

  What you leave out is written as "no" and the instrument (*no guitar*). A style never holds lyrics, a title, section tags, a tempo, a key or a length: Tempo, Key and Length are set in Settings. The enhancer is told whether the song is instrumental, so it describes no singer then; for a song YuE2 sings, the start of your lyrics tells it what the words are about, and it never copies them into the style. Stable Audio cannot sing, so it is never told the lyrics and a style for it describes no voice.

Sound has four options, and Music has six:

| Option | Choose | Values | Key |
|---|---|---|---|
| Space (Sound) | one | Keep, Small room, Large hall, Street, Open country, Forest, Underwater | `space` |
| Distance (Sound) | one | Keep, Close, Medium, Far | `distance` |
| Over time (Sound) | one | Keep, Steady, Builds, Hit then tail | `over_time` |
| Movement (Sound) | one | Keep, Still, Passing, Approaching, Receding | `movement` |
| Genre (Music) | one | Keep, then 32 genres in eight groups: Pop & rock, Electronic, Hip-hop & R&B, Jazz & blues, Folk & country, Classical & cinematic, World, Lo-fi & ambient | `genre` |
| Mood (Music) | one | Keep, Serene, Joyful, Romantic, Nostalgic, Melancholic, Mysterious, Dark, Dramatic, Epic, Hopeful, Energetic, Dreamy | `music_mood` |
| Instruments (Music) | any | Guitar, Piano, Drums, Bass, Strings, Synth, Brass, Saxophone, Violin, Flute, Organ, Percussion | `instruments` |
| Without (Music) | any | The same twelve; an instrument you ask for cannot also be left out | `without` |
| Singer (Music) | one | Keep, Female, Male, Duet, Choir | `singer` |
| Production (Music) | one | Keep, Polished, Lo-fi, Live, Vintage, Cinematic | `production` |
| Direction | free text | For example *keep the thunder far off* (up to 300 characters) | `note` |

**Singer** is offered only for a song the model sings: a model without lyrics (Stable Audio) and a song with no lyrics, or only `[section]` tags, are instrumental, and have no singer to choose.

Nothing is picked by default. With every choice at Keep the enhancer still rewrites your text into the model's shape.

The rewrite is made for the open model. It is kept as the **Enhanced** text beside your own, and it is out of date once your text changes or you pick another model: a Music style written for YuE2 shows ⚠ on its **Enhanced** tab when you switch to Stable Audio, and **Enhance again** writes it again in that model's shape. The Fast and Smart choice, the marks, Scratch, the stop ring and the last five versions work as they do for images (see [In the Web UI](#in-the-web-ui)).

The take keeps your own text and how it was enhanced: the options, the enhancer model and the kind of rewrite. **Reuse settings**, **Retry** and **Load settings** bring both back, with the rewrite in the **Enhanced** tab.

### Describe it and Fill in

**Describe it** is the one place that writes from an idea. A sentence such as *a sad folk song about leaving the coast* goes to the enhancer in a single request, and **Fill in** writes the fields you have left empty. Music and Speech have the box; Sound has none, since its prompt is already the idea.

| Take | Fill in can write | A field is empty when |
|---|---|---|
| Music on YuE2 | Style, Lyrics, Tempo, Key | It has no text (an Enhanced style does not count), or is Auto |
| Music on Stable Audio | Style, Tempo, Key (the model sings no lyrics) | The same |
| Speech | Text, Vibe | The text is blank, or the Delivery values are all at their defaults |

What it does:

- **It fills only what is empty.** A field you have written or set is left alone, and goes to the enhancer as context so the new fields suit it. The idea takes at most 1,000 characters.
- **The Style lands in your own Style box,** like Speech's Text, so it is yours to edit and Generate sends it as your text. **Enhance** then starts from it.
- **Names and quotes are copied exactly** into the lyrics and the text.
- **Your sections are kept.** When the Lyrics hold only `[section]` tags, Fill in follows those sections where it can, and **Undo fill** brings the tags back.
- **An instrumental has no lyrics.** If the idea asks for one, the Lyrics stay empty.
- **Speech** gets a Text in plain words and a Vibe. A mark such as `[sad]` appears only where the feeling clearly changes.

The Tempo and Key Fill in wrote, and the Vibe, carry a ✦ until you change them: edit the field and it is yours, and **Fill in again** rewrites only the fields that are empty or still marked. **Undo fill** puts back what the fill replaced, for the fields still marked. The take keeps the Style as your own text; the idea itself is not saved with it. Fill in uses the Fast or Smart enhancer chosen for the Audio tab.

### Lyrics helper

The lyrics helper works on the words you already have. It polishes your lyrics and fills an empty section to fit the rest; new lyrics from an idea come from Describe it, not from this helper. On a song of only `[section]` tags, the helper offers **Fill in from Describe it**, which writes the words into those sections. It is offered for a model that sings lyrics (YuE2); Stable Audio takes none.

**Improve lyrics** works on the whole song or on one section. **Tighten** is picked when the helper opens. Pick any of these, add a direction, or clear every pick for a general polish of rhythm and flow:

| Improvement | What it does |
|---|---|
| Tighten | Cuts filler and repeated ideas so each line is shorter and easier to sing, and keeps the meaning of every line |
| Fit the sections | Makes each section's length and weight suit its role (verses carry the story, choruses are short and catchy, a bridge is brief); whole song only |
| Smooth rhymes | Gives the line endings a clear, steady rhyme without bending the meaning |
| Simpler words | Swaps rare or formal words for plain ones that are easy to sing |

**Write** fills one section that has no words so that it fits the rest of the song: its language, voice, theme, line length and rhyme style, and about as many lines as the song's sections of the same name. A second chorus is written as a repeat of the first. A section is written only when the rest of the song has words to fit; on a song with none, Write points to Fill in.

What it keeps:

- **The sections.** Improving the whole song keeps your `[section]` tags as you wrote them, each on its own line, and a title above the first tag. A song without tags gets none, and a section without words stays without words. If the answer merges or drops sections, nothing changes and the helper says so.
- **Your names and quotes.** Names, dates, places and anything in double quotes are kept exactly.
- **Your language.** The words are written in the language of the lyrics.
- **One section.** Improving or writing one section changes only that section; the rest of the song is given as context.

The song's style (the Enhanced style when the take uses one, else your own) goes along as context, so the words suit it. Tempo, key and length are settings and do not go along.

Each result is a version of the lyrics, up to five with yours first, and Generate uses the one shown. The lyrics take at most 4,000 characters.

### Script helper

The script helper works on the script you already have in Speech's Text box. It marks the script up with delivery, or shortens or lengthens it to a reading time; a script written from an idea comes from Describe it, not from this helper. It is offered whenever the prompt enhancer is set up.

| Action | What it does |
|---|---|
| Fit to time | Shortens or lengthens the script so it reads in about the time you give, at the Pace in Settings |
| Add delivery | Adds delivery marks (`[sad]`, `[whisper]`, `[as set]`, ...), pauses (`…`) and line breaks where the meaning asks for them, and nothing else |

Pick either or both. With both, one request fits your Text and then adds delivery to the fitted text, so a run makes one version with one record and one **Stop**. When the delivery finds no place for a mark or pause, the fitted text is kept with a note.

What it keeps:

- **Your words.** Add delivery never changes a word, a quote mark or a bracket of yours: the result is your own text with only the marks, pauses and line breaks the helper added between your words. Marks and pauses you already wrote stay.
- **Only marks the app knows.** The helper adds only the marks the Insert menu offers (the emotions, `[whisper]`, `[shout]` and `[as set]`), at the start of a sentence or a line. A pause is `…`.
- **Your names and quotes.** A fit keeps names, numbers, dates and anything in double quotes as written.
- **Your language.** The script is edited in its own language.

**Fit to time** counts the characters that are read, not the marks, at the Pace in Settings. It starts at the script's current reading time rounded to the nearest five seconds, and takes 3 to 600 seconds. A fit that lands far from the time is kept and says so (*Came out at about 1:40.*). A voice that follows a recording is read as recorded, so its time is counted at a Normal pace, and Add delivery is not offered for it.

Each result is a **version** in the **Written** tab beside your own text, up to five, with a stepper. The helper always starts from your own Text, whichever tab is shown, and Generate sends the Written text. The Written text is out of date once your own text changes or you pick another model, and is still sent until you scratch it. The script takes at most 4,000 characters.

The take keeps your own text and what the helper did (the actions, and with a fit the time and Pace). **Reuse settings**, **Retry** and **Load settings** bring both back, with the script in the **Written** tab.

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
  - prompt: a ceramic teapot on a wooden table
    enhance:
      angle: overhead
      palette: custom
      colors: ["#1B3A4B", "#E07A5F"]   # quote hex codes
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

Jobs rewrite every prompt before the model loads. During that time, jobs respond to:

- **Skip** (++n++): skip the current rewrite; that image uses your prompt.
- **Quit** (++q++): stop before the model loads.
- **Pause** (++p++): wait until you resume.

**Repeat** and automatic retries keep the image's random choices and enhanced prompt, and only change the seed.

## Memory

The enhancer and the image or video model are never loaded together: the enhancer is unloaded, and its memory freed, before the model loads. Peak memory is the larger of the two, not their sum.

Clicking Enhance in the Web UI loads the enhancer (about 3 GB on macOS) and frees it after two idle minutes, or when a job starts.

## What is recorded

The enhanced prompt is what's stored in the image or video, so the Gallery shows and reuses the prompt that was actually used. The file also keeps your own prompt and how it was enhanced: the options, the enhancer model and Text or JSON.
