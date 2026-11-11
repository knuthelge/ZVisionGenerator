# Make speech and sound

Type a text and have it read aloud in one of nine built-in voices or in a voice of your own, or describe a sound and have it made. Everything runs on your own computer, in the **Audio** tab; its controls are described in [Audio tab](../web-ui/audio.md). Songs and instrumentals are in [Make music](music.md).

| Kind | Model | Longest take |
|---|---|---|
| **Speech** | Qwen3-TTS 1.7B | About 10 minutes (10,000 characters) |
| **Sound** | Stable Audio 3 Medium | 380 s |

Download sizes are in [Models and licences](../reference/models.md#audio-models) and what they need to run is in [Memory](../concepts/memory.md). Cloning and designing a voice each download more files the first time they are used. Stable Audio has licence terms that limit some uses; read them in [Licences](../reference/models.md#licences) before you use the output. For a first take, [Your first audio](../get-started/first-audio.md) goes step by step.

## Write the text { #speech }

Set **Kind** to **Speech** and type what should be read in **Text**, as plain text. It takes up to 10,000 characters, about 10 minutes of speech. Longer text is refused, so split it across takes. A long text is read in parts and joined into one take, every part in the same voice.

[Describe it](music.md#describe-it) can write the Text and pick a Vibe from an idea, and the [script helper](#script-helper) can mark up or fit a script you already have.

The model works out the language of the text itself, and the [`language`](../reference/configuration.md#audio_model_presets) setting of the model's preset can name one instead. It reads ten languages: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish and Italian.

### Insert a pause, a laugh or a delivery { #pauses-and-laughs }

**Insert** in the Speech tools row, or ++alt+enter++ in the text, puts a **Pause** (`…`), a **Laugh** (`haha`) or a [mark](#marks) at the caret. A line break stays a line break inside a part and pauses no longer than `…` does, so write what you want to hear: `…` for a pause and `haha` for a laugh. A delivery mark is not offered while the voice follows a recording.

**On a phone:** **Insert** opens as a sheet.

## Pick a voice { #voices }

**Voice** is the first group in Speech's **Settings**: your saved voices above the nine default voices. **Ryan** is the default.

The language of a default voice is its native one, not a limit: every voice reads all ten languages and carries its accent into the others. English text reads without an accent in **Ryan** or **Aiden**.

A saved voice is a short reference recording and what it says; any take can read its text in that voice. Saved voices live in `voices/` in the [data folder](../reference/data-folder.md) (`~/.ziv` by default) and never appear in the Gallery. A voice comes from a description, a recording or a speech take. The first two are made in the **Voices** window, which **Your voices** and **New voice**, under the dropdown, open; the third is made with **Use as** in the Gallery or the viewer.

**On a phone:** **Voice** is a chip that opens a sheet with the dropdown and the buttons below. The Voices window fits the screen, with its three tabs in a row.

### Design a voice

The **New voice** tab makes a voice from a description, with the model's VoiceDesign files, which download the first time you use it.

1. Describe the voice (up to 500 characters): age, gender, tone, pace and accent, such as `A warm British woman in her thirties, calm and clear`. Say how it should sound as well: warm, slow, whispered, excited.
2. Check the preview line. Every voice reads it, and it is saved with the voice as its transcript, so it must read in 3 to 30 seconds.
3. Make three candidates and play each one.
4. Choose one, name it and save it. The voice is chosen for the next takes.

The same description makes a different voice every time, so save the candidate you like: it is a recording that the model clones in every later take, and the voice stays the same. Candidates are kept in `voice-design/` in the data folder, never in the Gallery.

Delivery cannot change a saved voice (see [Delivery](#delivery)), so put the manner into the description: `A calm man in his fifties with a low voice, like a nature documentary narrator`.

### Clone a voice from a recording

The **From recording** tab follows a recording of someone's voice. It needs a recording of 3 to 30 seconds of speech, the words spoken in it, and the speaker's consent. Use a clean, close-miked recording without background noise: a phone, radio or old recording makes the cloned voice hiss and sound less like the speaker.

**Use once** makes it the voice of the next takes; naming and saving it keeps it in **Your voices** instead. The cloning files download the first time a take follows a recording or a saved voice.

### Use a take as a voice

A speech take of 30 seconds or less can become a voice. Open **Use as** in the Gallery, the viewer or the **⋯** menu of the finished take:

- **Voice** opens **Speech** with the take as a one-off recording and its text under **What it says**.
- **Saved voice** saves it into **Your voices**, after asking for a name and what it says. A take that followed a recording or a saved voice also asks for [consent](#consent).

Both are unavailable for a take longer than the model's voice limit. The take must be a speech take; music and sound takes offer neither. **Audio track**, which gives the take to a video, is not limited; see [Use a take elsewhere](music.md#use-a-take-elsewhere).

### Consent { #consent }

Cloning a voice needs the speaker's permission. Consent is asked per saved voice: a recording, or a take that followed one, asks once when you save it as a voice, and the voice remembers it. Used once, it isn't remembered, so it asks again after a page reload or a change of voice. Designed voices and the default voices are nobody's voice, so they don't ask. The consent is stored in the voice's record in `voices/` in the data folder, never in a take's file or the Gallery.

### Manage your voices

The **Voices** window has a tab to manage saved voices (**Your voices**), one to design a voice (**New voice**, when the open speech model can) and one to make a voice from a recording (**From recording**).

To rename or delete a saved voice:

1. Click **Your voices** and find the voice on the **Your voices** tab.
2. Click its **Rename** (pencil) or **Delete** (bin) button. A new name must not be taken by another voice.
3. **Delete** asks you to confirm. Takes made with the voice keep their sound.

A voice can't be deleted while a job runs.

## Set the delivery { #delivery }

**Delivery** is how a default voice says the text: emotion, manner (natural, whisper or shout), pace, pitch and expressiveness. The **Vibe** tiles set all five at once as a starting point, and each stays editable. With every setting neutral, the model reads in its own manner. A take saved with its settings brings the five values back on Reuse.

**Delivery works with the default voices only.** A saved voice or a recording reads as it was made, so a take that follows one has no delivery. To get a different manner from a saved voice, make a new voice and describe that manner when you [design it](#design-a-voice).

**On a phone:** **Delivery** is a chip that opens a sheet with the same Vibes and rows.

### Change the delivery partway with marks { #marks }

A **mark** in the text changes the delivery from that point on, for a default voice. It is a word in square brackets:

| Mark | Sets |
|---|---|
| `[sad]`, `[angry]`, `[proud]` and the other emotions | Emotion |
| `[whisper]`, `[shout]` | Manner |
| `[as set]` | Everything back to the delivery you set |

```text
I thought we were friends. [sad] You never even called. [as set] Anyway, welcome back.
```

**Insert** → **Delivery from here** puts a mark at the start of a sentence for you. A mark holds until the next mark or the end of the text. It sets one thing and the rest stays as set: `[whisper]` after `[sad]` is whispered in the delivery you set, not whispered and sad. The marks are not read aloud and do not count towards the 10,000 characters.

A saved voice or a recording reads the text as made, so its marks are left out.

A bracketed word that is not a mark, such as `[sadd]`, is refused before anything is made, and so is a text of marks alone. Other bracketed text, such as `[1]`, is read as written.

## Get help with the script { #script-helper }

The script helper works on the script already in Speech's **Text** box. It marks the script up with delivery, or shortens or lengthens it to a reading time. A script written from an idea comes from [Describe it](music.md#describe-it), not from the helper. **Script helper** in the Speech tools row opens a panel; it is offered wherever the prompt enhancer is set up and is unavailable while a take is being made. It always starts from your own **Text**, whichever tab is shown.

| Action | What it does |
|---|---|
| **Fit to time** | Shortens or lengthens the script so it reads in about the time you give, at the Pace in Settings |
| **Add delivery** | Adds delivery marks (`[sad]`, `[whisper]`, `[as set]`, …), pauses (`…`) and line breaks where the meaning asks for them, and nothing else |

Pick either or both. **Fit to** takes 3 to 600 seconds and starts at the script's current reading time rounded to the nearest five seconds. With both picked, one request fits the Text and then adds delivery, so a run makes one version with one record and one **Stop**. When the delivery finds no place for a mark, the fitted text is kept with a note. **Fast** or **Smart** beside the run button is the enhancer. **Add delivery** is left out while the voice follows a recording.

**On a phone:** the panel opens as a sheet.

Each result is a version in the **Written** tab beside your own text. ‹ **2 of 3** › steps through up to five, and you can edit the Written text by hand. Generate sends the Written text. It goes out of date once your own text changes or you pick another model, and is still sent until you scratch it. The script takes at most 4,000 characters.

??? note "What the helper keeps, and how a fit is counted"

    - **Your words.** Add delivery never changes a word, a quote mark or a bracket of yours: the result is your own text with only the marks, pauses and line breaks the helper added. Marks and pauses you already wrote stay.
    - **Only marks the app knows.** It adds only the marks the Insert menu offers (the emotions, `[whisper]`, `[shout]` and `[as set]`), at the start of a sentence or a line. A pause is `…`.
    - **Names, numbers, dates and quotes** in double quotes are kept as written, and the script is edited in its own language.
    - **Fit to time** counts the characters that are read, not the marks, at the Pace in Settings. A fit that lands far from the time is kept and says so. A voice that follows a recording is read as recorded, so its time is counted at a Normal pace.

The take keeps your own text and what the helper did: the actions, and with a fit the time and Pace. **Reuse settings**, **Retry** and **Load settings** bring both back, with the script in the **Written** tab.

## Make sound effects and ambience { #sound }

Set **Kind** to **Sound**. The model is Stable Audio 3 Medium, and the same download makes [music](music.md#stable-audio-under-music). **Prompt** describes what you hear, such as `rain on a tin roof, distant thunder`. Sound has no Describe it box, since the prompt is already the idea.

In **Settings**, the **Sound** section comes first:

- **Type** (Auto, Ambience, Effect or One-shot) says what kind of sound it is, and sets the length the sound has when you give none.
- **Length (s)** is how long the sound lasts, from 1 to 380 seconds. Typing a number locks it, whatever the type; emptying the field returns to Auto. **Reuse settings** brings back only a length you gave.

**On a phone:** **Type** and **Length** are chips after **Model**, each with a sheet.

### Polish the prompt with the enhancer

**Enhance** under the Prompt rewrites it into one short sentence for Stable Audio, at most 45 words, with no brackets, markdown, length or tempo. It describes only what is heard: the source, what it hits, the space, how it changes and how it moves. It adds no music or speech unless you ask for it. The take's **Type** and **Length** go along: an effect or a one-shot is written as one event, an ambience as a steady bed, and the rewrite describes only what fits in the take's length.

These options each start on **Keep**; with all of them on Keep, the enhancer still rewrites your text into the model's shape:

| Option | Values | Key |
|---|---|---|
| **Space** | **Keep**, **Small room**, **Large hall**, **Street**, **Open country**, **Forest**, **Underwater** | `space` |
| **Distance** | **Keep**, **Close**, **Medium**, **Far** | `distance` |
| **Over time** | **Keep**, **Steady**, **Builds**, **Hit then tail** | `over_time` |
| **Movement** | **Keep**, **Still**, **Passing**, **Approaching**, **Receding** | `movement` |
| **Direction** | Your own words, up to 300 characters, such as `keep the thunder far off` | `note` |

**Reset** puts every choice back to Keep and clears the direction. The rewrite is kept in the **Enhanced** tab beside your text, and **Reuse settings**, **Retry** and **Load settings** bring both back. The enhancer, its **Fast** and **Smart** choice and its versions work as they do for images; see [Prompt enhancer](../concepts/prompt-enhancer.md). The panel opens as a sheet on a phone or tablet.

## From the command line

`ziv-audio` makes the same takes; `qwen3_tts` reads speech and `sa3` makes sound.

```bash
ziv-audio -m qwen3_tts --prompt "And then it rained." --speaker aiden
ziv-audio -m sa3 --prompt "rain on a tin roof, distant thunder" --seconds 30
```

Recipes for saved voices, recordings and delivery are in [Make audio](../cli/audio.md#speech).
