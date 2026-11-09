# Audio

The app makes music, speech and sound on your own computer. There are three kinds; music and sound each have two models between them:

| Kind | Makes | Model | Alias | Download | Longest take |
|---|---|---|---|---|---|
| **Music** | Songs and instrumentals | YuE2 3B | `yue2` | about 8 GB | 6:00 |
| **Music** | Instrumentals | Stable Audio 3 Medium | `sa3` | about 10 GB, shared with its sound | 380 s |
| **Speech** | Text read aloud, in one of nine default voices or in a voice of your own | Qwen3-TTS 1.7B | `qwen3_tts` | about 4.5 GB, and about 4.5 GB more for cloning a voice and again for designing one, each the first time it is used | about 10 minutes (10,000 characters) |
| **Sound** | Ambience and sound effects | Stable Audio 3 Medium | `sa3` | about 10 GB | 380 s |

In the [Web UI](../web-ui/workspace.md#the-audio-tab) they are the **Audio** tab; in the [CLI](../cli/audio.md) they are `ziv audio`. Every take is saved as a [FLAC file](#takes-and-files) that you can play in the [Gallery](../web-ui/gallery.md#audio), reuse, or give to a video as its [audio track](video.md#audio-track).

The models run on Windows and Linux with an NVIDIA GPU (10 GB is enough) and on macOS with Apple silicon (32 GB of unified memory is enough). See [Memory](memory.md#audio). Each model downloads the first time you use it. YuE2 and Stable Audio 3 are not free for every use: read the terms in [Licences](models.md#licences) before you use the output.

## Music

Music is made by YuE2 or Stable Audio 3. With YuE2, music has two boxes: a **Style** and, optionally, **Lyrics**. Stable Audio takes the style alone: see [Stable Audio under Music](#stable-audio-under-music).

- **Style** (required) describes the music: genre, mood, instruments and the singer, such as `upbeat synth-pop, female vocal`.
- **Lyrics** are written in sections, each starting with a tag such as `[verse]` or `[chorus]`:

  ```text
  [verse]
  Streetlights hum along the avenue
  [chorus]
  Take me home tonight
  ```

What the Lyrics box holds decides what is made:

| Lyrics | Makes |
|---|---|
| Words under section tags | A **song**. Its length follows the lyrics |
| Empty | An **instrumental** of eight sections: `[Intro] [Instrumental] [Solo] [Instrumental] [Solo] [Bridge] [Instrumental] [Outro]` |
| Only section tags, such as `[Intro] [Instrumental] [Solo] [Outro]` | An **instrumental** with those sections |

The section names the app offers are `Intro`, `Verse`, `Pre-Chorus`, `Chorus`, `Post-Chorus`, `Bridge`, `Solo`, `Instrumental` and `Outro`. Any `[tag]` on a line of its own starts a section, with a name of your own too, and the tag is kept as you wrote it, in lowercase or title case. Words on the same line as a tag, or before the first tag, are lyrics like any other. A tag with no words under it is an instrumental section.

In the Web UI, [Describe it](prompt-enhancer.md#describe-it-and-fill-in) can write the Style, Lyrics, Tempo and Key from an idea.

There is no length setting. A song is as long as its lyrics make it, and an instrumental follows its section list: add tags for a longer piece and remove them for a shorter one.

The model stops at **6:00**. A take that reaches it fades out and is marked **Cut at 6:00**; shorten the lyrics to hear the ending.

An instrumental comes without a voice. Use `[Instrumental]` and `[Solo]` tags rather than `[verse]` and `[chorus]`, which the model reads as places to sing.

### Tempo and Key

Tempo (`--tempo`) and key (`--key`) steer how the music is composed. Left out, the model chooses. In the Web UI they are the **Tempo** and **Key** rows of the **Music** section in Settings; both read **Auto** until you set them, and **Reset** puts a row back to Auto.

- Tempo is a whole number of beats per minute, from 40 to 220.
- Key is one of 24: the twelve major keys (`C`, `Db`, `D`, `Eb`, `E`, `F`, `F#`, `G`, `Ab`, `A`, `Bb`, `B`) and the twelve minor ones, written with an `m` (`Cm`, `C#m`, `Dm`, and so on up to `Bm`). `F#m` is F♯ minor.

The song is composed in the tempo and key you set, in a major or a minor key alike, and the style text is not changed. If the model cannot plan in them, the take uses its own plan and says so in a note. The same settings and seed give the same take.

### Stable Audio under Music

Stable Audio 3 Medium also makes music: a piece of instrumental music from a **Style**. It takes no lyrics, so the Lyrics box is not shown for it and `--lyrics` is refused. Choose it under **Music** in the model picker, or with `-m sa3 --kind music`.

- **Length** (`--seconds`) is how long the piece lasts, from 1 second to 380. Left out, it is 30 seconds. In the Web UI it is the **Length (s)** row of the **Music** section, which YuE2 does not have: a YuE2 song is as long as its lyrics make it.
- **Tempo** and **Key** are written into the description the model is given, as `92 BPM, A minor`, and the piece is asked to be without vocals. The model reads them as part of the description, not as exact settings.

The same model and the same download make sound and music.

## Speech

**Text** is what is read, as plain text. It can hold up to 10,000 characters, about 10 minutes of speech; longer text is refused, so split it across takes. A long text is read in parts and joined into one take.

In the Web UI, [Describe it](prompt-enhancer.md#describe-it-and-fill-in) can write the Text and pick a Vibe from an idea, and the count under the box tells how long the text takes to read. The time is an estimate at the chosen Pace; a recording is counted at Normal.

### Voices

**Voice** is one library: **Your voices**, the ones you saved, above the nine **Default voices**.

| Default voice | Native language |
|---|---|
| **Ryan** (the default), **Aiden** | English |
| **Vivian**, **Serena**, **Uncle Fu** | Chinese |
| **Dylan** | Beijing Chinese |
| **Eric** | Sichuan Chinese |
| **Ono Anna** | Japanese |
| **Sohee** | Korean |

The language beside a default voice is its native one, not a limit: every voice reads all ten languages the model knows (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish and Italian) and carries its accent into the others. English text reads without an accent in **Ryan** or **Aiden**. The model works out the language of the text itself; the [`language`](configuration.md#audio_model_presets) setting of the model's preset can name one instead. Every part of a long text is read in the same voice.

Default voices come with the model.

#### Your voices

A saved voice is a short reference recording and what it says; any take can read its text in that voice. Saved voices live in `~/.ziv/voices/` and never appear in the Gallery.

A voice comes from a [description](#design-a-voice), a [recording](#follow-a-recording) or a speech take of 30 seconds or less (**Use as** → **Saved voice** in the Gallery).

Voices are made in the Web UI. The CLI uses them with `--voice NAME`; see [A saved voice](../cli/audio.md#a-saved-voice).

#### Design a voice

**New voice** makes a voice from a description, with the model's VoiceDesign checkpoint. Describe how the voice sounds (age, gender, tone, pace and accent), such as `A warm British woman in her thirties, calm and clear`, and the model makes three candidates that each read a **preview line**. The line starts as a sentence in the language of your Speech text and is saved with the voice as its transcript, so it must read in 3 to 30 seconds.

Candidates are kept in `~/.ziv/voice-design/`, never in the Gallery. Designing needs the VoiceDesign files, a separate download of about 4.5 GB the first time. Stopping a run keeps the candidates made so far.

The same description makes a different voice every time, so save the candidate you like: it is a recording that the model clones in every later take, and the voice stays the same. Delivery cannot change a saved voice (see [Delivery](#delivery)), so describe the manner in the description: `A calm man in his fifties with a low voice, like a nature documentary narrator`.

#### Follow a recording

A recording needs three things:

- The **recording**, 3 to 30 seconds of speech.
- **What it says**: the words spoken in the recording.
- A **consent tick**: *This is my voice, or I have the speaker's permission to clone it.* The app asks for it so that no voice is cloned without the speaker's consent.

!!! tip "Clone from a clean recording"
    Use a clean, close-miked recording without background noise. A phone, radio or old recording makes the cloned voice hiss and sound less like the speaker.

A recording can be used **once**, for the next takes, or **saved** as a voice. The cloning files (about 4.5 GB) download the first time a take follows a recording or a saved voice.

#### Consent

Cloning a voice needs the speaker's permission, which you confirm with a tick. A recording, or a take that followed one, asks once when you save it as a voice, and the voice remembers it. Used once, it isn't remembered, so it asks again after a page reload or a change of voice. Designed voices and the default voices are nobody's voice, so they don't ask.

The consent is stored in the voice's record in `~/.ziv/voices/`, never in a take's file or the Gallery.

### Delivery

**Delivery** is how a default voice says the text. It has five settings, set with `--emotion`, `--manner`, `--pace`, `--pitch` and `--expressiveness`. Left out, a setting stays at its neutral value:

| Flag | Choices | Neutral |
|---|---|---|
| `--emotion` | `neutral`, then 20: `elated`, `amused`, `enthusiastic`, `proud`, `content`, `affectionate`, `relieved`, `longing`, `thoughtful`, `determined`, `awed`, `surprised`, `confused`, `sad`, `bitter`, `ashamed`, `helpless`, `afraid`, `angry`, `disgusted` | `neutral` |
| `--manner` | `natural`, `whisper`, `shout` | `natural` |
| `--pace` | `very-slow`, `slow`, `normal`, `fast`, `very-fast` | `normal` |
| `--pitch` | `low`, `normal`, `high` | `normal` |
| `--expressiveness` | `low`, `normal`, `high` | `normal` |

In the Web UI the five settings are the rows of the **Delivery** section in Settings, under six **Vibes** that set all five at once:

| Vibe | Emotion | Manner | Pace | Pitch | Expressiveness |
|---|---|---|---|---|---|
| Calm narrator | thoughtful | natural | slow | normal | normal |
| Storyteller | content | natural | normal | normal | high |
| Excited host | enthusiastic | natural | fast | high | high |
| Announcer | determined | natural | normal | normal | low |
| Sad confession | sad | natural | slow | low | normal |
| Whispered secret | neutral | whisper | slow | low | low |

A Vibe is a starting point, not a mode: picking one only sets the five values, and each stays editable. A take saved with its settings brings the five values back on Reuse.

With every setting neutral, the model reads in its own manner.

**Delivery works with the default voices only.** A saved voice or a recording reads as it was made, so a take that follows one has no delivery, and the CLI refuses the delivery options together with `--voice`. To get a different manner from a saved voice, make a new voice and describe that manner when you [design it](#design-a-voice).

#### Marks

A **mark** in the text changes the delivery from that point on, for a default voice. It is a word in square brackets:

| Mark | Sets |
|---|---|
| `[sad]`, `[angry]`, `[proud]`, and the other 20 emotions | Emotion |
| `[whisper]`, `[shout]` | Manner |
| `[as set]` | Everything back to the delivery you set |

```text
I thought we were friends. [sad] You never even called. [as set] Anyway, welcome back.
```

In the Web UI, [**Insert → Delivery from here**](../web-ui/workspace.md#insert-into-the-text) puts a mark at the start of a sentence for you.

A mark holds until the next mark or the end of the text. A mark sets one thing and the rest stays as set: `[whisper]` after `[sad]` is whispered in the delivery you set, not whispered and sad. The marks are not read aloud and do not count towards the 10,000 characters. A saved voice or a recording reads the text as made, so its marks are left out and the take carries a note saying so.

A bracketed word that is not a mark, such as `[sadd]`, is refused before anything is made, with a message naming the word. Remove the brackets or use one of the marks above; for a pause write `…`, and for a laugh write `haha`. A text of marks alone is refused as empty. Other bracketed text, such as `[1]`, is read as written.

#### Pauses and laughs

Write what you want to hear. `…` is a short pause and `haha` is a laugh. A line break stays a line break inside a part, so the model reads it as written; it pauses no longer than `…` does. In the Web UI, the **Insert** menu adds both.

## Sound

**Prompt** describes what you hear, such as `rain on a tin roof, distant thunder`.

The sound type (`--sound-type`) says what kind of sound it is, and sets the length the sound has when you give none:

| `--sound-type` | Length when none is given |
|---|---|
| `auto` | 12 s |
| `ambience` | 12 s |
| `effect` | 5 s |
| `one-shot` | 2 s |

The length (`--seconds`) is how long the sound lasts, from 1 second to 380. Left out, it is the length of the type; a length you give is used as it is, whatever the type. Reuse settings brings back only the length you gave. In the Web UI, **Type** and **Length (s)** are the [Sound](../web-ui/workspace.md#sound) section: the empty Length field shows the number Auto will use.

## Takes and files

Every take is a lossless FLAC file named `music_<date>_<time>.flac`, `speech_…` or `sound_…`. The settings that made it are stored inside, which is how the Gallery's **Reuse settings** works for audio; the speaker's consent is not among them. For a cloned voice, the settings name the recording and what it says; a saved voice is shown by its name in the Gallery while it exists, and Reuse picks it again.

A take can be used as a video's audio track: **Use as** → **Audio track** in the Gallery or viewer opens **Video** with the file as the clip's [audio track](video.md#audio-track). A speech take of 30 seconds or less can also be used as the **Voice** of the next speech take, or saved with **Use as** → **Saved voice**.

One batch (**Batch size** in the Web UI, `--runs` in the CLI) makes several takes with a single model load, each with its own random seed unless you set one.

## Next

[The Audio tab](../web-ui/workspace.md#the-audio-tab){ .md-button .md-button--primary } [Generate audio in the CLI](../cli/audio.md){ .md-button }
