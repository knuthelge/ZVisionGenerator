# Make music

Write a style and, if you want a singer, lyrics, and the app makes a song or an instrumental on your own computer. The controls are described in [Audio tab](../web-ui/audio.md); speech and sound effects are in [Make speech and sound](speech-and-sound.md).

## Pick a model

In the **Audio** tab, set **Kind** to **Music** and choose a **Model**:

| Model | Makes | Length |
|---|---|---|
| **YuE2 3B** | Songs with a singer, and instrumentals | As long as the lyrics make it, up to 6:00 |
| **Stable Audio 3 Medium** | Instrumentals only | 1 to 380 s, 30 s to begin with |

Download sizes are in [Models and licences](../reference/models.md#audio-models) and what they need to run is in [Memory](../concepts/memory.md). Both models have licence terms that limit some uses, so read them in [Licences](../reference/models.md#licences) before you use the output.

The rest of this page is about YuE2. Stable Audio has its own section, [Make music with Stable Audio](#stable-audio-under-music).

## Write the style and lyrics { #music }

**Style** is required. It describes the music: genre, mood, instruments and the singer, such as `upbeat synth-pop, female vocal`.

**Lyrics** are optional. The box shows a song as one block per section, each with its words; **Edit as text** shows the whole lyrics as plain text and back. The block controls are in [Song blocks](../web-ui/audio.md#song-blocks).

**On a phone:** reorder sections with **Move up** and **Move down** in a block's **⋯** menu.

In the text form, a section starts at a `[tag]` on a line of its own, and the name can be your own:

```text
[verse]
Streetlights hum along the avenue
[chorus]
Take me home tonight
```

There is no length setting: shorten the lyrics or drop a section for a shorter song.

## Make an instrumental

What the Lyrics box holds decides what is made:

| Lyrics | Makes |
|---|---|
| Words under section tags | A **song** |
| Empty | An **instrumental** of eight sections: `[Intro] [Instrumental] [Solo] [Instrumental] [Solo] [Bridge] [Instrumental] [Outro]` |
| Only section tags, such as `[Intro] [Instrumental] [Solo] [Outro]` | An **instrumental** with those sections |

A section with no words is an instrumental section, so a list of empty blocks makes an instrumental of those sections. Each tag is one section: add sections for a longer piece and remove them for a shorter one.

Use `[Instrumental]` and `[Solo]`, not `[verse]` and `[chorus]`. The model reads `[verse]` and `[chorus]` as places to sing.

## Set tempo, key and length { #tempo-and-key }

Tempo and key steer how the music is composed. Left out, the model chooses. In **Settings**, the **Music** section has the rows:

- **Tempo** is a whole number of beats per minute, from 40 to 220.
- **Key** is a major key (`C`, `Db`, `F#`) or a minor key written with an `m` (`Cm`, `F#m`).

**On a phone:** **Tempo**, **Key** and, where the model has it, **Length** are chips after **Model**. Each opens a sheet with the same control and a **Reset**.

A YuE2 song is composed in the tempo and key you set, in a major or a minor key alike, and the Style text is not changed. If the model cannot plan in them, the take uses its own plan and says so in a note. The same settings and seed give the same take. A YuE2 song has no **Length (s)** row: its length comes from the lyrics.

## Make music with Stable Audio { #stable-audio-under-music }

Stable Audio 3 Medium makes an instrumental from a **Style**. Choose it under **Music** in the **Model** list. It takes no lyrics, so the **Lyrics** box is not shown for it.

- **Length (s)** is how long the piece lasts, from 1 to 380 seconds, 30 to begin with. It is a row of the **Music** section that YuE2 does not have.
- **Tempo** and **Key** are written into the description the model is given, as `92 BPM, A minor`, and the piece is asked to be without vocals. The model reads them as part of the description, not as exact settings.

It is the same model and the same download that makes [sound effects and ambience](speech-and-sound.md#sound).

## Write it from an idea with Describe it { #describe-it }

**Describe it** is a box at the top of Compose for Music and Speech, shown wherever the prompt enhancer is set up. Type an idea such as `a sad folk song about leaving the coast` and press **Fill in**. It writes only the fields you left empty (for YuE2 the Style, Lyrics, Tempo and Key; for Speech the Text and Vibe), and uses what you already wrote as context. The Style lands in your own Style box, so it is yours to edit. Names and quotes are copied exactly, an instrumental gets no lyrics, and sections you wrote as `[section]` tags are followed. Sound has no Describe it box, since its prompt is already the idea. The box is described in [Audio tab](../web-ui/audio.md#describe-it).

## Improve lyrics { #lyrics-helper }

The lyrics helper works on words you already have: it polishes them and fills an empty section to fit the rest. New lyrics from an idea come from [Describe it](#describe-it). It is offered for a model that sings (YuE2) wherever the prompt enhancer is set up; see [Lyrics helper](../web-ui/audio.md#lyrics-helper) for where it opens.

- **Improve lyrics** works on the whole song, or on one section from that block. Pick what to improve (tighten, fit the sections, smooth rhymes, simpler words), add a **Direction** such as `make it more hopeful`, or clear every pick for a general polish.
- **Write** fills one section that has no words so that it fits the rest of the song: its language, voice, theme, line length and rhyme style. It needs words elsewhere in the song.

??? note "What the helper keeps"

    - **The sections.** Your `[section]` tags stay as you wrote them, and a section without words stays without words. If the answer merges or drops sections, nothing changes and the helper says so.
    - **Your names and quotes.** Names, dates, places and anything in double quotes are kept exactly, in the language of the lyrics.
    - **One section.** Improving or writing one section changes only that section.
    - **The style as context.** The song's style goes along so the words suit it. Tempo, key and length do not.

Each result is a version of the lyrics, and Generate uses the one shown. The lyrics take at most 4,000 characters.

## Polish the style with the enhancer { #enhance-the-prompt }

**Enhance** under the Style rewrites it into the shape the open model was trained on, so a short idea becomes a style the model follows. It never rewrites the Lyrics; the [lyrics helper](#lyrics-helper) does. The enhancer, its **Fast** and **Smart** choice, the marks, **Scratch**, the stop ring and the last five versions work as they do for images; see [Prompt enhancer](../concepts/prompt-enhancer.md).

The same idea reads differently for each model:

| Model | The style is written as | Example |
|---|---|---|
| YuE2 | One line of comma-separated descriptors: short noun phrases, no sentences, no period | *Jazz, expressive lead vocal, piano, tenor saxophone, upright bass, brushed drums, warm* |
| Stable Audio 3 | One sentence of at most 45 words, plain English, with no singing: instruments carry the melody | *A warm jazz tune led by tenor saxophone and piano over upright bass and brushed drums.* |

Nothing is picked by default, and with every choice at **Keep** the enhancer still rewrites your text into the model's shape. These options steer it:

| Option | Choose | Values | Key |
|---|---|---|---|
| **Genre** | one | **Keep**, then 32 genres in eight groups: Pop & rock, Electronic, Hip-hop & R&B, Jazz & blues, Folk & country, Classical & cinematic, World, Lo-fi & ambient | `genre` |
| **Mood** | one | **Keep**, **Serene**, **Joyful**, **Romantic**, **Nostalgic**, **Melancholic**, **Mysterious**, **Dark**, **Dramatic**, **Epic**, **Hopeful**, **Energetic**, **Dreamy** | `music_mood` |
| **Instruments** | any | Guitar, Piano, Drums, Bass, Strings, Synth, Brass, Saxophone, Violin, Flute, Organ, Percussion | `instruments` |
| **Without** | any | The same twelve; an instrument you ask for cannot also be left out | `without` |
| **Singer** | one | **Keep**, **Female**, **Male**, **Duet**, **Choir** | `singer` |
| **Production** | one | **Keep**, **Polished**, **Lo-fi**, **Live**, **Vintage**, **Cinematic** | `production` |
| **Direction** | free text | In your own words, up to 300 characters, such as `keep the guitar fingerpicked` | `note` |

**Singer** is offered only for a song the model sings. Stable Audio, and a song with no lyrics or only `[section]` tags, are instrumental and have no singer to choose.

What the enhancer is told, and what it leaves out:

- What you leave out is written as `no` and the instrument (`no guitar`).
- A style never holds lyrics, a title, section tags, a tempo, a key or a length. Tempo, Key and Length are set in Settings, and a Stable Audio style describes no voice.
- It is told whether the song is instrumental, so it describes no singer then. For a song YuE2 sings, the start of your lyrics tells it what the words are about; it never copies them into the style. Stable Audio is never told the lyrics.

The rewrite is kept as the **Enhanced** text beside your own. It is out of date once your text changes or you pick another model: a style written for YuE2 shows ⚠ on its **Enhanced** tab when you switch to Stable Audio, and **Enhance again** writes it again in that model's shape. The take keeps your own text and how it was enhanced (the options, the enhancer model and the kind of rewrite). **Reuse settings**, **Retry** and **Load settings** bring both back, with the rewrite in the **Enhanced** tab.

**On a phone:** the Enhance panel opens as a sheet.

## Find your takes { #takes-and-files }

Every take is a lossless FLAC file named `music_<date>_<time>.flac` (`speech_…` and `sound_…` for the other kinds), saved in the output folder; see [Output files](../reference/data-folder.md#output-files). It appears in the [Gallery](../web-ui/gallery.md), which has an **Audio only** filter, and as a player in the Audio tab, where clicking or dragging the waveform seeks. Nothing plays by itself.

The settings that made the take are stored inside the file, which is how **Reuse settings** works for audio: it opens the Audio tab on the take's kind with the take's settings. For a take that was enhanced or filled in, your text and the rewrite come back separately, as described above.

**Batch size** makes several takes with a single model load, each with its own random seed unless you set one.

## Use a take elsewhere

**Use as** in the Gallery, the viewer or the **⋯** menu of the finished take has **Audio track**, which opens **Video** with the file as the clip's audio track. It is greyed, with the reason, when the open video model can't follow a track. See Add sound or an audio track in [Make video](video.md).

A speech take also offers **Voice** and **Saved voice**; see [Use a take as a voice](speech-and-sound.md#use-a-take-as-a-voice).

## From the command line

`ziv-audio` makes the same takes; the model you pick with `-m` decides what.

```bash
ziv-audio -m yue2 --prompt "upbeat synth-pop, female vocal" --lyrics-file song.txt --tempo 96 --key F#m
ziv-audio -m sa3 --kind music --prompt "warm lo-fi beat, soft keys" --tempo 80 --key Dm --seconds 90
```

Recipes for lyrics files and more are in [Make audio](../cli/audio.md#music).
