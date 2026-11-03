# Audio

The app makes music, speech and sound on your own computer. There are three kinds, each with one model:

| Kind | Makes | Model | Alias | Download | Longest take |
|---|---|---|---|---|---|
| **Music** | Songs and instrumentals | YuE2 3B | `yue2` | about 8 GB | 6:00 |
| **Speech** | Text read aloud, in a voice the model picks or one that follows a recording | Higgs TTS 3 4B | `higgs3` | about 10 GB | about 10 minutes (10,000 characters) |
| **Sound** | Ambience and sound effects | Stable Audio 3 Medium | `sa3` | about 10 GB | 380 s |

In the [Web UI](../web-ui/workspace.md#the-audio-tab) they are the **Audio** tab; in the [CLI](../cli/audio.md) they are `ziv audio`. Every take is saved as a [FLAC file](#takes-and-files) that you can play in the [Gallery](../web-ui/gallery.md#audio), reuse, or give to a video as its [audio track](video.md#audio-track).

The models run on Windows and Linux with an NVIDIA GPU, a 10 GB card being enough, and on macOS with Apple silicon. See [Memory](memory.md#audio) for what each needs.

Each model downloads the first time you use it. The Web UI asks first, with the size and the licence, and the CLI prints the licence and goes on. None of the three is free for every use: each one's terms are named at download, on the Models page and in [Licences](models.md#licences).

## Music

Music has two boxes: a **Style** and, optionally, **Lyrics**.

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

There is no length setting. A song is as long as its lyrics make it, and an instrumental follows its section list: add tags for a longer piece and remove them for a shorter one.

The model stops at **6:00**. A take that reaches it fades out over 2 seconds and is marked **Cut at 6:00** in the Web UI, and its settings record that it was cut. Shorten the lyrics to hear the ending.

To keep a voice out of an instrumental, write the section tags yourself and use `[Instrumental]` and `[Solo]` rather than `[verse]` and `[chorus]`: the model reads a verse or a chorus as a place to sing, so those sections bring a voice in much more often. The default list is built this way. A take can still occasionally carry a voice or wordless singing; generate again, or try another seed.

## Speech

**Text** is what is read. It can hold up to 10,000 characters, about 10 minutes of speech; longer text is refused, so split it across takes.

### Long text

The model reads about 600 characters faithfully in one pass, so the app splits longer text at sentence ends into parts of up to about 500 characters, reads them in order and joins them into one take. A sentence longer than a part is cut after a comma, a colon or a similar mark, and only then at a space. To make the join even, long silences at the edge of each part are trimmed, the parts' levels are brought close together, and a 300 ms pause separates them. The take records how many parts it was read in.

### Expressive tags

Higgs TTS 3 follows tags in the text that steer how a sentence is spoken. Every tag is written `<|category:value|>`. These are all the tags it knows; any other text in that shape may be read aloud as written or spoil the speech:

| Category | Values | Where it goes |
|---|---|---|
| `emotion` | `elation`, `amusement`, `enthusiasm`, `determination`, `pride`, `contentment`, `affection`, `relief`, `contemplation`, `confusion`, `surprise`, `awe`, `longing`, `arousal`, `anger`, `fear`, `disgust`, `bitterness`, `sadness`, `shame`, `helplessness` | At the start of a sentence |
| `style` | `singing`, `shouting`, `whispering` | At the start of a sentence |
| `prosody` | `speed_very_slow`, `speed_slow`, `speed_fast`, `speed_very_fast`, `pitch_low`, `pitch_high`, `expressive_high`, `expressive_low` | At the start of a sentence |
| `prosody` | `pause`, `long_pause` (about 0.4 to 0.7 s and 0.7 to 1.5 s) | Between words, where the pause should fall |
| `sfx` | `cough`, `laughter`, `crying`, `screaming`, `burping`, `humming`, `sigh`, `sniff`, `sneeze` | Where the sound should occur |

A sound effect is followed at once, with no space, by the matching sound as written out. Tags can be stacked:

```text
<|emotion:elation|>Welcome aboard, we are absolutely thrilled to have you here!
<|style:whispering|>Come closer, I have a little secret to share.
<|sfx:laughter|>Haha, so glad you could make it!
Hello there <|prosody:pause|> and welcome to the show.
<|emotion:elation|><|sfx:laughter|>Haha, welcome, welcome, we're so happy you're here!
```

A tag is never cut when long text is split into parts. The Web UI has no tag picker; type the tags into the text.

### Voices

| Voice | The take is read in |
|---|---|
| **Any** | A voice the model picks. It changes from take to take |
| **From a recording** | A voice that follows a recording of 3 to 30 seconds |

With **Any**, the first part sets the voice and every later part follows it, so a long text keeps one voice within a take. To keep a voice across separate takes, make takes until one fits and use it as the voice: in the Gallery, **Use as** → **Voice**.

**From a recording** needs three things:

- The **recording**, 3 to 30 seconds of speech.
- **What it says**: the words spoken in the recording, as the model needs them to follow the voice.
- A **consent tick**: *This is my voice, or I have the speaker's permission to clone it.* Higgs TTS 3's licence forbids cloning a voice without the speaker's consent.

The first part follows the recording, and each later part follows the first, so the voice stays the same through a long text.

!!! tip "Clone from a clean recording"
    Use a clean, close-miked recording without background noise. A phone, radio or old recording makes the cloned voice hiss, and more so in long text that is split into parts. It also sounds less like the speaker.

**The consent is never saved.** It is not in the file, the Gallery or the browser. A new recording, **Reset**, **Reuse settings**, **Use as** and a reload all clear the tick, and it is asked again. The CLI asks for it as `--voice-consent`. A take made with **Any** is the model's own voice, so using it as a voice needs no tick; a take that was itself cloned from a recording still does.

## Sound

**Prompt** describes what you hear, such as `rain on a tin roof, distant thunder`, and **Length (s)** is how long it lasts: from 1 second to 380, 10 by default. Sound is stereo.

The model can come out louder than full scale. A take that peaks above 0 dBFS is scaled down to −1 dBFS when it is saved, so it does not clip. The first Sound take after the model loads takes a few seconds longer than the ones after it.

## Takes and files

Every take is a FLAC file, 16-bit, at the model's own sample rate and channel count: music 48 kHz stereo, speech 24 kHz mono and sound 44.1 kHz stereo. FLAC is lossless and plays in every current browser. Files are named `music_<date>_<time>.flac`, `speech_…` or `sound_…`, and a name that is taken gets `_2`, `_3`….

The settings that made the take are stored inside the file: the model, the style, text or prompt, the lyrics, the length, the seed, whether it was cut, how many parts a speech take was read in and, for a cloned voice, the recording's file and what it says. The speaker's consent is not among them. The Gallery reads the settings back, which is how **Reuse settings** works for audio.

A take can be used as a video's audio track: **Use as** → **Audio track** in the Gallery or viewer opens **Video** with the file as the clip's [audio track](video.md#audio-track), from its start. A speech take of 30 seconds or less can also be used as the **Voice** of the next speech take.

One batch (**Batch size** in the Web UI, `--runs` in the CLI) makes several takes with a single model load, each with its own random seed unless you set one.

## Next

[The Audio tab](../web-ui/workspace.md#the-audio-tab){ .md-button .md-button--primary } [Generate audio in the CLI](../cli/audio.md){ .md-button }
