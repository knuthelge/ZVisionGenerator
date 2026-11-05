# Audio

The app makes music, speech and sound on your own computer. There are three kinds, each with one model:

| Kind | Makes | Model | Alias | Download | Longest take |
|---|---|---|---|---|---|
| **Music** | Songs and instrumentals | YuE2 3B | `yue2` | about 8 GB | 6:00 |
| **Speech** | Text read aloud, in a voice the model picks or one that follows a recording | Higgs TTS 3 4B | `higgs3` | about 10 GB | about 10 minutes (10,000 characters) |
| **Sound** | Ambience and sound effects | Stable Audio 3 Medium | `sa3` | about 10 GB | 380 s |

In the [Web UI](../web-ui/workspace.md#the-audio-tab) they are the **Audio** tab; in the [CLI](../cli/audio.md) they are `ziv audio`. Every take is saved as a [FLAC file](#takes-and-files) that you can play in the [Gallery](../web-ui/gallery.md#audio), reuse, or give to a video as its [audio track](video.md#audio-track).

The models run on Windows and Linux with an NVIDIA GPU (10 GB is enough) and on macOS with Apple silicon (32 GB of unified memory is enough). See [Memory](memory.md#audio). Each model downloads the first time you use it. None of the three is free for every use: read the terms in [Licences](models.md#licences) before you use the output.

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

The model stops at **6:00**. A take that reaches it fades out and is marked **Cut at 6:00**; shorten the lyrics to hear the ending.

To keep a voice out of an instrumental, use `[Instrumental]` and `[Solo]` tags rather than `[verse]` and `[chorus]`, which the model reads as places to sing. A take can still occasionally carry a voice; generate again, or try another seed.

## Speech

**Text** is what is read. It can hold up to 10,000 characters, about 10 minutes of speech; longer text is refused, so split it across takes. A long text is read in parts and joined into one take.

### Expressive tags

Higgs TTS 3 follows tags in the text that steer how a sentence is spoken. Type them into the text as `<|category:value|>`. These are all the tags it knows; any other text in that shape may be read aloud as written:

| Category | Values | Where it goes |
|---|---|---|
| `emotion` | `elation`, `amusement`, `enthusiasm`, `determination`, `pride`, `contentment`, `affection`, `relief`, `contemplation`, `confusion`, `surprise`, `awe`, `longing`, `arousal`, `anger`, `fear`, `disgust`, `bitterness`, `sadness`, `shame`, `helplessness` | At the start of a sentence |
| `style` | `singing`, `shouting`, `whispering` | At the start of a sentence |
| `prosody` | `speed_very_slow`, `speed_slow`, `speed_fast`, `speed_very_fast`, `pitch_low`, `pitch_high`, `expressive_high`, `expressive_low` | At the start of a sentence |
| `prosody` | `pause`, `long_pause` | Between words, where the pause should fall |
| `sfx` | `cough`, `laughter`, `crying`, `screaming`, `burping`, `humming`, `sigh`, `sniff`, `sneeze` | Where the sound should occur |

A sound effect is followed at once by the sound as written out. Tags can be stacked:

```text
<|emotion:elation|>Welcome aboard, we are absolutely thrilled to have you here!
<|style:whispering|>Come closer, I have a little secret to share.
<|sfx:laughter|>Haha, so glad you could make it!
Hello there <|prosody:pause|> and welcome to the show.
<|emotion:elation|><|sfx:laughter|>Haha, welcome, welcome, we're so happy you're here!
```


### Voices

| Voice | The take is read in |
|---|---|
| **Any** | A voice the model picks. It changes from take to take |
| **From a recording** | A voice that follows a recording of 3 to 30 seconds |

To keep a voice across separate takes, make takes with **Any** until one fits and use it as the voice: in the Gallery, **Use as** → **Voice**.

**From a recording** needs three things:

- The **recording**, 3 to 30 seconds of speech.
- **What it says**: the words spoken in the recording.
- A **consent tick**: *This is my voice, or I have the speaker's permission to clone it.* Higgs TTS 3's licence forbids cloning a voice without the speaker's consent.

!!! tip "Clone from a clean recording"
    Use a clean, close-miked recording without background noise. A phone, radio or old recording makes the cloned voice hiss and sound less like the speaker.

**The consent is never saved**, not in the file, the Gallery or the browser, so it is asked again whenever the voice changes or the page reloads. The CLI asks for it as `--voice-consent`. A take made with **Any** is the model's own voice, so using it as a voice needs no tick; a take that was itself cloned from a recording still does.

## Sound

**Prompt** describes what you hear, such as `rain on a tin roof, distant thunder`, and **Length (s)** is how long it lasts: from 1 second to 380, 10 by default.

## Takes and files

Every take is a lossless FLAC file named `music_<date>_<time>.flac`, `speech_…` or `sound_…`. The settings that made it are stored inside, which is how the Gallery's **Reuse settings** works for audio; the speaker's consent is not among them. For a cloned voice, the settings name the recording and what it says.

A take can be used as a video's audio track: **Use as** → **Audio track** in the Gallery or viewer opens **Video** with the file as the clip's [audio track](video.md#audio-track). A speech take of 30 seconds or less can also be used as the **Voice** of the next speech take.

One batch (**Batch size** in the Web UI, `--runs` in the CLI) makes several takes with a single model load, each with its own random seed unless you set one.

## Next

[The Audio tab](../web-ui/workspace.md#the-audio-tab){ .md-button .md-button--primary } [Generate audio in the CLI](../cli/audio.md){ .md-button }
