# Make audio

`ziv-audio` makes music, speech and sound. The model you pick with `-m` decides what is made: `yue2` and `sa3` (with `--kind music`) make songs and instrumentals, `qwen3_tts` reads speech, and `sa3` makes sound. It needs [ffmpeg](reference.md#ffmpeg). The guides [Make music](../guides/music.md) and [Make speech and sound](../guides/speech-and-sound.md) say what each setting does and why.

Takes are saved as FLAC, named `music_`, `speech_` or `sound_` plus the date and time, in the current folder or the one you give with `-o`, with their settings inside. `-r` makes several takes with one model load, each with its own random seed unless you set `--seed`. While a run goes, press ++n++ to skip the take, ++r++ to repeat it with a new seed, ++p++ to pause or ++q++ to quit.

`ziv-audio` takes no prompt file; to make many takes from a list, use a [shell loop](#speech-from-a-text-file).

## Music { #music }

### Make a song with lyrics

```bash
ziv-audio -m yue2 --prompt "upbeat synth-pop, female vocal" --lyrics-file song.txt --tempo 96 --key F#m
```

```text title="song.txt"
[verse]
Streetlights hum along the avenue
[chorus]
Take me home tonight
```

`--prompt` is the style, and the lyrics come from the file, or from `--lyrics "..."` in the command. You get a song; its length follows the lyrics. `--tempo` is beats per minute (40 to 220) and `--key` is `C`, `F#`, `Dm`, `F#m`… Leave either out and the model chooses. See [Write the style and lyrics](../guides/music.md#music).

### Make an instrumental

```bash
ziv-audio -m yue2 --prompt "calm piano, slow" --lyrics "[Intro] [Instrumental] [Solo] [Outro]"
```

Lyrics with only `[section]` tags, or none, make an instrumental, and the tags set its length. The command says whether it is making a song or an instrumental before it starts.

### Make music with Stable Audio

```bash
ziv-audio -m sa3 --kind music --prompt "warm lo-fi beat, soft keys" --tempo 80 --key Dm --seconds 90
```

Stable Audio takes no lyrics. `--seconds` sets the length, and left out it is 30 seconds. See [Make music with Stable Audio](../guides/music.md#stable-audio-under-music).

### Polish the style and make four takes

```bash
ziv-audio -m yue2 --prompt "sad folk" --lyrics-file song.txt --enhance genre=folk,instruments=guitar+violin,singer=female -r 4
```

`--enhance` rewrites the style into a fuller description once, before the model loads, and all four takes use it. Put `+` between several values. The keys and values are in [the audio enhance spec](reference.md#the-audio-spec). `--enhance-note "..."` is your own direction in words. See [Polish the style with the enhancer](../guides/music.md#enhance-the-prompt).

## Speech { #speech }

Without a voice option, the model reads in its default voice, `ryan`.

### Speak in a built-in voice

```bash
ziv-audio -m qwen3_tts --prompt "I thought we were friends. [sad] You never even called." --speaker aiden --pace slow --expressiveness high
```

`--speaker` picks the voice: `ryan`, `aiden`, `vivian`, `serena`, `uncle_fu`, `dylan`, `eric`, `ono_anna` or `sohee`. The delivery options are `--emotion`, `--manner` (`natural`, `whisper`, `shout`), `--pace`, `--pitch` and `--expressiveness`. A mark such as `[sad]` in the text changes the delivery from there on; see [Delivery marks](../guides/speech-and-sound.md#marks).

To start all five delivery options from a named style, use `--vibe`, then change one:

```bash
ziv-audio -m qwen3_tts --prompt "Good evening, and welcome to the show." --vibe announcer --pitch low
```

The styles are `calm-narrator`, `storyteller`, `excited-host`, `announcer`, `sad-confession` and `whispered-secret`. The delivery options work only with a built-in voice: they are refused with `--voice`, because a saved voice or a recording reads as it was made.

### Speak in a saved voice

```bash
ziv-audio -m qwen3_tts --prompt "And then it rained." --voice Morgan
```

`--voice` takes the name of a voice you saved in the Web UI, in any case, or its file slug. A saved voice brings its own transcript and consent, so `--voice-text` and `--voice-consent` are refused with it. A file by that name in the current folder is used as a recording first.

### Speak in the voice of a recording

```bash
ziv-audio -m qwen3_tts --prompt "Welcome back." --voice me.wav --voice-text "Hello, this is me." --voice-consent
```

The recording is 3 to 30 seconds long, and `--voice-text` is exactly what it says. `--voice-consent` confirms that the speaker allows their voice to be cloned; without it the command stops. A speech take made in a built-in voice can be the voice of another take without `--voice-consent`: give its file as `--voice` and its text as `--voice-text`.

The first time a take follows a recording or a saved voice, the cloning files (about 4.5 GB) download after the model's own. The command asks nothing before it downloads. A progress line shows the download, ++q++ stops it, and the licence terms it prints first name only the model's own files. See [Clone a voice from a recording](../guides/speech-and-sound.md#clone-a-voice-from-a-recording).

### Read several lines from a text file { #speech-from-a-text-file }

Put one line of speech on each line of a file:

```text title="lines.txt"
Welcome back. It's good to see you again.
The doors open at nine.
Please keep your tickets with you.
```

A loop makes one take per line, in a folder `lines`:

=== "macOS"

    ```bash
    while IFS= read -r -u 3 line; do
      [ -n "$line" ] || continue
      ziv-audio -m qwen3_tts --prompt "$line" --speaker aiden -o lines || break
    done 3< lines.txt
    ```

=== "Windows"

    ```powershell
    foreach ($line in Get-Content lines.txt) {
      if (-not $line.Trim()) { continue }
      ziv-audio -m qwen3_tts --prompt $line --speaker aiden -o lines
      if ($LASTEXITCODE -ne 0) { break }
    }
    ```

=== "Linux"

    ```bash
    while IFS= read -r -u 3 line; do
      [ -n "$line" ] || continue
      ziv-audio -m qwen3_tts --prompt "$line" --speaker aiden -o lines || break
    done 3< lines.txt
    ```

You get one `speech_<date>_<time>.flac` per line. Each line loads the model again, so it is slower than `-r`, which makes several takes of one text. The loop reads the file through a separate input (`3<`), so the command keeps the terminal and its keys. Press ++ctrl+c++ to stop the loop; a take that fails stops it too, because `ziv-audio` then exits with `2`. See [Run commands in a loop and in scripts](index.md#loops-and-scripts).

## Sound { #sound }

### Make a sound effect

```bash
ziv-audio -m sa3 --prompt "a heavy wooden door slams shut in a stone hall" --sound-type effect --seconds 4
```

`--sound-type` is `auto`, `ambience`, `effect` or `one-shot`, and sets the length when you give no `--seconds`: 12 s for `auto` and `ambience`, 5 s for `effect` and 2 s for `one-shot`. A length you give is used as it is, from 1 second up to 380.

### Make a sound of an exact length

```bash
ziv-audio -m sa3 --prompt "wind in the pines, a woodpecker tapping" --seconds 30 --enhance space=forest,distance=far
```

A sound takes `space`, `distance`, `over_time` and `movement` in `--enhance`, with the values in [the audio enhance spec](reference.md#the-audio-spec). Speech refuses `--seconds` and `--enhance`: its length follows the text, and it is read as written. See [Polish the prompt with the enhancer](../guides/speech-and-sound.md#polish-the-prompt-with-the-enhancer).

## Flags

Every flag, with its default and limits, is in the [command reference](reference.md#ziv-audio).

| Flag | Does | In the Web UI |
|---|---|---|
| `-m` | The model: `yue2`, `qwen3_tts` or `sa3` | **Model** |
| `--kind` | `music`, `speech` or `sound`, for a model that makes several | **Kind** |
| `--prompt` | The style of a song, the text to read, or the description of a sound | **Style**, **Text**, **Prompt** |
| `--lyrics`, `--lyrics-file` | Lyrics of a song (`yue2` only) | **Lyrics** |
| `--tempo`, `--key` | Tempo and key of a song | **Tempo**, **Key** |
| `--seconds` | Length in seconds (music from `sa3`, and sound) | **Length (s)** |
| `--sound-type` | The type of a sound | **Type** |
| `--speaker` | A built-in voice | **Voice** |
| `--voice` | A saved voice, or a recording | **Voice** |
| `--voice-text`, `--voice-consent` | What a recording says, and the speaker's permission | **What it says**, the consent tick |
| `--vibe` | A named delivery style | **Vibe** |
| `--emotion`, `--manner`, `--pace`, `--pitch`, `--expressiveness` | Delivery of a built-in voice | **Emotion**, **Manner**, **Pace**, **Pitch**, **Expressiveness** |
| `--enhance`, `--enhance-note`, `--no-enhance` | Rewrite a song's style or a sound's description first | **Enhance**, **Direction** |
| `--enhance-model` | The enhancer model | **Prompt enhancer model** (Config page) |
| `-r` | How many takes to make | **Batch size** |
| `--seed` | The seed, used for every take | **Seed** |
| `-o` | The output folder | **Output folder** (Config page) |
