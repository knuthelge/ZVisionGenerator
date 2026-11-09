# Generate audio

`ziv-audio` makes songs and instrumentals, speech and sound. Every option is in the [command reference](reference.md#ziv-audio); this page shows the common tasks. `--enhance` rewrites a sound's description or a song's style; Describe it and the lyrics and script helpers are Web UI only. What each kind does, the voice rules and the limits are in [Audio](../concepts/audio.md).

The model you pick with `-m` decides what is made: `yue2` music, `qwen3_tts` speech, `sa3` sound or music. `--kind music`, `--kind speech` or `--kind sound` names the kind; a model that makes one kind needs none (`sa3` makes two, and is a sound unless you say `--kind music`), and a kind the model doesn't make is refused with the kinds it does.

## A song

```bash
ziv-audio -m yue2 --prompt "upbeat synth-pop, female vocal" --lyrics-file song.txt
```

`--prompt` is the style. The lyrics come from `--lyrics` or a file, and are written in sections:

```text
[verse]
Streetlights hum along the avenue
[chorus]
Take me home tonight
```

The song is as long as its lyrics make it, up to six minutes.

### Tempo and key

```bash
ziv-audio -m yue2 --prompt "upbeat synth-pop, female vocal" --lyrics-file song.txt --tempo 96 --key F#m
```

`--tempo` is in beats per minute, 40 to 220; `--key` is one of the 24 keys, with an `m` for minor (`ziv-audio --help` lists them). Leave either out and the model chooses. See [Tempo and Key](../concepts/audio.md#tempo-and-key).

## An instrumental

Leave the lyrics out, or give only section tags:

```bash
ziv-audio -m yue2 --prompt "calm piano, slow"
ziv-audio -m yue2 --prompt "calm piano, slow" --lyrics "[Intro] [Instrumental] [Solo] [Outro]"
```

With no lyrics the instrumental has eight sections. Each tag is a section, so adding or removing tags makes the piece longer or shorter. See [Music](../concepts/audio.md#music).

Stable Audio makes an instrumental of a length you choose, 30 seconds when you give none:

```bash
ziv-audio -m sa3 --kind music --prompt "warm lo-fi beat, soft keys" --tempo 80 --key Dm --seconds 90
```

See [Stable Audio under Music](../concepts/audio.md#stable-audio-under-music).

## Speech

```bash
ziv-audio -m qwen3_tts --prompt "Welcome back. It's good to see you again."
```

Without a voice option, the model reads in its default voice, Ryan. The text can be up to 10,000 characters; see [Speech](../concepts/audio.md#speech).

### A default voice

```bash
ziv-audio -m qwen3_tts --prompt "Welcome back." --speaker aiden
```

`--speaker` takes the id of one of the model's nine default voices; `ziv-audio --help` lists them, and [Voices](../concepts/audio.md#voices) names them with their languages.

### Delivery

```bash
ziv-audio -m qwen3_tts --prompt "I thought we were friends. [sad] You never even called." \
  --pace slow --expressiveness high
ziv-audio -m qwen3_tts --prompt "Don't tell anyone." --manner whisper --emotion afraid
```

`--emotion`, `--manner`, `--pace`, `--pitch` and `--expressiveness` set how a default voice says the text, and marks such as `[sad]` in the text change it from there on. `--vibe` starts all five from a named style (`calm-narrator`, `storyteller`, `excited-host`, `announcer`, `sad-confession` or `whispered-secret`), and the options then change one. They are refused with `--voice`: a saved voice or a recording reads as it was made. See [Delivery](../concepts/audio.md#delivery).

### A voice from a recording

```bash
ziv-audio -m qwen3_tts --prompt "Welcome back." \
  --voice me.wav --voice-text "Hello, this is me." --voice-consent
```

The recording is 3 to 30 seconds long, and `--voice-text` is what it says. `--voice-consent` confirms the speaker allows their voice to be cloned; without it the command stops. The first time a take follows a recording or a saved voice, the cloning files (about 4.5 GB) download; the usual download prompt lists them first. See [Follow a recording](../concepts/audio.md#follow-a-recording).

A speech take made in a default voice can be the voice of another take without `--voice-consent`. Its text is what `--voice-text` says:

```bash
ziv-audio -m qwen3_tts --prompt "And then it rained." --voice speech_2026-11-01_10-00-00.flac --voice-text "Welcome back."
```

### A saved voice

```bash
ziv-audio -m qwen3_tts --prompt "And then it rained." --voice Morgan
```

`--voice` also takes the name of a voice saved in the Web UI, in any case, or its file slug. A file by that name in the current folder is used as a recording first. A saved voice brings its own transcript and consent, so `--voice-text` and `--voice-consent` are refused with it.

The CLI cannot save or design voices; see [Your voices](../concepts/audio.md#your-voices).

## Sound

```bash
ziv-audio -m sa3 --prompt "rain on a tin roof, distant thunder" --seconds 30
```

`--seconds` is the length, 1 second up to the model's limit. Left out, it is 12 seconds, or the length of the sound's type:

```bash
ziv-audio -m sa3 --prompt "a door slamming" --sound-type effect      # 5 s
ziv-audio -m sa3 --prompt "a single drum hit" --sound-type one-shot  # 2 s
```

`--sound-type` is `auto`, `ambience`, `effect` or `one-shot`. See [Sound](../concepts/audio.md#sound).

## Enhance the prompt

```bash
ziv-audio -m sa3 --prompt "rain on a tin roof" --enhance space=forest,movement=receding
ziv-audio -m yue2 --prompt "sad folk" --lyrics-file song.txt --enhance genre=folk,instruments=guitar+violin,singer=female
```

`--enhance` rewrites a sound's description or a song's style into the shape the model was trained on, and is written the same way as for [images](enhance.md#the-spec). Speech is read as written. See [Audio](../concepts/prompt-enhancer.md#audio).

## Several takes

```bash
ziv-audio -m sa3 --prompt "a busy market" --seconds 20 --runs 4
ziv-audio -m yue2 --prompt "lo-fi hip hop" --seed 42
```

`--runs` makes several takes with one model load, each with its own random seed unless you set `--seed`. While it runs, press ++n++ to skip the take, ++r++ to repeat it with a new seed, ++p++ to pause or ++q++ to quit.

## Files

Takes are saved as FLAC in the current folder, or the one you give with `-o`, with their settings inside. See [Takes and files](../concepts/audio.md#takes-and-files).

## Copying from the Web UI

The terminal button in the Web UI's *Settings* header copies the same command for the settings on screen.
