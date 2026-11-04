# Generate audio

`ziv-audio` makes songs and instrumentals, speech and sound. Every option is in the [command reference](reference.md#ziv-audio); this page shows the common tasks. What each kind does, the voice rules and the limits are in [Audio](../concepts/audio.md).

The model you pick with `-m` decides what is made: `yue2` music, `higgs3` speech, `sa3` sound.

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

## An instrumental

Leave the lyrics out, or give only section tags:

```bash
ziv-audio -m yue2 --prompt "calm piano, slow"
ziv-audio -m yue2 --prompt "calm piano, slow" --lyrics "[Intro] [Instrumental] [Solo] [Outro]"
```

With no lyrics the instrumental has eight sections. Each tag is a section, so adding or removing tags makes the piece longer or shorter. See [Music](../concepts/audio.md#music).

## Speech

```bash
ziv-audio -m higgs3 --prompt "Welcome back. It's good to see you again."
```

Without a voice, the model reads in a voice of its own. The text can be up to 10,000 characters; see [Speech](../concepts/audio.md#speech).

### A voice from a recording

```bash
ziv-audio -m higgs3 --prompt "Welcome back." \
  --voice me.wav --voice-text "Hello, this is me." --voice-consent
```

The recording is 3 to 30 seconds long, and `--voice-text` is what it says. `--voice-consent` confirms the speaker allows their voice to be cloned; without it the command stops. See [Voices](../concepts/audio.md#voices).

A speech take made in the model's own voice can be the voice of another take without `--voice-consent`. Its text is what `--voice-text` says:

```bash
ziv-audio -m higgs3 --prompt "And then it rained." --voice speech_2026-11-01_10-00-00.flac --voice-text "Welcome back."
```

## Sound

```bash
ziv-audio -m sa3 --prompt "rain on a tin roof, distant thunder" --seconds 30
```

`--seconds` is the length, 1 second up to the model's limit; the default is 10.

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
