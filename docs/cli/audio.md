# Generate audio

`ziv-audio` makes songs and instrumentals, speech and sound. Every option is in the [command reference](reference.md#ziv-audio); this page shows the common tasks. What each kind does, the voice rules and the limits are in [Audio](../concepts/audio.md).

The model you pick with `-m` decides what is made. Which models are available depends on the `audio_model_presets` section of your [config](../concepts/configuration.md#audio_model_presets); if it has none, `ziv-audio` says so.

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

The song is as long as its lyrics make it, up to six minutes. A take that reaches the limit ends in a fade-out, and its settings say it was cut.

## An instrumental

Leave the lyrics out, or give only section tags:

```bash
ziv-audio -m yue2 --prompt "calm piano, slow"
ziv-audio -m yue2 --prompt "calm piano, slow" --lyrics "[Intro] [Instrumental] [Solo] [Outro]"
```

With no lyrics, the instrumental has eight sections: `[Intro] [Instrumental] [Solo] [Instrumental] [Solo] [Bridge] [Instrumental] [Outro]`. Each tag is a section, so adding or removing tags makes the piece longer or shorter. Use `[Instrumental]` and `[Solo]` to keep vocals out; see [Music](../concepts/audio.md#music).

## Speech

```bash
ziv-audio -m higgs3 --prompt "Welcome back. It's good to see you again."
```

Without a voice, the model reads in a voice of its own. Long text is read in parts of about 500 characters and joined into one take; the ceiling is 10,000 characters, about ten minutes of speech. Longer text is refused, so split it across takes.

### A voice from a recording

```bash
ziv-audio -m higgs3 --prompt "Welcome back." \
  --voice me.wav --voice-text "Hello, this is me." --voice-consent
```

The recording is 3 to 30 seconds long, and `--voice-text` is what it says. Use a clean, close-miked recording: see [Clone from a clean recording](../concepts/audio.md#voices). `--voice-consent` confirms the speaker allows their voice to be cloned; without it the command stops with `Add --voice-consent: cloning a voice needs the speaker's permission.`

A speech take made without a voice recording can be the voice of another take without `--voice-consent`, as long as it is 30 seconds or less like any voice. Its text is what `--voice-text` says:

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

## Progress

While a take is made, the command shows what the model is doing and how far along it is, for example `Synthesis 12/64 · about 2 min left in the batch` for a bounded step or `Song tokens 1,840` for a step that ends on its own, and `Finishing...` while the sound is decoded. On a terminal this is one line that updates in place; in a pipe or a log file one line is printed each time the step changes. The time left covers all the takes still to make, this one included, and appears once the first take of a run has finished.

## The files

Takes are saved as FLAC, 16-bit, at the model's own sample rate and channel count, in the current folder or the one you give with `-o`. The settings that made a take (model, style or text, lyrics, length, seed and, for a cloned voice, the recording and its text) are stored inside the file, never the speaker's consent. The Web UI reads them back to reuse a take.

## Copying from the Web UI

The terminal button in the Web UI's *Settings* header copies the same command for the settings on screen. It adds `--voice-consent` only for a voice cloned from a recording.
