# Batches and prompt files

A prompt file holds many prompts, and `ziv-image` and `ziv-video` make every active one, one after another, with one model load. The guide [Run batches from prompt files](../guides/batches.md) says how the file works and what happens when a prompt fails.

## Write a prompt file

A prompt file is YAML. Each top-level key is a set, holding a list of prompts, and `{a|b}` picks one option at random every time a prompt runs:

```yaml title="prompts.yaml"
fox:
  - prompt: A red fox sitting in fresh snow, early morning light
  - prompt: A red fox {leaping|sleeping} in a {birch|pine} forest

lighthouse:
  - prompt: A lighthouse in a storm, waves crashing
  - active: false
    prompt: A lighthouse at calm sunset
```

`active: false` parks a prompt without deleting it. The full syntax, with snippets, negative prompts and the `enhance:` key, is in [Prompt syntax](../concepts/prompt-syntax.md#prompt-files).

## Run a prompt file

```bash
ziv-image -m klein9b -p prompts.yaml
```

Every active prompt runs once: two fox images and one lighthouse. Without `-p`, the command looks for `prompts.yaml` in the current folder. `--prompt` on the command line replaces the file for that run.

## Pick which prompts run

The command line has no set picker. To run one set, put it in a file of its own, or switch the others off with `active: false`:

```bash
ziv-image -m klein9b -p fox.yaml
```

To try one prompt before a whole file, give it with `--prompt`; the file is then not read.

## Run it more than once

```bash
ziv-image -m klein9b -p prompts.yaml -r 3
```

`-r 3` runs the whole file three times, in order: every prompt of run 1, then every prompt of run 2, and so on. Each run picks the `{a|b}` choices again and uses a new random seed, so the three fox-in-a-forest images differ. A fixed `--seed` is used for every run, which would make the same image each time for a prompt with no choices. All runs share one model load.

## Make a video from a prompt file

A video prompt can hold a `scene:` (who and where, the look, the camera) beside its `prompt:` (what happens):

```yaml title="clips.yaml"
snippets:
  courtyard: A knight in a grey castle courtyard, overcast, handheld camera.

castle:
  - scene: $courtyard
    prompt: She draws her sword and turns to the gate.
  - scene: $courtyard
    prompt: She sheathes the sword and walks toward the tower.
  - prompt: A raven lands on the wall.
```

```bash
ziv-video -m ltx-4 -p clips.yaml
ziv-video -m ltx-4 -p clips.yaml --scene "A knight in a grey castle courtyard, overcast."
```

Each entry makes a clip. In the second command, `--scene` fills in the entries that have no `scene:` of their own, here the raven. An image run ignores `scene:`, so one file can serve both commands. See [Scenes for video](../concepts/prompt-syntax.md#scenes-for-video).

A prompt file also works with `--extend`: every prompt continues the same clip and makes a file of its own. See [Continue a clip](video.md#extend).

## Enhance every prompt

```bash
ziv-image -m klein9b -p prompts.yaml --enhance style=photo,length=longer
ziv-image -m klein9b -p prompts.yaml --no-enhance
```

`--enhance` rewrites every prompt of the file, with its `{a|b}` choices already picked, before the model loads, and overrides the `enhance:` entries of the file. `--no-enhance` turns them all off. The values are in [the enhance spec](reference.md#enhance-spec).

## Where the files go

Each file starts with the name of its set and then the date and time, such as `fox_2026-10-04_14-03-22.png` or `castle_2026-10-04_14-03-22.mp4`. A name that is taken gets `_2`, `_3`… so nothing is overwritten. `-o folder` saves them in another folder; see [Where files go](index.md#where-files-go). Each file records the prompt it was made from, with its choices picked.

## Steer a run from the keyboard

In a terminal, ++n++ skips to the next prompt, ++q++ ends the batch, ++p++ pauses after the current one and ++r++ repeats it with a new seed. See [Pause, skip or stop a run](../guides/batches.md#keys-during-a-run). A prompt that fails is skipped with a warning and the run goes on.

## Many takes of audio

`ziv-audio` takes no prompt file. `-r` makes several takes of one prompt, and to make one take for each of many texts, loop over a text file in the shell: see [Read several lines from a text file](audio.md#speech-from-a-text-file).

## Flags

Every flag is in the [command reference](reference.md#ziv-image).

| Flag | Does | In the Web UI |
|---|---|---|
| `-p` | The prompt file; `prompts.yaml` when left out | **Prompt source** → **Prompt file** |
| `--prompt` | One prompt, instead of the file | **Prompt source** → **Inline** |
| `--scene` | `ziv-video`: a scene for entries that have none | **Default scene** |
| `-r` | How many times to run the whole file | **Batch size** |
| `--seed` | The seed, used for every run | **Seed** |
| `--enhance`, `--no-enhance` | Rewrite every prompt, or none | **Enhance each image when generating** |
| `-o` | The output folder | **Output folder** (Config page) |
