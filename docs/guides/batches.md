# Run batches from prompt files

A prompt file holds many prompts, grouped in named sets. Run it and every active prompt is made, one after another, from the Web UI or from a terminal.

## Write a prompt file

A prompt file is YAML: each top-level key is a set, holding a list of prompts.

```yaml title="prompts.yaml"
fox:
  - prompt: A red fox sitting in fresh snow, early morning light
  - prompt: A red fox {leaping|sleeping} in a {birch|pine} forest

lighthouse:
  - prompt: A lighthouse in a storm, waves crashing
```

There are two ways to write one:

- **On the Prompts page** (++g++ ++p++): **New file…** creates one, and you edit it without typing YAML. See [Prompts page](../web-ui/prompts.md).
- **In a text editor**, as above. Any name ending in `.yaml` works.

To keep a prompt you just wrote in the Workspace, click **Save to prompt file…** (the file icon in the prompt box's tools row). It appends the prompt to the set named `prompts`. See [Save a Workspace prompt](../web-ui/prompts.md#save-a-workspace-prompt).

The set's name starts the name of each file it makes: `fox_2026-10-04_14-03-22.png`. The `{a|b}` choices, snippets, negative prompts and video scenes are described in [Prompt syntax](../concepts/prompt-syntax.md).

### Turn prompts on and off { #turn-prompts-on-and-off }

`active: false` parks a prompt without deleting it. A run skips it.

```yaml
fox:
  - prompt: A red fox sitting in fresh snow
  - active: false
    prompt: A red fox at the beach   # skipped
```

On the Prompts page each prompt has an **active** switch.

## Pick a prompt set and run it

In the Web UI:

1. Choose **Image** or **Video** in the top bar, and set the model and settings you want for every prompt.
2. Set **Prompt source** to **Prompt file** and give the path of the file.
3. Click **Prompts…** to choose which prompts run, by set or one by one.
4. Press **Generate**.

**Edit** next to **Prompts…** opens the file on the [Prompts page](../web-ui/prompts.md), which can also queue one prompt or select the file for the Workspace.

From a terminal, give the file with `-p`:

```bash
ziv-image -m klein9b -p prompts.yaml
ziv-video -m ltx-8 -p clips.yaml
```

Every active prompt of every set runs once. `-p` defaults to `prompts.yaml` in the current folder, and `--prompt` overrides it. The command line has no set picker: to run one set, put it in a file of its own, or mark the others `active: false`.

Video takes the same files. A video prompt can also hold a `scene:`; see [Scenes for video](../concepts/prompt-syntax.md#scenes-for-video).

## Set runs and batch size

**Batch size** (`-r` or `--runs`) runs the whole selection that many times. Each run picks the `{a|b}` choices again and uses a new random seed, so a batch size of 3 gives three different images per prompt. Set a **Seed** to give every run the same one.

Audio batches work the same way: **Batch size** or `--runs` makes several takes with one model load.

In the Web UI you can keep working while a batch runs. Change the settings and press **Add to queue**; queued jobs run one after another. See [The queue](../web-ui/workspace.md#the-queue).

## Enhance a whole batch

The [prompt enhancer](../concepts/prompt-enhancer.md) can rewrite every prompt of a file.

- **In the Web UI**, turn on **Enhance each image when generating** in the Enhance panel. Every prompt, with its random choices already picked, is rewritten as its image starts.
- **In a file**, add `enhance:` to an entry to enhance it on every run. On the Prompts page each prompt has an ✨ **Enhance** menu for it. See [In prompt files](../concepts/prompt-enhancer.md#in-prompt-files).
- **On the command line**, `--enhance` rewrites every prompt:

```bash
ziv-image -m klein9b -p prompts.yaml --enhance style=photo,length=longer
```

All prompts are rewritten before the model loads, so the enhancer adds no memory during generation. `--enhance` overrides the `enhance:` entries of the file, and `--no-enhance` turns all of it off. The spec is in the [command reference](../cli/reference.md#enhance-spec).

## Pause, skip or stop a run { #keys-during-a-run }

A batch can take a while. In the Web UI, the job card steers it; see [While a job runs](../web-ui/workspace.md#while-a-job-runs).

In a terminal, `ziv-image`, `ziv-video` and `ziv-audio` read single keys: ++n++ skips to the next prompt, ++q++ ends the batch, ++p++ pauses after the current one (any key resumes) and ++r++ repeats the current one with a new seed. They work only in an interactive terminal, not when the output is piped or the command runs in the background.

With `--review`, each small render waits for ++y++ to upscale it, ++n++ to discard it or ++q++ to quit; see [Review before upscaling](../concepts/upscaling.md#review-before-upscaling).

## If a prompt fails { #if-a-prompt-fails }

A failed prompt doesn't stop the batch: the run records the failure and goes on to the next prompt. If every prompt fails, the job ends as failed, and the [Job log](../web-ui/job-log.md) keeps the reason. To see what a message means, look it up in [Troubleshooting](../reference/troubleshooting.md#prompts-and-prompt-files).

## From the command line

A prompt file is the command line's way of running many prompts:

```bash
ziv-image -m klein9b -p prompts.yaml -r 3
```

Picking prompts, runs, video scenes and file names are in [Batches and prompt files](../cli/batches.md).
