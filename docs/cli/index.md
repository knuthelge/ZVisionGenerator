# CLI

Everything you can make in the app you can make from a command, for batch runs, scripts and working over SSH. Each task has a page of recipes you can copy: what to type, and what you get. The [command reference](reference.md) lists every option. Error? See [Troubleshooting](../reference/troubleshooting.md).

The commands and the [Web UI](../web-ui/index.md) share the same models and their downloads, the same [`config.yaml`](../reference/configuration.md), the same [data folder](../reference/data-folder.md) (`~/.ziv` by default) and the same LoRAs. A file the command line makes records its settings like one from the Web UI, so you can open it in the Gallery, read its **Details** and use **Reuse settings**.

## The commands

| Command | Does | Recipes |
|---|---|---|
| `ziv-image` | Makes images | [Make images](images.md) |
| `ziv-video` | Makes videos and continues them | [Make video](video.md) |
| `ziv-audio` | Makes music, speech and sound | [Make audio](audio.md) |
| `ziv-model` | Converts checkpoints, imports LoRAs, lists models | [Models and LoRAs](models.md) |
| `ziv-ui` | Starts the [Web UI](../web-ui/index.md) | [Command reference](reference.md#ziv-ui-ziv-ui) |

To run many prompts from a file with `ziv-image` or `ziv-video`, see [Batches and prompt files](batches.md).

`ziv` runs the same commands as subcommands: `ziv image` is `ziv-image`, and likewise `ziv video`, `ziv audio`, `ziv model` and `ziv ui`. These docs spell them `ziv-image` and so on. `ziv` on its own prints the list of commands, `ziv --version` prints the version, and `--help` after any command lists its options: `ziv-image --help`.

## Your first commands

An image. `--prompt` is the prompt; without it, the command looks for a `prompts.yaml` file in the current folder and stops if there isn't one.

```bash
ziv-image -m klein4b --prompt "A red fox sitting in fresh snow, early morning light"
```

A video, with sound:

```bash
ziv-video -m ltx-4 --prompt "A cat walking through a garden, birds singing"
```

A spoken take:

```bash
ziv-audio -m qwen3_tts --prompt "Welcome back. It's good to see you again."
```

The models, LoRAs and model names you can use:

```bash
ziv-model list
```

It prints the models in `models/` in the data folder (the ones you converted, and the quantized copies `--quantize` stores beside them), your LoRAs, and the built-in model names, whether you have downloaded them or not. It does not say what is downloaded: a built-in model downloads the first time you use it.

The Web UI, kept running after you close the terminal, and stopped again later:

```bash
ziv-ui --background
ziv-ui --background-stop
```

`-m` picks the model by [alias](../reference/models.md#how-a-name-is-found). The first run of a model prints its licence and downloads it, the same download the Web UI uses. See [Licences](../reference/models.md#licences).

## Where files go { #where-files-go }

An image, video or take is saved in the folder you run the command from, as `prompt_<date>_<time>.png` for an inline prompt; a prompt file starts the name with the name of its set. `-o <folder>` saves it somewhere else, and makes the folder if it doesn't exist. The Web UI saves to its own output folder instead; see [Output files](../reference/data-folder.md#output-files).

Each file keeps the prompt and settings it was made with. To see a command's files in the Web UI's Gallery, save them in a folder of the output folder. A folder directly under `outputs` in the data folder is a [project](../concepts/projects.md#projects-on-the-command-line):

```bash
ziv-image -m klein4b --prompt "A night bus in the rain" -o "~/.ziv/outputs/Night Bus"
```

The Gallery shows it as the project **Night Bus** the next time it loads. If you moved the data folder or set another **Output folder**, use that one in place of `~/.ziv/outputs`.

## Copy a command from the Web UI

The terminal button in the Workspace's *Settings* header copies a command with the current model, prompt and settings. Paste it in a terminal to run the same job. See [Copy as a command](../web-ui/workspace.md#copy-as-a-command).

## Run commands in a loop and in scripts { #loops-and-scripts }

A command makes what you ask and exits, so a shell loop runs it again with something changed. Use `-r` for more of the same prompt and a [prompt file](batches.md) for many prompts: they make many images with one model load. Use a loop to change an option `-r` can't, such as a LoRA's weight, the guidance or the model. This makes one image at each of three weights of a LoRA named `my-style`, all from the same seed so that only the weight changes:

=== "macOS"

    ```bash
    for weight in 0.4 0.7 1.0; do
      ziv-image -m klein9b --prompt "A portrait of an old fisherman" --lora "my-style:$weight" --seed 42 -o lora-weights || break
    done
    ```

=== "Windows"

    ```powershell
    foreach ($weight in 0.4, 0.7, 1.0) {
      ziv-image -m klein9b --prompt "A portrait of an old fisherman" --lora "my-style:$weight" --seed 42 -o lora-weights
      if ($LASTEXITCODE -ne 0) { break }
    }
    ```

=== "Linux"

    ```bash
    for weight in 0.4 0.7 1.0; do
      ziv-image -m klein9b --prompt "A portrait of an old fisherman" --lora "my-style:$weight" --seed 42 -o lora-weights || break
    done
    ```

Every command loads the model again, so a loop is slower than `-r`. Press ++ctrl+c++ to stop a loop. The key ++q++ ends only the running command, which then exits with `0`, so the loop goes on with the next one.

Without a terminal, as in a script, a scheduled task or `ssh` without a terminal:

- **The keys don't work.** Skip, quit, pause and repeat (++n++, ++q++, ++p++, ++r++) are read from the terminal, so the only way to interrupt a run is to stop the process. `--review` needs a terminal and is refused without one.
- **Install ffmpeg first.** `ziv-video` and `ziv-audio` need it and ask before installing it, which a script can't answer; see [ffmpeg](reference.md#ffmpeg).
- **The exit status** is `0` when the command ends normally and non-zero when it stops: `2` for a refused option, a prompt file it can't read, a model that can't load, or a `ziv-audio` take that fails, with the reason on standard error; `1` for an error of `ziv-model` and for a missing or declined ffmpeg. An image or clip that fails partway through a run is reported as a warning and skipped, and the run still ends with `0`.
- **Use a fresh `-o` folder for each job.** Because a skipped image still exits with `0`, the way for a script to know it got what it asked for is to look in the folder: an empty one means nothing was made.

## What is Web UI only

Some things have no command:

- **Describe it**, the lyrics helper and the script helper on the Audio tab.
- **Saving and designing voices.** `ziv-audio` follows a voice you saved in the Web UI with `--voice`, or a recording.
- **Downloading and deleting models**, from the Models page. The commands download a model when you first use it.
- **The Gallery, the viewer, the Job log, projects, presets and the Prompts page.**
- **Upscaling a finished image or video.** The commands upscale while generating, with `--upscale`.
