# Your first image

This walks you from a fresh install to a finished image in the Web UI. It takes a few minutes plus the model download.

## 1. Start the Web UI

```bash
ziv ui
```

Your browser opens on the Workspace.

## 2. Download a model

The first time, a welcome dialog lists the image models with the size of each download. One is already picked for your machine:

- **FLUX.2 Klein 9B** when it has 32 GB of memory or more (unified memory on a Mac, GPU memory on a graphics card).
- **FLUX.2 Klein 4B** otherwise.

![The welcome dialog with FLUX.2 Klein 9B preselected](../assets/screenshots/placeholder.svg "first-run-dialog")

Press **Download**. The model becomes your default image model, and its progress shows in the Workspace model picker. You can carry on while it downloads.

!!! tip "Not now?"
    **Skip** closes the dialog for good. A model you never downloaded downloads by itself the first time a job needs it, and the job card shows the progress. You can also download any model later from the [Models](models.md) page.

## 3. Write a prompt

Check that **Text to Image** is selected in the top bar. Type what you want to see in the **Prompt** box under *Compose*:

```text
A red fox sitting in fresh snow, early morning light, close-up
```

Plain descriptions work best: the subject, where it is, the light, and how it's framed. See [Prompts](../concepts/prompts.md) for random choices and other tricks.

## 4. Generate

Press **Generate**, or ++ctrl+enter++ (++cmd+enter++ on a Mac).

![A running job: the live preview and the job card with its steps](../assets/screenshots/placeholder.svg "job-running")

The job card shows each phase: loading the model, the steps, and saving. While it works, the preview shows the image taking shape at 25, 50 and 75% of the steps. The first job is the slowest, because the model loads from disk; later jobs with the same model start faster.

## 5. Look at the result

The finished image fills the preview and appears first in the Gallery strip. Click it to open the viewer, where you can:

- see the settings it was made with under **Details** (++i++),
- **Reuse settings** (++r++) to load them back into the Workspace,
- **Upscale** it, **Download** it, or delete it.

Images are saved to the output folder, `~/.ziv/outputs` unless you [changed it](config.md). Every image remembers the settings it was made with.

## Make it better

- **Let the enhancer write the details.** Click ✨ **Enhance** in the prompt box and **Enhance prompt**. A local model rewrites your prompt with more visual detail, in the **Enhanced** tab. See [Prompt enhancer](../concepts/prompt-enhancer.md).
- **Try another seed.** Every job uses a new random seed. To get a variation of a result you like, **Reuse settings** and generate again with ++ctrl+shift+enter++ (a new seed).
- **Change the shape.** **Ratio** and **Resolution** in *Settings* set the size; `2:3` portrait is the default. See [Images](../concepts/images.md#sizes).
- **Queue more.** While a job runs, change the prompt or settings and press Generate again. The new job waits its turn. See [The queue](workspace.md#the-queue).

## Next

- [Your first video](first-video.md)
- [Workspace](workspace.md): every part of the screen.
