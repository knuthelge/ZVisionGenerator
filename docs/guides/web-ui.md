# Web UI Guide

The Web UI is the local browser interface for running generations, managing assets, and adjusting everyday settings.

## Start the Web UI

Use whichever command fits how you installed Z-Vision Generator:

```bash
# Installed command
ziv ui

# Shortcut: opens the same Web UI
ziv

# Standalone launcher
ziv-ui

# From a repository checkout
uv run ziv ui --no-browser
```

By default, the app starts on `http://127.0.0.1:8080/` and opens a browser tab automatically.

- Open the printed local address if the app chooses a different port.
- Open `http://127.0.0.1:8080/` or `http://127.0.0.1:8080/app` when you keep the default port.
- Use `--no-browser` if you want to start the server without opening a tab.
- Use `--host` and `--port` when you need a different bind address or preferred port.

Example:

```bash
ziv ui --host 127.0.0.1 --port 9000 --no-browser
```

Then open `http://127.0.0.1:9000/` in your browser.

## Main Pages

The top navigation switches between the four supported pages inside the app:

| Page | Direct URL | What to use it for |
|---|---|---|
| Workspace | `/app#/workspace` | Start new jobs, switch workflows, adjust per-run settings, and monitor the current session |
| Models | `/app#/models` | Review installed models and LoRAs, convert checkpoints, and import LoRAs |
| Gallery | `/app#/gallery` | Browse saved images and videos, preview them, reuse them in the workspace, and delete items |
| Configuration | `/app#/config` | Set default models, choose the output directory, and review runtime paths |

You usually do not need to type these URLs manually. Open the app once, then move between pages with the top bar.

## Use the Workspace

The Workspace is where you run image and video generations.

### Text to Image

1. Open **Workspace**.
2. Choose an image model.
3. Keep the workflow on **Text to Image**.
4. Enter your prompt.
5. Adjust size, ratio, steps, seed, LoRA, quantization, or image upscale if needed.
6. Click **Generate**.

The newest result appears in the preview area, and the current session also updates in the history pane.

### Image to Image

1. Switch the workflow to **Image to Image**.
2. Add a reference image.
3. Enter the prompt describing what should change or stay the same.
4. Adjust image strength and any other run settings.
5. Click **Generate**.

Use this when you want to restyle, refine, or iterate from an existing image instead of starting from scratch.

### Text to Video

Text-to-video is available when your installation and available models support video generation.

1. Switch the workflow to **Text to Video**.
2. Choose a video model.
3. Enter your prompt.
4. Adjust ratio, size, frame count, steps, seed, audio, or low-memory mode as needed.
5. Click **Generate**.

On Windows, the Web UI is currently image-only because video generation is not supported there.

### Image to Video

1. Switch the workflow to **Image to Video**.
2. Add the starting image.
3. Enter a prompt describing the motion or camera move.
4. Adjust video settings.
5. Click **Generate**.

This is the workflow to use when you already have a still frame and want the model to animate it.

### Job Controls and Session History

While a job is running, the Workspace shows its progress and any controls supported by that job, such as cancel, pause, or repeat.

The right-side history pane shows recent items from the current session. Use **Open Gallery** there when you want to browse the full saved output history.

### CLI-Only Advanced Controls

The Workspace intentionally stays limited to controls that are fully wired in the browser app.

- Advanced image post-processing flags such as sharpen, contrast, and saturation remain CLI-only.
- Video upscale controls and advanced image upscale tuning remain CLI-only.
- If you need those options, run the matching `ziv-image` or `ziv-video` command from the terminal.

## Use the Models Page

Open **Models** when you need to prepare assets for later runs.

- Review which image models, video models, and LoRAs are currently available.
- Check the model and LoRA directories shown at the top of the page.
- Convert a checkpoint into an installed model entry.
- Import a LoRA from a local file.
- Import a LoRA from Hugging Face when you have the needed access.

The page is for asset management. Actual generation still happens from **Workspace**.

## Use the Gallery

Open **Gallery** to work with outputs that have already been saved.

- Filter between all media, images only, or videos only.
- Sort newest first or oldest first.
- Select an item to inspect its details.
- Reuse an item in the Workspace to prefill a new run.
- Delete individual items or a selected batch.

This is the fastest way to revisit older results without searching the output folder manually.

## Use Configuration

Open **Configuration** to set persistent defaults that should apply to future sessions.

- Choose the default image model.
- Choose the default video model.
- Set the default base image size.
- Change the output directory.
- Review the active model cache and LoRA directories.
- Confirm whether a Hugging Face token is available to the current process.

Changes here affect future sessions and new runs. Per-run choices such as prompt, workflow, LoRA stack, seed, and many generation settings still belong in **Workspace**.

## Where Files Go

Generated media is written to the configured output directory. The same saved outputs appear in the Workspace session history and in the Gallery.

If you want future runs to save somewhere else, change **Output Directory** in **Configuration** and save.

## Troubleshooting

- If no browser tab opens, copy the printed local URL into your browser.
- If `8080` is busy, open the exact port printed in the terminal.
- If you do not see any video workflows, confirm that your platform and installed models support video generation.
- If downloads from Hugging Face fail for gated assets, start the app with `HF_TOKEN` set in your environment.