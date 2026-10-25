# Config page

The **Config** page (++g++ ++c++) holds the defaults the whole app uses. Saved settings apply from the next generation.

![The Config page with Settings and Folders and access](../assets/screenshots/placeholder.svg "config-page")

## Settings

| Setting | Sets | When empty |
|---|---|---|
| **Default image model** | The model the Workspace starts with for images | The first available image model |
| **Default video model** | The model the Workspace starts with for video | The first available video model |
| **Base resolution** | The image size preset, such as `m (832x1216)` | The app's base resolution, `m` |
| **Output folder** | Where images and videos are saved | `outputs` in the [data folder](../concepts/data-folder.md), which moves with `ZIV_DATA_DIR` |
| **Prompt enhancer model** | A Hugging Face repo (`owner/name`, optionally `@revision`) or a local folder with a chat model, used instead of Fast and Smart | Fast or Smart, chosen in the Enhance popover |

A setting you changed shows a reset mark; reset it to go back to the default. **Discard changes** undoes edits you haven't saved.

The settings are saved to `config.yaml` in the data folder, which also holds settings this page doesn't show. See [Configuration](../concepts/configuration.md).

## Folders and access

These are shown for reference and can't be changed here:

- **Model cache**: where Hugging Face downloads go, `~/.cache/huggingface/hub` unless `HF_HOME` or `HUGGINGFACE_HUB_CACHE` says otherwise when the app starts.
- **LoRAs folder**: `loras` in the data folder.
- **Hugging Face token**: whether a token was found, and where it comes from. See [Hugging Face token](../concepts/data-folder.md#hugging-face-token).
