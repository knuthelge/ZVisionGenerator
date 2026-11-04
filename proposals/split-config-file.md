# Split config.yaml into a file per area

**Status:** Proposed (2026-11-02)

## Idea

The packaged `zvisiongenerator/config.yaml` is one 786-line file with more than 20 sections, covering image, video, audio, the enhancer, the UI and model facts. Split it into a `config/` folder with a file per area, such as `image.yaml`, `video.yaml`, `audio.yaml`, `enhancer.yaml` and `ui.yaml`. The loader reads them into the same `dict` as today, so no code that reads config changes.

## Findings

- **One loader reads it:** `utils/config.py` `load_config()` reads the one packaged file, deep-merges `~/.ziv/config.yaml` over it, resolves `${…}` references and checks that the known sections are mappings. Reading a list of files in a fixed order fits into that one function.
- **Cross-file references:** 11 `${video_model_presets…}` references sit in `model_aliases`, so the LTX aliases point at video presets. They resolve after the merge, so they keep working across files as long as resolution stays last.
- **The user config can stay one file:** the Config page writes it (`web/config_contract.py`), and users edit it by hand. Its keys don't depend on which packaged file a section lives in.
- **The docs already split it by section:** `docs/concepts/configuration.md` has a heading per section. A sentence on which file holds which section is enough.
- **Packaging:** the folder must ship in the wheel, as `config.yaml` does now.
- **Do the metadata move first.** [Keep config.yaml for configuration](config-vs-model-metadata.md) moves model facts out first. Doing that first leaves smaller files with only settings in them, and avoids moving the same lines twice.
- **No migration:** the user config's keys don't change, so nothing breaks for users.
