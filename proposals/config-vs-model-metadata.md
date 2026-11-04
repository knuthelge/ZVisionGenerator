# Keep config.yaml for configuration

**Status:** Proposed (2026-11-03)

## Idea

`zvisiongenerator/config.yaml` has grown into a catalogue of facts about models alongside the real settings. Those facts are pinned repos, file lists, display names, download sizes and memory figures. Move them into packaged model metadata that the app reads directly. That would leave config.yaml as the settings a user may change, which `~/.ziv/config.yaml` is merged over.

## Findings

- **The file is 786 lines,** about 40 pinned revisions and about 50 size, file-list and memory entries.
- **Metadata, not settings:**
  - pinned repos in `model_aliases`, `moss_image` and the `video_model_presets` and `audio_model_presets` download lists;
  - display names and per-platform download sizes in `model_labels` and `prompt_enhancer`;
  - `model_alias_families`;
  - per-model memory figures;
  - `platforms` display names.
- **Settings:**
  - sizes, generation defaults, steps, guidance, schedulers, sharpening and upscale settings;
  - the per-model `settings` blocks;
  - the user's enhancer model;
  - `ui`.
- **A user config can override metadata:** because the user config is deep-merged over the whole file, it can change a pinned revision or a download size by accident. Nothing stops that, and the app has no use for it.
- **Licences went first.** The licence records were the first piece moved out, on the audio branch, to their own packaged data file.
- **Mixed presets need care.** Some presets mix the two in one block; an audio preset holds its label, files and memory next to its `settings`. Splitting them means a model id that ties a metadata entry to its settings.
- **No migration:** user configs that override a moved key stop having an effect. That would be stated in the changelog, with no migration code.

## Next step: one file declaration for downloads

Once the metadata is packaged, declare there which files each model needs: per platform, from which repo and revision, whole repo or a few patterns. Everything that today works out "what does this model need" separately would read that one declaration instead: download plans, whether a model is downloaded, the Models-page rows, the parts of a job's download plan, delete, matching a queued download, the used-repos list and the licence lookup.

- **Seven planners:** the download plans are computed in seven places (the shared plans, the Mac mflux and MLX packs, the CUDA diffusers, H3 and enhancer plans, matte, SeedVR2, and audio), plus the models built on first use.
- **Two ways to tell if a model is downloaded:** image rows look for the local folder, video and audio compare a plan with what is on disk.
- **Five delete functions** each rebuild cache folders and remove whole repo folders.
- **Used repos come from six sources:** the model aliases, video companions, built-model sources, audio repos, enhancer tiers and matte pins.
- **In-flight downloads are keyed per repo,** so a queued download needs extra matching to know a model's files are already coming.
- **Licence lookups re-plan** through a callback in six CLIs and the web UI.
- **Some lists can't be static:** image file lists resolved from the Hub (`model_index.json`, mflux definitions), user models and local folders, models built on first use (Ideogram Instant, MOSS) and the SeedVR2 tier tables. These become resolvers that return the same shape as a declaration.
- **Per-file delete needs care:** removing only a model's own files means counting references to the Hub-cache blobs behind the snapshot symlinks. It is the riskiest part.

Suggested order: audio, matte, the enhancer tiers and SeedVR2 first; then video; then image with its resolvers; per-file delete last. This builds on the metadata move above and goes on its own branch once audio has merged.
