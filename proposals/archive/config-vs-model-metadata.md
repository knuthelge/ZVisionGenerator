# Keep config.yaml for configuration

**Status:** Done (2026-11-08, v0.14.0; built on `refactor/config`)

## Idea

`zvisiongenerator/config.yaml` has grown into a catalogue of facts about models alongside the real settings. Those facts are pinned repos, file lists, display names, download sizes and memory figures. Move them into packaged model metadata that the app reads directly. That would leave config.yaml as the settings a user may change, which `~/.ziv/config.yaml` is merged over.

## What shipped

- **One packaged metadata file.** `utils/model_metadata.yaml` holds the facts: the licence records (folded in from `utils/licences.yaml`), the image and video families (capabilities, frame rules, versions, companions) and a `models` entry per built-in model with its name, sizes, memory figures, licences and pinned downloads per platform. It is read directly, cached, never merged with the user's config and never written to. `config.yaml` is settings only (693 lines down to 255), and a user override of a fact has no effect. The `models` entry and the config's settings meet by model id.
- **Declarations feed the planners.** `downloads/declaration.py` reads a model's targets and roles from the metadata, its own first and then its video family's companions. The download plans, the video rows' downloaded checks, the Models-page rows, a job's download parts, the SeedVR2, audio, matte and enhancer files and the models built on first use read it, so a pin is written once in the app (the MOSS-Image and BiRefNet pins are also in the `Makefile`, for the vendored code, and a contract test keeps them equal).
- **Delete and used repos from the same declaration.** One function answers what deleting a model frees, and the used-repos list is the union of the declared repos, replacing eight separate sources. The Hugging Face cache view marks only the repos this platform's models load.
- **Fixes found on the way.** A Windows/Linux LTX model with a file missing no longer shows as downloaded, deleting the last LTX-2.3 model on a Mac removes its Gemma 3 text encoder, and saving on the Config page keeps the comments in `config.yaml`.
- **Per-file delete was not needed.** No declaration splits a repo between models, so the reference counting behind the cache's blobs was never required.

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

## Next step: one file declaration for downloads (done)

With the metadata packaged, it declares which files each model needs: per platform, from which repo and revision, whole repo or a few patterns. Everything that had worked out "what does this model need" separately reads that one declaration: download plans, whether a model is downloaded, the Models-page rows, the parts of a job's download plan, delete, matching a queued download, the used-repos list and the licence lookup.

What it replaced:

- **Seven planners:** the download plans were computed in seven places (the shared plans, the Mac mflux and MLX packs, the CUDA diffusers, H3 and enhancer plans, matte, SeedVR2, and audio), plus the models built on first use.
- **Two ways to tell if a model is downloaded:** image rows looked for the local folder, video and audio compared a plan with what was on disk.
- **Five delete functions** each rebuilt cache folders and removed whole repo folders.
- **Used repos came from six sources:** the model aliases, video companions, built-model sources, audio repos, enhancer tiers and matte pins.
- **In-flight downloads are keyed per repo,** so a queued download needs extra matching to know a model's files are already coming. This is unchanged.
- **Licence lookups re-planned** through a callback in six CLIs and the web UI.

Lists that cannot be static stay with their loaders: image file lists resolved from the Hub (`model_index.json`, mflux definitions), user models and local folders. A declared target marks them with `files_from: loader`. The models built on first use (Ideogram Instant, MOSS) and the SeedVR2 tiers declare theirs with roles.

Per-file delete was left out: removing only a model's own files would mean counting references to the Hub-cache blobs behind the snapshot symlinks, and no declaration splits a repo between models.
