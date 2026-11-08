# How-to

Recipes for common changes. Each lists where the change goes and what has to move with it. Read [Conventions](conventions.md) first.

## Add a config key

1. Add the key with its default, and a comment, to the packaged `zvisiongenerator/config.yaml`. It must be a setting a user may change. A fact about a built-in model (a pin, a file list, a name, a size, a limit, a capability) goes in the model metadata instead; see [Add a built-in model](#add-a-built-in-model).
2. Read it from the config `dict` where it's used, with `.get()` and the same default. Pass the value inward as an argument rather than loading config deep in a helper.
3. If it's a new top-level section that must be a mapping, add it to the type check in `utils/config.load_config()`.
4. If users should change it from the Config page, add a `WritableConfigField` to `web/config_contract.py` and its label to `ConfigPage.svelte`.
5. Document it in [Configuration](../concepts/configuration.md). A test fails when a key is missing there.

## Add a CLI flag

1. Add the argument in `image_cli._build_parser()`, `video_cli._build_video_parser()` or `audio_cli._build_audio_parser()`. Flags shared by both, such as the enhance flags, go in `enhance_cli.py`.
2. Carry it through to where it's used. If it changes which stages run, see [Add a workflow stage](#add-a-workflow-stage).
3. If the Web UI has the same setting, keep the two in step: the form, the job arguments in `web/server.py`, and the command the Web UI copies (`frontend/src/lib/state/cliCommand.ts`). The audio commands are checked against `tests/fixtures/audio_cli_cases.json`, which both the Python and the Vitest tests read.
4. Test the parser option and the behavior.
5. Add it to the [command reference](../cli/reference.md), and to a how-to page if it's something people do. A test fails when a flag is missing from the reference.

## Add a workflow stage

A stage is a function `(request, artifacts) -> StageOutcome`: `ImageGenerationRequest` and `ImageWorkingArtifacts` for images, the video types for video. It reads the request, changes the working artifacts, and returns `success`, `skipped`, `retry` or `failed`.

```python
def contrast_stage(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome:
    """Apply contrast adjustment to current image."""
    if artifacts.image is None:
        return StageOutcome.success
    from zvisiongenerator.processing.contrast import adjust_contrast

    artifacts.image = adjust_contrast(artifacts.image, amount=request.contrast_amount)
    return StageOutcome.success
```

1. Put the work itself in a pure function, for example in `processing/` or `utils/`, and keep the stage a thin adapter.
2. Add the stage to `workflows/image_stages.py` or `workflows/video_stages.py`.
3. Add it to the builder in `workflows/__init__.py` (`build_workflow()`, `build_video_workflow()` or an upscale builder), under the condition that turns it on.
4. Test the pure function directly, and the stage with a mock backend and plain artifacts.

No stage loads the prompt enhancer: work that must happen before the model loads belongs in preflight. See [Extending preflight](workflows.md#extending-preflight).

## Add a built-in model

A built-in model is one entry under `models:` in `zvisiongenerator/utils/model_metadata.yaml`, keyed by its id. The alias, its name, its downloads, its sizes and its licences all read from it, and the only other file that names a video model is the prefix list in step 1. The header of the file describes the main keys and which keys each kind requires.

1. Add the entry, among the entries of its kind (the order of the entries is the order the Web UI lists them), with its `kind` and `name` (as the Web UI reads it) and its `downloads`: the targets per platform, or one list for all. Pin each to a commit (`owner/repo@revision`). Image aliases are unpinned for now, because image model detection refuses a pinned reference, and so are the SeedVR2 repos; a contract test fails on any other unpinned target. Point at the upstream repo when it is ungated, otherwise at a mirror whose weights match upstream's sha256. The first target with the default role is the alias's source; order is load-bearing (the model's own targets, then the roles its loader asks for, then its video family's companions). A target whose files the loader lists takes `files_from: loader` and no file list. A new video repo must also start with one of the prefixes in `utils/video_model_detect.py`, or the model is not recognised as a video model.
2. Add the keys its kind needs:
    - **Image:** `download_size`, read from a dry-run download plan against an empty cache, per platform when they differ; and `family` for a repo whose family can't be detected offline.
    - **Video:** `family`, which must be a `video_families` entry (`ltx` or `h3`; an LTX model also takes `version` for a `versions` overlay), and for `ltx` the `quantize` level, `4` or `8`. A contract test fails on an LTX entry with any other level, on an LTX pack that does not list its files on macOS and on a family or version that is not defined.
    - **Audio:** `family`, `kinds` (the first is the default) and `limits`; `lyrics`, `voices` and `platforms.cuda` / `platforms.mps` (`memory_gb`, or its own `family`) are optional. Add the model's settings to `audio_model_presets.<id>.settings` in `config.yaml` (step 4). A test over the packaged entries picks the model up by itself.
3. Add the ids of the licence records that cover the model to `licences`, in display order. Add a record under `licences:` for a licence that has none yet, free licences included; a record several models share is written once. A URL points to the licensor's original repository: a licence file pinned to a commit, or its model page when the repository is gated or has no licence file. Optional `attribution` holds a credit line the licence requires. `make check` fails on an entry that names a record that does not exist, a record no entry uses, a model that has no record, or a URL that is neither pinned nor a model page.
4. Put the model's defaults (steps, guidance, scheduler) in `config.yaml` as settings, in the section of its family: `model_presets`, `video_model_presets` or `audio_model_presets.<id>.settings`.
5. Update the [Models](../concepts/models.md) page (its Licences table lists every record), and `CHANGELOG.md`.

The metadata is parsed once and shared, and is never merged with the user's config. A reader that merges settings into facts copies at the merge, so it never writes into the shared parse: `family_block` and `builtin_aliases` return copies, and any new reader that adds a setting to a fact does the same. A session check in `tests/conftest.py` fails the run when anything wrote into the cached parse, so a miss shows up in `make check`. Tests change facts through `metadata_copy()` and `use_model_metadata` in `tests/_factories.py`.

## Add a model family

A new family touches every layer, so write a proposal first.

- **Metadata and config:** an `image_families` or `video_families` entry in `model_metadata.yaml` with its capability flags (`supports_negative_prompt`, `supports_img2img`, `supports_upscale`, `supports_scheduler`, `dimension_min`, … for an image family; the label, frame rule and capabilities for a video family), and the family's defaults in `model_presets` or `video_model_presets` of `config.yaml`. The Web UI reads capabilities from the metadata, never from the family name. A new scheduler also needs its class in `SCHEDULER_CLASSES` (`schedulers/`).
- **Detection:** `utils/image_model_detect.py` or `utils/video_model_detect.py`.
- **Backends:** loading and generation in `backends/mac/` and `backends/cuda/`, behind the existing protocol; on macOS, an entry in `backends/mac/prompt_encoding.py` (without one, the family keeps its text encoder loaded).
- **Downloads:** the planners in `backends/*/download_plan.py`, which must mirror the files the loader reads. See [Model downloads](downloads.md).
- **Memory:** an estimate in `utils/model_memory.py`, for the badges.
- **Previews:** latent-to-RGB factors in `core/latent_preview.py`, for the live preview.
- **Converter:** a `--model-type` in `converters/convert_checkpoint.py`, if fine-tunes should convert.

## Add a backend capability

When only some platforms can do something:

1. Declare it in `core/`, as a method on the existing protocol or an optional protocol of its own, such as `VideoExtender`.
2. Implement it in `backends/mac/` and/or `backends/cuda/`.
3. Add a `supports_<thing>()` function to `backends/__init__.py`, and use it everywhere else. Never check `sys.platform` outside `backends/__init__.py` to decide.
4. For the Web UI, hide the control through the workflow contract in `web/workspace_contract.py`, as `video_extend` does.

## Add an API route

1. Put the logic in its own module in `web/`, as plain functions that take what they need and return data. Keep the route in `web/server.py` to parsing the request and calling it.
2. Use the request helpers in `web/json_fields.py` for optional fields, and raise `HTTPException` with a `detail` the frontend can show.
3. Routes that change state are protected by `web/request_guard.py` automatically.
4. Test the module directly, and the route with FastAPI's `TestClient`.

## Add a Web UI feature

1. Put page-specific components in `frontend/src/features/<page>/`, and anything shared in `frontend/src/lib/`.
2. Build from the shared components and the `ui-*` classes, and follow the [design system](frontend.md#design-system): tokens, not raw colours; one main action per section; touch as a layer.
3. Ask for confirmation with `ConfirmDialog`, report outcomes with a toast, and send requests through `api`. See [Frontend](frontend.md).
4. Add `*.test.ts` tests next to the source, then run `make frontend-test`.
5. Run `make frontend-build` and commit the rebuilt files.
6. Document it in the [Web UI](../web-ui/index.md) section, with a [screenshot placeholder](docs.md#screenshots).
