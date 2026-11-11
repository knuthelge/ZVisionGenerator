# How-to

Recipes for common changes. Each lists where the change goes, what has to move with it and the test to run when you are done. Read [Conventions](conventions.md) first, and [Architecture](architecture.md) if you do not know where something lives.

## Add a config key

1. Add the key with its default, and a comment, to the packaged `zvisiongenerator/config.yaml`. It must be a setting a user may change. A fact about a built-in model (a pin, a file list, a name, a size, a limit, a capability) goes in the model metadata instead; see [Add a built-in model](#add-a-built-in-model).
2. Read it from the config `dict` where it's used, with `.get()` and the same default. Pass the value inward as an argument rather than loading config deep in a helper.
3. If it's a new top-level section that must be a mapping, add it to the type check in `utils/config.load_config()`.
4. If users should change it from the Config page, add a `WritableConfigField` to `web/config_contract.py` and its label to `ConfigPage.svelte`.
5. Document it in [Configuration](../reference/configuration.md). A test fails when a key is missing there.

**Test:** `uv run pytest tests/common/test_config.py tests/common/test_docs_reference.py`, plus the test of the module that reads the key.

## Add a CLI flag

A flag is wired into two places that each build the job's request: the command-line runner and the web server. Skip the second and a Web UI job either ignores the setting or, if the runner reads the new namespace field, fails with an `AttributeError`. A flag people also set in the Web UI has a third and fourth set of places: what is recorded in the output and read back for **Reuse settings**, and what presets and the draft know. Follow an existing sibling flag through the code first, for example `git grep -n contrast -- zvisiongenerator frontend/src`, and copy its path.

1. **Parser.** Add the argument in `image_cli._build_parser()`, `video_cli._build_video_parser()` or `audio_cli._build_audio_parser()`. Flags shared by several commands, such as the enhance flags, go in `enhance_cli.py`. Validate the value in the same module, where the other `parser.error(...)` checks are, so a bad value fails before anything downloads or loads.
2. **Request.** If the value changes what runs, add a field to the request dataclass (`ImageGenerationRequest` in `core/image_types.py`, or the video one in `core/video_types.py`). Fill it where the runner builds the request: `image_runner.py` builds an `ImageGenerationRequest` twice (a display request and the real one), and `video_runner.py` builds a `VideoGenerationRequest`. Settings that have a config default resolve there, as `contrast_amount` does; a new config key follows [Add a config key](#add-a-config-key).
3. **Stage.** If it changes which stages run or what one does, see [Add a workflow stage](#add-a-workflow-stage).
4. **Web server.** Wire it into the form parsing in `web/server.py`: the `argparse.Namespace` that `_parse_image_job()` or `_parse_video_job()` builds from the form, the checks in `_validate_image_args()` or `_validate_video_args()` (they repeat the CLI's rules in the user's words), and the request that the same function builds. A setting the form does not carry still needs its namespace field, set to its off value. Upscale jobs build their request in `web/upscale_api.py`.
5. **Record and Reuse.** `utils/provenance.py` writes the setting into the output (`_image_generation_details()`) and reads it back (`_generation_block()`); `web/gallery.py` turns the recorded value into the Reuse parameters (`_reuse_image_finishing_params()`).
6. **Presets and the workspace contract.** `web/presets.py` lists the keys a preset holds with their types (`_IMAGE_KEYS`, `_VIDEO_KEYS`) and drops every other key. `web/workspace_contract.py` names the control in the workflows that show it and the form fields each workflow clears (`clear_fields`).
7. **Frontend.** Add the form control (see [Add a Web UI feature](#add-a-web-ui-feature)) and the places that know the setting: `lib/state/draft.svelte.ts` (the default, and the maps between form fields and draft keys), `FIELD_SCOPE` in `lib/state/resetScope.ts`, `SETTING_ROWS` in `features/workspace/settingRows.ts`, the key lists in `lib/state/presets.ts`, `lib/state/jobSettings.ts` for Reuse, and the flag in the command the Web UI copies, `lib/state/cliCommand.ts`. The audio commands are checked against `tests/fixtures/audio_cli_cases.json`, which both the Python and the Vitest tests read.
8. **Docs.** Add the flag to the [command reference](../cli/reference.md), and to a guide if it is something people do. Add a `CHANGELOG.md` entry under `[Unreleased]`.

**Test:** add the parser case and the validation case to `tests/image/test_cli_parsing.py` (or `tests/video/test_video_cli.py`, `tests/audio/test_audio_cli.py`), and a web case to `tests/web/test_web_server.py`. Run those files, then `uv run pytest tests/common/test_docs_reference.py`, which fails when a flag is missing from the reference. The first-change walkthrough on the [front page](index.md#your-first-change) shows the loop for the command-line half.

Only some of steps 4 to 7 are enforced. `FIELD_SCOPE` and `SETTING_ROWS` have tests (and the type check) that fail when a draft key lacks an entry. Nothing fails when the namespace field, the provenance record, the Reuse parameter, the preset key, the `clear_fields` entry or the `draft.svelte.ts` maps are missed: the setting is just not recorded, reused, saved in a preset or cleared when the workflow changes. Check each by hand against the sibling flag.

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

**Test:** `uv run pytest tests/image/test_workflow.py` (or `tests/video/test_video_workflow.py`, `tests/video/test_video_stages.py`) and the new test of your function.

## Add a built-in model

A built-in model is one entry under `models:` in `zvisiongenerator/utils/model_metadata.yaml`, keyed by its id. The alias, its name, its downloads, its sizes and its licences all read from it, and the only other file that names a video model is the prefix list in step 1. The header of the file describes the main keys and which keys each kind requires.

1. Add the entry, among the entries of its kind (the order of the entries is the order the Web UI lists them), with its `kind` and `name` (as the Web UI reads it) and its `downloads`: the targets per platform, or one list for all. Pin each to a commit (`owner/repo@revision`). Image aliases are unpinned for now, because image model detection refuses a pinned reference, and so are the SeedVR2 repos; a contract test fails on any other unpinned target. Point at the upstream repo when it is ungated, otherwise at a mirror whose weights match upstream's sha256. The first target with the default role is the alias's source; order is load-bearing (the model's own targets, then the roles its loader asks for, then its video family's companions). A target whose files the loader lists takes `files_from: loader` and no file list. A new video repo must also start with one of the prefixes in `utils/video_model_detect.py`, or the model is not recognised as a video model.
2. Add the keys its kind needs:
    - **Image:** `download_size`, read from a dry-run download plan against an empty cache, per platform when they differ; and `family` for a repo whose family can't be detected offline.
    - **Video:** `family`, which must be a `video_families` entry (`ltx` or `h3`; an LTX model also takes `version` for a `versions` overlay), and for `ltx` the `quantize` level, `4` or `8`. A contract test fails on an LTX entry with any other level, on an LTX pack that does not list its files on macOS and on a family or version that is not defined.
    - **Audio:** `family`, `kinds` (the first is the default) and `limits`; `lyrics`, `voices`, the `voice` and `design` targets of `downloads` (the Base checkpoint that follows a voice, the VoiceDesign checkpoint that designs one; a speech model only) and `platforms.cuda` / `platforms.mps` (`memory_gb`, or its own `family`) are optional. Add the model's settings to `audio_model_presets.<id>.settings` in `config.yaml` (step 4). A test over the packaged entries picks the model up by itself.
3. Add the ids of the licence records that cover the model to `licences`, in display order. Add a record under `licences:` for a licence that has none yet, free licences included; a record several models share is written once. A URL points to the licensor's original repository: a licence file pinned to a commit, or its model page when the repository is gated or has no licence file. Optional `attribution` holds a credit line the licence requires. `make check` fails on an entry that names a record that does not exist, a record no entry uses, a model that has no record, or a URL that is neither pinned nor a model page.
4. Put the model's defaults (steps, guidance, scheduler) in `config.yaml` as settings, in the section of its family: `model_presets`, `video_model_presets` or `audio_model_presets.<id>.settings`.
5. Record the measured memory peak, where the entry takes one. Run the model for real on each platform you have, with the longest clip or take it supports, and note the peak accelerator memory in GiB (`torch.cuda.max_memory_allocated()` on CUDA, `mx.get_peak_memory()` for MLX). An audio entry records it as `memory_gb` under `platforms.cuda` and `platforms.mps`; an enhancer or SeedVR2 entry takes a `memory_gb` map keyed by `darwin`, `win32` and `linux`. Leave a platform's `memory_gb` out while it is unmeasured: the app then shows no memory badge for it instead of a guess. Image and video entries record none; `utils/model_memory.py` estimates them from the weights, and a new family needs an estimator there (see [Add a model family](#add-a-model-family)).
6. Update the [Models](../reference/models.md) page (its Licences table lists every record), and `CHANGELOG.md`.

The metadata is parsed once and shared, and is never merged with the user's config. A reader that merges settings into facts copies at the merge, so it never writes into the shared parse: `family_block` and `builtin_aliases` return copies, and any new reader that adds a setting to a fact does the same. A session check in `tests/conftest.py` fails the run when anything wrote into the cached parse, so a miss shows up in `make check`. Tests change facts through `metadata_copy()` and `use_model_metadata` in `tests/_factories.py`.

**Test:** `uv run pytest tests/contracts/test_model_metadata.py tests/common/test_model_metadata.py tests/common/test_docs_licences.py`, then `uv run pytest tests/downloads` if you added downloads. No test loads the model: run it once by hand on each platform you have.

## Add a model family

A new family touches every layer, so write a proposal first.

- **Metadata and config:** an `image_families` or `video_families` entry in `model_metadata.yaml` with its capability flags (`supports_negative_prompt`, `supports_img2img`, `supports_upscale`, `supports_scheduler`, `dimension_min`, … for an image family; the label, frame rule and capabilities for a video family), and the family's defaults in `model_presets` or `video_model_presets` of `config.yaml`. The Web UI reads capabilities from the metadata, never from the family name. A new scheduler also needs its class in `SCHEDULER_CLASSES` (`schedulers/`).
- **Detection:** `utils/image_model_detect.py` or `utils/video_model_detect.py`.
- **Backends:** loading and generation in `backends/mac/` and `backends/cuda/`, behind the existing protocol; on macOS, an entry in `backends/mac/prompt_encoding.py` (without one, the family keeps its text encoder loaded).
- **Downloads:** the planners in `backends/*/download_plan.py`, which must mirror the files the loader reads. See [Model downloads](downloads.md).
- **Memory:** an estimate in `utils/model_memory.py`, for the badges.
- **Previews:** latent-to-RGB factors in `core/latent_preview.py`, for the live preview.
- **LoRA keys:** a policy in `_POLICIES` and the family's module names in `LORA_MODULES` (`utils/lora_keys.py`), so a LoRA file is recognised and its keys are renamed for the family's loader. Bump `LORA_FAMILY_VERSION` in `utils/lora_family.py` when the rules change, or the LoRA index keeps the family it cached for files it already read.
- **Converter:** a `--model-type` in `converters/convert_checkpoint.py` (`MODEL_TYPES`), if fine-tunes should convert. The Web UI's converter list is written out by hand in `frontend/src/features/models/ModelsPage.svelte`, so add the type there too. No test checks that the two lists agree.

**Test:** the family needs a test in each layer it touches. Run `uv run pytest tests/image tests/video tests/downloads tests/backends tests/common/test_model_memory.py -k <family>`, and `uv run pytest tests/contracts` for the metadata.

## Add a backend capability

When only some platforms can do something:

1. Declare it in `core/`, as a method on the existing protocol or an optional protocol of its own, such as `VideoExtender`.
2. Implement it in `backends/mac/` and/or `backends/cuda/`.
3. Add a `supports_<thing>()` function to `backends/__init__.py`, and use it everywhere else. Never check `sys.platform` outside `backends/__init__.py` to decide.
4. For the Web UI, hide the control through the workflow contract in `web/workspace_contract.py`, as `video_extend` does.

**Test:** `uv run pytest tests/backends/test_backend_protocol.py tests/web/test_web_workspace_contract.py`, and a test per platform backend under `tests/backends/cuda` or `tests/backends/mac`.

## Add an API route

1. Put the logic in its own module in `web/`, as plain functions that take what they need and return data. Keep the route in `web/server.py` to parsing the request and calling it.
2. Use the request helpers in `web/json_fields.py` for optional fields, and raise `HTTPException` with a `detail` the frontend can show.
3. Routes that change state are protected by `web/request_guard.py` automatically.
4. Test the module directly, and the route with FastAPI's `TestClient`.

**Test:** the module's own test in `tests/web/`, and `uv run pytest tests/web/test_web_server.py tests/web/test_web_request_guard.py`. Run `uv run ziv ui`, open `/docs` and check that the route is listed; the routes are not copied into these docs.

## Add a Web UI feature

1. Put page-specific components in `frontend/src/features/<page>/`, and anything shared in `frontend/src/lib/`.
2. Build from the shared components and the `ui-*` classes, and follow the [design system](frontend.md#design-system): tokens, not raw colours; one main action per section; touch as a layer.
3. Ask for confirmation with `ConfirmDialog`, report outcomes with a toast, and send requests through `api`. See [Frontend](frontend.md).
4. Add `*.test.ts` tests next to the source, then run `make frontend-test`.
5. Run `make frontend-build` and commit the rebuilt files.
6. Document it in the [Web UI](../web-ui/index.md) section, with a [screenshot placeholder](docs.md#screenshots).

**Test:** `make frontend-test`; `make check` also runs the palette, hover and packaged-build checks.
