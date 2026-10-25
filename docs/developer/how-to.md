# How-to

Recipes for common changes. Each lists where the change goes and what has to move with it. Read [Conventions](conventions.md) first.

## Add a config key

1. Add the key with its default, and a comment, to the packaged `zvisiongenerator/config.yaml`.
2. Read it from the config `dict` where it's used, with `.get()` and the same default. Pass the value inward as an argument rather than loading config deep in a helper.
3. If it's a new top-level section that must be a mapping, add it to the type check in `utils/config.load_config()`.
4. If users should change it from the Config page, add a `WritableConfigField` to `web/config_contract.py` and its label to `ConfigPage.svelte`.
5. Document it in [Configuration](../concepts/configuration.md). A test fails when a key is missing there.

## Add a CLI flag

1. Add the argument in `image_cli._build_parser()` or `video_cli._build_video_parser()`. Flags shared by both, such as the enhance flags, go in `enhance_cli.py`.
2. Carry it through to where it's used. If it changes which stages run, see [Add a workflow stage](#add-a-workflow-stage).
3. If the Web UI has the same setting, keep the two in step: the form, the job arguments in `web/server.py`, and the command the Web UI copies (`frontend/src/lib/state/cliCommand.ts`).
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

## Add a built-in model alias

1. Add the alias to `model_aliases` in `config.yaml`: a repo, or a map per platform. Point at the upstream repo when it is ungated, otherwise at a mirror whose weights match upstream's sha256.
2. Add `model_labels.<alias>.name`, and for an image model `download_size_label` per platform, read from a dry-run download plan against an empty cache.
3. For a repo whose family can't be detected offline, add it to `model_alias_families`.
4. Update the [Models](../concepts/models.md) page, and `CHANGELOG.md`.

## Add a model family

A new family touches every layer, so write a proposal first.

- **Config:** a `model_presets` entry with its defaults and capability flags (`supports_negative_prompt`, `supports_img2img`, `supports_upscale`, `supports_scheduler`, `dimension_min`, …). The Web UI reads capabilities from here, never from the family name.
- **Detection:** `utils/image_model_detect.py` or `utils/video_model_detect.py`.
- **Backends:** loading and generation in `backends/mac/` and `backends/cuda/`, behind the existing protocol.
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
