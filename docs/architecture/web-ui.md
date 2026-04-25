# Web UI Contract

This note documents the maintained contract between the FastAPI web layer and the Svelte SPA.

## Purpose

The Web UI exposes a deliberately smaller surface than the CLI. The contract exists to keep that browser surface truthful:

- visible controls must map to real backend behavior
- defaults must come from the same config layering as the CLI
- unsupported CLI features must stay out of the Workspace UI

## Canonical Workflows

The SPA-facing workflow vocabulary is:

- `txt2img`
- `img2img`
- `txt2vid`
- `img2vid`

Legacy aliases such as `image`, `video`, `i2i`, and `i2v` may still be accepted when parsing old reuse URLs or metadata, but any newly emitted workspace URL must use the canonical values above.

## Workspace Bootstrap Contract

`/api/workspace` is the bootstrap endpoint for the SPA. It must provide enough data for the browser to render supported controls without inventing fallback values locally.

The response includes:

- image and video model lists
- LoRA options and recent history assets
- current image and video model selections
- global option sets for ratios, size options, schedulers, and quantize choices
- prompt source and prompt-file contract metadata
- per-model image defaults and per-model video defaults
- the canonical workflow contract metadata
- output directory and startup UI config

The image defaults payload is the source for visible image controls such as ratio, size, width, height, steps, guidance, quantize visibility, and reference-image strength. The video defaults payload is the source for visible video controls such as ratio, size, width, height, frame count, audio, and low-memory state.

`workflow_contract.definitions[workflow].visible_controls` is the browser's authoritative control inventory. The SPA must use that inventory, plus the current model defaults payload, to decide which controls can render for a workflow instead of hard-coded workflow-name checks.

## SPA Hydration Rules

The SPA draft store must treat local defaults as placeholders only. Effective values come from backend hydration.

- `draft.hydrateFromContext(ctx, preferredModel)` is the canonical hydration seam.
- Workflow changes go through `draft.onWorkflowChange(workflow, ctx)`.
- Model switches rehydrate dependent values from the matching backend defaults map.
- Visible default-backed fields such as width, height, and image strength must be replaced by hydrated backend values.
- Before `/api/workspace` resolves, the Workspace must render a loading-safe shell only: editable controls disabled, submit disabled, and no rendered numeric or select value presented as authoritative.
- URL or stored draft values may be staged before hydration, but they may only be applied to fields that remain visible in the hydrated workflow contract.

The SPA must not guess backend capabilities from hard-coded frontend fallbacks once workspace context has loaded.

## Prompt File Contract

Prompt-file support is backend-authoritative. The browser may cache the last successful path and option id locally, but it must re-validate that state through the backend before treating it as runnable.

Current invariants:

- `/api/workspace` publishes the prompt-source inventory and `prompt_file` contract metadata.
- `/api/picker` returns host-local picker status payloads for prompt-file browse flows.
- `/api/prompt-files/inspect` normalizes the submitted path, validates the extension, parses the YAML with the shared prompt loader, and returns stable option ids based on source positions.
- `/api/prompt-files/read` returns the normalized path, active option metadata, and raw file contents for the editor dialog.
- `/api/prompt-files/write` validates the edited YAML before replacing the file atomically; invalid saves must not mutate the file on disk.
- When the current option id disappears after a refresh or save, the SPA must clear the selection and block submission until the user picks another active option.

## Submit Rules

The browser submits only fields the user can currently see or intentionally set. The backend then applies CLI-aligned normalization and validation before creating generation requests.

Current invariants:

- explicit `image_strength=0.0` is preserved and must not fall through to a default value
- ratio and size resolution stay aligned with CLI helpers
- explicit width and height override ratio and size when provided
- quantize validation uses the shared web config option set rather than duplicated literals in multiple call sites
- image and video submit paths reject unsupported reference-image workflows when no image is provided
- `prompt_source=inline` submits inline prompt fields only
- `prompt_source=file` submits the normalized `prompts_file` path plus one `prompt_option_id`; inline prompt fields must stay out of the active payload
- file-mode submit resolution uses the shared prompt-file inspection helpers so stale or inactive option ids fail validation instead of silently drifting

## Truthful Control Boundary

The Workspace should render only controls that are fully wired end to end.

- supported controls are hydrated from `/api/workspace`, serialized by the SPA, and consumed by the backend
- CLI-only controls stay out of the rendered Workspace UI and out of end-user Web UI docs
- gallery and history reuse may downgrade to a compatible workflow or model, but the fallback reason must be surfaced to the UI via `reuse_state`

## Routes In Scope

The maintained browser surface is:

- `/app`
- `/app-static/*`
- `/media/{asset_path}`
- `/api/workspace`
- `/api/generate`
- `/api/history`
- `/api/gallery`
- `/api/config`
- `/api/models`
- `/jobs/{job_id}` and related SSE/control endpoints

Future web changes should extend this contract rather than reintroducing separate template-era behavior or web-only default logic.

## Host-Local Path Resolution

Browse operations and prompt-file edit flows resolve paths on the machine running the Z-Vision server process, not on the device running the browser. The `/api/picker` endpoint invokes the host-side file picker; `/api/prompt-files/read` and `/api/prompt-files/write` open and replace files on the server's local filesystem. The SPA renders whatever normalized path the backend returns and must not attempt to resolve or validate paths against the browser device's filesystem.

This seam is relevant for remote-access scenarios: a user accessing the UI from a second machine will see and edit paths that exist on the server machine, which may differ from paths on their own device.

## Browser-Excluded Features

The following CLI capabilities are intentionally absent from the web Workspace UI and must not be added without a corresponding backend submit contract:

- **Full prompt-file batch execution** — the CLI can iterate over every option in a prompts file in a single run; the web UI supports only single-option selection per submission.
- **Per-run output-directory override** — the CLI accepts `--output-dir` per invocation; the web UI uses the server's configured output directory unconditionally.
- **`upscale_save_pre`** — the CLI flag saves the pre-upscale image as a sidecar; there is no web UI field, submit path, or backend route for this behavior.

## Models Inventory Contract

`/api/models` is a narrow inventory endpoint for the Models page. It returns model and LoRA directories, discovered image models, discovered video models, discovered LoRAs, and the current process-level Hugging Face token status.

Placeholder inventory concepts that are not computed by the backend must stay out of this payload. Surface inventory fields should not be added unless the server can populate them with real, reviewable data.