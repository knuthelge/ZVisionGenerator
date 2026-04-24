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
- per-model image defaults and per-model video defaults
- the canonical workflow contract metadata
- output directory and startup UI config

The image defaults payload is the source for visible image controls such as ratio, size, width, height, steps, guidance, quantize visibility, and reference-image strength. The video defaults payload is the source for visible video controls such as ratio, size, width, height, frame count, audio, and low-memory state.

## SPA Hydration Rules

The SPA draft store must treat local defaults as placeholders only. Effective values come from backend hydration.

- `draft.hydrateFromContext(ctx, preferredModel)` is the canonical hydration seam.
- Workflow changes go through `draft.onWorkflowChange(workflow, ctx)`.
- Model switches rehydrate dependent values from the matching backend defaults map.
- Visible default-backed fields such as width, height, and image strength must be replaced by hydrated backend values.

The SPA must not guess backend capabilities from hard-coded frontend fallbacks once workspace context has loaded.

## Submit Rules

The browser submits only fields the user can currently see or intentionally set. The backend then applies CLI-aligned normalization and validation before creating generation requests.

Current invariants:

- explicit `image_strength=0.0` is preserved and must not fall through to a default value
- ratio and size resolution stay aligned with CLI helpers
- explicit width and height override ratio and size when provided
- quantize validation uses the shared web config option set rather than duplicated literals in multiple call sites
- image and video submit paths reject unsupported reference-image workflows when no image is provided

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