# Licences at download, and a dialog for every download

**Status:** Done (2026-11-02, v0.14.0; built on `feat/audio`)

## Problem

Several built-in models come with terms a user has to follow, and the app says so in one place only: a **Non-commercial** badge on the cut-out switch (`SettingsPane.svelte:435`, `ModelsPage.svelte:727`). The docs' licence column (`docs/concepts/models.md:7-15`) is partly wrong or vague:

- **FLUX.2 Klein 9B** (`klein9b`), the first-run pick on machines with 32 GB or more, is under the FLUX Non-Commercial License; the docs say "See its model page".
- **MiniMax H3** (`h3`) is not licensed in the European Union, the United Kingdom, South Korea or the United States ("Excluded Territories", section 5), and needs MiniMax's permission above USD 20M revenue; the docs say it "restricts use in some regions".
- **LTX** (`ltx-4`, `ltx-8`, `ltx25-4`) needs a paid licence from USD 10M annual revenue; the docs don't mention it.

The moment a user takes on these terms is the download, and that moment is also unannounced in most of the app: the Models page asks before downloading (`downloads.svelte.ts:258-288`), but **Generate downloads whatever the job is missing without asking** (`web_runner._ensure_downloads`, `web/web_runner.py:1020`), which can be 35–107 GB. The only warning is a help line in the model popover (`SettingPopover.svelte:292`).

The audio models ([audio-generation](audio-generation.md)) add three more licences.

## What the user sees

### One download dialog, wherever a download starts

Every Web UI action that would download something first shows the **Download …?** dialog the Models page already uses, then goes ahead or does nothing:

- **Question**: "Download FLUX.2 Klein 9B?" for one model, "Download 2 models?" for more.
- **Parts**: one row per model, even when there is only one: its name on the left and its size on the right ("size unknown" when it can't be checked). Under a restricted model, one line per licence record, indented behind a thin rule, in a neutral form: "FLUX Non-Commercial License v2.1 (Black Forest Labs): Non-commercial · Licence ↗ · NOTICE ↗". A model under two records (LTX-2.3 and Gemma, Stable Audio 3 and Gemma) gets two lines. When any part has lines, the dialog says once, above the parts: "By downloading you agree to the terms shown under each model."
- **Under the parts**: the total and the free space, as today (`lib/state/downloadConfirm.ts`), including the "may run out of space" warning and **Download anyway**. The links open the licence text (for the cut-out models, the model card: "Model card ↗") and, where one exists, the NOTICE, at the revision the app downloads, in a new tab. Enter on a focused link opens it and never confirms the download. There is no checkbox and nothing is stored: the dialog is a notice. On a phone the dialog is a sheet with full-width buttons.
- **Cancel** queues nothing. **Download** queues the job (or the Models-page download) as before; the job card then shows the download progress it shows today.
- When nothing is missing, there is no dialog. Files already downloading (from the Models page, the first-run dialog or another job) count as not missing, so picking a model in the first-run dialog and then pressing Generate doesn't ask about the same 35 GB twice. When Hugging Face can't be reached to size the download, the dialog says "Couldn't check the size" and still offers **Download anyway**; the job then fails as it does today if the download can't run.

Entry points that get the dialog:

| Where | What it can download |
|---|---|
| Workspace **Generate**, every tab (Image, Video, Audio), inline prompt or prompt file | the model (with the video latent upscaler when Upscale is on), the prompt enhancer when the job enhances, the cut-out models with Transparent background |
| **Upscale** from the viewer, the Gallery page or the Workspace (`lib/state/upscale.ts` `startUpscale`) | SeedVR2's tier, or the recorded image model for a creative upscale; the cut-out models when the source was transparent (`web_runner.py:834`) |
| Workspace **Enhance** button | the prompt enhancer |
| Models page **Download** and the first-run dialog | unchanged route; they gain the licence lines |

Left out on purpose:

- **LoRA downloads** (Discover, Hugging Face import, a Generate that waits for a LoRA): each is already started by an explicit Download, and LoRAs have their own licence display in Discover (`LoraDetail.svelte:93`).
- **Checkpoint conversion** on the Models page: an explicit form whose downloads are base components of a model the user already chose.
- **Repeat and Next** inside a running job, and jobs already queued: they download nothing new beyond what their submit showed.
- **The CLI and API clients**: no prompts. The CLI prints the licence line instead (below).

### The licence mark

A small, muted ⓘ button next to a restricted model, built on the existing `Tooltip` atom. Its accessible name is "Licence: <name>" ("Licences: <a> and <b>" for two records).

- **Hover or keyboard focus** with a mouse shows the tooltip, "<name>: <summary>" per record; never on touch, never while the popover is open.
- **Click, tap, Enter or Space** opens one small popover on every layout and input: per record its name, supplier and summary, then **Licence ↗** (or **Model card ↗**) and **NOTICE ↗** buttons (44 px on touch). Escape closes it and returns focus to the mark; a tap outside or Tab past the last link closes it. On touch the mark keeps a 44 px hit area without growing its row.
- **Where it sits**: after the name on every restricted row of the Models page (wide table, tall and phone lists); in the Workspace toolbar right after the **Model** dropdown, for the selected model only (never inside the dropdown, and none for `zit`, `klein4b` or `moss`); on a phone, at the end of each restricted model's row in the Model popover and after the Model dropdown in the All sheet's toolbar, since the Model chip is itself a button and can't hold one.
- **The cut-out models** use the same component: as the ⓘ on their Models-page row (in place of today's badge), and beside the Transparent background switch (inspector, `SettingsPane.svelte:435`, and the phone popover, `SettingPopover.svelte:348`) as a neutral badge reading ⓘ **Non-commercial**, whose accessible name is "Non-commercial. Licence: <name>". Its text comes from the matte record, which the workspace payload carries, so `TRANSPARENT_TERMS` (`features/workspace/transparent.ts`) goes.

### On the CLI

When `ziv image`, `ziv video` or `ziv audio` is about to download a restricted model it doesn't have (or the cut-out models for `--transparent`), it first prints:

```
By downloading you agree to these terms:
FLUX Non-Commercial License v2.1 (Black Forest Labs): Non-commercial
  https://huggingface.co/unsloth/FLUX.2-klein-9B/blob/455f1273c2f08f1fc93d725c2e6f1d7fc681cd94/LICENSE.md
  NOTICE: https://huggingface.co/unsloth/FLUX.2-klein-9B/blob/455f1273c2f08f1fc93d725c2e6f1d7fc681cd94/NOTICE
```

Nothing is printed when the files are already there, and nothing waits for input. A model without a record is never checked, so it costs no Hub request; a model whose plan can't be made offline counts as missing. `ziv model` has no download command (`converters/convert_checkpoint.py:755-790`) and gets none.

## The records

The records live in `zvisiongenerator/utils/licences.yaml`, a packaged file that is read directly and never merged with the user's config. Every URL is pinned to a revision, so the text linked is the text of the weights the app downloads.

```yaml
flux-nc:
  name: "FLUX Non-Commercial License v2.1"
  supplier: "Black Forest Labs"
  summary: "Non-commercial"
  url: "https://huggingface.co/unsloth/FLUX.2-klein-9B/blob/455f1273c2f08f1fc93d725c2e6f1d7fc681cd94/LICENSE.md"
  notice_url: "https://huggingface.co/unsloth/FLUX.2-klein-9B/blob/455f1273c2f08f1fc93d725c2e6f1d7fc681cd94/NOTICE"
  models: [klein9b]
```

`url_label` is optional and defaults to "Licence"; the cut-out record sets "Model card".

| Record | Supplier | Models | Summary | Licence (pinned) | NOTICE |
|---|---|---|---|---|---|
| FLUX Non-Commercial License v2.1 | Black Forest Labs | `klein9b` | Non-commercial | `unsloth/FLUX.2-klein-9B@455f1273…` `LICENSE.md` | same revision `NOTICE` |
| Krea 2 Community License | Krea | `krea2` | Free under $1M revenue | `krea-ai/krea-2@db3984fb…` `assets/hf_samples/LICENSE.pdf` (the card's link) | — |
| Ideogram Non-Commercial Model Agreement | Ideogram | `ideo` | Non-commercial | `unsloth/ideogram-4-fp8@6ac1d7a5…` `LICENSE.md` | same revision `NOTICE` |
| LTX-2 Community License Agreement | Lightricks | `ltx-4`, `ltx-8` | Free under $10M revenue | `Lightricks/LTX-2@a95ab856bf29407b6b066ede0abe1846050db56c` `LICENSE-2` | — |
| LTX-2.x Community License Agreement | Lightricks | `ltx25-4` | Free under $10M revenue | `unsloth/LTX-2.5-FP8@7b40724fc39bbb01d9b4d72638c0d263531c88c2` `LICENSE` | same revision `NOTICE` |
| MiniMax H3 Community License | MiniMax | `h3` | Not licensed in the EU, UK, South Korea or the USA; permission needed over $20M revenue; products show "Powered by MiniMax H3" | `MiniMaxAI/MiniMax-H3@42ed227e…` `LICENSE` | — |
| Gemma Terms of Use | Google | `ltx-4`, `ltx-8` (Gemma 3 text encoder), `sa3` (T5Gemma) | Gemma's use policy applies | the pinned copy `cocktailpeanut/stable-audio-3-medium@a30034d7…` `LICENSE_GEMMA.md` | — |
| BiRefNet (MIT) and ViTMatte (Apache-2.0) | BiRefNet and ViTMatte authors | `matte` | Trained on research datasets that don't allow commercial use | Model card: `hustvl/vitmatte-small-composition-1k@6a58ad76…` `README.md` (trained on Composition-1k) | — |
| CC BY-NC 4.0 | m-a-p | `yue2` | Non-commercial, credit required | `m-a-p/YuE2-3B@8f24312f…` `LICENSE` | — |
| Boson Higgs TTS 3 Research and Non-Commercial License | Boson AI | `higgs3` | Non-commercial; creators may publish with credit | `bosonai/higgs-tts-3-4b@239f63fb…` `LICENSE` | none upstream; the docs carry its NOTICE line |
| Stability AI Community License | Stability AI | `sa3` | Free under $1M revenue | `cocktailpeanut/stable-audio-3-medium@a30034d7…` `LICENSE.md` | same revision `NOTICE` |

The LTX-2.3 `LICENSE-2` link pins the last commit that changed the file (2026-08-26). Which Gemma each LTX alias loads on each platform is checked when the records are written (LTX-2.3 loads Gemma 3; LTX-2.5 loads Gemma 4, which is Apache-2.0). The three audio records arrive with the audio presets.

Not listed, being MIT or Apache-2.0: `klein4b`, `zit`, `moss`, `moss-sft`, SeedVR2, the prompt enhancers, H3's Qwen3-VL text encoder and latent upscaler.

`load_licences()` parses the file: `name`, `supplier`, `summary`, `url` and a non-empty `models` are required, `notice_url` is optional and every URL must contain a revision. `load_config()` still refuses a record that names a model that is not a known alias, audio preset or `matte`. A model may be in two records (LTX-2.3 and Gemma).

## Proposed change

1. **Records.** `utils/licences.yaml`, read by `utils/licences.py` (`LicenceRecord`; `load_licences()`; `parse_licences(raw)`; `licences_for(records, model_id)`; `licence_notice(...) -> tuple[str, ...]`, which returns the CLI lines when the model has a record and its targets are `None` or any is missing), and the unknown-model check in `load_config()`.
2. **One job download plan.** The targets each job type downloads are worked out today inside `WebRunner` (`image_plan`, `video_plan`, `enhancer_plan`, `matte_plan`, `_upscale_download_plan`, `_seedvr2_download_plan`; `web_runner.py:784-1061`). They move into a pure `web/job_downloads.py`, `job_download_parts(request, args, prompts_data, enhance_by_set, config, *, platform_key) -> tuple[JobDownloadPart, ...]`, each part with a display `label`, the `event_model` the runner's download events name today, the model id, the role and the targets. The runner's `_ensure_downloads` calls use it and the plan routes use it, so the dialog and the job can't disagree. The enhancer part uses the choice preflight makes (`prompt_enhancer_session.plan_job_enhancer`, with the inputs `preflight.py:158-194` passes), without loading anything; the `RuntimeError` it raises offline becomes an unchecked part, not a server error.
3. **Plan routes.** `POST /api/generate/plan` takes the same form as `/api/generate` (every mode, audio included; an unknown mode is a 422) and `POST /api/upscale/plan` the same body as `/api/upscale`; both parse with the code their submit route uses (the `_submit_*_job` functions split into parse and queue). They leave out every target that is already downloading, through the shared `DownloadFlights` (`server.py:186-190`, `downloads/flights.py:80`), and return a new multi-part summary (`{parts: [{label, model, repo, files, bytes}], total_bytes, space, unchecked, licences}`) built beside `model_download_plan.plan_summary` (`:148-186`), whose row shape stays as it is for the Models page. `GET /api/models/download/plan` gains the `licences` field. The Enhance button uses that route with `kind=enhancer`.
4. **The dialog.** One shared `confirmDownloads(plan)` in `lib/state/` builds the question, the parts with their licence lines, and the space line (`downloadConfirmText`), and calls `requestConfirm`. `ConfirmDialog` gains a `parts` prop (`{label, size, lines: {text, links}[]}[]`) rendered between the question and the info, and its Enter-to-confirm skips links as it already skips buttons. `downloadStore.request` (Models page, first run) and the Generate, Upscale (`lib/state/upscale.ts` `startUpscale`) and Enhance actions all call it, so licence lines are added in one place. Each caller stays busy while its plan request runs. The popover's help line (`SettingPopover.svelte:292`) changes to "Generate asks before downloading it; the Models page shows its size."
5. **The mark.** `LicenceMark.svelte`: a muted ⓘ button on the `Tooltip` atom (shown on `focusin`, suppressed on touch and while open), or a labelled badge with `label=`; one popover, portalled or fixed to the viewport so scrolling tables and pane edges don't clip it. Placed on every restricted row of `ModelsPage.svelte` (the matte row at `:727` included), in the Workspace toolbar after the Model dropdown, on each restricted row of the phone Model popover and in the All sheet's toolbar, and beside the Transparent switch (`SettingsPane.svelte:435`, `SettingPopover.svelte:348`). The Models inventory rows and the workspace payload carry the records: each model option in the payload, and the background-removal block (`web/workspace_api.py:420-441`).
6. **CLI.** `image_cli` and `video_cli` (and `audio_cli`) look up the record by the alias the user picked (`picked_model` in `image_cli.py:224`; `canonical_model_name` as at `video_cli.py:422`), before planning anything, and print `licence_notice` for the model and, with `--transparent`, the matte, before `ensure_built_model` (`image_cli.py:355`), `run_preflight` (`video_cli.py:605`) and `_fetch_matte_models` (`image_cli.py:115`).
7. **Docs and changelog.** A **Licences** section in `docs/concepts/models.md` lists every record with supplier, summary and links, and the attributions the licences ask of apps and creators ("Powered by Stability AI"; "Powered by MiniMax H3"; "Built with Higgs TTS 3 licensed from Boson AI USA, Inc." with Boson's NOTICE line; YuE2's repositories). The model tables' License column uses the record names. A test fails when a record is missing from that section. `docs/web-ui/workspace.md` and `models.md` describe the dialog and the licence mark. CHANGELOG `[Unreleased]`: "Generate, Upscale and Enhance now ask before downloading a model, with its size and free space, as the Models page does." and "The download dialog, the Model picker, the Models page and the CLI name the terms of models that have them, with links to the licence and its NOTICE." No page in `docs/` links to a proposal.

Done when: the parser refuses a record without a required key, an unpinned URL and an unknown model id; `job_download_parts` returns the same targets the runner downloads for an image job with Enhance and Transparent, a video job with upscale, an image upscale with SeedVR2, a creative one, the upscale of a transparent image and a video-file upscale, and an unchecked part when the enhancer can't be planned offline; `/api/generate/plan` returns parts and licences for `klein9b`, nothing for a downloaded `zit`, and nothing for a `klein9b` that is already downloading; Generate shows one dialog listing every missing part, Cancel queues nothing, and nothing missing or everything already downloading means no dialog (Vitest, with a `planGenerate` default of an empty plan in `test-utils/workspaceMocks.ts`); Upscale and Enhance ask the same way; a two-record part shows two lines; Enter on a licence link doesn't confirm; the licence mark shows a tooltip on hover and keyboard focus and one popover on click, tap, Enter and Space, closes on Escape with focus back on it, follows the selected model in the toolbar, ends each restricted row of the phone Model popover and never sits on the chip, and `zit`'s row and selection have none; `licence_notice` returns lines for a missing or unplannable restricted model and nothing for a present one or a model without a record; the CLI prints them before the steps named above (mocked); the docs test passes; `make frontend-build`.

## Left out on purpose

- **Acceptance**: no checkbox, stored acceptance or server refusal. The dialog is a notice.
- **A Licences page.**
- **Copied licence text**: links to pinned revisions are exact.
- **A prompt on the CLI**, and dialogs for LoRAs and conversions (see the entry-point list).

## Open questions

None. The summaries are drafts from the licence texts at the pinned revisions; each is checked against its text when the record is written.

## Build order

Built first on `feat/audio`, before the Audio tab, which adds its three records.

1. `feat(models): add licence records`: the records file, parser, `licence_notice`.
2. `refactor(web): plan a job's downloads in one place`: `job_download_parts`, used by the runner; no behaviour change.
3. `feat(web): ask before a job downloads models`: the plan routes, `confirmDownloads` (also used by `downloadStore.request`), Generate, Upscale, Enhance, the help line; CHANGELOG line.
4. `feat(web): show model licences at download, in the Model picker and on the Models page`: the dialog's parts and licence lines, `LicenceMark`, the matte badge, `make frontend-build`.
5. `feat(cli): name a model's licence before downloading it`.
6. `docs(models): list model licences`: the Concepts section and its test, the Web UI pages, CHANGELOG.
