# Backlog from the v1 review

**Status:** Backlog (2026-10-27)

Findings from the pre-v1 codebase review that were verified but deliberately left out of v1. Each is small enough to pick up on its own; none has a proposal of its own. Items are grouped by area, and each says what a user sees, where it sits, and what the fix would be. Decisions taken at the review are noted.

## Web UI

- **Job finished on another page is not announced** (desktop and tablet). The only lifecycle subscriber is the Workspace, which unsubscribes on unmount; the Gallery page never merges new outputs. Covered by [Know when a job is done](job-notifications.md).
- **A stalled SSE stream is never detected.** Recovery is driven by `onerror` only; the 4 s `/api/jobs` poll ignores its result while the followed job is non-terminal, and `lastEventAt` is display-only (`frontend/src/lib/api/sse.ts`, `lib/state/job.svelte.ts`). A half-open connection (phone changing Wi-Fi, laptop waking) leaves the card "running" until reload. Fix: treat a long silence or a poll that no longer lists the job as a reconnect, closing the stream first so a late `job_completed` does not finish the job twice.
- **Four failure toasts drop the server's reason**, against the documented `errorText(err, fallback)` rule: gallery delete (`GalleryPage.svelte`), workspace delete, remove from queue and clear queue (`WorkspacePage.svelte`). Decision: skipped for v1.
- **A UI save rewrites `config.yaml` without comments.** `web/config_contract.py` re-dumps with `yaml.safe_dump`; skipping the first-run dialog or saving Settings loses commented aliases. Prompt files already round-trip with ruamel. Decision: skipped for v1; either switch the writer to ruamel or say so in the Config page docs.
- **Uploaded reference images accumulate for good** in `<output>/.web_uploads/<uuid>`, including when the submit then fails validation (`web/server.py`, `_save_uploaded_reference_image`). Recorded settings point at them for Reuse, so a cleanup must keep referenced files.
- **Background-thread failures in the download lane and video conversion are not logged**; only `str(exc)` is kept (`web/download_lane.py`, `web/video_conversion.py`). Add `logger.exception` so a bug report has a traceback.
- **Deleting `ltx-4` on Windows/Linux also deletes `ltx-8`'s files** (found 2026-10-26). Both aliases point at the same ~104 GB diffusers download; `ltx-4` only adds a 4-bit stored copy. `model_delete_target` (`web/model_delete.py`) keeps a shared download only while another alias has stored quants, so deleting `ltx-4` removes the download `ltx-8` loads from, and the dialog never names `ltx-8`. Decision: "last holder takes the shared files". An alias holds a shared download once it was used (a job ran with it, or its row's Download was pressed) and not deleted since, recorded in the app database. Deleting a holder while another holder remains removes only its own files (its stored copies) and shows it as not downloaded; deleting the last holder removes the download too; an alias that was never used does not hold it. The same rule could replace MOSS-Image's built-folder rule. Its own standard run (discovery, PRD, review).
- **No test ties the frontend's SSE listener list to the server's event names.** `sse.ts` hand-lists the event types; a renamed or new server event is dropped silently. Correct today; add a contract test.
- **The docs screenshots of the prompt box are out of date.** `docs/assets/screenshots/workspace.webp` and `phone-workspace.webp` still show the old bars (the "used" pill, Clear and the word count in the tools row) from before the bars were restructured, and the Recent prompts section in `docs/web-ui/workspace.md` still uses the `placeholder.svg` "recent-prompts" shot. Retake them with the screenshot process in `docs/developer/docs.md`, using invented prompts and test images.

## CLI

- **Ctrl+C prints a full traceback** in every CLI, often from inside torch or hf_xet. `ziv video` has no key-based stop unless `--review` is on, so Ctrl+C is the normal way to stop a 30–100 GB first download. Harmless (terminal restored, download resumes). Catch `KeyboardInterrupt` at the entry points and print one line.
- **A custom image alias with a per-platform map lacking this platform crashes `ziv image`** with a `ValueError` traceback (`image_cli.py`, `resolve_model_path` not wrapped); `ziv video` catches it. Only custom aliases.
- **The CLI H3 converter prints a traceback on network and disk-full failures** during the component download (`converters/convert_checkpoint.py` catches only `ValueError`, `FileExistsError`, `RuntimeError`; `utils/hf_snapshot.py` does not wrap Hub errors). Nothing checks free space before writing a 12–38 GB converted folder. The Web UI path shows the last stderr line and is fine.
- **ffprobe is not checked.** `ensure_ffmpeg` checks ffmpeg only; ffmpeg without ffprobe gives `[Errno 2] ... 'ffprobe'`.
- **`ziv image -p <directory>`, or an unreadable prompts file, still tracebacks.** The early check in `image_cli.py` turns only `FileNotFoundError` into "Prompts file not found … Provide --prompt or a valid --prompts-file"; `Path.read_text` raises `IsADirectoryError` or `PermissionError` for those inputs. `ziv video` checks `Path(...).is_file()` first. Catch `OSError` or add the same pre-check. Decision: skipped for v1.
- **The prompts-file message lives in both CLIs.** The same sentence and check sit in `image_cli.py` and `video_cli.py`. One `require_prompts_file()` helper in `utils/prompts.py` would keep them from drifting. Decision: skipped for v1.
- **macOS: `ziv video --extend` on an H3 clip picks the LTX default model.** `video_cli.py` (`_model_available_locally`) judges H3 "not available locally" because the Mac plan downloads only some text-encoder shards; it then extends with `ui.default_models.video` and may start an LTX download. The Web UI judges by download plan. Needs a Mac to confirm.

## Generation and backends

- **One exception in an image batch aborts the batch with a traceback.** No image stage returns `StageOutcome.failed` or `.retry`, so the image runner's failure and retry branches are unreachable and nothing in the loop catches exceptions (`image_runner.py`); the video runner catches per iteration. CUDA image and LTX have no `torch.cuda.OutOfMemoryError` handling (SeedVR2 and H3 do). Decision: keep as is for v1.
- **Extend refuses clips made with the built-in `--upscale 2`** and says "extend the original instead", but the built-in upscale discards the half-size render (`utils/video_extend.py`, `provenance.py`). Documented as "extend first, upscale last". Reword the message for generated clips, or allow it.
- **CUDA: the first LTX run prepares copies for minutes with no phase shown and Cancel waits** (`backends/cuda/video/backend.py`, `_prepared_weights` only logs; `load_model` takes no `on_phase`/`cancelled`, unlike the image loader's `saving_quant` phase).
- **CUDA: a SIGKILL during a stored-copy save leaves a hidden multi-GB `.partial` folder**, swept only on a retry of the same target more than an hour later (`utils/stored_quant.py`, `sweep_stale_partials`). Ctrl-C and ordinary errors clean up.
- **CUDA H3: a LoRA trained on a ComfyUI pruned checkpoint fails minutes in** with a raw `mat1 and mat2 shapes cannot be multiplied`; the AdaLN down factor is never shape-checked (`h3/stored_copy.py`, `h3/adaln.py`). Check the shape at load and name the cause.
- **macOS H3 refuses LoRAs with AdaLN modules** that CUDA accepts (`backends/mac/h3_video.py`). Documented limitation; in hand in the Mac test.
- **Video provenance records the prompt template, not the resolved `{a|b}` choice** (`utils/provenance.py`), so an Extend without `--prompt` can pick the other option. Images record the resolved text.

- **LoRAs that don't fit fail late, silently or with the wrong message** (found 2026-10-28 while researching Discover LoRAs, sampling 1,710 Hugging Face LTX and MiniMax H3 headers). They fail after the prompt is encoded, so minutes into a job.
  - **macOS LTX silently ignores** LoRAs with no `lora_A`/`lora_B` pairs (LyCORIS, kohya, LoKr) and ones that match no module: about 9% of sampled LTX files. This happens in the vendored attach (`packages/ltx_core_mlx/loader/lora_adapters.py`), where the skipped list is dropped in `ltx_pipelines_mlx/_base.py`. Windows/Linux rejects the same files, so this is also a parity gap.
  - **Windows/Linux LTX gives misleading errors for prefix-less or kohya keys:** "Adapter name(s) … not in the list of present adapters", or at a weight other than 1.0 "does not support explicit LoRA scales" (`backends/cuda/video/backend.py::_apply_loras`).
  - **Zero-match and partial-match warnings never reach the job card.** `web/web_runner.py` doesn't capture warnings.
  - **The docs over-claim.** `docs/concepts/loras.md` says an unsupported file or scale "fails before generation with a message saying why".

  Fix: one clear error per case, using the loader's own verdict, surfaced on the job card; correct the docs.

## Code hygiene (no user impact)

- `backends/__init__.py` `__all__` omits `convert_h3_checkpoint`, `supports_video_audio`, `supports_video_extend` and `supports_video_file_upscale`, which other modules import.
- Dead config key `video_model_presets.h3.diffusers.attention` (`config.yaml`, `utils/h3_preset.py`): parsed and validated, never read.
- Dead code: `utils/h3_memory.h3_extend_window_frames` (duplicates `video_extend.window_frames`), `web_runner.WebRunner.get_job_result_path`, `job_contract.ACTIVE_STATUSES`, the job snapshot field `supports_controls` (duplicates `supported_controls`), `backends/cuda/image/backend._make_skip_callback`, the `GET /docs/assets/{asset_name}` route (always 404 once installed), the frontend `progress_text` SSE handler, `draft.reset()`.
- Duplicated helpers: `backends/mac/video.py` `_snapshot_download` repeats `utils/hf_snapshot.snapshot_download`; `workflows/video_stages.py` handles a `CalledProcessError` that `splice_clips` can never raise while `strip_audio` and `embed_mp4_config` still raise it.
- `backends/cuda/video/h3/lora.py` docstring mentions a planning label ("C11's `load_copy`"); `h3/conditioning.py` imports the private `_weight_files` from `decode.py`.
- The Windows CUDA install command is hand-copied into `docs/install.md` while the code carries it as `CUDA_INSTALL_COMMAND` in `backends/cuda/driver.py`; a later index change (cu130 to cu131) would leave the error message and the docs apart. A test in `tests/common/test_docs_reference.py` asserting the docs contain the constant would tie them. Decision: skipped for v1.
- Docs: `docs/developer/downloads.md` names `ziv convert --model-type h3`; the command is `ziv model model --model-type h3`. `docs/install.md` says Apple Silicon (M1 or newer) but the locked mlx and torch wheels need macOS 14. `core/video_backend.py` says `mode` is accepted and ignored for LTX; the CUDA LTX backend raises on a mismatched mode.
- `ltx25-h3-checkpoint-conversion.md` stays open until its two macOS findings are settled: W4A8 files that keep their records in the safetensors metadata, and the LTX connectors' design rule.

## Accepted as is

- `--host local` has no login; anyone on the network can use the UI and its file routes, including reading any YAML through the prompt-file document route. Communicated in [Other devices](../docs/web-ui/devices.md).
- A hand-broken `config.yaml`, or an output folder on an unplugged drive, makes every page answer a bare "Internal Server Error". Judged not worth mitigating.
