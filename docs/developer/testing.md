# Testing

The test suite runs on any machine, with or without a GPU, with no network and no ffmpeg. It never loads a real model. A few guards in `tests/conftest.py` make that true even when the code under test would otherwise reach out.

## Which check to run

| You changed | Run |
|---|---|
| One module | `uv run pytest tests/<area>/test_<module>.py -k <name>` |
| A layer | `uv run pytest tests/web` (or another area folder) |
| Python style | `make lint-fix && make format` |
| `frontend/` | `make frontend-test`, then `make frontend-build` and commit the output |
| Docs | `make docs-check` |
| A feature that spans layers | `uv run pytest tests -k h3` |
| Anything, before you finish | `make check` |
| A backend or a model, for real | an [end-to-end target](#end-to-end-runs) |

`make check` is the full CI gate: ruff lint and format check, pytest (`make test`, one worker per core), the frontend type checks and Vitest, the palette, hover and packaged-SPA drift checks, and a strict docs build.

## Layout

Tests live in `tests/<area>/test_<module>.py`, with functions named `test_<behavior>` and related tests grouped in `Test`-prefixed classes. Frontend tests sit next to their source as `*.test.ts`.

The areas are layers, not features:

| Area | Covers |
|---|---|
| `tests/web` | The server, its routes, the job queue, the gallery and the stores behind them |
| `tests/video` | Video runners, stages, frame rules, clip splicing, family capabilities |
| `tests/image` | Image runners, stages, upscaling, background removal, model detection |
| `tests/prompts` | Prompt documents, composition, the enhancer's requests and parsing |
| `tests/downloads` | Planners, the dry run, the child process, flights, the lane's collaborators |
| `tests/converters` | `ziv-model` and the checkpoint converters |
| `tests/contracts` | Pins on vendored and upstream code, and on the packaged data files |
| `tests/backends` | Backends, mirroring the package: `cuda/` and `mac/` beside the shared ones |
| `tests/audio` | The pure helpers and presets of the audio models |
| `tests/common` | Cross-cutting modules: config, metadata, paths, provenance, LoRA files, the docs checks |

Put a new file in the folder of the layer its subject module lives in. A folder needs at least four files, so anything smaller goes to `common`. A feature such as H3 or MOSS spans several layers; select it with `-k`.

## Platforms

CI runs `make check` on a macOS runner (`.github/workflows/ci.yml`). Tests that need mflux or MLX, which only a Mac installs, run there and skip themselves on Windows and Linux; a few tests that need Windows skip on the Mac. So a green `make check` on Linux has skipped a different set from the one CI runs: a change under `backends/mac/` is exercised only by CI or by your own Mac, and a change under `backends/cuda/` runs in the mocked tests everywhere. List what skipped with `uv run pytest -rs`, and run a backend for real on each platform you can, as [End-to-end runs](#end-to-end-runs) describes.

## Rules

- **Import shared code as `tests.<module>`.** `tests` and every area folder are packages. Never `from conftest import ...` and never import one test module from another. Plain builders used in more than one area live in `tests/_factories.py`, other cross-area helpers in `tests/_*.py` at the root, and a helper for one area in `tests/<area>/_helpers.py`.
- **Resolve repository paths through `tests/_paths.py`** (`REPO_ROOT`, `FIXTURES`), never from `__file__`.
- **Never load real backends or models.** Use `MagicMock` with controlled return values, for example a backend whose `.text_to_image.return_value` is a small `Image.new("RGB", ...)` (see `_make_mock_backend()` in `tests/image/test_workflow.py`). Build CLI arguments with `tests/_factories.py::_make_args(**overrides)`.
- **Test behavior and machine-readable contracts**: routes, config keys, storage keys, parser options, event names, file names, enum values, statuses, and the accessible names a control needs. Not prose, help text, CSS classes, source text or incidental selectors.
- Use `tmp_path` for files and `@pytest.mark.parametrize` for many input and output pairs. Skip heavy dependencies with `pytest.importorskip("torch")`, and platform-specific tests with `@pytest.mark.skipif(sys.platform == ...)`.
- `packaged_config` is a private copy of the packaged config a test may change; `packaged_config_shared` is a read-only session copy. Outside a fixture, `tests._factories.load_packaged_config()` returns the same copy.

## Protective fixtures

All of these are autouse in `tests/conftest.py`, so a new test is protected without asking.

| Guard | What it keeps out |
|---|---|
| `_isolated_app_db` | The real `ziv.sqlite3`, the job log included. Files a test creates are under 2 seconds old, which the gallery index never persists; to test caching, backdate them with `os.utime` (or pass `clock=` to `GalleryIndex`) and count reader calls |
| `_presets_in_tmp`, `_voices_in_tmp` | The real `~/.ziv/presets`, `~/.ziv/voices` and `~/.ziv/voice-design` |
| `_no_hub_downloads` | `huggingface_hub.hf_hub_download` and `snapshot_download`. A call fails the test afterwards, even when the code under test swallows the error. Use the `hub_offline` fixture for a test whose subject is "nothing cached, no network", or mock the function |
| `_no_hugging_face_token` | Your real Hugging Face token |
| `_no_media_tools` | `ffmpeg` and `ffprobe`: both are hidden from `shutil.which` and any start fails the test, so every machine behaves like CI, which has none. Mock the seam instead: `tests/_fake_media.py` has fake processes and PCM and frame bytes, and `tests/_fake_audio.py` stands in for saving, probing and reading takes. Test the commands we build, how we parse given output and the error paths, not what ffmpeg makes of them |
| `CUDA_VISIBLE_DEVICES=""` | The GPU. It is assigned for the whole run, workers included, so `torch.cuda.is_available()` is `False` and no test opens a CUDA context; mock `is_available` when a test needs it true. `tests/common/test_gpu_hidden.py` fails if the GPU becomes visible |
| `_no_web_download_planning` | Download planners for web jobs (`@pytest.mark.real_download_plans` keeps the real ones) |
| `_seedvr2_off_in_web_config` | Platform differences in the Web UI config: SeedVR2 is off unless a test turns it on |
| `_model_metadata_cache_unchanged` | Writes into the shared parse of `utils/model_metadata.yaml`: the run fails at the end if the cached parse differs from a fresh one. Change facts through `metadata_copy()` and `use_model_metadata` in `tests/_factories.py` |

Two more rules the guards cannot enforce:

- **Never run the real memory cleanup.** Patch `zvisiongenerator.preflight.release_accelerator_memory` (and `zvisiongenerator.web.web_runner.release_accelerator_memory` for web jobs), or pass `release_memory=` to `run_preflight`.
- **Download tests use fakes.** `check_target` and the worker take a fake `snapshot_download`, `run_in_child` takes a fake `popen`, and `ensure_downloaded` and `DownloadLane` take fake `present`, `check`, `run` and `ensure` callables and a fake clock.

Diffusers backend tests patch the lazy runtime loader, the torch CUDA checks, the upsampler download and the export helpers, so nothing instantiates a model. Pattern pins that need mflux, mlx-lm or diffusers skip themselves where those are not installed.

## Contract tests

`tests/contracts/` pins what we rely on but do not own: the names and signatures of vendored code and of upstream libraries, read by introspection or as a syntax tree when the library is not installed (Linux has no MLX). When a bump of mflux, mlx-lm, diffusers or a vendored package moves something we call, the contract test fails first and names it. Re-run the relevant ones on every such bump; [Vendored packages](vendored.md) says which.

The contracts directory also holds the checks of the packaged data: model metadata, curated LoRAs and the voice preview clips.

## Shared fixtures

Some fixtures are read by Python and by Vitest, so the two sides cannot drift. Change both when you change the rule:

- `tests/fixtures/audio_cli_cases.json` and `audio_settings_cases.json`: the audio settings the Web UI form and `ziv audio` accept, and the command the Web UI copies.
- `tests/fixtures/frame_point_cases.json`: how a time in a clip becomes a frame (`utils/video_point.py` and `frontend/src/lib/state/framePoint.ts`).
- `tests/audio/music_mode_cases.json`: the song-or-instrumental rule (`utils/music_mode.py` and `frontend/src/lib/state/musicMode.ts`).

## End-to-end runs

`make e2e` runs one tiny generation per model family through the real CLI and writes the results to `output/`. These load real models, so they need the hardware and the downloaded weights, and they are not part of `make check`.

| Target | Model variable | Default |
|---|---|---|
| `make e2e-image-zimage` | `ZIMAGE_MODEL` | `zit` |
| `make e2e-image-klein` | `KLEIN_MODEL` | `klein9b` |
| `make e2e-image-ideogram` | `IDEOGRAM_MODEL` | `ideo` |
| `make e2e-video-ltx` | `LTX_MODEL` | `ltx-8` |

`make e2e-image` and `make e2e-video` run the group; `make e2e` runs everything. Override a variable to run a different alias, for example a smaller model on a card with less memory:

```bash
make e2e-image-klein KLEIN_MODEL=klein4b
make e2e-video-ltx LTX_MODEL=ltx-4
```

There is no audio end-to-end target. Check an audio model by hand with `uv run ziv-audio`, and see [Add a built-in model](how-to.md#add-a-built-in-model) for measuring its memory.

The [prompt-enhancer bench](enhancer-bench.md) is the other check that loads a real model.
