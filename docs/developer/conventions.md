# Conventions

The rules every change follows. `AGENTS.md` in the repository holds the same rules for coding agents.

## Design: atomic units

Every module, function and class should be a single, self-contained unit of responsibility that can be understood, tested and replaced on its own.

- **One purpose per unit.** Each function does one thing; each module owns one concept. If a docstring needs "and", split the unit.
- **Small surface.** Expose the minimum. Keep helpers private (`_`-prefixed); public symbols go through `__init__.py` re-exports and `__all__`.
- **Pure over stateful.** Prefer functions that take inputs and return outputs. Push side effects (I/O, logging, mutation) to the edges.
- **Composition over inheritance.** Stages, processors and utilities are composed into pipelines, not inherited from.
- **Testable in isolation.** If a unit can't be tested without standing up half the system, it is too coupled.
- **Explicit dependencies.** Pass config, backends and I/O handles in as arguments rather than importing global state.
- **Reuse over duplication.** Check `utils/` and existing helpers before writing new code.

## Python

- Python 3.14+. Start every `.py` file with `from __future__ import annotations`.
- Modern type syntax: `str | None`, `list[str]`, `dict[str, Any]`. Never `Optional`, `Union`, `List`, `Dict` or `Tuple` from `typing`.
- Imports: standard library, third-party, then `zvisiongenerator.*`, with a blank line between groups.
- Guard heavy imports (torch, mflux, diffusers) with `TYPE_CHECKING`, or import them lazily.
- Line length 200, enforced by ruff.
- Google-style docstrings. Module docstrings on every non-trivial module. The first line is an imperative fragment; longer ones use `Args:`, `Returns:` and `Raises:`.

## Platform and backends

Backend and accelerator selection happens only in `backends/__init__.py`. Other code never branches on `sys.platform` to pick a backend; it asks a capability function such as `supports_video_extend()`. OS-level checks unrelated to backends (drive listings, ffmpeg install hints, terminal handling) are fine where they're needed. See [Backends](backends.md).

## Data types

`@dataclass(frozen=True)` for value objects (inputs, detection results). Plain `@dataclass` only for working state. No pydantic or attrs.

## Config

Config is a plain `dict` loaded from YAML, not a dataclass. Precedence: CLI flags, then the model preset's variant, then the model preset's family, then global defaults. See [Configuration](../concepts/configuration.md).

## Errors

Raise `ValueError`, `FileNotFoundError` or `RuntimeError` with a descriptive message. Use `warnings.warn(..., stacklevel=2)` for non-fatal conditions. Add a custom exception only when callers must catch it distinctly, such as `JobConflictError` in `web/web_runner.py`, or as a private sentinel.

## Tests

- Tests live in `tests/` as `test_<module>.py`, with functions named `test_<behavior>`. Group related tests in `Test`-prefixed classes. Frontend tests sit next to their source as `*.test.ts`.
- **Never load real backends or models.** Use `unittest.mock.MagicMock` with controlled return values, for example a backend whose `.text_to_image.return_value` is `Image.new("RGB", (64, 64))` (see `_make_mock_backend()` in `tests/test_workflow.py`).
- Build CLI arguments with `tests/_factories.py::_make_args(**overrides)`.
- Use the `tmp_path` fixture for files, and `@pytest.mark.parametrize` for many input/output pairs.
- Skip heavy dependencies with `pytest.importorskip("torch")`, and platform-specific tests with `@pytest.mark.skipif(sys.platform == ...)`.
- **Test behavior and machine-readable contracts**: routes, config keys, storage keys, parser options, event names, file names, enum values, statuses, and accessible names a control needs. Not prose, help text, CSS classes, source text or incidental selectors.

### Fixtures that protect your data

- `_isolated_app_db` (autouse) points the app database at a per-test folder, so no test touches the real `ziv.sqlite3`, the job log included. Files a test creates are under 2 seconds old, and the gallery index never persists those; to test caching, backdate them with `os.utime` (or pass `clock=` to `GalleryIndex`) and count reader calls.
- `_presets_in_tmp` (autouse) points the presets folder at a temporary data folder, so no test reads or writes the real `~/.ziv/presets`.
- Never run the real memory cleanup: patch `zvisiongenerator.preflight.release_accelerator_memory` (and `zvisiongenerator.web.web_runner.release_accelerator_memory` for web jobs), or pass `release_memory=` to `run_preflight`.

### Mocking heavy dependencies

- Diffusers backend tests patch the lazy runtime loader, torch CUDA checks, upsampler download and loading, export helpers and image loading, so the suite never instantiates a model, downloads weights or needs a GPU.
- Platform dispatch, alias resolution and Web inventory tests assert behavior through config and protocol boundaries, not backend internals.
- Download tests never use the network: `check_target` and the worker take a fake `snapshot_download`, `run_in_child` takes a fake `popen`, and `ensure_downloaded` and `DownloadLane` take fake `present`, `check`, `run` and `ensure` callables and a fake clock.
- Pattern pins that need mflux, mlx-lm or diffusers skip themselves where those aren't installed.
- An autouse guard fails any test that reaches `huggingface_hub.hf_hub_download` or `snapshot_download` unmocked, even when the code under test swallows the error. Use the `hub_offline` fixture for a test whose subject is "nothing cached, no network", or mock the function in the test.

### Tests that need tools

Tests that run the real `ffmpeg` and `ffprobe` (`tests/test_video_probe.py`, `tests/test_video_splice.py`, `tests/test_video_tail.py`) skip when the tools are missing. Run them locally with ffmpeg installed before changing how extensions are joined.

## Web UI

The frontend has its own conventions for confirmations, toasts, requests and its design system. See [Frontend](frontend.md).
