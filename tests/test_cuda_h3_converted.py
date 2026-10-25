"""Tests for converted MiniMax H3 folders on Windows/Linux: what the writer records and how the backend loads one."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from tests.test_h3_checkpoint import _TINY, _diffusers_state, _native
from zvisiongenerator.backends.cuda.video.h3 import backend as h3_backend
from zvisiongenerator.backends.cuda.video.h3 import converted
from zvisiongenerator.converters import h3_checkpoint
from zvisiongenerator.converters.h3_checkpoint import inspect_h3
from zvisiongenerator.utils.h3_converted import read_converted_h3, write_converted_h3

_KEYS = [(0.5,), (0.25, 1.0)]


@pytest.fixture(autouse=True)
def _nf4_format(monkeypatch):
    """Pin the NF4 storage format, so the tests run without bitsandbytes installed (as on macOS)."""
    monkeypatch.setattr(converted.stored_copy, "nf4_storage_format", lambda: "nf4-bitsandbytes-0.50")


def _copy() -> dict:
    return converted.stored_copy.copy_settings(0.0625, _KEYS)


def _current() -> dict:
    return {"storage": "nf4-folded", "turbo": "merged", "copy": _copy()}


@pytest.fixture
def checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(h3_checkpoint, "STOCK_CONFIG", _TINY)
    path = tmp_path / "ft.safetensors"
    save_file(_native(_diffusers_state(torch.Generator().manual_seed(0)), interleave=False), str(path))
    return inspect_h3(path)


def _preset() -> SimpleNamespace:
    return SimpleNamespace(schedules={8: object()}, lora=SimpleNamespace(repo_id="org/turbo", revision="abc", files=("turbo.safetensors",)), lora_scale=0.0625)


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "repo" / "transformer").mkdir(parents=True)
    (tmp_path / "repo" / "transformer" / "config.json").write_text("{}")
    return tmp_path / "repo"


@pytest.mark.parametrize("merge", [True, False])
def test_the_manifest_records_the_turbo_choice_the_fold_keys_and_the_source(monkeypatch, tmp_path, checkpoint, merge):
    written: dict = {}
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.h3.adaln.fold_keys", lambda schedules, repo_dir: _KEYS)
    monkeypatch.setattr(converted.h3_lora, "load_pairs", lambda path: {"module": "pair"})
    monkeypatch.setattr(converted.stored_copy, "write_folded", lambda tensors, config, turbo, keys, target, device, *, label: written.update(turbo=turbo, keys=keys, label=label))
    target = tmp_path / "out"
    target.mkdir()

    converted.write_converted(
        checkpoint, target, preset=_preset(), repo_dir=_repo(tmp_path), lora_file=tmp_path / "turbo.safetensors" if merge else None, time_embedder=None, device="cpu", report=lambda *_: None
    )

    manifest = read_converted_h3(target)
    assert written["turbo"] == (({"module": "pair"}, 0.0625) if merge else None)
    assert (manifest["storage"], manifest["turbo"]) == ("nf4-folded", "merged" if merge else "distilled")
    assert manifest["copy"] == converted.stored_copy.copy_settings(0.0625 if merge else None, _KEYS)
    assert manifest["source"] == "ft.safetensors"


def test_a_converted_folder_loads_its_own_copy_with_stocks_components(monkeypatch, tmp_path):
    folder = tmp_path / "myh3"
    folder.mkdir()
    write_converted_h3(folder, _current())
    files = SimpleNamespace(repo_dir=tmp_path / "repo")
    monkeypatch.setattr(h3_backend, "resolve_base_files", lambda preset, upscale: files)
    monkeypatch.setattr(h3_backend, "_all_fold_keys", lambda preset, repo_dir: _KEYS)

    assert h3_backend._converted_copy(folder, read_converted_h3(folder), _preset(), False) == (files, folder)


@pytest.mark.parametrize(
    "changes",
    [
        {"storage": "mlx-int8"},  # converted on a Mac
        {"copy": {"fold_keys": [[0.5]]}},  # other schedules
        {"copy": {"lora_scale": 0.125}},  # another turbo strength
        {"copy": None},  # no record of the copy's settings
    ],
)
def test_a_folder_from_the_other_platform_or_with_other_copy_settings_is_refused(monkeypatch, tmp_path, changes):
    folder = tmp_path / "myh3"
    folder.mkdir()
    manifest = _current()
    if "storage" in changes:
        manifest["storage"] = changes["storage"]
    if "copy" in changes:
        if changes["copy"] is None:
            del manifest["copy"]
        else:
            manifest["copy"].update(changes["copy"])
    write_converted_h3(folder, manifest)
    monkeypatch.setattr(h3_backend, "resolve_base_files", lambda preset, upscale: SimpleNamespace(repo_dir=tmp_path))
    monkeypatch.setattr(h3_backend, "_all_fold_keys", lambda preset, repo_dir: _KEYS)

    with pytest.raises(RuntimeError):
        h3_backend._converted_copy(folder, read_converted_h3(folder), _preset(), False)


def test_a_new_nf4_serialization_asks_for_a_new_conversion(monkeypatch, tmp_path):
    folder = tmp_path / "myh3"
    folder.mkdir()
    write_converted_h3(folder, _current())
    monkeypatch.setattr(h3_backend, "resolve_base_files", lambda preset, upscale: SimpleNamespace(repo_dir=tmp_path))
    monkeypatch.setattr(h3_backend, "_all_fold_keys", lambda preset, repo_dir: _KEYS)
    monkeypatch.setattr(converted.stored_copy, "nf4_storage_format", lambda: "nf4-bitsandbytes-99.0")

    with pytest.raises(RuntimeError):
        h3_backend._converted_copy(folder, read_converted_h3(folder), _preset(), False)


def test_a_converted_folder_plans_stocks_components_and_no_turbo_lora(tmp_path):
    from zvisiongenerator.backends.cuda.download_plan import h3_download_plan
    from zvisiongenerator.utils.h3_preset import HubFiles

    folder = tmp_path / "myh3"
    folder.mkdir()
    write_converted_h3(folder, {"storage": "nf4-folded", "turbo": "merged"})
    preset = SimpleNamespace(
        repo="org/h3",
        revision="r1",
        repo_allow=("transformer/*", "vae/*"),
        repo_ignore=(),
        text_encoder=HubFiles("org/te", "r2", ("a.safetensors",)),
        lora=HubFiles("org/turbo", "r3", ("t.safetensors",)),
        latent_upsampler=HubFiles("org/up", "r4", ("u.safetensors",)),
    )

    targets = h3_download_plan(str(folder), preset)

    assert {target.repo_id for target in targets} == {"org/h3", "org/te"}
    assert "transformer/*" not in next(target for target in targets if target.repo_id == "org/h3").allow_patterns
