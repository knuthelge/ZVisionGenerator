"""Tests for loading image models through stored quants."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from zvisiongenerator.image_model_loader import SAVING_QUANT_PHASE, LoadPlan, load_image_model, plan_image_model_load, save_stored_quant
from zvisiongenerator.utils.stored_quant import build_manifest, is_current, write_manifest

FORMAT = "fake-1"


class _FakeBackend:
    """Records loads and writes a minimal stored quant on save."""

    name = "fake"

    def __init__(self, *, backend_format: str | None = FORMAT, save_error: BaseException | None = None, save_gate: threading.Event | None = None):
        self._format = backend_format
        self._save_error = save_error
        self._save_gate = save_gate
        self.loads: list[dict] = []
        self.saved: list[str] = []
        self.save_finished = threading.Event()

    def stored_quant_format(self) -> str | None:
        return self._format

    def load_model(self, model_path, quantize=None, precision="bfloat16", lora_paths=None, lora_weights=None):
        self.loads.append({"path": model_path, "quantize": quantize, "lora_paths": lora_paths})
        return MagicMock(name=f"model-{len(self.loads)}"), "info"

    def save_quantized(self, model, path):
        try:
            if self._save_gate is not None:
                self._save_gate.wait(5)
            if self._save_error is not None:
                raise self._save_error
            (Path(path) / "transformer").mkdir(parents=True)
            (Path(path) / "transformer" / "0.safetensors").write_bytes(b"q")
            self.saved.append(path)
        finally:
            self.save_finished.set()


def _make_source(models_dir: Path, name: str = "atlas") -> Path:
    source = models_dir / name
    (source / "transformer").mkdir(parents=True)
    (source / "model_index.json").write_text(json.dumps({"transformer": ["diffusers", "Flux2Transformer"]}))
    (source / "transformer" / "config.json").write_text("{}")
    (source / "transformer" / "weights.safetensors").write_bytes(b"x" * 32)
    return source


def _partials(models_dir: Path) -> list[Path]:
    return [path for path in models_dir.iterdir() if path.name.endswith(".partial")]


@pytest.fixture
def models_dir(tmp_path):
    path = tmp_path / "models"
    path.mkdir()
    return path


class TestPlan:
    def test_no_quantize_loads_source(self, models_dir):
        source = _make_source(models_dir)
        assert plan_image_model_load(str(source), None, models_dir=models_dir, backend_format=FORMAT) == LoadPlan(str(source), None)

    def test_unsupported_backend_quantizes_at_load(self, models_dir):
        source = _make_source(models_dir)
        assert plan_image_model_load(str(source), 8, models_dir=models_dir, backend_format=None) == LoadPlan(str(source), 8)

    def test_model_outside_models_dir_quantizes_at_load(self, tmp_path, models_dir):
        source = _make_source(tmp_path / "elsewhere")
        assert plan_image_model_load(str(source), 8, models_dir=models_dir, backend_format=FORMAT) == LoadPlan(str(source), 8)

    def test_repo_id_quantizes_at_load(self, models_dir):
        assert plan_image_model_load("org/repo", 8, models_dir=models_dir, backend_format=FORMAT) == LoadPlan("org/repo", 8)

    def test_selected_stored_quant_loads_unquantized(self, models_dir):
        stored = models_dir / "atlas@q8"
        stored.mkdir()
        assert plan_image_model_load(str(stored), 4, models_dir=models_dir, backend_format=FORMAT) == LoadPlan(str(stored), None)

    def test_current_stored_quant_is_used(self, models_dir):
        source = _make_source(models_dir)
        stored = models_dir / "atlas@q8"
        stored.mkdir()
        write_manifest(stored, build_manifest(source, 8, FORMAT))

        assert plan_image_model_load(str(source), 8, models_dir=models_dir, backend_format=FORMAT) == LoadPlan(str(stored), None)

    def test_missing_stored_quant_is_created(self, models_dir):
        source = _make_source(models_dir)
        assert plan_image_model_load(str(source), 4, models_dir=models_dir, backend_format=FORMAT) == LoadPlan(str(source), 4, create=models_dir / "atlas@q4", source=source)


class TestAliasPlan:
    """Hugging Face models picked through an alias store their quant under the alias name."""

    def test_downloaded_alias_creates_a_copy_named_after_the_alias(self, tmp_path, models_dir):
        snapshot = _make_source(tmp_path / "hf-cache", "snapshot")
        plan = plan_image_model_load("org/repo", 8, models_dir=models_dir, backend_format=FORMAT, model_name="zit", find_local_dir=lambda _ref: snapshot)

        assert plan == LoadPlan("org/repo", 8, create=models_dir / "zit@q8", source=snapshot)

    def test_current_alias_copy_is_used(self, tmp_path, models_dir):
        snapshot = _make_source(tmp_path / "hf-cache", "snapshot")
        stored = models_dir / "zit@q8"
        stored.mkdir()
        write_manifest(stored, build_manifest(snapshot, 8, FORMAT))

        plan = plan_image_model_load("org/repo", 8, models_dir=models_dir, backend_format=FORMAT, model_name="zit", find_local_dir=lambda _ref: snapshot)

        assert plan == LoadPlan(str(stored), None)

    def test_new_revision_makes_the_alias_copy_stale(self, tmp_path, models_dir):
        old_snapshot = _make_source(tmp_path / "hf-cache", "old")
        new_snapshot = _make_source(tmp_path / "hf-cache", "new")
        (new_snapshot / "transformer" / "weights.safetensors").write_bytes(b"y" * 64)
        stored = models_dir / "zit@q8"
        stored.mkdir()
        write_manifest(stored, build_manifest(old_snapshot, 8, FORMAT))

        plan = plan_image_model_load("org/repo", 8, models_dir=models_dir, backend_format=FORMAT, model_name="zit", find_local_dir=lambda _ref: new_snapshot)

        assert plan.create == stored

    def test_alias_not_downloaded_yet_plans_a_copy_without_a_source(self, models_dir):
        plan = plan_image_model_load("org/repo", 4, models_dir=models_dir, backend_format=FORMAT, model_name="zit", find_local_dir=lambda _ref: None)

        assert plan == LoadPlan("org/repo", 4, create=models_dir / "zit@q4", source=None)

    def test_unmapped_name_is_a_raw_path_not_an_alias(self, tmp_path, models_dir):
        """A bare folder name in the current directory resolves to itself and gets no copy."""
        folder = _make_source(tmp_path, "mymodel")
        plan = plan_image_model_load("mymodel", 8, models_dir=models_dir, backend_format=FORMAT, model_name="mymodel", find_local_dir=lambda _ref: folder)

        assert plan == LoadPlan("mymodel", 8)

    @pytest.mark.parametrize("picked", [None, "org/repo", "~/models/x", "C:model", "..", "zit@q8"])
    def test_raw_repo_ids_and_paths_are_quantized_at_load(self, models_dir, picked):
        plan = plan_image_model_load("org/repo", 8, models_dir=models_dir, backend_format=FORMAT, model_name=picked, find_local_dir=lambda _ref: pytest.fail("not resolved"))

        assert plan == LoadPlan("org/repo", 8)


class TestLoad:
    def test_first_use_saves_and_keeps_the_loaded_model(self, models_dir):
        source = _make_source(models_dir)
        backend = _FakeBackend()
        phases: list[str] = []

        model, info = load_image_model(backend, str(source), quantize=8, models_dir=models_dir, on_phase=phases.append)

        assert backend.loads == [{"path": str(source), "quantize": 8, "lora_paths": None}]
        assert phases == [SAVING_QUANT_PHASE]
        stored = models_dir / "atlas@q8"
        assert is_current(stored, source, 8, FORMAT)
        assert (stored / "model_index.json").is_file()
        assert _partials(models_dir) == []
        assert info == "info"

    def test_first_use_with_loras_saves_lora_free_weights_then_loads_loras_on_the_copy(self, models_dir):
        source = _make_source(models_dir)
        backend = _FakeBackend()
        release = MagicMock()

        load_image_model(backend, str(source), quantize=8, models_dir=models_dir, lora_paths=["style.safetensors"], lora_weights=[1.0], release_memory=release)

        assert backend.loads == [
            {"path": str(source), "quantize": 8, "lora_paths": None},
            {"path": str(models_dir / "atlas@q8"), "quantize": None, "lora_paths": ["style.safetensors"]},
        ]
        release.assert_called_once()

    def test_existing_stored_quant_loads_with_loras_and_never_saves(self, models_dir):
        source = _make_source(models_dir)
        stored = models_dir / "atlas@q8"
        stored.mkdir()
        write_manifest(stored, build_manifest(source, 8, FORMAT))
        backend = _FakeBackend()
        phases: list[str] = []

        load_image_model(backend, str(source), quantize=8, models_dir=models_dir, lora_paths=["style.safetensors"], on_phase=phases.append)

        assert backend.loads == [{"path": str(stored), "quantize": None, "lora_paths": ["style.safetensors"]}]
        assert backend.saved == []
        assert phases == []

    def test_no_quantize_never_creates_a_copy(self, models_dir):
        source = _make_source(models_dir)
        backend = _FakeBackend()

        load_image_model(backend, str(source), quantize=None, models_dir=models_dir)

        assert backend.loads == [{"path": str(source), "quantize": None, "lora_paths": None}]
        assert not (models_dir / "atlas@q8").exists()

    def test_failed_save_warns_and_falls_back_to_quantizing_at_load(self, models_dir):
        source = _make_source(models_dir)
        backend = _FakeBackend(save_error=OSError("disk full"))

        with pytest.warns(UserWarning, match="disk full"):
            load_image_model(backend, str(source), quantize=8, models_dir=models_dir, lora_paths=["style.safetensors"])

        assert backend.loads[-1] == {"path": str(source), "quantize": 8, "lora_paths": ["style.safetensors"]}
        assert not (models_dir / "atlas@q8").exists()
        assert _partials(models_dir) == []

    def test_stop_during_save_skips_the_lora_reload(self, models_dir):
        source = _make_source(models_dir)
        backend = _FakeBackend()

        load_image_model(backend, str(source), quantize=8, models_dir=models_dir, lora_paths=["style.safetensors"], cancelled=lambda: True)

        assert backend.loads == [{"path": str(source), "quantize": 8, "lora_paths": None}]


class TestAliasLoad:
    def test_first_use_downloads_then_saves_under_the_alias(self, tmp_path, models_dir):
        snapshot = _make_source(tmp_path / "hf-cache", "snapshot")
        downloaded: dict[str, Path | None] = {"dir": None}
        backend = _FakeBackend()
        original_load = backend.load_model

        def _load_and_download(*args, **kwargs):
            downloaded["dir"] = snapshot
            return original_load(*args, **kwargs)

        backend.load_model = _load_and_download
        phases: list[str] = []

        load_image_model(backend, "org/repo", quantize=8, models_dir=models_dir, model_name="zit", on_phase=phases.append, find_local_dir=lambda _ref: downloaded["dir"])

        assert phases == [SAVING_QUANT_PHASE]
        assert is_current(models_dir / "zit@q8", snapshot, 8, FORMAT)

    def test_no_local_files_after_load_skips_the_save(self, models_dir):
        backend = _FakeBackend()
        phases: list[str] = []

        load_image_model(backend, "org/repo", quantize=8, models_dir=models_dir, model_name="zit", on_phase=phases.append, find_local_dir=lambda _ref: None)

        assert backend.saved == []
        assert phases == []
        assert not (models_dir / "zit@q8").exists()


class TestSaveStoredQuant:
    def test_stop_during_save_discards_the_copy_after_the_write(self, models_dir):
        source = _make_source(models_dir)
        backend = _FakeBackend()
        target = models_dir / "atlas@q8"

        saved = save_stored_quant(backend, MagicMock(), source=source, target=target, bits=8, backend_format=FORMAT, cancelled=lambda: True)

        assert saved is False
        assert backend.saved  # the write ran to completion before the stop took effect
        assert _partials(models_dir) == []
        assert not target.exists()

    def test_interrupt_during_save_discards_the_partial_and_propagates(self, models_dir):
        source = _make_source(models_dir)
        backend = _FakeBackend(save_error=KeyboardInterrupt())

        with pytest.raises(KeyboardInterrupt):
            save_stored_quant(backend, MagicMock(), source=source, target=models_dir / "atlas@q8", bits=8, backend_format=FORMAT)

        assert _partials(models_dir) == []

    def test_old_partials_from_an_interrupted_save_are_swept(self, models_dir):
        source = _make_source(models_dir)
        stale = models_dir / ".atlas@q8.deadbeef.partial"
        fresh = models_dir / ".atlas@q8.cafebabe.partial"
        other = models_dir / ".other@q8.deadbeef.partial"
        for folder in (stale, fresh, other):
            folder.mkdir()
        old = time.time() - 3600
        os.utime(stale, (old, old))
        os.utime(other, (old, old))

        assert save_stored_quant(_FakeBackend(), MagicMock(), source=source, target=models_dir / "atlas@q8", bits=8, backend_format=FORMAT)

        assert not stale.exists()
        assert fresh.exists()  # may belong to a save still running in another process
        assert other.exists()
