"""Tests for loading image models through stored quants."""

from __future__ import annotations

import json
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
        assert plan_image_model_load(str(source), 4, models_dir=models_dir, backend_format=FORMAT) == LoadPlan(str(source), 4, create=models_dir / "atlas@q4")


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
        gate = threading.Event()
        backend = _FakeBackend(save_gate=gate)

        try:
            load_image_model(backend, str(source), quantize=8, models_dir=models_dir, lora_paths=["style.safetensors"], cancelled=lambda: True)
        finally:
            gate.set()

        assert backend.loads == [{"path": str(source), "quantize": 8, "lora_paths": None}]


class TestSaveCancellation:
    def test_cancel_abandons_save_and_removes_partial_folder(self, models_dir):
        source = _make_source(models_dir)
        gate = threading.Event()
        backend = _FakeBackend(save_gate=gate)
        target = models_dir / "atlas@q8"

        started = time.monotonic()
        saved = save_stored_quant(backend, MagicMock(), source=source, target=target, bits=8, backend_format=FORMAT, cancelled=lambda: True, poll_seconds=0.01)
        assert saved is False
        assert time.monotonic() - started < 2  # returned without waiting for the write

        gate.set()
        assert backend.save_finished.wait(5)
        deadline = time.monotonic() + 5
        while _partials(models_dir) and time.monotonic() < deadline:
            time.sleep(0.01)

        assert _partials(models_dir) == []
        assert not target.exists()
