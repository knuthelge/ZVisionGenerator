"""Tests for the LoRA files the mflux backend hands to mflux: renamed copies for files whose names mflux cannot match."""

from __future__ import annotations

import contextlib
import sys
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="mflux/MLX backend is macOS-only")


@pytest.fixture
def image_mac(monkeypatch):
    pytest.importorskip("mflux")
    pytest.importorskip("mlx.core")
    import zvisiongenerator.backends.mac.image as module

    monkeypatch.setattr(module.ModelConfig, "precision", module.ModelConfig.precision)
    monkeypatch.setattr(module, "mx", MagicMock(name="mx"))
    monkeypatch.setattr(module, "load_resident", MagicMock(name="load_resident", side_effect=lambda build: build()))
    monkeypatch.setattr(module, "_upcast_model_weights", MagicMock())
    return module


def _fake_copies(monkeypatch, module, events):
    """Replace ``readable_lora_files`` with a recorder: each path ending ``peft.safetensors`` becomes a ``.renamed`` copy while the block is open."""

    @contextlib.contextmanager
    def readable(paths, model_family, loader):
        events.append(("open", list(paths), model_family, loader))
        try:
            yield [f"{path}.renamed" if path.endswith("peft.safetensors") else path for path in paths]
        finally:
            events.append(("close",))

    monkeypatch.setattr(module, "readable_lora_files", readable)


def _use_family(monkeypatch, module, family, constructor):
    monkeypatch.setattr(module, "detect_image_model", lambda _path: module.ImageModelInfo(family=family, is_distilled=False, size=None))
    monkeypatch.setattr(module.ModelConfig, "z_image", MagicMock(return_value="cfg"))
    monkeypatch.setattr(module, "_krea2_turbo_config", MagicMock(return_value="cfg"))
    model_class = MagicMock(name=constructor)
    monkeypatch.setattr(module, constructor, model_class)
    return model_class


def test_mflux_gets_each_lora_as_the_path_the_copies_give_for_mflux_and_the_models_family(image_mac, monkeypatch):
    module = image_mac
    family = "zimage"
    model_class = _use_family(monkeypatch, module, family, "ZImageTurbo")
    events: list[tuple] = []
    _fake_copies(monkeypatch, module, events)

    module.MfluxBackend().load_model("/models/m", lora_paths=["/l/peft.safetensors", "/l/plain.safetensors"], lora_weights=[0.5, 1.0])

    assert events[0] == ("open", ["/l/peft.safetensors", "/l/plain.safetensors"], family, "mflux")
    assert model_class.call_args.kwargs["lora_paths"] == ["/l/peft.safetensors.renamed", "/l/plain.safetensors"]
    assert model_class.call_args.kwargs["lora_scales"] == [0.5, 1.0]


def test_the_copies_are_removed_once_the_weights_are_resident_and_not_before(image_mac, monkeypatch):
    module = image_mac
    _use_family(monkeypatch, module, "zimage", "ZImageTurbo")
    events: list[tuple] = []
    _fake_copies(monkeypatch, module, events)
    monkeypatch.setattr(module, "load_resident", lambda build: events.append(("resident",)) or build())

    module.MfluxBackend().load_model("/models/m", lora_paths=["/l/peft.safetensors"], lora_weights=[1.0])

    assert [event[0] for event in events] == ["open", "resident", "close"]


def test_the_copies_are_removed_when_the_load_fails(image_mac, monkeypatch):
    module = image_mac
    model_class = _use_family(monkeypatch, module, "zimage", "ZImageTurbo")
    model_class.side_effect = RuntimeError("out of memory")
    events: list[tuple] = []
    _fake_copies(monkeypatch, module, events)

    with pytest.raises(RuntimeError, match="out of memory"):
        module.MfluxBackend().load_model("/models/m", lora_paths=["/l/peft.safetensors"], lora_weights=[1.0])

    assert events[-1] == ("close",)


def test_a_load_without_loras_makes_no_copies_and_passes_no_lora_arguments(image_mac, monkeypatch):
    module = image_mac
    model_class = _use_family(monkeypatch, module, "zimage", "ZImageTurbo")
    events: list[tuple] = []
    _fake_copies(monkeypatch, module, events)

    module.MfluxBackend().load_model("/models/m")

    assert "lora_paths" not in model_class.call_args.kwargs
    assert "lora_scales" not in model_class.call_args.kwargs
