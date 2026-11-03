"""Tests for the YuE2 streamed model loader and the weight size check, on a tiny model written to a temporary folder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import yue2.pipeline as kit_pipeline
from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM
from zvisiongenerator.backends.audio.yue2_load import _size_checked_identity, load_model, size_checked_weights


def _tiny_config() -> YuE2Config:
    return YuE2Config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=96,
        vocab_size=300,
        max_position_embeddings=128,
        max_latent_frames=64,
        latent_dim=8,
    )


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    model = YuE2ForCausalLM(_tiny_config()).to(torch.bfloat16)
    save_file({name: tensor.contiguous() for name, tensor in model.state_dict().items()}, str(tmp_path / "model.safetensors"))
    _tiny_config().save_pretrained(tmp_path)
    return tmp_path


def _write_manifest(directory: Path, **sizes: int) -> None:
    files = {name: {"bytes": size, "sha256": f"sha-{name}"} for name, size in sizes.items()}
    (directory / "weights_manifest.json").write_text(json.dumps({"schema": 1, "files": files}), encoding="utf-8")


class TestLoadModel:
    def test_the_model_holds_the_files_tensors_in_bf16_in_eval_mode(self, model_dir):
        from safetensors.torch import load_file

        saved = load_file(str(model_dir / "model.safetensors"))

        model = load_model(model_dir, pin=False)

        assert not model.training
        state = model.state_dict()
        assert set(state) == set(saved)
        assert all(torch.equal(state[name], tensor) for name, tensor in saved.items())
        assert {parameter.dtype for parameter in model.parameters()} == {torch.bfloat16}

    def test_the_tensors_are_pinned_as_they_are_read_and_the_model_keeps_those_very_tensors(self, model_dir, monkeypatch):
        pinned: dict[int, str] = {}

        def pin_memory(self: torch.Tensor, *args: object) -> torch.Tensor:
            pinned[self.data_ptr()] = "pinned"
            return self

        monkeypatch.setattr(torch.Tensor, "pin_memory", pin_memory)

        model = load_model(model_dir, pin=True)

        assert len(pinned) == len(model.state_dict())
        assert all(tensor.data_ptr() in pinned for tensor in model.state_dict().values())

    def test_nothing_is_pinned_without_the_pin_flag(self, model_dir, monkeypatch):
        monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, *args: pytest.fail("pinned"))

        load_model(model_dir, pin=False)

    def test_the_file_is_not_mapped(self, model_dir, monkeypatch):
        monkeypatch.setattr("safetensors.safe_open", lambda *args, **kwargs: pytest.fail("the file was mapped"))
        monkeypatch.setattr("safetensors.torch.load_file", lambda *args, **kwargs: pytest.fail("the file was mapped"))

        load_model(model_dir, pin=False)


class TestSizeCheck:
    def test_the_kit_still_has_the_function_this_replaces_and_its_pipeline_still_looks_it_up_by_name(self):
        assert callable(kit_pipeline.model_identity)
        assert "model_identity" in kit_pipeline.YuE2Pipeline.__init__.__code__.co_names

    def test_the_kits_function_is_replaced_inside_the_block_and_put_back_after(self):
        original = kit_pipeline.model_identity

        with size_checked_weights():
            assert kit_pipeline.model_identity is _size_checked_identity

        assert kit_pipeline.model_identity is original

    def test_the_kits_function_is_put_back_after_an_error(self):
        original = kit_pipeline.model_identity

        with pytest.raises(RuntimeError), size_checked_weights():
            raise RuntimeError("boom")

        assert kit_pipeline.model_identity is original

    def test_a_kit_without_the_function_fails_loudly(self, monkeypatch):
        monkeypatch.delattr(kit_pipeline, "model_identity")

        with pytest.raises(AttributeError), size_checked_weights():
            pass

    def test_matching_sizes_pass_and_the_record_has_the_kits_shape_with_the_manifests_hashes(self, model_dir):
        size = (model_dir / "model.safetensors").stat().st_size
        _write_manifest(model_dir, **{"model.safetensors": size})

        record = _size_checked_identity(model_dir)

        assert record["files"] == {"model.safetensors": {"sha256": "sha-model.safetensors", "bytes": size}}
        assert set(record) == {"files", "config_sha256"}

    def test_a_file_of_another_size_than_the_manifest_says_is_refused(self, model_dir):
        size = (model_dir / "model.safetensors").stat().st_size
        _write_manifest(model_dir, **{"model.safetensors": size + 1})

        with pytest.raises(ValueError, match=r"model\.safetensors is"):
            _size_checked_identity(model_dir)

    def test_the_size_is_not_compared_when_verifying_is_off(self, model_dir):
        _write_manifest(model_dir, **{"model.safetensors": 1})

        assert _size_checked_identity(model_dir, False)["files"]["model.safetensors"]["bytes"] > 1

    def test_a_file_the_manifest_does_not_list_is_refused(self, model_dir):
        _write_manifest(model_dir, **{"other.safetensors": 1})

        with pytest.raises(ValueError):
            _size_checked_identity(model_dir)

    def test_a_listed_file_that_is_missing_is_refused(self, model_dir):
        size = (model_dir / "model.safetensors").stat().st_size
        _write_manifest(model_dir, **{"model.safetensors": size, "model-2.safetensors": 5})

        with pytest.raises(ValueError, match="missing or unexpected"):
            _size_checked_identity(model_dir)

    def test_a_folder_without_a_manifest_still_needs_its_weights(self, model_dir):
        assert set(_size_checked_identity(model_dir)["files"]) == {"model.safetensors"}
        (model_dir / "model.safetensors").unlink()

        with pytest.raises(FileNotFoundError):
            _size_checked_identity(model_dir)
