"""Tests for offline detection of fully downloaded model weights."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zvisiongenerator.utils import model_files
from zvisiongenerator.utils.model_files import find_local_model_dir, has_complete_weights


def _touch(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_model_index(model_dir: Path) -> None:
    index = {
        "_class_name": "ZImagePipeline",
        "transformer": ["diffusers", "ZImageTransformer2DModel"],
        "text_encoder": ["transformers", "Qwen3Model"],
        "vae": ["diffusers", "AutoencoderKL"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "tokenizer": ["transformers", "Qwen2Tokenizer"],
        "safety_checker": [None, None],
    }
    _touch(model_dir / "model_index.json", json.dumps(index))


class TestHasCompleteWeights:
    def test_diffusers_layout_with_every_weighted_component(self, tmp_path):
        _write_model_index(tmp_path)
        for component in ("transformer", "text_encoder", "vae"):
            _touch(tmp_path / component / "model.safetensors")

        assert has_complete_weights(tmp_path) is True

    def test_diffusers_layout_missing_a_component_is_incomplete(self, tmp_path):
        _write_model_index(tmp_path)
        _touch(tmp_path / "transformer" / "model.safetensors")
        _touch(tmp_path / "vae" / "model.safetensors")

        assert has_complete_weights(tmp_path) is False

    def test_missing_shard_is_incomplete_even_with_a_stale_index(self, tmp_path):
        _touch(tmp_path / "model-00001-of-00002.safetensors")
        _touch(tmp_path / "model.safetensors.index.json", json.dumps({"weight_map": {"a": "model-00001-of-00005.safetensors"}}))

        assert has_complete_weights(tmp_path) is False
        _touch(tmp_path / "model-00002-of-00002.safetensors")
        assert has_complete_weights(tmp_path) is True

    @pytest.mark.parametrize("transformer", ["transformer.safetensors", "transformer-distilled.safetensors"])
    def test_ltx_mlx_layout_needs_transformer_and_decoder(self, tmp_path, transformer):
        _touch(tmp_path / "connector.safetensors")
        _touch(tmp_path / transformer)

        assert has_complete_weights(tmp_path) is False
        _touch(tmp_path / "vae_decoder.safetensors")
        assert has_complete_weights(tmp_path) is True

    def test_config_only_directory_is_not_downloaded(self, tmp_path):
        _write_model_index(tmp_path)

        assert has_complete_weights(tmp_path) is False


class TestFindLocalModelDir:
    def test_local_directory_with_weights(self, tmp_path):
        _touch(tmp_path / "model.safetensors")

        assert find_local_model_dir(str(tmp_path)) == tmp_path

    def test_unrecognised_reference_is_not_downloaded(self):
        assert find_local_model_dir("not a repo or path") is None

    def test_repo_id_uses_the_newest_complete_snapshot(self, tmp_path, monkeypatch):
        """Like mflux's offline resolution: an incomplete newer snapshot falls back to an older complete one."""
        import os

        snapshots = tmp_path / "models--owner--repo" / "snapshots"
        old = _touch(snapshots / "old" / "model.safetensors").parent
        new = snapshots / "new"
        new.mkdir()
        _touch(new / "model-00001-of-00002.safetensors")
        os.utime(old, (1, 1))
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))

        assert find_local_model_dir("owner/repo") == old
        _touch(new / "model-00002-of-00002.safetensors")
        assert find_local_model_dir("owner/repo") == new

    def test_repo_id_missing_from_cache_is_not_downloaded(self, tmp_path, monkeypatch):
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))

        assert find_local_model_dir("owner/repo") is None

    def test_pinned_revision_resolves_only_that_snapshot(self, tmp_path, monkeypatch):
        _touch(tmp_path / "model.safetensors")
        lookups: list[tuple[str, str | None]] = []

        def _snapshot_download(repo_id, *, revision, local_files_only):
            lookups.append((repo_id, revision))
            return str(tmp_path)

        monkeypatch.setattr("huggingface_hub.snapshot_download", _snapshot_download)

        assert find_local_model_dir("owner/repo@v2") == tmp_path
        assert lookups == [("owner/repo", "v2")]


def test_model_weight_files_ignore_files_outside_index_components(tmp_path):
    _write_model_index(tmp_path)
    for component in ("transformer", "text_encoder", "vae"):
        _touch(tmp_path / component / "model.safetensors")
    _touch(tmp_path / "single-file.safetensors")

    files = {path.relative_to(tmp_path).as_posix() for path in model_files.model_weight_files(tmp_path)}

    assert files == {"transformer/model.safetensors", "text_encoder/model.safetensors", "vae/model.safetensors"}


def test_ltx_text_encoder_repo_comes_from_the_pipeline_default(monkeypatch):
    import sys
    import types

    class _Pipeline:
        def __init__(self, model_dir, gemma_model_id="org/gemma-new", low_memory=True):
            pass

    monkeypatch.setattr(model_files, "_ltx_mlx_text_encoder_repo", None)
    monkeypatch.setitem(sys.modules, "ltx_pipelines_mlx", None)  # import fails: fall back, but do not cache it
    assert model_files.ltx_mlx_text_encoder_repo() == model_files._LTX_MLX_TEXT_ENCODER_FALLBACK

    monkeypatch.setitem(sys.modules, "ltx_pipelines_mlx", types.SimpleNamespace(TextToVideoPipeline=_Pipeline))
    assert model_files.ltx_mlx_text_encoder_repo() == "org/gemma-new"


def test_precision_variants_are_ignored_next_to_plain_weights(tmp_path):
    _touch(tmp_path / "transformer" / "diffusion_pytorch_model.safetensors")
    _touch(tmp_path / "transformer" / "diffusion_pytorch_model.fp16.safetensors")
    _touch(tmp_path / "vae" / "diffusion_pytorch_model.fp16-00001-of-00001.safetensors")  # only a variant: kept

    files = {path.relative_to(tmp_path).as_posix() for path in model_files.model_weight_files(tmp_path)}

    assert files == {"transformer/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.fp16-00001-of-00001.safetensors"}
