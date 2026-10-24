"""Tests for offline detection of fully downloaded model weights."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zvisiongenerator.utils import model_files
from zvisiongenerator.utils.model_files import LTX2_UNLOADED_COMPONENTS, cached_snapshot_dirs, find_local_model_dir, has_complete_weights, ltx2_text_encoder_class


def _touch(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_model_index(model_dir: Path, class_name: str = "ZImagePipeline") -> None:
    index = {
        "_class_name": class_name,
        "transformer": ["diffusers", "ZImageTransformer2DModel"],
        "text_encoder": ["transformers", "Qwen3Model"],
        "vae": ["diffusers", "AutoencoderKL"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "tokenizer": ["transformers", "Qwen2Tokenizer"],
        "safety_checker": [None, None],
    }
    _touch(model_dir / "model_index.json", json.dumps(index))


_PIN = "7b40724fc39bbb01d9b4d72638c0d263531c88c2"
_LTX25_WEIGHTED = {"audio_vae": 1, "connectors": 2, "text_encoder": 5, "transformer": 4, "vae": 1, "vocoder": 1}


def _write_ltx25_index(model_dir: Path, class_name: str = "LTX2Pipeline", text_encoder: str = "Gemma4UnifiedForConditionalGeneration") -> None:
    """Write a ``model_index.json`` shaped like the LTX-2.5 diffusers repo's."""
    index = {
        "_class_name": class_name,
        "audio_vae": ["diffusers", "AutoencoderKLLTX2Audio"],
        "connectors": ["ltx2", "LTX2TextConnectors"],
        "diffusion_decoder": ["diffusers", "LTX2VideoDiffusionDecoderModel"],
        "duration_head": ["ltx2", "LTX2DurationHead"],
        "processor": ["transformers", "Gemma4Processor"],
        "prompt_enhancer": ["transformers", "Gemma4ForConditionalGeneration"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", text_encoder],
        "tokenizer": ["transformers", "GemmaTokenizer"],
        "transformer": ["diffusers", "LTX2VideoTransformer3DModel"],
        "vae": ["diffusers", "AutoencoderKLLTX2Video"],
        "vocoder": ["ltx2", "LTX2VocoderWithBWE"],
    }
    _touch(model_dir / "model_index.json", json.dumps(index))


def _write_ltx25_snapshot(model_dir: Path) -> None:
    """Lay out a complete LTX-2.5 snapshot: sharded weights, config-only extras and an unlisted ``latent_upsampler``."""
    _write_ltx25_index(model_dir)
    for component, shards in _LTX25_WEIGHTED.items():
        _touch(model_dir / component / "config.json")
        if shards == 1:
            _touch(model_dir / component / "diffusion_pytorch_model.safetensors")
            continue
        for shard in range(1, shards + 1):
            _touch(model_dir / component / f"model-{shard:05d}-of-{shards:05d}.safetensors")
    _touch(model_dir / "latent_upsampler" / "diffusion_pytorch_model.safetensors")
    for component in ("duration_head", "diffusion_decoder", "scheduler", "tokenizer"):
        _touch(model_dir / component / "config.json")


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

    def test_an_ltx_2_repo_without_its_transformer_weights_is_complete(self, tmp_path):
        _write_model_index(tmp_path, "LTX2Pipeline")
        _touch(tmp_path / "transformer" / "config.json")
        _touch(tmp_path / "text_encoder" / "model.safetensors")
        _touch(tmp_path / "vae" / "model.safetensors")

        assert has_complete_weights(tmp_path) is True

    def test_an_ltx_2_3_repo_still_needs_its_other_weighted_components(self, tmp_path):
        _write_model_index(tmp_path, "LTX2Pipeline")
        _touch(tmp_path / "transformer" / "config.json")
        _touch(tmp_path / "vae" / "model.safetensors")

        assert has_complete_weights(tmp_path) is False

    def test_the_complete_ltx_2_5_snapshot_is_complete(self, tmp_path):
        _write_ltx25_snapshot(tmp_path)

        assert has_complete_weights(tmp_path) is True

    def test_an_ltx_2_5_snapshot_missing_a_transformer_shard_is_incomplete(self, tmp_path):
        _write_ltx25_snapshot(tmp_path)
        (tmp_path / "transformer" / "model-00004-of-00004.safetensors").unlink()

        assert has_complete_weights(tmp_path) is False

    def test_an_ltx_2_5_snapshot_with_only_the_transformer_config_is_incomplete(self, tmp_path):
        _write_ltx25_snapshot(tmp_path)
        for shard in (tmp_path / "transformer").glob("*.safetensors"):
            shard.unlink()

        assert has_complete_weights(tmp_path) is False

    def test_an_ltx_2_5_snapshot_missing_another_weighted_component_is_incomplete(self, tmp_path):
        _write_ltx25_snapshot(tmp_path)
        (tmp_path / "vocoder" / "diffusion_pytorch_model.safetensors").unlink()

        assert has_complete_weights(tmp_path) is False

    def test_an_ltx_2_5_index_alone_is_not_downloaded(self, tmp_path):
        _write_ltx25_index(tmp_path)

        assert has_complete_weights(tmp_path) is False

    def test_the_components_an_ltx_2_index_lists_but_the_app_never_loads_need_no_weights(self, tmp_path):
        _write_ltx25_snapshot(tmp_path)

        components = model_files.weighted_components(tmp_path)

        assert components is not None
        assert not set(components) & LTX2_UNLOADED_COMPONENTS
        assert set(components) == set(_LTX25_WEIGHTED)

    def test_a_non_ltx_pipeline_keeps_components_named_like_the_unloaded_ones(self, tmp_path):
        _write_ltx25_index(tmp_path, class_name="OtherPipeline")

        components = model_files.weighted_components(tmp_path)

        assert components is not None
        assert {"prompt_enhancer", "duration_head", "diffusion_decoder"} <= set(components)

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

    def test_ltx_mlx_layout_accepts_the_2_5_conv_decoder(self, tmp_path):
        _touch(tmp_path / "connector.safetensors")
        _touch(tmp_path / "transformer-distilled.safetensors")
        _touch(tmp_path / "text_encoder.safetensors")
        _touch(tmp_path / "text_encoder_config.json", "{}")

        assert has_complete_weights(tmp_path) is False
        _touch(tmp_path / "vae_decoder_conv.safetensors")
        assert has_complete_weights(tmp_path) is True

    def test_ltx_mlx_layout_with_a_bundled_encoder_config_needs_its_weights(self, tmp_path):
        """An interrupted download links the small config before the large weights."""
        _touch(tmp_path / "connector.safetensors")
        _touch(tmp_path / "transformer-distilled.safetensors")
        _touch(tmp_path / "vae_decoder_conv.safetensors")
        _touch(tmp_path / "text_encoder_config.json", "{}")

        assert has_complete_weights(tmp_path) is False
        _touch(tmp_path / "text_encoder.safetensors")
        assert has_complete_weights(tmp_path) is True

    def test_converted_layout_with_symlinked_base_components(self, tmp_path):
        base = tmp_path / "base"
        model = tmp_path / "converted"
        _write_model_index(model)
        _touch(model / "transformer" / "model.safetensors")
        for component in ("text_encoder", "vae"):
            _touch(base / component / "model.safetensors")
            (model / component).symlink_to(base / component, target_is_directory=True)

        assert has_complete_weights(model) is True
        (base / "vae" / "model.safetensors").unlink()
        assert has_complete_weights(model) is False

    def test_root_single_file_stands_in_for_a_missing_transformer_folder(self, tmp_path):
        # mflux downloads only Krea 2 Turbo's root turbo.safetensors, not the diffusers transformer/ shards.
        _write_model_index(tmp_path, "Krea2Pipeline")
        _touch(tmp_path / "turbo.safetensors")
        for component in ("text_encoder", "vae"):
            _touch(tmp_path / component / "model.safetensors")

        assert has_complete_weights(tmp_path) is True

    def test_root_file_without_text_encoder_is_still_incomplete(self, tmp_path):
        _write_model_index(tmp_path, "Krea2Pipeline")
        _touch(tmp_path / "turbo.safetensors")
        _touch(tmp_path / "vae" / "model.safetensors")

        assert has_complete_weights(tmp_path) is False

    def test_root_single_file_does_not_stand_in_for_other_pipelines(self, tmp_path):
        # FLUX.2 Klein ships a root single-file checkpoint next to its diffusers folders.
        _write_model_index(tmp_path, "Flux2KleinPipeline")
        _touch(tmp_path / "flux-2-klein-4b.safetensors")
        for component in ("text_encoder", "vae"):
            _touch(tmp_path / component / "model.safetensors")

        assert has_complete_weights(tmp_path) is False

    def test_config_only_directory_is_not_downloaded(self, tmp_path):
        _write_model_index(tmp_path)

        assert has_complete_weights(tmp_path) is False


class TestLtx2TextEncoderClass:
    def test_returns_the_class_of_an_ltx_2_3_index(self, tmp_path):
        _write_ltx25_index(tmp_path, text_encoder="Gemma3ForConditionalGeneration")

        assert ltx2_text_encoder_class(tmp_path) == "Gemma3ForConditionalGeneration"

    def test_returns_the_class_of_an_ltx_2_5_index(self, tmp_path):
        _write_ltx25_snapshot(tmp_path)

        assert ltx2_text_encoder_class(tmp_path) == "Gemma4UnifiedForConditionalGeneration"

    def test_accepts_the_image_to_video_pipeline(self, tmp_path):
        _write_ltx25_index(tmp_path, class_name="LTX2ImageToVideoPipeline")

        assert ltx2_text_encoder_class(tmp_path) == "Gemma4UnifiedForConditionalGeneration"

    def test_is_none_without_an_index(self, tmp_path):
        assert ltx2_text_encoder_class(tmp_path) is None

    def test_is_none_for_an_unreadable_index(self, tmp_path):
        _touch(tmp_path / "model_index.json", "{not json")

        assert ltx2_text_encoder_class(tmp_path) is None

    def test_is_none_for_a_non_ltx_2_index(self, tmp_path):
        _write_model_index(tmp_path, "ZImagePipeline")

        assert ltx2_text_encoder_class(tmp_path) is None

    @pytest.mark.parametrize("entry", [None, "Gemma3", ["transformers"], ["transformers", ""], ["transformers", 3], [None, None]])
    def test_is_none_for_a_malformed_text_encoder_entry(self, tmp_path, entry):
        index = {"_class_name": "LTX2Pipeline", "text_encoder": entry}
        _touch(tmp_path / "model_index.json", json.dumps(index))

        assert ltx2_text_encoder_class(tmp_path) is None

    def test_the_unloaded_components_are_the_four_the_app_never_reads(self):
        assert LTX2_UNLOADED_COMPONENTS == frozenset({"processor", "prompt_enhancer", "duration_head", "diffusion_decoder"})


class TestFindLocalModelDir:
    def test_a_pinned_ltx_2_5_mlx_pack_is_found_through_the_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
        snapshot = tmp_path / "models--owner--pack" / "snapshots" / _PIN
        for name in ("connector", "transformer-distilled", "vae_decoder_conv", "text_encoder"):
            _touch(snapshot / f"{name}.safetensors")
        _touch(snapshot / "text_encoder_config.json", "{}")

        assert find_local_model_dir(f"owner/pack@{_PIN}") == snapshot
        (snapshot / "text_encoder.safetensors").unlink()
        assert find_local_model_dir(f"owner/pack@{_PIN}") is None

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

    def _cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
        return tmp_path / "models--owner--repo"

    def test_a_pinned_commit_resolves_only_that_snapshot(self, tmp_path, monkeypatch):
        repo = self._cache(tmp_path, monkeypatch)
        pinned = _touch(repo / "snapshots" / "abc123" / "model.safetensors").parent
        _touch(repo / "snapshots" / "def456" / "model.safetensors")

        assert find_local_model_dir("owner/repo@abc123") == pinned
        assert find_local_model_dir("owner/repo@fff999") is None

    def test_a_pinned_snapshot_with_only_the_files_diffusers_loads_counts_as_downloaded(self, tmp_path, monkeypatch):
        """The Hub's own offline lookup would call this incomplete, as the repo has files (a README) that were never fetched."""
        repo = self._cache(tmp_path, monkeypatch)
        snapshot = repo / "snapshots" / "abc123"
        _write_model_index(snapshot)
        for component in ("transformer", "text_encoder", "vae"):
            _touch(snapshot / component / "model.safetensors")

        assert find_local_model_dir("owner/repo@abc123") == snapshot

    def test_the_pinned_ltx_2_5_snapshot_is_found_through_the_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
        snapshot = tmp_path / "models--unsloth--LTX-2.5-FP8" / "snapshots" / _PIN
        _write_ltx25_snapshot(snapshot)

        assert find_local_model_dir(f"unsloth/LTX-2.5-FP8@{_PIN}") == snapshot
        (snapshot / "transformer" / "model-00001-of-00004.safetensors").unlink()
        assert find_local_model_dir(f"unsloth/LTX-2.5-FP8@{_PIN}") is None

    def test_a_pinned_tag_resolves_through_its_ref(self, tmp_path, monkeypatch):
        repo = self._cache(tmp_path, monkeypatch)
        snapshot = _touch(repo / "snapshots" / "abc123" / "model.safetensors").parent
        (repo / "refs").mkdir()
        (repo / "refs" / "v2").write_text("abc123\n", encoding="utf-8")

        assert find_local_model_dir("owner/repo@v2") == snapshot

    def test_a_pinned_snapshot_without_all_its_weights_is_not_downloaded(self, tmp_path, monkeypatch):
        repo = self._cache(tmp_path, monkeypatch)
        _touch(repo / "snapshots" / "abc123" / "model-00001-of-00002.safetensors")

        assert find_local_model_dir("owner/repo@abc123") is None


def test_model_weight_files_ignore_files_outside_index_components(tmp_path):
    _write_model_index(tmp_path)
    for component in ("transformer", "text_encoder", "vae"):
        _touch(tmp_path / component / "model.safetensors")
    _touch(tmp_path / "single-file.safetensors")

    files = {path.relative_to(tmp_path).as_posix() for path in model_files.model_weight_files(tmp_path)}

    assert files == {"transformer/model.safetensors", "text_encoder/model.safetensors", "vae/model.safetensors"}


def test_root_single_file_counts_only_when_the_transformer_folder_is_empty(tmp_path):
    _write_model_index(tmp_path, "Krea2Pipeline")
    root = _touch(tmp_path / "turbo.safetensors")
    vae = _touch(tmp_path / "vae" / "model.safetensors")

    assert set(model_files.model_weight_files(tmp_path)) == {root, vae}

    shard = _touch(tmp_path / "transformer" / "model.safetensors")

    assert set(model_files.model_weight_files(tmp_path)) == {shard, vae}


def test_ltx_text_encoder_repo_comes_from_the_pipeline_default(monkeypatch):
    import sys
    import types

    class _Pipeline:
        def __init__(self, model_dir, gemma_model_id="org/gemma-new", low_memory=True):
            pass

    monkeypatch.setattr(model_files, "_ltx_mlx_text_encoder_repo", None)
    monkeypatch.setitem(sys.modules, "ltx_pipelines_mlx", None)  # import fails: fall back, but do not cache it
    assert model_files.ltx_mlx_text_encoder_repo() == model_files._LTX_MLX_TEXT_ENCODER_FALLBACK

    monkeypatch.setitem(sys.modules, "ltx_pipelines_mlx", types.SimpleNamespace(BasePipeline=_Pipeline))
    assert model_files.ltx_mlx_text_encoder_repo() == "org/gemma-new"


@pytest.mark.parametrize(
    ("present", "expected"),
    [
        (["transformer-distilled.safetensors"], "transformer-distilled.safetensors"),
        (["transformer-distilled.safetensors", "transformer-distilled-1.1.safetensors"], "transformer-distilled-1.1.safetensors"),
        (["transformer-distilled-1.0.safetensors", "transformer-distilled-1.1.safetensors"], "transformer-distilled-1.1.safetensors"),
        (
            ["transformer.safetensors", "transformer-distilled.safetensors", "transformer-distilled-1.1.safetensors"],
            "transformer.safetensors",
        ),
        ([], None),
    ],
)
def test_ltx_transformer_file_follows_the_pipeline_choice(tmp_path, present, expected):
    for name in present:
        _touch(tmp_path / name)

    found = model_files.ltx_mlx_transformer_file(tmp_path)

    assert (found.name if found else None) == expected


@pytest.mark.parametrize(
    ("present", "expected"),
    [
        (["spatial_upscaler_x2_v1_1.safetensors"], "spatial_upscaler_x2_v1_1.safetensors"),
        (["ltx-2.3-spatial-upscaler-x2.safetensors"], "ltx-2.3-spatial-upscaler-x2.safetensors"),
        (["spatial_upscaler_x2_v1_0.safetensors"], "spatial_upscaler_x2_v1_0.safetensors"),
        (["spatial_upscaler_x2_v1_1-1.0.safetensors", "spatial_upscaler_x2_v1_1-1.1.safetensors"], "spatial_upscaler_x2_v1_1-1.1.safetensors"),
        (["spatial_upscaler_x2_v1_1.safetensors", "ltx-2.3-spatial-upscaler-x2.safetensors"], "ltx-2.3-spatial-upscaler-x2.safetensors"),
        (["transformer.safetensors", "connector.safetensors"], None),
        ([], None),
    ],
)
def test_ltx_upsampler_file_follows_the_pipeline_names(tmp_path, present, expected):
    for name in present:
        _touch(tmp_path / name)

    found = model_files.ltx_mlx_upsampler_file(tmp_path)

    assert (found.name if found else None) == expected


def test_precision_variants_are_ignored_next_to_plain_weights(tmp_path):
    _touch(tmp_path / "transformer" / "diffusion_pytorch_model.safetensors")
    _touch(tmp_path / "transformer" / "diffusion_pytorch_model.fp16.safetensors")
    _touch(tmp_path / "vae" / "diffusion_pytorch_model.fp16-00001-of-00001.safetensors")  # only a variant: kept

    files = {path.relative_to(tmp_path).as_posix() for path in model_files.model_weight_files(tmp_path)}

    assert files == {"transformer/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.fp16-00001-of-00001.safetensors"}


def test_directory_bytes_counts_regular_files_and_skips_symlinks(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.bin").write_bytes(b"x" * 10)
    (tmp_path / "two.bin").write_bytes(b"y" * 5)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.bin"
    outside.write_bytes(b"z" * 100)
    (tmp_path / "link.bin").symlink_to(outside)

    assert model_files.directory_bytes(tmp_path) == 15


def test_directory_bytes_is_zero_for_a_missing_directory(tmp_path):
    assert model_files.directory_bytes(tmp_path / "missing") == 0


class TestHasCompleteWeightsOnly:
    def test_none_requires_every_declared_component(self, tmp_path):
        _write_model_index(tmp_path)
        for component in ("text_encoder", "vae"):
            _touch(tmp_path / component / "model.safetensors")

        assert has_complete_weights(tmp_path, only=None) is False

    def test_only_restricts_the_required_components(self, tmp_path):
        _write_model_index(tmp_path)
        for component in ("text_encoder", "vae"):
            _touch(tmp_path / component / "model.safetensors")

        assert has_complete_weights(tmp_path, only={"text_encoder", "vae"}) is True
        assert has_complete_weights(tmp_path, only={"transformer", "vae"}) is False

    def test_empty_set_requires_no_component(self, tmp_path):
        _write_model_index(tmp_path)

        assert has_complete_weights(tmp_path, only=set()) is True

    def test_shard_completeness_still_applies(self, tmp_path):
        _write_model_index(tmp_path)
        _touch(tmp_path / "vae" / "model-00001-of-00002.safetensors")

        assert has_complete_weights(tmp_path, only=set()) is False
        assert has_complete_weights(tmp_path, only={"vae"}) is False

    def test_unreadable_index_still_fails(self, tmp_path):
        _touch(tmp_path / "model_index.json", "{not json")

        assert has_complete_weights(tmp_path, only=set()) is False

    def test_missing_index_in_an_empty_folder_is_not_complete(self, tmp_path):
        assert has_complete_weights(tmp_path, only=set()) is False
        assert has_complete_weights(tmp_path, only={"vae"}) is False


class TestCachedSnapshotDirs:
    def test_lists_snapshots_newest_first(self, tmp_path, monkeypatch):
        import os

        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
        snapshots = tmp_path / "models--owner--repo" / "snapshots"
        old = _touch(snapshots / "old" / "a").parent
        new = _touch(snapshots / "new" / "a").parent
        os.utime(old, (1, 1))

        assert cached_snapshot_dirs("owner/repo", None) == [new, old]

    def test_revision_maps_through_refs_and_missing_repo_is_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
        repo = tmp_path / "models--owner--repo"
        snapshot = _touch(repo / "snapshots" / "abc123" / "a").parent
        _touch(repo / "refs" / "main", "abc123\n")

        assert cached_snapshot_dirs("owner/repo", "main") == [snapshot]
        assert cached_snapshot_dirs("owner/repo", "..") == []
        assert cached_snapshot_dirs("owner/missing", None) == []


@pytest.mark.parametrize(
    ("present", "expected"),
    [
        (["text_encoder.safetensors", "text_encoder_config.json"], "text_encoder.safetensors"),
        (["text_encoder.safetensors"], None),
        (["text_encoder_config.json"], None),
        ([], None),
    ],
)
def test_ltx_bundled_text_encoder_needs_both_files(tmp_path, present, expected):
    for name in present:
        _touch(tmp_path / name)

    found = model_files.ltx_mlx_bundled_text_encoder(tmp_path)

    assert (found.name if found else None) == expected


@pytest.mark.parametrize(
    ("present", "expected"),
    [
        (["vae_decoder_conv.safetensors"], "vae_decoder_conv.safetensors"),
        (["vae_decoder.safetensors"], "vae_decoder.safetensors"),
        (["vae_decoder.safetensors", "vae_decoder_conv.safetensors"], "vae_decoder_conv.safetensors"),
        (["vae_encoder_conv.safetensors"], None),
        ([], None),
    ],
)
def test_ltx_video_decoder_file_follows_the_pipeline_names(tmp_path, present, expected):
    for name in present:
        _touch(tmp_path / name)

    found = model_files.ltx_mlx_video_decoder_file(tmp_path)

    assert (found.name if found else None) == expected
