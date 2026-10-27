"""Tests for offline detection of fully downloaded model weights."""

from __future__ import annotations

import json

import pytest

from tests._snapshots import (
    LTX25_WEIGHTED,
    touch,
    write_h3_index,
    write_h3_snapshot,
    write_ltx25_index,
    write_ltx25_snapshot,
    write_model_index,
)

from zvisiongenerator.utils import model_files
from zvisiongenerator.utils.model_files import (
    LTX2_UNLOADED_COMPONENTS,
    cached_snapshot_dirs,
    find_local_model_dir,
    has_complete_weights,
    ltx2_text_encoder_class,
)


_PIN = "a1" * 20  # an opaque commit id; the tests only need some snapshot folder name


class TestHasCompleteWeights:
    def test_diffusers_layout_with_every_weighted_component(self, tmp_path):
        write_model_index(tmp_path)
        for component in ("transformer", "text_encoder", "vae"):
            touch(tmp_path / component / "model.safetensors")

        assert has_complete_weights(tmp_path) is True

    def test_diffusers_layout_missing_a_component_is_incomplete(self, tmp_path):
        write_model_index(tmp_path)
        touch(tmp_path / "transformer" / "model.safetensors")
        touch(tmp_path / "vae" / "model.safetensors")

        assert has_complete_weights(tmp_path) is False

    def test_an_ltx_2_repo_without_its_transformer_weights_is_complete(self, tmp_path):
        write_model_index(tmp_path, "LTX2Pipeline")
        touch(tmp_path / "transformer" / "config.json")
        touch(tmp_path / "text_encoder" / "model.safetensors")
        touch(tmp_path / "vae" / "model.safetensors")

        assert has_complete_weights(tmp_path) is True

    def test_an_ltx_2_3_repo_still_needs_its_other_weighted_components(self, tmp_path):
        write_model_index(tmp_path, "LTX2Pipeline")
        touch(tmp_path / "transformer" / "config.json")
        touch(tmp_path / "vae" / "model.safetensors")

        assert has_complete_weights(tmp_path) is False

    def test_the_complete_ltx_2_5_snapshot_is_complete(self, tmp_path):
        write_ltx25_snapshot(tmp_path)

        assert has_complete_weights(tmp_path) is True

    def test_an_ltx_2_5_snapshot_missing_a_transformer_shard_is_incomplete(self, tmp_path):
        write_ltx25_snapshot(tmp_path)
        (tmp_path / "transformer" / "model-00004-of-00004.safetensors").unlink()

        assert has_complete_weights(tmp_path) is False

    def test_an_ltx_2_5_snapshot_with_only_the_transformer_config_is_incomplete(self, tmp_path):
        write_ltx25_snapshot(tmp_path)
        for shard in (tmp_path / "transformer").glob("*.safetensors"):
            shard.unlink()

        assert has_complete_weights(tmp_path) is False

    def test_an_ltx_2_5_snapshot_missing_another_weighted_component_is_incomplete(self, tmp_path):
        write_ltx25_snapshot(tmp_path)
        (tmp_path / "vocoder" / "diffusion_pytorch_model.safetensors").unlink()

        assert has_complete_weights(tmp_path) is False

    def test_an_ltx_2_5_index_alone_is_not_downloaded(self, tmp_path):
        write_ltx25_index(tmp_path)

        assert has_complete_weights(tmp_path) is False

    def test_the_components_an_ltx_2_index_lists_but_the_app_never_loads_need_no_weights(self, tmp_path):
        write_ltx25_snapshot(tmp_path)

        components = model_files.weighted_components(tmp_path)

        assert components is not None
        assert not set(components) & LTX2_UNLOADED_COMPONENTS
        assert set(components) == set(LTX25_WEIGHTED)

    def test_a_non_ltx_pipeline_keeps_components_named_like_the_unloaded_ones(self, tmp_path):
        write_ltx25_index(tmp_path, class_name="OtherPipeline")

        components = model_files.weighted_components(tmp_path)

        assert components is not None
        assert {"prompt_enhancer", "duration_head", "diffusion_decoder"} <= set(components)

    def test_missing_shard_is_incomplete_even_with_a_stale_index(self, tmp_path):
        touch(tmp_path / "model-00001-of-00002.safetensors")
        touch(tmp_path / "model.safetensors.index.json", json.dumps({"weight_map": {"a": "model-00001-of-00005.safetensors"}}))

        assert has_complete_weights(tmp_path) is False
        touch(tmp_path / "model-00002-of-00002.safetensors")
        assert has_complete_weights(tmp_path) is True

    @pytest.mark.parametrize("transformer", ["transformer.safetensors", "transformer-distilled.safetensors"])
    def test_ltx_mlx_layout_needs_transformer_and_decoder(self, tmp_path, transformer):
        touch(tmp_path / "connector.safetensors")
        touch(tmp_path / transformer)

        assert has_complete_weights(tmp_path) is False
        touch(tmp_path / "vae_decoder.safetensors")
        assert has_complete_weights(tmp_path) is True

    def test_ltx_mlx_layout_accepts_the_2_5_conv_decoder(self, tmp_path):
        touch(tmp_path / "connector.safetensors")
        touch(tmp_path / "transformer-distilled.safetensors")
        touch(tmp_path / "text_encoder.safetensors")
        touch(tmp_path / "text_encoder_config.json", "{}")

        assert has_complete_weights(tmp_path) is False
        touch(tmp_path / "vae_decoder_conv.safetensors")
        assert has_complete_weights(tmp_path) is True

    def test_ltx_mlx_layout_with_a_bundled_encoder_config_needs_its_weights(self, tmp_path):
        """An interrupted download links the small config before the large weights."""
        touch(tmp_path / "connector.safetensors")
        touch(tmp_path / "transformer-distilled.safetensors")
        touch(tmp_path / "vae_decoder_conv.safetensors")
        touch(tmp_path / "text_encoder_config.json", "{}")

        assert has_complete_weights(tmp_path) is False
        touch(tmp_path / "text_encoder.safetensors")
        assert has_complete_weights(tmp_path) is True

    def test_converted_layout_with_symlinked_base_components(self, tmp_path):
        base = tmp_path / "base"
        model = tmp_path / "converted"
        write_model_index(model)
        touch(model / "transformer" / "model.safetensors")
        for component in ("text_encoder", "vae"):
            touch(base / component / "model.safetensors")
            (model / component).symlink_to(base / component, target_is_directory=True)

        assert has_complete_weights(model) is True
        (base / "vae" / "model.safetensors").unlink()
        assert has_complete_weights(model) is False

    def test_root_single_file_stands_in_for_a_missing_transformer_folder(self, tmp_path):
        # mflux downloads only Krea 2 Turbo's root turbo.safetensors, not the diffusers transformer/ shards.
        write_model_index(tmp_path, "Krea2Pipeline")
        touch(tmp_path / "turbo.safetensors")
        for component in ("text_encoder", "vae"):
            touch(tmp_path / component / "model.safetensors")

        assert has_complete_weights(tmp_path) is True

    def test_root_file_without_text_encoder_is_still_incomplete(self, tmp_path):
        write_model_index(tmp_path, "Krea2Pipeline")
        touch(tmp_path / "turbo.safetensors")
        touch(tmp_path / "vae" / "model.safetensors")

        assert has_complete_weights(tmp_path) is False

    def test_root_single_file_does_not_stand_in_for_other_pipelines(self, tmp_path):
        # FLUX.2 Klein ships a root single-file checkpoint next to its diffusers folders.
        write_model_index(tmp_path, "Flux2KleinPipeline")
        touch(tmp_path / "flux-2-klein-4b.safetensors")
        for component in ("text_encoder", "vae"):
            touch(tmp_path / component / "model.safetensors")

        assert has_complete_weights(tmp_path) is False

    def test_config_only_directory_is_not_downloaded(self, tmp_path):
        write_model_index(tmp_path)

        assert has_complete_weights(tmp_path) is False


class TestLtx2TextEncoderClass:
    def test_returns_the_class_of_an_ltx_2_3_index(self, tmp_path):
        write_ltx25_index(tmp_path, text_encoder="Gemma3ForConditionalGeneration")

        assert ltx2_text_encoder_class(tmp_path) == "Gemma3ForConditionalGeneration"

    def test_returns_the_class_of_an_ltx_2_5_index(self, tmp_path):
        write_ltx25_snapshot(tmp_path)

        assert ltx2_text_encoder_class(tmp_path) == "Gemma4UnifiedForConditionalGeneration"

    def test_accepts_the_image_to_video_pipeline(self, tmp_path):
        write_ltx25_index(tmp_path, class_name="LTX2ImageToVideoPipeline")

        assert ltx2_text_encoder_class(tmp_path) == "Gemma4UnifiedForConditionalGeneration"

    def test_is_none_without_an_index(self, tmp_path):
        assert ltx2_text_encoder_class(tmp_path) is None

    def test_is_none_for_an_unreadable_index(self, tmp_path):
        touch(tmp_path / "model_index.json", "{not json")

        assert ltx2_text_encoder_class(tmp_path) is None

    def test_is_none_for_a_non_ltx_2_index(self, tmp_path):
        write_model_index(tmp_path, "ZImagePipeline")

        assert ltx2_text_encoder_class(tmp_path) is None

    @pytest.mark.parametrize("entry", [None, ["transformers"], ["transformers", ""]])
    def test_is_none_for_a_malformed_text_encoder_entry(self, tmp_path, entry):
        index = {"_class_name": "LTX2Pipeline", "text_encoder": entry}
        touch(tmp_path / "model_index.json", json.dumps(index))

        assert ltx2_text_encoder_class(tmp_path) is None


class TestH3ModelIndex:
    def test_three_element_entries_count_as_weighted_components(self, tmp_path):
        write_h3_snapshot(tmp_path)

        assert set(model_files.weighted_components(tmp_path) or ()) == {"transformer", "vae"}

    def test_a_complete_snapshot_is_complete(self, tmp_path):
        write_h3_snapshot(tmp_path)

        assert has_complete_weights(tmp_path) is True

    def test_a_missing_transformer_shard_is_incomplete(self, tmp_path):
        write_h3_snapshot(tmp_path)
        (tmp_path / "transformer" / "model-00002-of-00003.safetensors").unlink()

        assert has_complete_weights(tmp_path) is False

    def test_the_index_alone_is_incomplete(self, tmp_path):
        write_h3_index(tmp_path)

        assert has_complete_weights(tmp_path) is False

    def test_a_partial_text_encoder_and_a_missing_reference_transformer_are_ignored(self, tmp_path):
        write_h3_snapshot(tmp_path)
        (tmp_path / "text_encoder" / "model-00005-of-00005.safetensors").unlink()
        for path in (tmp_path / "transformer_ref").glob("*.safetensors"):
            path.unlink()

        assert has_complete_weights(tmp_path) is True

    def test_another_pipeline_keeps_the_components_the_h3_pipeline_skips(self, tmp_path):
        write_h3_index(tmp_path, class_name="OtherModularPipeline")

        assert {"text_encoder", "transformer_ref"} <= set(model_files.weighted_components(tmp_path) or ())

    @pytest.mark.parametrize("entry", [["diffusers"], ["diffusers", "Model", {}, "extra"], [None, "Model", {}]])
    def test_malformed_entries_are_not_weighted(self, tmp_path, entry):
        touch(tmp_path / "model_index.json", json.dumps({"_class_name": "OtherPipeline", "unet": entry}))

        assert model_files.weighted_components(tmp_path) == ()

    def test_weightless_three_element_entries_are_not_weighted(self, tmp_path):
        spec = {"pretrained_model_name_or_path": "owner/h3"}
        index = {"_class_name": "OtherPipeline", "tokenizer": ["transformers", "Qwen2Tokenizer", spec], "scheduler": ["diffusers", "FlowMatchScheduler", spec]}
        touch(tmp_path / "model_index.json", json.dumps(index))

        assert model_files.weighted_components(tmp_path) == ()


class TestFindLocalModelDir:
    def test_a_pinned_ltx_2_5_mlx_pack_is_found_through_the_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
        snapshot = tmp_path / "models--owner--pack" / "snapshots" / _PIN
        for name in ("connector", "transformer-distilled", "vae_decoder_conv", "text_encoder"):
            touch(snapshot / f"{name}.safetensors")
        touch(snapshot / "text_encoder_config.json", "{}")

        assert find_local_model_dir(f"owner/pack@{_PIN}") == snapshot
        (snapshot / "text_encoder.safetensors").unlink()
        assert find_local_model_dir(f"owner/pack@{_PIN}") is None

    def test_local_directory_with_weights(self, tmp_path):
        touch(tmp_path / "model.safetensors")

        assert find_local_model_dir(str(tmp_path)) == tmp_path

    def test_unrecognised_reference_is_not_downloaded(self):
        assert find_local_model_dir("not a repo or path") is None

    def test_repo_id_uses_the_newest_complete_snapshot(self, tmp_path, monkeypatch):
        """Like mflux's offline resolution: an incomplete newer snapshot falls back to an older complete one."""
        import os

        snapshots = tmp_path / "models--owner--repo" / "snapshots"
        old = touch(snapshots / "old" / "model.safetensors").parent
        new = snapshots / "new"
        new.mkdir()
        touch(new / "model-00001-of-00002.safetensors")
        os.utime(old, (1, 1))
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))

        assert find_local_model_dir("owner/repo") == old
        touch(new / "model-00002-of-00002.safetensors")
        assert find_local_model_dir("owner/repo") == new

    def test_repo_id_missing_from_cache_is_not_downloaded(self, tmp_path, monkeypatch):
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))

        assert find_local_model_dir("owner/repo") is None

    def _cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
        return tmp_path / "models--owner--repo"

    def test_a_pinned_commit_resolves_only_that_snapshot(self, tmp_path, monkeypatch):
        repo = self._cache(tmp_path, monkeypatch)
        pinned = touch(repo / "snapshots" / "abc123" / "model.safetensors").parent
        touch(repo / "snapshots" / "def456" / "model.safetensors")

        assert find_local_model_dir("owner/repo@abc123") == pinned
        assert find_local_model_dir("owner/repo@fff999") is None

    def test_a_pinned_snapshot_with_only_the_files_diffusers_loads_counts_as_downloaded(self, tmp_path, monkeypatch):
        """The Hub's own offline lookup would call this incomplete, as the repo has files (a README) that were never fetched."""
        repo = self._cache(tmp_path, monkeypatch)
        snapshot = repo / "snapshots" / "abc123"
        write_model_index(snapshot)
        for component in ("transformer", "text_encoder", "vae"):
            touch(snapshot / component / "model.safetensors")

        assert find_local_model_dir("owner/repo@abc123") == snapshot

    def test_the_pinned_ltx_2_5_snapshot_is_found_through_the_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
        snapshot = tmp_path / "models--unsloth--LTX-2.5-FP8" / "snapshots" / _PIN
        write_ltx25_snapshot(snapshot)

        assert find_local_model_dir(f"unsloth/LTX-2.5-FP8@{_PIN}") == snapshot
        (snapshot / "transformer" / "model-00001-of-00004.safetensors").unlink()
        assert find_local_model_dir(f"unsloth/LTX-2.5-FP8@{_PIN}") is None

    def test_a_pinned_tag_resolves_through_its_ref(self, tmp_path, monkeypatch):
        repo = self._cache(tmp_path, monkeypatch)
        snapshot = touch(repo / "snapshots" / "abc123" / "model.safetensors").parent
        (repo / "refs").mkdir()
        (repo / "refs" / "v2").write_text("abc123\n", encoding="utf-8")

        assert find_local_model_dir("owner/repo@v2") == snapshot

    def test_a_pinned_snapshot_without_all_its_weights_is_not_downloaded(self, tmp_path, monkeypatch):
        repo = self._cache(tmp_path, monkeypatch)
        touch(repo / "snapshots" / "abc123" / "model-00001-of-00002.safetensors")

        assert find_local_model_dir("owner/repo@abc123") is None


def test_model_weight_files_ignore_files_outside_index_components(tmp_path):
    write_model_index(tmp_path)
    for component in ("transformer", "text_encoder", "vae"):
        touch(tmp_path / component / "model.safetensors")
    touch(tmp_path / "single-file.safetensors")

    files = {path.relative_to(tmp_path).as_posix() for path in model_files.model_weight_files(tmp_path)}

    assert files == {"transformer/model.safetensors", "text_encoder/model.safetensors", "vae/model.safetensors"}


def test_root_single_file_counts_only_when_the_transformer_folder_is_empty(tmp_path):
    write_model_index(tmp_path, "Krea2Pipeline")
    root = touch(tmp_path / "turbo.safetensors")
    vae = touch(tmp_path / "vae" / "model.safetensors")

    assert set(model_files.model_weight_files(tmp_path)) == {root, vae}

    shard = touch(tmp_path / "transformer" / "model.safetensors")

    assert set(model_files.model_weight_files(tmp_path)) == {shard, vae}


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
        touch(tmp_path / name)

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
        touch(tmp_path / name)

    found = model_files.ltx_mlx_upsampler_file(tmp_path)

    assert (found.name if found else None) == expected


def test_precision_variants_are_ignored_next_to_plain_weights(tmp_path):
    touch(tmp_path / "transformer" / "diffusion_pytorch_model.safetensors")
    touch(tmp_path / "transformer" / "diffusion_pytorch_model.fp16.safetensors")
    touch(tmp_path / "vae" / "diffusion_pytorch_model.fp16-00001-of-00001.safetensors")  # only a variant: kept

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
        write_model_index(tmp_path)
        for component in ("text_encoder", "vae"):
            touch(tmp_path / component / "model.safetensors")

        assert has_complete_weights(tmp_path, only=None) is False

    def test_only_restricts_the_required_components(self, tmp_path):
        write_model_index(tmp_path)
        for component in ("text_encoder", "vae"):
            touch(tmp_path / component / "model.safetensors")

        assert has_complete_weights(tmp_path, only={"text_encoder", "vae"}) is True
        assert has_complete_weights(tmp_path, only={"transformer", "vae"}) is False

    def test_empty_set_requires_no_component(self, tmp_path):
        write_model_index(tmp_path)

        assert has_complete_weights(tmp_path, only=set()) is True

    def test_shard_completeness_still_applies(self, tmp_path):
        write_model_index(tmp_path)
        touch(tmp_path / "vae" / "model-00001-of-00002.safetensors")

        assert has_complete_weights(tmp_path, only=set()) is False
        assert has_complete_weights(tmp_path, only={"vae"}) is False

    def test_unreadable_index_still_fails(self, tmp_path):
        touch(tmp_path / "model_index.json", "{not json")

        assert has_complete_weights(tmp_path, only=set()) is False

    def test_missing_index_in_an_empty_folder_is_not_complete(self, tmp_path):
        assert has_complete_weights(tmp_path, only=set()) is False
        assert has_complete_weights(tmp_path, only={"vae"}) is False


class TestCachedSnapshotDirs:
    def test_lists_snapshots_newest_first(self, tmp_path, monkeypatch):
        import os

        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
        snapshots = tmp_path / "models--owner--repo" / "snapshots"
        old = touch(snapshots / "old" / "a").parent
        new = touch(snapshots / "new" / "a").parent
        os.utime(old, (1, 1))

        assert cached_snapshot_dirs("owner/repo", None) == [new, old]

    def test_revision_maps_through_refs_and_missing_repo_is_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
        repo = tmp_path / "models--owner--repo"
        snapshot = touch(repo / "snapshots" / "abc123" / "a").parent
        touch(repo / "refs" / "main", "abc123\n")

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
        touch(tmp_path / name)

    found = model_files.ltx_mlx_bundled_text_encoder(tmp_path)

    assert (found.name if found else None) == expected


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        (["text_encoder.safetensors", "text_encoder_config.json", "connector.safetensors"], True),
        (["text_encoder.safetensors"], False),
        (["text_encoder_config.json"], False),
        (["connector.safetensors"], False),
        ([], False),
    ],
)
def test_ltx_pack_file_list_bundles_a_text_encoder_with_both_files(files, expected):
    assert model_files.ltx_mlx_files_bundle_text_encoder(files) is expected


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
        touch(tmp_path / name)

    found = model_files.ltx_mlx_video_decoder_file(tmp_path)

    assert (found.name if found else None) == expected


class TestIsMossModel:
    @pytest.mark.parametrize(("class_name", "expected"), [("MOSSImagePipeline", True), ("Ideogram4Pipeline", False)])
    def test_reads_the_pipeline_class(self, tmp_path, class_name, expected):
        (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": class_name}))

        assert model_files.is_moss_model(tmp_path) is expected

    def test_a_folder_without_an_index_is_not_moss(self, tmp_path):
        assert model_files.is_moss_model(tmp_path) is False
