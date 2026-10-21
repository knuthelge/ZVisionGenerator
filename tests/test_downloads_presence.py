"""Tests for offline presence of download targets and partial-file cleanup."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zvisiongenerator.downloads.presence import remove_partial_files, target_present_offline
from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils import model_files

_INSTANT_PATTERNS = ("model_index.json", "transformer/config.json", "LICENSE.md", "NOTICE", "scheduler/*", "text_encoder/*", "tokenizer/*", "vae/*")
_CUDA_KLEIN_PATTERNS = ("model_index.json", "scheduler/*", "text_encoder/*.safetensors", "tokenizer/*", "transformer/*.safetensors", "vae/*.safetensors")
_MFLUX_KLEIN_PATTERNS = ("text_encoder/*.safetensors", "tokenizer/*.json", "transformer/*.safetensors", "vae/*.safetensors")
_MFLUX_KREA_PATTERNS = ("turbo.safetensors", "text_encoder/*.safetensors", "tokenizer/*", "vae/*.safetensors")


def _touch(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_index(snapshot: Path, class_name: str = "Flux2KleinPipeline") -> None:
    index = {
        "_class_name": class_name,
        "transformer": ["diffusers", "Transformer"],
        "text_encoder": ["transformers", "Qwen3Model"],
        "vae": ["diffusers", "AutoencoderKL"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "tokenizer": ["transformers", "Qwen2Tokenizer"],
    }
    _touch(snapshot / "model_index.json", json.dumps(index))


def _components_snapshot(snapshot: Path, *, transformer: bool = False, class_name: str = "Flux2KleinPipeline") -> Path:
    _write_index(snapshot, class_name)
    _touch(snapshot / "transformer" / "config.json")
    _touch(snapshot / "scheduler" / "scheduler_config.json")
    _touch(snapshot / "tokenizer" / "tokenizer.json")
    _touch(snapshot / "text_encoder" / "model.safetensors")
    _touch(snapshot / "vae" / "diffusion_pytorch_model.safetensors")
    if transformer:
        _touch(snapshot / "transformer" / "diffusion_pytorch_model.safetensors")
    return snapshot


@pytest.fixture
def cache(tmp_path, monkeypatch) -> Path:
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(model_files, "huggingface_cache_dir", lambda: cache_dir)
    return cache_dir


def _snapshot(cache: Path, repo_id: str = "owner/repo", commit: str = "c0ffee") -> Path:
    return cache / f"models--{repo_id.replace('/', '--')}" / "snapshots" / commit


class TestLocalDirTargets:
    def test_literal_patterns_present(self, tmp_path):
        _touch(tmp_path / "a.safetensors")
        _touch(tmp_path / "sub" / "b.json")
        target = DownloadTarget("o/r", allow_patterns=("a.safetensors", "sub/b.json"), local_dir=str(tmp_path))

        assert target_present_offline(target) is True

    def test_missing_file_absent(self, tmp_path):
        _touch(tmp_path / "a.safetensors")
        target = DownloadTarget("o/r", allow_patterns=("a.safetensors", "b.json"), local_dir=str(tmp_path))

        assert target_present_offline(target) is False

    def test_glob_or_missing_patterns_absent(self, tmp_path):
        _touch(tmp_path / "a.safetensors")

        assert target_present_offline(DownloadTarget("o/r", allow_patterns=("*.safetensors",), local_dir=str(tmp_path))) is False
        assert target_present_offline(DownloadTarget("o/r", local_dir=str(tmp_path))) is False


class TestLiteralCacheTargets:
    def test_every_file_must_be_cached(self, tmp_path):
        existing = _touch(tmp_path / "blob")
        lookups: list = []

        def cached_file(repo_id, filename, *, revision=None):
            lookups.append((repo_id, filename, revision))
            return str(existing) if filename != "missing.json" else None

        present = DownloadTarget("o/r", revision="v1", allow_patterns=("a.json", "b.json"))
        absent = DownloadTarget("o/r", allow_patterns=("a.json", "missing.json"))

        assert target_present_offline(present, cached_file=cached_file) is True
        assert lookups == [("o/r", "a.json", "v1"), ("o/r", "b.json", "v1")]
        assert target_present_offline(absent, cached_file=cached_file) is False

    def test_nonexistent_path_or_sentinel_is_absent(self, tmp_path):
        target = DownloadTarget("o/r", allow_patterns=("a.json",))

        assert target_present_offline(target, cached_file=lambda *a, **k: str(tmp_path / "gone")) is False
        assert target_present_offline(target, cached_file=lambda *a, **k: object()) is False


class TestWholeRepoTargets:
    def test_complete_snapshot_present(self, cache):
        snapshot = _components_snapshot(_snapshot(cache), transformer=True)

        assert snapshot.is_dir()
        assert target_present_offline(DownloadTarget("owner/repo")) is True

    def test_missing_shard_absent(self, cache):
        snapshot = _components_snapshot(_snapshot(cache), transformer=True)
        (snapshot / "text_encoder" / "model.safetensors").unlink()
        _touch(snapshot / "text_encoder" / "model-00001-of-00002.safetensors")

        assert target_present_offline(DownloadTarget("owner/repo")) is False

    def test_no_snapshot_absent(self, cache):
        assert target_present_offline(DownloadTarget("owner/repo")) is False

    def test_injected_snapshots_are_used(self, tmp_path):
        snapshot = _components_snapshot(tmp_path / "snap", transformer=True)
        seen: list = []

        def snapshots(repo_id, revision):
            seen.append((repo_id, revision))
            return [snapshot]

        assert target_present_offline(DownloadTarget("o/r", revision="v2"), snapshots=snapshots) is True
        assert seen == [("o/r", "v2")]

    def test_revision_ref_is_respected(self, cache):
        repo_dir = cache / "models--owner--repo"
        _components_snapshot(repo_dir / "snapshots" / "good", transformer=True)
        _components_snapshot(repo_dir / "snapshots" / "partial")
        _touch(repo_dir / "refs" / "stable", "good")
        _touch(repo_dir / "refs" / "wip", "partial")

        assert target_present_offline(DownloadTarget("owner/repo", revision="stable")) is True
        assert target_present_offline(DownloadTarget("owner/repo", revision="wip")) is False
        assert target_present_offline(DownloadTarget("owner/repo", revision="unknown")) is False


class TestGlobCacheTargets:
    def test_instant_components_present_without_transformer_weights(self, cache):
        _components_snapshot(_snapshot(cache))
        _touch(_snapshot(cache) / "LICENSE.md")
        _touch(_snapshot(cache) / "NOTICE")

        assert target_present_offline(DownloadTarget("owner/repo", allow_patterns=_INSTANT_PATTERNS)) is True

    def test_optional_literal_files_are_not_checked(self, cache):
        _components_snapshot(_snapshot(cache))

        assert target_present_offline(DownloadTarget("owner/repo", allow_patterns=_INSTANT_PATTERNS)) is True

    def test_covered_component_without_weights_absent(self, cache):
        snapshot = _components_snapshot(_snapshot(cache))
        (snapshot / "vae" / "diffusion_pytorch_model.safetensors").unlink()

        assert target_present_offline(DownloadTarget("owner/repo", allow_patterns=_INSTANT_PATTERNS)) is False

    def test_covered_transformer_without_weights_absent(self, cache):
        snapshot = _components_snapshot(_snapshot(cache))
        assert (snapshot / "transformer" / "config.json").is_file()

        assert target_present_offline(DownloadTarget("owner/repo", allow_patterns=_CUDA_KLEIN_PATTERNS)) is False
        assert target_present_offline(DownloadTarget("owner/repo", allow_patterns=_MFLUX_KLEIN_PATTERNS)) is False

    @pytest.mark.parametrize("patterns", [_INSTANT_PATTERNS, _CUDA_KLEIN_PATTERNS, _MFLUX_KLEIN_PATTERNS])
    def test_snapshot_without_a_model_index_is_not_complete(self, cache, patterns):
        # An interrupted download can leave small files (scheduler, tokenizer) but no weights and no index yet.
        _touch(_snapshot(cache) / "scheduler" / "scheduler_config.json")
        _touch(_snapshot(cache) / "tokenizer" / "tokenizer.json")

        assert target_present_offline(DownloadTarget("owner/repo", allow_patterns=patterns)) is False

    @pytest.mark.parametrize("patterns", [_INSTANT_PATTERNS, _CUDA_KLEIN_PATTERNS])
    def test_empty_snapshot_folder_absent(self, cache, patterns):
        _snapshot(cache).mkdir(parents=True)

        assert target_present_offline(DownloadTarget("owner/repo", allow_patterns=patterns)) is False

    def test_cuda_klein_plan_present_without_root_configs(self, cache):
        snapshot = _components_snapshot(_snapshot(cache), transformer=True)
        assert not (snapshot / "config.json").exists()
        patterns = (*_CUDA_KLEIN_PATTERNS, "config.json", "README.md")

        assert target_present_offline(DownloadTarget("owner/repo", allow_patterns=patterns)) is True

    def test_mflux_klein_patterns_on_diffusers_snapshot_present(self, cache):
        _components_snapshot(_snapshot(cache), transformer=True)

        assert target_present_offline(DownloadTarget("owner/repo", allow_patterns=_MFLUX_KLEIN_PATTERNS)) is True

    def test_missing_shard_in_covered_component_absent(self, cache):
        snapshot = _components_snapshot(_snapshot(cache), transformer=True)
        (snapshot / "transformer" / "diffusion_pytorch_model.safetensors").unlink()
        _touch(snapshot / "transformer" / "diffusion_pytorch_model-00001-of-00003.safetensors")
        _touch(snapshot / "transformer" / "diffusion_pytorch_model-00002-of-00003.safetensors")

        assert target_present_offline(DownloadTarget("owner/repo", allow_patterns=_MFLUX_KLEIN_PATTERNS)) is False

    def test_mflux_krea_requires_its_literal_weights_file(self, cache):
        snapshot = _components_snapshot(_snapshot(cache), class_name="Krea2Pipeline")
        target = DownloadTarget("owner/repo", allow_patterns=_MFLUX_KREA_PATTERNS)

        assert target_present_offline(target) is False

        _touch(snapshot / "turbo.safetensors")

        assert target_present_offline(target) is True

    def test_no_snapshot_absent(self, cache):
        assert target_present_offline(DownloadTarget("owner/repo", allow_patterns=_INSTANT_PATTERNS)) is False


class TestRemovePartialFiles:
    def test_cache_target_removes_only_blob_incompletes(self, tmp_path):
        repo_dir = tmp_path / "models--owner--repo"
        _touch(repo_dir / "blobs" / "aaa.incomplete")
        _touch(repo_dir / "blobs" / "bbb.incomplete")
        keep_blob = _touch(repo_dir / "blobs" / "ccc")
        keep_snapshot = _touch(repo_dir / "snapshots" / "x" / "d.incomplete")
        other = _touch(tmp_path / "models--owner--other" / "blobs" / "eee.incomplete")

        assert remove_partial_files(DownloadTarget("owner/repo"), cache_dir=tmp_path) == 2

        assert not list((repo_dir / "blobs").glob("*.incomplete"))
        assert keep_blob.exists()
        assert keep_snapshot.exists()
        assert other.exists()

    def test_cache_dir_defaults_to_the_hub_cache(self, cache, monkeypatch):
        monkeypatch.setattr("zvisiongenerator.downloads.presence.huggingface_cache_dir", lambda: cache)
        partial = _touch(cache / "models--owner--repo" / "blobs" / "a.incomplete")

        assert remove_partial_files(DownloadTarget("owner/repo")) == 1
        assert not partial.exists()

    def test_local_dir_target_removes_nested_incompletes(self, tmp_path):
        download = tmp_path / ".cache" / "huggingface" / "download"
        _touch(download / "a.safetensors.incomplete")
        _touch(download / "sub" / "b.incomplete")
        keep_meta = _touch(download / "a.safetensors.metadata")
        keep_file = _touch(tmp_path / "c.incomplete")

        assert remove_partial_files(DownloadTarget("o/r", local_dir=str(tmp_path))) == 2

        assert keep_meta.exists()
        assert keep_file.exists()
        assert not list(download.rglob("*.incomplete"))

    def test_missing_folders_return_zero(self, tmp_path):
        assert remove_partial_files(DownloadTarget("o/r"), cache_dir=tmp_path) == 0
        assert remove_partial_files(DownloadTarget("o/r", local_dir=str(tmp_path / "nope"))) == 0

    def test_oserror_on_one_file_is_skipped(self, tmp_path, monkeypatch):
        repo_dir = tmp_path / "models--o--r"
        stuck = _touch(repo_dir / "blobs" / "a.incomplete")
        _touch(repo_dir / "blobs" / "b.incomplete")
        real_unlink = Path.unlink

        def flaky_unlink(self, *args, **kwargs):
            if self == stuck:
                raise PermissionError("locked")
            return real_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", flaky_unlink)

        assert remove_partial_files(DownloadTarget("o/r"), cache_dir=tmp_path) == 1
        assert stuck.exists()
