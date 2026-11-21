"""Tests for whether the Web UI sees a model as downloaded."""

from __future__ import annotations

import ctypes.util  # noqa: F401 - the macOS modules import it lazily; under the patched platform it would look for macOS-only parts
import json
import struct
from pathlib import Path

import pytest

from tests._factories import load_packaged_config

from zvisiongenerator.utils.model_metadata import builtin_aliases
from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.web import model_status as model_status_module
from zvisiongenerator.web.model_status import model_downloaded, nothing_to_download

_GIB = 1024**3


def _finder(mapping: dict[str, Path]):
    return lambda reference: mapping.get(reference)


def _write_sized(path: Path, gib: float) -> Path:
    """Write a header-only safetensors file whose weights load as *gib* GiB of bfloat16."""
    header = json.dumps({"w": {"dtype": "BF16", "shape": [int(gib * _GIB / 2 / 1024), 1024], "data_offsets": [0, 0]}}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header)
    return path


def _make_mlx_pack(root: Path, *, bundled: bool) -> Path:
    """Build a fake LTX MLX pack: LTX-2.5 (bundled Gemma 4, conv decoder) or LTX-2.3 (plain decoder)."""
    pack = root / ("ltx25" if bundled else "ltx23")
    _write_sized(pack / "transformer-distilled-1.1.safetensors", 10.5)
    _write_sized(pack / "connector.safetensors", 5.9)
    _write_sized(pack / ("vae_decoder_conv.safetensors" if bundled else "vae_decoder.safetensors"), 0.75)
    if bundled:
        _write_sized(pack / "text_encoder.safetensors", 7.5)
        (pack / "text_encoder_config.json").write_text("{}", encoding="utf-8")
    return pack


_GEMMA_REPO = "mlx-community/gemma-3-12b-it-4bit"


@pytest.fixture
def darwin(monkeypatch):
    import zvisiongenerator.backends as backends

    monkeypatch.setattr(backends.sys, "platform", "darwin")


def _ltx_downloaded(name, pack, *, gemma=None, present=None):
    """Return whether the macOS alias *name* whose pinned pack is the folder *pack* is downloaded; *gemma* is the Gemma 3 folder when it is on disk."""
    config = load_packaged_config()
    ref = builtin_aliases()[name]["darwin"]
    folders = {ref: pack, **({_GEMMA_REPO: gemma} if gemma is not None else {})}
    present = present or (lambda target: target.repo_id != _GEMMA_REPO or gemma is not None)
    return model_downloaded(ref, kind="video", find_local_dir=_finder(folders), model_name=name, app_config=config, target_present=present)


@pytest.mark.usefixtures("darwin")
class TestMlxVideoDownloaded:
    def _gemma_dir(self, root: Path) -> Path:
        _write_sized(root / "gemma" / "model-00001-of-00001.safetensors", 7.5)
        return root / "gemma"

    def test_a_bundled_pack_is_downloaded_without_gemma_3(self, tmp_path):
        pack = _make_mlx_pack(tmp_path, bundled=True)
        seen = []

        downloaded = _ltx_downloaded("ltx25-4", pack, present=lambda target: (seen.append(target.repo_id), True)[1])

        assert downloaded is True
        assert seen == ["MarceloFSO/ltx-2.5-mlx-q4-fast"]

    def test_a_bundled_pack_is_downloaded_the_same_with_or_without_a_gemma_3_copy(self, tmp_path):
        pack = _make_mlx_pack(tmp_path, bundled=True)

        assert _ltx_downloaded("ltx25-4", pack) == _ltx_downloaded("ltx25-4", pack, gemma=self._gemma_dir(tmp_path))

    @pytest.mark.parametrize("unpinned", [False, True])
    def test_a_custom_name_for_the_bundled_pack_is_downloaded_with_no_gemma_3_cached(self, tmp_path, unpinned):
        pack = _make_mlx_pack(tmp_path, bundled=True)
        config = load_packaged_config()
        ref = builtin_aliases()["ltx25-4"]["darwin"]
        ref = ref.split("@")[0] if unpinned else ref
        seen = []

        downloaded = model_downloaded(
            ref,
            kind="video",
            find_local_dir=_finder({ref: pack}),
            model_name="my-custom-ltx",
            app_config=config,
            target_present=lambda target: (seen.append(target.repo_id), True)[1],
        )

        assert downloaded is True
        assert seen == ["MarceloFSO/ltx-2.5-mlx-q4-fast"]

    def test_a_2_3_pack_without_gemma_3_is_not_downloaded(self, tmp_path):
        pack = _make_mlx_pack(tmp_path, bundled=False)

        assert _ltx_downloaded("ltx-4", pack) is False

    def test_a_2_3_pack_with_a_missing_file_is_not_downloaded_even_with_gemma_3(self, tmp_path):
        pack = _make_mlx_pack(tmp_path, bundled=False)
        packs = []

        def present(target):
            packs.append(target.allow_patterns)
            return target.repo_id == _GEMMA_REPO  # the pack's literal files are not all cached

        assert _ltx_downloaded("ltx-4", pack, gemma=self._gemma_dir(tmp_path), present=present) is False
        assert any(patterns and "transformer-distilled-1.1.safetensors" in patterns for patterns in packs)

    def test_a_2_3_pack_with_gemma_3_is_downloaded(self, tmp_path):
        pack = _make_mlx_pack(tmp_path, bundled=False)

        assert _ltx_downloaded("ltx-4", pack, gemma=self._gemma_dir(tmp_path)) is True

    def test_a_pack_that_is_not_on_disk_is_not_downloaded(self):
        config = load_packaged_config()
        ref = builtin_aliases()["ltx-8"]["darwin"]

        assert model_downloaded(ref, kind="video", find_local_dir=_finder({}), model_name="ltx-8", app_config=config, target_present=lambda target: False) is False

    def test_a_local_mlx_folder_with_only_the_marker_file_is_not_downloaded(self, tmp_path):
        folder = tmp_path / "ltx-mlx"
        _write_sized(folder / "connector.safetensors", 0.1)

        assert model_downloaded(str(folder), kind="video", model_name="ltx-4", app_config=load_packaged_config(), target_present=lambda target: True) is False

    def test_a_local_folder_without_weights_is_not_downloaded(self, tmp_path):
        folder = tmp_path / "ltx-local"
        folder.mkdir()

        assert model_downloaded(str(folder), kind="video", model_name="ltx-4", app_config=load_packaged_config(), target_present=lambda target: True) is False


@pytest.mark.usefixtures("darwin")
class TestTrimmedPackOnDisk:
    """The Models page reads a pinned pack from the real Hub cache layout: every listed file must be there."""

    @pytest.fixture
    def pack(self, tmp_path, monkeypatch):
        import huggingface_hub.constants

        from zvisiongenerator.utils.ltx_facts import ltx_load_options

        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "cache"))
        files = ltx_load_options("ltx-4", "darwin", upscale=False)["mlx_packs"][builtin_aliases()["ltx-4"]["darwin"]]
        snapshot = tmp_path / "cache" / f"models--{files.repo_id.replace('/', '--')}" / "snapshots" / files.revision
        for name in files.files:
            (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
            (snapshot / name).write_bytes(b"x")
        return snapshot

    @staticmethod
    def _status(snapshot, tmp_path):
        from zvisiongenerator.downloads.presence import target_present_offline

        config = load_packaged_config()
        ref = builtin_aliases()["ltx-4"]["darwin"]
        gemma = tmp_path / "gemma"
        gemma.mkdir(exist_ok=True)
        return model_downloaded(
            ref,
            kind="video",
            find_local_dir=_finder({ref: snapshot, _GEMMA_REPO: gemma}),
            model_name="ltx-4",
            app_config=config,
            target_present=lambda target: target.repo_id == _GEMMA_REPO or target_present_offline(target),
        )

    def test_a_folder_with_every_listed_file_is_downloaded(self, pack, tmp_path):
        assert self._status(pack, tmp_path) is True

    def test_a_folder_missing_one_listed_file_is_not_downloaded(self, pack, tmp_path):
        (pack / "audio_vae.safetensors").unlink()

        assert self._status(pack, tmp_path) is False


class TestH3Downloaded:
    """MiniMax H3 is downloaded when every target of its plan is on disk."""

    @pytest.fixture
    def darwin(self, monkeypatch):
        import zvisiongenerator.backends as backends

        monkeypatch.setattr(backends.sys, "platform", "darwin")

    @staticmethod
    def _downloaded(present):
        return model_downloaded(
            "MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe",
            kind="video",
            find_local_dir=lambda reference: None,
            model_name="h3",
            app_config=load_packaged_config(),
            family="h3",
            target_present=present,
        )

    def test_macos_checks_the_three_targets_of_the_mac_plan(self, darwin):
        seen = []

        def present(target):
            seen.append((target.repo_id, target.revision))
            return True

        assert self._downloaded(present) is True
        assert [repo for repo, _revision in seen] == ["water1234/MiniMax-H3-MLX-Argus-Calibrated-INT8", "MiniMaxAI/MiniMax-H3", "lightx2v/Minimax-h3-Turbo"]

    def test_one_absent_target_is_not_downloaded(self, darwin):
        assert self._downloaded(lambda target: target.repo_id != "MiniMaxAI/MiniMax-H3") is False

    def test_a_discrete_gpu_checks_the_targets_of_its_plan(self, monkeypatch):
        import zvisiongenerator.backends as backends

        monkeypatch.setattr(backends.sys, "platform", "linux")
        seen = []

        assert self._downloaded(lambda target: (seen.append(target.repo_id), True)[1]) is True
        assert seen == ["MiniMaxAI/MiniMax-H3", "unsloth/Qwen3-VL-32B-Instruct-bnb-4bit", "lightx2v/Minimax-h3-Turbo"]

    def test_the_status_agrees_with_the_backends_offline_resolution(self, darwin, tmp_path, monkeypatch):
        """SC-10: the same cache decides ``downloaded`` and whether the backend would download."""
        import huggingface_hub.constants

        from zvisiongenerator.backends.mac.h3_video import resolve_mlx_files
        from zvisiongenerator.downloads.plans import video_plan
        from zvisiongenerator.utils import model_files
        from zvisiongenerator.utils.h3_preset import h3_mlx_preset, h3_turbo_lora

        cache = tmp_path / "cache"
        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))
        monkeypatch.setattr(model_files, "huggingface_cache_dir", lambda: cache)
        config = load_packaged_config()
        ref = "MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe"

        def no_network(**_kwargs):
            raise AssertionError("downloaded")

        def resolves() -> bool:
            try:
                resolve_mlx_files(ref, h3_mlx_preset(), lora=h3_turbo_lora("darwin")[0], snapshot_download=no_network)
            except AssertionError:
                return False
            return True

        def status() -> bool:
            return model_downloaded(ref, kind="video", find_local_dir=lambda reference: None, model_name="h3", app_config=config, family="h3")

        assert (status(), resolves()) == (False, False)
        plan = video_plan(ref, family="h3", config=config, model_name="h3", platform_key="darwin", upscale=False)
        for target in plan:
            for name in target.allow_patterns:
                path = cache / f"models--{target.repo_id.replace('/', '--')}" / "snapshots" / target.revision / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
        assert (status(), resolves()) == (True, True)
        (cache / "models--MiniMaxAI--MiniMax-H3" / "snapshots" / "42ed227ee7df40d41602854ae760620d6eb651fe" / "FL2VA" / "audio_vae" / "model.safetensors").unlink()
        assert (status(), resolves()) == (False, False)


class TestModelDownloaded:
    def test_a_missing_model_is_not_downloaded(self):
        assert model_downloaded("owner/repo", kind="image", find_local_dir=_finder({}), app_config={}) is False

    def test_a_model_whose_folder_resolves_is_downloaded(self, tmp_path):
        assert model_downloaded("owner/repo", kind="image", find_local_dir=_finder({"owner/repo": tmp_path}), app_config={}) is True


class TestCudaVideoDownloaded:
    @pytest.fixture
    def linux(self, monkeypatch):
        import zvisiongenerator.backends as backends

        monkeypatch.setattr(backends.sys, "platform", "linux")

    @staticmethod
    def _downloaded(tmp_path, *, present=lambda target: True, name="ltx-4"):
        """Return whether the Windows/Linux LTX alias *name*, whose repo folder is on disk, is downloaded given the plan targets *present* says are there."""
        config = load_packaged_config()
        ref = builtin_aliases()[name]["linux"]
        return model_downloaded(ref, kind="video", find_local_dir=_finder({ref: tmp_path}), model_name=name, app_config=config, target_present=present)

    @pytest.mark.usefixtures("linux")
    def test_a_video_model_with_every_plan_target_present_is_downloaded(self, tmp_path):
        seen = []

        def present(target):
            seen.append(target.repo_id)
            return True

        assert self._downloaded(tmp_path, present=present) is True
        assert seen == ["diffusers/LTX-2.3-Distilled-Diffusers", "Lightricks/LTX-2.3"]

    @pytest.mark.usefixtures("linux")
    def test_a_video_model_with_the_repo_but_not_the_transformer_file_is_not_downloaded(self, tmp_path):
        assert self._downloaded(tmp_path, present=lambda target: target.repo_id == "diffusers/LTX-2.3-Distilled-Diffusers") is False

    @pytest.mark.usefixtures("linux")
    def test_a_local_diffusers_folder_without_weights_is_not_downloaded(self, tmp_path):
        folder = tmp_path / "ltx-local"
        folder.mkdir()
        (folder / "model_index.json").write_text("{}", encoding="utf-8")

        assert model_downloaded(str(folder), kind="video", model_name="ltx-4", app_config=load_packaged_config(), target_present=lambda target: True) is False

    @pytest.mark.usefixtures("linux")
    def test_a_video_model_with_no_local_folder_is_not_downloaded_even_when_its_plan_is_present(self):
        config = load_packaged_config()
        ref = builtin_aliases()["ltx-4"]["linux"]

        assert model_downloaded(ref, kind="video", find_local_dir=_finder({}), model_name="ltx-4", app_config=config, target_present=lambda target: True) is False


class TestNothingToDownload:
    """One rule for every surface: a model has nothing to fetch when it is downloaded, or built on first use with all its sources on disk."""

    _SOURCES = (DownloadTarget("owner/base"), DownloadTarget("owner/instant"))

    @pytest.fixture
    def built_on_first_use(self, monkeypatch):
        monkeypatch.setattr(model_status_module, "built_model_plan", lambda path, **_: self._SOURCES if path == "/models/ideo" else None)

    def test_a_downloaded_model_has_nothing_to_download(self):
        assert nothing_to_download("/models/zit", downloaded=True, target_present=lambda target: False) is True

    def test_a_model_that_is_not_downloaded_and_not_built_on_first_use_still_needs_its_download(self, built_on_first_use):
        assert nothing_to_download("/models/zit", downloaded=False, target_present=lambda target: True) is False

    def test_an_unbuilt_model_with_every_source_on_disk_needs_none(self, built_on_first_use):
        assert nothing_to_download("/models/ideo", downloaded=False, target_present=lambda target: True) is True

    def test_an_unbuilt_model_missing_one_source_still_needs_its_download(self, built_on_first_use):
        assert nothing_to_download("/models/ideo", downloaded=False, target_present=lambda target: target.repo_id != "owner/instant") is False
