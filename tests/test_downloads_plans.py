"""Tests for composing download plans: Instant sources, image/video/enhancer dispatch and SeedVR2 tier files."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled

from zvisiongenerator.downloads.plans import enhancer_plan, image_plan, instant_plan, seedvr2_plan, video_plan
from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils.ideogram4_instant import COMPONENT_PATTERNS, instant_downloads_dir, instant_source, write_build_record
from zvisiongenerator.utils.seedvr2 import FP8_WEIGHTS, MLX_WEIGHTS, SEEDVR2_MIXED_REPO, SEEDVR2_REPO

_CONFIG = {
    "ideogram4_instant": {
        "transformer": {"repo": "owner/instant", "revision": "a" * 40, "filename": "instant.safetensors"},
        "components": {"linux": "owner/base-nf4@" + "b" * 40, "darwin": "owner/base-fp8@" + "c" * 40},
        "transformer_format": {"linux": "bf16", "darwin": "fp8"},
    },
    "video_model_presets": {
        "ltx": {
            "diffusers": {
                "transformer": {"repo": "owner/ltx-transformer", "filename": "t.safetensors"},
                "latent_upsampler": {"repo": "owner/ltx-up", "filename": "u.safetensors"},
            }
        }
    },
}


def _patch_backend(monkeypatch, name: str, result=None, side_effect=None) -> MagicMock:
    fake = MagicMock(return_value=result, side_effect=side_effect)
    monkeypatch.setattr(f"zvisiongenerator.backends.{name}", fake, raising=False)
    return fake


@pytest.fixture
def models_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
    folder = tmp_path / "models"
    folder.mkdir()
    return folder


class TestInstantPlan:
    def test_a_model_that_is_not_the_instant_folder_is_not_planned(self, models_dir):
        assert instant_plan(str(models_dir / "zit"), config=_CONFIG, platform_key="linux") is None
        assert instant_plan("owner/repo", config=_CONFIG, platform_key="linux") is None

    def test_a_built_folder_needs_nothing(self, models_dir):
        folder = models_dir / "ideo"
        folder.mkdir()
        write_build_record(folder, instant_source(_CONFIG, "linux"))

        assert instant_plan(str(folder), config=_CONFIG, platform_key="linux") == ()

    def test_an_unbuilt_folder_needs_the_components_and_the_transformer_file(self, models_dir):
        folder = models_dir / "ideo"

        plan = instant_plan(str(folder), config=_CONFIG, platform_key="linux")

        assert plan == (
            DownloadTarget("owner/base-nf4", "b" * 40, COMPONENT_PATTERNS),
            DownloadTarget("owner/instant", "a" * 40, ("instant.safetensors",), local_dir=str(instant_downloads_dir(folder))),
        )

    def test_the_platform_picks_the_components_repo(self, models_dir):
        plan = instant_plan(str(models_dir / "ideo"), config=_CONFIG, platform_key="darwin")

        assert plan is not None
        assert (plan[0].repo_id, plan[0].revision) == ("owner/base-fp8", "c" * 40)

    def test_a_build_from_other_sources_is_planned_again(self, models_dir):
        folder = models_dir / "ideo"
        folder.mkdir()
        write_build_record(folder, instant_source(_CONFIG, "linux"))

        plan = instant_plan(str(folder), config=_CONFIG, platform_key="darwin")

        assert plan is not None
        assert len(plan) == 2

    def test_the_downloads_folder_is_the_hidden_sibling(self, tmp_path):
        assert instant_downloads_dir(tmp_path / "ideo") == tmp_path / ".ideo.download"


class TestImagePlan:
    def test_the_instant_folder_is_planned_before_the_backend(self, monkeypatch, models_dir):
        backend = _patch_backend(monkeypatch, "image_download_plan", ())

        plan = image_plan(str(models_dir / "ideo"), family="ideogram4", config=_CONFIG, platform_key="linux")

        assert plan is not None
        assert len(plan) == 2
        backend.assert_not_called()

    def test_without_a_config_the_backend_plans_even_the_instant_folder(self, monkeypatch, models_dir):
        backend = _patch_backend(monkeypatch, "image_download_plan", ())

        assert image_plan(str(models_dir / "ideo"), family="ideogram4", config=None, platform_key="linux") == ()
        backend.assert_called_once_with(str(models_dir / "ideo"), family="ideogram4")

    def test_another_model_goes_to_the_backend(self, monkeypatch, models_dir):
        target = DownloadTarget("owner/zit", None, ("vae/*",))
        backend = _patch_backend(monkeypatch, "image_download_plan", (target,))

        plan = image_plan("owner/zit", family="zimage", config=_CONFIG, platform_key="linux")

        assert plan == (target,)
        backend.assert_called_once_with("owner/zit", family="zimage")

    @pytest.mark.parametrize(
        "error",
        [LookupError("no model_index.json"), httpx.ConnectError("down"), httpx.ReadTimeout("slow"), OfflineModeIsEnabled("offline"), LocalEntryNotFoundError("not cached")],
    )
    def test_an_unreadable_file_list_cannot_be_planned(self, monkeypatch, models_dir, error):
        _patch_backend(monkeypatch, "image_download_plan", side_effect=error)

        assert image_plan("owner/zit", family="zimage", config=None, platform_key="linux") is None

    def test_other_errors_propagate(self, monkeypatch, models_dir):
        _patch_backend(monkeypatch, "image_download_plan", side_effect=RuntimeError("unsupported platform"))

        with pytest.raises(RuntimeError, match="unsupported"):
            image_plan("owner/zit", family="zimage", config=None, platform_key="linux")


class TestVideoPlan:
    def test_the_configured_companions_are_passed_to_the_backend(self, monkeypatch):
        target = DownloadTarget("owner/ltx")
        backend = _patch_backend(monkeypatch, "video_download_plan", (target,))

        plan = video_plan("owner/ltx", family="ltx", config=_CONFIG, model_name="ltx", upscale=True)

        assert plan == (target,)
        backend.assert_called_once_with(
            "owner/ltx",
            family="ltx",
            transformer_source=("owner/ltx-transformer", "t.safetensors"),
            latent_upsampler=("owner/ltx-up", "u.safetensors"),
            upscale=True,
            repo_ignore=(),
            text_encoder_bundled=False,
        )

    def test_the_versions_repo_ignore_is_passed_to_the_backend(self, monkeypatch):
        backend = _patch_backend(monkeypatch, "video_download_plan", ())
        config = {"video_model_presets": {"ltx": {"versions": {"v": {"aliases": ["ltx-v"], "diffusers": {"repo_ignore": ["a/*", "b"]}}}}}}

        video_plan("owner/ltx", family="ltx", config=config, model_name="ltx-v", upscale=False)

        assert backend.call_args.kwargs["repo_ignore"] == ("a/*", "b")

    def test_without_upscale_no_upsampler_is_requested(self, monkeypatch):
        backend = _patch_backend(monkeypatch, "video_download_plan", ())

        video_plan("owner/ltx", family="ltx", config=_CONFIG, model_name="ltx", upscale=False)

        assert backend.call_args.kwargs["latent_upsampler"] is None
        assert backend.call_args.kwargs["upscale"] is False

    def test_a_malformed_config_propagates(self, monkeypatch):
        _patch_backend(monkeypatch, "video_download_plan", ())
        config = {"video_model_presets": {"ltx": {"diffusers": {"transformer": {"repo": ""}}}}}

        with pytest.raises(ValueError, match="transformer"):
            video_plan("owner/ltx", family="ltx", config=config, model_name="ltx", upscale=False)

    def test_network_errors_cannot_be_planned(self, monkeypatch):
        _patch_backend(monkeypatch, "video_download_plan", side_effect=httpx.ConnectError("down"))

        assert video_plan("owner/ltx", family="ltx", config=_CONFIG, model_name="ltx", upscale=False) is None


class TestSeedVR2Plan:
    def test_mlx_downloads_the_tier_files_into_the_cache(self, models_dir):
        plan = seedvr2_plan("seedvr2-3b", models_dir=models_dir, weights=MLX_WEIGHTS)

        assert plan == (DownloadTarget(SEEDVR2_REPO, None, ("seedvr2_ema_3b_fp16.safetensors", "ema_vae_fp16.safetensors")),)

    def test_a_current_stored_copy_needs_nothing(self, monkeypatch, models_dir):
        monkeypatch.setattr("zvisiongenerator.seedvr2_loader.seedvr2_downloaded", lambda model, folder, weights: True)

        assert seedvr2_plan("seedvr2-7b", models_dir=models_dir, weights=MLX_WEIGHTS) == ()

    def test_fp8_7b_downloads_one_target_per_repo_into_the_weights_folder(self, models_dir):
        folder = str(models_dir / ".seedvr2" / "seedvr2-7b@fp8")

        plan = seedvr2_plan("seedvr2-7b", models_dir=models_dir, weights=FP8_WEIGHTS)

        assert plan == (
            DownloadTarget(SEEDVR2_MIXED_REPO, None, ("seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",), local_dir=folder),
            DownloadTarget(SEEDVR2_REPO, None, ("ema_vae_fp16.safetensors",), local_dir=folder),
        )

    def test_fp8_3b_shares_one_repo(self, models_dir):
        plan = seedvr2_plan("seedvr2-3b", models_dir=models_dir, weights=FP8_WEIGHTS)

        assert len(plan) == 1
        assert plan[0].repo_id == SEEDVR2_REPO
        assert plan[0].allow_patterns == ("seedvr2_ema_3b_fp8_e4m3fn.safetensors", "ema_vae_fp16.safetensors")
        assert plan[0].local_dir == str(models_dir / ".seedvr2" / "seedvr2-3b@fp8")

    def test_downloaded_fp8_files_need_nothing(self, models_dir):
        folder = models_dir / ".seedvr2" / "seedvr2-3b@fp8"
        folder.mkdir(parents=True)
        for _repo, name in FP8_WEIGHTS.files["seedvr2-3b"]:
            (folder / name).write_bytes(b"x")

        assert seedvr2_plan("seedvr2-3b", models_dir=models_dir, weights=FP8_WEIGHTS) == ()

    def test_one_missing_fp8_file_is_still_planned(self, models_dir):
        folder = models_dir / ".seedvr2" / "seedvr2-3b@fp8"
        folder.mkdir(parents=True)
        (folder / "ema_vae_fp16.safetensors").write_bytes(b"x")

        assert len(seedvr2_plan("seedvr2-3b", models_dir=models_dir, weights=FP8_WEIGHTS)) == 1

    def test_the_planned_names_cover_mflux_patterns(self):
        mflux_weights = pytest.importorskip("mflux.models.seedvr2.weights.seedvr2_weight_definition")
        for model, definition in (("seedvr2-3b", mflux_weights.SeedVR2WeightDefinition3B), ("seedvr2-7b", mflux_weights.SeedVR2WeightDefinition7B)):
            planned = {name for _repo, name in MLX_WEIGHTS.files[model]}
            assert set(definition.get_download_patterns()) <= planned


class TestEnhancerPlan:
    def test_the_backend_dispatch_answers(self, monkeypatch):
        target = DownloadTarget("owner/enhancer", "rev", ("*.json",))
        backend = _patch_backend(monkeypatch, "enhancer_download_plan", (target,))

        assert enhancer_plan("owner/enhancer", "rev") == (target,)
        backend.assert_called_once_with("owner/enhancer", "rev")

    def test_a_local_folder_needs_nothing(self, monkeypatch, tmp_path):
        _patch_backend(monkeypatch, "enhancer_download_plan", ())

        assert enhancer_plan(str(tmp_path), None) == ()


def test_the_target_round_trips_through_json(models_dir):
    plan = instant_plan(str(models_dir / "ideo"), config=_CONFIG, platform_key="linux")

    assert plan is not None
    assert all(DownloadTarget.from_json(target.to_json()) == target for target in plan)
    assert json.loads(plan[1].to_json())["local_dir"].endswith(".ideo.download")


class TestThroughTheRealDispatch:
    """Call the composers without patching ``zvisiongenerator.backends``: the argument names must fit the real planners."""

    @pytest.fixture(autouse=True)
    def _cuda_platform(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "linux")

    def test_the_image_plan_reads_the_index_and_selects_the_component_folders(self, monkeypatch):
        from zvisiongenerator.backends.cuda import download_plan as cuda_plan

        seen: list[tuple[str, str | None]] = []

        def read_index(repo_id: str, revision: str | None) -> dict:
            seen.append((repo_id, revision))
            return {"_class_name": "Pipeline", "vae": ["diffusers", "AutoencoderKL"], "scheduler": ["diffusers", "Scheduler"], "safety_checker": [None, None]}

        monkeypatch.setattr(cuda_plan, "_read_hub_model_index", read_index)

        plan = image_plan("owner/zit@abc123", family="zimage", config=None, platform_key="linux")

        assert seen == [("owner/zit", "abc123")]
        assert plan is not None
        (target,) = plan
        assert (target.repo_id, target.revision, target.local_dir) == ("owner/zit", "abc123", None)
        assert target.allow_patterns is not None
        assert {"model_index.json", "vae/*", "scheduler/*"} <= set(target.allow_patterns)
        assert "safety_checker/*" not in target.allow_patterns
        assert target.ignore_patterns == cuda_plan.DIFFUSERS_IGNORE

    def test_an_unreadable_index_leaves_the_image_plan_unmade(self, monkeypatch):
        from zvisiongenerator.backends.cuda import download_plan as cuda_plan

        def read_index(repo_id: str, revision: str | None) -> dict:
            raise httpx.ConnectError("down")

        monkeypatch.setattr(cuda_plan, "_read_hub_model_index", read_index)

        assert image_plan("owner/zit", family="zimage", config=None, platform_key="linux") is None

    def test_a_local_folder_and_an_instant_folder_need_no_index(self, monkeypatch, models_dir, tmp_path):
        from zvisiongenerator.backends.cuda import download_plan as cuda_plan

        def read_index(repo_id: str, revision: str | None) -> dict:
            raise AssertionError("the index must not be read")

        monkeypatch.setattr(cuda_plan, "_read_hub_model_index", read_index)

        assert image_plan(str(tmp_path), family="zimage", config=None, platform_key="linux") == ()
        instant = image_plan(str(models_dir / "ideo"), family="ideogram4", config=_CONFIG, platform_key="linux")
        assert instant is not None
        assert [target.repo_id for target in instant] == ["owner/base-nf4", "owner/instant"]

    def test_the_video_plan_takes_the_configured_companions(self):
        from zvisiongenerator.backends.cuda.download_plan import SEPARATE_TRANSFORMER_IGNORE

        plan = video_plan("owner/ltx", family="ltx", config=_CONFIG, model_name="ltx-4", upscale=True)

        assert plan == (
            DownloadTarget("owner/ltx", ignore_patterns=SEPARATE_TRANSFORMER_IGNORE),
            DownloadTarget("owner/ltx-transformer", allow_patterns=("t.safetensors",)),
            DownloadTarget("owner/ltx-up", allow_patterns=("u.safetensors",)),
        )

    def test_the_video_plan_leaves_the_upsampler_out_without_upscale(self):
        plan = video_plan("owner/ltx", family="ltx", config=_CONFIG, model_name="ltx-4", upscale=False)

        assert plan is not None
        assert [target.repo_id for target in plan] == ["owner/ltx", "owner/ltx-transformer"]

    def test_the_ltx25_plan_is_the_pinned_repo_alone_with_and_without_upscale(self, monkeypatch, tmp_path):
        from zvisiongenerator.utils import config as config_module
        from zvisiongenerator.utils.config import load_config

        monkeypatch.setattr(config_module, "get_ziv_data_dir", lambda: tmp_path / ".ziv")
        monkeypatch.setattr("zvisiongenerator.utils.paths.get_ziv_data_dir", lambda: tmp_path / ".ziv")
        monkeypatch.setattr("zvisiongenerator.backends.sys.platform", "linux")
        config = load_config()
        pin = config["model_aliases"]["ltx25-4"]["linux"]
        expected = (DownloadTarget("unsloth/LTX-2.5-FP8", "7b40724fc39bbb01d9b4d72638c0d263531c88c2", None, ("LTX-2.5-*.safetensors", "samples/*")),)

        assert pin == "unsloth/LTX-2.5-FP8@7b40724fc39bbb01d9b4d72638c0d263531c88c2"
        assert video_plan(pin, family="ltx", config=config, model_name="ltx25-4", upscale=False) == expected
        assert video_plan(pin, family="ltx", config=config, model_name="ltx25-4", upscale=True) == expected

    @pytest.mark.parametrize(("name", "bits"), [("ltx-4", 4), ("ltx-8", 8)])
    def test_the_ltx_2_3_plan_is_unchanged(self, monkeypatch, tmp_path, name, bits):
        """2.3 guard: the packaged ``ltx-4`` and ``ltx-8`` plans are the repo, the transformer and the upsampler, with no revision."""
        from zvisiongenerator.utils import config as config_module
        from zvisiongenerator.utils.config import load_config

        monkeypatch.setattr(config_module, "get_ziv_data_dir", lambda: tmp_path / ".ziv")
        monkeypatch.setattr("zvisiongenerator.utils.paths.get_ziv_data_dir", lambda: tmp_path / ".ziv")
        monkeypatch.setattr("zvisiongenerator.backends.sys.platform", "linux")
        config = load_config()
        repo = config["model_aliases"][name]["linux"]

        plan = video_plan(repo, family="ltx", config=config, model_name=name, upscale=True)

        assert plan == (
            DownloadTarget(repo, None, None, ("transformer/*.safetensors", "transformer/*.safetensors.index.json"), None),
            DownloadTarget("Lightricks/LTX-2.3", None, ("ltx-2.3-22b-distilled-1.1.safetensors",), None, None),
            DownloadTarget("Lightricks/LTX-2.3", None, ("ltx-2.3-spatial-upscaler-x2-1.1.safetensors",), None, None),
        )

    @staticmethod
    def _darwin_config(monkeypatch, tmp_path):
        from zvisiongenerator.backends.mac import download_plan as mac_plan
        from zvisiongenerator.utils import config as config_module
        from zvisiongenerator.utils.config import load_config

        monkeypatch.setattr(config_module, "get_ziv_data_dir", lambda: tmp_path / ".ziv")
        monkeypatch.setattr("zvisiongenerator.utils.paths.get_ziv_data_dir", lambda: tmp_path / ".ziv")
        monkeypatch.setattr("zvisiongenerator.backends.sys.platform", "darwin")
        monkeypatch.setattr(mac_plan, "ltx_mlx_text_encoder_repo", lambda: "owner/gemma3")
        return load_config()

    def test_a_macos_version_alias_plans_no_gemma_3_and_keeps_its_revision(self, monkeypatch, tmp_path):
        config = self._darwin_config(monkeypatch, tmp_path)
        config["video_model_presets"]["ltx"]["versions"]["ltx25"]["aliases"].append("ltx-ver")
        pin = "owner/ltx25-mlx@" + "d" * 40

        assert video_plan(pin, family="ltx", config=config, model_name="ltx-ver", upscale=False) == (DownloadTarget("owner/ltx25-mlx", "d" * 40),)

    def test_the_macos_ltx25_plan_is_the_pinned_mirror_alone_with_upscale(self, monkeypatch, tmp_path):
        config = self._darwin_config(monkeypatch, tmp_path)
        pin = config["model_aliases"]["ltx25-4"]["darwin"]

        assert pin == "MarceloFSO/ltx-2.5-mlx-q4-fast@281bf05bd23207d6feef708fc2e270f84ba992b5"
        assert video_plan(pin, family="ltx", config=config, model_name="ltx25-4", upscale=True) == (DownloadTarget("MarceloFSO/ltx-2.5-mlx-q4-fast", "281bf05bd23207d6feef708fc2e270f84ba992b5"),)

    @pytest.mark.parametrize("name", ["ltx-4", "ltx-8"])
    def test_the_macos_ltx_2_3_plan_is_the_repo_then_gemma_3(self, monkeypatch, tmp_path, name):
        """2.3 guard: the packaged ``ltx-4`` and ``ltx-8`` Mac plans are the whole repo with no revision, then Gemma 3."""
        from zvisiongenerator.backends.mac.download_plan import MLX_LM_ALLOW_PATTERNS

        config = self._darwin_config(monkeypatch, tmp_path)
        repo = config["model_aliases"][name]["darwin"]

        for upscale in (False, True):
            assert video_plan(repo, family="ltx", config=config, model_name=name, upscale=upscale) == (
                DownloadTarget(repo),
                DownloadTarget("owner/gemma3", None, MLX_LM_ALLOW_PATTERNS, None, None),
            )

    def test_the_enhancer_plan_uses_the_transformers_patterns(self):
        from zvisiongenerator.backends.cuda.download_plan import TRANSFORMERS_ALLOW_PATTERNS

        assert enhancer_plan("owner/llm", "rev") == (DownloadTarget("owner/llm", "rev", TRANSFORMERS_ALLOW_PATTERNS),)
