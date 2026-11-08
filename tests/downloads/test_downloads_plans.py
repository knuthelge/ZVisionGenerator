"""Tests for composing download plans: Instant sources, image/video/enhancer dispatch and SeedVR2 tier files."""

from __future__ import annotations

from unittest.mock import MagicMock

from huggingface_hub.utils import httpx
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled

from tests._factories import load_packaged_config, metadata_copy, use_model_metadata, video_config

from zvisiongenerator.utils.model_metadata import builtin_aliases
from zvisiongenerator.downloads.declaration import declared_repo_ids
from zvisiongenerator.downloads.plans import built_model_plan, enhancer_plan, image_plan, matte_plan, seedvr2_plan, video_model_downloaded, video_plan
from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils.matte import matte_pins
from zvisiongenerator.utils.model_metadata import model_entry
from zvisiongenerator.utils.paths import split_huggingface_revision
from zvisiongenerator.utils.ideogram4_instant import instant_downloads_dir, instant_source, write_build_record
from zvisiongenerator.utils.seedvr2 import FP8_WEIGHTS, MLX_WEIGHTS, seedvr2_files

SEEDVR2_REPO = "numz/SeedVR2_comfyUI"
SEEDVR2_MIXED_REPO = "AInVFX/SeedVR2_comfyUI"
BIREFNET, VITMATTE = matte_pins()

_CONFIG = video_config()


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


class TestBuiltModelPlan:
    def test_a_model_that_is_not_the_instant_folder_is_not_planned(self, models_dir):
        assert built_model_plan(str(models_dir / "zit"), platform_key="linux") is None
        assert built_model_plan("owner/repo", platform_key="linux") is None

    def test_a_built_folder_needs_nothing(self, models_dir):
        folder = models_dir / "ideo"
        folder.mkdir()
        write_build_record(folder, instant_source("linux"))

        assert built_model_plan(str(folder), platform_key="linux") == ()

    @pytest.mark.parametrize("platform_key", ["darwin", "win32", "linux"])
    def test_an_unbuilt_folder_needs_the_components_and_the_transformer_file(self, models_dir, platform_key):
        folder = models_dir / "ideo"
        declared = {target["role"]: target for target in model_entry("ideo")["downloads"][platform_key]}

        plan = built_model_plan(str(folder), platform_key=platform_key)

        assert plan is not None and len(plan) == 2
        components, transformer = plan
        assert (f"{components.repo_id}@{components.revision}", components.allow_patterns, components.local_dir) == (declared["components"]["ref"], tuple(declared["components"]["files"]), None)
        assert (f"{transformer.repo_id}@{transformer.revision}", transformer.allow_patterns) == (declared["transformer"]["ref"], tuple(declared["transformer"]["files"]))
        assert transformer.local_dir == str(instant_downloads_dir(folder))

    def test_the_platform_picks_the_components_repo(self, models_dir):
        darwin = built_model_plan(str(models_dir / "ideo"), platform_key="darwin")
        linux = built_model_plan(str(models_dir / "ideo"), platform_key="linux")

        assert darwin is not None and linux is not None
        assert darwin[0].repo_id != linux[0].repo_id

    def test_a_build_from_other_sources_is_planned_again(self, models_dir):
        folder = models_dir / "ideo"
        folder.mkdir()
        write_build_record(folder, instant_source("linux"))

        plan = built_model_plan(str(folder), platform_key="darwin")

        assert plan is not None
        assert len(plan) == 2

    def test_a_folder_that_ziv_did_not_build_needs_nothing_because_the_build_refuses_it(self, models_dir):
        folder = models_dir / "ideo"
        folder.mkdir()
        (folder / "weights.safetensors").write_bytes(b"x")

        assert built_model_plan(str(folder), platform_key="linux") == ()

    def test_an_empty_folder_is_planned_like_a_missing_one(self, models_dir):
        (models_dir / "ideo").mkdir()

        plan = built_model_plan(str(models_dir / "ideo"), platform_key="linux")

        assert plan is not None and len(plan) == 2

    def test_the_downloads_folder_is_the_hidden_sibling(self, tmp_path):
        assert instant_downloads_dir(tmp_path / "ideo") == tmp_path / ".ideo.download"


class TestImagePlan:
    def test_the_instant_folder_is_planned_before_the_backend(self, monkeypatch, models_dir):
        backend = _patch_backend(monkeypatch, "image_download_plan", ())

        plan = image_plan(str(models_dir / "ideo"), family="ideogram4", platform_key="linux")

        assert plan is not None
        assert len(plan) == 2
        backend.assert_not_called()

    def test_another_model_goes_to_the_backend(self, monkeypatch, models_dir):
        target = DownloadTarget("owner/zit", None, ("vae/*",))
        backend = _patch_backend(monkeypatch, "image_download_plan", (target,))

        plan = image_plan("owner/zit", family="zimage", platform_key="linux")

        assert plan == (target,)
        backend.assert_called_once_with("owner/zit", family="zimage")

    @pytest.mark.parametrize(
        "error",
        [LookupError("no model_index.json"), httpx.ConnectError("down"), httpx.ReadTimeout("slow"), OfflineModeIsEnabled("offline"), LocalEntryNotFoundError("not cached")],
    )
    def test_an_unreadable_file_list_cannot_be_planned(self, monkeypatch, models_dir, error):
        _patch_backend(monkeypatch, "image_download_plan", side_effect=error)

        assert image_plan("owner/zit", family="zimage", platform_key="linux") is None

    def test_other_errors_propagate(self, monkeypatch, models_dir):
        _patch_backend(monkeypatch, "image_download_plan", side_effect=RuntimeError("unsupported platform"))

        with pytest.raises(RuntimeError, match="unsupported"):
            image_plan("owner/zit", family="zimage", platform_key="linux")


class TestVideoPlan:
    def test_the_ignore_of_the_models_target_is_passed_to_the_backend(self, monkeypatch):
        backend = _patch_backend(monkeypatch, "video_download_plan", ())

        video_plan("owner/ltx", family="ltx", config=_CONFIG, model_name="ltx25-4", platform_key="linux", upscale=False)

        assert backend.call_args.kwargs["repo_ignore"] == ("LTX-2.5-*.safetensors", "samples/*")

    def test_a_model_of_no_version_passes_its_own_ignore_and_bundles_no_text_encoder(self, monkeypatch):
        backend = _patch_backend(monkeypatch, "video_download_plan", ())

        video_plan("owner/ltx", family="ltx", config=_CONFIG, model_name="ltx-4", platform_key="linux", upscale=False)

        assert backend.call_args.kwargs["repo_ignore"] == ("transformer/*.safetensors", "transformer/*.safetensors.index.json")
        assert backend.call_args.kwargs["text_encoder_bundled"] is False

    def test_a_repo_with_no_metadata_entry_skips_the_transformer_weights_its_single_file_transformer_replaces(self, monkeypatch):
        backend = _patch_backend(monkeypatch, "video_download_plan", ())

        video_plan("owner/ltx", family="ltx", config=_CONFIG, model_name=None, platform_key="linux", upscale=False)

        assert backend.call_args.kwargs["repo_ignore"] == ("transformer/*.safetensors", "transformer/*.safetensors.index.json")

    def test_a_version_bundles_its_text_encoder(self, monkeypatch):
        backend = _patch_backend(monkeypatch, "video_download_plan", ())

        video_plan("owner/ltx", family="ltx", config=_CONFIG, model_name="ltx25-4", platform_key="darwin", upscale=False)

        assert backend.call_args.kwargs["text_encoder_bundled"] is True

    def test_a_malformed_h3_setting_propagates(self, monkeypatch):
        _patch_backend(monkeypatch, "h3_download_plan", ())
        config = {"video_model_presets": {"h3": {"diffusers": {"ffn_chunk_tokens": 0}}}}

        with pytest.raises(ValueError, match="ffn_chunk_tokens"):
            video_plan("MiniMaxAI/MiniMax-H3", family="h3", config=config, model_name="h3", platform_key="linux", upscale=False)

    def test_h3_is_planned_from_its_preset_by_the_h3_planner(self, monkeypatch):
        import zvisiongenerator.backends as backends
        from zvisiongenerator.utils.h3_preset import h3_preset

        monkeypatch.setattr(backends.sys, "platform", "linux")
        config = load_packaged_config()
        upscale = True
        target = DownloadTarget("owner/h3")
        h3_backend = _patch_backend(monkeypatch, "h3_download_plan", (target,))
        ltx_backend = _patch_backend(monkeypatch, "video_download_plan", ())

        plan = video_plan("MiniMaxAI/MiniMax-H3@rev", family="h3", config=config, model_name="h3", platform_key="linux", upscale=upscale)

        assert plan == (target,)
        h3_backend.assert_called_once_with("MiniMaxAI/MiniMax-H3@rev", h3_preset(config, "linux"), upscale=upscale)
        ltx_backend.assert_not_called()

    def test_h3_on_macos_is_planned_from_the_mlx_preset_and_the_turbo_lora(self, monkeypatch):
        import zvisiongenerator.backends as backends
        from zvisiongenerator.utils.h3_preset import h3_mlx_preset, h3_turbo_lora

        monkeypatch.setattr(backends.sys, "platform", "darwin")
        config = load_packaged_config()

        plan = video_plan("MiniMaxAI/MiniMax-H3@rev", family="h3", config=config, model_name="h3", platform_key="darwin", upscale=False)

        assert [target.repo_id for target in plan] == ["water1234/MiniMax-H3-MLX-Argus-Calibrated-INT8", "MiniMaxAI/MiniMax-H3", "lightx2v/Minimax-h3-Turbo"]
        assert plan[0].allow_patterns == h3_mlx_preset().transformer.files
        assert plan[2].allow_patterns == h3_turbo_lora("darwin")[0].files

    def test_network_errors_cannot_be_planned(self, monkeypatch):
        _patch_backend(monkeypatch, "video_download_plan", side_effect=httpx.ConnectError("down"))

        assert video_plan("owner/ltx", family="ltx", config=_CONFIG, model_name="ltx", platform_key="linux", upscale=False) is None


class TestSeedVR2Plan:
    def test_mlx_downloads_the_tier_files_into_the_cache(self, models_dir):
        plan = seedvr2_plan("seedvr2-3b", platform_key="darwin", models_dir=models_dir, weights=MLX_WEIGHTS)

        assert plan == (DownloadTarget(SEEDVR2_REPO, None, ("seedvr2_ema_3b_fp16.safetensors", "ema_vae_fp16.safetensors")),)

    def test_a_current_stored_copy_needs_nothing(self, monkeypatch, models_dir):
        monkeypatch.setattr("zvisiongenerator.seedvr2_loader.seedvr2_downloaded", lambda model, folder, weights, *, platform_key: True)

        assert seedvr2_plan("seedvr2-7b", platform_key="darwin", models_dir=models_dir, weights=MLX_WEIGHTS) == ()

    def test_fp8_7b_downloads_one_target_per_repo_into_the_weights_folder(self, models_dir):
        folder = str(models_dir / ".seedvr2" / "seedvr2-7b@fp8")

        plan = seedvr2_plan("seedvr2-7b", platform_key="linux", models_dir=models_dir, weights=FP8_WEIGHTS)

        assert plan == (
            DownloadTarget(SEEDVR2_MIXED_REPO, None, ("seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",), local_dir=folder),
            DownloadTarget(SEEDVR2_REPO, None, ("ema_vae_fp16.safetensors",), local_dir=folder),
        )

    def test_fp8_3b_shares_one_repo(self, models_dir):
        plan = seedvr2_plan("seedvr2-3b", platform_key="linux", models_dir=models_dir, weights=FP8_WEIGHTS)

        assert len(plan) == 1
        assert plan[0].repo_id == SEEDVR2_REPO
        assert plan[0].allow_patterns == ("seedvr2_ema_3b_fp8_e4m3fn.safetensors", "ema_vae_fp16.safetensors")
        assert plan[0].local_dir == str(models_dir / ".seedvr2" / "seedvr2-3b@fp8")

    def test_downloaded_fp8_files_need_nothing(self, models_dir):
        folder = models_dir / ".seedvr2" / "seedvr2-3b@fp8"
        folder.mkdir(parents=True)
        for _repo, name in seedvr2_files("seedvr2-3b", "linux"):
            (folder / name).write_bytes(b"x")

        assert seedvr2_plan("seedvr2-3b", platform_key="linux", models_dir=models_dir, weights=FP8_WEIGHTS) == ()

    def test_one_missing_fp8_file_is_still_planned(self, models_dir):
        folder = models_dir / ".seedvr2" / "seedvr2-3b@fp8"
        folder.mkdir(parents=True)
        (folder / "ema_vae_fp16.safetensors").write_bytes(b"x")

        assert len(seedvr2_plan("seedvr2-3b", platform_key="linux", models_dir=models_dir, weights=FP8_WEIGHTS)) == 1

    def test_the_planned_names_cover_mflux_patterns(self):
        mflux_weights = pytest.importorskip("mflux.models.seedvr2.weights.seedvr2_weight_definition")
        for model, definition in (("seedvr2-3b", mflux_weights.SeedVR2WeightDefinition3B), ("seedvr2-7b", mflux_weights.SeedVR2WeightDefinition7B)):
            planned = {name for _repo, name in seedvr2_files(model, "darwin")}
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


class TestMattePlan:
    def test_two_pinned_targets_with_exact_files(self):
        assert matte_plan() == (
            DownloadTarget("ZhengPeng7/BiRefNet", BIREFNET.revision, ("model.safetensors",)),
            DownloadTarget("hustvl/vitmatte-small-composition-1k", VITMATTE.revision, ("config.json", "model.safetensors", "preprocessor_config.json")),
        )

    def test_the_pins_are_the_declared_birefnet_and_vitmatte_targets(self):
        birefnet, vitmatte = matte_pins()

        assert (birefnet.repo_id, birefnet.files) == ("ZhengPeng7/BiRefNet", ("model.safetensors",))
        assert (vitmatte.repo_id, vitmatte.files) == ("hustvl/vitmatte-small-composition-1k", ("config.json", "model.safetensors", "preprocessor_config.json"))

    def test_the_revisions_are_full_commit_hashes(self):
        for pin in (BIREFNET, VITMATTE):
            assert len(pin.revision) == 40
            assert set(pin.revision) <= set("0123456789abcdef")


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

        plan = image_plan("owner/zit@abc123", family="zimage", platform_key="linux")

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

        assert image_plan("owner/zit", family="zimage", platform_key="linux") is None

    def test_a_local_folder_and_an_instant_folder_need_no_index(self, monkeypatch, models_dir, tmp_path):
        from zvisiongenerator.backends.cuda import download_plan as cuda_plan

        def read_index(repo_id: str, revision: str | None) -> dict:
            raise AssertionError("the index must not be read")

        monkeypatch.setattr(cuda_plan, "_read_hub_model_index", read_index)

        assert image_plan(str(tmp_path), family="zimage", platform_key="linux") == ()
        instant = image_plan(str(models_dir / "ideo"), family="ideogram4", platform_key="linux")
        assert instant is not None
        assert [target.repo_id for target in instant] == list(declared_repo_ids("ideo", "linux"))

    def test_the_video_plan_takes_the_declared_companions(self):
        from zvisiongenerator.utils.ltx_facts import SEPARATE_TRANSFORMER_IGNORE

        plan = video_plan("owner/ltx", family="ltx", config=_CONFIG, model_name="ltx-4", platform_key="linux", upscale=True)
        lightricks = ("Lightricks/LTX-2.3", "c1521179de480464f366a98a9d9b2acc4cc097b9")

        assert plan == (
            DownloadTarget("owner/ltx", ignore_patterns=SEPARATE_TRANSFORMER_IGNORE),
            DownloadTarget(*lightricks, allow_patterns=("ltx-2.3-22b-distilled-1.1.safetensors",)),
            DownloadTarget(*lightricks, allow_patterns=("ltx-2.3-spatial-upscaler-x2-1.1.safetensors",)),
        )

    def test_the_video_plan_leaves_the_upsampler_out_without_upscale(self):
        plan = video_plan("owner/ltx", family="ltx", config=_CONFIG, model_name="ltx-4", platform_key="linux", upscale=False)

        assert plan is not None
        assert [target.repo_id for target in plan] == ["owner/ltx", "Lightricks/LTX-2.3"]

    @staticmethod
    def _config(monkeypatch, platform_key: str):
        monkeypatch.setattr("zvisiongenerator.backends.sys.platform", platform_key)
        return load_packaged_config()

    def test_the_ltx25_plan_is_the_pinned_repo_alone_with_and_without_upscale(self, monkeypatch):
        config = self._config(monkeypatch, "linux")
        pin = builtin_aliases()["ltx25-4"]["linux"]
        repo, revision = pin.split("@")
        ignore = ("LTX-2.5-*.safetensors", "samples/*")
        expected = (DownloadTarget(repo, revision, None, ignore),)

        assert video_plan(pin, family="ltx", config=config, model_name="ltx25-4", platform_key="linux", upscale=False) == expected
        assert video_plan(pin, family="ltx", config=config, model_name="ltx25-4", platform_key="linux", upscale=True) == expected

    @pytest.mark.parametrize(("name", "bits"), [("ltx-4", 4), ("ltx-8", 8)])
    def test_the_ltx_2_3_plan_is_the_repo_the_transformer_and_the_upsampler(self, monkeypatch, name, bits):
        config = self._config(monkeypatch, "linux")
        repo = builtin_aliases()[name]["linux"]
        lightricks = ("Lightricks/LTX-2.3", "c1521179de480464f366a98a9d9b2acc4cc097b9")

        plan = video_plan(repo, family="ltx", config=config, model_name=name, platform_key="linux", upscale=True)

        assert plan == (
            DownloadTarget(*split_huggingface_revision(repo), None, ("transformer/*.safetensors", "transformer/*.safetensors.index.json"), None),
            DownloadTarget(*lightricks, ("ltx-2.3-22b-distilled-1.1.safetensors",), None, None),
            DownloadTarget(*lightricks, ("ltx-2.3-spatial-upscaler-x2-1.1.safetensors",), None, None),
        )

    def test_a_macos_version_model_plans_no_gemma_3_and_keeps_its_revision(self, monkeypatch, tmp_path):
        config = self._config(monkeypatch, "darwin")
        data = metadata_copy()
        files = data["models"]["ltx25-4"]["downloads"]["darwin"][0]["files"]
        data["models"]["ltx-ver"] = {
            "kind": "video",
            "family": "ltx",
            "version": "ltx25",
            "quantize": 4,
            "name": "Other LTX-2.5",
            "downloads": {"darwin": [{"ref": "owner/ltx25-mlx@" + "d" * 40, "files": files}]},
        }
        use_model_metadata(monkeypatch, tmp_path, data)
        pin = "owner/ltx25-mlx@" + "d" * 40

        assert video_plan(pin, family="ltx", config=config, model_name="ltx-ver", platform_key="darwin", upscale=False) == (DownloadTarget("owner/ltx25-mlx", "d" * 40, tuple(files)),)

    def test_the_macos_ltx25_plan_is_the_pinned_packs_files_alone_with_upscale(self, monkeypatch):
        config = self._config(monkeypatch, "darwin")
        pin = builtin_aliases()["ltx25-4"]["darwin"]
        repo, revision = pin.split("@")
        files = tuple(model_entry("ltx25-4")["downloads"]["darwin"][0]["files"])

        for upscale in (False, True):
            assert video_plan(pin, family="ltx", config=config, model_name="ltx25-4", platform_key="darwin", upscale=upscale) == (DownloadTarget(repo, revision, files),)

    @pytest.mark.parametrize("name", ["ltx-4", "ltx-8"])
    def test_the_macos_ltx_2_3_plan_is_the_pinned_packs_files_then_the_pinned_gemma_3(self, monkeypatch, name):
        from zvisiongenerator.backends.mac.download_plan import MLX_LM_ALLOW_PATTERNS
        from zvisiongenerator.utils.paths import split_huggingface_revision

        config = self._config(monkeypatch, "darwin")
        ref = builtin_aliases()[name]["darwin"]
        repo, revision = split_huggingface_revision(ref)
        gemma_repo, gemma_revision = "mlx-community/gemma-3-12b-it-4bit", "86cc6a8dedbc456dd0e4af01a9d09f396f77e558"
        files = tuple(model_entry(name)["downloads"]["darwin"][0]["files"])

        assert revision is not None
        for upscale in (False, True):
            assert video_plan(ref, family="ltx", config=config, model_name=name, platform_key="darwin", upscale=upscale) == (
                DownloadTarget(repo, revision, files),
                DownloadTarget(gemma_repo, gemma_revision, MLX_LM_ALLOW_PATTERNS),
            )

    def test_a_macos_pack_plans_only_its_pinned_files_whatever_the_reference_and_the_name(self, monkeypatch):
        """No whole-repo target: the packs hold other transformers, LoRAs and upsamplers that must not be downloaded."""
        config = self._config(monkeypatch, "darwin")

        for name, targets in (("ltx-4", 2), ("ltx-8", 2), ("ltx25-4", 1)):
            repo, revision = builtin_aliases()[name]["darwin"].split("@")
            files = tuple(model_entry(name)["downloads"]["darwin"][0]["files"])
            for ref in (f"{repo}@{revision}", repo, repo.upper()):
                for picked in (name, "my-custom-ltx"):
                    plan = video_plan(ref, family="ltx", config=config, model_name=picked, platform_key="darwin", upscale=True)

                    assert plan is not None
                    assert plan[0] == DownloadTarget(repo, revision, files)
                    assert len(plan) == targets  # the 2.3 packs are followed by the pinned Gemma 3
                    assert all(target.revision and target.allow_patterns for target in plan)

    def test_a_pack_repo_pinned_to_another_revision_is_planned_whole_at_that_revision(self, monkeypatch):
        config = self._config(monkeypatch, "darwin")
        repo = builtin_aliases()["ltx-4"]["darwin"].split("@")[0]

        plan = video_plan(f"{repo}@{'e' * 40}", family="ltx", config=config, model_name="ltx-4", platform_key="darwin", upscale=False)

        assert plan is not None and plan[0] == DownloadTarget(repo, "e" * 40)

    def test_the_enhancer_plan_uses_the_transformers_patterns(self):
        from zvisiongenerator.backends.cuda.download_plan import TRANSFORMERS_ALLOW_PATTERNS

        assert enhancer_plan("owner/llm", "rev") == (DownloadTarget("owner/llm", "rev", TRANSFORMERS_ALLOW_PATTERNS),)


class TestVideoModelDownloaded:
    _CONFIG = video_config()

    def _plan(self, monkeypatch, targets):
        monkeypatch.setattr("zvisiongenerator.downloads.plans.video_plan", lambda *args, **kwargs: targets)

    def test_it_is_true_when_every_target_of_the_plan_is_present(self, monkeypatch):
        first, second = DownloadTarget("o/a", "r", ("x",)), DownloadTarget("o/b")
        self._plan(monkeypatch, (first, second))
        seen = []

        assert video_model_downloaded("o/a", family="ltx", config=self._CONFIG, model_name="ltx", platform_key="linux", target_present=lambda target: (seen.append(target), True)[1])
        assert seen == [first, second]

    def test_one_missing_target_is_not_downloaded(self, monkeypatch):
        self._plan(monkeypatch, (DownloadTarget("o/a"), DownloadTarget("o/b")))

        assert not video_model_downloaded("o/a", family="ltx", config=self._CONFIG, model_name="ltx", platform_key="linux", target_present=lambda target: target.repo_id == "o/a")

    def test_a_plan_that_cannot_be_made_is_not_downloaded(self, monkeypatch):
        self._plan(monkeypatch, None)

        assert not video_model_downloaded("o/a", family="ltx", config=self._CONFIG, model_name="ltx", platform_key="linux", target_present=lambda target: True)

    def test_an_empty_plan_of_an_existing_folder_is_downloaded(self, monkeypatch, tmp_path):
        self._plan(monkeypatch, ())

        assert video_model_downloaded(str(tmp_path), family="ltx", config=self._CONFIG, model_name="ltx", platform_key="linux", target_present=lambda target: False)

    @pytest.mark.parametrize("reference", ["{root}/deleted", "~/ziv-deleted-model"])
    def test_a_local_path_that_is_no_folder_is_not_downloaded(self, monkeypatch, tmp_path, reference):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        self._plan(monkeypatch, ())

        assert not video_model_downloaded(reference.format(root=tmp_path), family="ltx", config=self._CONFIG, model_name="ltx", platform_key="linux", target_present=lambda target: True)

    def test_a_folder_that_was_deleted_is_not_downloaded(self, monkeypatch, tmp_path):
        self._plan(monkeypatch, ())
        folder = tmp_path / "model"
        folder.mkdir()
        folder.rmdir()

        assert not video_model_downloaded(str(folder), family="ltx", config=self._CONFIG, model_name="ltx", platform_key="linux", target_present=lambda target: True)
