"""Tests for Models-page row plans, ``download_needed`` and the plan summary behind the download confirm."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import huggingface_hub
import huggingface_hub.constants
import pytest

from zvisiongenerator.backends.cuda import download_plan as cuda_plan
from zvisiongenerator.downloads.space import SpaceNeed
from zvisiongenerator.downloads.target import DownloadTarget, TargetCheck
from zvisiongenerator.utils.ideogram4_instant import instant_source, write_build_record
from zvisiongenerator.utils.model_memory import MemoryBudget
from zvisiongenerator.utils.seedvr2 import FP8_WEIGHTS, SeedVR2Settings, SeedVR2Setup, SeedVR2Tier
from zvisiongenerator.web import model_download_plan as plan_module
from zvisiongenerator.web import workspace_api as workspace_api_module
from zvisiongenerator.web.model_download_plan import PlanUnchecked, RowPlan, plan_summary, row_download_needed, row_plan
from zvisiongenerator.web.model_inventory import ImageInventoryEntry, VideoInventoryEntry
from zvisiongenerator.web.workspace_api import build_models_response

_TIER = SeedVR2Tier("seedvr2-7b", weights_gb=9.26, activation_floor_gb=10.79, activation_base_gb=1.88, activation_gb_per_mp=0.83)
_CONFIG = {
    "prompt_enhancer": {"model": "owner/enhancer"},
    "ideogram4_instant": {
        "transformer": {"repo": "owner/instant", "revision": "a" * 40, "filename": "instant.safetensors"},
        "components": {"linux": "owner/base@" + "b" * 40, "darwin": "owner/base@" + "b" * 40, "win32": "owner/base@" + "b" * 40},
        "transformer_format": {"linux": "bf16", "darwin": "fp8", "win32": "bf16"},
    },
    "video_model_presets": {
        "ltx": {
            "mlx_text_encoder": "owner/gemma",
            "diffusers": {
                "transformer": {"repo": "owner/ltx-transformer", "filename": "t.safetensors"},
                "latent_upsampler": {"repo": "owner/ltx-up", "filename": "u.safetensors"},
            },
        }
    },
}


@pytest.fixture
def env(monkeypatch, tmp_path):
    """An isolated data dir and Hub cache."""
    monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(hub))
    (tmp_path / "models").mkdir()
    return SimpleNamespace(data_dir=tmp_path, models_dir=tmp_path / "models", hub=hub)


def _web_config(env, *, image=(), video=(), seedvr2=None, app_config=None) -> SimpleNamespace:
    return SimpleNamespace(
        app_config=_CONFIG if app_config is None else app_config,
        data_dir=str(env.data_dir),
        models_dir=str(env.models_dir),
        image_inventory=tuple(image),
        video_inventory=tuple(video),
        seedvr2=seedvr2,
        quantize_options=(4, 8),
        loras_dir=str(env.data_dir / "loras"),
    )


def _image(name: str, path: str, family: str = "zimage", source: str = "alias") -> ImageInventoryEntry:
    return ImageInventoryEntry(name=name, family=family, size=None, source=source, resolved_path=path)


def _video(name: str, path: str) -> VideoInventoryEntry:
    return VideoInventoryEntry(name=name, family="ltx", supports_i2v=True, source="alias", resolved_path=path)


def _seedvr2(downloaded: bool = False) -> SeedVR2Setup:
    settings = SeedVR2Settings(model="auto", softness=0.75, video_softness=0.0, reference_megapixels=16, tiers=(_TIER,))
    return SeedVR2Setup(settings=settings, tier=_TIER, budget_bytes=None, downloaded=downloaded, weights=FP8_WEIGHTS)


def _no_planner_may_run(monkeypatch) -> None:
    for name in ("image_plan", "video_plan", "seedvr2_plan", "enhancer_plan"):
        monkeypatch.setattr(plan_module, name, lambda *a, _name=name, **k: pytest.fail(f"{_name} ran"))


class TestRowPlan:
    def test_an_image_row_is_planned_with_its_resolved_path_family_and_config(self, env, monkeypatch):
        targets = (DownloadTarget("owner/zit"),)
        planner = MagicMock(return_value=targets)
        monkeypatch.setattr(plan_module, "image_plan", planner)
        web_config = _web_config(env, image=[_image("zit", "owner/zit")])

        plan = row_plan("model", "zit", web_config)

        assert plan == RowPlan("model", "zit", "zit", targets)
        planner.assert_called_once_with("owner/zit", family="zimage", config=web_config.app_config, platform_key=sys.platform)

    def test_a_plan_that_cannot_be_made_has_no_targets(self, env, monkeypatch):
        monkeypatch.setattr(plan_module, "image_plan", lambda *a, **k: None)

        assert row_plan("model", "zit", _web_config(env, image=[_image("zit", "owner/zit")])).targets is None

    def test_a_video_row_is_planned_with_every_part(self, env, monkeypatch):
        targets = (DownloadTarget("owner/ltx"),)
        planner = MagicMock(return_value=targets)
        monkeypatch.setattr(plan_module, "video_plan", planner)
        web_config = _web_config(env, video=[_video("ltx-8", "owner/ltx")])

        plan = row_plan("model", "ltx-8", web_config)

        assert (plan.kind, plan.name, plan.label, plan.targets) == ("model", "ltx-8", "ltx-8", targets)
        planner.assert_called_once_with("owner/ltx", family="ltx", config=web_config.app_config, model_name="ltx-8", upscale=True)

    def test_a_row_is_labelled_by_its_configured_name(self, env, monkeypatch):
        monkeypatch.setattr(plan_module, "image_plan", MagicMock(return_value=(DownloadTarget("owner/zit"),)))
        monkeypatch.setattr(plan_module, "video_plan", MagicMock(return_value=(DownloadTarget("owner/ltx"),)))
        labels = {"zit": {"name": "Z-Image Turbo"}, "ltx-8": {"name": "LTX-2.3 8-bit"}}
        web_config = _web_config(env, image=[_image("zit", "owner/zit")], video=[_video("ltx-8", "owner/ltx")], app_config={**_CONFIG, "model_labels": labels})

        image, video = row_plan("model", "zit", web_config), row_plan("model", "ltx-8", web_config)
        summary = plan_summary(image, check=lambda t: None, present=lambda t: True, cache_dir=lambda: env.data_dir)

        assert (image.name, image.label, video.name, video.label) == ("zit", "Z-Image Turbo", "ltx-8", "LTX-2.3 8-bit")
        assert (summary["name"], summary["label"]) == ("zit", "Z-Image Turbo")

    def test_the_seedvr2_row_is_planned_by_its_row_name(self, env, monkeypatch):
        targets = (DownloadTarget("owner/seedvr2", None, ("a.safetensors",), local_dir="/x"),)
        planner = MagicMock(return_value=targets)
        monkeypatch.setattr(plan_module, "seedvr2_plan", planner)
        web_config = _web_config(env, seedvr2=_seedvr2())

        plan = row_plan("model", "seedvr2-7b@fp8", web_config)

        assert (plan.name, plan.label, plan.targets) == ("seedvr2-7b@fp8", "seedvr2-7b@fp8", targets)
        planner.assert_called_once_with("seedvr2-7b", models_dir=env.models_dir, weights=FP8_WEIGHTS)

    def test_an_enhancer_row_is_planned_by_tier(self, env, monkeypatch):
        targets = (DownloadTarget("owner/enhancer"),)
        planner = MagicMock(return_value=targets)
        monkeypatch.setattr(plan_module, "enhancer_plan", planner)

        plan = row_plan("enhancer", "fast", _web_config(env))

        assert (plan.kind, plan.name, plan.targets) == ("enhancer", "fast", targets)
        assert plan.label
        planner.assert_called_once_with("owner/enhancer", None)

    @pytest.mark.parametrize(("kind", "name"), [("model", "nope"), ("enhancer", "nope"), ("enhancer", "smart"), ("lora", "zit")])
    def test_an_unknown_row_or_kind_is_a_value_error(self, env, kind, name):
        with pytest.raises(ValueError):
            row_plan(kind, name, _web_config(env, image=[_image("zit", "owner/zit")]))


class TestDownloadNeeded:
    def test_an_image_row_needs_a_download_exactly_when_it_is_not_downloaded_and_no_target_planner_runs(self, env, monkeypatch):
        _no_planner_may_run(monkeypatch)
        web_config = _web_config(env, image=[_image("zit", "owner/zit")])

        assert row_download_needed("model", "zit", web_config, downloaded=False) is True
        assert row_download_needed("model", "zit", web_config, downloaded=True) is False
        assert row_download_needed("model", "zit", web_config, downloaded=None) is False

    def test_a_built_instant_folder_needs_nothing(self, env):
        folder = env.models_dir / "ideo"
        folder.mkdir()
        write_build_record(folder, instant_source(_CONFIG, sys.platform))
        web_config = _web_config(env, image=[_image("ideo", str(folder), family="ideogram4", source="installed")])

        assert row_download_needed("model", "ideo", web_config, downloaded=False, present=lambda target: False) is False

    def test_an_unbuilt_instant_folder_needs_its_sources_until_they_are_present(self, env):
        folder = env.models_dir / "ideo"
        folder.mkdir()
        web_config = _web_config(env, image=[_image("ideo", str(folder), family="ideogram4", source="installed")])
        present: set[str] = set()

        def _present(target: DownloadTarget) -> bool:
            return target.repo_id in present

        assert row_download_needed("model", "ideo", web_config, downloaded=True, present=_present) is True
        present.add("owner/base")
        assert row_download_needed("model", "ideo", web_config, downloaded=True, present=_present) is True
        present.add("owner/instant")
        assert row_download_needed("model", "ideo", web_config, downloaded=True, present=_present) is False

    def test_cuda_ltx_with_the_repo_but_not_the_transformer_needs_a_download(self, env, monkeypatch):
        monkeypatch.setattr("zvisiongenerator.backends.video_download_plan", cuda_plan.video_download_plan)
        web_config = _web_config(env, video=[_video("ltx-8", "owner/ltx")])
        present = {"owner/ltx"}

        def _present(target: DownloadTarget) -> bool:
            return target.repo_id in present

        assert row_download_needed("model", "ltx-8", web_config, downloaded=True, present=_present) is True
        present.update({"owner/ltx-transformer", "owner/ltx-up"})
        assert row_download_needed("model", "ltx-8", web_config, downloaded=True, present=_present) is False

    def test_the_ltx25_row_plans_one_pinned_target_and_is_needed_until_it_is_present(self, env, monkeypatch):
        from zvisiongenerator.utils.config import load_config

        monkeypatch.setattr("zvisiongenerator.backends.sys.platform", "linux")
        config = load_config()
        pin = config["model_aliases"]["ltx25-4"]["linux"]
        web_config = _web_config(env, video=[_video("ltx25-4", pin)], app_config=config)
        target = DownloadTarget("unsloth/LTX-2.5-FP8", "7b40724fc39bbb01d9b4d72638c0d263531c88c2", None, ("LTX-2.5-*.safetensors", "samples/*"))

        assert row_plan("model", "ltx25-4", web_config).targets == (target,)
        assert row_download_needed("model", "ltx25-4", web_config, downloaded=True, present=lambda t: False) is True
        assert row_download_needed("model", "ltx25-4", web_config, downloaded=True, present=lambda t: t == target) is False

    def test_a_video_plan_that_cannot_be_made_falls_back_to_the_downloaded_status(self, env, monkeypatch):
        monkeypatch.setattr(plan_module, "video_plan", lambda *a, **k: None)
        web_config = _web_config(env, video=[_video("ltx-8", "owner/ltx")])

        assert row_download_needed("model", "ltx-8", web_config, downloaded=False) is True
        assert row_download_needed("model", "ltx-8", web_config, downloaded=True) is False

    def test_a_seedvr2_row_needs_a_download_until_its_targets_are_present(self, env, monkeypatch):
        target = DownloadTarget("owner/seedvr2", None, ("a.safetensors",), local_dir=str(env.models_dir / "w"))
        monkeypatch.setattr(plan_module, "seedvr2_plan", lambda *a, **k: (target,))
        web_config = _web_config(env, seedvr2=_seedvr2())

        assert row_download_needed("model", "seedvr2-7b@fp8", web_config, downloaded=False, present=lambda t: False) is True
        assert row_download_needed("model", "seedvr2-7b@fp8", web_config, downloaded=False, present=lambda t: True) is False

    def test_a_downloaded_seedvr2_row_needs_nothing(self, env, monkeypatch):
        _no_planner_may_run(monkeypatch)

        assert row_download_needed("model", "seedvr2-7b@fp8", _web_config(env, seedvr2=_seedvr2(True)), downloaded=True) is False

    def test_an_enhancer_row_needs_a_download_until_present_and_never_when_unavailable(self, env):
        web_config = _web_config(env)
        monkeypatch_present = {"owner/enhancer"}

        def _present(target: DownloadTarget) -> bool:
            return target.repo_id in monkeypatch_present

        assert row_download_needed("enhancer", "fast", web_config, downloaded=False, present=lambda t: False) is True
        assert row_download_needed("enhancer", "fast", web_config, downloaded=True, present=_present) is False
        assert row_download_needed("enhancer", "fast", web_config, downloaded=False, unavailable="too big", present=lambda t: False) is False

    def test_an_unknown_row_is_a_value_error(self, env):
        with pytest.raises(ValueError):
            row_download_needed("model", "nope", _web_config(env), downloaded=False)


class TestModelsResponse:
    @pytest.fixture
    def web_config(self, env, monkeypatch):
        monkeypatch.setattr(workspace_api_module, "list_loras", lambda _data_dir: [])
        monkeypatch.setattr(workspace_api_module, "memory_budget", lambda: None)
        return _web_config(
            env,
            image=[_image("zit", "owner/zit")],
            video=[_video("ltx-8", "owner/ltx")],
            seedvr2=_seedvr2(),
        )

    def test_every_row_kind_carries_download_needed_as_a_bool(self, web_config):
        payload = build_models_response(web_config, token_source=None)

        rows = [*payload["image_models"], *payload["video_models"], *payload["enhancer_models"]]
        assert {row.get("name", row.get("tier")) for row in rows} == {"zit", "seedvr2-7b@fp8", "ltx-8", "fast"}
        assert all(isinstance(row["download_needed"], bool) for row in rows)

    def test_the_listing_builds_offline_without_any_network_or_planner_that_reads_files(self, web_config, monkeypatch):
        """SC-4: building the Models page never plans an image model, never checks the Hub and never downloads."""

        def _raise(*_args, **_kwargs):
            raise AssertionError("network or download call while listing models")

        monkeypatch.setattr("zvisiongenerator.backends.image_download_plan", _raise)
        monkeypatch.setattr(plan_module, "image_plan", lambda *a, **k: pytest.fail("image_plan ran while listing models"))
        monkeypatch.setattr(plan_module, "check_target", lambda *a, **k: pytest.fail("check_target ran while listing models"))
        monkeypatch.setattr(huggingface_hub, "snapshot_download", _raise)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _raise)

        payload = build_models_response(web_config, token_source=None)

        rows = {row["name"]: row for row in (*payload["image_models"], *payload["video_models"])}
        assert rows["zit"]["download_needed"] is True
        assert rows["ltx-8"]["download_needed"] is True
        assert payload["enhancer_models"][0]["download_needed"] is True

    def test_one_row_that_cannot_be_planned_warns_and_reads_false(self, web_config, monkeypatch):
        def _boom(*_args, **_kwargs):
            raise ValueError("malformed video config")

        monkeypatch.setattr(plan_module, "video_plan", _boom)

        with pytest.warns(UserWarning, match="ltx-8"):
            payload = build_models_response(web_config, token_source=None)

        assert payload["video_models"][0]["download_needed"] is False
        assert payload["image_models"][0]["download_needed"] is True

    def test_an_unavailable_enhancer_tier_needs_no_download(self, env, monkeypatch):
        rows = workspace_api_module._enhancer_model_rows(
            {"prompt_enhancer": {"model": "owner/enhancer", "smart": {"model": "owner/smart"}}},
            downloaded=lambda repo, revision: False,
            budget_bytes=lambda: None,
            present=lambda target: False,
        )

        by_tier = {row["tier"]: row for row in rows}
        assert by_tier["fast"]["download_needed"] is True
        assert by_tier["smart"]["unavailable"]
        assert by_tier["smart"]["download_needed"] is False


class TestPlanSummary:
    def _plan(self, *targets: DownloadTarget | None) -> RowPlan:
        return RowPlan("model", "zit", "zit", tuple(targets))

    def test_sums_the_missing_parts_and_reports_the_free_space_as_one_object(self, tmp_path):
        first, second = DownloadTarget("owner/a"), DownloadTarget("owner/b")
        checks = {first: TargetCheck(3, 1_000), second: TargetCheck(1, 500)}
        space = MagicMock(return_value=SpaceNeed(path=tmp_path, needed_bytes=1_500, free_bytes=1_000))

        summary = plan_summary(self._plan(first, second), check=checks.get, present=lambda t: False, space=space, cache_dir=lambda: tmp_path)

        assert summary == {
            "kind": "model",
            "name": "zit",
            "label": "zit",
            "parts": [{"repo": "owner/a", "files": 3, "bytes": 1_000}, {"repo": "owner/b", "files": 1, "bytes": 500}],
            "total_bytes": 1_500,
            "space": {"path": str(tmp_path), "needed_bytes": 1_500, "free_bytes": 1_000, "enough": False},
        }
        assert space.call_args.kwargs == {"cache_dir": tmp_path}
        assert space.call_args.args[0] == [(first, checks[first]), (second, checks[second])]

    def test_targets_on_disk_are_skipped_and_never_checked(self, tmp_path):
        present, missing = DownloadTarget("owner/a"), DownloadTarget("owner/b")
        check = MagicMock(return_value=TargetCheck(2, 40))

        summary = plan_summary(
            self._plan(present, missing),
            check=check,
            present=lambda target: target is present,
            space=lambda items, *, cache_dir: SpaceNeed(tmp_path, 40, 100),
            cache_dir=lambda: tmp_path,
        )

        assert [part["repo"] for part in summary["parts"]] == ["owner/b"]
        check.assert_called_once_with(missing)

    def test_nothing_missing_gives_no_parts_zero_bytes_and_no_space(self, tmp_path):
        target = DownloadTarget("owner/a")
        space = MagicMock()

        for present, check in ((lambda t: True, MagicMock()), (lambda t: False, lambda t: TargetCheck(0, 0))):
            summary = plan_summary(self._plan(target), check=check, present=present, space=space, cache_dir=lambda: tmp_path)

            assert (summary["parts"], summary["total_bytes"], summary["space"]) == ([], 0, None)
        space.assert_not_called()

    def test_an_unknown_part_size_makes_the_total_unknown(self, tmp_path):
        first, second = DownloadTarget("owner/a"), DownloadTarget("owner/b")
        checks = {first: TargetCheck(1, 10), second: TargetCheck(1, None)}

        summary = plan_summary(self._plan(first, second), check=checks.get, present=lambda t: False, space=lambda items, *, cache_dir: SpaceNeed(tmp_path, 10, 100), cache_dir=lambda: tmp_path)

        assert summary["total_bytes"] is None
        assert [part["bytes"] for part in summary["parts"]] == [10, None]
        assert summary["space"]["enough"] is True

    def test_a_target_the_hub_cannot_be_asked_about_is_unchecked(self, tmp_path):
        with pytest.raises(PlanUnchecked):
            plan_summary(self._plan(DownloadTarget("owner/a")), check=lambda t: None, present=lambda t: False, cache_dir=lambda: tmp_path)

    def test_a_plan_without_targets_is_unchecked(self, tmp_path):
        with pytest.raises(PlanUnchecked):
            plan_summary(RowPlan("model", "zit", "zit", None), cache_dir=lambda: tmp_path)

    def test_a_hub_refusal_passes_through_as_a_runtime_error(self, tmp_path):
        def _refuse(target):
            raise RuntimeError(f"Could not download {target.repo_id}: gated")

        with pytest.raises(RuntimeError, match="owner/a"):
            plan_summary(self._plan(DownloadTarget("owner/a")), check=_refuse, present=lambda t: False, cache_dir=lambda: tmp_path)


class TestInstantStatus:
    """The Workspace counts an unbuilt Ideogram 4 Instant folder as downloaded once its sources are on disk."""

    @pytest.fixture
    def present(self, monkeypatch):
        repos: set[str] = set()
        real = workspace_api_module.row_download_needed
        monkeypatch.setattr(workspace_api_module, "row_download_needed", lambda *a, **k: real(*a, present=lambda target: target.repo_id in repos, **k))
        monkeypatch.setattr(workspace_api_module, "memory_budget", lambda: MemoryBudget(10 * 1024**3))
        return repos

    def _status(self, env, *, instant_sources_count: bool = True) -> dict:
        folder = env.models_dir / "ideo"
        web_config = _web_config(env, image=[_image("ideo", str(folder), family="ideogram4", source="installed")])
        resolve = workspace_api_module._status_resolver(web_config, {}, lambda path: Path(path) if Path(path).is_dir() else None, instant_sources_count=instant_sources_count)
        return resolve(web_config.image_inventory[0], "image")

    def test_complete_sources_without_a_built_folder_read_as_downloaded_without_a_memory_fit(self, env, present):
        present.update({"owner/base", "owner/instant"})

        assert self._status(env) == {"downloaded": True, "memory_fit": None}

    def test_missing_sources_read_as_not_downloaded(self, env, present):
        present.add("owner/base")

        assert self._status(env) == {"downloaded": False, "memory_fit": None}

    def test_the_models_page_resolver_keeps_reporting_an_unbuilt_folder_as_not_downloaded(self, env, present):
        present.update({"owner/base", "owner/instant"})

        assert self._status(env, instant_sources_count=False) == {"downloaded": False, "memory_fit": None}

    def test_a_built_folder_is_reported_from_its_files(self, env, present):
        folder = env.models_dir / "ideo"
        folder.mkdir()
        write_build_record(folder, instant_source(_CONFIG, sys.platform))

        assert self._status(env)["downloaded"] is True

    def test_other_models_are_unaffected(self, env, present):
        present.add("owner/zit")
        web_config = _web_config(env, image=[_image("zit", "owner/zit")])
        resolve = workspace_api_module._status_resolver(web_config, {}, lambda path: None, instant_sources_count=True)

        assert resolve(web_config.image_inventory[0], "image")["downloaded"] is False
