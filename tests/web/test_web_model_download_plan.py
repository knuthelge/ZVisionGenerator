"""Tests for Models-page row plans, ``download_needed`` and the row summary behind the download confirm."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import huggingface_hub
import huggingface_hub.constants
import pytest

from tests._audio_presets import audio_config
from tests._factories import load_packaged_config, use_built_model_pins, use_enhancer_models, video_config

from zvisiongenerator.utils.model_metadata import builtin_aliases
from zvisiongenerator.backends.cuda import download_plan as cuda_plan
from zvisiongenerator.downloads.space import SpaceNeed
from zvisiongenerator.downloads.target import DownloadTarget, TargetCheck
from zvisiongenerator.utils.ideogram4_instant import instant_source, write_build_record
from zvisiongenerator.utils.matte import matte_pins
from zvisiongenerator.utils.model_memory import MemoryBudget
from zvisiongenerator.utils.seedvr2 import FP8_WEIGHTS, SeedVR2Settings, SeedVR2Setup, SeedVR2Tier, seedvr2_tier_label
from zvisiongenerator.web import model_download_plan as plan_module
from zvisiongenerator.web import workspace_api as workspace_api_module
from zvisiongenerator.utils.licences import MATTE_LICENCE_MODEL, load_licences
from zvisiongenerator.web import job_downloads as job_downloads_module
from zvisiongenerator.web.model_download_plan import MATTE_ROW_NAME, parts_summary, row_download_needed, row_plan
from zvisiongenerator.web.model_inventory import ImageInventoryEntry, VideoInventoryEntry
from zvisiongenerator.web.workspace_api import build_models_response

BIREFNET, VITMATTE = matte_pins()
MATTE_LABEL = "Background removal"
MATTE_DOWNLOAD_SIZE_LABEL = "548 MB"
_TIER = SeedVR2Tier("seedvr2-7b", weights_gb=9.26, activation_floor_gb=10.79, activation_base_gb=1.88, activation_gb_per_mp=0.83)
_CONFIG = {**audio_config(), **video_config()}


@pytest.fixture(autouse=True)
def _pinned_instant_sources(monkeypatch, tmp_path):
    """Pin Ideogram 4 Instant's sources to ``owner/base`` and ``owner/instant`` on every platform."""
    components = dict.fromkeys(("darwin", "win32", "linux"), "owner/base@" + "b" * 40)
    use_built_model_pins(monkeypatch, tmp_path, None, components=components, transformer="owner/instant@" + "a" * 40)


@pytest.fixture(autouse=True)
def _enhancer_models(monkeypatch, tmp_path, _pinned_instant_sources):
    """Pin the Fast enhancer to ``owner/enhancer`` and the Smart one to ``owner/smart`` on every platform."""
    use_enhancer_models(monkeypatch, tmp_path, fast="owner/enhancer", smart="owner/smart")


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
    for module, name in ((job_downloads_module, "image_plan"), (job_downloads_module, "video_plan"), (job_downloads_module, "seedvr2_plan"), (plan_module, "enhancer_plan")):
        monkeypatch.setattr(module, name, lambda *a, _name=name, **k: pytest.fail(f"{_name} ran"))


class TestRowPlan:
    @pytest.mark.parametrize(
        ("kind", "name", "planner", "inventory", "expected_kwargs"),
        [
            pytest.param("model", "zit", "image_plan", {"image": [_image("zit", "owner/zit")]}, {}, id="image"),
            pytest.param("model", "ltx-8", "video_plan", {"video": [_video("ltx-8", "owner/ltx")]}, {"upscale": True}, id="video-includes-the-upscaler-parts"),
            pytest.param("model", "seedvr2-7b@fp8", "seedvr2_plan", {"seedvr2": _seedvr2()}, {"weights": FP8_WEIGHTS}, id="seedvr2-picks-the-weights-folder"),
            pytest.param("enhancer", "fast", "enhancer_plan", {}, {}, id="enhancer"),
        ],
    )
    def test_each_row_kind_is_planned_by_its_own_planner(self, env, monkeypatch, kind, name, planner, inventory, expected_kwargs):
        targets = (DownloadTarget("owner/target"),)
        mock = MagicMock(return_value=targets)
        monkeypatch.setattr(job_downloads_module, planner, mock)

        plan = row_plan(kind, name, _web_config(env, **inventory))

        assert plan.targets == targets
        mock.assert_called_once()
        assert {key: mock.call_args.kwargs[key] for key in expected_kwargs} == expected_kwargs

    def test_a_seedvr2_row_and_an_enhancer_row_read_as_the_jobs_name_them(self, env, monkeypatch):
        monkeypatch.setattr(job_downloads_module, "seedvr2_plan", lambda *a, **k: ())
        monkeypatch.setattr(job_downloads_module, "enhancer_plan", lambda *a, **k: ())

        seedvr2 = row_plan("model", "seedvr2-7b@fp8", _web_config(env, seedvr2=_seedvr2()))
        enhancer = row_plan("enhancer", "fast", _web_config(env))

        assert seedvr2.label == seedvr2_tier_label("seedvr2-7b", FP8_WEIGHTS)
        assert enhancer.label == "Fast enhancer model"

    def test_the_background_removal_row_plans_both_pinned_repos(self, env):
        plan = row_plan("matte", MATTE_ROW_NAME, _web_config(env))

        assert (plan.model_id, plan.label) == (MATTE_LICENCE_MODEL, MATTE_LABEL)
        assert plan.targets == (
            DownloadTarget(BIREFNET.repo_id, BIREFNET.revision, BIREFNET.files),
            DownloadTarget(VITMATTE.repo_id, VITMATTE.revision, VITMATTE.files),
        )

    def test_a_background_removal_plan_that_cannot_be_made_has_no_targets(self, env, monkeypatch):
        monkeypatch.setattr(job_downloads_module, "matte_plan", lambda: None)

        assert row_plan("matte", MATTE_ROW_NAME, _web_config(env)).targets is None

    def test_a_plan_that_cannot_be_made_has_no_targets(self, env, monkeypatch):
        monkeypatch.setattr(job_downloads_module, "image_plan", lambda *a, **k: None)

        assert row_plan("model", "zit", _web_config(env, image=[_image("zit", "owner/zit")])).targets is None

    def test_a_row_is_labelled_by_its_metadata_name(self, env, monkeypatch):
        monkeypatch.setattr(job_downloads_module, "image_plan", MagicMock(return_value=(DownloadTarget("owner/zit"),)))
        monkeypatch.setattr(job_downloads_module, "video_plan", MagicMock(return_value=(DownloadTarget("owner/ltx"),)))
        web_config = _web_config(env, image=[_image("zit", "owner/zit")], video=[_video("ltx-8", "owner/ltx")])

        image, video = row_plan("model", "zit", web_config), row_plan("model", "ltx-8", web_config)

        assert [(part.model_id, part.label) for part in (image, video)] == [("zit", "Z-Image Turbo"), ("ltx-8", "LTX-2.3 8-bit")]

    @pytest.mark.parametrize(("kind", "name"), [("model", "nope"), ("enhancer", "nope"), ("matte", "nope"), ("lora", "zit")])
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
        write_build_record(folder, instant_source(sys.platform))
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

    @pytest.mark.real_download_plans
    def test_cuda_ltx_with_the_repo_but_not_the_transformer_needs_a_download(self, env, monkeypatch):
        monkeypatch.setattr("zvisiongenerator.backends.video_download_plan", cuda_plan.video_download_plan)
        monkeypatch.setattr("zvisiongenerator.backends.sys.platform", "linux")
        web_config = _web_config(env, video=[_video("ltx-8", "owner/ltx")])
        present = {"owner/ltx"}

        def _present(target: DownloadTarget) -> bool:
            return target.repo_id in present

        assert row_download_needed("model", "ltx-8", web_config, downloaded=True, present=_present) is True
        present.add("Lightricks/LTX-2.3")
        assert row_download_needed("model", "ltx-8", web_config, downloaded=True, present=_present) is False

    @pytest.mark.real_download_plans
    def test_the_ltx25_row_plans_one_pinned_target_and_is_needed_until_it_is_present(self, env, monkeypatch):
        monkeypatch.setattr("zvisiongenerator.backends.sys.platform", "linux")
        config = load_packaged_config()
        pin = builtin_aliases()["ltx25-4"]["linux"]
        web_config = _web_config(env, video=[_video("ltx25-4", pin)], app_config=config)
        target = DownloadTarget("unsloth/LTX-2.5-FP8", "7b40724fc39bbb01d9b4d72638c0d263531c88c2", None, ("LTX-2.5-*.safetensors", "samples/*"))

        assert row_plan("model", "ltx25-4", web_config).targets == (target,)
        assert row_download_needed("model", "ltx25-4", web_config, downloaded=True, present=lambda t: False) is True
        assert row_download_needed("model", "ltx25-4", web_config, downloaded=True, present=lambda t: t == target) is False

    def test_a_video_plan_that_cannot_be_made_falls_back_to_the_downloaded_status(self, env, monkeypatch):
        monkeypatch.setattr(job_downloads_module, "video_plan", lambda *a, **k: None)
        web_config = _web_config(env, video=[_video("ltx-8", "owner/ltx")])

        assert row_download_needed("model", "ltx-8", web_config, downloaded=False) is True
        assert row_download_needed("model", "ltx-8", web_config, downloaded=True) is False

    def test_a_seedvr2_row_needs_a_download_until_its_targets_are_present(self, env, monkeypatch):
        target = DownloadTarget("owner/seedvr2", None, ("a.safetensors",), local_dir=str(env.models_dir / "w"))
        monkeypatch.setattr(job_downloads_module, "seedvr2_plan", lambda *a, **k: (target,))
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

    def test_the_background_removal_row_needs_a_download_until_both_repos_are_present(self, env):
        web_config = _web_config(env)
        present = {BIREFNET.repo_id}

        def _present(target: DownloadTarget) -> bool:
            return target.repo_id in present

        assert row_download_needed("matte", MATTE_ROW_NAME, web_config, downloaded=None, present=lambda t: False) is True
        assert row_download_needed("matte", MATTE_ROW_NAME, web_config, downloaded=None, present=_present) is True
        present.add(VITMATTE.repo_id)
        assert row_download_needed("matte", MATTE_ROW_NAME, web_config, downloaded=None, present=_present) is False

    def test_a_background_removal_plan_that_cannot_be_made_needs_nothing(self, env, monkeypatch):
        monkeypatch.setattr(plan_module, "matte_plan", lambda: None)

        assert row_download_needed("matte", MATTE_ROW_NAME, _web_config(env), downloaded=None, present=lambda t: False) is False

    def test_an_unknown_row_is_a_value_error(self, env):
        with pytest.raises(ValueError):
            row_download_needed("model", "nope", _web_config(env), downloaded=False)

    def test_an_unknown_background_removal_row_is_a_value_error(self, env):
        with pytest.raises(ValueError):
            row_download_needed("matte", "nope", _web_config(env), downloaded=None)


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
        assert {row.get("name", row.get("tier")) for row in rows} == {"zit", "seedvr2-7b@fp8", "ltx-8", "fast", "smart"}
        assert all(isinstance(row["download_needed"], bool) for row in rows)

    def test_the_listing_builds_offline_without_any_network_or_planner_that_reads_files(self, web_config, monkeypatch):
        """SC-4: building the Models page never plans an image model, never checks the Hub and never downloads."""

        def _raise(*_args, **_kwargs):
            raise AssertionError("network or download call while listing models")

        monkeypatch.setattr("zvisiongenerator.backends.image_download_plan", _raise)
        monkeypatch.setattr(job_downloads_module, "image_plan", lambda *a, **k: pytest.fail("image_plan ran while listing models"))
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

        monkeypatch.setattr(job_downloads_module, "video_plan", _boom)

        with pytest.warns(UserWarning, match="ltx-8"):
            payload = build_models_response(web_config, token_source=None)

        assert payload["video_models"][0]["download_needed"] is False
        assert payload["image_models"][0]["download_needed"] is True

    def test_the_background_removal_row_has_the_fields_the_page_reads(self, web_config):
        payload = build_models_response(web_config, token_source=None)

        row = payload["background_removal"]
        assert set(row) == {"name", "label", "downloaded", "disk_size_label", "download_size_label", "deletable", "download_needed", "licences"}
        assert (row["name"], row["label"], row["download_size_label"]) == (MATTE_ROW_NAME, MATTE_LABEL, MATTE_DOWNLOAD_SIZE_LABEL)
        assert (row["downloaded"], row["deletable"], row["download_needed"], row["disk_size_label"]) == (False, False, True, None)

    def test_the_background_removal_row_reads_downloaded_and_deletable_once_both_repos_are_cached(self, env):
        for pin in (BIREFNET, VITMATTE):
            blob = env.hub / f"models--{pin.repo_id.replace('/', '--')}" / "blobs" / "weights"
            blob.parent.mkdir(parents=True)
            blob.write_bytes(b"x" * 2048)

        row = workspace_api_module._background_removal_row(present=lambda target: True)

        assert (row["downloaded"], row["deletable"], row["download_needed"]) == (True, True, False)
        assert row["disk_size_label"]

    def test_an_unavailable_enhancer_tier_needs_no_download(self, env, monkeypatch):
        rows = workspace_api_module._enhancer_model_rows(
            {},
            downloaded=lambda repo, revision: False,
            budget_bytes=lambda: None,
            present=lambda target: False,
        )

        by_tier = {row["tier"]: row for row in rows}
        assert by_tier["fast"]["download_needed"] is True
        assert by_tier["smart"]["unavailable"]
        assert by_tier["smart"]["download_needed"] is False


class TestRowSummary:
    """A Models-page row is summarised as one part, the shape a job's plan has."""

    def test_a_background_removal_row_is_one_part_that_sums_both_repos(self, env, tmp_path):
        part = row_plan("matte", MATTE_ROW_NAME, _web_config(env))
        checks = {target: TargetCheck(1, 100) for target in part.targets}

        summary = parts_summary(
            (part,),
            check=checks.get,
            present=lambda t: False,
            space=MagicMock(return_value=SpaceNeed(path=tmp_path, needed_bytes=200, free_bytes=1_000)),
            cache_dir=lambda: tmp_path,
            licences=load_licences(),
        )

        assert summary["parts"] == [{"label": MATTE_LABEL, "model": MATTE_LICENCE_MODEL, "bytes": 200}]
        assert (summary["total_bytes"], summary["unchecked"]) == (200, False)

    def test_each_target_is_looked_up_on_disk_once_while_a_part_is_summarised(self, env, tmp_path):
        part = row_plan("matte", MATTE_ROW_NAME, _web_config(env))
        looked_up: list[object] = []

        def absent(target):
            looked_up.append(target)
            return False

        parts_summary(
            (part,),
            committed=[DownloadTarget("a/elsewhere")],
            check=lambda _target: TargetCheck(1, 100),
            present=absent,
            space=MagicMock(return_value=None),
            cache_dir=lambda: tmp_path,
            licences=load_licences(),
        )

        assert looked_up == list(part.targets)

    def test_a_row_s_licences_follow_its_model(self, env, tmp_path, monkeypatch):
        monkeypatch.setattr(job_downloads_module, "image_plan", MagicMock(return_value=(DownloadTarget("owner/a"),)))
        monkeypatch.setattr(job_downloads_module, "enhancer_plan", MagicMock(return_value=(DownloadTarget("owner/a"),)))
        web_config = _web_config(env, image=[_image("klein9b", "owner/klein"), _image("zit", "owner/zit")])
        kwargs = {"check": lambda t: TargetCheck(1, 10), "present": lambda t: False, "space": MagicMock(return_value=None), "cache_dir": lambda: tmp_path, "licences": load_licences()}

        klein, matte, enhancer, zit = (
            parts_summary((row_plan(kind, name, web_config),), **kwargs) for kind, name in (("model", "klein9b"), ("matte", MATTE_ROW_NAME), ("enhancer", "fast"), ("model", "zit"))
        )

        assert [(entry["model"], entry["supplier"]) for entry in klein["licences"]] == [("klein9b", "Black Forest Labs")]
        assert [(entry["model"], entry["url_label"]) for entry in matte["licences"]] == [(MATTE_LICENCE_MODEL, "Dataset terms")]
        assert [(entry["model"], entry["supplier"]) for entry in enhancer["licences"]] == [("enhancer-fast", "Qwen")]
        assert [(entry["model"], entry["supplier"]) for entry in zit["licences"]] == [("zit", "Tongyi-MAI")]


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
        write_build_record(folder, instant_source(sys.platform))

        assert self._status(env)["downloaded"] is True

    def test_other_models_are_unaffected(self, env, present):
        present.add("owner/zit")
        web_config = _web_config(env, image=[_image("zit", "owner/zit")])
        resolve = workspace_api_module._status_resolver(web_config, {}, lambda path: None, instant_sources_count=True)

        assert resolve(web_config.image_inventory[0], "image")["downloaded"] is False
