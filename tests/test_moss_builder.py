"""Tests for the MOSS-Image builder: the folder it assembles, when it rebuilds, its download plan and its call sites."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests._factories import _patch_image_main_dependencies

import zvisiongenerator.image_cli as image_cli
from zvisiongenerator import moss_builder
from zvisiongenerator.backends.cuda.download_plan import image_download_plan
from zvisiongenerator.built_models import ensure_built_model
from zvisiongenerator.downloads.plans import built_model_plan, image_plan
from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.image_model_loader import LOADING_PHASE
from zvisiongenerator.moss_builder import build_moss_model, ensure_moss_model, moss_model_index
from zvisiongenerator.utils import built_model
from zvisiongenerator.utils.built_model import BUILD_RECORD_NAME, BUILDING_PHASE
from zvisiongenerator.utils.moss_build import (
    LINKED_COMPONENTS,
    SFT_IGNORE,
    SFT_PATTERNS,
    TURBO_LORA_NAME,
    MossSource,
    build_record,
    is_built,
    moss_source,
)
from zvisiongenerator.web import web_runner as web_runner_module

_SFT = "owner/sft@" + "a" * 40
_CONFIG = {"moss_image": {"sft": _SFT, "turbo_lora": {"repo": "owner/lora", "revision": "b" * 40, "filename": "lora.safetensors"}}}


def _source(sft: str = _SFT, lora_revision: str = "b" * 40) -> MossSource:
    return MossSource(sft, "owner/lora", lora_revision, "lora.safetensors")


def _config(sft: str = _SFT, lora_revision: str = "b" * 40) -> dict:
    return {"moss_image": {"sft": sft, "turbo_lora": {"repo": "owner/lora", "revision": lora_revision, "filename": "lora.safetensors"}}}


@pytest.fixture
def models(monkeypatch, tmp_path) -> Path:
    monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
    return tmp_path / "models"


def _downloaders(tmp_path: Path, calls: list[str] | None = None) -> dict:
    """Return fake downloaders that lay the SFT snapshot and the LoRA out in a cache-like folder."""
    cache = tmp_path / "cache"
    snapshot = cache / "sft"
    for name in LINKED_COMPONENTS:
        (snapshot / name).mkdir(parents=True, exist_ok=True)
        (snapshot / name / "weights.safetensors").write_bytes(name.encode())
    (snapshot / "LICENSE").write_text("MIT")
    (snapshot / "model_index.json").write_text(json.dumps({"_class_name": "MOSSImagePipeline", "_name_or_path": "/somewhere", "scheduler": ["x", "y"]}))
    lora = cache / "lora.safetensors"
    lora.write_bytes(b"lora")

    def download_sft(source):
        if calls is not None:
            calls.append("sft")
        return snapshot

    def download_lora(source):
        if calls is not None:
            calls.append("lora")
        return lora

    return {"download_sft": download_sft, "download_lora": download_lora}


class TestBuildMossModel:
    def test_the_sft_folder_links_the_files_and_flags_the_index(self, models, tmp_path):
        calls: list[str] = []
        target = models / "moss-sft"

        build_moss_model(_source(), target, turbo=False, **_downloaders(tmp_path, calls))

        assert calls == ["sft"]
        assert is_built(target, _source())
        index = json.loads((target / "model_index.json").read_text())
        assert index["is_distilled"] is False
        assert "_name_or_path" not in index
        assert index["_class_name"] == "MOSSImagePipeline"
        assert not (target / TURBO_LORA_NAME).exists()
        assert json.loads((target / BUILD_RECORD_NAME).read_text()) == build_record(_source(), turbo=False)
        assert (target / "LICENSE").read_text() == "MIT"
        for name in LINKED_COMPONENTS:
            assert os.path.samefile(target / name / "weights.safetensors", tmp_path / "cache" / "sft" / name / "weights.safetensors")

    def test_the_turbo_folder_links_the_lora_and_marks_the_index_distilled(self, models, tmp_path):
        calls: list[str] = []
        target = models / "moss"

        build_moss_model(_source(), target, turbo=True, **_downloaders(tmp_path, calls))

        assert calls == ["sft", "lora"]
        assert is_built(target, _source())
        assert json.loads((target / "model_index.json").read_text())["is_distilled"] is True
        assert os.path.samefile(target / TURBO_LORA_NAME, tmp_path / "cache" / "lora.safetensors")
        assert json.loads((target / BUILD_RECORD_NAME).read_text()) == build_record(_source(), turbo=True)

    def test_the_record_is_written_last_and_nothing_else_is_left_beside_the_folder(self, models, tmp_path, monkeypatch):
        order: list[str] = []
        real_flush = built_model.flush_to_disk

        def flush(folder):
            order.append("record" if (folder / BUILD_RECORD_NAME).exists() else "flush")
            real_flush(folder)

        monkeypatch.setattr("zvisiongenerator.utils.built_model.flush_to_disk", flush)
        target = models / "moss"

        build_moss_model(_source(), target, turbo=True, **_downloaders(tmp_path))

        assert order == ["flush"]  # the record did not exist when the files were flushed
        assert [path.name for path in models.iterdir()] == ["moss"]

    def test_the_default_downloads_request_the_pinned_model_files_and_no_scripts(self, monkeypatch, tmp_path):
        import huggingface_hub

        snapshot = MagicMock(return_value=str(tmp_path / "snap"))
        single = MagicMock(return_value=str(tmp_path / "lora.safetensors"))
        monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", single)
        assert moss_builder._download_sft(_source()) == tmp_path / "snap"
        assert moss_builder._download_lora(_source()) == tmp_path / "lora.safetensors"

        snapshot.assert_called_once_with("owner/sft", revision="a" * 40, allow_patterns=list(SFT_PATTERNS), ignore_patterns=list(SFT_IGNORE))
        single.assert_called_once_with("owner/lora", "lora.safetensors", revision="b" * 40)

    def test_a_failed_build_discards_the_partial_folder(self, models, tmp_path):
        downloaders = _downloaders(tmp_path)
        downloaders["download_lora"] = lambda source: (_ for _ in ()).throw(OSError("network"))
        target = models / "moss"

        with pytest.raises(OSError, match="network"):
            build_moss_model(_source(), target, turbo=True, **downloaders)

        assert not target.exists() and not list(models.glob(".moss.*.partial"))

    def test_a_failure_while_linking_discards_the_partial_folder(self, models, tmp_path):
        downloaders = _downloaders(tmp_path)
        (tmp_path / "cache" / "sft" / "model_index.json").unlink()
        target = models / "moss-sft"

        with pytest.raises(FileNotFoundError):
            build_moss_model(_source(), target, turbo=False, **downloaders)

        assert not target.exists() and not list(models.glob(".moss-sft.*.partial"))

    def test_the_model_index_drops_only_the_copy_specific_path(self):
        assert moss_model_index({"_name_or_path": "x", "a": 1}, turbo=True) == {"a": 1, "is_distilled": True}


class TestEnsureMossModel:
    def test_other_models_are_left_alone(self, models, monkeypatch):
        monkeypatch.setattr("zvisiongenerator.moss_builder.build_moss_model", lambda *a, **k: pytest.fail("built"))
        phases: list[str] = []

        ensure_moss_model(str(models / "zit"), _CONFIG, on_phase=phases.append)
        ensure_moss_model("owner/MOSS-Image-SFT", _CONFIG, on_phase=phases.append)

        assert phases == []

    def test_a_missing_folder_is_built_and_reports_its_phases(self, models, monkeypatch):
        built: list[tuple[MossSource, Path, bool]] = []
        monkeypatch.setattr("zvisiongenerator.moss_builder.build_moss_model", lambda source, target, *, turbo, cancelled=None: built.append((source, target, turbo)) or True)
        phases: list[str] = []

        ensure_moss_model(str(models / "moss"), _CONFIG, on_phase=phases.append)
        ensure_moss_model(str(models / "moss-sft"), _CONFIG, on_phase=phases.append)

        assert built == [(moss_source(_CONFIG), models / "moss", True), (moss_source(_CONFIG), models / "moss-sft", False)]
        assert phases == [BUILDING_PHASE, LOADING_PHASE, BUILDING_PHASE, LOADING_PHASE]

    def test_a_current_build_is_reused(self, models, monkeypatch, tmp_path):
        build_moss_model(_source(), models / "moss", turbo=True, **_downloaders(tmp_path))
        monkeypatch.setattr("zvisiongenerator.moss_builder.build_moss_model", lambda *a, **k: pytest.fail("rebuilt"))
        phases: list[str] = []

        ensure_moss_model(str(models / "moss"), _CONFIG, on_phase=phases.append)

        assert phases == []

    def test_a_pin_change_rebuilds_and_a_lora_pin_change_leaves_sft_alone(self, models, monkeypatch, tmp_path):
        build_moss_model(_source(), models / "moss", turbo=True, **_downloaders(tmp_path))
        build_moss_model(_source(), models / "moss-sft", turbo=False, **_downloaders(tmp_path))
        built: list[str] = []
        monkeypatch.setattr("zvisiongenerator.moss_builder.build_moss_model", lambda source, target, *, turbo, cancelled=None: built.append(target.name) or True)

        lora_changed = _config(lora_revision="c" * 40)
        ensure_moss_model(str(models / "moss"), lora_changed)
        ensure_moss_model(str(models / "moss-sft"), lora_changed)
        assert built == ["moss"]

        sft_changed = _config(sft="owner/sft@" + "d" * 40)
        ensure_moss_model(str(models / "moss-sft"), sft_changed)
        assert built == ["moss", "moss-sft"]

    def test_an_invalid_block_raises_for_a_moss_folder_only(self, models):
        ensure_moss_model(str(models / "zit"), {})
        with pytest.raises(ValueError, match="moss_image"):
            ensure_moss_model(str(models / "moss"), {})


class TestMossPlan:
    def test_other_models_have_no_moss_plan(self, models):
        assert built_model_plan(str(models / "zit"), config=_CONFIG, platform_key=sys.platform) is None
        assert built_model_plan("owner/MOSS-Image-SFT", config=_CONFIG, platform_key=sys.platform) is None

    def test_the_turbo_folder_plans_the_sft_files_and_the_lora(self, models):
        assert built_model_plan(str(models / "moss"), config=_CONFIG, platform_key=sys.platform) == (
            DownloadTarget("owner/sft", "a" * 40, SFT_PATTERNS, SFT_IGNORE),
            DownloadTarget("owner/lora", "b" * 40, ("lora.safetensors",)),
        )

    def test_the_sft_folder_plans_only_the_sft_files(self, models):
        assert built_model_plan(str(models / "moss-sft"), config=_CONFIG, platform_key=sys.platform) == (DownloadTarget("owner/sft", "a" * 40, SFT_PATTERNS, SFT_IGNORE),)

    def test_a_built_folder_needs_nothing(self, models, tmp_path):
        build_moss_model(_source(), models / "moss", turbo=True, **_downloaders(tmp_path))

        assert built_model_plan(str(models / "moss"), config=_CONFIG, platform_key=sys.platform) == ()

    def test_image_plan_plans_a_moss_folder_from_the_config_and_never_a_raw_repo(self, models):
        targets = image_plan(str(models / "moss-sft"), family="moss", config=_CONFIG, platform_key=sys.platform)

        assert targets == (DownloadTarget("owner/sft", "a" * 40, SFT_PATTERNS, SFT_IGNORE),)
        assert image_download_plan("OpenMOSS-Team/MOSS-Image-SFT", family="moss") == ()


class TestForeignFolder:
    @pytest.mark.parametrize("name", ["moss", "moss-sft"])
    def test_a_non_empty_folder_without_a_build_record_is_never_replaced(self, models, tmp_path, name):
        calls: list[str] = []
        other = models / name
        other.mkdir(parents=True)
        (other / "weights.safetensors").write_bytes(b"someone else's model")

        with pytest.raises(RuntimeError, match="Rename or move"):
            build_moss_model(_source(), other, turbo=name == "moss", **_downloaders(tmp_path, calls))

        assert calls == [] and (other / "weights.safetensors").read_bytes() == b"someone else's model"

    def test_the_plan_asks_for_no_download(self, models):
        other = models / "moss"
        other.mkdir(parents=True)
        (other / "weights.safetensors").write_bytes(b"x")

        assert built_model_plan(str(other), config=_CONFIG, platform_key=sys.platform) == ()

    def test_an_empty_folder_is_built_in(self, models, tmp_path):
        (models / "moss-sft").mkdir(parents=True)

        build_moss_model(_source(), models / "moss-sft", turbo=False, **_downloaders(tmp_path))

        assert is_built(models / "moss-sft", _source())

    def test_a_folder_with_an_outdated_record_is_rebuilt(self, models, tmp_path):
        build_moss_model(_source(), models / "moss-sft", turbo=False, **_downloaders(tmp_path))
        changed = _source(sft="owner/sft@" + "d" * 40)

        build_moss_model(changed, models / "moss-sft", turbo=False, **_downloaders(tmp_path))

        assert is_built(models / "moss-sft", changed)


class TestCancelled:
    def test_a_stop_after_the_download_discards_the_partial_folder(self, models, tmp_path):
        target = models / "moss"

        built = build_moss_model(_source(), target, turbo=True, cancelled=lambda: True, **_downloaders(tmp_path))

        assert built is False
        assert not target.exists() and not list(models.glob(".moss.*.partial"))

    def test_ensure_reports_a_stopped_build_without_the_loading_phase(self, models, monkeypatch):
        monkeypatch.setattr("zvisiongenerator.moss_builder.build_moss_model", lambda *a, **k: False)
        phases: list[str] = []

        assert ensure_moss_model(str(models / "moss"), _CONFIG, on_phase=phases.append, cancelled=lambda: True) is False
        assert phases == [BUILDING_PHASE]


class TestEnsureBuiltModel:
    def test_a_moss_folder_goes_to_the_moss_builder_with_the_cancel_check(self, models, monkeypatch):
        seen: list[tuple] = []
        monkeypatch.setattr("zvisiongenerator.moss_builder.build_moss_model", lambda source, target, *, turbo, cancelled=None: seen.append((target.name, turbo, cancelled)) or True)
        stop = lambda: False  # noqa: E731

        assert ensure_built_model(str(models / "moss"), _CONFIG, platform_key="linux", cancelled=stop) is True

        assert seen == [("moss", True, stop)]

    def test_another_model_is_left_alone(self, models):
        assert ensure_built_model(str(models / "zit"), {}, platform_key="linux") is True
        assert ensure_built_model("owner/MOSS-Image-SFT", {}, platform_key="linux") is True


class TestCallSites:
    def test_the_cli_prepares_the_model_after_preflight_and_reports_the_building_phase(self, monkeypatch, capsys):
        _patch_image_main_dependencies(monkeypatch)
        ensure = MagicMock(side_effect=lambda model, config, *, platform_key, on_phase, cancelled: (on_phase(BUILDING_PHASE), on_phase(LOADING_PHASE), True)[2])
        monkeypatch.setattr(image_cli, "ensure_built_model", ensure)
        monkeypatch.setattr(image_cli, "model_display_name", lambda config, name: "Pretty Label")
        monkeypatch.setattr(sys, "argv", ["ziv-image", "-m", "owner/model", "--prompt", "a cat"])

        image_cli.main()

        assert ensure.call_args.args[0] == "owner/model"
        assert "Pretty Label" in capsys.readouterr().out

    def test_a_cli_build_error_ends_with_a_usage_error(self, monkeypatch, capsys):
        backend = _patch_image_main_dependencies(monkeypatch)
        monkeypatch.setattr(image_cli, "ensure_built_model", MagicMock(side_effect=OSError("disk full")))
        monkeypatch.setattr(image_cli, "model_display_name", lambda config, name: "Pretty Label")
        monkeypatch.setattr(sys, "argv", ["ziv-image", "-m", "owner/model", "--prompt", "a cat"])

        with pytest.raises(SystemExit) as exc_info:
            image_cli.main()

        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "Pretty Label" in err and "disk full" in err
        backend.load_model.assert_not_called()

    def test_the_web_runner_prepares_the_model_before_loading_and_emits_the_building_phase(self, monkeypatch):
        events: list[dict] = []
        order: list[str] = []
        monkeypatch.setattr(web_runner_module, "get_backend", lambda: MagicMock())

        def ensure(model_ref, config, *, platform_key, on_phase, cancelled):
            order.append("ensure")
            on_phase(BUILDING_PHASE)
            return True

        monkeypatch.setattr(web_runner_module, "ensure_built_model", ensure)
        monkeypatch.setattr(web_runner_module, "load_image_model", lambda *a, **k: order.append("load") or (MagicMock(), MagicMock()))
        request = MagicMock(model_name="moss", lora_paths=None, lora_weights=None)

        web_runner_module._load_image_model_with_progress(request, "/models/moss", None, events.append, MagicMock(), config=_CONFIG)

        assert order == ["ensure", "load"]
        assert {"type": "model_loading", "mode": "image", "model": "moss", "phase": BUILDING_PHASE} in events
