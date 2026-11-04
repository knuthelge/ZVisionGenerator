"""Tests for the audio side of the workspace payload: the workflows the contract lists, what each hides per model, and the model rows."""

from __future__ import annotations

import pytest

from tests._audio_presets import audio_config, higgs3_preset, sa3_preset, yue2_preset
from tests._factories import load_packaged_config
from tests._licences import use_licences
from tests.web._helpers import make_web_config, make_workspace_bootstrap_view

from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils.model_memory import GIB, MemoryBudget
from zvisiongenerator.web import audio_models as audio_models_module
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web import workspace_api as workspace_api_module
from zvisiongenerator.web.workspace_contract import (
    AUDIO_WORKFLOW_VALUES,
    CANONICAL_WORKFLOW_VALUES,
    WORKFLOW_DEFINITIONS,
    build_workflow_contract,
    canonicalize_workflow,
    workflow_mode,
)

_VIDEO_AND_IMAGE = ["txt2img", "img2img", "txt2vid", "img2vid", "extend"]


def _on(monkeypatch, key: str) -> None:
    """Make this machine ``cuda`` or ``mps`` for the contract and the model rows."""
    monkeypatch.setattr(audio_models_module, "audio_platform_key", lambda: key)


def _row(monkeypatch, model_id: str, key: str, *, budget, present=lambda _target: True, **overrides):
    """Return the Workspace row of *model_id* on a *key* machine with *budget*, from the fixture presets (or the *overrides*)."""
    _on(monkeypatch, key)
    rows = audio_models_module.audio_model_rows(audio_config(**overrides), budget, target_present=present)
    return next(row for row in rows if row["id"] == model_id)


class TestWorkflowValues:
    def test_the_audio_workflows_are_canonical_and_belong_to_the_audio_mode(self):
        assert set(AUDIO_WORKFLOW_VALUES) <= set(CANONICAL_WORKFLOW_VALUES)
        assert [workflow_mode(name) for name in AUDIO_WORKFLOW_VALUES] == ["audio"] * 3
        assert [canonicalize_workflow(name) for name in AUDIO_WORKFLOW_VALUES] == list(AUDIO_WORKFLOW_VALUES)
        assert all(WORKFLOW_DEFINITIONS[name]["mode"] == "audio" for name in AUDIO_WORKFLOW_VALUES)


class TestListedWorkflows:
    def test_a_config_without_audio_presets_lists_no_audio_workflow_so_the_tab_stays_hidden(self):
        contract = web_server._build_workflow_contract({**load_packaged_config(), "audio_model_presets": {}})

        assert contract["values"] == _VIDEO_AND_IMAGE
        assert set(contract["definitions"]) == set(_VIDEO_AND_IMAGE)

    @pytest.mark.parametrize("key", ["cuda", "mps"])
    def test_the_packaged_config_lists_every_audio_workflow_on_both_platforms(self, monkeypatch, key):
        _on(monkeypatch, key)

        contract = web_server._build_workflow_contract(load_packaged_config())

        assert contract["values"] == [*_VIDEO_AND_IMAGE, "music", "speech", "sound"]

    def test_only_the_kinds_that_have_a_runnable_model_are_listed(self, monkeypatch):
        _on(monkeypatch, "cuda")

        assert web_server._build_workflow_contract(audio_config(sa3=sa3_preset()))["values"] == [*_VIDEO_AND_IMAGE, "sound"]
        assert web_server._build_workflow_contract(audio_config(yue2=yue2_preset(), higgs3=higgs3_preset()))["values"] == [*_VIDEO_AND_IMAGE, "music", "speech"]

    def test_a_machine_with_neither_cuda_nor_mps_lists_nothing(self, monkeypatch):
        def refuse():
            raise RuntimeError("no accelerator")

        monkeypatch.setattr(audio_models_module, "audio_platform_key", refuse)

        assert web_server._build_workflow_contract(audio_config())["values"] == _VIDEO_AND_IMAGE


class TestRunnableModels:
    def test_the_presets_of_the_config_run_where_there_is_cuda_or_mps(self, monkeypatch):
        _on(monkeypatch, "cuda")

        presets = audio_models_module.runnable_audio_models(audio_config(sa3=sa3_preset()))

        assert [preset.id for preset in presets] == ["sa3"]

    def test_no_preset_runs_on_a_machine_with_neither(self, monkeypatch):
        def refuse():
            raise RuntimeError("no accelerator")

        monkeypatch.setattr(audio_models_module, "audio_platform_key", refuse)

        assert audio_models_module.runnable_audio_models(audio_config(sa3=sa3_preset())) == ()
        with pytest.raises(ValueError, match="no accelerator"):
            audio_models_module.find_audio_model(audio_config(sa3=sa3_preset()), "sa3")


class TestVisibleControls:
    @pytest.fixture
    def definitions(self):
        return build_workflow_contract(audio_kinds={"music", "speech", "sound"})["definitions"]

    def test_each_workflow_shows_its_own_boxes(self, definitions):
        assert definitions["music"]["visible_controls"] == ["workflow", "model", "music_style", "lyrics", "runs", "seed"]
        assert definitions["speech"]["visible_controls"] == ["workflow", "model", "speech_text", "voice", "voice_path", "voice_text", "voice_consent", "runs", "seed"]
        assert definitions["sound"]["visible_controls"] == ["workflow", "model", "sound_prompt", "length", "runs", "seed"]

    def test_no_audio_workflow_offers_an_image_video_or_prompt_file_control(self, definitions):
        image_and_video = {control for name in _VIDEO_AND_IMAGE for control in WORKFLOW_DEFINITIONS[name]["visible_controls"]} - {"workflow", "model", "runs", "seed"}

        for name in AUDIO_WORKFLOW_VALUES:
            assert not image_and_video & set(definitions[name]["visible_controls"])
            assert definitions[name]["requires_reference_image"] is False and definitions[name]["requires_source_video"] is False


class TestMemoryAndDownloadStatus:
    @pytest.mark.parametrize(
        ("budget_gb", "status"),
        [(10, "fits"), (7, "tight"), (6, "too_large")],
    )
    def test_a_cuda_model_compares_its_measured_peak_with_the_cards_memory(self, monkeypatch, budget_gb, status):
        row = _row(monkeypatch, "yue2", "cuda", budget=MemoryBudget(round(budget_gb * GIB), system_bytes=64 * GIB))  # measured at 6.7 GiB on CUDA

        assert row["memory_fit"] == {"kind": "discrete", "budget_gb": round(budget_gb, 1), "by_quantize": {"none": {"status": status, "required_gb": 6.7}}}

    def test_a_mac_compares_the_mps_peak_with_the_recommended_working_set(self, monkeypatch):
        raw = yue2_preset()
        raw["platforms"]["mps"] = {"memory_gb": 12.0}

        fit = _row(monkeypatch, "yue2", "mps", budget=MemoryBudget(16 * GIB), yue2=raw)["memory_fit"]
        small = _row(monkeypatch, "yue2", "mps", budget=MemoryBudget(6 * GIB), yue2=raw)["memory_fit"]

        assert fit == {"kind": "unified", "budget_gb": 16.0, "by_quantize": {"none": {"status": "fits", "required_gb": 12.0}}}
        assert small["by_quantize"]["none"]["status"] == "too_large"

    def test_an_unmeasured_mac_peak_or_an_unknown_budget_gives_no_estimate(self, monkeypatch):
        assert _row(monkeypatch, "yue2", "mps", budget=MemoryBudget(16 * GIB))["memory_fit"] is None  # the Mac peak is unmeasured
        assert _row(monkeypatch, "yue2", "cuda", budget=None)["memory_fit"] is None

    def test_a_model_is_downloaded_when_every_repository_of_its_plan_is_on_disk(self, monkeypatch):
        on_disk = {"m-a-p/YuE2-3B"}

        status = _row(monkeypatch, "yue2", "cuda", budget=None, present=lambda target: target.repo_id in on_disk)["downloaded"]
        on_disk.add("m-a-p/YuE2-Vae")
        after = _row(monkeypatch, "yue2", "cuda", budget=None, present=lambda target: target.repo_id in on_disk)["downloaded"]

        assert status is False and after is True


class TestWorkspacePayload:
    @staticmethod
    def build(web_config, monkeypatch, budget=MemoryBudget(10 * GIB, system_bytes=64 * GIB)):
        monkeypatch.setattr(workspace_api_module, "memory_budget", lambda: budget)
        return workspace_api_module.build_workspace_response(
            web_config,
            [],
            active_job=None,
            prompt_sources=["inline"],
            default_prompt_source="inline",
            prompt_file_contract={},
            workflow_contract=web_server._build_workflow_contract(web_config.app_config),
            build_bootstrap_view=lambda _cfg: make_workspace_bootstrap_view(),
        )

    def test_a_config_without_audio_presets_has_no_audio_models(self, monkeypatch):
        web_config = make_web_config()
        web_config.app_config.update({**load_packaged_config(), "audio_model_presets": {}})
        _on(monkeypatch, "cuda")

        workspace = self.build(web_config, monkeypatch)

        assert workspace["audio_models"] == []
        assert workspace["workflow_contract"]["values"] == _VIDEO_AND_IMAGE

    def test_the_packaged_config_offers_the_three_models_with_their_licences(self, monkeypatch):
        web_config = make_web_config()
        web_config.app_config.update(load_packaged_config())
        _on(monkeypatch, "cuda")

        workspace = self.build(web_config, monkeypatch)

        rows = {row["id"]: row for row in workspace["audio_models"]}
        assert list(rows) == ["yue2", "higgs3", "sa3"]
        assert [len(rows[model]["licences"]) for model in rows] == [1, 2, 2]

    def test_each_runnable_preset_is_a_row_in_config_order(self, monkeypatch, tmp_path):
        web_config = make_web_config()
        web_config.app_config.update(audio_config())
        use_licences(
            monkeypatch,
            tmp_path,
            {
                "yue2-nc": {
                    "name": "CC BY-NC 4.0 with individual creator permission",
                    "supplier": "M-A-P",
                    "summary": "Non-commercial; individual creators may monetise their outputs, companies need a licence",
                    "url": "https://github.com/multimodal-art-projection/YuE/blob/3acdec8cf1b97603bb696e4c6c4560348ffbb0b7/MODEL_LICENSE",
                    "models": ["yue2"],
                }
            },
        )
        _on(monkeypatch, "cuda")
        monkeypatch.setattr(audio_models_module, "target_present_offline", lambda target: target.repo_id.startswith("m-a-p/"))

        workspace = self.build(web_config, monkeypatch)

        rows = {row["id"]: row for row in workspace["audio_models"]}
        assert list(rows) == ["yue2", "higgs3", "sa3"]
        assert rows["yue2"] == {
            "id": "yue2",
            "label": "YuE2 3B",
            "kind": "music",
            "type": "audio",
            "limits": {"max_seconds": 360},
            "instrumental_sections": "[Intro] [Instrumental] [Solo] [Instrumental] [Solo] [Bridge] [Instrumental] [Outro]",
            "licences": [
                {
                    "name": "CC BY-NC 4.0 with individual creator permission",
                    "supplier": "M-A-P",
                    "summary": "Non-commercial; individual creators may monetise their outputs, companies need a licence",
                    "url": "https://github.com/multimodal-art-projection/YuE/blob/3acdec8cf1b97603bb696e4c6c4560348ffbb0b7/MODEL_LICENSE",
                    "url_label": "Licence",
                    "attribution": None,
                }
            ],
            "downloaded": True,
            "memory_fit": {"kind": "discrete", "budget_gb": 10.0, "by_quantize": {"none": {"status": "fits", "required_gb": 6.7}}},
        }
        assert rows["higgs3"]["limits"] == {"max_text_chars": 10000, "min_voice_seconds": 3, "max_voice_seconds": 30}
        assert rows["sa3"]["limits"] == {"max_seconds": 380, "min_seconds": 1.0, "default_seconds": 10.0}
        assert rows["higgs3"]["instrumental_sections"] is None and rows["sa3"]["instrumental_sections"] is None
        assert rows["sa3"]["downloaded"] is False  # none of its repositories is under m-a-p/
        assert rows["sa3"]["licences"] == []

    def test_a_status_that_cannot_be_read_leaves_the_row_with_unknown_status(self, monkeypatch):
        web_config = make_web_config()
        web_config.app_config.update(audio_config(sa3=sa3_preset()))
        _on(monkeypatch, "cuda")

        def broken(_target: DownloadTarget) -> bool:
            raise OSError("cache unreadable")

        monkeypatch.setattr(audio_models_module, "target_present_offline", broken)

        with pytest.warns(UserWarning, match="Could not determine status for audio model 'sa3'"):
            workspace = self.build(web_config, monkeypatch)

        assert [(row["id"], row["downloaded"], row["memory_fit"]) for row in workspace["audio_models"]] == [("sa3", None, None)]
