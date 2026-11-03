"""Tests for the audio side of the workspace payload: the workflows the contract lists, what each hides per model, and the model rows."""

from __future__ import annotations

import pytest

from tests._audio_presets import audio_config, higgs3_preset, sa3_preset, yue2_preset
from tests._factories import load_packaged_config
from tests._licences import use_licences
from tests.web._helpers import make_web_config, make_workspace_bootstrap_view

from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils.audio_preset import AudioCapabilities, parse_audio_presets
from zvisiongenerator.utils.model_memory import GIB, MemoryBudget
from zvisiongenerator.web import audio_models as audio_models_module
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web import workspace_api as workspace_api_module
from zvisiongenerator.web.model_status import describe_audio_status
from zvisiongenerator.web.workspace_contract import (
    AUDIO_WORKFLOW_VALUES,
    CANONICAL_WORKFLOW_VALUES,
    WORKFLOW_DEFINITIONS,
    build_workflow_contract,
    canonicalize_workflow,
    hidden_audio_controls,
    workflow_mode,
)

_VIDEO_AND_IMAGE = ["txt2img", "img2img", "txt2vid", "img2vid", "extend"]


def _on(monkeypatch, key: str) -> None:
    """Make this machine ``cuda`` or ``mps`` for the contract and the model rows."""
    monkeypatch.setattr(audio_models_module, "audio_platform_key", lambda: key)


def _presets(**overrides):
    return {preset.id: preset for preset in parse_audio_presets(audio_config(**overrides) if overrides else audio_config())}


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

    @pytest.mark.parametrize(
        ("caps", "hidden"),
        [
            ({"lyrics": True, "duration": True, "voice_clone": True}, []),
            ({"lyrics": False, "duration": True, "voice_clone": True}, ["lyrics"]),
            ({"lyrics": True, "duration": False, "voice_clone": True}, ["length"]),
            ({"lyrics": True, "duration": True, "voice_clone": False}, ["voice", "voice_consent", "voice_path", "voice_text"]),
            ({"lyrics": False, "duration": False, "voice_clone": False}, ["length", "lyrics", "voice", "voice_consent", "voice_path", "voice_text"]),
        ],
    )
    def test_a_capability_a_model_lacks_hides_its_controls(self, caps, hidden):
        assert hidden_audio_controls(AudioCapabilities(**caps)) == hidden

    def test_the_hidden_controls_are_all_controls_of_some_audio_workflow(self):
        every = {control for name in AUDIO_WORKFLOW_VALUES for control in WORKFLOW_DEFINITIONS[name]["visible_controls"]}

        assert set(hidden_audio_controls(AudioCapabilities(False, False, False))) <= every


class TestMemoryAndDownloadStatus:
    @pytest.mark.parametrize(
        ("budget_gb", "status"),
        [(10, "fits"), (7, "tight"), (6, "too_large")],
    )
    def test_a_cuda_model_compares_its_measured_peak_with_the_cards_memory(self, budget_gb, status):
        preset = _presets()["yue2"]  # measured at 6.7 GiB on CUDA

        fit = describe_audio_status(preset, "cuda", budget=MemoryBudget(round(budget_gb * GIB), system_bytes=64 * GIB), target_present=lambda _t: True)["memory_fit"]

        assert fit == {"kind": "discrete", "budget_gb": round(budget_gb, 1), "by_quantize": {"none": {"status": status, "required_gb": 6.7}}}

    def test_a_mac_compares_the_mps_peak_with_the_recommended_working_set(self):
        raw = yue2_preset()
        raw["platforms"]["mps"]["memory_gb"] = 12.0
        preset = _presets(yue2=raw)["yue2"]

        fit = describe_audio_status(preset, "mps", budget=MemoryBudget(16 * GIB), target_present=lambda _t: True)["memory_fit"]

        assert fit == {"kind": "unified", "budget_gb": 16.0, "by_quantize": {"none": {"status": "fits", "required_gb": 12.0}}}
        small = describe_audio_status(preset, "mps", budget=MemoryBudget(6 * GIB), target_present=lambda _t: True)["memory_fit"]
        assert small["by_quantize"]["none"]["status"] == "too_large"

    def test_an_unmeasured_mac_peak_or_an_unknown_budget_gives_no_estimate(self):
        preset = _presets()["yue2"]  # the Mac peak is null until the Mac gate fills it in

        assert describe_audio_status(preset, "mps", budget=MemoryBudget(16 * GIB), target_present=lambda _t: True)["memory_fit"] is None
        assert describe_audio_status(preset, "cuda", budget=None, target_present=lambda _t: True)["memory_fit"] is None

    def test_a_model_is_downloaded_when_every_repository_of_its_plan_is_on_disk(self):
        preset = _presets()["yue2"]
        on_disk = {"m-a-p/YuE2-3B"}

        status = describe_audio_status(preset, "cuda", budget=None, target_present=lambda target: target.repo_id in on_disk)
        on_disk.add("m-a-p/YuE2-Vae")
        after = describe_audio_status(preset, "cuda", budget=None, target_present=lambda target: target.repo_id in on_disk)

        assert status["downloaded"] is False and after["downloaded"] is True


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
            "capabilities": {"lyrics": True, "duration": False, "voice_clone": False},
            "limits": {"max_seconds": 360},
            "instrumental_sections": "[Intro] [Instrumental] [Solo] [Instrumental] [Solo] [Bridge] [Instrumental] [Outro]",
            "hidden_controls": ["length", "voice", "voice_consent", "voice_path", "voice_text"],
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
        assert rows["higgs3"]["capabilities"]["voice_clone"] is True and rows["higgs3"]["hidden_controls"] == ["length", "lyrics"]
        assert rows["sa3"]["limits"] == {"max_seconds": 380, "min_seconds": 1.0}
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
