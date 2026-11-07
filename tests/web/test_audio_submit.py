"""Tests for audio jobs through ``/api/generate``: the form, its refusals, the plan, and the job the runner makes of it."""

from __future__ import annotations

import io
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from tests._audio_presets import audio_config, qwen3_tts_preset, sa3_two_kind_preset, yue2_preset
from tests._factories import load_packaged_config
from tests._fake_audio import fake_read_config, install_fake_audio, write_fake_audio
from tests._paths import FIXTURES
from tests.audio._helpers import case_request, clip, fake_backend, patch_runner, voice_folder
from tests.web._helpers import make_web_config, wait_for_status

from zvisiongenerator.core.audio_backend import AudioCancelled
from zvisiongenerator.core.audio_types import AudioGenerationRequest, SpeechDelivery, VoiceRef
from zvisiongenerator.downloads.ensure import DownloadReport
from zvisiongenerator.downloads.flights import DownloadFlights
from zvisiongenerator.downloads.space import SpaceNeed
from zvisiongenerator.downloads.target import DownloadProgress, TargetCheck
from zvisiongenerator.utils.audio_preset import parse_audio_presets
from zvisiongenerator.web.audio_job import AudioFields, audio_request_from_fields
from zvisiongenerator.utils.interactive import SkipSignal
from zvisiongenerator.web import audio_models as audio_models_module
from zvisiongenerator.web import model_download_plan as plan_module
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web import web_runner as web_runner_module
from zvisiongenerator.web.job_contract import generation_supported_controls

CASES = json.loads((FIXTURES / "audio_cli_cases.json").read_text(encoding="utf-8"))
SETTINGS_CASES = json.loads((FIXTURES / "audio_settings_cases.json").read_text(encoding="utf-8"))["cases"]
_GIB = 1024**3


def _set_platform(monkeypatch, key: str) -> None:
    """Make this machine ``cuda`` or ``mps`` for the form's checks and the contract."""
    monkeypatch.setattr(audio_models_module, "audio_platform_key", lambda: key)


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Serve the Web UI with the three fixture presets on a CUDA machine; the queue records what it is given instead of running it."""
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path / "out")
    web_config.app_config.update(audio_config())
    submitted: list[dict] = []
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    monkeypatch.setattr(web_server, "require_ffmpeg", lambda: None)
    install_fake_audio(monkeypatch)
    monkeypatch.setattr(web_server.web_runner, "submit_audio_request_job", lambda **kwargs: submitted.append(kwargs) or "job-123")
    _set_platform(monkeypatch, "cuda")
    with TestClient(web_server.app) as client:
        env = SimpleNamespace(client=client, web_config=web_config, submitted=submitted, root=tmp_path)
        env.post = lambda path="/api/generate", **fields: client.post(path, data={"mode": "audio", **fields})
        yield env


def _refused(response, *needles: str) -> str:
    """Assert *response* is a 422 whose message holds every needle, and return the message."""
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    for needle in needles:
        assert needle in detail
    return detail


class TestSubmit:
    def test_a_song_is_queued_as_an_audio_job(self, env):
        response = env.post(workflow="music", model="yue2", prompt="  upbeat synth-pop  ", lyrics="[verse]\nLa la", runs="2", seed="7")

        assert response.status_code == 200
        body = response.json()
        assert (body["job_id"], body["workflow"], body["job_type"], body["model"], body["model_label"], body["runs"]) == ("job-123", "music", "Music", "yue2", "YuE2 3B", 2)
        assert body["media_type"] == "audio" and body["width"] is None and body["height"] is None
        assert body["supported_controls"] == list(generation_supported_controls(review=False))
        (job,) = env.submitted
        assert job["request"] == AudioGenerationRequest(preset="yue2", kind="music", prompt="upbeat synth-pop", lyrics="[verse]\nLa la")
        assert (job["preset"].id, job["runs"], job["seed"], job["output_dir"]) == ("yue2", 2, 7, str(env.root / "out"))
        assert job["config"] is env.web_config.app_config
        assert job["context"]["settings"]["prompt"] == "  upbeat synth-pop  "  # Load settings restores the form as it was sent

    def test_the_summary_of_a_queued_job_does_not_repeat_the_model(self, env):
        sound = env.post(model="sa3", prompt="rain", seconds="3").json()
        speech = env.post(model="qwen3_tts", prompt="Welcome back.").json()

        assert sound["meta"] == "3 s"
        assert speech["meta"] == "13 characters"
        assert sound["model_label"] not in sound["meta"]

    def test_the_queued_job_keeps_the_models_download_targets_for_the_next_plan(self, env):
        env.post(workflow="music", model="yue2", prompt="synth-pop")

        (job,) = env.submitted
        assert [target.repo_id for target in job["planned_targets"]] == ["m-a-p/YuE2-3B", "m-a-p/YuE2-Vae"]

    def test_the_workflow_follows_the_model_when_the_form_names_none(self, env):
        assert env.post(model="sa3", prompt="rain").json()["workflow"] == "sound"

    def test_runs_default_to_one_and_the_seed_to_random(self, env):
        env.post(model="sa3", prompt="rain")

        assert (env.submitted[0]["runs"], env.submitted[0]["seed"]) == (1, None)

    def test_an_audio_workflow_is_not_an_image_or_video_workflow(self, env):
        for mode in ("image", "video"):
            response = env.client.post("/api/generate", data={"mode": mode, "workflow": "music", "prompt": "x"})

            _refused(response, "music", "mode=audio")
        assert env.submitted == []

    def test_a_job_is_refused_while_the_queue_holds_memory_for_another_task(self, env, monkeypatch):
        def busy(**_kwargs):
            raise web_runner_module.JobConflictError("Prompt enhancement in progress.")

        monkeypatch.setattr(web_server.web_runner, "submit_audio_request_job", busy)

        assert env.post(model="sa3", prompt="rain").status_code == 409


class TestKinds:
    """The form's ``workflow`` is the kind of the take; the model decides what that kind takes."""

    @pytest.fixture
    def two_kinds(self, env):
        env.web_config.app_config.update(audio_config(yue2=yue2_preset(), qwen3_tts=qwen3_tts_preset(), sa3=sa3_two_kind_preset()))
        return env

    def test_a_model_of_two_kinds_makes_the_kind_the_form_names(self, two_kinds):
        sound = two_kinds.post(workflow="sound", model="sa3", prompt="rain").json()
        music = two_kinds.post(workflow="music", model="sa3", prompt="calm piano").json()

        assert (sound["workflow"], music["workflow"]) == ("sound", "music")
        assert [job["request"].kind for job in two_kinds.submitted] == ["sound", "music"]

    def test_a_form_without_a_workflow_makes_the_first_kind_of_the_model(self, two_kinds):
        assert two_kinds.post(model="sa3", prompt="rain").json()["workflow"] == "sound"
        assert two_kinds.post(model="yue2", prompt="piano").json()["workflow"] == "music"

    def test_music_of_a_model_without_lyrics_takes_a_length_and_no_lyrics(self, two_kinds):
        two_kinds.post(workflow="music", model="sa3", prompt="calm piano", lyrics="[verse]\nLa", seconds="45")
        two_kinds.post(workflow="music", model="sa3", prompt="calm piano")

        assert [(job["request"].lyrics, job["request"].seconds) for job in two_kinds.submitted] == [("", 45.0), ("", 30.0)]

    def test_the_length_of_music_is_checked_against_the_models_limit(self, two_kinds):
        _refused(two_kinds.post(workflow="music", model="sa3", prompt="x", seconds="381"), "seconds must be between 1 and 380; Stable Audio 3 Medium makes songs of that length.")

    def test_music_of_a_model_with_lyrics_ignores_the_length(self, two_kinds):
        two_kinds.post(workflow="music", model="yue2", prompt="piano", lyrics="[verse]\nLa", seconds="45")

        assert (two_kinds.submitted[0]["request"].lyrics, two_kinds.submitted[0]["request"].seconds) == ("[verse]\nLa", None)

    @pytest.mark.parametrize(
        ("model", "workflow", "message"),
        [
            ("sa3", "speech", "Stable Audio 3 Medium makes sound and music, not speech."),
            ("yue2", "sound", "YuE2 3B makes music, not sound."),
            ("qwen3_tts", "music", "Qwen3-TTS 1.7B makes speech, not music."),
        ],
    )
    def test_a_kind_the_model_lacks_is_refused_naming_what_it_makes(self, two_kinds, model, workflow, message):
        assert _refused(two_kinds.post(workflow=workflow, model=model, prompt="x")) == message
        assert two_kinds.submitted == []

    def test_a_recording_is_not_read_for_a_model_that_cannot_speak(self, two_kinds, tmp_path):
        response = two_kinds.post(workflow="speech", model="sa3", prompt="x", voice_path=str(tmp_path / "missing.wav"), voice_text="x")

        _refused(response, "makes sound and music, not speech")


class TestRecentPrompts:
    def test_each_kind_is_remembered_under_its_own_list_and_a_song_keeps_its_lyrics(self, env):
        env.post(model="yue2", prompt="  synth-pop ", lyrics="[verse]\r\nLa la")
        env.post(model="qwen3_tts", prompt="Welcome back.")
        env.post(model="sa3", prompt="rain on a roof")

        def remembered(kind: str) -> list[tuple[str, str]]:
            return [(entry["prompt"], entry["scene"]) for entry in web_server.recent_prompts.search("", kind)]

        assert remembered("music") == [("synth-pop", "[verse]\nLa la")]
        assert remembered("speech") == [("Welcome back.", "")]
        assert remembered("sound") == [("rain on a roof", "")]


class TestNormalisation:
    def test_the_style_and_the_sound_prompt_are_trimmed_and_speech_is_kept_as_typed(self, env):
        env.post(model="yue2", prompt="  calm piano \n")
        env.post(model="sa3", prompt=" rain on a roof ", seconds="12")
        env.post(model="qwen3_tts", prompt="  Welcome back.\n\nSecond line.  ")

        music, sound, speech = (job["request"] for job in env.submitted)
        assert music.prompt == "calm piano" and sound.prompt == "rain on a roof"
        assert speech.prompt == "  Welcome back.\n\nSecond line.  "

    def test_the_browsers_line_breaks_become_the_ones_the_shell_hands_the_cli(self, env):
        env.post(model="yue2", prompt="style", lyrics="[verse]\r\nHello\r\n[chorus]\r\nLa")
        env.post(model="qwen3_tts", prompt="One.\r\nTwo.")

        assert env.submitted[0]["request"].lyrics == "[verse]\nHello\n[chorus]\nLa"
        assert env.submitted[1]["request"].prompt == "One.\nTwo."

    def test_blank_lyrics_make_an_instrumental(self, env):
        env.post(model="yue2", prompt="calm piano", lyrics="  \n ")

        assert env.submitted[0]["request"].lyrics == ""

    def test_a_field_of_another_kind_is_ignored_as_the_form_hides_it(self, env):
        env.post(model="sa3", prompt="rain", lyrics="[verse]\nx", voice_path="/nowhere.wav", voice_text="x", seconds="5")
        env.post(model="qwen3_tts", prompt="Hello.", lyrics="[verse]\nx", seconds="5")
        env.post(model="yue2", prompt="piano", seconds="5", voice_text="x", voice_consent="on")

        assert [(job["request"].lyrics, job["request"].seconds, job["request"].voice) for job in env.submitted] == [("", 5.0, None), ("", None, None), ("", None, None)]

    def test_a_sound_without_a_length_is_twelve_seconds(self, env):
        env.post(model="sa3", prompt="rain")

        assert env.submitted[0]["request"].seconds == 12.0


class TestFormMatchesCopyCli:
    """The form posts what Copy CLI reads, so the job it queues makes the takes the printed command makes (same fixture as the CLI and Vitest tests)."""

    @pytest.fixture
    def voices(self, tmp_path):
        return voice_folder(tmp_path)

    @pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
    def test_the_form_of_a_case_makes_the_takes_of_its_command(self, env, monkeypatch, voices, case):
        fields = {key: ([item for item in value] if isinstance(value, list) else str(value).replace("/voices/", f"{voices}/")) for key, value in case["settings"].items()}
        backend = fake_backend(clip(0.2))
        patch_runner(monkeypatch, backend)

        response = env.client.post("/api/generate", data=fields)
        assert response.status_code == 200, response.text
        (job,) = env.submitted
        from zvisiongenerator.audio_runner import run_audio_batch

        run_audio_batch(job["request"], job["preset"], job["config"], runs=job["runs"], seed=job["seed"], output_dir=str(env.root / "takes"))

        assert [call.args[0] for call in backend.generate.call_args_list] == [case_request(item, voices) for item in case["requests"]]


def _settings_request(item: dict, voices) -> AudioGenerationRequest:
    """Return the request a ``audio_settings_cases.json`` case expects (its seed is set per take by the runner), with its ``/voices/`` paths moved into *voices*."""
    voice = item["voice"]
    return AudioGenerationRequest(
        **{key: value for key, value in item.items() if key not in ("voice", "delivery")},
        voice=None if voice is None else VoiceRef(voice["path"].replace("/voices/", f"{voices}/"), voice["text"]),
        delivery=SpeechDelivery(**item["delivery"]) if item["delivery"] else None,
    )


def _settings_fields(form: dict) -> AudioFields:
    """Return the fields of a case's form that ``audio_request_from_fields`` reads, the enhance record parsed."""
    return AudioFields(
        prompt=form["prompt"],
        seconds=float(form["seconds"]) if form.get("seconds") else None,
        **{name: form.get(name) for name in ("tempo", "key", "emotion", "manner", "pace", "pitch", "expressiveness", "sound_type", "prompt_input")},
        enhance_record=json.loads(form["enhance_record"]) if form.get("enhance_record") else None,
    )


class TestSettingsRoundTrip:
    """The Retry and Reuse form of a take that sets every new value makes the take's own request (the Python half of the round trip; the same cases run in Vitest)."""

    @pytest.fixture
    def voices(self, tmp_path):
        return voice_folder(tmp_path)

    @pytest.fixture(autouse=True)
    def _models(self, env):
        env.web_config.app_config.update(audio_config(yue2=yue2_preset(), qwen3_tts=qwen3_tts_preset(), sa3=sa3_two_kind_preset()))

    def _post(self, env, monkeypatch, voices, form: dict):
        fields = {key: ([item for item in value] if isinstance(value, list) else str(value).replace("/voices/", f"{voices}/")) for key, value in form.items()}
        patch_runner(monkeypatch, fake_backend(clip(0.2)))
        response = env.client.post("/api/generate", data=fields)
        assert response.status_code == 200, response.text
        return env.submitted[-1]

    @pytest.mark.parametrize("which", ["form", "reuse_form"])
    @pytest.mark.parametrize("case", SETTINGS_CASES, ids=[case["name"] for case in SETTINGS_CASES])
    def test_the_form_makes_the_request_of_the_take(self, env, monkeypatch, voices, case, which):
        form = case.get(which, case["form"])
        expected = _settings_request(case["request"], voices)

        job = self._post(env, monkeypatch, voices, form)

        assert job["request"] == expected

    @pytest.mark.parametrize("case", SETTINGS_CASES, ids=[case["name"] for case in SETTINGS_CASES])
    def test_the_job_records_the_form_as_sent_for_retry_and_load_settings(self, env, monkeypatch, voices, case):
        job = self._post(env, monkeypatch, voices, case["form"])

        recorded = {key: value for key, value in job["context"]["settings"].items()}
        sent = {key: (value if isinstance(value, list) else str(value).replace("/voices/", f"{voices}/")) for key, value in case["form"].items() if key != "voice_consent"}
        assert recorded == sent

    @pytest.mark.parametrize("case", [case for case in SETTINGS_CASES if "enhance_record" in case["form"]], ids=lambda case: case["name"])
    def test_an_enhanced_text_keeps_its_own_text_and_record_in_the_request(self, case):
        item = case["request"]
        preset = next(model for model in parse_audio_presets(audio_config(yue2=yue2_preset(), qwen3_tts=qwen3_tts_preset(), sa3=sa3_two_kind_preset())) if model.id == item["preset"])

        request = audio_request_from_fields(preset, item["kind"], _settings_fields(case["form"]))

        assert request == _settings_request(item, None)


class TestEnhancedText:
    """The Enhanced text of a take: the form's ``prompt_input`` and ``enhance_record`` reach the request, and a record that cannot be read never fails the job."""

    _RECORD = {"settings": {"genre": "folk"}, "model": "example/enhancer-model", "tier": "turbo", "format": "text", "task": "style"}

    def _submit(self, env, **fields):
        response = env.post(workflow="music", model="yue2", prompt="Folk, piano", **fields)
        assert response.status_code == 200, response.text
        return env.submitted[-1]["request"]

    def test_an_enhanced_style_keeps_the_users_own_text_and_the_normalized_record(self, env):
        request = self._submit(env, prompt_input="sad folk", enhance_record=json.dumps(self._RECORD))

        assert request.prompt == "Folk, piano" and request.prompt_input == "sad folk"
        assert request.enhance_record == {**self._RECORD, "tier": "custom"}  # the tier is worked out from the model, not trusted

    def test_the_users_own_text_may_be_empty(self, env):
        request = self._submit(env, prompt_input="", enhance_record=json.dumps(self._RECORD))

        assert request.prompt_input == "" and request.enhance_record is not None

    def test_the_users_own_text_without_a_record_is_not_kept(self, env):
        request = self._submit(env, prompt_input="sad folk")

        assert (request.prompt_input, request.enhance_record) == (None, None)

    @pytest.mark.parametrize(
        "record", ["not json", "[]", json.dumps({**_RECORD, "task": "poem"}), json.dumps({**_RECORD, "settings": {"genre": "nope"}}), json.dumps({"settings": {}, "task": "style"})]
    )
    def test_a_record_that_cannot_be_read_is_dropped_with_a_warning_and_the_job_still_runs(self, env, record):
        with pytest.warns(UserWarning, match="Ignoring the enhance record"):
            request = self._submit(env, prompt_input="sad folk", enhance_record=record)

        assert (request.prompt, request.prompt_input, request.enhance_record) == ("Folk, piano", None, None)


class TestSettingsRefusals:
    @pytest.mark.parametrize(
        ("fields", "message"),
        [
            ({"model": "yue2", "tempo": "300"}, "Tempo must be 40 to 220 BPM."),
            ({"model": "yue2", "tempo": "quick"}, "Tempo must be 40 to 220 BPM."),
            ({"model": "yue2", "key": "H"}, "Key must be one of:"),
            ({"model": "qwen3_tts", "emotion": "grumpy"}, "Emotion must be one of:"),
            ({"model": "qwen3_tts", "manner": "sing"}, "Manner must be one of:"),
            ({"model": "sa3", "sound_type": "song"}, "Sound type must be one of:"),
        ],
    )
    def test_a_bad_value_is_refused_in_the_users_words(self, env, fields, message):
        _refused(env.post(prompt="x", **fields), message)
        assert env.submitted == []

    def test_a_bad_value_of_another_kinds_setting_is_ignored_as_the_form_hides_it(self, env):
        assert env.post(model="sa3", prompt="rain", tempo="300", key="H", emotion="grumpy").status_code == 200
        assert env.post(model="yue2", prompt="pop", sound_type="song", pace="warp").status_code == 200

    def test_delivery_with_a_recording_is_ignored_even_when_wrong(self, env, monkeypatch, tmp_path):
        voices = voice_folder(tmp_path)

        response = env.post(model="qwen3_tts", prompt="Hi", voice_path=str(voices / "me.wav"), voice_text="Hello", voice_consent="on", emotion="grumpy")

        assert response.status_code == 200 and env.submitted[0]["request"].delivery is None


class TestRefusals:
    def test_speech_over_ten_thousand_characters_is_refused_with_the_models_message(self, env):
        response = env.post(model="qwen3_tts", prompt="a" * 10_001)

        _refused(response, "Shorten the text", "10,000 characters")
        assert env.submitted == []

    @pytest.mark.parametrize(("model", "message"), [("yue2", "Write a style first."), ("qwen3_tts", "Write the text to read first."), ("sa3", "Describe the sound first.")])
    def test_a_missing_prompt_names_what_to_write(self, env, model, message):
        assert _refused(env.post(model=model, prompt="  \n")) == message
        assert _refused(env.post(model=model)) == message

    def test_a_missing_model_lists_the_ones_that_run(self, env):
        _refused(env.post(prompt="x"), "Field 'model' is required", "yue2, qwen3_tts, sa3")

    def test_an_unknown_model_lists_the_ones_that_run(self, env):
        _refused(env.post(model="nope", prompt="x"), "Unknown audio model 'nope'. Choose one of: yue2, qwen3_tts, sa3.")

    def test_a_machine_without_audio_models_names_the_config_section(self, env):
        env.web_config.app_config.update({**load_packaged_config(), "audio_model_presets": {}})

        _refused(env.post(model="yue2", prompt="x"), "No audio models are configured", "audio_model_presets")

    @pytest.mark.parametrize("seed", ["-1", str(2**63)])
    def test_a_seed_no_model_takes_is_refused_before_anything_loads(self, env, seed):
        _refused(env.post(model="sa3", prompt="x", seed=seed), "seed")
        assert env.submitted == []

    def test_runs_below_one_are_refused(self, env):
        _refused(env.post(model="sa3", prompt="x", runs="-1"), "Runs must be at least 1")

    @pytest.mark.parametrize("seconds", ["0.5", "381", "nan", "inf"])
    def test_a_length_out_of_range_is_refused_with_the_clis_wording(self, env, seconds):
        _refused(env.post(model="sa3", prompt="x", seconds=seconds), "seconds must be between 1 and 380; Stable Audio 3 Medium makes sounds of that length.")

    def test_a_length_that_is_not_a_number_is_refused(self, env):
        _refused(env.post(model="sa3", prompt="x", seconds="long"), "'seconds' must be a number")


class TestSpeaker:
    def test_speech_without_a_speaker_is_read_in_the_models_default_voice(self, env):
        env.post(model="qwen3_tts", prompt="Hello.")

        assert env.submitted[0]["request"].speaker == "ryan"

    def test_the_voice_the_form_chose_is_the_speaker_of_the_take(self, env):
        env.post(model="qwen3_tts", prompt="Hello.", speaker="serena")

        assert env.submitted[0]["request"].speaker == "serena"

    def test_a_blank_speaker_is_the_default_voice(self, env):
        env.post(model="qwen3_tts", prompt="Hello.", speaker="")

        assert env.submitted[0]["request"].speaker == "ryan"

    def test_a_voice_the_model_lacks_is_refused_with_the_models_label(self, env):
        _refused(env.post(model="qwen3_tts", prompt="Hello.", speaker="nobody"), "Pick one of Qwen3-TTS 1.7B's voices.")
        assert env.submitted == []

    def test_the_plan_refuses_a_voice_the_model_lacks_as_the_submit_does(self, env):
        _refused(env.post("/api/generate/plan", model="qwen3_tts", prompt="Hello.", speaker="nobody"), "Pick one of Qwen3-TTS 1.7B's voices.")

    def test_a_speaker_is_ignored_by_the_kinds_that_have_no_voices(self, env):
        env.post(model="yue2", prompt="piano", speaker="ryan")
        env.post(model="sa3", prompt="rain", speaker="ryan")

        assert [job["request"].speaker for job in env.submitted] == [None, None]

    def test_the_job_settings_keep_the_speaker_for_retry_and_load_settings(self, env):
        env.post(model="qwen3_tts", prompt="Hello.", speaker="eric")

        assert env.submitted[0]["context"]["settings"]["speaker"] == "eric"

    def test_a_recording_has_no_speaker_whatever_the_form_says(self, env, tmp_path):
        voices = voice_folder(tmp_path)

        env.post(model="qwen3_tts", prompt="More.", speaker="eric", voice_path=str(voices / "any.flac"), voice_text="Hello", voice_consent="false")

        assert env.submitted[0]["request"].speaker is None


class TestVoice:
    @pytest.fixture
    def voices(self, tmp_path):
        return voice_folder(tmp_path)

    def speak(self, env, voices, name, **fields):
        return env.post(model="qwen3_tts", prompt="More words.", voice_path=str(voices / name), voice_text="Hello, this is me.", **fields)

    def test_a_recording_needs_the_speakers_consent(self, env, voices):
        _refused(self.speak(env, voices, "me.wav"), "permission")
        _refused(self.speak(env, voices, "me.wav", voice_consent="false"), "permission")
        assert env.submitted == []

    def test_a_ticked_box_lets_a_recording_through(self, env, voices):
        response = self.speak(env, voices, "me.wav", voice_consent=["false", "on"])

        assert response.status_code == 200
        assert env.submitted[0]["request"].voice == VoiceRef(str(voices / "me.wav"), "Hello, this is me.")

    def test_a_take_made_without_a_reference_voice_needs_no_consent(self, env, voices):
        assert self.speak(env, voices, "any.flac", voice_consent="false").status_code == 200

    def test_a_cloned_take_needs_consent_like_a_recording(self, env, voices):
        _refused(self.speak(env, voices, "cloned.flac"), "permission")
        assert self.speak(env, voices, "cloned.flac", voice_consent="on").status_code == 200

    def test_a_voice_needs_its_transcript(self, env, voices):
        response = env.post(model="qwen3_tts", prompt="x", voice_path=str(voices / "me.wav"), voice_consent="on")

        _refused(response, "Write what the recording says first.")

    def test_a_voice_outside_three_to_thirty_seconds_is_refused(self, env, tmp_path):
        short = tmp_path / "short.flac"
        write_fake_audio(clip(1.5).samples, 24000, short, {"kind": "speech", "prompt": "x"})

        _refused(env.post(model="qwen3_tts", prompt="x", voice_path=str(short), voice_text="Hi"), "The voice recording is 1.5 seconds long; use one between 3 and 30 seconds.")

    def test_a_voice_that_cannot_be_read_is_refused(self, env, tmp_path):
        broken = tmp_path / "broken.flac"
        broken.write_bytes(b"not audio")

        _refused(env.post(model="qwen3_tts", prompt="x", voice_path=str(broken), voice_text="Hi", voice_consent="on"), "Can't read")

    def test_a_voice_that_is_not_a_file_on_the_host_is_refused(self, env, tmp_path):
        _refused(env.post(model="qwen3_tts", prompt="x", voice_path=str(tmp_path / "gone.wav"), voice_text="Hi", voice_consent="on"), "voice_path", "existing host-local file")
        _refused(env.post(model="qwen3_tts", prompt="x", voice_path="https://example.com/a.wav", voice_text="Hi", voice_consent="on"), "host-local file path")

    def test_a_voice_path_is_used_as_typed_with_the_home_folder_expanded(self, env, voices, monkeypatch):
        monkeypatch.setenv("HOME", str(voices))
        monkeypatch.setenv("USERPROFILE", str(voices))

        response = env.post(model="qwen3_tts", prompt="x", voice_path="~/any.flac", voice_text="Hi")

        assert response.status_code == 200
        assert env.submitted[0]["request"].voice.path == str(voices / "any.flac")

    def test_an_upload_wins_over_the_typed_path_and_is_saved_with_its_name(self, env, voices):
        upload = ("My Voice.flac", io.BytesIO((voices / "any.flac").read_bytes()), "audio/flac")

        response = env.client.post(
            "/api/generate",
            data={"mode": "audio", "model": "qwen3_tts", "prompt": "x", "voice_path": str(voices / "me.wav"), "voice_text": "Welcome back."},
            files={"voice_file": upload},
        )

        assert response.status_code == 200, response.text
        saved = env.submitted[0]["request"].voice.path
        assert saved.startswith(str(env.root / "out" / ".web_uploads")) and saved.endswith("/My Voice.flac")
        assert env.submitted[0]["context"]["settings"]["voice_path"] == saved

    def test_an_upload_that_is_not_audio_is_refused_by_name(self, env):
        response = env.client.post(
            "/api/generate", data={"mode": "audio", "model": "qwen3_tts", "prompt": "x", "voice_text": "Hi"}, files={"voice_file": ("notes.txt", io.BytesIO(b"hi"), "text/plain")}
        )

        _refused(response, "Voice recording must be a WAV, MP3, M4A, FLAC, or OGG file.")

    def test_the_job_log_record_of_a_cloned_voice_job_holds_no_consent(self, env, voices, monkeypatch, tmp_path):
        from zvisiongenerator.web.job_log import JobLog

        self.speak(env, voices, "cloned.flac", voice_consent=["false", "on"])
        job = env.submitted[0]
        log = JobLog(lambda: tmp_path / "log.sqlite3")
        monkeypatch.setattr(web_runner_module, "run_audio_batch", lambda *_a, **_k: None)
        monkeypatch.setattr(web_runner_module, "release_accelerator_memory", lambda: None)
        monkeypatch.setattr(web_runner_module, "require_ffmpeg", lambda: None)
        monkeypatch.setattr(web_runner_module, "ensure_downloaded", lambda *_a, **_k: "ready")
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01, job_log=log)
        try:
            job_id = runner.submit_audio_request_job(request=job["request"], preset=job["preset"], config=job["config"], runs=1, seed=None, output_dir=job["output_dir"], context=job["context"])
            wait_for_status(runner, job_id, "completed", timeout=5)
            runner._executor.submit(lambda: None).result(timeout=2.0)
        finally:
            runner.shutdown()

        (entry,) = log.page("all", 1, 10)["entries"]
        assert entry["job_type"] == "audio" and entry["settings"]["voice_text"] == "Hello, this is me." and "voice_consent" not in entry["settings"]

    def test_the_tick_reaches_no_store_from_the_form_to_the_saved_file(self, env, voices, monkeypatch):
        """Trace one ticked submit through every place it is kept: the job's context, the recent prompts and the take's embedded settings."""
        from zvisiongenerator.audio_runner import run_audio_batch

        patch_runner(monkeypatch, fake_backend(clip(0.2)))
        response = self.speak(env, voices, "cloned.flac", voice_consent=["false", "on"])
        assert response.status_code == 200
        (job,) = env.submitted
        saved = run_audio_batch(job["request"], job["preset"], job["config"], runs=1, seed=3, output_dir=str(env.root / "takes"))

        assert "consent" not in json.dumps(job["context"]).lower()
        remembered = web_server.recent_prompts.search("", "speech")
        assert [entry["prompt"] for entry in remembered] == ["More words."] and "consent" not in json.dumps(remembered).lower()
        assert not any("consent" in key for key in fake_read_config(saved[0]))

    def test_a_loaded_or_retried_cloned_job_must_be_ticked_again(self, env, voices):
        self.speak(env, voices, "cloned.flac", voice_consent="on")
        settings = env.submitted[0]["context"]["settings"]

        _refused(env.post(**settings), "permission")


class TestMissingFfmpeg:
    """Without ffmpeg a recording cannot be read, so the form is refused with the ffmpeg message before it is judged on the recording."""

    @pytest.fixture
    def no_ffmpeg(self, env, monkeypatch, tmp_path):
        def missing() -> None:
            raise RuntimeError("ffmpeg is required but was not found on PATH.")

        monkeypatch.setattr(web_server, "require_ffmpeg", missing)
        env.voices = voice_folder(tmp_path)
        return env

    @pytest.mark.parametrize("path", ["/api/generate", "/api/generate/plan"])
    def test_a_recording_form_gets_the_ffmpeg_message_not_a_consent_refusal(self, no_ffmpeg, path):
        response = no_ffmpeg.post(path, model="qwen3_tts", prompt="More words.", voice_path=str(no_ffmpeg.voices / "me.wav"), voice_text="Hello, this is me.")

        detail = _refused(response, "ffmpeg")
        assert "permission" not in detail
        assert no_ffmpeg.submitted == []

    @pytest.mark.parametrize("path", ["/api/generate", "/api/generate/plan"])
    def test_a_form_with_no_recording_is_refused_the_same_way(self, no_ffmpeg, path):
        _refused(no_ffmpeg.post(path, model="sa3", prompt="rain"), "ffmpeg")


class TestPlan:
    @pytest.fixture
    def plan_env(self, env, monkeypatch, tmp_path):
        """Serve the plan with a fake Hub and disk."""
        env.present = set()
        monkeypatch.setattr(web_server.download_flights, "active", DownloadFlights().active)
        real_summary = plan_module.parts_summary
        monkeypatch.setattr(
            web_server,
            "parts_summary",
            lambda parts, **kwargs: real_summary(
                parts,
                **kwargs,
                check=lambda target: TargetCheck(missing_files=2, missing_bytes=3 * _GIB),
                present=lambda target: target.repo_id in env.present,
                space=lambda items, *, cache_dir: SpaceNeed(path=cache_dir, needed_bytes=sum(c.missing_bytes for _, c in items), free_bytes=100 * _GIB),
                cache_dir=lambda: tmp_path,
            ),
        )
        return env

    def test_the_plan_of_an_audio_form_lists_the_models_repositories_and_never_queues(self, plan_env):
        response = plan_env.post("/api/generate/plan", model="yue2", prompt="calm piano")

        assert response.status_code == 200
        plan = response.json()
        # One part for the model, summing its two repositories (the model and the VAE).
        assert [(part["label"], part["model"], part["bytes"]) for part in plan["parts"]] == [("YuE2 3B", "yue2", 6 * _GIB)]
        assert plan["total_bytes"] == 6 * _GIB
        assert plan_env.submitted == []

    @pytest.mark.parametrize(
        ("model", "suppliers"),
        [("yue2", ["M-A-P"]), ("qwen3_tts", ["Qwen"]), ("sa3", ["Stability AI", "Google"])],
    )
    def test_the_plan_carries_the_packaged_records_of_each_audio_model(self, plan_env, model, suppliers):
        licences = plan_env.post("/api/generate/plan", model=model, prompt="calm piano").json()["licences"]

        assert [(entry["model"], entry["supplier"]) for entry in licences] == [(model, supplier) for supplier in suppliers]
        assert all(entry["url"].startswith("https://") for entry in licences)

    def test_a_model_on_disk_has_nothing_to_download(self, plan_env):
        plan_env.present = {"m-a-p/YuE2-3B", "m-a-p/YuE2-Vae"}

        plan = plan_env.post("/api/generate/plan", model="yue2", prompt="calm piano").json()

        assert plan["parts"] == [] and plan["licences"] == []

    def test_an_audio_form_is_refused_by_the_plan_as_the_submit_refuses_it(self, plan_env):
        fields = {"model": "qwen3_tts", "prompt": "a" * 10_001}

        planned = plan_env.post("/api/generate/plan", **fields)
        submitted = plan_env.post("/api/generate", **fields)

        assert planned.status_code == submitted.status_code == 422
        assert planned.json() == submitted.json()

    @pytest.mark.parametrize(("model", "fields"), [("yue2", {"prompt": "calm piano"}), ("sa3", {"prompt": "rain", "seconds": "10"}), ("qwen3_tts", {"prompt": "Hello there."})])
    def test_the_plan_of_each_kind_is_the_same_with_or_without_the_voice_file(self, plan_env, model, fields):
        voice = {"voice_text": "Welcome back."} if model == "qwen3_tts" else {}
        data = {"mode": "audio", "model": model, **fields, **voice}
        files = {"voice_file": ("me.wav", b"RIFF....", "audio/wav")} if model == "qwen3_tts" else None

        full = plan_env.client.post("/api/generate/plan", data=data, files=files)
        bare = plan_env.client.post("/api/generate/plan", data=data, files={"voice_file": ("me.wav", b"", "audio/wav")} if files else None)

        assert full.status_code == bare.status_code == 200, (full.text, bare.text)
        assert bare.json() == full.json()
        assert bare.json()["parts"]
        assert not (plan_env.root / "out" / ".web_uploads").exists()

    def test_the_plan_refuses_the_voice_file_type_the_submit_refuses(self, plan_env):
        data = {"mode": "audio", "model": "qwen3_tts", "prompt": "Hello there.", "voice_text": "Welcome back."}

        planned = plan_env.client.post("/api/generate/plan", data=data, files={"voice_file": ("notes.txt", b"", "text/plain")})
        submitted = plan_env.client.post("/api/generate", data=data, files={"voice_file": ("notes.txt", b"hi", "text/plain")})

        assert planned.status_code == submitted.status_code == 422
        assert planned.json() == submitted.json()

    def test_the_plan_leaves_the_content_of_the_voice_file_to_the_submit(self, plan_env):
        data = {"mode": "audio", "model": "qwen3_tts", "prompt": "Hello there.", "voice_text": "Welcome back."}
        voice = {"voice_file": ("me.wav", b"", "audio/wav")}

        planned = plan_env.client.post("/api/generate/plan", data=data, files=voice)
        submitted = plan_env.client.post("/api/generate", data=data, files={"voice_file": ("me.wav", b"hi", "audio/wav")})

        assert planned.status_code == 200  # no consent tick and no readable recording: both checked on the file, at submit
        assert submitted.status_code == 422

    def test_the_voice_fields_are_still_checked_by_the_plan(self, plan_env):
        files = {"voice_file": ("me.wav", b"", "audio/wav")}

        missing_text = plan_env.client.post("/api/generate/plan", data={"mode": "audio", "model": "qwen3_tts", "prompt": "Hello there."}, files=files)
        no_prompt = plan_env.client.post("/api/generate/plan", data={"mode": "audio", "model": "qwen3_tts", "prompt": "", "voice_text": "Hi"}, files=files)

        _refused(missing_text, "Write what the recording says first.")
        _refused(no_prompt, "Write the text to read first.")


def _ready(targets, *, flights, report, cancelled):
    """Stand in for the downloads: report one download and say the files are ready."""
    report(DownloadReport("m-a-p/YuE2-3B", 1, 2, DownloadProgress(10, 100, 5.0, 18.0)))
    return "ready"


class TestRunner:
    """``submit_audio_request_job`` runs the real batch loop with a fake backend and reports it as the job's events."""

    @pytest.fixture
    def runner(self, monkeypatch):
        monkeypatch.setattr(web_runner_module, "require_ffmpeg", lambda: None)
        monkeypatch.setattr(web_runner_module, "release_accelerator_memory", lambda: None)
        web_config = make_web_config()
        web_config.app_config.update(audio_config())
        monkeypatch.setattr(web_runner_module, "load_web_config", lambda: web_config)  # the gallery JSON of a saved take reads only the config
        _set_platform(monkeypatch, "cuda")
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01, download_flights=DownloadFlights())
        runner.ensure = MagicMock(side_effect=_ready)  # the downloads: one report, then ready
        monkeypatch.setattr(web_runner_module, "ensure_downloaded", runner.ensure)
        yield runner
        runner.shutdown()

    @staticmethod
    def submit(runner, tmp_path, **kwargs):
        preset = next(item for item in parse_audio_presets(audio_config()) if item.id == "sa3")
        request = AudioGenerationRequest(preset="sa3", kind="sound", prompt="rain on a roof", seconds=2.0)
        defaults = {
            "request": request,
            "preset": preset,
            "config": audio_config(),
            "runs": 2,
            "seed": 5,
            "output_dir": str(tmp_path),
            "context": {"workflow": "sound", "model": "sa3", "prompt": "rain on a roof"},
        }
        return runner.submit_audio_request_job(**{**defaults, **kwargs})

    @staticmethod
    def events(runner, job_id):
        return [dict(event) for event in runner._get_job(job_id).history]

    def test_a_job_makes_each_take_and_reports_the_runners_events(self, runner, monkeypatch, tmp_path):
        backend = fake_backend(clip(0.3, rate=44100, channels=2))

        def generate(request, on_progress, cancelled):
            on_progress("Sampling", 1, 2, 1, 1)
            on_progress("Sampling", 2, 2, 1, 1)
            return clip(0.3, rate=44100, channels=2)

        backend.generate.side_effect = generate
        patch_runner(monkeypatch, backend)

        job_id = self.submit(runner, tmp_path)
        snapshot = wait_for_status(runner, job_id, "completed", timeout=10)

        assert snapshot["status"] == "completed"
        events = self.events(runner, job_id)
        assert [event["type"] for event in events if event["type"] not in ("audio_progress", "model_downloading")] == [
            "job_submitted",
            "job_started",
            "model_loading",
            "batch_started",
            "prompt_started",
            "generation_started",
            "generation_finished",
            "prompt_started",
            "generation_started",
            "generation_finished",
            "batch_completed",
            "job_completed",
        ]
        progress = [event for event in events if event["type"] == "audio_progress"]
        assert [(e["phase"], e["step"], e["total"], e["part"], e["n_parts"], e["mode"]) for e in progress[:2]] == [("Sampling", 1, 2, 1, 1, "audio"), ("Sampling", 2, 2, 1, 1, "audio")]
        assert all(e["job_type"] == "audio" and e["total_runs"] == 2 and "run_index" in e for e in progress)
        (download,) = [event for event in events if event["type"] == "model_downloading"]
        assert (download["mode"], download["role"], download["model"], download["repo"], download["part"], download["parts"], download["bytes_done"], download["bytes_total"]) == (
            "audio",
            "model",
            "Stable Audio 3 Medium",
            "m-a-p/YuE2-3B",
            1,
            2,
            10,
            100,
        )
        assert runner.ensure.call_args.kwargs["flights"] is runner.download_flights
        finished = [event for event in events if event["type"] == "generation_finished"]
        assert [event["status"] for event in finished] == ["success", "success"]
        assert all(event["asset"]["media_type"] == "audio" and event["asset"]["kind"] == "sound" and event["asset"]["has_reusable_config"] for event in finished)
        assert len(snapshot["outputs"]) == 2 and len(list(tmp_path.glob("*.flac"))) == 2
        assert events[-1]["outputs"] == snapshot["outputs"]

    def test_a_failed_take_ends_the_job_failed_and_keeps_the_takes_before_it(self, runner, monkeypatch, tmp_path):
        backend = fake_backend(clip(0.3, rate=44100, channels=2))
        takes = iter([clip(0.3, rate=44100, channels=2), RuntimeError("Stable Audio 3 Medium ran out of GPU memory. Close other apps that use the GPU and try again.")])

        def generate(request, on_progress, cancelled):
            take = next(takes)
            if isinstance(take, Exception):
                raise take
            return take

        backend.generate.side_effect = generate
        patch_runner(monkeypatch, backend)

        job_id = self.submit(runner, tmp_path)
        snapshot = wait_for_status(runner, job_id, "failed", timeout=10)

        assert snapshot["status"] == "failed" and snapshot["terminal_event"] == "job_failed"
        events = self.events(runner, job_id)
        assert events[-1]["type"] == "job_failed" and "ran out of GPU memory" in events[-1]["message"]
        assert [event["status"] for event in events if event["type"] == "generation_finished"] == ["success"]
        assert len(snapshot["outputs"]) == 1 and len(list(tmp_path.glob("*.flac"))) == 1
        assert backend.close.called

    def test_a_stop_before_the_model_loads_ends_the_job_cancelled(self, runner, monkeypatch, tmp_path):
        backend = fake_backend(clip(0.3))
        patch_runner(monkeypatch, backend)
        runner.ensure.side_effect = lambda targets, *, flights, report, cancelled: "stopped"

        job_id = self.submit(runner, tmp_path)
        snapshot = wait_for_status(runner, job_id, "cancelled", timeout=10)

        assert snapshot["status"] == "cancelled" and snapshot["terminal_event"] == "job_cancelled"
        assert not backend.generate.called

    def test_a_stop_queued_before_the_job_starts_cancels_it_before_any_download_or_load(self, runner, monkeypatch, tmp_path):
        backend = fake_backend(clip(0.3))
        patch_runner(monkeypatch, backend)
        control = SkipSignal()
        control.queue_action("quit")
        events: list[dict] = []
        preset = next(item for item in parse_audio_presets(audio_config()) if item.id == "sa3")

        runner._run_audio_request(
            request=AudioGenerationRequest(preset="sa3", kind="sound", prompt="rain", seconds=2.0),
            preset=preset,
            config=audio_config(),
            runs=2,
            seed=5,
            output_dir=str(tmp_path),
            progress_callback=events.append,
            control_signal=control,
        )

        assert events == [{"type": "batch_cancelled", "mode": "audio", "completed_iterations": 0, "total_iterations": 2}]
        assert not runner.ensure.called and not backend.load.called

    def test_a_stop_during_the_model_load_ends_the_job_cancelled_and_frees_the_backend(self, runner, monkeypatch, tmp_path):
        backend = fake_backend(clip(0.3))
        patch_runner(monkeypatch, backend)
        loading = threading.Event()

        def load(preset, device, cancelled=None):
            loading.set()
            deadline = time.monotonic() + 5
            while not cancelled():
                assert time.monotonic() < deadline
                time.sleep(0.01)
            raise AudioCancelled

        backend.load.side_effect = load

        job_id = self.submit(runner, tmp_path)
        assert loading.wait(5)
        runner.queue_job_control(job_id, "quit")
        snapshot = wait_for_status(runner, job_id, "cancelled", timeout=10)

        assert snapshot["terminal_event"] == "job_cancelled"
        assert not backend.generate.called and backend.close.called

    def test_the_job_is_exclusive_and_offers_the_image_controls(self, runner, monkeypatch, tmp_path):
        release = threading.Event()
        monkeypatch.setattr(web_runner_module, "run_audio_batch", lambda *_a, **_k: release.wait(5))

        job_id = self.submit(runner, tmp_path)
        wait_for_status(runner, job_id, "running", timeout=5)
        try:
            snapshot = runner.get_job_snapshot(job_id)
            assert snapshot["supported_controls"] == list(generation_supported_controls(review=False))
            assert runner._get_job(job_id).exclusive and runner._get_job(job_id).job_type == "audio"
            assert runner.list_jobs()["active_job"]["job_id"] == job_id
        finally:
            release.set()

    def test_the_job_runs_the_batch_with_its_own_arguments(self, runner, monkeypatch, tmp_path):
        seen = {}
        monkeypatch.setattr(web_runner_module, "run_audio_batch", lambda *args, **kwargs: seen.update(args=args, kwargs=kwargs))

        job_id = self.submit(runner, tmp_path)
        wait_for_status(runner, job_id, "completed", timeout=5)

        request, preset, config = seen["args"]
        kwargs = seen["kwargs"]
        assert (request.prompt, preset.id) == ("rain on a roof", "sa3") and config == audio_config()
        assert (kwargs["runs"], kwargs["seed"], kwargs["output_dir"]) == (2, 5, str(tmp_path))
        assert kwargs["skip_signal"] is runner._get_job(job_id).control_signal

    def _run_qwen(self, runner, monkeypatch, tmp_path, **request_fields):
        patch_runner(monkeypatch, fake_backend(clip(0.3)))
        config = audio_config(qwen3_tts=qwen3_tts_preset())
        preset = next(item for item in parse_audio_presets(config) if item.id == "qwen3_tts")
        request = AudioGenerationRequest(preset="qwen3_tts", kind="speech", prompt="Hello there.", **request_fields)

        job_id = self.submit(runner, tmp_path, request=request, preset=preset, config=config, runs=1, context={"workflow": "speech", "model": "qwen3_tts", "prompt": "Hello there."})
        wait_for_status(runner, job_id, "completed", timeout=10)
        return [target.repo_id for target in runner.ensure.call_args.args[0]]

    def test_a_built_in_voice_job_downloads_customvoice_alone(self, runner, monkeypatch, tmp_path):
        assert self._run_qwen(runner, monkeypatch, tmp_path, speaker="aiden") == ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"]

    def test_a_job_with_a_recording_downloads_the_cloning_checkpoint_too(self, runner, monkeypatch, tmp_path):
        repos = self._run_qwen(runner, monkeypatch, tmp_path, voice=VoiceRef("/voices/me.wav", "Hello, this is me."))

        assert repos == ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"]


class TestQwenCloningFiles:
    """Qwen3-TTS downloads its cloning checkpoint only for a take that follows a recording: the plan and the queued job say so alike."""

    _BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    _CUSTOM = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

    @pytest.fixture
    def qwen_env(self, env, monkeypatch, tmp_path):
        """Serve the plan with a fake Hub and disk, and Qwen3-TTS among the models."""
        env.web_config.app_config.update(audio_config(qwen3_tts=qwen3_tts_preset()))
        monkeypatch.setattr(web_server.download_flights, "active", DownloadFlights().active)
        real_summary = plan_module.parts_summary
        monkeypatch.setattr(
            web_server,
            "parts_summary",
            lambda parts, **kwargs: real_summary(
                parts,
                **kwargs,
                check=lambda target: TargetCheck(missing_files=2, missing_bytes=3 * _GIB),
                present=lambda target: False,
                space=lambda items, *, cache_dir: SpaceNeed(path=cache_dir, needed_bytes=sum(c.missing_bytes for _, c in items), free_bytes=100 * _GIB),
                cache_dir=lambda: tmp_path,
            ),
        )
        return env

    def test_a_built_in_voice_take_plans_and_queues_customvoice_alone(self, qwen_env):
        plan = qwen_env.post("/api/generate/plan", model="qwen3_tts", prompt="Hello there.").json()
        qwen_env.post(model="qwen3_tts", prompt="Hello there.")

        assert plan["parts"][0]["bytes"] == 3 * _GIB
        assert [target.repo_id for target in qwen_env.submitted[0]["planned_targets"]] == [self._CUSTOM]

    def test_a_take_with_a_recording_also_plans_the_cloning_checkpoint(self, qwen_env):
        data = {"mode": "audio", "model": "qwen3_tts", "prompt": "Hello there.", "voice_text": "Welcome back."}

        plan = qwen_env.client.post("/api/generate/plan", data=data, files={"voice_file": ("me.wav", b"", "audio/wav")}).json()

        assert plan["parts"][0]["bytes"] == 6 * _GIB

    def test_a_queued_take_with_a_recording_holds_both_checkpoints(self, qwen_env, tmp_path):
        voices = voice_folder(tmp_path)

        qwen_env.post(model="qwen3_tts", prompt="Hello there.", voice_path=str(voices / "me.wav"), voice_text="Hello, this is me.", voice_consent="on")

        assert [target.repo_id for target in qwen_env.submitted[0]["planned_targets"]] == [self._CUSTOM, self._BASE]
