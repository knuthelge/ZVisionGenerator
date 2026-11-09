"""Test the voice design routes: plan, queue, the candidates, their audio, saving one and clearing them, with a fake designer backend."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tests._audio_presets import audio_config
from tests._speech_fakes import RATE, tone
from tests.audio._helpers import patch_runner
from tests.web._helpers import make_web_config, wait_for_status

from zvisiongenerator.downloads.flights import DownloadFlights
from zvisiongenerator.utils.voice_candidates import DesignBatch, add_batch, candidates_dir, read_batches, write_candidate
from zvisiongenerator.utils.voice_library import list_voices
from zvisiongenerator.web import audio_models as audio_models_module
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web import web_runner as web_runner_module
from zvisiongenerator.web.job_log import JobLog

_DESCRIPTION = "A calm, low voice with a slow pace."
_LINE = "Hello there, this is how I sound when I read a story aloud."
_BODY = {"model": "qwen3_tts", "description": _DESCRIPTION, "text": _LINE}


class FakeDesigner:
    """A backend that designs voices: one tone per candidate (*seconds*, the last repeats); it can wait on *gate* before designing."""

    def __init__(self, *seconds: float, gate: threading.Event | None = None, fail: str | None = None) -> None:
        self.seconds = list(seconds) or [4.0]
        self.gate, self.fail = gate, fail
        self.designed = 0
        self.started = threading.Event()
        self.sample_rate = RATE

    def load(self, preset, platform_key, cancelled=None) -> None: ...

    def close(self) -> None: ...

    def design_voice(self, description: str, text: str, seed: int):
        self.started.set()
        if self.gate is not None:
            self.gate.wait(timeout=10)
        if self.fail:
            raise RuntimeError(self.fail)
        value = self.seconds[min(self.designed, len(self.seconds) - 1)]
        self.designed += 1
        return tone(value)


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Serve the Web UI with the fixture audio presets on a CUDA machine and a real queue whose downloads are on disk; the job log is a temporary database."""
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path / "out")
    web_config.app_config.update(audio_config())
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    monkeypatch.setattr(audio_models_module, "audio_platform_key", lambda: "cuda")
    monkeypatch.setattr(web_runner_module, "release_accelerator_memory", lambda: None)
    monkeypatch.setattr(web_runner_module, "ensure_downloaded", lambda *_a, **_k: "ready")
    log = JobLog(lambda: tmp_path / "log.sqlite3")
    runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01, download_flights=DownloadFlights(), job_log=log)
    monkeypatch.setattr(web_server, "web_runner", runner)
    monkeypatch.setattr(web_server, "job_log", log)
    state = SimpleNamespace(web_config=web_config, runner=runner, log=log, root=tmp_path, backend=FakeDesigner())
    with TestClient(web_server.app) as client:
        state.client = client

        def design(backend: FakeDesigner | None = None, body: dict | None = None, *, wait: str = "completed"):
            """Queue a job with *backend* behind the runner, wait for *wait* and return its id."""
            state.backend = backend or state.backend
            patch_runner(monkeypatch, state.backend)
            response = client.post("/api/voices/design", json=body or _BODY)
            assert response.status_code == 200, response.text
            state.submitted = response.json()
            job_id = state.submitted["job_id"]
            if wait:
                assert wait_for_status(runner, job_id, wait, timeout=10)["status"] == wait
            return job_id

        state.design = design
        yield state
    runner.shutdown()


def _design_payload(env) -> dict:
    return env.client.get("/api/voices/design").json()


def _candidates(env) -> list[dict]:
    """Return every candidate of the payload, newest batch first."""
    return [item for batch in _design_payload(env)["batches"] for item in batch["candidates"]]


_LONG_AGO = "2026-01-01T00:00:00+00:00"
_LATER = "2999-01-01T00:00:00+00:00"


def _write_batch(seconds: list[float], *, batch_id: str = "batch1", description: str = _DESCRIPTION, text: str = _LINE, model: str = "qwen3_tts", created_at: str = _LONG_AGO) -> DesignBatch:
    """Add a finished batch of tones of the given lengths, made at *created_at*, to the candidate folder."""
    folder = candidates_dir()
    items = tuple(write_candidate(folder, tone(length), RATE) for length in seconds)
    batch = DesignBatch(batch_id, model, description, text, created_at, items)
    add_batch(folder, batch)
    return batch


class TestPlan:
    def test_the_plan_is_the_design_checkpoint_alone(self, env, monkeypatch):
        seen = []
        monkeypatch.setattr(web_server, "_summarize_job_parts", lambda parts: seen.append(parts) or {"parts": []})

        response = env.client.post("/api/voices/design/plan", json=_BODY)

        assert response.status_code == 200
        (parts,) = seen
        assert [part.label for part in parts] == ["Qwen3-TTS 1.7B VoiceDesign"]
        assert [target.repo_id for target in parts[0].targets] == ["Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"]

    @pytest.mark.parametrize(
        ("changes", "needle"),
        [
            ({"model": "sa3"}, "cannot design voices"),
            ({"model": "nope"}, "nope"),
            ({"model": None}, "model"),
            ({"description": "  "}, "Describe the voice"),
            ({"description": "x" * 501}, "500"),
            ({"text": ""}, "line"),
            ({"text": "x" * 501}, "500"),
        ],
    )
    def test_a_bad_request_is_refused_with_422_by_the_plan_and_the_submit(self, env, changes, needle):
        body = {**_BODY, **changes}
        for path in ("/api/voices/design/plan", "/api/voices/design"):
            response = env.client.post(path, json=body)

            assert response.status_code == 422
            assert needle in response.json()["detail"]
        assert env.runner.get_active_exclusive_job_snapshot() is None


class TestSubmit:
    def test_the_job_has_its_own_context_without_settings(self, env):
        gate = threading.Event()
        job_id = env.design(FakeDesigner(gate=gate), wait="")
        snapshot = env.client.get(f"/jobs/{job_id}").json()
        gate.set()
        wait_for_status(env.runner, job_id, "completed", timeout=10)
        body = env.submitted

        assert "context" not in body and "settings" not in body
        assert (body["workflow"], body["job_type"], body["media_type"]) == ("voice_design", "Voice design", "audio")
        assert (body["model"], body["model_label"], body["prompt"], body["runs"], body["meta"]) == ("qwen3_tts", "Qwen3-TTS 1.7B VoiceDesign", _DESCRIPTION, 3, "3 voices")
        assert body["supported_controls"] == ["quit"]
        assert (body["job_id"], body["events_url"], body["status_url"]) == (job_id, f"/jobs/{job_id}/events", f"/jobs/{job_id}")
        assert "queue_position" in body and body["created_at"] and "output_dir" not in body
        assert snapshot["workflow"] == "voice_design" and snapshot["settings"] == {} and snapshot["supported_controls"] == ["quit"]

    def test_a_job_that_fails_reports_the_backends_error(self, env):
        job_id = env.design(FakeDesigner(fail="made no sound"), wait="failed")

        events = [dict(event) for event in env.runner._get_job(job_id).history]
        assert events[-1]["type"] == "job_failed" and "made no sound" in events[-1]["message"]

    def test_quit_while_queued_cancels_the_job(self, env, monkeypatch):
        gate = threading.Event()
        blocker = FakeDesigner(gate=gate)
        blocker_id = env.design(blocker, wait="")
        assert blocker.started.wait(timeout=5)
        second = env.client.post("/api/voices/design", json=_BODY).json()["job_id"]

        assert env.client.post(f"/jobs/{second}/controls/quit").status_code == 200
        gate.set()

        assert wait_for_status(env.runner, second, "cancelled", timeout=10)["status"] == "cancelled"
        wait_for_status(env.runner, blocker_id, "completed", timeout=10)

    def test_other_controls_are_refused(self, env):
        gate = threading.Event()
        job_id = env.design(FakeDesigner(gate=gate), wait="")
        try:
            assert env.client.post(f"/jobs/{job_id}/controls/pause").status_code == 409
        finally:
            gate.set()
            wait_for_status(env.runner, job_id, "completed", timeout=10)


class TestCandidates:
    def test_no_candidates_give_an_empty_list(self, env):
        assert _design_payload(env) == {"batches": [], "unpicked": 0}

    def test_a_finished_job_leaves_three_candidates_with_ids_and_its_batch_details(self, env):
        env.design(FakeDesigner(4.0, 5.5, 6.0))

        payload = _design_payload(env)

        (batch,) = payload["batches"]
        assert read_batches(candidates_dir())[0].batch_id == batch["batch_id"]
        assert (batch["description"], batch["text"]) == (_DESCRIPTION, _LINE)
        assert batch["created_at"]
        assert [(item["number"], item["seconds"], item["saveable"], item["reason"], item["saved_as"]) for item in batch["candidates"]] == [
            (1, 4.0, True, None, None),
            (2, 5.5, True, None, None),
            (3, 6.0, True, None, None),
        ]
        assert len({item["id"] for item in batch["candidates"]}) == 3
        assert payload["unpicked"] == 3

    def test_two_jobs_accumulate_newest_batch_first_with_stable_ids_and_running_numbers(self, env):
        env.design(FakeDesigner(4.0))
        first = _design_payload(env)["batches"][0]
        env.design(FakeDesigner(5.0), body={**_BODY, "description": "A bright voice.", "text": "Another line, long enough to read."})

        payload = _design_payload(env)

        newest, oldest = payload["batches"]
        assert oldest == first | {"candidates": [{**item} for item in first["candidates"]]}
        assert (newest["description"], newest["text"]) == ("A bright voice.", "Another line, long enough to read.")
        assert [item["number"] for item in oldest["candidates"]] == [1, 2, 3]
        assert [item["number"] for item in newest["candidates"]] == [4, 5, 6]
        assert len({item["id"] for batch in payload["batches"] for item in batch["candidates"]}) == 6
        assert payload["unpicked"] == 6

    def test_a_candidate_outside_the_voice_limits_is_not_saveable_with_its_reason_and_is_not_unpicked(self, env):
        _write_batch([2.0, 4.0, 31.0])

        payload = _design_payload(env)

        reasons = [(item["saveable"], item["reason"]) for item in payload["batches"][0]["candidates"]]
        assert reasons == [(False, "Too short to save as a voice; make the line longer."), (True, None), (False, "Too long to save as a voice; make the line shorter.")]
        assert payload["unpicked"] == 1

    def test_a_batch_of_an_unavailable_model_is_listed_as_not_saveable_beside_the_others(self, env):
        _write_batch([4.0], batch_id="gone", model="gone")
        _write_batch([4.0], batch_id="here")

        payload = _design_payload(env)

        gone = next(batch for batch in payload["batches"] if batch["batch_id"] == "gone")
        here = next(batch for batch in payload["batches"] if batch["batch_id"] == "here")
        assert [(item["saveable"], "not available" in item["reason"]) for item in gone["candidates"]] == [(False, True)]
        assert [item["saveable"] for item in here["candidates"]] == [True]
        assert payload["unpicked"] == 1


class TestReadyMarker:
    """``unpicked`` counts the unsaved candidates of the batches made since the last save or Clear voices."""

    def test_a_batch_made_after_the_save_raises_it_again_for_its_own_candidates_only(self, env):
        first = _write_batch([4.0, 5.0], batch_id="first")
        env.client.post(f"/api/voices/design/{first.candidates[0].id}/save", json={"name": "Anna"})

        _write_batch([4.0, 5.0, 6.0], batch_id="second", created_at=_LATER)

        assert _design_payload(env)["unpicked"] == 3

    def test_a_batch_made_before_the_save_stays_uncounted_even_when_it_is_newer_than_the_saved_one(self, env):
        older = _write_batch([4.0], batch_id="older", created_at="2026-01-01T00:00:00+00:00")
        _write_batch([4.0, 5.0], batch_id="newer", created_at="2026-02-01T00:00:00+00:00")

        env.client.post(f"/api/voices/design/{older.candidates[0].id}/save", json={"name": "Anna"})

        assert _design_payload(env)["unpicked"] == 0

    def test_clearing_the_voices_starts_the_count_over(self, env):
        batch = _write_batch([4.0])
        env.client.post(f"/api/voices/design/{batch.candidates[0].id}/save", json={"name": "Anna"})
        env.client.delete("/api/voices/design/candidates")

        _write_batch([4.0, 5.0], batch_id="fresh")

        assert _design_payload(env)["unpicked"] == 2


class TestCandidateAudio:
    def test_a_candidate_is_served_as_wav_by_its_id(self, env):
        batch = _write_batch([4.0, 5.0])
        candidate = batch.candidates[1]

        response = env.client.get(f"/api/voices/design/{candidate.id}/audio")

        assert response.status_code == 200 and response.headers["content-type"] == "audio/wav"
        assert "no-store" not in response.headers["cache-control"]
        assert response.content == (candidates_dir() / f"{candidate.id}.wav").read_bytes()

    def test_an_unknown_id_is_404(self, env):
        _write_batch([4.0])

        assert env.client.get("/api/voices/design/nope/audio").status_code == 404

    def test_a_cleared_candidate_is_404(self, env):
        (candidate,) = _write_batch([4.0]).candidates
        env.client.delete("/api/voices/design/candidates")

        assert env.client.get(f"/api/voices/design/{candidate.id}/audio").status_code == 404


class TestSave:
    def test_a_candidate_is_saved_as_a_designed_voice_with_the_line_as_its_text(self, env):
        batch = _write_batch([4.0, 5.0])

        response = env.client.post(f"/api/voices/design/{batch.candidates[1].id}/save", json={"name": "Storyteller"})

        assert response.status_code == 200
        voice = response.json()
        assert (voice["name"], voice["origin"], voice["text"], voice["description"], voice["seconds"]) == ("Storyteller", "design", _LINE, _DESCRIPTION, 5.0)
        assert "consent" not in voice and "path" not in voice
        (saved,) = list_voices()
        assert (saved.origin, saved.consent, saved.text) == ("design", "not_needed", _LINE)
        assert env.client.get(f"/api/voices/{voice['slug']}/audio").status_code == 200

    def test_saving_keeps_the_others_and_marks_the_saved_one_with_its_name_and_clears_the_marker(self, env):
        batch = _write_batch([4.0, 5.0, 6.0])
        picked = batch.candidates[1]

        assert env.client.post(f"/api/voices/design/{picked.id}/save", json={"name": "Morgan"}).status_code == 200

        payload = _design_payload(env)
        assert [(item["id"], item["saved_as"]) for item in payload["batches"][0]["candidates"]] == [
            (batch.candidates[0].id, None),
            (picked.id, "Morgan"),
            (batch.candidates[2].id, None),
        ]
        assert payload["unpicked"] == 0
        assert env.client.get(f"/api/voices/design/{picked.id}/audio").status_code == 200

    def test_the_transcript_is_the_line_of_the_candidates_own_batch(self, env):
        older = _write_batch([4.0], batch_id="older", text="The line of the older batch, read aloud.")
        _write_batch([5.0], batch_id="newer", text="The line of the newer batch, read aloud.")

        env.client.post(f"/api/voices/design/{older.candidates[0].id}/save", json={"name": "Anna"})

        (saved,) = list_voices()
        assert (saved.seconds, saved.text) == (4.0, "The line of the older batch, read aloud.")

    def test_saving_a_candidate_a_second_time_is_409_and_makes_no_voice(self, env):
        (candidate,) = _write_batch([4.0]).candidates
        assert env.client.post(f"/api/voices/design/{candidate.id}/save", json={"name": "Anna"}).status_code == 200

        response = env.client.post(f"/api/voices/design/{candidate.id}/save", json={"name": "Bea"})

        assert response.status_code == 409
        assert len(list_voices()) == 1

    def test_a_saved_candidate_shows_its_voices_current_name(self, env):
        (candidate,) = _write_batch([4.0]).candidates
        voice = env.client.post(f"/api/voices/design/{candidate.id}/save", json={"name": "Anna"}).json()

        assert env.client.post(f"/api/voices/{voice['slug']}/rename", json={"name": "Anna Lund"}).status_code == 200

        assert [item["saved_as"] for item in _candidates(env)] == ["Anna Lund"]

    def test_deleting_the_saved_voice_makes_its_candidate_unpicked_and_savable_again(self, env):
        batch = _write_batch([4.0, 5.0])
        voice = env.client.post(f"/api/voices/design/{batch.candidates[0].id}/save", json={"name": "Anna"}).json()

        assert env.client.delete(f"/api/voices/{voice['slug']}").status_code == 200

        assert _design_payload(env)["unpicked"] == 0
        assert [item["saved_as"] for item in _candidates(env)] == [None, None]
        assert env.client.get(f"/api/voices/design/{batch.candidates[0].id}/audio").status_code == 200
        again = env.client.post(f"/api/voices/design/{batch.candidates[0].id}/save", json={"name": "Anna"})
        assert again.status_code == 200
        assert [item["saved_as"] for item in _candidates(env)] == ["Anna", None]

    @pytest.mark.parametrize("candidate_id", ["nope", "0", "0123456789abcdef"])
    def test_an_unknown_id_is_404_and_saves_nothing(self, env, candidate_id):
        _write_batch([4.0])

        response = env.client.post(f"/api/voices/design/{candidate_id}/save", json={"name": "Anna"})

        assert response.status_code == 404
        assert list_voices() == []

    def test_a_cleared_candidate_is_404_for_a_stale_page(self, env):
        (candidate,) = _write_batch([4.0]).candidates
        env.client.delete("/api/voices/design/candidates")
        _write_batch([4.0], batch_id="fresh")

        response = env.client.post(f"/api/voices/design/{candidate.id}/save", json={"name": "Anna"})

        assert response.status_code == 404
        assert list_voices() == []

    @pytest.mark.parametrize("seconds", [2.0, 31.0])
    def test_a_candidate_outside_the_limits_is_422(self, env, seconds):
        (candidate,) = _write_batch([seconds]).candidates

        response = env.client.post(f"/api/voices/design/{candidate.id}/save", json={"name": "Anna"})

        assert response.status_code == 422
        assert list_voices() == []

    def test_a_candidate_of_an_unavailable_model_is_422(self, env):
        (candidate,) = _write_batch([4.0], model="gone").candidates

        response = env.client.post(f"/api/voices/design/{candidate.id}/save", json={"name": "Anna"})

        assert response.status_code == 422

    def test_a_taken_name_is_409_and_a_blank_name_422_and_the_candidate_stays_unpicked(self, env):
        first, second = _write_batch([4.0, 5.0]).candidates
        assert env.client.post(f"/api/voices/design/{first.id}/save", json={"name": "Anna"}).status_code == 200

        assert env.client.post(f"/api/voices/design/{second.id}/save", json={"name": "anna"}).status_code == 409
        assert env.client.post(f"/api/voices/design/{second.id}/save", json={"name": " "}).status_code == 422
        assert len(list_voices()) == 1
        assert [item["saved_as"] for item in _candidates(env)] == ["Anna", None]


class TestClear:
    def test_clearing_removes_every_batch_and_file_and_keeps_the_saved_voices(self, env):
        batch = _write_batch([4.0, 5.0])
        _write_batch([4.0], batch_id="two")
        env.client.post(f"/api/voices/design/{batch.candidates[0].id}/save", json={"name": "Anna"})

        response = env.client.delete("/api/voices/design/candidates")

        assert response.status_code == 200 and response.json() == {"status": "ok"}
        assert _design_payload(env) == {"batches": [], "unpicked": 0}
        assert list(candidates_dir().glob("*")) == []
        assert [voice.name for voice in list_voices()] == ["Anna"]

    def test_clearing_nothing_is_fine(self, env):
        assert env.client.delete("/api/voices/design/candidates").status_code == 200

    def test_a_voice_named_design_is_still_deleted_by_its_slug(self, env):
        voice = env.client.post(f"/api/voices/design/{_write_batch([4.0]).candidates[0].id}/save", json={"name": "Design"}).json()

        assert env.client.delete(f"/api/voices/{voice['slug']}").status_code == 200
        assert list_voices() == []

    def test_clearing_is_refused_while_a_design_job_runs_and_works_after_it(self, env):
        _write_batch([4.0])
        gate = threading.Event()
        job_id = env.design(FakeDesigner(gate=gate), wait="")
        try:
            response = env.client.delete("/api/voices/design/candidates")
            assert response.status_code == 409
            assert len(_candidates(env)) == 1
        finally:
            gate.set()
            wait_for_status(env.runner, job_id, "completed", timeout=10)

        assert env.client.delete("/api/voices/design/candidates").status_code == 200
        assert _candidates(env) == []

    def test_clearing_is_refused_while_a_design_job_is_queued(self, env):
        gate = threading.Event()
        blocker = FakeDesigner(gate=gate)
        blocker_id = env.design(blocker, wait="")
        assert blocker.started.wait(timeout=5)
        queued_id = env.client.post("/api/voices/design", json=_BODY).json()["job_id"]
        try:
            assert env.client.delete("/api/voices/design/candidates").status_code == 409
        finally:
            gate.set()
            wait_for_status(env.runner, blocker_id, "completed", timeout=10)
            wait_for_status(env.runner, queued_id, "completed", timeout=10)


class TestEndToEnd:
    def test_plan_queue_pick_and_save_leave_the_outputs_and_the_gallery_alone(self, env, monkeypatch):
        monkeypatch.setattr(web_server, "_summarize_job_parts", lambda parts: {"parts": [part.label for part in parts]})
        out = env.root / "out"
        out.mkdir()

        assert env.client.post("/api/voices/design/plan", json=_BODY).json() == {"parts": ["Qwen3-TTS 1.7B VoiceDesign"]}
        gallery_before = env.client.get("/api/gallery").json()
        env.design(FakeDesigner(4.0, 5.0, 6.0))
        candidates = _candidates(env)
        assert [item["saveable"] for item in candidates] == [True, True, True]
        audio = env.client.get(f"/api/voices/design/{candidates[2]['id']}/audio")
        assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
        saved = env.client.post(f"/api/voices/design/{candidates[2]['id']}/save", json={"name": "Storyteller"})

        assert saved.status_code == 200 and saved.json()["origin"] == "design" and saved.json()["text"] == _LINE
        assert list(out.iterdir()) == []
        assert env.client.get("/api/gallery").json() == gallery_before
        assert env.submitted["save_project"] is None
        assert env.client.get("/api/voices").json()["voices"][0]["slug"] == saved.json()["slug"]
        assert env.log.page("all", 1, 10)["entries"] == []
        assert sorted(path.suffix for path in candidates_dir().iterdir()) == [".wav", ".wav", ".wav", ".yaml"]
        assert [path.name for path in env.root.iterdir() if path.name not in ("out", "log.sqlite3")] == []

    def test_a_stopped_job_and_a_failed_job_keep_the_earlier_batches_and_what_they_made(self, env):
        env.design(FakeDesigner(4.0))
        (first,) = _design_payload(env)["batches"]
        gate = threading.Event()
        backend = FakeDesigner(5.0, gate=gate)
        job_id = env.design(backend, wait="")
        assert backend.started.wait(timeout=5)
        env.client.post(f"/jobs/{job_id}/controls/quit")
        gate.set()
        assert wait_for_status(env.runner, job_id, "cancelled", timeout=10)["status"] == "cancelled"
        env.design(FakeDesigner(fail="boom"), wait="failed")

        batches = _design_payload(env)["batches"]

        assert [len(batch["candidates"]) for batch in batches] == [1, 3]
        assert batches[1] == first
