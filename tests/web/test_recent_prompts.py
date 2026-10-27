"""Tests for the recent prompts: entry builder, app-database store, recording on generate and the routes."""

from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest
from fastapi.testclient import TestClient

from tests.web._helpers import make_resolved_image_defaults, make_web_config, patch_image_submit_dependencies
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web.config import WebUiDefaultModels
from zvisiongenerator.web.recent_prompts import RecentPrompts, RecentPromptsError, recent_prompt_entry
from zvisiongenerator.web.web_runner import JobConflictError

ENTRY_KEYS = {"format", "prompt", "scene", "negative_prompt", "mode", "used_at"}
CAPTION = '{\n  "high_level_description": "a red fox"\n}'


def _build(**overrides) -> dict | None:
    values = dict(prompt_source=None, prompt="a lake", json_prompt=None, scene=None, negative_prompt=None, mode="image", used_at=100.0)
    values.update(overrides)
    return recent_prompt_entry(**values)


def _entry(prompt: str = "a lake", *, format: str = "text", scene: str = "", negative_prompt: str = "", mode: str = "image", used_at: float = 1.0) -> dict:
    return {"format": format, "prompt": prompt, "scene": scene, "negative_prompt": negative_prompt, "mode": mode, "used_at": used_at}


class TestRecentPromptEntry:
    def test_an_image_submit_keeps_the_negative_prompt_and_drops_the_scene(self):
        entry = _build(prompt="  a lake ", scene="a scene", negative_prompt=" blurry ")

        assert entry == _entry("a lake", negative_prompt="blurry", used_at=100.0)
        assert set(entry) == ENTRY_KEYS

    def test_a_video_submit_keeps_the_scene_and_stores_no_negative_prompt(self):
        entry = _build(mode="video", prompt="she waves", scene=" a pier ", negative_prompt="blurry")

        assert entry == _entry("she waves", scene="a pier", mode="video", used_at=100.0)

    def test_a_video_submit_with_only_a_scene_is_kept(self):
        assert _build(mode="video", prompt=None, scene="a pier") == _entry("", scene="a pier", mode="video", used_at=100.0)

    def test_a_json_caption_is_stored_as_sent_with_the_json_format(self):
        entry = _build(prompt=None, json_prompt=f" {CAPTION} ", negative_prompt="blurry", scene="ignored")

        assert entry == _entry(CAPTION, format="json", negative_prompt="blurry", used_at=100.0)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"prompt_source": "file", "json_prompt": CAPTION},
            {"prompt": "  "},
            {"mode": "video", "prompt": None, "scene": None},
        ],
    )
    def test_file_and_blank_submits_are_not_remembered(self, overrides):
        assert _build(**overrides) is None


@pytest.fixture()
def store(tmp_path) -> RecentPrompts:
    return RecentPrompts(lambda: tmp_path / "ziv.sqlite3")


def _prompts(store: RecentPrompts, query: str = "", mode: str = "image") -> list[str]:
    return [entry["prompt"] for entry in store.search(query, mode)]


class TestRecentPromptsStore:
    def test_a_repeat_is_one_row_that_becomes_the_newest_with_the_new_time(self, store):
        store.record(_entry("a", used_at=1.0))
        store.record(_entry("b", used_at=2.0))
        store.record(_entry("a", used_at=3.0))

        entries = store.search("", "image")

        assert [entry["prompt"] for entry in entries] == ["a", "b"]
        assert entries[0]["used_at"] == 3.0

    @pytest.mark.parametrize("other", [{"format": "json"}, {"negative_prompt": "blurry"}, {"scene": "a pier"}, {"mode": "video"}])
    def test_the_same_prompt_with_another_format_scene_negative_or_mode_is_another_row(self, store, other):
        store.record(_entry("a"))
        store.record(_entry("a", **other))

        assert len(store.search("", "image")) + len(store.search("", "video")) == 2

    def test_there_is_no_cap(self, store):
        for index in range(120):
            store.record(_entry(f"p{index}", used_at=float(index)))

        assert store.clear("image") == 120

    def test_search_returns_only_the_newest_matches_up_to_the_limit(self, store):
        for index in range(60):
            store.record(_entry(f"fox {index}", used_at=float(index)))
        store.record(_entry("a cat", used_at=100.0))

        assert _prompts(store, "fox")[:2] == ["fox 59", "fox 58"]
        assert len(store.search("fox", "image")) == 50
        assert len(store.search("", "image")) == 50


class TestNewerSchema:
    def test_a_newer_schema_is_refused_and_left_untouched(self, tmp_path):
        db = tmp_path / "ziv.sqlite3"
        with closing(sqlite3.connect(db)) as conn:
            conn.execute("CREATE TABLE schema_versions (component TEXT PRIMARY KEY, version INTEGER NOT NULL) WITHOUT ROWID")
            conn.execute("INSERT INTO schema_versions VALUES ('recent_prompts', 2)")
            conn.commit()
        store = RecentPrompts(lambda: db)

        for call in (lambda: store.record(_entry()), lambda: store.search("", "image"), lambda: store.clear("image")):
            with pytest.raises(RecentPromptsError):
                call()

        with closing(sqlite3.connect(db)) as conn:
            assert conn.execute("SELECT version FROM schema_versions WHERE component = 'recent_prompts'").fetchone() == (2,)


class _FailingStore:
    def record(self, entry):
        raise RecentPromptsError("disk full")

    def search(self, query, mode):
        raise RecentPromptsError("disk full")

    def clear(self, mode):
        raise RecentPromptsError("disk full")


def _stored(mode: str = "image") -> list[dict]:
    return web_server.recent_prompts.search("", mode)


@pytest.fixture()
def image_submit(monkeypatch, tmp_path) -> list[dict[str, object]]:
    """Make ``POST /api/generate`` accept image submits on a JSON-capable model without loading anything."""
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path)
    web_config.default_models = WebUiDefaultModels(image="ideo", video="ltx-8")
    submitted: list[dict[str, object]] = []
    patch_image_submit_dependencies(
        monkeypatch,
        model_info=ImageModelInfo(family="ideogram4", is_distilled=False, size=None),
        defaults=make_resolved_image_defaults(supports_json_prompt=True),
        submitted=submitted,
    )
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    return submitted


def _post_image(**data):
    with TestClient(web_server.app) as client:
        return client.post("/api/generate", data={"mode": "image", "model": "ideo", "workflow": "txt2img", **data})


class TestRecordOnGenerate:
    def test_an_accepted_image_submit_is_recorded_as_sent(self, image_submit, monkeypatch):
        monkeypatch.setattr(web_server.time, "time", lambda: 1234.5)

        response = _post_image(prompt="  a red fox ", negative_prompt="blurry", scene="ignored")

        assert response.status_code == 200
        assert _stored() == [_entry("a red fox", negative_prompt="blurry", used_at=1234.5)]

    def test_the_prompt_field_is_recorded_not_the_users_original_text(self, image_submit):
        _post_image(prompt="an enhanced fox in golden light", prompt_input="a fox")

        assert [entry["prompt"] for entry in _stored()] == ["an enhanced fox in golden light"]

    @pytest.mark.parametrize("error", [ValueError("bad size"), JobConflictError("busy")])
    def test_a_refused_submit_records_nothing(self, image_submit, monkeypatch, error):
        def refuse(_form, _web_config):
            raise error

        monkeypatch.setattr(web_server, "_submit_image_job", refuse)

        response = _post_image(prompt="a red fox")

        assert response.status_code == (409 if isinstance(error, JobConflictError) else 422)
        assert _stored() == []

    def test_a_recording_failure_warns_and_the_submit_still_succeeds(self, image_submit, monkeypatch):
        monkeypatch.setattr(web_server, "recent_prompts", _FailingStore())

        with pytest.warns(UserWarning):
            response = _post_image(prompt="a red fox")

        assert response.status_code == 200
        assert response.json()["job_id"] == "job-123"


class TestRecentPromptsRoutes:
    def test_listing_returns_the_entries_key_newest_first(self):
        web_server.recent_prompts.record(_entry("a", used_at=1.0))
        web_server.recent_prompts.record(_entry("b", format="json", used_at=2.0))

        with TestClient(web_server.app) as client:
            response = client.get("/api/recent-prompts", params={"mode": "image"})

        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"entries"}
        assert [entry["prompt"] for entry in body["entries"]] == ["b", "a"]
        assert set(body["entries"][0]) == ENTRY_KEYS

    def test_the_list_holds_only_the_asked_mode(self):
        web_server.recent_prompts.record(_entry("a still"))
        web_server.recent_prompts.record(_entry("she waves", scene="a pier", mode="video"))

        with TestClient(web_server.app) as client:
            image = client.get("/api/recent-prompts", params={"mode": "image"}).json()["entries"]
            video = client.get("/api/recent-prompts", params={"mode": "video"}).json()["entries"]

        assert [entry["prompt"] for entry in image] == ["a still"]
        assert [(entry["prompt"], entry["scene"]) for entry in video] == [("she waves", "a pier")]

    @pytest.mark.parametrize("params", [{}, {"mode": "audio"}])
    def test_a_missing_or_unknown_mode_is_a_422(self, params):
        with TestClient(web_server.app) as client:
            assert client.get("/api/recent-prompts", params=params).status_code == 422
            assert client.delete("/api/recent-prompts", params=params).status_code == 422

    def test_q_filters_on_prompt_and_scene_ignoring_case(self):
        web_server.recent_prompts.record(_entry("a red fox"))
        web_server.recent_prompts.record(_entry("a cat"))
        web_server.recent_prompts.record(_entry("walks", scene="Snowy Hill", mode="video"))

        with TestClient(web_server.app) as client:
            by_prompt = client.get("/api/recent-prompts", params={"mode": "image", "q": "FOX"}).json()["entries"]
            by_scene = client.get("/api/recent-prompts", params={"mode": "video", "q": "snowy"}).json()["entries"]
            blank = client.get("/api/recent-prompts", params={"mode": "image", "q": "  "}).json()["entries"]

        assert [entry["prompt"] for entry in by_prompt] == ["a red fox"]
        assert [entry["prompt"] for entry in by_scene] == ["walks"]
        assert len(blank) == 2

    def test_delete_clears_only_the_asked_mode_and_returns_the_count(self):
        web_server.recent_prompts.record(_entry("a"))
        web_server.recent_prompts.record(_entry("b"))
        web_server.recent_prompts.record(_entry("c", mode="video"))

        with TestClient(web_server.app) as client:
            deleted = client.delete("/api/recent-prompts", params={"mode": "image"})
            image = client.get("/api/recent-prompts", params={"mode": "image"})
            video = client.get("/api/recent-prompts", params={"mode": "video"})

        assert deleted.json() == {"deleted_count": 2}
        assert image.json() == {"entries": []}
        assert [entry["prompt"] for entry in video.json()["entries"]] == ["c"]

    @pytest.mark.parametrize(
        "call",
        [lambda client: client.get("/api/recent-prompts", params={"mode": "image"}), lambda client: client.delete("/api/recent-prompts", params={"mode": "image"})],
    )
    def test_a_storage_error_is_a_500_with_the_reason(self, monkeypatch, call):
        monkeypatch.setattr(web_server, "recent_prompts", _FailingStore())

        with TestClient(web_server.app) as client:
            response = call(client)

        assert response.status_code == 500
        assert "disk full" in response.json()["detail"]
