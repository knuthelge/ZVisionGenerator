"""Tests for the prompt-builder routes: load, save, preview, create and append."""

from __future__ import annotations

import threading
import time
from typing import Any

from fastapi.testclient import TestClient
import pytest

from zvisiongenerator.web import prompt_builder
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web.prompt_builder import PromptFileChangedError, append_entry_payload, file_revision, save_document_payload

SAMPLE = """\
snippets:
  light: soft light

# Portraits
portrait:
  - prompt: a woman, $light
  - prompt: a man
    active: false
"""


def _write(tmp_path, text: str = SAMPLE):
    path = tmp_path / "prompts.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _load(client: TestClient, path) -> dict[str, Any]:
    response = client.post("/api/prompt-files/document", json={"path": str(path)})
    assert response.status_code == 200
    return response.json()


def _save(client: TestClient, path, loaded: dict[str, Any], document: dict[str, Any], **extra: Any):
    return client.put("/api/prompt-files/document", json={"path": str(path), "revision": loaded["revision"], "document": document, **extra})


class TestLoadRoute:
    def test_returns_document_revision_and_text(self, tmp_path):
        path = _write(tmp_path)

        with TestClient(web_server.app) as client:
            payload = _load(client, path)

        assert payload["revision"] == file_revision(SAMPLE.encode())
        assert payload["raw_text"] == SAMPLE
        assert "problem" not in payload
        assert [axis["key"] for axis in payload["enhance_matrix"]["axes"]] == ["style", "mood", "theme", "details", "camera", "pace", "motion", "sound", "length"]
        document = payload["document"]
        assert document["snippets"] == [{"id": "n0", "name": "light", "value": {"kind": "text", "text": "soft light"}}]
        assert [entry["id"] for entry in document["sets"][0]["entries"]] == ["s0.e0", "s0.e1"]
        assert document["sets"][0]["entries"][1]["active"] is False

    def test_invalid_structure_returns_problem_for_repair(self, tmp_path):
        path = _write(tmp_path, "portrait: [\n")

        with TestClient(web_server.app) as client:
            payload = _load(client, path)

        assert "document" not in payload
        assert payload["raw_text"] == "portrait: [\n"
        assert isinstance(payload["problem"], str) and payload["problem"]

    def test_rejects_wrong_extension(self, tmp_path):
        path = tmp_path / "prompts.txt"
        path.write_text(SAMPLE, encoding="utf-8")

        with TestClient(web_server.app) as client:
            response = client.post("/api/prompt-files/document", json={"path": str(path)})

        assert response.status_code == 422


class TestSaveRoute:
    def test_saves_and_returns_new_revision_options_and_id_maps(self, tmp_path):
        path = _write(tmp_path)

        with TestClient(web_server.app) as client:
            loaded = _load(client, path)
            document = loaded["document"]
            portrait = document["sets"][0]
            # Move the active entry after the inactive one and rename the set.
            portrait["entries"].reverse()
            portrait["name"] = "people"
            response = _save(client, path, loaded, document)

        assert response.status_code == 200
        payload = response.json()
        written = path.read_text(encoding="utf-8")
        assert payload["raw_text"] == written
        assert payload["revision"] == file_revision(written.encode())
        assert "# Portraits\npeople:" in written
        assert payload["option_id_map"] == {"portrait:0": "people:1"}
        assert payload["ids"]["s0.e0"] == "s0.e1"
        assert payload["warnings"] == []

    def test_stale_revision_is_a_conflict_and_does_not_write(self, tmp_path):
        path = _write(tmp_path)

        with TestClient(web_server.app) as client:
            loaded = _load(client, path)
            path.write_text(SAMPLE + "\nother:\n  - prompt: x\n", encoding="utf-8")
            response = _save(client, path, loaded, loaded["document"])

        assert response.status_code == 409
        assert "other:" in path.read_text(encoding="utf-8")

    def test_force_overwrites_onto_the_loaded_text(self, tmp_path):
        path = _write(tmp_path)

        with TestClient(web_server.app) as client:
            loaded = _load(client, path)
            path.write_text(SAMPLE + "\nother:\n  - prompt: x\n", encoding="utf-8")
            response = _save(client, path, loaded, loaded["document"], force=True, base_text=loaded["raw_text"])

        assert response.status_code == 200
        assert path.read_text(encoding="utf-8") == SAMPLE

    def test_force_without_base_text_is_rejected(self, tmp_path):
        path = _write(tmp_path)

        with TestClient(web_server.app) as client:
            loaded = _load(client, path)
            response = _save(client, path, loaded, loaded["document"], force=True)

        assert response.status_code == 422

    def test_errors_block_the_save(self, tmp_path):
        path = _write(tmp_path)

        with TestClient(web_server.app) as client:
            loaded = _load(client, path)
            document = loaded["document"]
            document["sets"][0]["entries"][0]["prompt"] = {"kind": "text", "text": "a $missing"}
            response = _save(client, path, loaded, document)

        assert response.status_code == 422
        assert path.read_text(encoding="utf-8") == SAMPLE

    def test_malformed_document_is_rejected(self, tmp_path):
        path = _write(tmp_path)

        with TestClient(web_server.app) as client:
            loaded = _load(client, path)
            response = _save(client, path, loaded, {"sets": [{"id": "s0", "name": "x", "entries": [{"id": "e", "prompt": {"kind": "nope"}}]}]})

        assert response.status_code == 422

    def test_duplicate_ids_are_rejected(self, tmp_path):
        path = _write(tmp_path)

        with TestClient(web_server.app) as client:
            loaded = _load(client, path)
            document = loaded["document"]
            entries = document["sets"][0]["entries"]
            entries[1]["id"] = entries[0]["id"]
            save = _save(client, path, loaded, document)
            preview = client.post("/api/prompt-files/preview", json={"document": document})

        assert save.status_code == 422
        assert preview.status_code == 422
        assert path.read_text(encoding="utf-8") == SAMPLE

    def test_enhance_warnings_are_returned(self, tmp_path):
        path = _write(tmp_path)

        with TestClient(web_server.app) as client:
            loaded = _load(client, path)
            document = loaded["document"]
            document["sets"][0]["entries"][0]["enhance"] = {"style": "nope"}
            response = _save(client, path, loaded, document)

        assert response.status_code == 200
        assert len(response.json()["warnings"]) == 1


class TestPreviewRoute:
    def test_returns_previews_problems_uses_and_roll(self, tmp_path):
        path = _write(tmp_path)

        with TestClient(web_server.app) as client:
            document = _load(client, path)["document"]
            document["sets"][0]["entries"][1]["prompt"] = {"kind": "text", "text": "a {cat|cat} and $nothing"}
            response = client.post("/api/prompt-files/preview", json={"document": document, "roll_entry_id": "s0.e0"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["entries"]["s0.e0"] == {"prompt": "a woman, soft light", "negative": None}
        assert "s0.e1" not in payload["entries"]
        assert [(problem["target"], problem["severity"]) for problem in payload["problems"]] == [("s0.e1", "warning")]
        assert payload["snippet_uses"] == {"n0": 1}
        assert payload["rolled"] == {"entry_id": "s0.e0", "prompt": "a woman, soft light"}


class TestCreateRoute:
    def test_creates_an_empty_prompt_file(self, tmp_path):
        with TestClient(web_server.app) as client:
            response = client.post("/api/prompt-files/create", json={"directory": str(tmp_path), "name": "new"})
            created = response.json()["path"]
            loaded = _load(client, created)

        assert created == str((tmp_path / "new.yaml").resolve())
        assert (tmp_path / "new.yaml").read_text(encoding="utf-8") == ""
        assert loaded["document"] == {"snippets": [], "sets": []}

    def test_refuses_existing_file_missing_folder_and_separators(self, tmp_path):
        _write(tmp_path)

        with TestClient(web_server.app) as client:
            existing = client.post("/api/prompt-files/create", json={"directory": str(tmp_path), "name": "prompts.yaml"})
            missing = client.post("/api/prompt-files/create", json={"directory": str(tmp_path / "nope"), "name": "a"})
            nested = client.post("/api/prompt-files/create", json={"directory": str(tmp_path), "name": "a/b"})

        assert [existing.status_code, missing.status_code, nested.status_code] == [422, 422, 422]


SCENE_SAMPLE = """\
snippets:
  castle: A castle at dusk.

knights:
  - prompt: A knight rides in.
    scene: $castle
  - prompt: A dog barks.
"""


class TestScene:
    def test_load_returns_scene_or_null_per_entry(self, tmp_path):
        path = _write(tmp_path, SCENE_SAMPLE)

        with TestClient(web_server.app) as client:
            entries = _load(client, path)["document"]["sets"][0]["entries"]

        assert entries[0]["scene"] == {"kind": "text", "text": "$castle"}
        assert entries[1]["scene"] is None

    def test_save_writes_scene_before_prompt_and_reloads_it(self, tmp_path):
        path = _write(tmp_path, SCENE_SAMPLE)
        scene = {"kind": "text", "text": "A barn at dawn."}

        with TestClient(web_server.app) as client:
            loaded = _load(client, path)
            document = loaded["document"]
            document["sets"][0]["entries"][1]["scene"] = scene
            response = _save(client, path, loaded, document)
            reloaded = _load(client, path)

        assert response.status_code == 200
        assert "  - scene: A barn at dawn.\n    prompt: A dog barks.\n" in path.read_text(encoding="utf-8")
        assert reloaded["document"]["sets"][0]["entries"][1]["scene"] == scene
        assert response.json()["document"]["sets"][0]["entries"][1]["scene"] == scene

    def test_payload_without_a_scene_key_means_no_scene(self, tmp_path):
        path = _write(tmp_path, SCENE_SAMPLE)

        with TestClient(web_server.app) as client:
            loaded = _load(client, path)
            document = loaded["document"]
            for entry in document["sets"][0]["entries"]:
                entry.pop("scene")
            response = _save(client, path, loaded, document)

        assert response.status_code == 200
        assert "scene:" not in path.read_text(encoding="utf-8")

    def test_malformed_scene_payload_is_rejected(self, tmp_path):
        path = _write(tmp_path, SCENE_SAMPLE)

        with TestClient(web_server.app) as client:
            loaded = _load(client, path)
            document = loaded["document"]
            document["sets"][0]["entries"][0]["scene"] = {"kind": "nope"}
            response = _save(client, path, loaded, document)

        assert response.status_code == 422

    def test_undefined_snippet_in_scene_blocks_the_save(self, tmp_path):
        path = _write(tmp_path, SCENE_SAMPLE)

        with TestClient(web_server.app) as client:
            loaded = _load(client, path)
            document = loaded["document"]
            document["sets"][0]["entries"][0]["scene"] = {"kind": "text", "text": "$missing"}
            response = _save(client, path, loaded, document)

        assert response.status_code == 422
        assert path.read_text(encoding="utf-8") == SCENE_SAMPLE

    def test_preview_carries_scene_only_when_present_and_counts_snippet_use(self, tmp_path):
        path = _write(tmp_path, SCENE_SAMPLE)

        with TestClient(web_server.app) as client:
            document = _load(client, path)["document"]
            response = client.post("/api/prompt-files/preview", json={"document": document})

        payload = response.json()
        assert payload["entries"]["s0.e0"] == {"prompt": "A knight rides in.", "negative": None, "scene": "A castle at dusk."}
        assert payload["entries"]["s0.e1"] == {"prompt": "A dog barks.", "negative": None}
        assert payload["snippet_uses"] == {"n0": 1}
        assert payload["problems"] == []


_EXTENSIONS = (".yaml", ".yml")


def _append(client: TestClient, path, **fields: Any):
    return client.post("/api/prompt-files/append", json={"path": str(path), **fields})


def _slow(monkeypatch, name: str, delay: float) -> None:
    """Make ``prompt_builder.<name>`` sleep *delay* seconds before it runs, to widen a race window."""
    original = getattr(prompt_builder, name)

    def slowed(*args: Any, **kwargs: Any) -> Any:
        time.sleep(delay)
        return original(*args, **kwargs)

    monkeypatch.setattr(prompt_builder, name, slowed)


class TestAppendRoute:
    def test_appends_to_a_new_prompts_set_and_keeps_the_rest(self, tmp_path):
        path = _write(tmp_path)

        with TestClient(web_server.app) as client:
            response = _append(client, path, prompt="a harbour at night", scene="the quay", negative="blurry")
            loaded = _load(client, path)

        body = response.json()
        assert response.status_code == 200
        assert set(body) == {"path", "revision", "entry_id", "set_name"}
        assert body["path"] == str(path.resolve())
        assert body["set_name"] == "prompts"
        assert body["entry_id"] == "s1.e0"
        assert body["revision"] == file_revision(path.read_bytes()) == loaded["revision"]
        last = loaded["document"]["sets"][-1]
        assert last["name"] == "prompts"
        assert [entry["id"] for entry in last["entries"]] == ["s1.e0"]
        assert last["entries"][0]["prompt"] == {"kind": "text", "text": "a harbour at night"}
        assert last["entries"][0]["scene"] == {"kind": "text", "text": "the quay"}
        text = path.read_text(encoding="utf-8")
        assert "# Portraits" in text
        assert "light: soft light" in text

    def test_a_second_append_lands_after_the_first(self, tmp_path):
        path = _write(tmp_path)

        with TestClient(web_server.app) as client:
            first = _append(client, path, prompt="one")
            second = _append(client, path, prompt="two")
            loaded = _load(client, path)

        assert first.json()["entry_id"] == "s1.e0"
        assert second.json()["entry_id"] == "s1.e1"
        texts = [entry["prompt"]["text"] for entry in loaded["document"]["sets"][-1]["entries"]]
        assert texts == ["one", "two"]

    def test_appends_to_a_file_made_by_the_create_route(self, tmp_path):
        with TestClient(web_server.app) as client:
            created = client.post("/api/prompt-files/create", json={"directory": str(tmp_path), "name": "fresh"}).json()["path"]
            response = _append(client, created, prompt="first words")
            loaded = _load(client, created)

        assert response.status_code == 200
        assert response.json()["entry_id"] == "s0.e0"
        assert [doc_set["name"] for doc_set in loaded["document"]["sets"]] == ["prompts"]
        assert len(loaded["document"]["sets"][0]["entries"]) == 1

    def test_no_active_or_enhance_key_is_written(self, tmp_path):
        path = _write(tmp_path, "")

        with TestClient(web_server.app) as client:
            _append(client, path, prompt="plain", negative="  ")

        assert path.read_text(encoding="utf-8") == "prompts:\n  - prompt: plain\n"

    @pytest.mark.parametrize(
        "fields",
        [
            {"prompt": ""},
            {"prompt": "   "},
            {},
            {"prompt": 5},
            {"prompt": "ok", "scene": []},
            {"prompt": "ok", "negative": 3},
            {"prompt": "uses $bill"},
        ],
    )
    def test_bad_input_is_a_422_and_leaves_the_file_unchanged(self, tmp_path, fields):
        path = _write(tmp_path)
        before = path.read_bytes()

        with TestClient(web_server.app) as client:
            response = _append(client, path, **fields)

        assert response.status_code == 422
        assert path.read_bytes() == before
        assert response.json()["detail"]

    def test_missing_path_nonexistent_file_and_wrong_extension_are_422(self, tmp_path):
        text_file = tmp_path / "notes.txt"
        text_file.write_text("keep", encoding="utf-8")

        with TestClient(web_server.app) as client:
            no_path = client.post("/api/prompt-files/append", json={"prompt": "x"})
            missing = _append(client, tmp_path / "nope.yaml", prompt="x")
            wrong_extension = _append(client, text_file, prompt="x")

        assert [no_path.status_code, missing.status_code, wrong_extension.status_code] == [422, 422, 422]
        assert not (tmp_path / "nope.yaml").exists()
        assert text_file.read_text(encoding="utf-8") == "keep"

    def test_a_file_that_is_not_yaml_is_a_422_and_unchanged(self, tmp_path):
        path = _write(tmp_path, "a: [\n")

        with TestClient(web_server.app) as client:
            response = _append(client, path, prompt="x")

        assert response.status_code == 422
        assert path.read_text(encoding="utf-8") == "a: [\n"

    def test_prompts_that_is_not_a_list_is_a_422_and_unchanged(self, tmp_path):
        path = _write(tmp_path, "prompts:\n")

        with TestClient(web_server.app) as client:
            response = _append(client, path, prompt="x")

        assert response.status_code == 422
        assert response.json()["detail"]
        assert path.read_text(encoding="utf-8") == "prompts:\n"

    def test_an_error_elsewhere_in_the_file_blocks_the_append(self, tmp_path):
        path = _write(tmp_path, "portrait:\n  - prompt: ''\n")
        before = path.read_bytes()

        with TestClient(web_server.app) as client:
            response = _append(client, path, prompt="fine")

        assert response.status_code == 422
        assert response.json()["detail"]
        assert path.read_bytes() == before


class TestAppendLocking:
    def test_two_concurrent_appends_both_land(self, tmp_path, monkeypatch):
        path = _write(tmp_path)
        _slow(monkeypatch, "load_prompt_document", 0.05)
        threads = [threading.Thread(target=append_entry_payload, args=(str(path), text), kwargs={"accepted_extensions": _EXTENSIONS}) for text in ("one", "two")]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        text = path.read_text(encoding="utf-8")
        assert "prompt: one" in text
        assert "prompt: two" in text

    def test_a_save_racing_an_append_does_not_lose_the_append(self, tmp_path, monkeypatch):
        path = _write(tmp_path)
        with TestClient(web_server.app) as client:
            loaded = _load(client, path)
        document = loaded["document"]
        document["sets"][0]["entries"][0]["prompt"]["text"] = "edited"
        outcome: dict[str, PromptFileChangedError] = {}

        def save() -> None:
            try:
                save_document_payload(str(path), loaded["revision"], document, accepted_extensions=_EXTENSIONS)
            except PromptFileChangedError as exc:
                outcome["save"] = exc

        _slow(monkeypatch, "_document_from_payload", 0.1)  # only the save calls it
        saver = threading.Thread(target=save)
        appender = threading.Thread(target=append_entry_payload, args=(str(path), "added"), kwargs={"accepted_extensions": _EXTENSIONS})
        saver.start()
        time.sleep(0.01)
        appender.start()
        saver.join()
        appender.join()

        text = path.read_text(encoding="utf-8")
        assert ("prompt: edited" in text and "prompt: added" in text) or isinstance(outcome.get("save"), PromptFileChangedError)
