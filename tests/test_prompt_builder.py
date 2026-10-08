"""Tests for the prompt-builder routes: load, save, preview and create."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from zvisiongenerator.web import server as web_server
from zvisiongenerator.web.prompt_builder import file_revision

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
        assert [axis["key"] for axis in payload["enhance_matrix"]["axes"]] == ["style", "mood", "details", "length", "motion"]
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
