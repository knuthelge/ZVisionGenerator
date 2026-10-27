"""Tests for the in-app path browser's folder listing."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from zvisiongenerator.web import path_browser
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web.path_browser import browse_places, create_folder, default_place_dir, list_directory, resolve_purpose


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "styles").mkdir()
    (tmp_path / "Characters").mkdir()
    (tmp_path / ".cache").mkdir()
    (tmp_path / "ink.safetensors").write_bytes(b"12345")
    (tmp_path / "Grain.SAFETENSORS").write_bytes(b"1")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    (tmp_path / ".hidden.safetensors").write_bytes(b"1")
    return tmp_path


def _names(listing) -> list[str]:
    return [entry.name for entry in listing.entries]


def test_lists_folders_first_and_only_files_the_purpose_accepts(tree):
    listing = list_directory(resolve_purpose("lora_file"), str(tree), default_dir=tree)

    assert _names(listing) == ["Characters", "styles", "Grain.SAFETENSORS", "ink.safetensors"]
    assert listing.hidden_count == 1
    assert listing.path == str(tree)
    assert listing.parent == str(tree.parent)
    ink = next(entry for entry in listing.entries if entry.name == "ink.safetensors")
    assert (ink.kind, ink.size, ink.selectable) == ("file", 5, True)
    assert all(not entry.selectable for entry in listing.entries if entry.kind == "directory")


def test_show_all_lists_other_files_as_not_selectable_and_show_hidden_adds_dotfiles(tree):
    listing = list_directory(resolve_purpose("lora_file"), str(tree), default_dir=tree, show_hidden=True, show_all=True)

    assert listing.hidden_count == 0
    assert {".cache", ".hidden.safetensors", "notes.txt"} <= set(_names(listing))
    assert next(entry for entry in listing.entries if entry.name == "notes.txt").selectable is False


def test_folder_purpose_selects_folders_and_hides_files(tree):
    listing = list_directory(resolve_purpose("output_directory"), str(tree), default_dir=tree)

    assert _names(listing) == ["Characters", "styles"]
    assert listing.hidden_count == 3
    assert all(entry.selectable for entry in listing.entries)


def test_a_file_path_lists_its_folder_and_focuses_the_file(tree):
    listing = list_directory(resolve_purpose("lora_file"), str(tree / "ink.safetensors"), default_dir=tree)

    assert listing.path == str(tree)
    assert listing.focus == "ink.safetensors"


def test_an_empty_path_opens_the_default_folder_or_home_when_it_is_missing(tree, monkeypatch, tmp_path_factory):
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setattr(path_browser.Path, "home", lambda: home)

    assert list_directory(resolve_purpose("lora_file"), "", default_dir=tree).path == str(tree)
    assert list_directory(resolve_purpose("lora_file"), " ", default_dir=tree / "missing").path == str(home)


def test_a_linked_file_opens_the_folder_that_holds_the_link(tree, tmp_path_factory):
    blobs = tmp_path_factory.mktemp("blobs")
    (blobs / "abc123").write_bytes(b"1")
    snapshot = tree / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").symlink_to(blobs / "abc123")

    listing = list_directory(resolve_purpose("checkpoint_file"), str(snapshot / "model.safetensors"))

    assert listing.path == str(snapshot)
    assert listing.focus == "model.safetensors"
    assert [entry.name for entry in listing.entries if entry.selectable] == ["model.safetensors"]


def test_default_place_dir_follows_the_purpose(tmp_path):
    folders = {"home": tmp_path / "h", "output_root": tmp_path / "o", "data_dir": tmp_path / "z"}

    assert default_place_dir(resolve_purpose("lora_file"), **folders) == tmp_path / "z" / "loras"
    assert default_place_dir(resolve_purpose("output_directory"), **folders) == tmp_path / "o"
    assert default_place_dir(resolve_purpose("prompt_file"), **folders) == tmp_path / "h"


def test_the_filesystem_root_has_no_parent():
    root = Path(Path.cwd().anchor)

    assert list_directory(resolve_purpose("output_directory"), str(root), default_dir=root).parent is None


@pytest.mark.parametrize(("path_text", "error"), [("relative/dir", ValueError), ("/definitely/not/here/zvision", FileNotFoundError)])
def test_rejects_relative_and_missing_paths(tree, path_text, error):
    with pytest.raises(error):
        list_directory(resolve_purpose("lora_file"), path_text, default_dir=tree)


def test_unknown_purpose_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        resolve_purpose("unknown")


def test_places_flag_the_purpose_default_and_add_drives(tmp_path):
    places = browse_places(resolve_purpose("lora_file"), home=tmp_path, output_root=tmp_path / "out", data_dir=tmp_path / "ziv", drives=("C:\\", "D:\\"))

    assert [place.id for place in places] == ["home", "output", "models", "loras", "drive:C:\\", "drive:D:\\"]
    assert [place.id for place in places if place.default] == ["loras"]
    assert places[3].path == str(tmp_path / "ziv" / "loras")
    assert places[4].label == "C:"


@pytest.mark.parametrize(
    ("purpose", "default"),
    [("output_directory", "output"), ("source_video", "output"), ("reference_image", "output"), ("checkpoint_file", "models"), ("prompt_file", "home"), ("prompt_file_folder", "home")],
)
def test_each_purpose_opens_at_its_own_place(tmp_path, purpose, default):
    places = browse_places(resolve_purpose(purpose), home=tmp_path, output_root=tmp_path, data_dir=tmp_path)

    assert [place.id for place in places if place.default] == [default]


def test_create_folder_makes_one_plain_folder(tmp_path):
    created = create_folder(resolve_purpose("output_directory"), str(tmp_path), " renders ")

    assert created == tmp_path / "renders"
    assert created.is_dir()
    with pytest.raises(FileExistsError):
        create_folder(resolve_purpose("output_directory"), str(tmp_path), "renders")


@pytest.mark.parametrize("name", ["", "..", "a/b", "a\\b"])
def test_create_folder_rejects_names_that_are_not_one_folder(tmp_path, name):
    with pytest.raises(ValueError):
        create_folder(resolve_purpose("output_directory"), str(tmp_path), name)


def test_create_folder_is_only_for_folder_purposes(tmp_path):
    with pytest.raises(ValueError):
        create_folder(resolve_purpose("lora_file"), str(tmp_path), "new")


# ── Routes ───────────────────────────────────────────────────────────────────


@pytest.fixture
def client(tree, monkeypatch):
    monkeypatch.setattr(web_server, "_media_output_root", lambda: tree)
    monkeypatch.setattr(web_server, "get_ziv_data_dir", lambda: tree)
    monkeypatch.setattr(web_server, "host_drives", lambda: ())
    with TestClient(web_server.app) as test_client:
        yield test_client


def test_browse_route_returns_the_listing_contract(client, tree):
    response = client.get("/api/browse", params={"purpose": "lora_file", "path": str(tree)})

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"path", "parent", "separator", "focus", "entries", "hidden_count"}
    assert set(payload["entries"][0]) == {"name", "path", "kind", "size", "modified", "selectable"}
    assert payload["hidden_count"] == 1


def test_browse_route_passes_hidden_and_all(client, tree):
    payload = client.get("/api/browse", params={"purpose": "lora_file", "path": str(tree), "hidden": "true", "all": "true"}).json()

    assert payload["hidden_count"] == 0
    assert ".cache" in [entry["name"] for entry in payload["entries"]]


def test_browse_route_opens_an_empty_path_at_the_purpose_place(client, tree):
    payload = client.get("/api/browse", params={"purpose": "output_directory"}).json()

    assert payload["path"] == str(tree)


@pytest.mark.parametrize(
    ("params", "status"),
    [({"purpose": "unknown"}, 422), ({"purpose": "lora_file", "path": "/definitely/not/here/zvision"}, 404), ({"purpose": "lora_file", "path": "relative"}, 422)],
)
def test_browse_route_errors(client, params, status):
    response = client.get("/api/browse", params=params)

    assert response.status_code == status
    assert set(response.json()) == {"detail"}


def test_browse_route_reports_permission_denied_as_403(client, tree, monkeypatch):
    def _denied(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied", str(tree))

    monkeypatch.setattr(web_server, "list_directory", _denied)

    assert client.get("/api/browse", params={"purpose": "lora_file", "path": str(tree)}).status_code == 403


def test_places_route_lists_places(client):
    payload = client.get("/api/browse/places", params={"purpose": "checkpoint_file"}).json()

    assert [place["id"] for place in payload["places"]] == ["home", "output", "models", "loras"]
    assert [place["id"] for place in payload["places"] if place["default"]] == ["models"]
    assert payload["purpose"] == {"name": "checkpoint_file", "kind": "existing_file", "label": "Checkpoint file", "accepted_extensions": [".safetensors"]}


def test_folder_route_creates_and_reports_conflicts(client, tree):
    body = {"purpose": "output_directory", "path": str(tree), "name": "renders"}

    created = client.post("/api/browse/folder", json=body)
    again = client.post("/api/browse/folder", json=body)
    for_files = client.post("/api/browse/folder", json={**body, "purpose": "lora_file", "name": "other"})

    assert created.status_code == 200
    assert created.json() == {"path": str(tree / "renders")}
    assert again.status_code == 409
    assert for_files.status_code == 422
