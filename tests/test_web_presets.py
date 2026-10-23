"""Test the preset store and the /api/presets routes."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest
import yaml

from zvisiongenerator.web import presets as store
from zvisiongenerator.web import server as web_server


@pytest.fixture
def client():
    with TestClient(web_server.app) as test_client:
        yield test_client


def _body(name: str = "Soft portrait", mode: str = "image", model: str | None = "my-image-model", **settings) -> dict:
    return {"name": name, "mode": mode, "model": model, "settings": settings}


def _files() -> list[str]:
    """Return the preset files as ``<mode>/<slug>.yaml``."""
    directory = store._presets_dir()
    return sorted(path.relative_to(directory).as_posix() for path in directory.glob("*/*.yaml")) if directory.is_dir() else []


def test_an_empty_store_lists_nothing(client):
    response = client.get("/api/presets")

    assert response.status_code == 200
    assert response.json() == {"presets": []}


def test_create_update_rename_delete_round_trip(client):
    created = client.post("/api/presets", json=_body(steps=9, guidance=0, ratio="2:3"))
    updated = client.put("/api/presets/image/soft-portrait", json=_body(steps=12))
    renamed = client.post("/api/presets/image/soft-portrait/rename", json={"name": "Hard portrait"})
    after_rename = client.get("/api/presets").json()["presets"]
    files_after_rename = _files()
    deleted = client.delete("/api/presets/image/hard-portrait")

    assert created.status_code == 200
    preset = created.json()
    assert (preset["slug"], preset["name"], preset["mode"], preset["model"]) == ("soft-portrait", "Soft portrait", "image", "my-image-model")
    assert (preset["settings"]["guidance"], preset["settings"]["ratio"], preset["settings"]["quantize"]) == (0.0, "2:3", None)
    assert updated.json()["settings"]["steps"] == 12
    assert (renamed.json()["slug"], renamed.json()["name"], renamed.json()["settings"]["steps"]) == ("hard-portrait", "Hard portrait", 12)
    assert [item["slug"] for item in after_rename] == ["hard-portrait"]
    assert files_after_rename == ["image/hard-portrait.yaml"]
    assert deleted.json() == {"status": "ok"}
    assert client.get("/api/presets").json() == {"presets": []}


def test_the_list_is_sorted_by_name_across_modes(client):
    client.post("/api/presets", json=_body(name="beta"))
    client.post("/api/presets", json=_body(name="Alpha", mode="video", model=None))
    client.post("/api/presets", json=_body(name="Charlie"))

    assert [item["name"] for item in client.get("/api/presets").json()["presets"]] == ["Alpha", "beta", "Charlie"]


def test_create_with_a_taken_slug_is_409_within_a_mode(client):
    client.post("/api/presets", json=_body(name="Clip", mode="video", model=None))

    response = client.post("/api/presets", json=_body(name="clip", mode="video", model=None))

    assert response.status_code == 409
    assert _files() == ["video/clip.yaml"]


def test_the_same_name_can_exist_in_both_modes(client):
    image = client.post("/api/presets", json=_body(name="Clip"))
    video = client.post("/api/presets", json=_body(name="Clip", mode="video", model=None))
    renamed = client.post("/api/presets/video/clip/rename", json={"name": "Other"})
    deleted = client.delete("/api/presets/image/clip")

    assert (image.status_code, video.status_code, renamed.status_code, deleted.status_code) == (200, 200, 200, 200)
    assert _files() == ["video/other.yaml"]


def test_the_folder_decides_the_mode_and_a_body_mode_is_ignored_on_update(client):
    client.post("/api/presets", json=_body(name="Clip", mode="video", model=None))

    updated = client.put("/api/presets/video/clip", json=_body(name="Clip", mode="image", model=None, steps=7))

    assert updated.status_code == 200
    assert (updated.json()["mode"], updated.json()["settings"]["steps"]) == ("video", 7)
    assert "mode" not in yaml.safe_load((store._presets_dir() / "video" / "clip.yaml").read_text(encoding="utf-8"))
    assert _files() == ["video/clip.yaml"]


def test_image_strength_and_runs_are_stored_and_runs_is_stored_for_video(client):
    image = client.post("/api/presets", json=_body(name="Img", image_strength=0.7, runs=3)).json()["settings"]
    video = client.post("/api/presets", json=_body(name="Vid", mode="video", runs=2, image_strength=0.7)).json()["settings"]

    assert (image["image_strength"], image["runs"]) == (0.7, 3)
    assert video["runs"] == 2
    assert "image_strength" not in video


def test_unknown_keys_are_ignored(client):
    response = client.post("/api/presets", json={**_body(steps=5, bogus=1, frames=49), "extra": 1})

    assert response.status_code == 200
    assert response.json()["settings"]["steps"] == 5
    assert "bogus" not in response.json()["settings"]
    assert "frames" not in response.json()["settings"]


@pytest.mark.parametrize("settings", [{"steps": True}, {"steps": "9"}, {"guidance": "x"}, {"review": 1}])
def test_a_setting_of_the_wrong_type_is_422(client, settings):
    assert client.post("/api/presets", json=_body(**settings)).status_code == 422
    assert _files() == []


@pytest.mark.parametrize(
    "body",
    [
        _body(name=""),
        _body(name="***"),
        {"name": "Soft portrait", "mode": "audio", "model": None, "settings": {}},
        {"name": "Soft portrait", "mode": "image", "model": 5, "settings": {}},
        {"name": "Soft portrait", "mode": "image", "model": None, "settings": 5},
    ],
)
def test_an_invalid_body_is_422(client, body):
    assert client.post("/api/presets", json=body).status_code == 422


def test_update_needs_an_existing_preset_and_the_same_slug(client):
    client.post("/api/presets", json=_body())

    assert client.put("/api/presets/image/missing", json=_body(name="Missing")).status_code == 404
    assert client.put("/api/presets/video/soft-portrait", json=_body()).status_code == 404
    assert client.put("/api/presets/image/soft-portrait", json=_body(name="Other name")).status_code == 422
    assert client.put("/api/presets/image/soft-portrait", json=_body(name="SOFT Portrait")).status_code == 200


@pytest.mark.parametrize("slug", ["Bad", "-x", "a--b", "a_b"])
def test_a_malformed_path_slug_is_422(client, slug):
    assert client.put(f"/api/presets/image/{slug}", json=_body()).status_code == 422
    assert client.post(f"/api/presets/image/{slug}/rename", json={"name": "New"}).status_code == 422
    assert client.delete(f"/api/presets/image/{slug}").status_code == 422


def test_an_invalid_path_mode_is_422(client):
    assert client.put("/api/presets/audio/clip", json=_body()).status_code == 422
    assert client.post("/api/presets/audio/clip/rename", json={"name": "New"}).status_code == 422
    assert client.delete("/api/presets/audio/clip").status_code == 422


def test_rename_to_an_existing_preset_is_409_and_keeps_both(client):
    client.post("/api/presets", json=_body(name="One"))
    client.post("/api/presets", json=_body(name="Two"))

    assert client.post("/api/presets/image/one/rename", json={"name": "two"}).status_code == 409
    assert _files() == ["image/one.yaml", "image/two.yaml"]


def test_rename_and_delete_of_a_missing_preset_are_404(client):
    assert client.post("/api/presets/image/missing/rename", json={"name": "New"}).status_code == 404
    assert client.delete("/api/presets/video/missing").status_code == 404


def test_a_rename_that_keeps_the_slug_rewrites_the_file_in_place(client):
    client.post("/api/presets", json=_body())

    response = client.post("/api/presets/image/soft-portrait/rename", json={"name": "SOFT Portrait"})

    assert (response.json()["slug"], response.json()["name"]) == ("soft-portrait", "SOFT Portrait")
    assert _files() == ["image/soft-portrait.yaml"]


def test_an_unwritable_store_is_a_500(client, monkeypatch):
    def _fail(*_args, **_kwargs):
        raise PermissionError("read-only disk")

    monkeypatch.setattr(store, "write_text_atomic", _fail)

    assert client.post("/api/presets", json=_body()).status_code == 500


def test_invalid_files_are_left_out_of_the_list(client):
    client.post("/api/presets", json=_body(name="Good"))
    (store._presets_dir() / "image" / "broken.yaml").write_text("name: [", encoding="utf-8")
    (store._presets_dir() / "video" / "wrong.yaml").parent.mkdir(exist_ok=True)
    (store._presets_dir() / "video" / "wrong.yaml").write_text("name: x\nsettings: 5\n", encoding="utf-8")

    assert [item["slug"] for item in client.get("/api/presets").json()["presets"]] == ["good"]


def test_writes_from_another_origin_are_refused(client):
    response = client.post("/api/presets", json=_body(), headers={"Origin": "https://evil.example"})

    assert response.status_code == 403
    assert _files() == []


@pytest.mark.parametrize(
    ("name", "slug"),
    [("Soft Look", "soft-look"), ("  Crème brûlée!  ", "creme-brulee"), ("A  B--C", "a-b-c"), ("x" * 100, "x" * 64)],
)
def test_slugify(name, slug):
    assert store._slugify(name) == slug
