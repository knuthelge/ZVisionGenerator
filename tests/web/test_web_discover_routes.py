"""Tests for the Discover LoRAs routes: the curated list, the Hugging Face proxy, downloads, triggers and deletes."""

from __future__ import annotations

import dataclasses
import hashlib
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pytest
import yaml

from tests.web._helpers import COMMIT, SHA, FakeHub, model_json, safetensors_bytes
from zvisiongenerator.downloads.ensure import READY
from zvisiongenerator.downloads.flights import DownloadFlights
from zvisiongenerator.utils.curated_loras import curated_for_family
from zvisiongenerator.web import config_contract as config_contract_module
from zvisiongenerator.web import hub_client
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web.download_lane import DownloadLane
from zvisiongenerator.web.lora_details import describe_installed
from zvisiongenerator.web.lora_download import LoraInstaller
from zvisiongenerator.web.lora_download import LoraSource
from zvisiongenerator.web.lora_sources import SOURCES, LoraSourceRow

_PAYLOAD = b"lora-bytes" * 20
_PAYLOAD_SHA = hashlib.sha256(_PAYLOAD).hexdigest()


class _Ensure:
    """A scripted download that writes each target's file, after the test releases it when ``hold`` is set."""

    def __init__(self) -> None:
        self.calls: list = []
        self.release = threading.Event()
        self.release.set()

    def __call__(self, targets, *, flights, report, cancelled, **_):
        self.calls.append(tuple(targets))
        assert self.release.wait(5.0)
        for target in targets:
            path = Path(target.local_dir, target.allow_patterns[0])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_PAYLOAD)
        return READY


@pytest.fixture
def env(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "loras").mkdir(parents=True)
    hub = FakeHub()
    ensure = _Ensure()
    lane = DownloadLane(flights=DownloadFlights(), ensure=ensure)
    previews: list[str] = []

    def preview(url: str):
        previews.append(url)
        return "image/webp", b"IMG"

    monkeypatch.setattr(config_contract_module, "get_ziv_data_dir", lambda: data)
    monkeypatch.setattr(web_server, "get_ziv_data_dir", lambda: data)
    monkeypatch.setattr(web_server, "hub_fetch", hub)
    monkeypatch.setattr(web_server, "download_lane", lane)
    monkeypatch.setattr(web_server, "lora_installer", LoraInstaller(lane, loras_dir=lambda: data / "loras", sources=SOURCES, preview=preview, background=lambda run: run()))
    monkeypatch.setattr(web_server, "web_runner", MagicMock(**{"get_active_exclusive_job_snapshot.return_value": None}))
    with TestClient(web_server.app) as client:
        yield SimpleNamespace(client=client, hub=hub, ensure=ensure, lane=lane, data=data, loras=data / "loras", previews=previews)
    ensure.release.set()
    lane.shutdown(join_timeout=5.0)


@pytest.fixture
def hf_on(env):
    env.client.post("/api/discover/settings", json={"hf_enabled": True})
    return env


def _wait_state(env, item_id: str, state: str) -> dict:
    for _ in range(500):
        item = next((entry for entry in env.lane.snapshot()["items"] if entry["id"] == item_id), None)
        if item is not None and item["state"] == state:
            return item
        threading.Event().wait(0.01)
    raise AssertionError(f"{item_id} never reached {state}: {env.lane.snapshot()}")


def _zimage_entry():
    return curated_for_family("zimage")[0]


def _hf_enabled(env) -> bool:
    return env.client.get("/api/discover/curated", params={"family": "zimage"}).json()["hf_enabled"]


class TestSettings:
    def test_hugging_face_is_off_until_the_user_chooses_to_show_it(self, env):
        assert _hf_enabled(env) is False

        assert env.client.post("/api/discover/settings", json={"hf_enabled": True}).json() == {"hf_enabled": True}

        assert _hf_enabled(env) is True
        assert yaml.safe_load((env.data / "config.yaml").read_text(encoding="utf-8")) == {"ui": {"discover_huggingface": True}}
        assert env.client.post("/api/discover/settings", json={"hf_enabled": False}).json() == {"hf_enabled": False}
        assert _hf_enabled(env) is False
        assert yaml.safe_load((env.data / "config.yaml").read_text(encoding="utf-8")) == {"ui": {"discover_huggingface": False}}

    @pytest.mark.parametrize("body", [{}, {"hf_enabled": "yes"}, [True]])
    def test_the_setting_must_be_a_boolean(self, env, body):
        assert env.client.post("/api/discover/settings", json=body).status_code == 422

    @pytest.mark.parametrize(
        "path",
        [
            "/api/discover/hf?family=zimage",
            "/api/discover/hf/repo?repo=o/r",
            "/api/discover/hf/check?repo=o/r&file=a.safetensors&family=zimage",
        ],
    )
    def test_every_hugging_face_read_is_refused_until_then_and_the_hub_is_not_asked(self, env, path):
        response = env.client.get(path)

        assert response.status_code == 403
        assert env.hub.calls == []

    def test_a_download_of_a_repository_file_is_refused_until_then_but_a_curated_one_is_not(self, env):
        refused = env.client.post("/api/discover/download", json={"repo": "o/r", "file": "a.safetensors"})
        allowed = env.client.post("/api/discover/download", json={"curated_id": _zimage_entry().id})

        assert refused.status_code == 403
        assert allowed.status_code == 200


class TestCurated:
    def test_lists_the_entries_of_the_model_with_their_contract_keys(self, env):
        body = env.client.get("/api/discover/curated", params={"family": "zimage"}).json()

        assert {"family", "size", "version", "hf_enabled", "entries"} <= set(body)
        assert [entry["id"] for entry in body["entries"]] == [entry.id for entry in curated_for_family("zimage")]
        first = body["entries"][0]
        assert {
            "id",
            "name",
            "display_name",
            "author",
            "description",
            "section",
            "repo",
            "revision",
            "file",
            "size_bytes",
            "size_label",
            "trigger",
            "weight",
            "licence",
            "licence_label",
            "non_commercial",
            "previews",
            "model_card_url",
            "state",
            "download",
        } <= set(first)
        assert first["state"] == "available"
        assert "thumbnail_url" not in first

    def test_only_the_entries_for_the_asked_size_or_version_are_listed(self, env):
        nine = env.client.get("/api/discover/curated", params={"family": "flux2_klein", "size": "9b"}).json()["entries"]
        ltx25 = env.client.get("/api/discover/curated", params={"family": "ltx", "version": "2.5"}).json()["entries"]

        assert nine and {entry["size_variant"] for entry in nine} == {"9b"}
        assert ltx25 and {entry["version"] for entry in ltx25} == {"2.5"}
        assert env.client.get("/api/discover/curated", params={"family": "unknown_family"}).json()["entries"] == []

    def test_an_entry_shows_installed_when_its_file_has_its_name_and_size(self, env):
        entry = _zimage_entry()
        with open(env.loras / f"{entry.name}.safetensors", "wb") as handle:
            handle.truncate(entry.size_bytes)

        shown = {item["id"]: item for item in env.client.get("/api/discover/curated", params={"family": "zimage"}).json()["entries"]}

        assert shown[entry.id]["state"] == "installed"

    def test_an_entry_whose_name_holds_another_file_is_name_taken(self, env):
        entry = _zimage_entry()
        (env.loras / f"{entry.name}.safetensors").write_bytes(b"mine")

        shown = {item["id"]: item for item in env.client.get("/api/discover/curated", params={"family": "zimage"}).json()["entries"]}

        assert shown[entry.id]["state"] == "name_taken"

    def test_an_entry_downloading_shows_its_lane_item(self, env):
        env.ensure.release.clear()
        entry = _zimage_entry()
        started = env.client.post("/api/discover/download", json={"curated_id": entry.id}).json()

        shown = {item["id"]: item for item in env.client.get("/api/discover/curated", params={"family": "zimage"}).json()["entries"]}

        assert shown[entry.id]["state"] == "downloading"
        assert shown[entry.id]["download"]["id"] == started["item"]["id"]
        env.ensure.release.set()


class TestHubProxy:
    def test_lists_a_page_through_the_server(self, hf_on):
        listing = {"id": "o/a", "downloads": 9, "likes": 1, "gated": False, "tags": [], "siblings": [{"rfilename": "a.safetensors"}], "cardData": {"license": "mit"}}
        curated = _zimage_entry()
        hf_on.hub.listing["Tongyi-MAI/Z-Image-Turbo"] = ([listing, {**listing, "id": curated.repo}], "NEXT")

        body = hf_on.client.get("/api/discover/hf", params={"family": "zimage", "q": "pixel", "sort": "likes"}).json()

        assert [item["repo"] for item in body["items"]] == ["o/a", curated.repo]
        assert [item["curated_id"] is not None for item in body["items"]] == [False, True]
        assert body["next_cursor"]
        assert "search=pixel" in hf_on.hub.urls[0]
        again = hf_on.client.get("/api/discover/hf", params={"family": "zimage", "cursor": body["next_cursor"]})
        assert again.status_code == 200
        assert "cursor=NEXT" in hf_on.hub.urls[1]

    def test_an_ltx_model_lists_only_the_adapters_of_its_version(self, hf_on):
        hf_on.hub.listing["Lightricks/LTX-2.5"] = ([], None)

        body = hf_on.client.get("/api/discover/hf", params={"family": "ltx", "version": "2.5"}).json()

        assert body["base_models"] == ["Lightricks/LTX-2.5"]

    @pytest.mark.parametrize("params", [{"family": "nothing"}, {"family": "zimage", "sort": "random"}, {"family": "zimage", "cursor": "garbage"}])
    def test_bad_arguments_are_422(self, hf_on, params):
        assert hf_on.client.get("/api/discover/hf", params=params).status_code == 422

    def test_the_default_reader_is_the_anonymous_bounded_one(self):
        assert isinstance(web_server.hub_fetch, hub_client.CachedFetch)
        assert web_server.hub_fetch._fetch is hub_client.http_fetch

    def test_a_hub_that_cannot_be_reached_is_reported_without_breaking_the_app(self, hf_on, monkeypatch):
        def down(url, **_):
            raise hub_client.HubError("Could not reach Hugging Face.")

        monkeypatch.setattr(web_server, "hub_fetch", down)

        response = hf_on.client.get("/api/discover/hf", params={"family": "zimage"})

        assert response.status_code == 503

    @pytest.mark.parametrize("repo", ["../x", "o/../x", "o/r/extra", "https://evil.example/a/b", "o"])
    def test_a_repo_that_is_not_an_id_is_refused_without_asking_the_hub(self, hf_on, repo):
        for path in ("/api/discover/hf/repo", "/api/discover/hf/check"):
            response = hf_on.client.get(path, params={"repo": repo, "file": "a.safetensors", "family": "zimage"})
            assert response.status_code == 422, path

        assert hf_on.hub.calls == []

    @pytest.mark.parametrize("file", ["../a.safetensors", "/a.safetensors", "\\a.safetensors", "a.bin", "x/../../a.safetensors"])
    def test_a_file_that_is_not_a_safetensors_path_is_refused_without_asking_the_hub(self, hf_on, file):
        response = hf_on.client.get("/api/discover/hf/check", params={"repo": "o/r", "file": file, "family": "zimage"})

        assert response.status_code == 422
        assert hf_on.hub.calls == []

    def test_repo_detail_gives_the_files_the_preselected_one_and_the_folded_snapshots(self, hf_on):
        files = {"s_000001000.safetensors": 10, "s_000002000_(recommended).safetensors": 20, "s_000003000.safetensors": 30}
        hf_on.hub.models["o/Sunbleached-Z"] = model_json("o/Sunbleached-Z", files=files)

        body = hf_on.client.get("/api/discover/hf/repo", params={"repo": "o/Sunbleached-Z"}).json()

        assert body["repo"] == "o/Sunbleached-Z"
        assert body["revision"] == COMMIT
        assert body["preselected"] == "s_000002000_(recommended).safetensors"
        assert [item["file"] for item in body["files"]] == ["s_000002000_(recommended).safetensors"]
        assert [item["file"] for item in body["other_files"]] == ["s_000001000.safetensors", "s_000003000.safetensors"]
        assert (body["licence"], body["base_model"], body["instance_prompt"], body["model_card_url"]) == (
            "apache-2.0",
            "Tongyi-MAI/Z-Image-Turbo",
            "trg, more",
            "https://huggingface.co/o/Sunbleached-Z",
        )
        chosen = body["files"][0]
        assert (chosen["size_bytes"], chosen["sha256"], chosen["save_as"], chosen["renamed"], chosen["state"]) == (20, SHA, "sunbleached_z", False, "available")

    def test_repo_detail_reports_a_name_clash_before_anything_downloads(self, hf_on):
        hf_on.hub.models["o/Style"] = model_json("o/Style")
        (hf_on.loras / "style.safetensors").write_bytes(b"mine")

        chosen = hf_on.client.get("/api/discover/hf/repo", params={"repo": "o/Style"}).json()["files"][0]

        assert (chosen["save_as"], chosen["renamed"], chosen["state"]) == ("style-2", True, "available")
        assert hf_on.ensure.calls == []

    def test_repo_detail_names_the_installed_copy_and_says_a_replaced_file_is_new(self, hf_on):
        hf_on.hub.models["o/Style"] = model_json("o/Style", files={"a.safetensors": 20})
        (hf_on.loras / "style.safetensors").write_bytes(b"other repository")
        (hf_on.loras / "style-2.safetensors").write_bytes(b"x" * 20)
        SOURCES.record(LoraSourceRow(name="style-2", size_bytes=20, repo="o/Style", revision="b" * 40, file="a.safetensors", sha256=SHA))

        installed = hf_on.client.get("/api/discover/hf/repo", params={"repo": "o/Style"}).json()["files"][0]
        SOURCES.record(LoraSourceRow(name="style-2", size_bytes=20, repo="o/Style", revision="b" * 40, file="a.safetensors", sha256="c" * 64))
        replaced = hf_on.client.get("/api/discover/hf/repo", params={"repo": "o/Style"}).json()["files"][0]

        assert (installed["save_as"], installed["renamed"], installed["state"]) == ("style-2", False, "installed")
        assert (replaced["save_as"], replaced["renamed"], replaced["state"]) == ("style-3", True, "available")

    def test_a_file_of_a_curated_entry_carries_its_id(self, hf_on):
        entry = _zimage_entry()
        hf_on.hub.models[entry.repo] = model_json(entry.repo, files={entry.file: entry.size_bytes})

        body = hf_on.client.get("/api/discover/hf/repo", params={"repo": entry.repo}).json()

        assert body["files"][0]["curated_id"] == entry.id

    def test_gated_and_adult_repositories_are_refused_and_a_missing_one_is_404(self, hf_on):
        hf_on.hub.models["o/gated"] = model_json("o/gated", gated="auto")
        hf_on.hub.models["o/adult"] = model_json("o/adult", tags=["not-for-all-audiences"])

        assert hf_on.client.get("/api/discover/hf/repo", params={"repo": "o/gated"}).status_code == 403
        assert hf_on.client.get("/api/discover/hf/repo", params={"repo": "o/adult"}).status_code == 403
        assert hf_on.client.get("/api/discover/hf/repo", params={"repo": "o/none"}).status_code == 404

    def test_the_family_check_names_the_model_from_the_header(self, hf_on):
        hf_on.hub.models["o/r"] = model_json("o/r", files={"a.safetensors": 5_000_000_000})
        hf_on.hub.files[("o/r", "a.safetensors")] = safetensors_bytes(metadata={"ss_base_model_version": "krea2"}, tail=3 * 1024 * 1024)

        body = hf_on.client.get("/api/discover/hf/check", params={"repo": "o/r", "file": "a.safetensors", "family": "zimage"}).json()

        assert body == {"status": "mismatch", "family": "krea2", "size_variant": None, "expected": {"family": "zimage", "size_variant": None}}

    def test_the_family_check_only_reads_files_the_repository_lists(self, hf_on):
        hf_on.hub.models["o/r"] = model_json("o/r")

        response = hf_on.client.get("/api/discover/hf/check", params={"repo": "o/r", "file": "other.safetensors", "family": "zimage"})

        assert response.status_code == 404
        assert not any("/resolve/" in url for url in hf_on.hub.urls)


class TestDownload:
    @pytest.fixture
    def entry(self, monkeypatch):
        """A curated entry whose pinned size and SHA-256 are those of the scripted download."""
        entry = dataclasses.replace(_zimage_entry(), size_bytes=len(_PAYLOAD), sha256=_PAYLOAD_SHA)
        monkeypatch.setattr(web_server, "curated_by_id", lambda curated_id: entry if curated_id == entry.id else None)
        return entry

    def test_a_curated_entry_downloads_installs_and_records_its_source(self, env, entry):
        started = env.client.post("/api/discover/download", json={"curated_id": entry.id}).json()
        _wait_state(env, started["item"]["id"], "done")

        assert (started["name"], started["status"], started["renamed"], started["trigger"], started["trigger_source"]) == (entry.name, "queued", False, entry.trigger, "curated")
        assert started["item"]["kind"] == "lora"
        assert (env.loras / f"{entry.name}.safetensors").read_bytes() == _PAYLOAD
        row = SOURCES.rows([entry.name])[entry.name]
        assert (row.curated_id, row.sha256, row.revision) == (entry.id, _PAYLOAD_SHA, entry.revision)

    def test_deleting_the_file_forgets_its_finished_download(self, env, entry, monkeypatch):
        monkeypatch.setattr(web_server, "load_web_config", lambda: SimpleNamespace(data_dir=str(env.data)))
        started = env.client.post("/api/discover/download", json={"curated_id": entry.id}).json()
        _wait_state(env, started["item"]["id"], "done")

        assert env.client.delete(f"/api/loras/{entry.name}").status_code == 200

        assert [item for item in env.lane.snapshot()["items"] if item["kind"] == "lora"] == []

    def test_each_file_of_a_repository_has_its_own_name_and_state_while_another_downloads_and_once_it_is_installed(self, hf_on):
        files = {"a.safetensors": len(_PAYLOAD), "b.safetensors": len(_PAYLOAD)}
        hf_on.hub.models["o/Style"] = model_json("o/Style", files=files)
        for sibling in hf_on.hub.models["o/Style"]["siblings"][1:]:
            sibling["lfs"]["sha256"] = _PAYLOAD_SHA

        def shown() -> dict[str, tuple[str, str]]:
            body = hf_on.client.get("/api/discover/hf/repo", params={"repo": "o/Style"}).json()
            return {item["file"]: (item["save_as"], item["state"]) for item in [*body["files"], *body["other_files"]]}

        assert shown() == {"a.safetensors": ("style", "available"), "b.safetensors": ("style", "available")}
        hf_on.ensure.release.clear()
        started = hf_on.client.post("/api/discover/download", json={"repo": "o/Style", "file": "a.safetensors", "name": "style"}).json()

        assert shown() == {"a.safetensors": ("style", "downloading"), "b.safetensors": ("style-2", "available")}
        hf_on.ensure.release.set()
        _wait_state(hf_on, started["item"]["id"], "done")
        assert shown() == {"a.safetensors": ("style", "installed"), "b.safetensors": ("style-2", "available")}

    def test_a_curated_name_held_by_another_file_is_a_409_and_nothing_downloads(self, env, entry):
        (env.loras / f"{entry.name}.safetensors").write_bytes(b"mine")

        response = env.client.post("/api/discover/download", json={"curated_id": entry.id})

        assert response.status_code == 409
        assert env.ensure.calls == []

    def test_an_unknown_curated_id_is_404(self, env):
        assert env.client.post("/api/discover/download", json={"curated_id": "nope"}).status_code == 404

    @pytest.mark.parametrize("body", [[], {}, {"repo": "o/r"}, {"file": "a.safetensors"}])
    def test_a_request_must_name_a_curated_id_or_a_repo_and_file(self, hf_on, body):
        assert hf_on.client.post("/api/discover/download", json=body).status_code == 422

    def test_a_repository_file_downloads_under_a_slug_of_the_repo_name_with_the_cards_trigger(self, hf_on):
        hf_on.hub.models["o/Pixel-Art-Style"] = model_json("o/Pixel-Art-Style", files={"pytorch_lora_weights.safetensors": len(_PAYLOAD)})
        hf_on.hub.models["o/Pixel-Art-Style"]["siblings"][1]["lfs"]["sha256"] = _PAYLOAD_SHA

        started = hf_on.client.post("/api/discover/download", json={"repo": "o/Pixel-Art-Style", "file": "pytorch_lora_weights.safetensors"}).json()
        _wait_state(hf_on, started["item"]["id"], "done")

        assert (started["name"], started["renamed"], started["trigger"], started["trigger_source"]) == ("pixel_art_style", False, "trg, more", "card")
        assert (hf_on.loras / "pixel_art_style.safetensors").read_bytes() == _PAYLOAD
        (target,) = hf_on.ensure.calls[0]
        assert (target.repo_id, target.revision, target.anonymous, target.local_dir is not None) == ("o/Pixel-Art-Style", COMMIT, True, True)
        row = SOURCES.rows(["pixel_art_style"])["pixel_art_style"]
        assert (row.repo, row.revision, row.sha256, row.auto_trigger, row.trigger_source) == ("o/Pixel-Art-Style", COMMIT, _PAYLOAD_SHA, "trg, more", "card")

    def test_a_repository_download_keeps_the_cards_facts_and_its_first_preview_for_the_installed_tab(self, hf_on):
        card = {"license": "apache-2.0", "instance_prompt": "trg", "widget": [{"output": {"url": "images/a.png"}}, {"output": {"url": "images/b.png"}}]}
        hf_on.hub.models["o/Pixel-Art-Style"] = model_json("o/Pixel-Art-Style", files={"w.safetensors": len(_PAYLOAD)}, cardData=card, downloads=1234)
        hf_on.hub.models["o/Pixel-Art-Style"]["siblings"][1]["lfs"]["sha256"] = _PAYLOAD_SHA

        started = hf_on.client.post("/api/discover/download", json={"repo": "o/Pixel-Art-Style", "file": "w.safetensors"}).json()
        _wait_state(hf_on, started["item"]["id"], "done")

        row = SOURCES.rows(["pixel_art_style"])["pixel_art_style"]
        assert (row.author, row.downloads, row.licence, row.has_image) == ("o", 1234, "apache-2.0", True)
        assert hf_on.previews == [f"https://huggingface.co/o/Pixel-Art-Style/resolve/{COMMIT}/images/a.png"]
        image = hf_on.client.get("/api/discover/installed-thumbnails/pixel_art_style")
        assert (image.status_code, image.headers["content-type"], image.content) == (200, "image/webp", b"IMG")

    def test_a_repository_without_a_preview_keeps_the_facts_and_no_image(self, hf_on):
        hf_on.hub.models["o/Plain"] = model_json("o/Plain", files={"w.safetensors": len(_PAYLOAD)})
        hf_on.hub.models["o/Plain"]["siblings"][1]["lfs"]["sha256"] = _PAYLOAD_SHA

        started = hf_on.client.post("/api/discover/download", json={"repo": "o/Plain", "file": "w.safetensors"}).json()
        _wait_state(hf_on, started["item"]["id"], "done")

        row = SOURCES.rows(["plain"])["plain"]
        assert (row.author, row.downloads, row.has_image) == ("o", 5, False)
        assert hf_on.previews == []
        assert hf_on.client.get("/api/discover/installed-thumbnails/plain").status_code == 404

    @pytest.mark.parametrize("name", ["my lora", "style #1", "100% real", "a?b&c=d", "ümlaut"])
    def test_the_address_of_an_installed_preview_serves_a_name_with_special_characters(self, env, name):
        (env.loras / f"{name}.safetensors").write_bytes(b"x" * 10)
        SOURCES.record(LoraSourceRow(name=name, size_bytes=10, repo="o/r", revision=COMMIT, file="f.safetensors", sha256=SHA, has_image=True, downloaded_at=7), ("image/webp", b"IMG"))
        described = describe_installed(env.loras, [name], SOURCES.rows([name]))[name]

        response = env.client.get(described["hub"]["thumbnail_url"])

        assert (response.status_code, response.content) == (200, b"IMG")

    @pytest.mark.parametrize("name", ["missing", "..%2Fx"])
    def test_a_name_without_a_kept_preview_has_no_thumbnail(self, env, name):
        assert env.client.get(f"/api/discover/installed-thumbnails/{name}").status_code == 404

    @pytest.mark.parametrize(
        ("body", "status"),
        [
            ({"repo": "../x", "file": "a.safetensors"}, 422),
            ({"repo": "o/r", "file": "../a.safetensors"}, 422),
            ({"repo": "o/r", "file": "missing.safetensors"}, 404),
            ({"repo": "o/none", "file": "a.safetensors"}, 404),
        ],
    )
    def test_bad_requests_are_refused_before_any_download(self, hf_on, body, status):
        hf_on.hub.models["o/r"] = model_json("o/r")

        assert hf_on.client.post("/api/discover/download", json=body).status_code == status
        assert hf_on.ensure.calls == []

    def test_a_gated_or_adult_repository_is_never_downloaded(self, hf_on):
        hf_on.hub.models["o/gated"] = model_json("o/gated", gated="manual")

        response = hf_on.client.post("/api/discover/download", json={"repo": "o/gated", "file": "lora.safetensors"})

        assert response.status_code == 422
        assert hf_on.ensure.calls == []


class TestImportOnTheLane:
    def test_the_route_returns_at_once_with_the_lane_item_and_the_file_arrives_later(self, env, monkeypatch):
        env.ensure.release.clear()
        monkeypatch.setattr(web_server, "import_source", lambda repo, filename: LoraSource("o/Imported-Lora", COMMIT, "weights/Style_V2.safetensors", len(_PAYLOAD), _PAYLOAD_SHA, anonymous=False))

        response = env.client.post("/api/models/import-lora/hf", json={"repo_id": "o/Imported-Lora"})

        body = response.json()
        assert response.status_code == 200
        assert (body["status"], body["outcome"], body["name"], body["download"]["kind"]) == ("ok", "queued", "Style_V2", "lora")
        assert not (env.loras / "Style_V2.safetensors").exists()
        env.ensure.release.set()
        _wait_state(env, body["download"]["id"], "done")
        assert (env.loras / "Style_V2.safetensors").is_file()
        assert env.ensure.calls[0][0].anonymous is False

    def test_a_typed_name_that_is_taken_is_a_409_and_an_unnamed_import_is_numbered(self, env, monkeypatch):
        monkeypatch.setattr(web_server, "import_source", lambda repo, filename: LoraSource("o/Imported-Lora", COMMIT, "weights/Style_V2.safetensors", len(_PAYLOAD), _PAYLOAD_SHA, anonymous=False))
        (env.loras / "mine.safetensors").write_bytes(b"x")
        (env.loras / "Style_V2.safetensors").write_bytes(b"x")

        taken = env.client.post("/api/models/import-lora/hf", json={"repo_id": "o/Imported-Lora", "name": "mine"})
        numbered = env.client.post("/api/models/import-lora/hf", json={"repo_id": "o/Imported-Lora"})

        assert taken.status_code == 409
        assert numbered.json()["name"] == "Style_V2-2"

    def test_a_file_stem_that_is_not_a_valid_name_is_a_400(self, env, monkeypatch):
        monkeypatch.setattr(web_server, "import_source", lambda repo, filename: LoraSource("o/Imported-Lora", COMMIT, "a..b.safetensors", len(_PAYLOAD), _PAYLOAD_SHA, anonymous=False))

        assert env.client.post("/api/models/import-lora/hf", json={"repo_id": "o/Imported-Lora"}).status_code == 400
        assert env.ensure.calls == []

    def test_a_bad_repository_is_a_400(self, env):
        assert env.client.post("/api/models/import-lora/hf", json={"repo_id": "../x"}).status_code == 400


class TestDeleteAndTriggers:
    @pytest.fixture
    def installed(self, env, monkeypatch):
        monkeypatch.setattr(web_server, "load_web_config", lambda: SimpleNamespace(data_dir=str(env.data)))
        path = env.loras / "style.safetensors"
        path.write_bytes(safetensors_bytes())
        return path

    def test_a_delete_during_a_job_is_refused_and_keeps_the_file(self, env, installed, monkeypatch):
        monkeypatch.setattr(web_server.web_runner, "get_active_exclusive_job_snapshot", lambda: {"job_id": "j"})

        response = env.client.delete("/api/loras/style")

        assert response.status_code == 409
        assert installed.exists()

    def test_a_delete_removes_the_file_and_the_record_of_its_source(self, env, installed):
        SOURCES.set_user_trigger("style", installed.stat().st_size, "ohwx")

        assert env.client.delete("/api/loras/style").status_code == 200

        assert not installed.exists()
        assert SOURCES.rows(["style"]) == {}

    def test_a_typed_trigger_is_stored_cleaned_and_wins_over_every_other_source(self, env, installed):
        response = env.client.put("/api/loras/style/trigger", json={"trigger": "  ohwx   person "})

        assert response.json() == {"name": "style", "trigger": "ohwx person", "trigger_source": "user", "user_trigger": "ohwx person"}
        cleared = env.client.put("/api/loras/style/trigger", json={"trigger": None}).json()
        assert (cleared["trigger"], cleared["trigger_source"], cleared["user_trigger"]) == (None, None, None)

    @pytest.mark.parametrize("body", [{}, {"trigger": 5}, []])
    def test_the_trigger_must_be_text_or_null(self, env, installed, body):
        assert env.client.put("/api/loras/style/trigger", json=body).status_code == 422

    @pytest.mark.parametrize("name", ["missing", "..%2Fstyle"])
    def test_a_trigger_for_a_lora_that_is_not_there_is_404(self, env, installed, name):
        assert env.client.put(f"/api/loras/{name}/trigger", json={"trigger": "x"}).status_code == 404

    def test_the_installed_listings_carry_each_loras_trigger_and_source(self, env, installed):
        from tests.web._helpers import make_web_config

        SOURCES.record(
            LoraSourceRow(name="style", size_bytes=installed.stat().st_size, repo="o/r", revision=COMMIT, file="f.safetensors", sha256=SHA, auto_trigger="t3chnic4lly", trigger_source="card")
        )
        web_config = make_web_config()
        web_config.loras_dir = str(env.loras)
        web_config.lora_options = ("style",)
        from zvisiongenerator.web import workspace_api

        row = workspace_api._lora_rows(web_config)[0]

        assert (row["trigger"], row["trigger_source"], row["source"]["repo"]) == ("t3chnic4lly", "card", "o/r")
