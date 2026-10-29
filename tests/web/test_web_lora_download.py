"""Tests for installing LoRAs through the download lane: names, staging, verification and the atomic rename."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import threading
from pathlib import Path

import pytest
from huggingface_hub.utils import filter_repo_objects

from zvisiongenerator.downloads.ensure import READY, STOPPED
from zvisiongenerator.downloads.flights import DownloadFlights
from zvisiongenerator.utils import app_db
from zvisiongenerator.utils.curated_loras import load_curated_loras
from zvisiongenerator.web.discover_hub import RepoFile
from zvisiongenerator.web.download_lane import DownloadLane
from zvisiongenerator.web.discover_api import curated_source
from zvisiongenerator.web.lora_download import HubFacts, LoraInstaller, LoraSource, import_source, staging_dir
from zvisiongenerator.web import lora_staging
from zvisiongenerator.web.lora_sources import LoraSourceRow, LoraSources

_WAIT = 5.0
_BYTES = b"lora-weights" * 10
_SHA = hashlib.sha256(_BYTES).hexdigest()
_COMMIT = "e" * 40


class _FakeEnsure:
    """A scripted ``ensure_downloaded`` that writes each target's file into its folder, optionally after a test releases it."""

    def __init__(self, content: bytes = _BYTES, outcome: str = READY, *, hold: bool = False, repo_files: list[str] | None = None) -> None:
        self.content = content
        self.repo_files = repo_files
        self.outcome = outcome
        self.calls: list = []
        self.started = threading.Event()
        self.release = threading.Event()
        if not hold:
            self.release.set()

    def __call__(self, targets, *, flights, report, cancelled, **_):
        self.calls.append(tuple(targets))
        self.started.set()
        assert self.release.wait(_WAIT)
        if self.outcome == READY:
            for target in targets:
                # Fetch what the Hub would: the repository files the allow patterns match as globs.
                files = list(filter_repo_objects(self.repo_files, allow_patterns=target.allow_patterns)) if self.repo_files else [target.allow_patterns[0]]
                for file in files:
                    path = Path(target.local_dir, file)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(self.content)
        return self.outcome


@pytest.fixture
def loras(tmp_path) -> Path:
    folder = tmp_path / "loras"
    folder.mkdir()
    return folder


@pytest.fixture
def sources() -> LoraSources:
    return LoraSources(app_db.app_db_path)


def _installer(loras: Path, sources: LoraSources, ensure: _FakeEnsure) -> tuple[LoraInstaller, DownloadLane]:
    lane = DownloadLane(flights=DownloadFlights(), ensure=ensure)
    return LoraInstaller(lane, loras_dir=lambda: loras, sources=sources), lane


def _source(**fields) -> LoraSource:
    base = LoraSource("owner/Some-Repo", _COMMIT, "weights.safetensors", len(_BYTES), _SHA, trigger="trg", trigger_source="card", anonymous=True)
    return dataclasses.replace(base, **fields)


def _entry(**fields):
    entry = dataclasses.replace(load_curated_loras()[0], id="pick-1", name="pick_one", size_bytes=len(_BYTES), sha256=_SHA, trigger="curated trigger")
    return dataclasses.replace(entry, **fields)


def _wait(lane: DownloadLane, item_id: str, state: str) -> dict:
    for _ in range(int(_WAIT / 0.01)):
        item = next((entry for entry in lane.snapshot()["items"] if entry["id"] == item_id), None)
        if item is not None and item["state"] == state:
            return item
        threading.Event().wait(0.01)
    raise AssertionError(f"{item_id} never reached {state}: {lane.snapshot()}")


def _files(loras: Path) -> list[str]:
    return sorted(path.name for path in loras.iterdir())


class TestInstall:
    def test_a_download_is_staged_outside_the_cache_verified_and_renamed_into_place_once(self, loras, sources, monkeypatch):
        ensure = _FakeEnsure()
        installer, lane = _installer(loras, sources, ensure)
        renames = []
        real = os.replace
        monkeypatch.setattr(os, "replace", lambda a, b: renames.append((Path(a).name, Path(b).name)) or real(a, b))

        result = installer.start(curated_source(_entry()), name="pick_one")
        _wait(lane, result.item["id"], "done")

        (target,) = ensure.calls[0]
        assert target.local_dir == str(staging_dir(loras, "pick_one"))
        assert (target.allow_patterns, target.revision, target.anonymous) == ((load_curated_loras()[0].file,), load_curated_loras()[0].revision, True)
        assert renames == [(load_curated_loras()[0].file, "pick_one.safetensors")]
        assert _files(loras) == ["pick_one.safetensors"]
        assert (loras / "pick_one.safetensors").read_bytes() == _BYTES

    def test_a_file_whose_name_has_glob_characters_downloads_as_itself(self, loras, sources):
        file = "[FLUX.2.Klein]Style_Redmond.safetensors"
        ensure = _FakeEnsure(repo_files=[file, "FStyle_Redmond.safetensors", "other.safetensors"])
        installer, lane = _installer(loras, sources, ensure)

        result = installer.start(curated_source(_entry(file=file)), name="pick_one")
        _wait(lane, result.item["id"], "done")

        assert _files(loras) == ["pick_one.safetensors"]

    def test_the_source_is_recorded_with_the_checked_hash_and_the_curateds_trigger(self, loras, sources):
        installer, lane = _installer(loras, sources, _FakeEnsure())

        result = installer.start(curated_source(_entry()), name="pick_one")
        _wait(lane, result.item["id"], "done")

        row = sources.rows(["pick_one"])["pick_one"]
        assert (row.curated_id, row.sha256, row.revision, row.auto_trigger, row.trigger_source, row.size_bytes) == (
            "pick-1",
            _SHA,
            load_curated_loras()[0].revision,
            "curated trigger",
            "curated",
            len(_BYTES),
        )

    def test_a_sha_mismatch_fails_the_item_and_leaves_no_file_behind(self, loras, sources):
        installer, lane = _installer(loras, sources, _FakeEnsure(b"tampered"))

        result = installer.start(_source(size_bytes=None), name="odd")
        _wait(lane, result.item["id"], "failed")

        assert _files(loras) == []
        assert sources.rows(["odd"]) == {}

    def test_a_wrong_size_fails_the_item_and_leaves_no_file_behind(self, loras, sources):
        installer, lane = _installer(loras, sources, _FakeEnsure())

        result = installer.start(_source(size_bytes=len(_BYTES) + 1), name="odd")
        _wait(lane, result.item["id"], "failed")

        assert _files(loras) == []

    def test_a_stopped_download_leaves_no_file_and_no_staging_folder(self, loras, sources):
        installer, lane = _installer(loras, sources, _FakeEnsure(outcome=STOPPED))

        result = installer.start(_source(), name="odd")
        _wait(lane, result.item["id"], "cancelled")

        assert _files(loras) == []

    def test_a_file_that_appears_under_the_name_meanwhile_is_not_replaced(self, loras, sources):
        ensure = _FakeEnsure(hold=True)
        installer, lane = _installer(loras, sources, ensure)
        result = installer.start(_source(), name="odd")
        assert ensure.started.wait(_WAIT)
        (loras / "odd.safetensors").write_bytes(b"mine")

        ensure.release.set()
        _wait(lane, result.item["id"], "failed")

        assert (loras / "odd.safetensors").read_bytes() == b"mine"
        assert _files(loras) == ["odd.safetensors"]

    def test_a_file_in_a_subfolder_of_the_repository_installs_under_the_name(self, loras, sources):
        installer, lane = _installer(loras, sources, _FakeEnsure())

        result = installer.start(_source(file="loras/v2/weights.safetensors"), name="nested")
        _wait(lane, result.item["id"], "done")

        assert _files(loras) == ["nested.safetensors"]

    def test_the_item_carries_the_name_the_source_and_the_expected_size(self, loras, sources):
        installer, _lane = _installer(loras, sources, _FakeEnsure(hold=True))

        item = installer.start(curated_source(_entry()), name="pick_one").item

        assert (item["kind"], item["name"], item["curated_id"], item["size_bytes"]) == ("lora", "pick_one", "pick-1", len(_BYTES))
        assert (item["source_repo"], item["source_file"]) == (load_curated_loras()[0].repo, load_curated_loras()[0].file)

    def test_the_item_says_whether_it_downloads_anonymously(self, loras, sources):
        installer, _lane = _installer(loras, sources, _FakeEnsure(hold=True))

        curated = installer.start(curated_source(_entry()), name="pick_one").item
        own = installer.start(_source(anonymous=False), name="mine").item

        assert (curated["anonymous"], own["anonymous"]) == (True, False)


_NEEDS_FLOCK = pytest.mark.skipif(lora_staging.fcntl is None and lora_staging.msvcrt is None, reason="needs flock or msvcrt")


class TestStaging:
    @_NEEDS_FLOCK
    def test_a_folder_left_by_a_killed_server_is_swept_but_one_a_running_download_holds_stays(self, loras, sources):
        ensure = _FakeEnsure(hold=True)
        installer, lane = _installer(loras, sources, ensure)
        left = staging_dir(loras, "left")
        left.mkdir()
        (left / ".in-use").write_bytes(b"")
        (left / "weights.safetensors").write_bytes(b"partial")
        result = installer.start(_source(), name="busy")
        assert ensure.started.wait(_WAIT)
        busy = staging_dir(loras, "busy")
        Path(busy, "weights.safetensors").write_bytes(b"partial")

        installer.sweep_staging()

        assert not left.exists()
        assert Path(busy, "weights.safetensors").is_file()
        ensure.release.set()
        _wait(lane, result.item["id"], "done")
        assert _files(loras) == ["busy.safetensors"]


class TestPreview:
    """The preview picture is optional: it is fetched after the item is done and never holds it back or fails it."""

    _FACTS = HubFacts("owner", 5, "mit", "https://huggingface.co/owner/r/resolve/a/i.png")

    def _installer(self, loras, sources, preview):
        lane = DownloadLane(flights=DownloadFlights(), ensure=_FakeEnsure())
        return LoraInstaller(lane, loras_dir=lambda: loras, sources=sources, preview=preview), lane

    def test_a_slow_image_host_does_not_hold_the_item(self, loras, sources):
        asked, release = threading.Event(), threading.Event()

        def slow(url):
            asked.set()
            assert release.wait(_WAIT)
            return "image/webp", b"IMG"

        installer, lane = self._installer(loras, sources, slow)

        result = installer.start(_source(facts=self._FACTS), name="mine")
        _wait(lane, result.item["id"], "done")

        assert asked.wait(_WAIT)
        assert sources.rows(["mine"])["mine"].has_image is False
        release.set()
        for _ in range(int(_WAIT / 0.01)):
            if sources.image("mine") is not None:
                break
            threading.Event().wait(0.01)
        assert sources.image("mine") == ("image/webp", b"IMG")
        assert sources.rows(["mine"])["mine"].has_image is True

    @pytest.mark.parametrize("failure", [RuntimeError("boom"), OSError("down")])
    def test_a_preview_that_raises_leaves_a_done_download_with_its_source_row(self, loras, sources, failure):
        done = threading.Event()

        def broken(url):
            done.set()
            raise failure

        installer, lane = self._installer(loras, sources, broken)

        result = installer.start(_source(facts=self._FACTS), name="mine")
        _wait(lane, result.item["id"], "done")

        assert done.wait(_WAIT)
        row = sources.rows(["mine"])["mine"]
        assert (row.repo, row.has_image) == ("owner/Some-Repo", False)
        assert (loras / "mine.safetensors").is_file()

    def test_a_preview_that_arrives_after_the_file_was_deleted_is_dropped(self, loras, sources):
        release = threading.Event()
        installer, lane = self._installer(loras, sources, lambda url: release.wait(_WAIT) and ("image/webp", b"IMG"))

        result = installer.start(_source(facts=self._FACTS), name="mine")
        _wait(lane, result.item["id"], "done")
        sources.forget("mine")
        release.set()
        threading.Event().wait(0.1)

        assert sources.image("mine") is None


class TestOneDownload:
    def test_two_uses_of_one_curated_entry_make_one_download(self, loras, sources):
        ensure = _FakeEnsure(hold=True)
        installer, lane = _installer(loras, sources, ensure)

        first = installer.start(curated_source(_entry()), name="pick_one")
        assert ensure.started.wait(_WAIT)
        second = installer.start(curated_source(_entry()), name="pick_one")
        ensure.release.set()
        _wait(lane, first.item["id"], "done")

        assert (first.status, second.status) == ("queued", "running")
        assert second.item["id"] == first.item["id"]
        assert len(ensure.calls) == 1

    def test_two_uses_of_one_open_file_make_one_download_under_one_name(self, loras, sources):
        ensure = _FakeEnsure(hold=True)
        installer, lane = _installer(loras, sources, ensure)

        first = installer.start(_source())
        second = installer.start(_source())
        ensure.release.set()
        _wait(lane, first.item["id"], "done")

        assert (first.name, second.name, second.status, second.renamed) == ("some_repo", "some_repo", "running", False)
        assert len(ensure.calls) == 1
        assert _files(loras) == ["some_repo.safetensors"]

    def test_simultaneous_uses_make_one_download(self, loras, sources):
        ensure = _FakeEnsure(hold=True)
        installer, lane = _installer(loras, sources, ensure)
        barrier = threading.Barrier(6)
        results = []

        def press() -> None:
            barrier.wait(_WAIT)
            results.append(installer.start(curated_source(_entry()), name="pick_one"))

        threads = [threading.Thread(target=press) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(_WAIT)
        ensure.release.set()
        _wait(lane, results[0].item["id"], "done")

        assert len({result.item["id"] for result in results}) == 1
        assert len(ensure.calls) == 1

    def test_two_files_of_one_repository_are_two_downloads_in_folders_of_their_own(self, loras, sources):
        ensure = _FakeEnsure()
        installer, lane = _installer(loras, sources, ensure)

        first = installer.start(_source(file="a.safetensors"), name="alpha")
        second = installer.start(_source(file="b.safetensors"), name="beta")
        _wait(lane, second.item["id"], "done")

        targets = [call[0] for call in ensure.calls]
        assert len({target.local_dir for target in targets}) == 2
        assert len({target.key for target in targets}) == 2
        assert first.item["id"] != second.item["id"]
        assert _files(loras) == ["alpha.safetensors", "beta.safetensors"]

    def test_an_import_and_a_discover_of_one_file_make_one_download(self, loras, sources):
        ensure = _FakeEnsure(hold=True)
        installer, lane = _installer(loras, sources, ensure)

        discover = installer.start(_source())
        imported = installer.start(_source(anonymous=False), name=None, suffix_on_clash=False)
        ensure.release.set()
        _wait(lane, discover.item["id"], "done")

        assert imported.status == "running"
        assert imported.item["id"] == discover.item["id"]

    def test_an_import_of_another_file_under_a_held_name_is_refused_not_merged(self, loras, sources):
        ensure = _FakeEnsure(hold=True)
        installer, _lane = _installer(loras, sources, ensure)
        installer.start(curated_source(_entry()), name="pick_one")

        with pytest.raises(FileExistsError):
            installer.start(_source(file="other.safetensors"), name="pick_one", suffix_on_clash=False)

        ensure.release.set()
        assert len(ensure.calls) <= 1

    def test_the_file_of_a_finished_download_is_installed_not_downloaded_again(self, loras, sources):
        ensure = _FakeEnsure()
        installer, lane = _installer(loras, sources, ensure)
        first = installer.start(_source())
        _wait(lane, first.item["id"], "done")

        again = installer.start(_source())

        assert (again.status, again.name, again.item) == ("installed", "some_repo", None)
        assert len(ensure.calls) == 1


class TestNames:
    def test_an_open_file_defaults_to_a_slug_of_the_repository_name(self, loras, sources):
        installer, _lane = _installer(loras, sources, _FakeEnsure(hold=True))

        decision = installer.decide(_source(repo="renderartist/Technically-Color-Z-Image-Turbo", file="pytorch_lora_weights.safetensors"))

        assert (decision.name, decision.status, decision.renamed) == ("technically_color_z_image_turbo", "free", False)

    def test_a_name_on_disk_gets_a_number_before_any_bytes_move(self, loras, sources):
        ensure = _FakeEnsure()
        installer, _lane = _installer(loras, sources, ensure)
        (loras / "some_repo.safetensors").write_bytes(b"other")
        (loras / "some_repo-2.safetensors").write_bytes(b"other")

        decision = installer.decide(_source())

        assert (decision.name, decision.renamed) == ("some_repo-3", True)
        assert ensure.calls == []

    def test_a_name_held_by_a_running_download_gets_a_number(self, loras, sources):
        ensure = _FakeEnsure(hold=True)
        installer, _lane = _installer(loras, sources, ensure)
        installer.start(_source(file="a.safetensors"))

        second = installer.start(_source(file="b.safetensors"))

        assert (second.name, second.renamed) == ("some_repo-2", True)

    def test_a_typed_name_is_used_and_a_taken_one_is_numbered_for_discover_but_refused_for_import(self, loras, sources):
        installer, _lane = _installer(loras, sources, _FakeEnsure(hold=True))
        (loras / "mine.safetensors").write_bytes(b"other")

        assert installer.decide(_source(), name="mine").name == "mine-2"
        assert installer.decide(_source(), name="fresh").name == "fresh"
        with pytest.raises(FileExistsError):
            installer.decide(_source(), name="mine", suffix_on_clash=False)

    def test_names_clash_regardless_of_case(self, loras, sources):
        installer, _lane = _installer(loras, sources, _FakeEnsure(hold=True))
        (loras / "Some_Repo.safetensors").write_bytes(b"other")

        assert installer.decide(_source()).name == "some_repo-2"

    @pytest.mark.parametrize("name", ["../evil", "a/b", "a\\b"])
    def test_an_unusable_typed_name_is_refused(self, loras, sources, name):
        installer, _lane = _installer(loras, sources, _FakeEnsure(hold=True))

        with pytest.raises(ValueError):
            installer.decide(_source(), name=name)

    def test_a_blank_typed_name_means_no_name(self, loras, sources):
        installer, _lane = _installer(loras, sources, _FakeEnsure(hold=True))

        assert installer.decide(_source(), name="").name == "some_repo"

    def test_a_curated_entry_whose_file_is_there_counts_as_installed_and_downloads_nothing(self, loras, sources):
        ensure = _FakeEnsure()
        installer, _lane = _installer(loras, sources, ensure)
        (loras / "pick_one.safetensors").write_bytes(_BYTES)

        result = installer.start(curated_source(_entry()), name="pick_one")

        assert (result.status, result.name, result.item) == ("installed", "pick_one", None)
        assert ensure.calls == []

    def test_a_curated_name_taken_by_a_different_file_is_refused_before_any_download(self, loras, sources):
        ensure = _FakeEnsure()
        installer, _lane = _installer(loras, sources, ensure)
        (loras / "pick_one.safetensors").write_bytes(b"something else entirely")

        with pytest.raises(FileExistsError):
            installer.start(curated_source(_entry()), name="pick_one")

        assert ensure.calls == []

    def test_a_curated_entry_must_be_given_its_name(self, loras, sources):
        installer, _lane = _installer(loras, sources, _FakeEnsure())

        with pytest.raises(ValueError):
            installer.decide(curated_source(_entry()))


def _install_by_hand(loras: Path, sources: LoraSources, name: str, source: LoraSource, content: bytes = _BYTES) -> None:
    """Put *content* in the LoRA folder as *name* and record *source* as where it came from."""
    (loras / f"{name}.safetensors").write_bytes(content)
    sources.record(LoraSourceRow(name=name, size_bytes=len(content), repo=source.repo, revision=source.revision, file=source.file, sha256=source.sha256))


class TestInstalledFile:
    def test_a_file_installed_under_a_numbered_name_is_found_and_downloads_nothing(self, loras, sources):
        ensure = _FakeEnsure()
        installer, _lane = _installer(loras, sources, ensure)
        (loras / "some_repo.safetensors").write_bytes(b"another repository's file")
        _install_by_hand(loras, sources, "some_repo-2", _source())

        result = installer.start(_source())

        assert (result.status, result.name, result.renamed, result.item) == ("installed", "some_repo-2", False, None)
        assert ensure.calls == []

    def test_a_new_commit_with_the_same_file_hash_is_still_installed(self, loras, sources):
        installer, _lane = _installer(loras, sources, _FakeEnsure())
        _install_by_hand(loras, sources, "some_repo", _source())

        decision = installer.decide(_source(revision="f" * 40))

        assert (decision.name, decision.status) == ("some_repo", "installed")

    def test_a_replaced_file_is_a_new_version_under_a_free_name(self, loras, sources):
        ensure = _FakeEnsure(b"new weights")
        installer, lane = _installer(loras, sources, ensure)
        _install_by_hand(loras, sources, "some_repo", _source())
        new_sha = hashlib.sha256(b"new weights").hexdigest()

        result = installer.start(_source(revision="f" * 40, sha256=new_sha, size_bytes=len(b"new weights")))
        _wait(lane, result.item["id"], "done")

        assert (result.status, result.name, result.renamed) == ("queued", "some_repo-2", True)
        assert _files(loras) == ["some_repo-2.safetensors", "some_repo.safetensors"]
        assert (loras / "some_repo.safetensors").read_bytes() == _BYTES

    def test_a_row_whose_file_is_gone_or_has_another_size_does_not_count(self, loras, sources):
        installer, _lane = _installer(loras, sources, _FakeEnsure(hold=True))
        _install_by_hand(loras, sources, "gone", _source())
        (loras / "gone.safetensors").unlink()
        _install_by_hand(loras, sources, "edited", _source())
        (loras / "edited.safetensors").write_bytes(b"edited by hand")

        assert installer.decide(_source()).status == "free"

    def test_an_import_with_no_name_finds_the_installed_copy_and_a_typed_name_asks_for_that_name(self, loras, sources):
        ensure = _FakeEnsure(hold=True)
        installer, _lane = _installer(loras, sources, ensure)
        _install_by_hand(loras, sources, "some_repo-2", _source())

        imported = installer.start(_source(anonymous=False), name=None, suffix_on_clash=False)
        typed = installer.decide(_source(), name="mine", suffix_on_clash=False)

        assert (imported.status, imported.name) == ("installed", "some_repo-2")
        assert (typed.status, typed.name) == ("free", "mine")
        assert ensure.calls == []

    def test_a_download_running_under_a_numbered_name_is_found_by_its_source(self, loras, sources):
        ensure = _FakeEnsure(hold=True)
        installer, _lane = _installer(loras, sources, ensure)
        (loras / "some_repo.safetensors").write_bytes(b"another repository's file")
        first = installer.start(_source())

        second = installer.start(_source())

        assert (first.name, second.name, second.status) == ("some_repo-2", "some_repo-2", "running")
        assert second.item["id"] == first.item["id"]


def test_pending_sizes_cover_only_downloads_that_are_still_running(loras, sources):
    ensure = _FakeEnsure(hold=True)
    installer, lane = _installer(loras, sources, ensure)
    first = installer.start(curated_source(_entry()), name="pick_one")
    installer.start(_source(size_bytes=None, sha256=None, file="x.safetensors"), name="unknown_size")

    assert installer.pending_sizes() == {"pick_one": len(_BYTES)}
    ensure.release.set()
    _wait(lane, first.item["id"], "done")
    assert installer.pending_sizes() == {}


class TestImportSource:
    @staticmethod
    def _read(files, prompt=None):
        return lambda repo: (_COMMIT, tuple(files), prompt)

    def test_takes_the_only_file_with_its_hub_hash_and_the_cards_trigger_and_uses_the_users_token(self):
        files = [RepoFile("only.safetensors", 123, "f" * 64, None, False)]

        source = import_source("o/r", None, read_repo=self._read(files, "t3chnic4lly, more"))

        assert (source.file, source.size_bytes, source.sha256, source.revision) == ("only.safetensors", 123, "f" * 64, _COMMIT)
        assert (source.trigger, source.trigger_source, source.anonymous, source.curated_id) == ("t3chnic4lly, more", "card", False, None)

    def test_several_files_need_a_choice_and_the_choice_must_exist(self):
        files = [RepoFile("a.safetensors", 1, None, None, False), RepoFile("b.safetensors", 1, None, None, False)]

        with pytest.raises(ValueError, match="Multiple"):
            import_source("o/r", None, read_repo=self._read(files))
        with pytest.raises(ValueError, match="not in repo"):
            import_source("o/r", "c.safetensors", read_repo=self._read(files))
        assert import_source("o/r", "b.safetensors", read_repo=self._read(files)).file == "b.safetensors"

    @pytest.mark.parametrize(("repo", "filename"), [("../x", None), ("o/r", "../a.safetensors"), ("o/r", "a.bin")])
    def test_bad_input_is_refused_before_the_hub_is_read(self, repo, filename):
        def never(_repo):
            raise AssertionError("the Hub must not be read")

        with pytest.raises(ValueError):
            import_source(repo, filename, read_repo=never)

    def test_a_repository_without_safetensors_files_is_refused(self):
        with pytest.raises(ValueError, match="No .safetensors"):
            import_source("o/r", None, read_repo=self._read([]))


class TestFetchPreview:
    @staticmethod
    def _png() -> bytes:
        import io

        from PIL import Image

        out = io.BytesIO()
        Image.new("RGB", (800, 600), (10, 120, 200)).save(out, format="PNG")
        return out.getvalue()

    def test_reads_the_image_without_credentials_and_returns_the_small_picture(self, monkeypatch):
        from zvisiongenerator.web import hub_client
        from zvisiongenerator.web.lora_download import fetch_preview

        seen = []

        def fetch(url, *, max_bytes):
            seen.append((url, max_bytes))
            return hub_client.HubResponse(200, self._png(), {})

        monkeypatch.setattr(hub_client, "http_fetch", fetch)

        content_type, data = fetch_preview("https://huggingface.co/o/r/resolve/a/i.png")

        assert content_type == "image/webp" and data
        assert [url for url, _ in seen] == ["https://huggingface.co/o/r/resolve/a/i.png"]

    @pytest.mark.parametrize(
        "answer",
        [None, "missing", "cut-off", "bad-bytes"],
    )
    def test_a_preview_that_cannot_be_read_gives_nothing(self, monkeypatch, answer):
        from zvisiongenerator.web import hub_client
        from zvisiongenerator.web.lora_download import fetch_preview

        def fetch(url, *, max_bytes):
            if answer is None:
                raise hub_client.HubError("Could not reach Hugging Face.")
            if answer == "missing":
                return hub_client.HubResponse(404, b"", {})
            if answer == "cut-off":
                return hub_client.HubResponse(200, self._png(), {}, truncated=True)
            return hub_client.HubResponse(200, b"not an image", {})

        monkeypatch.setattr(hub_client, "http_fetch", fetch)

        assert fetch_preview("https://huggingface.co/o/r/resolve/a/i.png") is None

    def test_a_url_off_the_hub_gives_nothing(self):
        from zvisiongenerator.web.lora_download import fetch_preview

        assert fetch_preview("https://evil.example/i.png") is None
