"""Tests for holding and sweeping the staging folders of LoRA downloads."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from zvisiongenerator.web import lora_staging
from zvisiongenerator.web.lora_staging import StagingClaim, sweep_abandoned


class _FakeMsvcrt:
    """``msvcrt`` with ``locking`` done by ``flock``, so the Windows path runs on macOS and Linux (each file object has its own lock, as on Windows)."""

    LK_NBLCK = 2

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def locking(self, fd: int, mode: int, nbytes: int) -> None:
        import fcntl

        self.calls.append((mode, nbytes))
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.fixture(params=["native", "windows"])
def platform(request, monkeypatch) -> SimpleNamespace:
    """Run a test with the lock of this machine, and again with the lock Windows uses (``msvcrt.locking``) on a machine that has ``flock``."""
    fake = _FakeMsvcrt()
    if request.param == "windows":
        pytest.importorskip("fcntl")
        monkeypatch.setattr(lora_staging, "fcntl", None)
        monkeypatch.setattr(lora_staging, "msvcrt", fake)
    elif lora_staging.fcntl is None and lora_staging.msvcrt is None:
        pytest.skip("needs flock or msvcrt")
    return SimpleNamespace(name=request.param, msvcrt=fake)


@pytest.fixture
def loras(tmp_path) -> Path:
    folder = tmp_path / "loras"
    folder.mkdir()
    return folder


def _abandon(folder: Path) -> Path:
    """Leave a staging folder behind as a killed server does: a partial file and a lock file nobody holds."""
    folder.mkdir()
    (folder / ".in-use").write_bytes(b"")
    (folder / "weights.safetensors").write_bytes(b"partial" * 100)
    (folder / ".cache" / "huggingface").mkdir(parents=True)
    return folder


@pytest.mark.usefixtures("platform")
class TestClaim:
    def test_a_claim_empties_the_folder_and_release_removes_it(self, loras):
        folder = _abandon(loras / ".download-odd")
        claim = StagingClaim(folder)

        assert claim.acquire() == folder
        assert [path.name for path in folder.iterdir()] == [".in-use"]

        claim.release()
        assert not folder.exists()

    def test_a_folder_another_open_claim_holds_is_refused_and_left_alone(self, loras):
        folder = loras / ".download-odd"
        first = StagingClaim(folder)
        first.acquire()
        (folder / "weights.safetensors").write_bytes(b"in use")
        second = StagingClaim(folder)

        with pytest.raises(RuntimeError, match="already using"):
            second.acquire()
        second.release()

        assert (folder / "weights.safetensors").read_bytes() == b"in use"
        first.release()

    def test_releasing_a_claim_that_was_never_acquired_removes_nothing(self, loras):
        folder = _abandon(loras / ".download-odd")

        StagingClaim(folder).release()

        assert (folder / "weights.safetensors").is_file()


@pytest.mark.usefixtures("platform")
class TestSweep:
    def test_an_abandoned_folder_goes_with_its_partial_file(self, loras):
        gone = _abandon(loras / ".download-gone")
        (loras / "keep.safetensors").write_bytes(b"mine")

        sweep_abandoned(loras)

        assert not gone.exists()
        assert [path.name for path in loras.iterdir()] == ["keep.safetensors"]

    def test_a_folder_a_download_holds_stays(self, loras):
        held = loras / ".download-busy"
        claim = StagingClaim(held)
        claim.acquire()
        (held / "weights.safetensors").write_bytes(b"in use")
        gone = _abandon(loras / ".download-gone")

        sweep_abandoned(loras)

        assert (held / "weights.safetensors").read_bytes() == b"in use"
        assert not gone.exists()
        claim.release()

    def test_an_empty_folder_without_a_lock_file_goes_but_one_with_files_stays(self, loras):
        (loras / ".download-empty").mkdir()
        unknown = loras / ".download-unknown"
        unknown.mkdir()
        (unknown / "weights.safetensors").write_bytes(b"?")

        sweep_abandoned(loras)

        assert not (loras / ".download-empty").exists()
        assert (unknown / "weights.safetensors").is_file()

    def test_other_folders_and_files_are_never_touched(self, loras):
        (loras / ".cache").mkdir()
        (loras / ".cache" / "x").write_bytes(b"x")
        (loras / "style.safetensors").write_bytes(b"x")

        sweep_abandoned(loras)

        assert sorted(path.name for path in loras.iterdir()) == [".cache", "style.safetensors"]

    def test_a_missing_folder_is_not_an_error(self, tmp_path):
        sweep_abandoned(tmp_path / "nothing")


def _stick(monkeypatch, stuck: Path) -> None:
    """Make the files in *stuck* impossible to delete, as when a virus scanner holds one open."""
    real_unlink = Path.unlink

    def unlink(self, missing_ok=False):
        if self.parent == stuck:
            raise PermissionError(13, "in use", str(self))
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)


@pytest.mark.usefixtures("platform")
class TestFolderThatCannotBeEmptied:
    def test_the_download_stages_in_a_fresh_locked_folder_and_the_old_one_is_left(self, loras, monkeypatch):
        stuck = _abandon(loras / ".download-odd")
        _stick(monkeypatch, stuck)
        claim = StagingClaim(stuck)

        staged = claim.acquire()

        assert staged != stuck
        assert (staged.parent, staged.name.startswith(".download-odd")) == (loras, True)
        assert claim.folder == staged
        assert [path.name for path in staged.iterdir()] == [".in-use"]
        assert (stuck / "weights.safetensors").is_file()
        sweep_abandoned(loras)
        assert staged.is_dir()
        claim.release()
        assert not staged.exists()

    def test_the_old_folder_goes_with_a_later_sweep_once_it_can_be_removed(self, loras, monkeypatch):
        stuck = _abandon(loras / ".download-odd")
        with monkeypatch.context() as patch:
            _stick(patch, stuck)
            claim = StagingClaim(stuck)
            claim.acquire()
            claim.release()

        sweep_abandoned(loras)

        assert not stuck.exists()


class TestWindowsLock:
    @pytest.fixture(autouse=True)
    def windows(self, platform):
        if platform.name != "windows":
            pytest.skip("the Windows lock is exercised through the simulated one")
        self.fake = platform.msvcrt

    def test_a_claim_locks_one_byte_of_the_lock_file_without_waiting(self, loras):
        claim = StagingClaim(loras / ".download-odd")

        claim.acquire()

        assert self.fake.calls == [(self.fake.LK_NBLCK, 1)]
        claim.release()

    def test_the_lock_file_is_let_go_before_the_folder_is_removed(self, loras, monkeypatch):
        import fcntl

        claim = StagingClaim(loras / ".download-odd")
        folder = claim.acquire()
        seen = []
        real_rmtree = lora_staging.shutil.rmtree

        def rmtree(path, **kwargs):
            with open(folder / ".in-use", "rb") as probe:
                try:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    seen.append("free")
                except OSError:
                    seen.append("held")
            real_rmtree(path, **kwargs)

        monkeypatch.setattr(lora_staging.shutil, "rmtree", rmtree)

        claim.release()

        assert seen == ["free"]
        assert not folder.exists()

    def test_the_sweep_lets_go_of_a_lock_file_before_removing_its_folder(self, loras, monkeypatch):
        import fcntl

        gone = _abandon(loras / ".download-gone")
        seen = []
        real_rmtree = lora_staging.shutil.rmtree

        def rmtree(path, **kwargs):
            with open(gone / ".in-use", "rb") as probe:
                try:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    seen.append("free")
                except OSError:
                    seen.append("held")
            real_rmtree(path, **kwargs)

        monkeypatch.setattr(lora_staging.shutil, "rmtree", rmtree)

        sweep_abandoned(loras)

        assert seen == ["free"]
        assert not gone.exists()
