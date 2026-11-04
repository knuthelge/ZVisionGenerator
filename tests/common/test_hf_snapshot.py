"""Tests for the pinned-pack lookup by Hugging Face reference and the cached-snapshot lookup."""

from __future__ import annotations

import pytest

from zvisiongenerator.utils.h3_preset import HubFiles
from zvisiongenerator.utils import hf_snapshot
from zvisiongenerator.utils.hf_snapshot import cached_snapshot, pack_for_reference, snapshot_with

PIN = "a1" * 20
PACK = HubFiles("Owner/Pack", PIN, ("a.safetensors",))
PACKS = {f"Owner/Pack@{PIN}": PACK}


@pytest.mark.parametrize("reference", ["Owner/Pack", "owner/pack", f"owner/PACK@{PIN}"])
def test_a_pack_is_named_by_its_repo_id_in_any_case_with_no_revision_or_its_own(reference: str) -> None:
    assert pack_for_reference(PACKS, reference) is PACK


@pytest.mark.parametrize("reference", ["Owner/Pack@" + "b2" * 20, "Owner/Other", "./Owner/Pack"])
def test_another_revision_repo_or_a_local_path_names_no_pack(reference: str) -> None:
    assert pack_for_reference(PACKS, reference) is None


def test_no_packs_name_nothing() -> None:
    assert pack_for_reference({}, "Owner/Pack") is None


def test_the_first_pack_of_a_repo_answers_an_unpinned_reference_and_each_pin_its_own() -> None:
    other = HubFiles("Owner/Pack", "c3" * 20, ("b.safetensors",))
    packs = {**PACKS, f"Owner/Pack@{'c3' * 20}": other}

    assert pack_for_reference(packs, "Owner/Pack") is PACK
    assert pack_for_reference(packs, f"Owner/Pack@{'c3' * 20}") is other


def _snapshots(monkeypatch, *folders):
    monkeypatch.setattr(hf_snapshot, "cached_snapshot_dirs", lambda repo_id, revision: list(folders))


def test_the_cached_snapshot_is_the_first_folder_that_holds_every_listed_file(monkeypatch, tmp_path) -> None:
    partial, complete = tmp_path / "partial", tmp_path / "complete"
    partial.mkdir()
    (partial / "other.json").write_text("x")
    complete.mkdir()
    (complete / "a.safetensors").write_text("x")
    _snapshots(monkeypatch, partial, complete)

    assert cached_snapshot(PACK) == complete


def test_no_cached_snapshot_with_every_file_gives_none(monkeypatch, tmp_path) -> None:
    _snapshots(monkeypatch, tmp_path)

    assert cached_snapshot(PACK) is None


def test_snapshot_with_uses_the_cached_snapshot_and_downloads_only_without_one(monkeypatch, tmp_path) -> None:
    (tmp_path / "a.safetensors").write_text("x")
    downloads: list[dict] = []

    def download(**kwargs):
        downloads.append(kwargs)
        return str(tmp_path / "downloaded")

    _snapshots(monkeypatch, tmp_path)
    assert snapshot_with(PACK, download) == tmp_path
    assert downloads == []

    _snapshots(monkeypatch)
    assert snapshot_with(PACK, download) == tmp_path / "downloaded"
    assert downloads == [{"repo_id": "Owner/Pack", "revision": PIN, "allow_patterns": ["a.safetensors"]}]


def _cache_with(root, repo_id: str, commit: str, names: tuple[str, ...], *, ref: str | None = None) -> None:
    """Write a Hub cache layout under *root*: one snapshot of *commit* holding *names*, and ``refs/<ref>`` pointing at it."""
    repo_dir = root / f"models--{repo_id.replace('/', '--')}"
    for name in names:
        blob = repo_dir / "blobs" / f"blob-{name.replace('/', '-')}"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_text("x")
        link = repo_dir / "snapshots" / commit / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(blob)
    if ref is not None:
        (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
        (repo_dir / "refs" / ref).write_text(commit)


@pytest.mark.parametrize(
    ("revision", "commit", "names", "ref"),
    [
        (PIN, PIN, ("config.json", "model.safetensors"), None),
        (PIN, PIN, ("config.json",), None),
        (PIN, "d4" * 20, ("config.json", "model.safetensors"), None),
        ("main", "e5" * 20, ("config.json", "model.safetensors"), "main"),
        ("main", "e5" * 20, ("config.json",), "main"),
    ],
    ids=["complete", "a-file-missing", "other-commit", "ref-complete", "ref-a-file-missing"],
)
def test_a_pack_is_downloaded_for_the_plan_exactly_when_the_loader_finds_it_in_the_cache(monkeypatch, tmp_path, revision, commit, names, ref) -> None:
    """The download plan's presence test and the loaders' cache lookup answer "downloaded" the same way."""
    from huggingface_hub import constants

    from zvisiongenerator.downloads.presence import target_present_offline
    from zvisiongenerator.downloads.target import DownloadTarget

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    _cache_with(tmp_path, "Owner/Pack", commit, names, ref=ref)
    files = HubFiles("Owner/Pack", revision, ("config.json", "model.safetensors"))

    planned = target_present_offline(DownloadTarget(files.repo_id, files.revision, files.files))

    assert planned == (cached_snapshot(files) is not None) == (len(names) == 2 and commit != "d4" * 20)
