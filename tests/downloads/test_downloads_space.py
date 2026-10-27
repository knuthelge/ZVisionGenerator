"""Tests for the free-space estimate."""

from __future__ import annotations

from types import SimpleNamespace

from zvisiongenerator.downloads.space import SpaceNeed, space_need
from zvisiongenerator.downloads.target import DownloadTarget, TargetCheck


def _usage(free: int):
    calls: list = []

    def disk_usage(path):
        calls.append(path)
        return SimpleNamespace(free=free)

    disk_usage.calls = calls  # type: ignore[attr-defined]
    return disk_usage


def test_cache_targets_sum_missing_bytes_under_cache_dir(tmp_path):
    items = [(DownloadTarget("o/a"), TargetCheck(2, 100)), (DownloadTarget("o/b"), TargetCheck(1, 50))]
    usage = _usage(1000)

    need = space_need(items, cache_dir=tmp_path, disk_usage=usage)

    assert need == SpaceNeed(path=tmp_path, needed_bytes=150, free_bytes=1000)
    assert need.enough is True


def test_only_local_dir_targets_use_first_local_folder_ancestor(tmp_path):
    first = tmp_path / "first" / "model"
    second = tmp_path / "second"
    items = [
        (DownloadTarget("o/a", local_dir=str(first)), TargetCheck(1, 10)),
        (DownloadTarget("o/b", local_dir=str(second)), TargetCheck(1, 20)),
    ]

    need = space_need(items, cache_dir=tmp_path / "cache", disk_usage=_usage(5))

    assert need.path == tmp_path
    assert need.needed_bytes == 30


def test_mixed_plan_uses_cache_drive_with_sum_of_both(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    items = [
        (DownloadTarget("o/comps"), TargetCheck(3, 700)),
        (DownloadTarget("o/transformer", local_dir=str(tmp_path / "elsewhere")), TargetCheck(1, 300)),
    ]
    usage = _usage(10_000)

    need = space_need(items, cache_dir=cache, disk_usage=usage)

    assert need.path == cache
    assert need.needed_bytes == 1000
    assert usage.calls == [cache]


def test_missing_cache_folder_walks_up_to_existing_ancestor(tmp_path):
    need = space_need([(DownloadTarget("o/a"), TargetCheck(1, 1))], cache_dir=tmp_path / "no" / "such" / "cache", disk_usage=_usage(1))

    assert need.path == tmp_path


def test_targets_with_nothing_missing_are_ignored(tmp_path):
    local = tmp_path / "local"
    items = [
        (DownloadTarget("o/done"), TargetCheck(0, 0)),
        (DownloadTarget("o/need", local_dir=str(local)), TargetCheck(1, 40)),
    ]

    need = space_need(items, cache_dir=tmp_path / "cache", disk_usage=_usage(100))

    assert need.path == tmp_path
    assert need.needed_bytes == 40


def test_nothing_missing_returns_none(tmp_path):
    assert space_need([(DownloadTarget("o/a"), TargetCheck(0, 0))], cache_dir=tmp_path, disk_usage=_usage(1)) is None
    assert space_need([], cache_dir=tmp_path, disk_usage=_usage(1)) is None


def test_unknown_sizes_count_as_zero_and_short_space_is_not_enough(tmp_path):
    items = [(DownloadTarget("o/a"), TargetCheck(1, None)), (DownloadTarget("o/b"), TargetCheck(1, 500))]

    need = space_need(items, cache_dir=tmp_path, disk_usage=_usage(499))

    assert need.needed_bytes == 500
    assert need.enough is False
