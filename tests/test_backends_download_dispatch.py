"""Tests for the platform dispatch of the download planners in ``zvisiongenerator.backends``."""

from __future__ import annotations

from typing import Any

import pytest

import zvisiongenerator.backends as backends
from zvisiongenerator.backends.cuda import download_plan as cuda_plan
from zvisiongenerator.backends.mac import download_plan as mac_plan
from zvisiongenerator.downloads.target import DownloadTarget

_PLANNERS = ("image_download_plan", "video_download_plan", "enhancer_download_plan")


def test_planners_are_exported() -> None:
    assert set(_PLANNERS) <= set(backends.__all__)


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_cuda_platforms_dispatch_to_the_cuda_planner(monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
    monkeypatch.setattr(backends.sys, "platform", platform)
    index = {"vae": ["diffusers", "AutoencoderKL"]}

    (image,) = backends.image_download_plan("owner/repo", family="zimage", read_index=lambda repo, revision: index)
    video = backends.video_download_plan("owner/ltx", family="ltx", transformer_source=("a/b", "t.safetensors"), upscale=False)
    enhancer = backends.enhancer_download_plan("owner/llm", "rev")

    assert image.ignore_patterns == cuda_plan.DIFFUSERS_IGNORE
    assert video == (DownloadTarget("owner/ltx", ignore_patterns=cuda_plan.SEPARATE_TRANSFORMER_IGNORE), DownloadTarget("a/b", allow_patterns=("t.safetensors",)))
    assert enhancer == (DownloadTarget("owner/llm", "rev", cuda_plan.TRANSFORMERS_ALLOW_PATTERNS),)


def test_cuda_dispatch_forwards_the_repo_ignore_and_keeps_the_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends.sys, "platform", "linux")
    ignore = ("LTX-2.5-*.safetensors", "samples/*")

    plan = backends.video_download_plan("owner/ltx@abc", family="ltx", repo_ignore=ignore)

    assert plan == (DownloadTarget("owner/ltx", "abc", ignore_patterns=ignore),)


def test_cuda_dispatch_without_a_repo_ignore_is_the_plain_2_3_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    """2.3 guard: no ``repo_ignore`` gives the plan the dispatch always gave."""
    monkeypatch.setattr(backends.sys, "platform", "linux")

    assert backends.video_download_plan("owner/ltx", family="ltx", transformer_source=("a/b", "t.safetensors"), upscale=False) == (
        DownloadTarget("owner/ltx", None, None, cuda_plan.SEPARATE_TRANSFORMER_IGNORE),
        DownloadTarget("a/b", None, ("t.safetensors",)),
    )


def test_the_planner_receives_the_repo_ignore(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(backends.sys, "platform", "linux")
    monkeypatch.setattr(cuda_plan, "video_download_plan", lambda model_ref, **kwargs: (seen.append(kwargs), ())[1])

    backends.video_download_plan("owner/ltx", family="ltx", repo_ignore=("x/*",))
    backends.video_download_plan("owner/ltx", family="ltx")

    assert [kwargs["repo_ignore"] for kwargs in seen] == [("x/*",), ()]


def test_macos_dispatch_accepts_the_repo_ignore(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends.sys, "platform", "darwin")
    monkeypatch.setattr(mac_plan, "ltx_mlx_text_encoder_repo", lambda: "owner/gemma")

    assert backends.video_download_plan("owner/ltx", family="ltx", repo_ignore=("x/*",)) == (DownloadTarget("owner/ltx"), DownloadTarget("owner/gemma", allow_patterns=mac_plan.MLX_LM_ALLOW_PATTERNS))


def test_macos_dispatch_forwards_text_encoder_bundled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends.sys, "platform", "darwin")
    monkeypatch.setattr(mac_plan, "ltx_mlx_text_encoder_repo", lambda: "owner/gemma")

    assert backends.video_download_plan("owner/ltx@abc", family="ltx", text_encoder_bundled=True) == (DownloadTarget("owner/ltx", "abc"),)


def test_cuda_dispatch_ignores_text_encoder_bundled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends.sys, "platform", "linux")

    assert backends.video_download_plan("owner/ltx", family="ltx", text_encoder_bundled=True) == backends.video_download_plan("owner/ltx", family="ltx")


def test_macos_dispatches_to_the_mac_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends.sys, "platform", "darwin")
    monkeypatch.setattr(mac_plan, "ltx_mlx_text_encoder_repo", lambda: "owner/gemma")
    monkeypatch.setattr(mac_plan, "_mflux_patterns", lambda family: ["a/*"])

    assert backends.image_download_plan("owner/repo", family="zimage") == (DownloadTarget("owner/repo", None, ("a/*",)),)
    assert backends.video_download_plan("owner/ltx", family="ltx") == (DownloadTarget("owner/ltx"), DownloadTarget("owner/gemma", allow_patterns=mac_plan.MLX_LM_ALLOW_PATTERNS))
    assert backends.enhancer_download_plan("owner/llm", None) == (DownloadTarget("owner/llm", None, mac_plan.MLX_LM_ALLOW_PATTERNS),)


def test_macos_image_dispatch_ignores_the_index_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends.sys, "platform", "darwin")
    monkeypatch.setattr(mac_plan, "_mflux_patterns", lambda family: ["a/*"])

    def reader(repo_id: str, revision: str | None) -> dict[str, Any]:
        raise AssertionError("macOS does not read model_index.json")

    assert backends.image_download_plan("owner/repo", family="zimage", read_index=reader)


def test_unsupported_platform_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends.sys, "platform", "freebsd")

    with pytest.raises(RuntimeError, match="freebsd"):
        backends.image_download_plan("owner/repo", family="zimage")
    with pytest.raises(RuntimeError, match="freebsd"):
        backends.video_download_plan("owner/ltx", family="ltx")
    with pytest.raises(RuntimeError, match="freebsd"):
        backends.enhancer_download_plan("owner/llm", None)
