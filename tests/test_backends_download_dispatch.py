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
