"""Tests for the platform dispatch of the download planners in ``zvisiongenerator.backends``."""

from __future__ import annotations

from typing import Any

import pytest

import zvisiongenerator.backends as backends
from zvisiongenerator.backends.cuda import download_plan as cuda_plan
from zvisiongenerator.backends.mac import download_plan as mac_plan
from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils.h3_preset import HubFiles


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


def test_macos_dispatch_forwards_text_encoder_bundled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends.sys, "platform", "darwin")

    assert backends.video_download_plan("owner/ltx@abc", family="ltx", text_encoder_bundled=True) == (DownloadTarget("owner/ltx", "abc"),)


def test_macos_dispatch_plans_a_pinned_pack_as_its_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends.sys, "platform", "darwin")
    pack = HubFiles("owner/ltx", "abc", ("t.safetensors", "c.json"))

    plan = backends.video_download_plan("owner/ltx@abc", family="ltx", mlx_packs={"owner/ltx@abc": pack}, mlx_text_encoder="owner/gemma@def")

    assert plan == (DownloadTarget("owner/ltx", "abc", ("t.safetensors", "c.json")), DownloadTarget("owner/gemma", "def", mac_plan.MLX_LM_ALLOW_PATTERNS))


def test_cuda_dispatch_ignores_text_encoder_bundled_and_the_mlx_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends.sys, "platform", "linux")
    pack = HubFiles("owner/ltx", "abc", ("t.safetensors",))

    plan = backends.video_download_plan("owner/ltx", family="ltx", text_encoder_bundled=True, mlx_packs={"owner/ltx": pack}, mlx_text_encoder="owner/gemma")

    assert plan == backends.video_download_plan("owner/ltx", family="ltx")


def test_macos_dispatches_to_the_mac_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends.sys, "platform", "darwin")
    monkeypatch.setattr(mac_plan, "_mflux_patterns", lambda family: ["a/*"])
    monkeypatch.setattr(mac_plan, "_mflux_revision", lambda repo_id: None)

    assert backends.image_download_plan("owner/repo", family="zimage") == (DownloadTarget("owner/repo", None, ("a/*",)),)
    assert backends.video_download_plan("owner/ltx", family="ltx", repo_ignore=("x/*",), mlx_text_encoder="owner/gemma") == (
        DownloadTarget("owner/ltx"),
        DownloadTarget("owner/gemma", allow_patterns=mac_plan.MLX_LM_ALLOW_PATTERNS),
    )
    assert backends.enhancer_download_plan("owner/llm", None) == (DownloadTarget("owner/llm", None, mac_plan.MLX_LM_ALLOW_PATTERNS),)


def test_macos_image_dispatch_ignores_the_index_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends.sys, "platform", "darwin")
    monkeypatch.setattr(mac_plan, "_mflux_patterns", lambda family: ["a/*"])
    monkeypatch.setattr(mac_plan, "_mflux_revision", lambda repo_id: None)

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


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_cuda_platforms_dispatch_h3_to_the_cuda_h3_planner(monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
    monkeypatch.setattr(backends.sys, "platform", platform)
    seen: list[tuple[str, object, bool]] = []
    monkeypatch.setattr(cuda_plan, "h3_download_plan", lambda model_ref, preset, *, upscale=False: (seen.append((model_ref, preset, upscale)), ())[1])
    preset = object()

    assert backends.h3_download_plan("owner/h3", preset, upscale=True) == ()
    assert seen == [("owner/h3", preset, True)]


def test_a_platform_without_an_h3_backend_has_no_h3_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends.sys, "platform", "freebsd")

    with pytest.raises(RuntimeError, match="MiniMax H3"):
        backends.h3_download_plan("owner/h3", object())


def test_macos_dispatches_h3_to_the_mlx_planner_with_the_lora(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends.sys, "platform", "darwin")
    seen: list[tuple[str, object, object]] = []
    sentinel = (DownloadTarget("o/r"),)
    monkeypatch.setattr(mac_plan, "h3_mlx_download_plan", lambda model_ref, preset, lora: (seen.append((model_ref, preset, lora)), sentinel)[1])
    preset, lora = object(), object()

    assert backends.h3_download_plan("owner/h3", preset, upscale=True, lora=lora) == sentinel
    assert seen == [("owner/h3", preset, lora)]


def test_macos_h3_plan_needs_the_lora(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends.sys, "platform", "darwin")

    with pytest.raises(ValueError, match="turbo LoRA"):
        backends.h3_download_plan("owner/h3", object())
