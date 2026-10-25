"""Tests for the platform dispatch of the download planners in ``zvisiongenerator.backends``."""

from __future__ import annotations

from typing import Any

import pytest

import zvisiongenerator.backends as backends
from zvisiongenerator.backends.cuda import download_plan as cuda_plan
from zvisiongenerator.backends.mac import download_plan as mac_plan
from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils.h3_preset import HubFiles

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

    plan = backends.video_download_plan("owner/ltx", family="ltx", repo_ignore=("x/*",), mlx_text_encoder="owner/gemma")

    assert plan == (DownloadTarget("owner/ltx"), DownloadTarget("owner/gemma", allow_patterns=mac_plan.MLX_LM_ALLOW_PATTERNS))


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

    assert backends.image_download_plan("owner/repo", family="zimage") == (DownloadTarget("owner/repo", None, ("a/*",)),)
    assert backends.video_download_plan("owner/ltx", family="ltx", mlx_text_encoder="owner/gemma") == (
        DownloadTarget("owner/ltx"),
        DownloadTarget("owner/gemma", allow_patterns=mac_plan.MLX_LM_ALLOW_PATTERNS),
    )
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


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_the_cuda_platforms_register_the_h3_video_backend(monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
    monkeypatch.setattr(backends.sys, "platform", platform)

    assert backends.video_family_available("h3") is True
    assert backends.video_family_available("ltx") is True


def test_macos_registers_the_mlx_h3_video_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends.sys, "platform", "darwin")
    monkeypatch.setattr(backends, "VIDEO_BACKENDS", {})

    assert backends.video_family_available("h3") is True
    assert backends.get_video_backend("h3").name == "h3"


def test_get_video_backend_builds_the_h3_backend_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    from zvisiongenerator.backends.cuda.video.h3.backend import H3VideoBackend

    monkeypatch.setattr(backends.sys, "platform", "linux")
    monkeypatch.setattr(backends, "VIDEO_BACKENDS", {})

    backend = backends.get_video_backend("h3")

    assert isinstance(backend, H3VideoBackend)
    assert backend.name == "h3"
    assert backends.get_video_backend("h3") is backend
