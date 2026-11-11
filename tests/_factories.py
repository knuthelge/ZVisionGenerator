"""Plain builders shared by test modules in more than one area.

Import them with ``from tests._factories import ...``; they are functions, not fixtures, so a test calls them with the overrides it needs.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml
from PIL import Image

from zvisiongenerator.core.image_types import EditWork, ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.core.job_plan import JobPlan
from zvisiongenerator.utils import model_metadata as model_metadata_module
from zvisiongenerator.core.video_types import ExtendSource
from zvisiongenerator.preflight import plan_iterations
from zvisiongenerator.utils.provenance import build_image_config_payload
from zvisiongenerator.utils.video_model_detect import VideoModelInfo
from zvisiongenerator.utils.video_probe import VideoProbe


# What ``resolve_defaults`` returns for a FLUX.2 Klein model in the image CLI: it edits with a mask and takes reference pictures.
KLEIN_EDIT_DEFAULTS = {"steps": 4, "guidance": 1.0, "scheduler": None, "supports_mask_edit": True, "max_reference_images": 4}


def _make_args(**overrides):
    """Return a minimal argparse Namespace for run_batch()."""
    defaults = dict(
        ratio="2:3",
        size="m",
        width=None,
        height=None,
        runs=1,
        seed=42,
        steps=4,
        guidance=1.0,
        scheduler=None,
        upscale=None,
        upscale_denoise=None,
        upscale_steps=None,
        upscale_guidance=None,
        upscale_sharpen=False,
        upscale_save_pre=False,
        review=False,
        image_path=None,
        image_strength=0.5,
        reference_paths=None,
        output="outputs",
        model="test-model",
        sharpen=True,
        contrast=False,
        saturation=False,
        transparent=False,
        lora_paths=None,
        lora_weights=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _make_plan(prompts_data, args, *, config=None, enhance_by_set=None, scene=None, scene_by_set=None) -> JobPlan:
    """Return the preflight plan of a job without enhancement rewrites (statuses stay ``off``)."""
    generation = (config or {}).get("generation", {})
    return JobPlan(
        plan_iterations(
            prompts_data,
            runs=args.runs,
            seed=args.seed,
            seed_min=generation.get("seed_min", 4),
            seed_max=generation.get("seed_max", 2**32 - 1),
            json_prompt=bool(getattr(args, "json_prompt_enabled", False)),
            disabled=bool(getattr(args, "no_enhance", False)),
            override=getattr(args, "enhance", None),
            enhance_by_set=enhance_by_set,
            scene=scene,
            scene_by_set=scene_by_set,
        )
    )


def make_edit_payload(
    source: str | Path = "/pictures/door.png", *, painted: bool = False, extend: tuple[str, int] | None = None, size: tuple[int, int] = (640, 480), cutout: Image.Image | None = None, **overrides: Any
) -> dict[str, Any]:
    """Return the config an edit of *source* embeds, built by the real provenance writer; *overrides* are extra request fields."""
    request = ImageGenerationRequest(
        backend=MagicMock(),
        model=object(),
        prompt="a red door",
        model_name="zit",
        edit_source=str(source),
        edit_paint=np.ones((4, 4), dtype=bool) if painted else None,
        edit_extend=extend,
        seed=7,
        steps=9,
        guidance=3.5,
        scheduler="beta",
        width=832,
        height=1216,
        **overrides,
    )
    artifacts = ImageWorkingArtifacts(image=Image.new("RGB", size))
    artifacts.metadata["edit_source"] = {"path": str(source), "width": 512, "height": 480}
    artifacts.generation_time = 3.14
    artifacts.edit = EditWork(canvas=Image.new("RGB", (512, 480)), cutout=cutout)
    return build_image_config_payload(request, artifacts)


def _make_video_args(**overrides):
    """Return a minimal argparse Namespace for video CLI / run_video_batch()."""
    defaults = dict(
        model="dgrauet/ltx-2.3-mlx-q4",
        prompt=None,
        prompts_file="prompts.yaml",
        image_path=None,
        ratio="16:9",
        size="m",
        width=704,
        height=448,
        num_frames=49,
        steps=8,
        seed=42,
        runs=1,
        low_memory=True,
        output=".",
        format="mp4",
        lora=None,
        lora_paths=[],
        lora_weights=[],
        upscale=None,
        upscale_steps=None,
        no_audio=False,
        audio=True,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def make_extend_source(**overrides: Any) -> ExtendSource:
    """Return a checked 24 fps, 512x256, 49-frame source with sound that was recorded with a scene and an action."""
    values: dict[str, Any] = dict(
        path=Path("/videos/clip.mp4"),
        width=512,
        height=256,
        fps=24.0,
        fps_text="24/1",
        frame_count=49,
        has_audio=True,
        video_start=0.0,
        scene="A woman in a red raincoat on a rainy street.",
        prompt="She opens a black umbrella.",
    )
    values.update(overrides)
    return ExtendSource(**values)


def make_probe(**overrides: Any) -> VideoProbe:
    """Return the probe of a clean 512x256, 24 fps, 73-frame h264 clip with audio and no colour tags."""
    values: dict[str, Any] = dict(
        coded_width=512,
        coded_height=256,
        width=512,
        height=256,
        codec="h264",
        profile="High",
        pix_fmt="yuv420p",
        bits_per_sample=8,
        sample_aspect_ratio="1:1",
        r_frame_rate="24/1",
        avg_frame_rate="24/1",
        time_base="1/12288",
        field_order="progressive",
        color_space=None,
        color_primaries=None,
        color_trc=None,
        rotation=0,
        frame_count=73,
        duration=3.04,
        video_start=0.0,
        format_start=0.0,
        has_audio=True,
    )
    values.update(overrides)
    return VideoProbe(**values)


def _make_mock_video_backend(name="ltx"):
    """Return a MagicMock satisfying the VideoBackend Protocol."""
    mock = MagicMock()
    mock.name = name
    mock.text_to_video.return_value = Path("/tmp/test.mp4")
    mock.image_to_video.return_value = Path("/tmp/test.mp4")
    mock.load_model.return_value = (
        MagicMock(),
        VideoModelInfo(family=name, backend=name, supports_i2v=True, default_fps=24, frame_alignment=8, resolution_alignment=32),
    )
    return mock


def _patch_image_main_dependencies(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Stub everything ``image_cli.main`` loads and return the fake backend; the batch run itself is a mock."""
    import zvisiongenerator.image_cli as image_cli
    import zvisiongenerator.utils.lora as lora_utils

    backend = MagicMock()
    backend.name = "stub-image-backend"
    backend.load_model = MagicMock(return_value=(object(), MagicMock(family="zimage", size=None)))

    monkeypatch.setattr(
        image_cli,
        "load_config",
        lambda: {
            "generation": {"default_ratio": "2:3", "default_size": "m"},
            "sizes": {"2:3": {"m": {"width": 832, "height": 1216}}},
        },
    )
    monkeypatch.setattr(image_cli, "select_ratio_size_defaults", lambda *args, **kwargs: ("2:3", "m"))
    monkeypatch.setattr(image_cli, "resolve_model_path", lambda model, **kwargs: model)
    monkeypatch.setattr(image_cli, "resolve_defaults", lambda *args, **kwargs: {"steps": 10, "guidance": 0.5, "scheduler": None})
    monkeypatch.setattr(image_cli, "validate_scheduler", lambda *args, **kwargs: None)
    monkeypatch.setattr(image_cli, "detect_image_model", lambda _: MagicMock(family="zimage", size=None))
    monkeypatch.setattr(image_cli, "get_backend", lambda: backend)
    monkeypatch.setattr(lora_utils, "resolve_lora_path", lambda name: f"/resolved/{Path(name).name}")
    monkeypatch.setattr(image_cli, "run_batch", MagicMock())
    monkeypatch.setattr(lora_utils.os.path, "isfile", lambda _path: True)
    return backend


@functools.cache
def shared_packaged_config() -> dict[str, Any]:
    """Return the packaged ``config.yaml`` parsed once per process, with an empty data dir so no user override applies; never mutate it."""
    from zvisiongenerator.utils import config as config_module

    with tempfile.TemporaryDirectory() as empty, pytest.MonkeyPatch.context() as patch:
        data_dir = Path(empty) / ".ziv"
        patch.setenv("ZIV_DATA_DIR", str(data_dir))
        patch.setattr(config_module, "get_ziv_data_dir", lambda: data_dir)
        return config_module.load_config()


def load_packaged_config() -> dict[str, Any]:
    """Return a private deep copy of the packaged config (parsed once per process), so a test may change it freely.

    Inside a test, the ``packaged_config`` fixture returns the same thing.
    """
    return copy.deepcopy(shared_packaged_config())


def video_config(**families: dict[str, Any]) -> dict[str, Any]:
    """Return a config with the video settings of the shipped families (``ltx``, ``h3``); a family given by keyword replaces its settings.

    These are the only video keys a config carries; the facts of a family (its rule, capabilities and files) are the model metadata's.
    """
    settings: dict[str, Any] = {
        "ltx": {"default_steps": 8, "upscale": {"default_upscale_steps": 8, "existing_video_steps": 2}},
        "h3": {"default_steps": 8, "default_frames": 124, "diffusers": {"ffn_chunk_tokens": 4096}},
    }
    settings.update(families)
    return {"video_model_presets": settings}


def metadata_copy() -> dict[str, Any]:
    """Return a private deep copy of the packaged model metadata, so a test may change it freely.

    Never write into ``load_model_metadata()``'s result itself: it is the shared cached parse. Pass the edited copy through
    :func:`use_model_metadata`.
    """
    from zvisiongenerator.utils.model_metadata import load_model_metadata

    return copy.deepcopy(load_model_metadata())


_PACKAGED_METADATA = yaml.load(model_metadata_module.METADATA_FILE.read_text(encoding="utf-8"), Loader=model_metadata_module.YAML_LOADER)
"""The packaged metadata, parsed once at import (before any test serves other metadata); read through ``packaged_metadata_copy``."""


def packaged_metadata_copy() -> dict[str, Any]:
    """Return a private deep copy of the packaged metadata, even while a test serves other metadata."""
    return copy.deepcopy(_PACKAGED_METADATA)


def use_built_model_pins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ids: tuple[str, ...] | None = ("ideo", "moss", "moss-sft"),
    *,
    sft: str = "org/moss-sft@" + "d" * 40,
    lora: str = "org/moss-turbo@" + "e" * 40,
    components: dict[str, str] | None = None,
    transformer: str = "org/instant-transformer@" + "a" * 40,
    keep: tuple[str, ...] = (),
) -> None:
    """Serve metadata holding only the models built on first use among *ids* (every model for ``None``), pinned to test repos.

    The packaged models named in *keep* stay as they are.

    *components* maps a platform key to the Ideogram 4 Instant components repo (default: ``org/ideo-nf4`` on every platform) and
    *transformer* is its transformer reference. The MOSS entries pin *sft* and the Turbo LoRA *lora*.
    """
    data = packaged_metadata_copy()
    if ids is not None:
        data["models"] = {model_id: data["models"][model_id] for model_id in (*ids, *keep)}
    for model_id in ("moss", "moss-sft"):
        for target in data["models"].get(model_id, {}).get("downloads", ()):
            target["ref"] = sft if target["role"] == "sft" else lora
    pins = components or dict.fromkeys(("darwin", "win32", "linux"), "org/ideo-nf4@" + "c" * 40)
    for platform_key, pin in pins.items():
        for target in data["models"].get("ideo", {}).get("downloads", {}).get(platform_key, ()):
            target["ref"] = pin if target["role"] == "components" else transformer
    use_model_metadata(monkeypatch, tmp_path, data)


def use_model_entries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, models: dict[str, dict[str, Any]] | None = None, *, video_families: bool = False, enhancers: bool = False) -> None:
    """Make the metadata hold only the ``models`` entries in *models* (none by default), so a test sees no built-in alias but its own.

    Entries need a ``kind`` (``image`` or ``video``) and ``downloads`` for every platform. The metadata has no licences and no
    image families; it keeps the packaged ``video_families`` when *video_families* is set and the packaged enhancer entries when
    *enhancers* is set (every Web UI config reads them).
    """
    packaged = packaged_metadata_copy()
    entries = {model_id: packaged["models"][model_id] for model_id in ("enhancer-fast", "enhancer-smart")} if enhancers else {}
    data: dict[str, Any] = {"licences": {}, "models": {**entries, **(models or {})}}
    if video_families:
        data["video_families"] = packaged["video_families"]
    use_model_metadata(monkeypatch, tmp_path, data)


def use_model_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data: dict[str, Any]) -> None:
    """Make the metadata readers read *data* instead of the packaged file for this test.

    A section *data* leaves out is served empty, as the readers expect every section of the packaged file to exist.
    """
    from zvisiongenerator.utils import model_metadata

    text = yaml.safe_dump({"licences": {}, "image_families": {}, "video_families": {}, "models": {}, **data}, sort_keys=False)  # file order is meaningful: aliases and tiers follow it
    # Named by content: the parse is cached per path, so a second call in one test with other data must not reuse a path.
    path = tmp_path / f"model_metadata-{hashlib.sha256(text.encode()).hexdigest()[:12]}.yaml"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(model_metadata, "METADATA_FILE", path)


def use_enhancer_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    fast: str | None = None,
    smart: str | None = None,
    smart_memory_gb: float | None = None,
    fast_size: str | None = None,
    smart_size: str | None = None,
) -> None:
    """Serve the packaged metadata with the enhancer tiers changed to what is given, the same on every platform; a tier left out stays as packaged.

    *fast* and *smart* are a reference (``repo`` or ``repo@revision``).
    """
    data = metadata_copy()
    for model_id, ref, size in (("enhancer-fast", fast, fast_size), ("enhancer-smart", smart, smart_size)):
        entry = data["models"][model_id]
        if ref is not None:
            entry["downloads"] = {platform_key: [{"ref": ref, "files_from": "loader"}] for platform_key in ("darwin", "win32", "linux")}
        if size is not None:
            entry["download_size"] = size
    if smart_memory_gb is not None:
        data["models"]["enhancer-smart"]["memory_gb"] = smart_memory_gb
    use_model_metadata(monkeypatch, tmp_path, data)
