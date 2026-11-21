"""The video medium of the memory bench: the clip each case draws, its ``ziv-video`` arguments and the app's verdict for it."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from bench.memory.image import reference_names, source_path, write_synthetic_picture
from bench.memory.machine import case_machine, verdict_row

OUTPUT_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm"})
LTX_FPS = 24
"""The frame rate of an LTX clip (the ``extend_fps`` of its family), which turns an extension's seconds into frames."""


def prepare(case: dict[str, Any], cases: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Set the ``width``, ``height`` and ``frames`` of *case*: the ``video_sizes`` preset of its ``ratio`` and ``size``, unless the case gives its own ``width`` and ``height`` (then no ``size``) or ``frames``.

    An Extend case (``extend_source``) draws at its source clip's size, and its frames are the window the app generates: the carried frames plus the
    ``extend_seconds`` it adds.
    """
    if case.get("extend_source"):
        from zvisiongenerator.utils.video_extend import resolve_extend_frames, window_frames

        source = cases["sources"][case["extend_source"]]
        preset = config["video_sizes"][source.get("ratio", case["ratio"])][source["size"]]
        added, _note = resolve_extend_frames(frames=None, seconds=case["extend_seconds"], fps=LTX_FPS, strict=False)
        case["width"], case["height"], case["frames"] = preset["width"], preset["height"], window_frames(added)
        case["pixels"] = case["width"] * case["height"]
        return
    preset = {} if case.get("width") and case.get("height") else config["video_sizes"][case["ratio"]][case["size"]]
    case["width"], case["height"] = case.get("width") or preset["width"], case.get("height") or preset["height"]
    case["frames"] = case.get("frames") or preset["frames"]
    case["pixels"] = case["width"] * case["height"]


def start_name(case: Mapping[str, Any]) -> str:
    """Return the name of the synthetic start picture of *case*: one per clip size, so cases of different sizes do not share a file."""
    return f"start-{case['width']}x{case['height']}"


def clip_path(run_dir: Path, name: str) -> Path:
    """Return where the source clip *name* (an entry of the case file's ``sources``) lives under *run_dir*."""
    return run_dir / "sources" / f"{name}.mp4"


def ensure_sources(case: Mapping[str, Any], run_dir: Path, cases: Mapping[str, Any]) -> None:
    """Make what *case* needs to start from, when it is not there yet.

    The synthetic reference pictures come at the ``reference`` size of the case file (the clip's own size without one), a ``start_image`` case's
    picture at the clip's size, and an Extend case's source clip is rendered by the app (see :func:`ensure_clip`).

    Raises:
        subprocess.CalledProcessError: A source clip could not be made.
        FileNotFoundError: The source clip's render wrote no video file.
    """
    size = cases.get("reference") or {"width": case["width"], "height": case["height"]}
    for name in reference_names(case.get("references", 0)):
        path = source_path(run_dir, name)
        if not path.exists():
            write_synthetic_picture(path, size["width"], size["height"], seed=sum(map(ord, name)))
    if case.get("start_image"):
        name = start_name(case)
        if not source_path(run_dir, name).exists():
            write_synthetic_picture(source_path(run_dir, name), case["width"], case["height"], seed=sum(map(ord, name)))
    if case.get("extend_source"):
        ensure_clip(case["extend_source"], cases["sources"][case["extend_source"]], case, run_dir, cases)


def ensure_clip(name: str, spec: Mapping[str, Any], case: Mapping[str, Any], run_dir: Path, cases: Mapping[str, Any]) -> None:
    """Render the source clip *name* once, in a process of its own before the case's: the app makes it text-to-video from the case file's prompt.

    *spec* is the ``sources`` entry: a ``model``, a ``size`` (and ``ratio``) of ``video_sizes`` and the ``frames``. The clip is the app's own
    output from a prompt, nobody's footage. It is made at the case's seed and steps, with the enhancer off.
    """
    target = clip_path(run_dir, name)
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as scratch:
        argv = [
            "-m",
            spec["model"],
            "--runs",
            "1",
            "--seed",
            str(case["seed"]),
            "--steps",
            str(case["steps"]),
            "--prompt",
            cases["prompt"],
            "--no-enhance",
            "--ratio",
            spec.get("ratio", case["ratio"]),
            "--size",
            spec["size"],
            "--frames",
            str(spec["frames"]),
            "--output",
            scratch,
        ]
        subprocess.run([sys.executable, "-c", "from zvisiongenerator.video_cli import main; main()", *argv], check=True)
        clips = [path for path in Path(scratch).iterdir() if path.suffix.lower() in OUTPUT_SUFFIXES]
        if not clips:
            raise FileNotFoundError(f"The bench's source clip {name} was not written.")
        shutil.move(str(clips[0]), target)


def case_argv(case: Mapping[str, Any], run_dir: Path) -> list[str]:
    """Return the ``ziv-video`` arguments that run *case*, writing its clip into ``run_dir/out/<id>``.

    An Extend case continues its source clip by ``extend_seconds`` and carries no size or frames: the app takes the source's size.
    """
    if case.get("extend_source"):
        return [
            "-m",
            case["model"],
            "--runs",
            "1",
            "--seed",
            str(case["seed"]),
            "--steps",
            str(case["steps"]),
            "--prompt",
            case["prompt"],
            "--no-enhance",
            "--extend",
            str(clip_path(run_dir, case["extend_source"])),
            "--extend-seconds",
            str(case["extend_seconds"]),
            "--output",
            str(run_dir / "out" / case["id"]),
        ]
    argv = [
        "-m",
        case["model"],
        "--runs",
        "1",
        "--seed",
        str(case["seed"]),
        "--steps",
        str(case["steps"]),
        "--prompt",
        case["prompt"],
        "--no-enhance",
        *(["--ratio", case["ratio"], "--size", case["size"]] if case.get("size") else ["--width", str(case["width"]), "--height", str(case["height"])]),
        "--frames",
        str(case["frames"]),
        "--output",
        str(run_dir / "out" / case["id"]),
    ]
    if not case.get("low_memory", True):
        argv.append("--no-low-memory")
    if case.get("upscale"):
        argv += ["--upscale", str(case["upscale"])]
    if case.get("lora_path"):
        argv += ["--lora", case["lora_path"]]
    if case.get("start_image"):
        argv += ["--image", str(source_path(run_dir, start_name(case)))]
    if case.get("references"):
        argv += ["--reference", ",".join(str(source_path(run_dir, name)) for name in reference_names(case["references"]))]
    return argv


def run_cli(argv: list[str]) -> None:
    """Run the app's ``ziv-video`` entry point with *argv* in this process."""
    from zvisiongenerator.video_cli import main

    sys.argv = ["ziv-video", *argv]
    main()


def install_probes(case: Mapping[str, Any]) -> Callable[[], dict[str, Any]] | None:
    """Return nothing: video has no loader placement to watch or force."""
    return None


def estimate(case: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the app's memory verdict for *case*'s clip, or ``None`` when the family has no estimate, the weights cannot be sized or the machine has no budget.

    The job is built by the app's own builder (:func:`zvisiongenerator.utils.video_memory_job.clip_memory_job`) from the case's size, mode, references and steps (the transformer
    forwards of an H3 clip), with the LoRA's recorded size in place of its file.
    """
    from zvisiongenerator.utils.config import load_config
    from zvisiongenerator.utils.memory_verdict import MemoryWeights, judge
    from zvisiongenerator.utils.model_metadata import alias_map
    from zvisiongenerator.utils.paths import canonical_model_name, resolve_model_path
    from zvisiongenerator.utils.video_memory_job import clip_memory_job
    from zvisiongenerator.utils.weight_totals import resolve_ltx_weights

    where = case_machine(case)
    if where is None:
        return None
    if case["family"] == "ltx":
        platform_key = "darwin" if where.mac else sys.platform
        aliases = alias_map(load_config())
        name = canonical_model_name(case["model"], aliases=aliases, platform_key=platform_key)
        resolved = resolve_ltx_weights(
            name, resolve_model_path(name, aliases=aliases, platform_key=platform_key), platform_key=platform_key, **({"find_local_dir": lambda _path: None} if where.foreign_mac else {})
        )
        if resolved is None:
            return None
        weights = resolved.weights
    elif case["family"] == "h3":
        weights = MemoryWeights()
    else:
        return None
    job = clip_memory_job(
        case["family"],
        width=case["width"],
        height=case["height"],
        frames=case["frames"],
        low_memory=case.get("low_memory", True),
        mode="extend" if case.get("extend_source") else "i2v" if case.get("start_image") else "t2v",
        references=case.get("references", 0),
        upscale=bool(case.get("upscale")),
        steps=case.get("steps"),
        lora_bytes=case.get("lora_bytes", 0),
    )
    return verdict_row(judge(job, where.budget, weights), where.budget)
