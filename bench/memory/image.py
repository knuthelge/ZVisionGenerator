"""The image medium of the memory bench: the canvas each case draws, its ``ziv-image`` arguments, synthetic pictures and the app's verdict for it."""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bench.memory.machine import case_machine, verdict_row
from bench.memory.measure import platform_label
from bench.memory.probes import trace_mlx_job, watch_placement

if TYPE_CHECKING:
    from zvisiongenerator.utils.memory_verdict import MemoryJob, MemoryVerdict, MemoryWeights
    from zvisiongenerator.utils.model_memory import MemoryBudget

OUTPUT_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})


def prepare(case: dict[str, Any], cases: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Set the ``width``, ``height`` and ``pixels`` the model draws for *case*."""
    case["width"], case["height"] = case_canvas(case, config["sizes"], cases["pictures"])
    case["pixels"] = case["width"] * case["height"]


def case_canvas(case: Mapping[str, Any], sizes: Mapping[str, Any], pictures: Mapping[str, Any]) -> tuple[int, int]:
    """Return the ``(width, height)`` the model draws for *case*.

    A plain or upscaled case is the preset's output size, or its own ``canvas`` (an upscale refines back to it); an edit is the canvas of
    ``edit_model_size`` for the picture and the Extend.

    Args:
        case: An expanded case.
        sizes: The app's ``sizes`` table: ratio, then size, then ``width`` and ``height``.
        pictures: The case file's synthetic pictures by name.
    """
    if case.get("edit"):
        from zvisiongenerator.processing.edit_region import edit_model_size
        from zvisiongenerator.utils.edit_request import parse_extend

        picture = pictures[case["edit"]]
        extend = parse_extend(case["extend"]) if case.get("extend") else None
        return edit_model_size(picture["width"], picture["height"], masked=False, extend=extend)
    if case.get("canvas"):
        return case["canvas"]["width"], case["canvas"]["height"]
    preset = sizes[case["ratio"]][case["size"]]
    return preset["width"], preset["height"]


def source_path(run_dir: Path, name: str) -> Path:
    """Return where the synthetic picture or reference *name* lives under *run_dir*."""
    return run_dir / "sources" / f"{name}.png"


def reference_names(count: int) -> list[str]:
    """Return the names of *count* distinct synthetic references."""
    return [f"reference-{number}" for number in range(1, count + 1)]


def case_argv(case: Mapping[str, Any], run_dir: Path) -> list[str]:
    """Return the ``ziv-image`` arguments that run *case*, writing its result into ``run_dir/out/<id>``.

    An edit keeps the picture's own size, so it carries no ratio, size or upscale; the CLI refuses them with ``--edit``. A case marked ``lora`` runs
    with the file ``run --lora`` gave it (``lora_path``, which the results never record).
    """
    out = run_dir / "out" / case["id"]
    argv = [
        "-m",
        case["model"],
        *(["-q", str(case["quantize"])] if case["quantize"] else []),
        "--runs",
        "1",
        "--seed",
        str(case["seed"]),
        "--steps",
        str(case["steps"]),
        "--prompt",
        case["prompt"],
        "--no-enhance",
        "--output",
        str(out),
    ]
    if case.get("lora_path"):
        argv += ["--lora", case["lora_path"]]
    if case.get("edit"):
        argv += ["--edit", str(source_path(run_dir, case["edit"]))]
        if case.get("extend"):
            argv += ["--extend", case["extend"]]
        return argv
    argv += ["--ratio", case["ratio"], "--size", case["size"]]
    if case.get("canvas"):
        argv += ["--width", str(case["canvas"]["width"]), "--height", str(case["canvas"]["height"])]
    if case["references"]:
        argv += ["--reference", ",".join(str(source_path(run_dir, name)) for name in reference_names(case["references"]))]
    if case.get("upscale"):
        argv += ["--upscale", str(case["upscale"])]
    return argv


def write_synthetic_picture(path: Path, width: int, height: int, seed: int) -> None:
    """Write a seeded synthetic picture of colour gradients, rectangles and noise to *path*; it is nobody's photograph."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    phase = rng.uniform(0.0, 1.0, size=3).astype(np.float32)[None, None, :]
    canvas = 0.5 + 0.5 * np.sin(2 * np.pi * (x * (1.0 + phase) + y * (2.0 - phase)))
    canvas = np.broadcast_to(canvas, (height, width, 3)).copy()
    for _ in range(12):
        x0, y0 = int(rng.integers(0, width - 8)), int(rng.integers(0, height - 8))
        x1, y1 = min(width, x0 + int(rng.integers(8, max(9, width // 3)))), min(height, y0 + int(rng.integers(8, max(9, height // 3))))
        canvas[y0:y1, x0:x1] = rng.uniform(0.0, 1.0, size=3)
    canvas += rng.normal(0.0, 0.03, size=canvas.shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(canvas, 0.0, 1.0) * 255).astype(np.uint8)).save(path)


def ensure_sources(case: Mapping[str, Any], run_dir: Path, cases: Mapping[str, Any]) -> None:
    """Write the synthetic picture and references *case* needs, when they are not there yet."""
    wanted = {name: cases["pictures"][name] for name in [case["edit"]]} if case.get("edit") else {}
    wanted |= {name: cases["reference"] for name in reference_names(case["references"])}
    for name, size in wanted.items():
        path = source_path(run_dir, name)
        if not path.exists():
            write_synthetic_picture(path, size["width"], size["height"], seed=sum(map(ord, name)))


def run_cli(argv: list[str]) -> None:
    """Run the app's ``ziv-image`` entry point with *argv* in this process."""
    from zvisiongenerator.image_cli import main

    sys.argv = ["ziv-image", *argv]
    main()


def install_probes(case: Mapping[str, Any]) -> Callable[[], dict[str, Any]] | None:
    """Watch, and when the case forces one, override the CUDA loader's placement; macOS keeps fixed placements, so it traces the job's memory per denoise step and around the VAE decode instead."""
    if platform_label() == "mac":
        return trace_mlx_job()
    return watch_placement(case.get("placement"))


@contextlib.contextmanager
def _placement_forced(placement: str | None) -> Iterator[None]:
    """Make the app's own verdict judge *placement* where it would choose among its candidates, for the block; ``None`` leaves the app's choice alone.

    The app picks the placement itself (:func:`~zvisiongenerator.utils.memory_verdict.cuda_image_placement`, which takes the first of
    :func:`~zvisiongenerator.utils.memory_verdict.image_placement_candidates` that fits), so its verdict has no override; the bench forces q4 placements
    through the loader (a q8 or unquantized model always streams) and judges the one that ran, through the same engine. Ideogram 4 and MOSS-Image have one placement per level and
    never read the candidates.
    """
    if placement is None:
        yield
        return
    from zvisiongenerator.utils import memory_verdict

    real = memory_verdict.image_placement_candidates
    memory_verdict.image_placement_candidates = lambda _quantize: (placement,)
    try:
        yield
    finally:
        memory_verdict.image_placement_candidates = real


def judge_forced_placement(job: MemoryJob, machine: MemoryBudget, weights: MemoryWeights, placement: str | None) -> MemoryVerdict:
    """Return the app's verdict for *job* with its weights forced into *placement* (``"resident"``, ``"turns"`` or ``"streamed"``), which the app's own choice would not take; ``None`` is the app's choice."""
    from zvisiongenerator.utils.memory_verdict import judge

    with _placement_forced(placement):
        return judge(job, machine, weights)


def _case_job(case: Mapping[str, Any], cases: Mapping[str, Any]) -> MemoryJob:
    """Return the app's memory job for *case*, built by the app's own builders (:mod:`zvisiongenerator.utils.image_memory_job`) from the sizes the case records in place of files.

    An edit draws the canvas of its picture and extend and holds the picture as its one reference; an upscale refines a render and an Extend redraws part of a picture, so both start
    from a picture at the canvas size. A plain image holds ``references`` pictures of the case file's ``reference`` size.
    """
    from zvisiongenerator.utils.image_memory_job import sized_edit_job, sized_image_job, starts_from_picture

    lora_bytes = case.get("lora_bytes", 0)
    if case.get("edit"):
        from zvisiongenerator.utils.edit_request import parse_extend

        picture = cases["pictures"][case["edit"]]
        return sized_edit_job((picture["width"], picture["height"]), masked=False, extend=parse_extend(case["extend"]) if case.get("extend") else None, lora_bytes=lora_bytes)
    reference = cases["reference"]
    return sized_image_job(
        case["width"],
        case["height"],
        reference_megapixels=(reference["width"] * reference["height"] / 1e6,) * case.get("references", 0),
        lora_bytes=lora_bytes,
        from_picture=starts_from_picture(upscale=case.get("upscale"), start_image=False),
    )


def estimate(case: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the app's memory verdict for *case* at its size and references, or ``None`` when the model has none or the machine has no budget.

    The row is :func:`bench.memory.machine.verdict_row`, with ``ram_mapped`` besides. A case's ``budget_gib`` makes the app read a card of that size (a Mac: ``total_gib`` is its unified memory), and
    its ``placement`` makes the verdict judge that placement (:func:`_placement_forced`), as does the ``loaded_placement`` a finished row records.
    """
    from zvisiongenerator.utils.model_metadata import alias_map
    from zvisiongenerator.utils.paths import canonical_model_name, resolve_model_path
    from zvisiongenerator.utils.weight_totals import resolve_image_weights

    from bench.memory.cases import load_cases
    from zvisiongenerator.utils.config import load_config

    where = case_machine(case)
    if where is None:
        return None
    platform_key = "darwin" if where.mac else sys.platform
    aliases = alias_map(load_config())
    name = canonical_model_name(case["model"], aliases=aliases, platform_key=platform_key)
    # A case measures a downloaded model, so its files give the weights.
    resolved = resolve_image_weights(
        name, resolve_model_path(name, aliases=aliases, platform_key=platform_key), platform_key=platform_key, **({"find_local_dir": lambda _path: None} if where.foreign_mac else {})
    )
    if resolved is None:
        return None
    job = replace(_case_job(case, load_cases("image")), family=case["family"], model_size=resolved.model_size, quantize=case.get("quantize"))
    # A forced placement, or the one the loader took (a hold or a budget can make it step down), is the one the estimate judges.
    verdict = judge_forced_placement(job, where.budget, resolved.weights, None if where.mac else case.get("placement") or case.get("loaded_placement"))
    # Unquantized weights that stream are mapped from disk: the estimate counts them, so the measured RAM counts the file-backed pages too.
    return verdict_row(verdict, where.budget) | {"ram_mapped": not where.mac and verdict.placement == "streamed" and (resolved.stored_bits or case.get("quantize")) is None}
