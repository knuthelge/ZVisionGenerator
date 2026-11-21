"""Check the bench's results against the engine's estimates (the calibration rule), and turn image results into size terms."""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from bench.memory.measure import says_out_of_memory
from zvisiongenerator.utils.memory_grade import CUDA_GPU_RATIOS, TOO_LARGE, grade
from zvisiongenerator.utils.memory_offsets import CUDA_OFFSET_LINUX_BYTES, CUDA_OFFSET_WINDOWS_BYTES

GIB = 1024**3
REFERENCE_PIXELS = 1024 * 1024
# The largest picture and reference count a job reaches (an edit draws the picture at its own size: 2048x2048 is 4.19 MP;
# Klein takes up to 4 references). A slope counts as a term when it adds 1 GiB or more by there.
REACHABLE_PIXELS = 2048 * 2048
REACHABLE_REFERENCES = 4
# The calibration rule: a job that ran never reads Too large and its estimate is within the tolerance of what it measured, on either side;
# a job that ran out of memory reads Too large against the card less what other programs held.
TOLERANCE_RATIO = 0.10
TOLERANCE_FLOOR_BYTES = GIB


def tolerance_bytes(measured: int) -> int:
    """Return how far an estimate may sit from what a case measured: 10 % of it, but at least 1 GiB."""
    return max(round(TOLERANCE_RATIO * measured), TOLERANCE_FLOOR_BYTES)


def _is_probe(row: Mapping[str, Any]) -> bool:
    """Return whether *row* is a probe of the app, not a plain run: a row recorded by an experiment the bench no longer runs (an emptied allocator cache, a cleared MLX cache or another MLX cache cap)."""
    return bool(row.get("empty_cache") or row.get("clear_mlx_cache_before_decode") or row.get("mlx_cache_limit_gib") is not None)


def _usable(row: Mapping[str, Any]) -> bool:
    """Return whether *row* ran and has a peak, and is not a probe of the app: the size terms and checks read plain runs only."""
    return row.get("status") == "ok" and row.get("platform_peak_bytes") is not None and not _is_probe(row)


def allocated_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the CUDA rows that recorded an allocated peak, with that peak in place of the reserved one.

    The estimate follows the allocated peak: the memory tensors held, which the picture size and the references move. The
    reserved peak also holds what the allocator keeps for reuse (a streamed transformer: the allocated peak stays flat while
    the reserved one climbs to the card's limit) and the fragmentation between blocks, so it says little about what a job needs.
    """
    return [dict(row) | {"platform_peak_bytes": row["platform_allocated_peak_bytes"]} for row in rows if row.get("platform_allocated_peak_bytes") is not None]


def _plain(row: Mapping[str, Any]) -> bool:
    return not row.get("edit") and not row.get("upscale")


def _slope(low: tuple[int, int], high: tuple[int, int], per: int, reach: float) -> int:
    """Return the bytes added per *per* units between ``(units, peak)`` points *low* and *high*, or 0 when that slope adds under 1 GiB by *reach* units beyond *low*."""
    if high[0] <= low[0]:
        return 0
    slope = (high[1] - low[1]) / ((high[0] - low[0]) / per)
    return round(slope) if slope * reach >= GIB else 0


def model_terms(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return each model's size and reference terms from its plain rows.

    The size term is the slope in bytes per megapixel from the model's smallest to its largest plain size without
    references, and the reference term the bytes per reference from the smallest size with none to the most it was run
    with; each counts only when it would add 1 GiB or more at the largest picture and reference count a job reaches
    (:data:`REACHABLE_PIXELS`, :data:`REACHABLE_REFERENCES`), otherwise it is 0 (the peak is flat).

    Args:
        rows: Result rows; failed rows and rows without a platform peak are ignored.

    Returns:
        One dict per ``(family, platform, model, quantize)``: the two terms and the peaks and deltas they come from.
    """
    groups: dict[tuple[str, str, str, int | None], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if _usable(row) and _plain(row):
            groups[(row["family"], row["platform"], row["model"], row.get("quantize"))].append(row)
    found = []
    for (family, platform, model, quantize), group in groups.items():
        by_pixels: dict[int, int] = {}
        for row in group:
            if not row["references"]:
                by_pixels[row["pixels"]] = max(by_pixels.get(row["pixels"], 0), row["platform_peak_bytes"])
        if not by_pixels:
            continue
        small, large = min(by_pixels), max(by_pixels)
        by_references: dict[int, int] = {}
        for row in group:
            if row["pixels"] == small:
                by_references[row["references"]] = max(by_references.get(row["references"], 0), row["platform_peak_bytes"])
        most = max(by_references)
        found.append(
            {
                "family": family,
                "platform": platform,
                "model": model,
                "quantize": quantize,
                "small_pixels": small,
                "large_pixels": large,
                "small_peak_bytes": by_pixels[small],
                "large_peak_bytes": by_pixels[large],
                "size_delta_bytes": by_pixels[large] - by_pixels[small],
                "per_megapixel": _slope((small, by_pixels[small]), (large, by_pixels[large]), 1_000_000, (REACHABLE_PIXELS - REFERENCE_PIXELS) / 1_000_000),
                "most_references": most,
                "reference_delta_bytes": by_references[most] - by_references[0] if 0 in by_references else None,
                "per_reference": _slope((0, by_references[0]), (most, by_references[most]), 1, REACHABLE_REFERENCES) if 0 in by_references else 0,
            }
        )
    return found


def size_terms(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str, int | None], dict[str, int]]:
    """Return ``per_megapixel`` and ``per_reference`` in bytes for each ``(family, platform, quantize)``, taking the larger when several models share a family.

    Levels stay apart because the weights sit on the accelerator differently at each (streamed, resident, offloaded).
    A term of 0 means the peak does not grow by 1 GiB over the sizes and references a job reaches.
    """
    terms: dict[tuple[str, str, int | None], dict[str, int]] = {}
    for model in model_terms(rows):
        entry = terms.setdefault((model["family"], model["platform"], model["quantize"]), {"per_megapixel": 0, "per_reference": 0})
        entry["per_megapixel"] = max(entry["per_megapixel"], model["per_megapixel"])
        entry["per_reference"] = max(entry["per_reference"], model["per_reference"])
    return terms


def predicted_peak(base_bytes: int, terms: Mapping[str, int], *, pixels: int, references: int) -> int:
    """Return the peak the estimator would give: *base_bytes* at 1 MP plus the size and reference terms, as the image estimate adds them."""
    return base_bytes + terms["per_megapixel"] * max(0, pixels - REFERENCE_PIXELS) // 1_000_000 + terms["per_reference"] * references


def edit_checks(rows: list[Mapping[str, Any]], terms: Mapping[tuple[str, str, int | None], Mapping[str, int]]) -> list[dict[str, Any]]:
    """Compare each edit row's peak with what the terms predict for its canvas and one reference (the picture itself).

    Returns:
        One dict per usable edit row with a plain 1 MP row of the same model: ``id``, ``observed_bytes``, ``predicted_bytes`` and ``error_bytes``.
    """
    bases = {(row["family"], row["platform"], row["model"], row.get("quantize")): row["small_peak_bytes"] for row in model_terms(rows)}
    checks = []
    for row in rows:
        key = (row["family"], row["platform"], row["model"], row.get("quantize"))
        if _usable(row) and row.get("edit") and key in bases and (row["family"], row["platform"], row.get("quantize")) in terms:
            predicted = predicted_peak(bases[key], terms[(row["family"], row["platform"], row.get("quantize"))], pixels=row["pixels"], references=1)
            checks.append({"id": row["id"], "observed_bytes": row["platform_peak_bytes"], "predicted_bytes": predicted, "error_bytes": row["platform_peak_bytes"] - predicted})
    return checks


def upscale_checks(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compare each upscaled row's peak with the plain row of the same model, quantize and output size.

    ``escalate`` is true when the upscaled job peaked more than 1 GiB above the plain one: the estimate would need an upscale input.
    """
    plain: dict[tuple[str, str, int | None, int], int] = {}
    for row in rows:
        if _usable(row) and _plain(row) and not row["references"]:
            key = (row["platform"], row["model"], row.get("quantize"), row["pixels"])
            plain[key] = max(plain.get(key, 0), row["platform_peak_bytes"])
    checks = []
    for row in rows:
        key = (row["platform"], row["model"], row.get("quantize"), row["pixels"])
        if _usable(row) and row.get("upscale") and key in plain:
            delta = row["platform_peak_bytes"] - plain[key]
            checks.append({"id": row["id"], "upscaled_bytes": row["platform_peak_bytes"], "plain_bytes": plain[key], "delta_bytes": delta, "escalate": delta > GIB})
    return checks


def driver_view(row: Mapping[str, Any]) -> dict[str, int] | None:
    """Return what the card took besides the job's tensors at its tightest moment, from a row's driver readings.

    ``taken`` is the driver's total less its lowest free memory (the CUDA context, the allocator's reserve and every other
    program); ``others`` what the other programs held before the job started; ``beside_tensors`` is ``taken - others`` less the allocated
    peak: the context, the allocator's reserve beyond the tensors and fragmentation, which the CUDA offset stands for.
    ``driver_reserved`` is what the driver keeps from the card's nvidia-smi total. ``None`` when the row has no driver readings.
    """
    keys = ("gpu_free_min_bytes", "gpu_driver_total_bytes", "gpu_baseline_used_bytes", "gpu_smi_total_bytes", "platform_allocated_peak_bytes")
    if any(row.get(key) is None for key in keys):
        return None
    taken = row["gpu_driver_total_bytes"] - row["gpu_free_min_bytes"]
    return {
        "taken": taken,
        "others": row["gpu_baseline_used_bytes"],
        "beside_tensors": taken - row["gpu_baseline_used_bytes"] - row["platform_allocated_peak_bytes"],
        "driver_reserved": row["gpu_smi_total_bytes"] - row["gpu_driver_total_bytes"],
    }


def _gib(value: int | None) -> str:
    return "-" if value is None else f"{value / GIB:.2f}"


def format_report(rows: list[Mapping[str, Any]]) -> str:
    """Return the results as text: each row's peaks and estimate, and, from the CUDA rows that recorded an allocated peak, the model and family terms and the edit and upscale checks."""
    lines = ["case                       platform  MP     peak   alloc  footprint  rssAnon  rssFile  rssShmem  anon+shm  +file  vmHWM  estimate  taken  others  beside  status"]
    for row in rows:
        mp = f"{row['pixels'] / 1e6:.2f}" if row.get("pixels") else "-"
        driver = driver_view(row) or {}
        lines.append(
            f"{row['id']:<26} {row['platform']:<8} {mp:>5} {_gib(row.get('platform_peak_bytes')):>7} {_gib(row.get('platform_allocated_peak_bytes')):>6} {_gib(row.get('footprint_peak_bytes')):>10} {_gib(row.get('rss_anon_peak_bytes')):>8} {_gib(row.get('rss_file_peak_bytes')):>8} "
            f"{_gib(row.get('rss_shmem_peak_bytes')):>9} {_gib(row.get('rss_anon_shmem_peak_bytes')):>9} {_gib(row.get('rss_anon_shmem_file_peak_bytes')):>6} {_gib(row.get('vm_hwm_bytes')):>6} {_gib((row.get('estimate') or {}).get('gpu_bytes')):>9} {_gib(driver.get('taken')):>6} {_gib(driver.get('others')):>7} {_gib(driver.get('beside_tensors')):>7}  {row['status']}{': ' + row['error'] if row.get('error') else ''}"
        )
    lines += [
        "",
        "GiB. peak is CUDA max_memory_reserved or MLX peak, alloc the CUDA max_memory_allocated (what the estimate follows), footprint a Mac process's peak phys_footprint (what Activity Monitor's Memory column shows for the app); estimate is the app's number for the case's size, and references (GPU bytes; on CUDA with the offset). anon+shm and +file are the highest anonymous + shared memory, and that plus the file-backed pages, the process held in one 0.1 s sample (the RAM figure); vmHWM is the kernel's own resident high-water mark of the process (anonymous + shared + file-backed, exact), to be read against +file. taken is what the card gave out at the job's tightest moment, others what other programs held before it, beside the rest after the tensors (context, allocator reserve, fragmentation).",
        "",
    ]
    fitted = allocated_rows(rows)
    for model in model_terms(fitted):
        lines.append(
            f"{model['family']}/{model['platform']} {model['model']} q{model['quantize']}: size {_gib(model['small_peak_bytes'])} -> {_gib(model['large_peak_bytes'])} GiB "
            f"(delta {_gib(model['size_delta_bytes'])}), per_megapixel {model['per_megapixel']}; "
            f"references 0 -> {model['most_references']} delta {_gib(model['reference_delta_bytes'])} GiB, per_reference {model['per_reference']}"
        )
    lines += ["", "Terms per family and quantize level from the allocated peak (bytes; 0 = flat, under 1 GiB at the largest size and reference count a job reaches):"]
    for (family, platform, quantize), term in sorted(size_terms(fitted).items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or 0)):
        lines.append(f"  {family}/{platform} q{quantize}: per_megapixel={term['per_megapixel']} per_reference={term['per_reference']}")
    for check in edit_checks(fitted, size_terms(fitted)):
        lines.append(f"edit {check['id']}: observed {_gib(check['observed_bytes'])} GiB, predicted {_gib(check['predicted_bytes'])} GiB, error {_gib(check['error_bytes'])} GiB")
    for check in upscale_checks(fitted):
        lines.append(f"upscale {check['id']}: {_gib(check['upscaled_bytes'])} vs plain {_gib(check['plain_bytes'])} GiB, delta {_gib(check['delta_bytes'])}{' ESCALATE' if check['escalate'] else ''}")
    return "\n".join(lines)


# --- the calibration rule ---------------------------------------------------------------------------------------------


def on_this_system(row: Mapping[str, Any]) -> bool:
    """Return whether the engine's estimate here can stand for a row: any row but a CUDA row measured on another operating system (:func:`_os`), whose estimate carries that system's offset and budget."""
    return row.get("platform") != "cuda" or _os(row) == sys.platform


def reestimate(rows: Iterable[Mapping[str, Any]], estimate_for: Callable[[Mapping[str, Any]], dict[str, Any] | None]) -> list[dict[str, Any]]:
    """Return the rows with the estimate *estimate_for* gives now in place of the stored one, so a changed constant is judged against runs already made.

    A row whose estimate cannot be made here (its model is not downloaded, or the machine has no budget) keeps its stored estimate.
    """
    result = []
    for row in rows:
        try:
            fresh = estimate_for(row)
        except Exception:  # noqa: BLE001 -- a model that is gone or unreadable leaves the stored estimate
            fresh = None
        result.append(dict(row) | ({} if fresh is None else {"estimate": fresh}))
    return result


def card_the_job_had(row: Mapping[str, Any]) -> int | None:
    """Return the GPU memory a row's job had to itself, context included, in the units the verdict grades against: the card, less what other programs (a hold in another process among them)
    held before the job started and less what an in-process hold took (``hold_charged_bytes``, which the driver's baseline does not count: the hold was taken after it), or ``None`` without the driver's readings.

    The card is the driver's total, or the budget the run's own estimate graded against (``run_budget_bytes``, saved from the stored estimate before :func:`reestimate`) when that is smaller:
    on Windows the budget sits below the driver's total, and the verdict's figures are in the budget's units. On Linux the two are equal. A row run under ``--budget-gib`` has a simulated budget,
    not the real card, so it counts from the driver's total.
    """
    total, others = row.get("gpu_driver_total_bytes"), row.get("gpu_baseline_used_bytes")
    if total is None or others is None:
        return None
    budget = None if row.get("budget_gib") else row.get("run_budget_bytes")
    return min(total, budget or total) - others - (row.get("hold_charged_bytes") or 0)


def _hold_gib(row: Mapping[str, Any]) -> float | None:
    """Return the hold a row ran under in GiB, whether the job's own process took it (``hold_in_process_gib``) or, in a row of an older run, another process did (``hold_gib``), or ``None`` for no hold."""
    return row.get("hold_gib") or row.get("hold_in_process_gib") or None


def _os(row: Mapping[str, Any]) -> str | None:
    """Return the operating system a row was measured on (``sys.platform``: ``linux``, ``win32``), or ``None`` for a row that records none (a Mac row of the first runs)."""
    return row.get("os")


def _case_key(row: Mapping[str, Any]) -> tuple[str | None, str, Any]:
    """Return what a hold bracket belongs to: the case on one operating system, at the card the app read (``budget_gib``; a case run under a smaller simulated card is its own case)."""
    return _os(row), row["id"], row.get("budget_gib")


def _offset(row: Mapping[str, Any]) -> int:
    """Return the CUDA offset of the operating system a row was measured on: the Windows constant for a ``win32`` row, the Linux one otherwise."""
    return CUDA_OFFSET_WINDOWS_BYTES if _on_windows(row) else CUDA_OFFSET_LINUX_BYTES


def _job(row: Mapping[str, Any]) -> tuple[Any, ...] | None:
    """Return what makes two CUDA image rows the same job (operating system, model, quantization level, size, frames, references, edit, extend and upscale, the LoRA bytes, the placement the loader took and
    the card the app read), or ``None`` for a row that is not a CUDA image row or names no model.

    A LoRA adds to the job's memory, so a LoRA row is never the twin of the plain case; nor is a row run under a simulated card (``budget_gib``) the twin of one on the real card.
    """
    if row.get("platform") != "cuda" or row.get("media") != "image" or not row.get("model"):
        return None
    keys = ("os", "media", "model", "quantize", "width", "height", "frames", "references", "edit", "extend", "upscale", "loaded_placement", "budget_gib")
    return tuple(row.get(key) for key in keys) + (row.get("lora_bytes") or 0,)


def with_brackets(rows: Iterable[Mapping[str, Any]], *, report: bool = True) -> list[dict[str, Any]]:
    """Return the rows with ``bracket_bytes`` on every row of a case that was run under holds and both ran and ran out of memory.

    Brackets and twins are kept per operating system (:func:`_os`) and per card the app read (:func:`_case_key`): a Windows row is never judged by a Linux bracket, the outcomes of one case id on two
    systems are not merged, and a row run under ``--budget-gib`` neither takes nor spoils the bracket of the same case on the full card.

    The bracket is ``(low, high)``: the largest card the case ran out of memory on and the smallest it ran on (:func:`card_the_job_had`), so the least memory
    the job runs on lies between them. A case with only one outcome, or whose outcomes overlap (a run that failed for another reason), has none; an overlap is
    printed with both cards (unless *report* is false) so that it is not dropped silently.

    A row that ran without a bracket of its own, but whose job is identical to a probed case's (:func:`_job`; image rows only, the other media measure each case on its own), gets that case's bracket as ``twin_bracket_bytes``
    and its id as ``twin_case``: the bench measured that job's least memory, so the plain row is judged by it as well.
    """
    ran: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    failed: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for row in rows:
        card = card_the_job_had(row)
        if card is None or row.get("platform") != "cuda":
            continue
        if row.get("status") == "ok":
            ran[_case_key(row)].append(card)
        elif _ran_out_of_memory(row):
            failed[_case_key(row)].append(card)
    brackets = {}
    for case in sorted(ran.keys() & failed.keys(), key=str):
        if max(failed[case]) < min(ran[case]):
            brackets[case] = (max(failed[case]), min(ran[case]))
        elif report:
            where = f" on a {case[2]:g} GiB budget" if case[2] else ""
            print(f"NO BRACKET {case[1]} ({case[0]}){where}: its outcomes overlap, out of memory on {_gib(max(failed[case]))} GiB but ran on {_gib(min(ran[case]))} GiB")
    twins: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    for row in rows:
        if _case_key(row) in brackets and _job(row) is not None:
            twins.setdefault(_job(row), _case_key(row))
    result = []
    for row in rows:
        row = dict(row)
        if _case_key(row) in brackets:
            row["bracket_bytes"] = brackets[_case_key(row)]
        elif row.get("status") == "ok" and (twin := twins.get(_job(row))) is not None:
            row["twin_case"], row["twin_bracket_bytes"] = twin[1], brackets[twin]
        result.append(row)
    return result


def measured_gpu_bytes(row: Mapping[str, Any]) -> int | None:
    """Return the figure the estimate is compared with, in the quantity the bench measures, or ``None`` when the row has no peak to give it.

    On CUDA that is the least memory the job runs on. Where the case was probed with holds (:func:`with_brackets`), or an identical job was, it is the
    middle of the bracket; otherwise the allocated peak plus the CUDA offset stands in for it (image, video and audio alike), which is the free-card peak: some phases
    (a VAE decode) take less when the card is tight, so the stand-in can sit above the least the job needs. The reserved peak depends on the card's free
    memory (the allocator keeps what it can), so it says little about what a job needs. On a Mac it is the process's lifetime peak footprint,
    the figure the estimate is fitted to; a row recorded without one has no figure and is left out of the rule.
    """
    if row.get("platform") == "cuda":
        if bracket := row.get("bracket_bytes") or row.get("twin_bracket_bytes"):
            return sum(bracket) // 2
        peak = row.get("platform_allocated_peak_bytes")
        return None if peak is None else peak + _offset(row)
    return row.get("footprint_peak_bytes")


def _basis(row: Mapping[str, Any]) -> str:
    """Return what a row's measured GPU figure rests on: ``bracket`` (the middle of its hold probes), ``twin bracket`` (the middle of an identical job's), ``peak+offset`` (not measured: the stand-in), or a Mac's footprint (``-`` when the row has none)."""
    if row.get("platform") != "cuda":
        return "footprint" if row.get("footprint_peak_bytes") is not None else "-"
    return "bracket" if row.get("bracket_bytes") else ("twin bracket" if row.get("twin_bracket_bytes") else "peak+offset")


def _on_windows(row: Mapping[str, Any]) -> bool:
    """Return whether a row was measured on Windows (rows record ``os``)."""
    return row.get("os") == "win32"


def measured_ram_bytes(row: Mapping[str, Any]) -> int | None:
    """Return the RAM the job held at its peak, or ``None`` where the platform gave nothing.

    On Linux that is the highest anonymous plus shared memory the process held in one reading (``rss_anon_shmem_peak_bytes``); where the estimate counts
    weights mapped from disk (``ram_mapped``), the file-backed pages are in that reading's sum too (``rss_anon_shmem_file_peak_bytes``). On Windows it is the
    process's highest private working set (``private_working_set_peak_bytes``): the resident pages no other process shares, pinned host memory included.

    The figure is the simultaneous peak, not the sum of each counter's own peak: the counters peak at different moments, so that sum overstates what the
    job needed. A row recorded before the simultaneous peaks were kept has none and gives ``None``; its separate peaks cannot be combined honestly, so the
    case is measured again.
    """
    if _on_windows(row):
        return row.get("private_working_set_peak_bytes")
    mapped = (row.get("estimate") or {}).get("ram_mapped")
    return row.get("rss_anon_shmem_file_peak_bytes" if mapped else "rss_anon_shmem_peak_bytes")


SAMPLER_GAP_BYTES = GIB // 2
"""How far the kernel's high-water mark may sit above the sampler's simultaneous peak before the sampler is judged to miss peaks (0.5 GiB)."""


def sampler_gap_bytes(row: Mapping[str, Any]) -> int | None:
    """Return how far the kernel's ``VmHWM`` of the case's process sits above the sampler's simultaneous anon + shared + file-backed peak, or ``None`` without both.

    ``VmHWM`` is the exact resident high-water mark and counts anonymous, shared and file-backed pages, so it is compared with ``rss_anon_shmem_file_peak_bytes``
    (the file-backed pages included whether or not the estimate counts them); the RAM figure of a row that does not count mapped weights leaves the file-backed
    pages out and sits below both. A gap above :data:`SAMPLER_GAP_BYTES` means the 0.1 s sampler missed a short peak. Each case runs in a process of its own, so the
    mark is the case's; a Linux row recorded before ``vm_hwm_bytes`` was kept has none.
    """
    high_water, sampled = row.get("vm_hwm_bytes"), row.get("rss_anon_shmem_file_peak_bytes")
    return None if high_water is None or sampled is None else high_water - sampled


def _beyond_the_whole_memory(estimate: Mapping[str, Any]) -> bool:
    """Return whether a Mac verdict read Too large because its need is more than the machine's whole memory, which a job can still run in by swapping."""
    total, need = estimate.get("unified_bytes"), estimate.get("gpu_bytes")
    return bool(total) and need is not None and need > total


def _ran_out_of_memory(row: Mapping[str, Any]) -> bool:
    return row.get("status") == "failed" and says_out_of_memory(row.get("error") or "")


def _reads_too_large_beside_others(row: Mapping[str, Any], estimate: Mapping[str, Any]) -> bool:
    """Return whether a case that ran out of memory reads Too large against the card less what other programs held (the card the job had).

    Only a CUDA GPU figure has others to subtract; a RAM verdict, or a row without their reading, is judged as the verdict stands.
    """
    if estimate["status"] == TOO_LARGE:
        return True
    budget, need, others = estimate.get("budget_bytes"), estimate.get("gpu_bytes"), row.get("gpu_baseline_used_bytes")
    if row.get("platform") != "cuda" or budget is None or need is None or others is None:
        return False
    return grade(need, budget - others, CUDA_GPU_RATIOS) == TOO_LARGE


def _check(row: Mapping[str, Any], name: str, ok: bool, estimate: int | None, measured: int | None) -> dict[str, Any]:
    return {"row": row["_row"], "id": row["id"], "check": name, "ok": ok, "estimate_bytes": estimate, "measured_bytes": measured, "hold_gib": _hold_gib(row), "budget_gib": row.get("budget_gib")}


def rule_checks(rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the calibration rule to each row's estimate.

    A job that ran must read below Too large (a Mac job estimated beyond the machine's whole memory reads Too large and can still run, by swapping),
    with a GPU estimate within :func:`tolerance_bytes` of its measured figure (:func:`measured_gpu_bytes`) on either side and a RAM estimate within the
    same tolerance of its measured RAM; a job that ran out of memory must read Too large against the card less what other programs held
    (:func:`_reads_too_large_beside_others`). A case run under a budget judges a card it did not run on, so its status is not checked, only the coverage. A forced-placement case is judged
    in the placement it forced. A failure that is not an out-of-memory error is a setup failure, so it is left out.

    Args:
        rows: Result rows with their ``estimate`` (see :func:`reestimate`).

    Returns:
        The checks (``row``, the row's position; ``id``, ``check``, ``ok``, ``estimate_bytes``, ``measured_bytes``, ``hold_gib``, ``budget_gib``) and the rows
        left out with their reason (``row``, ``id``, ``reason``).
    """
    checks: list[dict[str, Any]] = []
    left_out: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        row = dict(row) | {"_row": position}
        estimate = row.get("estimate") or {}
        reason = None
        if _is_probe(row):
            reason = "a probe of the app, not of the estimate"
        elif row.get("status") == "failed" and not _ran_out_of_memory(row):
            reason = "a setup failure"
        elif estimate.get("status") is None:
            reason = "no estimate"
        elif row["status"] == "ok" and measured_gpu_bytes(row) is None:
            reason = "no peak to compare"
        if reason is not None:
            left_out.append({"row": position, "id": row["id"], "reason": reason})
            continue
        # A hold or a budget makes the verdict judge a card the job did not have to itself, so its outcome says nothing about the verdict's status.
        judged_card = bool(_hold_gib(row) or row.get("budget_gib") or row.get("total_gib"))
        if row["status"] == "failed":
            if not judged_card:
                checks.append(_check(row, "out of memory reads Too large", _reads_too_large_beside_others(row, estimate), estimate.get("gpu_bytes"), None))
            continue
        measured, estimated = measured_gpu_bytes(row), estimate.get("gpu_bytes")
        if not (row.get("budget_gib") or row.get("total_gib")):
            checks.append(_check(row, "ran reads below Too large", estimate["status"] != "too_large" or _beyond_the_whole_memory(estimate), estimated, measured))
        if estimated is not None:
            name = "GPU estimate matches the peak" + (f" (judged by the probes of {row['twin_case']})" if row.get("twin_bracket_bytes") else "")
            checks.append(_check(row, name, abs(estimated - measured) <= tolerance_bytes(measured), estimated, measured))
        ram, ram_estimate = measured_ram_bytes(row), estimate.get("system_bytes")
        if ram is not None and ram_estimate is not None:
            checks.append(_check(row, "RAM estimate matches the peak", abs(ram_estimate - ram) <= tolerance_bytes(ram), ram_estimate, ram))
    return checks, left_out


def _placement(row: Mapping[str, Any]) -> str:
    """Return the placement a row ran in, or ``estimated>loaded`` when the loader stepped down from what the verdict assumed."""
    estimated, loaded = (row.get("estimate") or {}).get("placement"), row.get("loaded_placement")
    if row.get("placement"):
        return f"{loaded or row['placement']}!"
    return f"{estimated}>{loaded}" if estimated and loaded and estimated != loaded else str(loaded or estimated or "-")


def format_rule(rows: list[Mapping[str, Any]], checks: list[Mapping[str, Any]], left_out: list[Mapping[str, Any]]) -> str:
    """Return the rule's result as text: each row's estimate next to its measurement, every failed check, and the rows left out."""
    failed, skipped = {check["row"] for check in checks if not check["ok"]}, {item["row"] for item in left_out}
    lines = ["case                         hold budget placement  status     est GPU  meas GPU  est RAM  meas RAM  vmHWM  hwm gap  footprint  outcome         basis        rule"]
    for position, row in enumerate(rows):
        estimate = row.get("estimate") or {}
        outcome = "ran" if row.get("status") == "ok" else ("out of memory" if _ran_out_of_memory(row) else "failed")
        verdict = "FAIL" if position in failed else ("-" if position in skipped else "ok")
        lines.append(
            f"{row['id']:<28} {_gib(int(_hold_gib(row) * GIB) if _hold_gib(row) else None):>4} {_gib(int(row['budget_gib'] * GIB) if row.get('budget_gib') else None):>6} {_placement(row):<10} "
            f"{str(estimate.get('status') or '-'):<9} {_gib(estimate.get('gpu_bytes')):>8} {_gib(measured_gpu_bytes(row)):>9} {_gib(estimate.get('system_bytes')):>8} {_gib(measured_ram_bytes(row)):>9}  {_gib(row.get('vm_hwm_bytes')):>5}  {_gib(sampler_gap_bytes(row)):>7}  {_gib(row.get('footprint_peak_bytes')):>9}  {outcome:<14}  {_basis(row):<12} {verdict}"
        )
    lines += [
        "",
        "GiB. est is the engine's estimate now; meas the least memory the job runs on: on CUDA the middle of the case's hold bracket (basis bracket; twin bracket where an identical job was probed instead) or, where it has no probes, the allocated peak plus the CUDA offset (basis peak+offset, not measured), a Mac's lifetime peak footprint; RAM is the highest anon + shmem in one sample (with the file-backed pages where the estimate counts mapped weights; Windows: the private working set's peak, pinned memory included); vmHWM is the kernel's resident high-water mark of the case's process (anon + shmem + file-backed, exact) and hwm gap is vmHWM less the sampler's simultaneous anon + shmem + file-backed peak (the like-for-like reading, whatever the RAM figure counts), where a gap above 0.5 GiB means the sampler missed a peak; footprint is a Mac process's peak phys_footprint (what Activity Monitor's Memory column shows for the app). A placement marked ! was forced; a>b means the loader stepped down from a to b.",
    ]
    for item in left_out:
        lines.append(f"left out: {item['id']} ({item['reason']})")
    missed = [(row["id"], gap) for row in rows if (gap := sampler_gap_bytes(row)) is not None and gap > SAMPLER_GAP_BYTES]
    if missed:
        lines.append("")
        lines += [f"SAMPLER MISSED A PEAK {case}: VmHWM is {_gib(gap)} GiB above the sampled anon + shmem + file peak" for case, gap in missed]
    failures = [check for check in checks if not check["ok"]]
    lines.append("")
    for check in failures:
        lines.append(f"FAIL {check['id']}: {check['check']} (estimate {_gib(check['estimate_bytes'])} GiB, measured {_gib(check['measured_bytes'])} GiB)")
    lines.append(f"The calibration rule {'fails' if failures else 'holds'}: {len(checks) - len(failures)} of {len(checks)} checks pass in {len({check['id'] for check in checks})} cases.")
    return "\n".join(lines)


def hold_brackets(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return what each probed case needed: the card it ran out of memory on, the card it ran on, and the offset those cards leave above its free-card peak.

    A case is probed by running it under growing holds (``run --hold-in-process-gib``: the job's own process takes that much of the card, and :func:`card_the_job_had` subtracts what the hold took), and the job either runs or
    runs out of memory. ``low_bytes`` and ``high_bytes`` are the cards it had (:func:`card_the_job_had`: its own context included) at the largest hold it failed
    under and the smallest it ran under; the least memory the job runs on lies between them. ``offset_low_bytes`` and ``offset_high_bytes`` are those cards less the
    case's free-card allocated peak: where a job's peak does not shrink under a hold they bracket the CUDA offset, and where a phase takes less memory on a tight card
    (a VAE decode) they fall below it.
    """
    rows = with_brackets(rows, report=False)
    peaks: dict[tuple[Any, ...], int] = {}
    for row in rows:
        if row.get("status") == "ok" and row.get("platform_allocated_peak_bytes") is not None:
            peaks[_case_key(row)] = max(peaks.get(_case_key(row), 0), row["platform_allocated_peak_bytes"])
    found: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if row.get("bracket_bytes") and _case_key(row) not in found:
            low, high = row["bracket_bytes"]
            peak = peaks.get(_case_key(row))
            found[_case_key(row)] = {
                "id": row["id"],
                "os": _os(row),
                "budget_gib": row.get("budget_gib"),
                "low_bytes": low,
                "high_bytes": high,
                "peak_bytes": peak,
                "offset_low_bytes": None if peak is None else low - peak,
                "offset_high_bytes": None if peak is None else high - peak,
            }
    return list(found.values())


def format_probes(probes: list[Mapping[str, Any]]) -> str:
    """Return the hold brackets as text, one line per probed case: the cards it ran out of memory and ran on, and the offset they leave above its free-card peak."""
    if not probes:
        return ""
    lines = [
        "",
        "Hold probes (the least memory a probed job runs on lies between the card it ran out of memory on and the one it ran on; the CUDA offset is one per operating system, shown on each line):",
    ]
    for probe in probes:
        middle = (probe["low_bytes"] + probe["high_bytes"]) // 2
        offsets = (
            ""
            if probe["peak_bytes"] is None
            else f", free-card peak {_gib(probe['peak_bytes'])}, offset {probe['offset_low_bytes'] / GIB:+.2f}..{probe['offset_high_bytes'] / GIB:+.2f} (CUDA offset on {probe['os']} {_gib(_offset(probe))})"
        )
        budget = f" (budget {probe['budget_gib']:g})" if probe.get("budget_gib") else ""
        lines.append(f"  {probe['id']}{budget}: out of memory on {_gib(probe['low_bytes'])}, ran on {_gib(probe['high_bytes'])}, figure {_gib(middle)}{offsets}")
    return "\n".join(lines)
