"""Compare two bench runs as a Markdown report: check rates overall and per tag, speed, and the cases that changed most."""

from __future__ import annotations

import difflib
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench.enhance.checks import CHECK_NAMES, case_score
from bench.enhance.runner import META_FILE, RESULTS_FILE

_TOP_CHANGED = 10


@dataclass(frozen=True)
class Run:
    """One bench run read back from disk."""

    label: str
    meta: dict[str, Any]
    records: tuple[dict[str, Any], ...]


def load_run(run_dir: Path) -> Run:
    """Read a run's ``meta.json`` and ``results.jsonl``.

    Raises:
        FileNotFoundError: When *run_dir* has no results.
    """
    results = run_dir / RESULTS_FILE
    if not results.exists():
        raise FileNotFoundError(f"No {RESULTS_FILE} in {run_dir}.")
    meta_path = run_dir / META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    records = tuple(json.loads(line) for line in results.read_text(encoding="utf-8").splitlines() if line.strip())
    return Run(label=run_dir.name, meta=meta, records=records)


def check_rates(records: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> dict[str, tuple[float | None, int]]:
    """Return each check's mean over the records where it applies, with how many records that was."""
    rates: dict[str, tuple[float | None, int]] = {}
    for name in CHECK_NAMES:
        values = [float(record["checks"][name]) for record in records if record["checks"].get(name) is not None]
        rates[name] = (sum(values) / len(values) if values else None, len(values))
    return rates


def mean_score(records: list[dict[str, Any]]) -> float | None:
    """Return the mean ``case_score`` of *records*, or None when there are none."""
    return statistics.fmean(case_score(record["checks"]) for record in records) if records else None


def rewrite_seconds(record: dict[str, Any]) -> float:
    """Return the model time of one record: all its calls, retries included."""
    return sum(attempt["seconds"] for attempt in record["attempts"])


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value * 100:.0f}%"


def _delta(base: float | None, new: float | None) -> str:
    if base is None or new is None:
        return "–"
    points = (new - base) * 100
    return f"{points:+.0f}"


def _median(values: list[float]) -> str:
    return f"{statistics.median(values):.1f}s" if values else "–"


def _fence(text: str) -> str:
    return "```text\n" + (text.strip() or "(empty)") + "\n```"


def _by_tag(records: tuple[dict[str, Any], ...]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for tag in record["tags"]:
            groups[tag].append(record)
    return groups


def _by_case(records: tuple[dict[str, Any], ...]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["case"]].append(record)
    return groups


def _header(base: Run, new: Run) -> list[str]:
    lines = [f"# Enhancer bench: {base.label} → {new.label}", "", "| | " + base.label + " | " + new.label + " |", "|---|---|---|"]
    for key in ("repo", "revision", "commit", "dirty", "platform", "temperature", "seeds", "cases", "load_seconds"):
        lines.append(f"| {key} | {base.meta.get(key, '–')} | {new.meta.get(key, '–')} |")
    return lines


def _checks_table(base: Run, new: Run) -> list[str]:
    base_rates, new_rates = check_rates(base.records), check_rates(new.records)
    lines = ["", "## Checks", "", f"| Check | {base.label} | {new.label} | Δ points | n |", "|---|---|---|---|---|"]
    for name in CHECK_NAMES:
        (b, bn), (n, nn) = base_rates[name], new_rates[name]
        lines.append(f"| {name} | {_pct(b)} | {_pct(n)} | {_delta(b, n)} | {bn}/{nn} |")
    lines.append(f"| **score** | {_pct(mean_score(list(base.records)))} | {_pct(mean_score(list(new.records)))} | {_delta(mean_score(list(base.records)), mean_score(list(new.records)))} | |")
    base_ratio = [record["checks"]["length_ratio"] for record in base.records if record["checks"].get("length_ratio") is not None]
    new_ratio = [record["checks"]["length_ratio"] for record in new.records if record["checks"].get("length_ratio") is not None]
    if base_ratio or new_ratio:
        median = lambda values: f"{statistics.median(values):.2f}" if values else "–"  # noqa: E731
        lines.append(f"| median length ratio | {median(base_ratio)} | {median(new_ratio)} | | |")
    return lines


def _tags_table(base: Run, new: Run) -> list[str]:
    base_tags, new_tags = _by_tag(base.records), _by_tag(new.records)
    lines = ["", "## Score per tag", "", f"| Tag | {base.label} | {new.label} | Δ points |", "|---|---|---|---|"]
    for tag in sorted(set(base_tags) | set(new_tags)):
        b, n = mean_score(base_tags.get(tag, [])), mean_score(new_tags.get(tag, []))
        lines.append(f"| {tag} | {_pct(b)} | {_pct(n)} | {_delta(b, n)} |")
    return lines


def _speed_table(base: Run, new: Run) -> list[str]:
    lines = ["", "## Speed", "", f"| | {base.label} | {new.label} |", "|---|---|---|"]
    lines.append(f"| median seconds per rewrite | {_median([rewrite_seconds(r) for r in base.records])} | {_median([rewrite_seconds(r) for r in new.records])} |")
    first = lambda run: [a["first_text_seconds"] for r in run.records for a in r["attempts"][:1] if a["first_text_seconds"] is not None]  # noqa: E731
    lines.append(f"| median time to first text | {_median(first(base))} | {_median(first(new))} |")
    return lines


def _record_block(label: str, record: dict[str, Any]) -> list[str]:
    checks = record["checks"]
    failed = [name for name in CHECK_NAMES if checks.get(name) is False or (name == "keep" and checks.get(name) not in (None, 1.0))]
    lines = [f"**{label}** (seed {record['seed']}, score {_pct(case_score(checks))}{', failed: ' + ', '.join(failed) if failed else ''})", ""]
    if record["error"]:
        lines.append(f"Error: {record['error']}")
    if record["scene"]:
        lines += ["Scene:", _fence(record["scene"])]
    lines.append(_fence(record["output"]))
    if record["warnings"]:
        lines.append("Warnings: " + "; ".join(record["warnings"]))
    return lines


def _changed_cases(base: Run, new: Run) -> list[str]:
    base_cases, new_cases = _by_case(base.records), _by_case(new.records)
    shared = sorted(set(base_cases) & set(new_cases))
    deltas = [(abs((mean_score(new_cases[case]) or 0) - (mean_score(base_cases[case]) or 0)), case) for case in shared]
    deltas.sort(key=lambda item: (-item[0], item[1]))
    lines = ["", f"## Cases that changed most (top {_TOP_CHANGED})"]
    for delta, case in deltas[:_TOP_CHANGED]:
        b_record, n_record = base_cases[case][0], new_cases[case][0]
        lines += ["", f"### {case} ({_pct(mean_score(base_cases[case]))} → {_pct(mean_score(new_cases[case]))})", ""]
        lines += ["Input:", _fence((b_record["input"]["scene"] + "\n" if b_record["input"]["scene"] else "") + b_record["input"]["prompt"]), ""]
        lines += _record_block(base.label, b_record) + [""] + _record_block(new.label, n_record)
        if b_record["system_prompt"] != n_record["system_prompt"]:
            diff = difflib.unified_diff(b_record["system_prompt"].splitlines(), n_record["system_prompt"].splitlines(), base.label, new.label, lineterm="")
            lines += ["", "System prompt diff:", "```diff", *diff, "```"]
    return lines


def compare_runs(base: Run, new: Run) -> str:
    """Return the Markdown report comparing *base* with *new*."""
    lines = _header(base, new) + _checks_table(base, new) + _tags_table(base, new) + _speed_table(base, new) + _changed_cases(base, new)
    return "\n".join(lines) + "\n"
