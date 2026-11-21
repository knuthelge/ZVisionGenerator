"""Command line for the memory bench.

``run`` renders the cases of one medium's ``cases/<media>.yaml`` through the app's own CLI (``ziv-image``, ``ziv-video`` or ``ziv-audio``), each in
a fresh process, and records each job's peak memory, the driver's view of the card and, on a Mac, the process's footprint. ``--budget-gib`` makes the app see a smaller card, to run the
placement a smaller card would get; ``--hold-in-process-gib`` takes that much GPU memory in each case's own process, as another program on the card would. ``fit`` judges a results
file against the engine's estimates (the calibration rule) and exits non-zero when the rule fails. ``one`` is what ``run`` starts per case.
See ``docs/developer/memory-bench.md``. Run one family per job::

    uv run python -m bench.memory run --media image --family zimage --label cuda
    uv run python -m bench.memory fit bench/memory/runs/image/cuda/results.jsonl
    uv run python -m bench.memory run --media image --label mac        # macOS: every family

If your machine runs GPU jobs through a queue, run the ``run`` command there.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

from bench.memory.cases import MEDIA, expand_cases, load_cases
from bench.memory.fit import GIB, allocated_rows, format_probes, format_report, format_rule, hold_brackets, on_this_system, reestimate, rule_checks, size_terms, with_brackets
from bench.memory.holds import hold_in_process
from bench.memory.measure import platform_label, run_case
from bench.memory.media import medium
from bench.memory.probes import show_machine_of

RUNS_DIR = Path(__file__).with_name("runs")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIMEOUT_SECONDS = 3600
LORA_ENV = "ZIV_BENCH_LORA"
"""The environment variable that names the LoRA file a case marked ``lora`` runs with, when ``run --lora`` is not given."""


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Return the rows of the results file *path*, or none when it does not exist yet."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _spawn(case: dict[str, Any], run_dir: Path, timeout: int) -> dict[str, Any]:
    """Run *case* in a fresh process and return its row; a process that dies (killed for memory, or timed out) is a failed row."""
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / "row.json"
        command = [sys.executable, "-m", "bench.memory", "one", json.dumps(case), "--run-dir", str(run_dir), "--out", str(out)]
        try:
            code = subprocess.run(command, cwd=REPO_ROOT, timeout=timeout, check=False).returncode
            reason = f"the process exited with {code} without a result"
        except subprocess.TimeoutExpired:
            code, reason = None, f"timed out after {timeout} s"
        if out.exists():
            return json.loads(out.read_text(encoding="utf-8"))
    return {key: value for key, value in case.items() if key != "prompt"} | {"platform": platform_label(), "os": sys.platform, "status": "failed", "error": reason, "platform_peak_bytes": None}


def _write_meta(run_dir: Path, args: argparse.Namespace) -> None:
    """Record the run's settings: commit, platform, library versions and arguments. No host name, user or device serial."""
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT, check=False).stdout.strip()
    versions = {}
    for package in ("torch", "mlx", "diffusers", "mflux"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    meta = {
        "commit": commit,
        "platform": platform_label(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "versions": versions,
        "args": vars(args) | {"func": None},
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def _key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Return what makes a result its own: the case and the card it ran on (a hold in the job's own process or, in an older row, in another one, a budget or a total reruns a case)."""
    return row["id"], row.get("hold_gib"), row.get("hold_in_process_gib"), row.get("budget_gib"), row.get("total_gib")


def _latest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return *rows* with only the last of each case (``run --force`` appends, so a rerun leaves its older rows in the file), in the order of those last rows."""
    latest = {_key(row): position for position, row in enumerate(rows)}
    return [row for position, row in enumerate(rows) if latest[_key(row)] == position]


def _attach_lora(case: dict[str, Any], lora: str | None, *, dry_run: bool) -> bool:
    """Give a case marked ``lora`` the file *lora* (or ``ZIV_BENCH_LORA``) names, and its size; return whether the case can run.

    A case without a file cannot run. A dry run lists it with a placeholder.

    Raises:
        FileNotFoundError: The named file does not exist.
    """
    path = lora or os.environ.get(LORA_ENV)
    if not path:
        if dry_run:
            case["lora_path"] = "<lora file>"
        return dry_run
    file = Path(path).expanduser()
    if not file.is_file():
        raise FileNotFoundError(f"The LoRA file for case {case['id']} does not exist: {file}")
    case["lora_path"], case["lora_bytes"] = str(file), file.stat().st_size
    return True


def _cmd_run(args: argparse.Namespace) -> int:
    from zvisiongenerator.utils.config import load_config

    if args.total_gib and platform_label() != "mac":
        raise SystemExit("--total-gib needs a Mac: a CUDA card keeps this machine's system memory.")
    if args.hold_in_process_gib and platform_label() == "mac":
        raise SystemExit("--hold-in-process-gib needs a CUDA card.")
    if args.budget_gib and platform_label() == "mac":
        raise SystemExit("--budget-gib needs a CUDA card: a Mac is judged against its total memory (--total-gib).")
    module = medium(args.media)
    cases = load_cases(args.media)
    chosen = expand_cases(cases, media=args.media, platform=platform_label(), families=args.family, ids=args.case)
    config = load_config()
    run_dir = RUNS_DIR / args.media / (args.label or platform_label())
    results = run_dir / "results.jsonl"
    done = {_key(row) for row in read_rows(results) if row["status"] == "ok" and not row.get("downloaded")}
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_meta(run_dir, args)
    for case in chosen:
        if args.hold_in_process_gib:
            case["hold_in_process_gib"] = args.hold_in_process_gib
        if args.budget_gib:
            case["budget_gib"] = args.budget_gib
        if args.total_gib:
            case["total_gib"] = args.total_gib
        if args.steps is not None:
            case["steps"] = args.steps
        if case.get("lora") and not _attach_lora(case, args.lora, dry_run=args.dry_run):
            print(f"{case['id']}: needs a LoRA file (run --lora PATH, or set {LORA_ENV}), skipped")
            continue
        module.prepare(case, cases, config)
        if args.dry_run:
            size = f"{case['width']}x{case['height']}  " if case.get("width") else ""
            print(f"{case['id']}: {size}ziv-{args.media} {' '.join(module.case_argv(case, run_dir))}")
            continue
        if _key(case) in done and not args.force:
            print(f"{case['id']}: done, skipped")
            continue
        module.ensure_sources(case, run_dir, cases)
        print(f"{case['id']} ...", flush=True)
        row = _measure(case, run_dir, results, args.timeout)
        footprint = f", footprint {row['footprint_peak_bytes'] / 1024**3:.2f} GiB" if row.get("footprint_peak_bytes") else ""
        print(
            f"{case['id']}: {row['status']}, peak {(row.get('platform_allocated_peak_bytes') or row.get('platform_peak_bytes') or 0) / 1024**3:.2f} GiB{footprint}, {row.get('seconds')} s{', ' + row['error'] if row.get('error') else ''}",
            flush=True,
        )
    return 0


def _measure(case: dict[str, Any], run_dir: Path, results: Path, timeout: int) -> dict[str, Any]:
    """Run *case*, append its row to *results* and return the row that counts.

    A case that downloaded or built a model while it ran (``downloaded``) overstates its peak, so it runs once more with the model in place and
    that row, appended last, is the case's result. The first row stays in the file with ``downloaded`` set.
    """
    row = _spawn(case, run_dir, timeout)
    row.pop("lora_path", None)  # the results record the LoRA's size, never where it lives
    _append_row(results, row)
    if row["status"] != "ok" or not row.get("downloaded"):
        return row
    print(f"{case['id']}: fetched or built a model while it ran, measuring again with it in place", flush=True)
    row = _spawn(case, run_dir, timeout)
    row.pop("lora_path", None)
    _append_row(results, row)
    return row


def _cmd_one(args: argparse.Namespace) -> int:
    case = json.loads(args.case)
    module = medium(case["media"])
    probes = module.install_probes(case)
    if case.get("budget_gib") or case.get("total_gib"):
        show_machine_of(case.get("budget_gib"), case.get("total_gib"))
    run_dir = Path(args.run_dir)
    hold = None if not case.get("hold_in_process_gib") else lambda: hold_in_process(case["hold_in_process_gib"])
    row = run_case(case, module.case_argv(case, run_dir), module.run_cli, run_dir / "out" / case["id"], estimate=module.estimate, outputs=module.OUTPUT_SUFFIXES, probes=probes, hold=hold)
    Path(args.out).write_text(json.dumps(row), encoding="utf-8")
    return 0


def _cmd_ui(args: argparse.Namespace) -> int:
    """Apply the budget in this process, then start the Web UI as ``ziv ui`` does with the remaining arguments."""
    from zvisiongenerator.web import main as web_main

    if "--background" in args.extra:
        raise SystemExit("--background starts the server in another process, which would not see the simulated machine.")
    try:
        show_machine_of(args.budget_gib, args.total_gib)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    web_main(args.extra, prog="ziv ui")
    return 0


def _estimate_for(row: dict[str, Any]) -> dict[str, Any] | None:
    return medium(row.get("media", "image")).estimate(row)


def _cmd_fit(args: argparse.Namespace) -> int:
    rows = [row for path in args.results for row in _latest_rows(read_rows(Path(path)))]
    mac = {"mac_unified_bytes": round(args.total_gib * GIB)} if args.total_gib else {}
    # The budget the run graded against, kept before a fresh estimate replaces the row's (it is this machine's budget, not the run's): the cards the rows had are counted in it.
    rows = [row | {"run_budget_bytes": (row.get("estimate") or {}).get("budget_bytes")} for row in rows]
    if not args.stored:
        elsewhere = [row for row in rows if not on_this_system(row)]
        if elsewhere:
            print(
                f"{len(elsewhere)} rows measured on another operating system ({', '.join(sorted({str(row.get('os')) for row in elsewhere}))}) are judged on their stored estimates: this machine's cannot stand for them.",
                file=sys.stderr,
            )
        estimate_for = _estimate_for if not args.total_gib else lambda row: _estimate_for(row | mac)
        rows = reestimate(rows, lambda row: estimate_for(row) if on_this_system(row) else None)
    if args.json:
        terms = size_terms(allocated_rows(rows))
        print(
            json.dumps(
                {f"{family}/{platform_}/q{quantize}": term for (family, platform_, quantize), term in sorted(terms.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or 0))}, indent=2
            )
        )
        return 0
    rows = with_brackets(rows)
    checks, left_out = rule_checks(rows)
    print(format_rule(rows, checks, left_out) + format_probes(hold_brackets(rows)))
    if args.terms:
        print("\n" + format_report([row for row in rows if row.get("media", "image") == "image"]))
    return 1 if any(not check["ok"] for check in checks) else 0


def main(argv: list[str] | None = None) -> int:
    """Run the bench command named in *argv* and return its exit code (``fit`` returns 1 when the calibration rule fails)."""
    parser = argparse.ArgumentParser(prog="python -m bench.memory")
    commands = parser.add_subparsers(required=True)

    run = commands.add_parser("run", help="Run the cases, each in a fresh process through the medium's CLI, and record their peaks.")
    run.add_argument("--media", choices=MEDIA, default="image", help="The medium to run (default: image).")
    run.add_argument("--family", action="append", help="A family to run (repeat for more; default: all).")
    run.add_argument("--case", action="append", help="A case id to run (repeat for more; default: all).")
    run.add_argument("--label", help="Results folder under runs/<media>/ (default: the platform, cuda or mac).")
    run.add_argument("--steps", type=int, help="Steps for every case instead of the case file's.")
    run.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="Seconds a case may run before it is stopped and recorded as failed.")
    run.add_argument(
        "--hold-in-process-gib",
        type=float,
        help="Take this many GiB of GPU memory in each case's own process before the job, through the CUDA driver API and outside PyTorch's allocator, as another program on the card would. The row records hold_charged_bytes.",
    )
    run.add_argument(
        "--budget-gib",
        type=float,
        help="CUDA: make the app read a card of this many GiB for every case, so it picks the placement and gives the verdict that card would (the run still uses this machine's card).",
    )
    run.add_argument(
        "--total-gib", type=float, help="Mac: make the app read this many GiB of unified memory for every case, so it gives the verdict that Mac would (the run still uses this machine's memory)."
    )
    run.add_argument("--lora", help=f"The LoRA file the cases marked `lora` run with (default: ${LORA_ENV}); those cases are skipped without one. It must suit the family you run.")
    run.add_argument("--force", action="store_true", help="Run cases that already have an ok result again.")
    run.add_argument("--dry-run", action="store_true", help="Print each case's size and CLI arguments; run nothing and write nothing.")
    run.set_defaults(func=_cmd_run)

    one = commands.add_parser("one", help="Run one case in this process (what run starts per case).")
    one.add_argument("case", help="The expanded case as JSON.")
    one.add_argument("--run-dir", required=True)
    one.add_argument("--out", required=True, help="File to write the case's row to.")
    one.set_defaults(func=_cmd_one)

    ui = commands.add_parser("ui", help="Start the Web UI as a smaller machine: the app reads this card or total memory for its memory tags and verdicts. Other arguments go to `ziv ui`.")
    ui.add_argument("--budget-gib", type=float, help="CUDA: the card's GiB the app reads.")
    ui.add_argument("--total-gib", type=float, help="Mac: the total unified memory the app reads.")
    ui.set_defaults(func=_cmd_ui)

    fit = commands.add_parser("fit", help="Judge results files against the engine's estimates and exit 1 when the calibration rule fails.")
    fit.add_argument("results", nargs="+", help="One or more results.jsonl files from run.")
    fit.add_argument("--stored", action="store_true", help="Use the estimates stored in the rows instead of asking the engine now (run fit on the machine that made the rows, or the cards differ).")
    fit.add_argument(
        "--total-gib",
        type=float,
        help="Judge macOS rows against a Mac with this much total memory in GiB, on any machine (the weights come from the shipped table when the model is not downloaded here).",
    )
    fit.add_argument("--terms", action="store_true", help="Also print the image size terms, and the edit and upscale checks.")
    fit.add_argument("--json", action="store_true", help="Print only the image size terms, as JSON (fitted to the allocated peak of the CUDA rows that recorded it).")
    fit.set_defaults(func=_cmd_fit)

    args, extra = parser.parse_known_args(argv)
    if extra and args.func is not _cmd_ui:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    args.extra = extra
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
