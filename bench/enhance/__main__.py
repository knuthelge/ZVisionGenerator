"""Command line for the enhancer bench: ``run`` a model over the cases, ``compare`` two runs."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from bench.enhance.cases import load_cases, select_cases
from bench.enhance.report import compare_runs, load_run

RUNS_DIR = Path(__file__).with_name("runs")
_LABEL_RE = re.compile(r"^[\w.-]+$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bench.enhance", description="Measure what the prompt enhancer writes.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run the enhancer over the cases and write runs/LABEL/.")
    run.add_argument("--label", required=True, help="Name of the run folder (letters, digits, '.', '_', '-').")
    run.add_argument("--model", default="fast", help="fast (default), smart, or REPO[@REVISION].")
    run.add_argument("--tags", default=None, help="Comma-separated tags; run only cases with any of them.")
    run.add_argument("--cases", default=None, help="Comma-separated case ids; run only these.")
    run.add_argument("--seeds", type=int, default=3, help="Seeds per case (default 3).")
    run.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help=argparse.SUPPRESS)
    rescore = commands.add_parser("rescore", help="Recompute the checks of stored runs from their outputs (after changing checks or expectations).")
    rescore.add_argument("labels", nargs="+", help="Labels (or folders) of the runs to rescore.")
    rescore.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help=argparse.SUPPRESS)
    sheet = commands.add_parser("sheet", help="Render the rewrites of image cases as images, side by side per run (loads an image model).")
    sheet.add_argument("runs", nargs="+", help="Labels (or folders) of the runs, one column each after the user's prompt.")
    sheet.add_argument("--label", required=True, help="Name of the sheet folder under runs/.")
    sheet.add_argument("--tags", default=None, help="Comma-separated tags; only cases with any of them.")
    sheet.add_argument("--cases", default=None, help="Comma-separated case ids.")
    sheet.add_argument("--model", default="zit", help="Image model alias (default zit).")
    sheet.add_argument("--seed", type=int, default=73914826, help="Seed for every image (default 73914826).")
    sheet.add_argument("--ratio", default="2:3", help="Aspect ratio (default 2:3).")
    sheet.add_argument("--size", default="s", help="Size preset (default s).")
    sheet.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help=argparse.SUPPRESS)
    compare = commands.add_parser("compare", help="Write a Markdown report comparing two runs.")
    compare.add_argument("base", help="Label (or folder) of the baseline run.")
    compare.add_argument("new", help="Label (or folder) of the run to compare.")
    compare.add_argument("--out", type=Path, default=None, help="Report path (default runs/compare-BASE-NEW.md).")
    compare.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help=argparse.SUPPRESS)
    return parser


def _split(value: str | None) -> set[str] | None:
    return {part.strip() for part in value.split(",") if part.strip()} if value else None


def _run_dir(runs_dir: Path, name: str) -> Path:
    path = Path(name)
    return path if path.is_dir() else runs_dir / name


def _cmd_run(args: argparse.Namespace) -> int:
    from zvisiongenerator.backends import create_prompt_enhancer
    from zvisiongenerator.backends.prompt_enhancer_session import ensure_available, smart_budget_bytes
    from zvisiongenerator.utils.config import load_config, resolve_enhancer_model
    from zvisiongenerator.utils.prompt_enhance import enhance_options

    from bench.enhance.runner import run_bench

    if not _LABEL_RE.match(args.label):
        raise SystemExit(f"Invalid --label '{args.label}'.")
    if args.seeds < 1:
        raise SystemExit("--seeds must be at least 1.")
    cases = select_cases(load_cases(), tags=_split(args.tags), ids=_split(args.cases))
    if not cases:
        raise SystemExit("No cases match --tags/--cases.")
    config = load_config()
    repo, revision = resolve_enhancer_model(config, platform_key=sys.platform, cli_model=args.model, budget_bytes=smart_budget_bytes(args.model))
    ensure_available(repo, revision)
    results = run_bench(
        cases,
        factory=create_prompt_enhancer,
        repo=repo,
        revision=revision,
        model_choice=args.model,
        config=config,
        options=enhance_options(config),
        seeds=args.seeds,
        out_dir=args.runs_dir / args.label,
    )
    print(f"Wrote {results}")
    return 0


def _cmd_rescore(args: argparse.Namespace) -> int:
    from zvisiongenerator.utils.config import load_config
    from zvisiongenerator.utils.prompt_enhance import enhance_options

    from bench.enhance.runner import rescore_run

    config = load_config()
    cases = load_cases()
    for label in args.labels:
        count = rescore_run(_run_dir(args.runs_dir, label), cases, config=config, options=enhance_options(config))
        print(f"Rescored {count} results in {label}")
    return 0


def _cmd_sheet(args: argparse.Namespace) -> int:
    from bench.enhance.sheet import build_sheet

    if not _LABEL_RE.match(args.label):
        raise SystemExit(f"Invalid --label '{args.label}'.")
    runs = tuple(load_run(_run_dir(args.runs_dir, name)) for name in args.runs)
    cases = tuple(case.id for case in select_cases(load_cases(), tags=_split(args.tags), ids=_split(args.cases)))
    sheet = build_sheet(runs, cases, args.runs_dir / args.label, model=args.model, seed=args.seed, ratio=args.ratio, size=args.size)
    print(f"Wrote {sheet}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    base, new = load_run(_run_dir(args.runs_dir, args.base)), load_run(_run_dir(args.runs_dir, args.new))
    out = args.out or args.runs_dir / f"compare-{base.label}-{new.label}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(compare_runs(base, new), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the bench command line."""
    args = _parser().parse_args(argv)
    commands = {"run": _cmd_run, "rescore": _cmd_rescore, "sheet": _cmd_sheet, "compare": _cmd_compare}
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
