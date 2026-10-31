"""Command line for the transparent-background bench.

Two environments are involved. ``render`` and ``pair-render`` drive the app's image CLI and run in the app's
environment. ``matte`` and ``sheet`` need the matting libraries (rembg with onnxruntime-gpu, transformers, pymatting,
opencv) and run in a separate environment, so the app's dependencies stay as they are::

    python -m bench.transparent render --model zit --background grey
    python -m bench.transparent render --model klein4b --background plain --tilt white:1.0 --folder tiltw-1.0
    python -m bench.transparent pair-render
    <bench python> -m bench.transparent matte
    <bench python> -m bench.transparent sheet
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

CASES_PATH = Path(__file__).with_name("cases.yaml")
RUNS_DIR = Path(__file__).with_name("runs")
SEED = 73914826
PAIR_MODEL = "klein4b"
PAIR_EDIT = "Change only the background to pure solid black. Keep the subject exactly the same: same pose, shape, colours, lighting and every strand and edge."
KEY_BACKGROUNDS = ("green", "magenta", "blue", "cyan")
# A white/black pair is two render folders of PAIR_MODEL: the edit pair, the same seed with two prompts, and the
# noise-tilt pairs (tiltw-S / tiltk-S, one per tilt strength S).
FIXED_PAIRS = {"edit": ("white", "black"), "prompt": ("white", "blackp")}
HYBRID_TOLERANCE = 0.03


def load_cases(path: Path = CASES_PATH) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(backgrounds, subjects)``: prompt suffix by background name and subject prompt by id."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(data["backgrounds"]), dict(data["subjects"])


def prompts_file_text(subjects: dict[str, str], suffix: str) -> str:
    """Return a prompt file with one set per subject, so one image-CLI run renders every subject with one model load."""
    return yaml.safe_dump({name: [{"prompt": prompt + suffix}] for name, prompt in subjects.items()}, sort_keys=False, width=10_000)


def find_render(folder: Path, name: str) -> Path | None:
    """Return the newest image the CLI wrote for set *name* (``{name}_{timestamp}.png``) in *folder*, or None."""
    matches = sorted(path for path in folder.glob(f"{name}_*.png") if not path.stem.endswith("_pre"))
    return matches[-1] if matches else None


def _ziv_image() -> str:
    return str(Path(sys.executable).with_name("ziv-image"))


def _base_args(model: str, quantize: int | None, out_dir: Path) -> list[str]:
    args = [_ziv_image(), "-m", model, "--seed", str(SEED), "--ratio", "1:1", "-s", "m", "-o", str(out_dir), "--no-enhance", "--no-sharpen"]
    return args + (["-q", str(quantize)] if quantize else [])


def _cmd_render(args: argparse.Namespace) -> int:
    backgrounds, subjects = load_cases()
    if args.subjects:
        subjects = {name: subjects[name] for name in args.subjects.split(",")}
    out_dir = args.runs_dir / "renders" / args.model / (args.folder or args.background)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = out_dir / "prompts.yaml"
    prompts.write_text(prompts_file_text(subjects, backgrounds[args.background]), encoding="utf-8")
    command = [
        *_base_args(args.model, args.quantize, out_dir),
        "-p",
        str(prompts),
        *(["--upscale", str(args.upscale)] if args.upscale else []),
        *(["-W", str(args.pixels), "-H", str(args.pixels)] if args.pixels else []),
    ]
    if not args.tilt:
        return subprocess.call(command)
    return subprocess.call([sys.executable, "-m", "bench.transparent.tilt", *command[1:]], env={**os.environ, "ZIV_BENCH_TILT": args.tilt})


def _cmd_pair_render(args: argparse.Namespace) -> int:
    _, subjects = load_cases()
    white_dir = args.runs_dir / "renders" / PAIR_MODEL / "white"
    black_dir = args.runs_dir / "renders" / PAIR_MODEL / "black"
    black_dir.mkdir(parents=True, exist_ok=True)
    for name in subjects:
        white = find_render(white_dir, name)
        if white is None:
            print(f"Skipping {name}: no white render in {white_dir}", file=sys.stderr)
            continue
        if find_render(black_dir, name) is not None:
            continue
        scratch = black_dir / ".tmp"
        scratch.mkdir(exist_ok=True)
        code = subprocess.call([*_base_args(PAIR_MODEL, None, scratch), "--reference", str(white), "--prompt", PAIR_EDIT])
        if code:
            return code
        for path in scratch.glob("*.png"):
            path.rename(black_dir / f"{name}_{path.name}")
    return 0


def _read_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _read_alpha(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).getchannel("A"), dtype=np.float32) / 255.0


def _matte_one(rgb: np.ndarray, method: str, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, float]:
    from bench.transparent.methods import METHODS

    start = time.perf_counter()
    alpha, foreground = METHODS[method](rgb, mask)
    return alpha, foreground, time.perf_counter() - start


def _write(out: Path, alpha: np.ndarray, foreground: np.ndarray) -> None:
    from bench.transparent.sheet import to_rgba

    out.parent.mkdir(parents=True, exist_ok=True)
    to_rgba(alpha, foreground).save(out)


def pairs(model_dir: Path) -> dict[str, tuple[str, str]]:
    """Return the white/black pairs rendered for a model, as ``{name: (light folder, dark folder)}``."""
    folders = {path.name for path in model_dir.iterdir() if path.is_dir()} if model_dir.is_dir() else set()
    found = {name: pair for name, pair in FIXED_PAIRS.items() if set(pair) <= folders}
    for folder in sorted(folders):
        if folder.startswith("tiltw-") and (dark := "tiltk-" + folder.removeprefix("tiltw-")) in folders:
            found["tilt-" + folder.removeprefix("tiltw-")] = (folder, dark)
    return found


def applies(method: str, background: str, model_methods: tuple[str, ...]) -> bool:
    """Return whether *method* is meant for renders in the *background* folder."""
    light = background == "white" or background.startswith("tiltw-")
    dark = background in ("black", "blackp") or background.startswith("tiltk-")
    if method == "chroma":
        return background in KEY_BACKGROUNDS
    if method in ("ink", "ink-solid"):
        return light
    if method == "screen":
        return dark
    if method == "birefnet-hr-matting+fg":
        return background == "grey-up2"
    return method in model_methods and (background.startswith("grey") or background == "green" or light)


def _cmd_matte(args: argparse.Namespace) -> int:
    from bench.transparent.alpha import border_colour
    from bench.transparent.checks import matte_checks
    from bench.transparent.methods import CPU_METHODS, KEY_METHODS, MASK_SOURCE, MODEL_METHODS, release_models

    _, subjects = load_cases()
    renders, mattes = args.runs_dir / "renders", args.runs_dir / "mattes"
    methods = list(MODEL_METHODS) if args.methods is None else [part for part in args.methods.split(",") if part]
    results = mattes / "results.jsonl"
    results.parent.mkdir(parents=True, exist_ok=True)
    sources = [
        (model_dir.name, background.name, name, source)
        for model_dir in sorted(path for path in renders.iterdir() if path.is_dir())
        for background in sorted(path for path in model_dir.iterdir() if path.is_dir())
        for name in subjects
        if (source := find_render(background, name)) is not None
    ]
    with results.open("a", encoding="utf-8") as log:

        def emit(record: dict) -> None:
            log.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)

        # One method at a time over every render, each freed before the next loads: two models at once overflow 10 GB.
        for method in [*methods, *KEY_METHODS]:
            for model, background, name, source in sources:
                out = mattes / model / background / method / f"{name}.png"
                if not applies(method, background, MODEL_METHODS + CPU_METHODS) or (out.exists() and not args.force):
                    continue
                base = MASK_SOURCE.get(method)
                base_path = mattes / model / background / str(base) / f"{name}.png"
                if base and not base_path.exists():
                    print(f"Skipping {method} for {model}/{background}/{name}: no {base} mask yet", file=sys.stderr)
                    continue
                rgb = _read_rgb(source)
                mask = _read_alpha(base_path) if base else None
                alpha, foreground, seconds = _matte_one(rgb, method, mask)
                _write(out, alpha, foreground)
                emit({"model": model, "background": background, "method": method, "subject": name, "seconds": round(seconds, 3), **matte_checks(alpha, foreground, border_colour(rgb))})
            release_models()
        for record in _pair_mattes(renders / PAIR_MODEL, mattes / PAIR_MODEL, subjects):
            emit(record)
        for record in _auto_key(renders / PAIR_MODEL, mattes / PAIR_MODEL, subjects):
            emit(record)
    return 0


def _pair_mattes(renders: Path, mattes: Path, subjects: dict[str, str]) -> list[dict]:
    """Write the difference and hybrid mattes of every pair, and return their records with the subject drift."""
    from bench.transparent.alpha import border_colour, difference_matte, hybrid_alpha, subject_drift
    from bench.transparent.checks import matte_checks
    from bench.transparent.methods import foreground_for

    records = []
    for pair, (light, dark) in pairs(renders).items():
        for name in subjects:
            white, black = find_render(renders / light, name), find_render(renders / dark, name)
            core_path = mattes / light / "birefnet" / f"{name}.png"
            if white is None or black is None:
                continue
            on_white, on_black = _read_rgb(white), _read_rgb(black)
            if on_white.shape != on_black.shape:
                print(f"Skipping pair {pair}/{name}: sizes differ", file=sys.stderr)
                continue
            alpha, foreground, residual = difference_matte(on_white, on_black)
            out_dir = mattes / f"pair-{pair}"
            _write(out_dir / "difference" / f"{name}.png", alpha, foreground)
            base = {"model": PAIR_MODEL, "background": f"pair-{pair}", "subject": name, "residual": float(residual.mean())}
            records.append({**base, "method": "difference", **matte_checks(alpha, foreground, border_colour(on_white))})
            if not core_path.exists():
                continue
            core = _read_alpha(core_path)
            mixed = hybrid_alpha(core, alpha, residual, tolerance=HYBRID_TOLERANCE)
            trusted = (residual <= HYBRID_TOLERANCE)[..., None]
            colour = np.where(trusted, foreground, foreground_for(on_white, mixed))
            _write(out_dir / "hybrid" / f"{name}.png", mixed, colour)
            drift = subject_drift(on_white, on_black, core)
            records.append({**base, "method": "hybrid", "drift": drift, "trusted": float(trusted.mean()), **matte_checks(mixed, colour, border_colour(on_white))})
    return records


def _auto_key(renders: Path, mattes: Path, subjects: dict[str, str]) -> list[dict]:
    """Pick each subject's key colour from its grey render and copy that key's chroma matte to ``auto/chroma-auto``."""
    import shutil

    from bench.transparent.alpha import KEYS, key_conflict

    records = []
    for name in subjects:
        grey, mask_path = find_render(renders / "grey", name), mattes / "grey" / "birefnet" / f"{name}.png"
        keyed = {key: mattes / key / "chroma" / f"{name}.png" for key in KEY_BACKGROUNDS if (mattes / key / "chroma" / f"{name}.png").exists()}
        if grey is None or not mask_path.exists() or len(keyed) < 2:
            continue
        rgb, mask = _read_rgb(grey), _read_alpha(mask_path)
        conflicts = {key: key_conflict(rgb, mask, KEYS[key]) for key in keyed}
        chosen = min(conflicts, key=conflicts.__getitem__)
        out = mattes / "auto" / "chroma-auto" / f"{name}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(keyed[chosen], out)
        records.append({"model": PAIR_MODEL, "background": "auto", "method": "chroma-auto", "subject": name, "chosen": chosen, "conflicts": conflicts})
    return records


def _source_folder(background: str) -> str:
    """Return the render folder whose image heads a sheet row for mattes in *background*."""
    if background == "auto":
        return "grey"
    if background.startswith("pair-"):
        pair = background.removeprefix("pair-")
        return FIXED_PAIRS[pair][0] if pair in FIXED_PAIRS else "tiltw-" + pair.removeprefix("tilt-")
    return background


def _cmd_sheet(args: argparse.Namespace) -> int:
    from bench.transparent.sheet import contact_sheet

    _, subjects = load_cases()
    renders, mattes, sheets = args.runs_dir / "renders", args.runs_dir / "mattes", args.runs_dir / "sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    box = tuple(float(part) for part in args.crop.split(",")) if args.crop else None
    for model_dir in sorted(path for path in mattes.iterdir() if path.is_dir()):
        for background_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            methods = sorted(path.name for path in background_dir.iterdir() if path.is_dir())
            source_dir = renders / model_dir.name / _source_folder(background_dir.name)
            rows = []
            for name in subjects:
                paths = [find_render(source_dir, name)] + [_existing(background_dir / method / f"{name}.png") for method in methods]
                if any(paths[1:]):
                    rows.append((name, paths))
            if not rows:
                continue
            sheet = contact_sheet(rows, ["render", *methods], size=args.size, box=box)
            suffix = "-crop" if box else ""
            out = sheets / f"{model_dir.name}-{background_dir.name}{suffix}.jpg"
            sheet.save(out, quality=88)
            print(f"Wrote {out}")
    return 0


def _existing(path: Path) -> Path | None:
    return path if path.exists() else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bench.transparent", description="Compare transparent-background methods.")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="command", required=True)
    render = commands.add_parser("render", help="Render every subject with one model on one background (loads an image model).")
    render.add_argument("--model", required=True, help="Image model alias, e.g. zit, klein4b.")
    render.add_argument("--background", required=True, help="Background name from cases.yaml, e.g. grey, green, magenta, white, black or plain.")
    render.add_argument("--quantize", type=int, default=None, help="Quantize level passed to the image CLI (4 or 8).")
    render.add_argument("--folder", default=None, help="Render folder name (default: the background name).")
    render.add_argument("--subjects", default=None, help="Comma-separated subject ids (default: all).")
    render.add_argument("--upscale", type=int, default=None, help="Upscale factor passed to the image CLI (2 or 4).")
    render.add_argument("--pixels", type=int, default=None, help="Square output size in pixels (default: size m, 1024).")
    render.add_argument("--tilt", default=None, help="Tilt klein's starting noise, e.g. white:1.0 or black:1.0 (see tilt.py).")
    commands.add_parser("pair-render", help=f"Edit each {PAIR_MODEL} white render onto black with the same seed (loads {PAIR_MODEL}).")
    matte = commands.add_parser("matte", help="Run the matte methods over every render and log the checks (bench environment).")
    matte.add_argument("--methods", default=None, help="Comma-separated model methods; '' for none (default: all). chroma and the pair always run where they apply.")
    matte.add_argument("--force", action="store_true", help="Redo mattes that already exist.")
    sheet = commands.add_parser("sheet", help="Write one contact sheet per model and background.")
    sheet.add_argument("--size", type=int, default=256, help="Tile size in pixels (default 256).")
    sheet.add_argument("--crop", default=None, help="Crop every image to LEFT,TOP,RIGHT,BOTTOM fractions first, for edge detail.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the bench command line."""
    args = _parser().parse_args(argv)
    commands = {"render": _cmd_render, "pair-render": _cmd_pair_render, "matte": _cmd_matte, "sheet": _cmd_sheet}
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
