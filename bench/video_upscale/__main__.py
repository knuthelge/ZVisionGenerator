"""Command line for the video-upscale spike (macOS).

``render`` drives the app's video CLI to make the source clips of ``cases.yaml``. ``run`` upscales them with LTX's
stage 2 (and a bicubic baseline) and records time and peak memory per phase. ``sheet`` writes a blind review page::

    uv run python -m bench.video_upscale render
    uv run python -m bench.video_upscale run --label steps --steps 1,2,3 --bicubic
    uv run python -m bench.video_upscale run --label iclora --iclora 1.0
    uv run python -m bench.video_upscale run --label prompt --clips face,walk5 --steps 2 --empty-prompt
    uv run python -m bench.video_upscale run --label tiles --clips walk10 --steps 2 --tiles 2x1x1
    uv run python -m bench.video_upscale encode-check --clips medium5,walk10
    uv run python -m bench.video_upscale sheet steps
    uv run python -m bench.video_upscale review          # http://<this Mac>:8765/ on the local network
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import yaml

CASES_PATH = Path(__file__).with_name("cases.yaml")
RUNS_DIR = Path(__file__).with_name("runs")
SOURCES_DIR = RUNS_DIR / "sources"
# Lightricks' 2x Pixel Spatial Upscaler IC-LoRA for LTX-2.3 (gated: accept the terms and log in to Hugging Face).
ICLORA_REPO = "Lightricks/LTX-2.3-22b-IC-LoRA-Pixel-Spatial-Upscaler"
ICLORA_FILE = "ltx-2.3-22b-ic-lora-pixel-spatial-upscaler-x2-0.9.safetensors"


def load_cases(path: Path = CASES_PATH) -> dict[str, Any]:
    """Return the bench cases: ``model``, ``seed`` and ``clips`` by id."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _select(clips: dict[str, Any], names: str | None) -> dict[str, Any]:
    if not names:
        return clips
    return {name: clips[name] for name in names.split(",")}


def _ziv_video() -> str:
    return str(Path(sys.executable).with_name("ziv-video"))


def _run_cli(arguments: list[str], out_dir: Path) -> Path:
    """Run the video CLI with *arguments* writing into *out_dir* and return the one new video it wrote."""
    before = set(out_dir.glob("*.mp4"))
    subprocess.run([_ziv_video(), *arguments, "-o", str(out_dir), "--no-enhance"], check=True)
    new = sorted(set(out_dir.glob("*.mp4")) - before)
    if len(new) != 1:
        raise RuntimeError(f"Expected one new video in {out_dir}, found {[path.name for path in new]}.")
    return new[0]


def _cmd_render(args: argparse.Namespace) -> int:
    cases = load_cases()
    for name, case in _select(cases["clips"], args.clips).items():
        target = SOURCES_DIR / f"{name}.mp4"
        if target.exists() and not args.force:
            print(f"{name}: {target} exists, skipped")
            continue
        work = SOURCES_DIR / name
        work.mkdir(parents=True, exist_ok=True)
        seed = str(cases["seed"])
        clip = _run_cli(["-m", cases["model"], "--seed", seed, "-s", case["size"], "--frames", str(case["frames"]), "--scene", case["scene"], "--prompt", case["prompt"]], work)
        for extend in case.get("extends", []):
            clip = _run_cli(["--extend", str(clip), "--extend-frames", str(extend["frames"]), "--seed", seed, "--prompt", extend["prompt"]], work)
        shutil.copy2(clip, target)
        print(f"{name}: {target}")
    return 0


def _iclora_file() -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(ICLORA_REPO, ICLORA_FILE)


def _method_name(steps: int, *, empty_prompt: bool, tiles: tuple[int, int, int], encode_tiles: tuple[int, int] | None = None) -> str:
    name = f"refine-{steps}"
    if empty_prompt:
        name += "-noprompt"
    if tiles != (1, 1, 1):
        name += "-t" + "x".join(str(count) for count in tiles)
    if encode_tiles is not None:
        name += f"-e{encode_tiles[0]}x{encode_tiles[1]}"
    return name


def _parse_tiles(text: str) -> tuple[int, int, int]:
    frames, height, width = (int(part) for part in text.lower().split("x"))
    return frames, height, width


def _cmd_run(args: argparse.Namespace) -> int:
    from bench.video_upscale.refine import RefineSettings, bicubic_upscale, estimate_fit, iclora_upscale, load_upscale_model, upscale_video
    from zvisiongenerator.utils.provenance import original_video_workflow
    from zvisiongenerator.utils.video_upscale import ltx_upscale_prompt
    from zvisiongenerator.video_file_upscale_runner import read_video_upscale_source

    cases = load_cases()
    clips = _select(cases["clips"], args.clips)
    run_dir = RUNS_DIR / args.label
    run_dir.mkdir(parents=True, exist_ok=True)
    tiles = _parse_tiles(args.tiles)
    encode_tiles = tuple(int(part) for part in args.encode_tiles.lower().split("x")) if args.encode_tiles else None
    steps_list = [int(step) for step in args.steps.split(",")] if args.steps else []
    metrics_path = run_dir / "metrics.jsonl"
    _write_meta(run_dir, args)

    read = {name: read_video_upscale_source(SOURCES_DIR / f"{name}.mp4") for name in clips}
    sources = {name: source for name, (source, _config) in read.items()}
    model = load_upscale_model(args.model or cases["model"], low_memory=args.low_memory) if steps_list or args.iclora else None
    lora_path = _iclora_file() if args.iclora else None
    seed = cases["seed"]
    for name, source in sources.items():
        clip_dir = run_dir / name
        clip_dir.mkdir(exist_ok=True)
        print(f"{name}: {source.width}x{source.height}, {source.frame_count} frames at {source.fps:g} fps")
        extended = original_video_workflow(read[name][1]) == "extend"
        prompt = "" if args.empty_prompt else ltx_upscale_prompt(source.scene, source.prompt, extended=extended)
        print(f"  prompt: {prompt!r}")
        suffix = "-noprompt" if args.empty_prompt else ""
        jobs: list[tuple[str, Callable[[Path], dict[str, Any]]]] = []
        if args.bicubic:
            jobs.append(("bicubic", partial(bicubic_upscale, source)))
        for steps in steps_list:
            settings = RefineSettings(steps=steps, prompt=prompt, seed=seed, tiles=tiles, tile_overlap=args.tile_overlap, encode_tiles=encode_tiles)
            jobs.append((_method_name(steps, empty_prompt=args.empty_prompt, tiles=tiles, encode_tiles=encode_tiles), partial(upscale_video, model, source, settings)))
        for strength in [float(value) for value in args.iclora.split(",")] if args.iclora else []:
            run = partial(iclora_upscale, model, source, lora_path=lora_path, strength=strength, prompt=prompt, seed=seed)
            jobs.append((f"iclora-{strength:g}{suffix}", lambda output, run=run: run(output_path=output)))
        fit = estimate_fit(model, source, low_memory=args.low_memory) if model is not None else None
        for method, job in jobs:
            output = clip_dir / f"{method}.mp4"
            if output.exists() and not args.force:
                print(f"  {method}: exists, skipped")
                continue
            print(f"  {method} ...")
            try:
                result = job(output)
                status = "ok"
            except Exception as exc:  # noqa: BLE001 -- a bench records every failure (out of memory included) and moves on
                result, status = {"error": f"{type(exc).__name__}: {exc}"}, "failed"
                print(f"  {method}: {result['error']}")
            record = {"clip": name, "method": method, "status": status, "frames": source.frame_count, "fps": source.fps, "output": f"{source.width * 2}x{source.height * 2}", "estimate": fit, **result}
            if status == "ok":
                record["seconds_per_output_second"] = round(result["seconds"] / (source.frame_count / source.fps), 1)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"  {method}: {status} {result.get('seconds', '')} s, peak {result.get('peak_gib', '')} GiB")
    return 0


def _cmd_encode_check(args: argparse.Namespace) -> int:
    from bench.video_upscale.refine import encode_check, load_upscale_model
    from zvisiongenerator.video_file_upscale_runner import read_video_upscale_source

    cases = load_cases()
    model = load_upscale_model(args.model or cases["model"])
    tiles = [(int(size), int(frames)) for size, frames in (item.split("x") for item in args.tiles.split(","))]
    out = RUNS_DIR / "encode-check.jsonl"
    for name in _select(cases["clips"], args.clips):
        source = read_video_upscale_source(SOURCES_DIR / f"{name}.mp4")[0]
        for record in encode_check(model, source, tiles):
            record = {"clip": name, **record}
            print(record)
            with out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    from bench.video_upscale.review import serve

    serve(RUNS_DIR, args.host, args.port)
    return 0


def _write_meta(run_dir: Path, args: argparse.Namespace) -> None:
    from zvisiongenerator.backends.mac.memory import memory_budget_bytes, memory_size_bytes

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {
        "commit": commit,
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "memory_gib": round((memory_size_bytes() or 0) / 1024**3, 1),
        "budget_gib": round((memory_budget_bytes() or 0) / 1024**3, 1),
        "args": vars(args) | {"func": None},
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def _cmd_sheet(args: argparse.Namespace) -> int:
    from bench.video_upscale.sheet import build_sheet
    from zvisiongenerator.video_file_upscale_runner import read_video_upscale_source

    cases = load_cases()
    run_dirs = [RUNS_DIR / label for label in args.labels]
    out_dir = run_dirs[0] if len(run_dirs) == 1 else RUNS_DIR / "-".join(args.labels)
    methods_by_clip: dict[str, list[str]] = {}
    for run_dir in run_dirs:
        for clip_dir in sorted(path for path in run_dir.iterdir() if path.is_dir() and path.name in cases["clips"]):
            for video in sorted(path for path in clip_dir.glob("*.mp4") if not path.stem.endswith(".raw")):
                if out_dir != run_dir:
                    (out_dir / clip_dir.name).mkdir(parents=True, exist_ok=True)
                    link = out_dir / clip_dir.name / video.name
                    if not link.exists():
                        link.symlink_to(video.resolve())
                methods_by_clip.setdefault(clip_dir.name, []).append(video.stem)
    frame_counts = {clip: read_video_upscale_source(SOURCES_DIR / f"{clip}.mp4")[0].frame_count for clip in methods_by_clip}
    sheet = build_sheet(out_dir, cases["clips"], {clip: sorted(set(methods)) for clip, methods in methods_by_clip.items()}, frame_counts, SOURCES_DIR)
    print(sheet)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the bench command named in *argv*."""
    parser = argparse.ArgumentParser(prog="python -m bench.video_upscale")
    commands = parser.add_subparsers(required=True)

    render = commands.add_parser("render", help="Render the source clips with the app's video CLI.")
    render.add_argument("--clips", help="Comma-separated clip ids (default: all).")
    render.add_argument("--force", action="store_true", help="Render clips that already exist again.")
    render.set_defaults(func=_cmd_render)

    run = commands.add_parser("run", help="Upscale the source clips and record time and memory.")
    run.add_argument("--label", required=True, help="Run folder under runs/.")
    run.add_argument("--clips", help="Comma-separated clip ids (default: all).")
    run.add_argument("--model", help="Model alias (default: the cases' model).")
    run.add_argument("--steps", default="", help="Comma-separated refine step counts, 1-3.")
    run.add_argument("--bicubic", action="store_true", help="Also write the bicubic baseline.")
    run.add_argument("--iclora", default="", help="Comma-separated IC-LoRA strengths to also run, e.g. 1.0,0.6.")
    run.add_argument("--empty-prompt", action="store_true", help="Refine with an empty prompt instead of the recorded one.")
    run.add_argument("--tiles", default="1x1x1", help="DiT tile counts as FRAMESxHEIGHTxWIDTH.")
    run.add_argument("--encode-tiles", default="", help="Encoder tile size as PIXELSxFRAMES (default: the encoder's own).")
    run.add_argument("--tile-overlap", type=int, default=1, help="Tile overlap in latent cells.")
    run.add_argument("--low-memory", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--force", action="store_true", help="Redo methods whose output exists.")
    run.set_defaults(func=_cmd_run)

    encode = commands.add_parser("encode-check", help="Compare encoder tile sizes: peak memory, time and latent difference.")
    encode.add_argument("--clips", help="Comma-separated clip ids (default: all).")
    encode.add_argument("--model", help="Model alias (default: the cases' model).")
    encode.add_argument("--tiles", default="512x48,512x32,384x32", help="Comma-separated PIXELSxFRAMES encoder tile sizes.")
    encode.set_defaults(func=_cmd_encode_check)

    review = commands.add_parser("review", help="Serve the blind-review site on the local network.")
    review.add_argument("--host", default="0.0.0.0", help="Address to listen on (default: every interface).")
    review.add_argument("--port", type=int, default=8765)
    review.set_defaults(func=_cmd_review)

    sheet = commands.add_parser("sheet", help="Write a blind review sheet for one or more runs.")
    sheet.add_argument("labels", nargs="+")
    sheet.set_defaults(func=_cmd_sheet)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
