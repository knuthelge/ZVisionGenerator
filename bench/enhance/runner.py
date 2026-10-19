"""Run bench cases through a real enhancer, recording every model call, and write the results as JSON lines."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.utils.prompt_enhance import EnhanceContext, enhance_context, enhance_prompt, resolve_enhance_ceiling, word_count
from zvisiongenerator.utils.prompt_enhance_caption import enhance_caption
from zvisiongenerator.utils.prompt_enhance_video import enhance_video_prompt

from bench.enhance.cases import Case, case_seeds
from bench.enhance.checks import caption_format, failed_checks, length_plan, score_output

if TYPE_CHECKING:
    from zvisiongenerator.core.prompt_enhancer import PromptEnhancer

RESULTS_FILE = "results.jsonl"
META_FILE = "meta.json"


@dataclass
class RecordingEnhancer:
    """Pass every call through to *inner* and keep its messages, raw output and timings in ``calls``."""

    inner: PromptEnhancer
    calls: list[dict[str, Any]] = field(default_factory=list)
    clock: Callable[[], float] = time.perf_counter

    @property
    def repo(self) -> str:
        return self.inner.repo

    @property
    def revision(self) -> str | None:
        return self.inner.revision

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        seed: int,
        max_tokens: int,
        temperature: float,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[str]:
        """Yield *inner*'s text deltas and record the call once the stream ends."""
        start = self.clock()
        first: float | None = None
        raw = ""
        try:
            for delta in self.inner.generate(messages, seed=seed, max_tokens=max_tokens, temperature=temperature, cancelled=cancelled):
                if first is None and delta:
                    first = self.clock() - start
                raw += delta
                yield delta
        finally:
            seconds = self.clock() - start
            self.calls.append(
                {
                    "messages": messages,
                    "seed": seed,
                    "raw": raw,
                    "seconds": round(seconds, 3),
                    "first_text_seconds": None if first is None else round(first, 3),
                    "words": word_count(raw),
                }
            )

    def close(self) -> None:
        self.inner.close()


def case_context(case: Case) -> EnhanceContext:
    """Return the rewrite context for *case*, built as production builds it from a job's shape."""
    context = case.context
    return enhance_context(
        mode=case.media,
        width=context.get("width"),
        height=context.get("height"),
        ratio=context.get("ratio"),
        frames=context.get("frames"),
        fps=context.get("fps"),
        audio=bool(context.get("audio")),
        reference_image=bool(context.get("reference_image")),
        extends=bool(context.get("extends")),
        previous=context.get("previous"),
        extract_from=context.get("extract_from"),
        prompt_is_original=bool(context.get("prompt_is_original")),
        new_frames=context.get("new_frames"),
    )


def run_case(enhancer: RecordingEnhancer, case: Case, seed: int, *, config: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    """Enhance *case* once with *seed* through the entry point preflight uses, and return its scored result record."""
    ceiling = resolve_enhance_ceiling(config, family=case.family, mode=case.media)
    shared = dict(
        seed=seed,
        ceiling=ceiling,
        context=case_context(case),
        length_cfg=options["length"],
        temperature=options["temperature"],
        max_tokens=options["max_new_tokens"],
    )
    enhancer.calls.clear()
    output, scene, error, format_ok = "", "", None, None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if case.mode == "video":
                pair = enhance_video_prompt(enhancer, case.scene, case.prompt, case.settings, **shared)
                output, scene, format_ok = pair.prompt, pair.scene, not pair.format_fallback
            elif case.mode == "caption":
                output = enhance_caption(enhancer, case.prompt, case.settings, **shared).prompt
                format_ok = caption_format(output)
            else:
                output = enhance_prompt(enhancer, case.prompt, case.settings, mode="image", **shared).prompt
        except Exception as exc:  # noqa: BLE001 - a failed rewrite is a result to record, not a crash
            error = str(exc) or type(exc).__name__
    calls = list(enhancer.calls)
    checks, plan_record = _score(case, output=output, scene=scene, error=error, attempts=len(calls), format_ok=format_ok, ceiling=ceiling, options=options)
    first = calls[0]["messages"] if calls else []
    return {
        "case": case.id,
        "seed": seed,
        "tags": list(case.tags),
        "mode": case.mode,
        "input": {"prompt": case.prompt, "scene": case.scene, "settings": case.raw_settings, "context": case.context},
        "output": output,
        "scene": scene,
        "error": error,
        "warnings": [str(warning.message) for warning in caught],
        "system_prompt": next((message["content"] for message in first if message["role"] == "system"), ""),
        "user_message": next((message["content"] for message in first if message["role"] == "user"), ""),
        "plan": plan_record,
        "checks": checks,
        "attempts": [{key: call[key] for key in ("seed", "raw", "seconds", "first_text_seconds", "words")} for call in calls],
    }


def _score(case: Case, *, output: str, scene: str, error: str | None, attempts: int, format_ok: bool | None, ceiling: int, options: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return the checks of one result and the length plan they measured against (None for captions and failures)."""
    if error is not None:
        return failed_checks(), None
    plan, measures = length_plan(case, ceiling=ceiling, length_cfg=options["length"])
    checks = score_output(case, output=output, scene=scene, attempts=attempts, format_ok=format_ok, plan=plan, measures=measures)
    plan_record = None if plan is None else {"measures": measures, "target": plan.target, "lo": plan.lo, "hi": plan.hi, "clamped": plan.clamped}
    return checks, plan_record


def rescore_record(record: dict[str, Any], case: Case, *, config: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    """Return *record* with its checks recomputed from its stored output and the current *case* expectations; no model runs.

    A video's format (whether the pair parsed without fallback) cannot be told from the stored text, so it is kept.
    """
    format_ok = caption_format(record["output"]) if case.mode == "caption" else record["checks"].get("format")
    ceiling = resolve_enhance_ceiling(config, family=case.family, mode=case.media)
    checks, plan_record = _score(case, output=record["output"], scene=record["scene"], error=record["error"], attempts=len(record["attempts"]), format_ok=format_ok, ceiling=ceiling, options=options)
    return {**record, "tags": list(case.tags), "checks": checks, "plan": plan_record}


def rescore_run(run_dir: Path, cases: tuple[Case, ...], *, config: dict[str, Any], options: dict[str, Any]) -> int:
    """Recompute the checks of every result in *run_dir* whose case still exists, rewrite ``results.jsonl``, and return how many."""
    path = run_dir / RESULTS_FILE
    by_id = {case.id: case for case in cases}
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rescored = [rescore_record(record, by_id[record["case"]], config=config, options=options) if record["case"] in by_id else record for record in records]
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in rescored), encoding="utf-8")
    return sum(record["case"] in by_id for record in records)


def git_state(cwd: Path | None = None) -> dict[str, Any]:
    """Return the current commit and whether the working tree has changes; empty values outside a git checkout."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=cwd).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True, cwd=cwd).stdout.strip())
    except OSError, subprocess.CalledProcessError:
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def run_bench(
    cases: tuple[Case, ...],
    *,
    factory: Callable[[str, str | None], PromptEnhancer],
    repo: str,
    revision: str | None,
    model_choice: str,
    config: dict[str, Any],
    options: dict[str, Any],
    seeds: int,
    out_dir: Path,
    echo: Callable[[str], None] = print,
) -> Path:
    """Load the enhancer once, run every case with *seeds* seeds, and write ``results.jsonl`` and ``meta.json`` to *out_dir*.

    Results are appended and flushed per case, so a stopped run keeps what it finished.

    Raises:
        FileExistsError: When *out_dir* already holds results.
    """
    results_path = out_dir / RESULTS_FILE
    if results_path.exists():
        raise FileExistsError(f"{results_path} already exists; pick another label.")
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    enhancer = RecordingEnhancer(factory(repo, revision))
    load_seconds = round(time.perf_counter() - start, 3)
    meta = {
        "label": out_dir.name,
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **git_state(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "model_choice": model_choice,
        "repo": repo,
        "revision": revision,
        "temperature": options["temperature"],
        "max_new_tokens": options["max_new_tokens"],
        "seeds": seeds,
        "cases": len(cases),
        "load_seconds": load_seconds,
    }
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    echo(f"Loaded {repo} in {load_seconds:.1f}s; running {len(cases)} cases x {seeds} seeds.")
    try:
        with results_path.open("a", encoding="utf-8") as handle:
            for index, case in enumerate(cases, start=1):
                for seed in case_seeds(case.id, seeds):
                    record = run_case(enhancer, case, seed, config=config, options=options)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    seconds = sum(attempt["seconds"] for attempt in record["attempts"])
                    status = "FAILED" if record["error"] else "ok"
                    echo(f"[{index}/{len(cases)}] {case.id} seed {seed}: {status} ({seconds:.1f}s, {len(record['attempts'])} call(s))")
    finally:
        enhancer.close()
    return results_path
