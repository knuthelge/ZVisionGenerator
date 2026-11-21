"""The audio medium of the memory bench: its ``ziv-audio`` arguments and the app's verdict, which is the model's measured peak against the machine."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from bench.memory.machine import case_machine, verdict_row

OUTPUT_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a"})


def prepare(case: dict[str, Any], cases: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Set the ``pixels`` a row carries to 0: audio has no picture size."""
    case["pixels"] = 0


def voice_path(run_dir: Path) -> Path:
    """Return the recording a cloned-voice case follows: the synthetic take under ``sources/`` (a ``.wav`` placeholder before it exists)."""
    found = sorted((run_dir / "sources").glob("voice.*"))
    return found[0] if found else run_dir / "sources" / "voice.wav"


def ensure_sources(case: Mapping[str, Any], run_dir: Path, cases: Mapping[str, Any]) -> None:
    """Make the recording a case with a ``voice_text`` follows, once: the app reads that text in its first built-in voice, in a process of its own before the case's.

    A speech take the app made in a built-in voice needs no consent to clone, and it is nobody's recording. The other cases make their audio from a prompt alone.

    Raises:
        subprocess.CalledProcessError: The take could not be made.
        FileNotFoundError: The take wrote no audio file.
    """
    sources = run_dir / "sources"
    if not case.get("voice_text") or any(sources.glob("voice.*")):
        return
    sources.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=sources) as scratch:
        argv = ["-m", case["model"], "--kind", "speech", "--runs", "1", "--seed", str(case["seed"]), "--prompt", case["voice_text"], "--output", scratch]
        subprocess.run([sys.executable, "-c", "from zvisiongenerator.audio_cli import main; main()", *argv], check=True)
        takes = [path for path in Path(scratch).iterdir() if path.suffix.lower() in OUTPUT_SUFFIXES]
        if not takes:
            raise FileNotFoundError("The bench's voice recording was not written.")
        shutil.move(str(takes[0]), sources / f"voice{takes[0].suffix.lower()}")


def case_argv(case: Mapping[str, Any], run_dir: Path) -> list[str]:
    """Return the ``ziv-audio`` arguments that run *case*, writing its take into ``run_dir/out/<id>``.

    A case's ``prompt_repeat`` repeats its prompt that many times (a long text to read), ``lyrics`` are a song's, and a ``voice_text`` follows
    the recording of :func:`ensure_sources`.
    """
    prompt = " ".join([case["prompt"]] * case.get("prompt_repeat", 1))
    argv = ["-m", case["model"], "--kind", case["kind"], "--runs", "1", "--seed", str(case["seed"]), "--prompt", prompt, "--output", str(run_dir / "out" / case["id"])]
    if case.get("seconds"):
        argv += ["--seconds", str(case["seconds"])]
    if case.get("lyrics"):
        argv += ["--lyrics", case["lyrics"]]
    if case.get("voice_text"):
        argv += ["--voice", str(voice_path(run_dir)), "--voice-text", case["voice_text"]]
    return argv


def run_cli(argv: list[str]) -> None:
    """Run the app's ``ziv-audio`` entry point with *argv* in this process."""
    from zvisiongenerator.audio_cli import main

    sys.argv = ["ziv-audio", *argv]
    main()


def install_probes(case: Mapping[str, Any]) -> Callable[[], dict[str, Any]] | None:
    """Return nothing: audio has no loader placement to watch or force."""
    return None


def estimate(case: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the app's memory verdict for *case*'s model, or ``None`` when the model has no measured peak here or the machine has no budget."""
    from zvisiongenerator.utils.audio_memory import audio_verdict
    from zvisiongenerator.utils.audio_request import find_audio_preset
    from zvisiongenerator.utils.config import load_config

    where = case_machine(case)
    if where is None:
        return None
    preset = find_audio_preset(load_config(), case["model"], missing="the case has no model")
    verdict = audio_verdict(preset, "mps" if where.mac else "cuda", where.budget)
    return None if verdict is None else verdict_row(verdict, where.budget)
