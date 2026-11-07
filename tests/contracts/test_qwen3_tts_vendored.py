"""Pin the vendored ``qwen_tts`` and ``faster_qwen3_tts`` trees: the commits, the licenses and the parts ``make update-qwen3-tts`` leaves out.

The trees are read as text, so the test runs on every platform. The wheel must resolve without a demo's dependencies, which is why
the command-line modules (the only code that imports ``gradio``) are not vendored.
"""

from __future__ import annotations

import re

import pytest

from tests._paths import REPO_ROOT

PACKAGES = REPO_ROOT / "packages"
MAKEFILE = REPO_ROOT / "Makefile"

# (package folder, Makefile variable of its pinned commit, text its license starts with, files left out)
TREES = [
    ("qwen_tts", "QWEN_TTS_COMMIT", "Apache License", ("cli", "__main__.py")),
    ("faster_qwen3_tts", "FASTER_QWEN3_TTS_COMMIT", "MIT License", ("cli.py",)),
]


def _makefile_commit(variable: str) -> str:
    match = re.search(rf"^{variable}\s*\?=\s*(\S+)\s*$", MAKEFILE.read_text(encoding="utf-8"), re.MULTILINE)
    assert match is not None, f"Makefile defines no {variable}"
    return match.group(1)


@pytest.mark.parametrize(("folder", "variable", "_license", "_excluded"), TREES)
def test_pinned_commit_matches_the_makefile(folder, variable, _license, _excluded):
    commit = _makefile_commit(variable)

    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    assert (PACKAGES / folder / "UPSTREAM_COMMIT").read_text(encoding="utf-8").strip() == commit


@pytest.mark.parametrize(("folder", "_variable", "license_start", "_excluded"), TREES)
def test_the_license_is_vendored(folder, _variable, license_start, _excluded):
    assert license_start in (PACKAGES / folder / "LICENSE").read_text(encoding="utf-8")


@pytest.mark.parametrize(("folder", "_variable", "_license", "excluded"), TREES)
def test_the_command_line_modules_are_not_vendored(folder, _variable, _license, excluded):
    for name in excluded:
        assert not (PACKAGES / folder / name).exists(), name


@pytest.mark.parametrize(("folder", "_variable", "_license", "_excluded"), TREES)
def test_nothing_imports_gradio_or_a_dropped_module(folder, _variable, _license, _excluded):
    pattern = re.compile(r"^\s*(?:from|import)\s+(?:gradio|qwen_tts\.cli|faster_qwen3_tts\.cli|\.cli)\b", re.MULTILINE)
    offenders = [str(path.relative_to(PACKAGES)) for path in (PACKAGES / folder).rglob("*.py") if pattern.search(path.read_text(encoding="utf-8"))]

    assert offenders == []
