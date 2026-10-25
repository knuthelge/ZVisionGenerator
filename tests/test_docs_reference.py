"""Tests that the reference docs list every CLI option and every user-facing config key in its own section."""

from __future__ import annotations

import argparse
import importlib.resources
import re
from pathlib import Path

import pytest
import yaml

from zvisiongenerator.converters.convert_checkpoint import _build_model_parser
from zvisiongenerator.image_cli import _build_parser as _build_image_parser
from zvisiongenerator.video_cli import _build_video_parser
from zvisiongenerator.web import _build_parser as _build_ui_parser

_DOCS = Path(__file__).resolve().parent.parent / "docs"
_CLI_REFERENCE = _DOCS / "cli" / "reference.md"
_CONFIG_REFERENCE = _DOCS / "concepts" / "configuration.md"

# Sections whose keys users set; each key must appear under the section's own heading.
_KEYED_SECTIONS = (
    "generation",
    "video_generation",
    "sharpening",
    "contrast",
    "saturation",
    "upscale",
    "seedvr2",
    "first_run",
    "ui",
    "prompt_enhancer",
)


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    """Return *parser*'s subcommands by name, or an empty dict when it has none."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def _long_options(parser: argparse.ArgumentParser) -> set[str]:
    """Return the long options of *parser* itself, without its subcommands' options."""
    return {opt for action in parser._actions for opt in action.option_strings if opt.startswith("--") and opt != "--help"}


def _section(text: str, name: str) -> str:
    """Return the body of the ``##`` section whose heading has *name* in an inline code span."""
    for chunk in re.split(r"^## ", text, flags=re.MULTILINE)[1:]:
        heading, _, body = chunk.partition("\n")
        if f"`{name}`" in heading:
            return body
    raise AssertionError(f"No section headed `{name}`")


def _documented(text: str, name: str) -> bool:
    """Return whether *name* appears as a whole word inside one inline code span of *text*.

    A dotted *name* such as ``smart.model`` must appear in full, and a plain name must not be the tail of a dotted path,
    so ``smart.model`` does not count for ``model``.
    """
    pattern = re.compile(r"(?<![\w.-])" + re.escape(name) + r"(?![\w-])")
    return any(pattern.search(span) for span in re.findall(r"`([^`\n]+)`", text))


# Dicts keyed by these are per-platform or per-length values, not settings of their own.
_VALUE_MAP_KEYS = (frozenset({"darwin", "win32", "linux"}), frozenset({"shorter", "same", "longer", "extra"}))


def _setting_keys(mapping: dict, prefix: str = "") -> list[str]:
    """Return every setting in *mapping* as a dotted path, descending into nested settings but not into per-platform or per-length values."""
    names: list[str] = []
    for key, value in mapping.items():
        names.append(prefix + key)
        if isinstance(value, dict) and not any(set(value) <= keys for keys in _VALUE_MAP_KEYS):
            names += _setting_keys(value, f"{prefix}{key}.")
    return names


def _packaged_config() -> dict:
    """Load the packaged config.yaml without any user override."""
    ref = importlib.resources.files("zvisiongenerator").joinpath("config.yaml")
    return yaml.safe_load(ref.read_text(encoding="utf-8"))


def _cli_cases() -> list[tuple[str, argparse.ArgumentParser]]:
    """Return each command's reference heading with the parser whose options belong under it."""
    cases = [
        ("ziv-ui", _build_ui_parser(prog="ziv-ui")),
        ("ziv-image", _build_image_parser(prog="ziv-image")),
        ("ziv-video", _build_video_parser(prog="ziv-video")),
    ]
    cases += [(f"ziv-model {name}", sub) for name, sub in _subparsers(_build_model_parser(prog="ziv-model")).items()]
    return cases


@pytest.mark.parametrize(("heading", "parser"), _cli_cases(), ids=[heading for heading, _ in _cli_cases()])
def test_cli_reference_lists_every_option(heading: str, parser: argparse.ArgumentParser) -> None:
    section = _section(_CLI_REFERENCE.read_text(encoding="utf-8"), heading)
    missing = sorted(opt for opt in _long_options(parser) if not _documented(section, opt))
    assert not missing, f"Add these options under `{heading}` in {_CLI_REFERENCE.relative_to(_DOCS.parent)}: {missing}"


def test_config_reference_lists_every_section() -> None:
    text = _CONFIG_REFERENCE.read_text(encoding="utf-8")
    missing = sorted(section for section in _packaged_config() if not _documented(text, section))
    assert not missing, f"Add these sections to {_CONFIG_REFERENCE.relative_to(_DOCS.parent)}: {missing}"


@pytest.mark.parametrize("section", _KEYED_SECTIONS)
def test_config_reference_lists_every_key(section: str) -> None:
    body = _section(_CONFIG_REFERENCE.read_text(encoding="utf-8"), section)
    missing = sorted(key for key in _setting_keys(_packaged_config()[section]) if not (_documented(body, key) or _documented(body, f"{section}.{key}")))
    assert not missing, f"Add these `{section}` keys under its heading in {_CONFIG_REFERENCE.relative_to(_DOCS.parent)}: {missing}"
