"""Tests that every repository path the developer docs name in backticks still exists.

A *path token* is an inline code span of a page in ``docs/developer/`` that contains a ``/`` or ends in the extension of a file the
repository keeps (``.py``, ``.md``, ``.yaml`` and so on). Pages are written relative to the places a contributor works in, so a token
is looked up under the repository root, the ``zvisiongenerator`` package, ``frontend`` and ``frontend/src`` and the folders under
them that the docs abbreviate, and a bare file name matches a file of that name anywhere in the repository. A token without an
extension may also name a module (``.py``, ``.ts``, ``.svelte``) or a folder. ``module.name`` names a symbol of a module, which the
module's source must mention.

Not path tokens, and skipped: anything with a ``<...>`` placeholder, a glob, a URL, a space (a command line or a phrase), a route
(it starts with ``/``), a location in the user's data folder (it starts with ``~``), an ``owner/repo@revision`` reference, a bare
extension and the text of a link to another site (an upstream repository, which is not part of this one).
Fenced code blocks are not read, and a ``::`` suffix (a test or a class) is cut before the lookup.
"""

from __future__ import annotations

import os
import re
from functools import cache
from pathlib import Path

import pytest

from tests._paths import REPO_ROOT

_PAGES = sorted((REPO_ROOT / "docs" / "developer").glob("*.md"))
_SEARCH_ROOTS = (
    REPO_ROOT,
    REPO_ROOT / "zvisiongenerator",
    REPO_ROOT / "zvisiongenerator" / "backends",
    REPO_ROOT / "docs",
    REPO_ROOT / "frontend",
    REPO_ROOT / "frontend" / "src",
    REPO_ROOT / "frontend" / "src" / "lib",
    REPO_ROOT / "frontend" / "src" / "lib" / "components",
)
# Extensions of the files the repository keeps; a token with any other ending (``ziv.sqlite3``, ``results.jsonl``) names something a run makes.
_FILE_EXTENSIONS = frozenset({"py", "md", "yaml", "yml", "toml", "json", "ts", "svelte", "mjs", "css", "html", "svg", "mp3", "sh", "txt"})
_MODULE_SUFFIXES = (".py", ".ts", ".svelte", ".svelte.ts")
# Files and folders that exist only while the app, the e2e targets or the benches run.
_CREATED_AT_RUN_TIME = frozenset({"output", ".web_uploads", "ziv-quant.json", "ziv-h3.json", "meta.json", "runs"})
# Folders no source lives in, left out of the lookup for a bare file name.
_PRUNED_DIRECTORIES = frozenset({".git", ".venv", "node_modules", "__pycache__", "site", ".ruff_cache", ".pytest_cache"})
# The frontend's import aliases (``$lib/api/client``) and the folder each stands for.
_ALIASES = {"$lib": "lib", "$features": "features", "$app": "app"}
_CODE_SPAN = re.compile(r"`([^`\n]+)`")
_EXTERNAL_LINK_TEXT = re.compile(r"\[`[^`\n]+`\]\(https?://")
_FENCED_BLOCK = re.compile(r"^(```|~~~).*?^\1\s*$", re.MULTILINE | re.DOTALL)
_NOT_A_PATH = re.compile(r"[\s<>*{}()|=,;@?\[\]]|://")


def _path_tokens(text: str) -> list[str]:
    """Return the path tokens of Markdown *text*, in page order, without the fenced code blocks."""
    tokens = []
    body = _FENCED_BLOCK.sub("", text)
    for link_text in _EXTERNAL_LINK_TEXT.findall(body):
        body = body.replace(link_text, "")
    for span in _CODE_SPAN.findall(body):
        token = span.split("::", 1)[0].removeprefix("./").rstrip(":.")
        if not token or token[0] in "/~-" or _NOT_A_PATH.search(token) or ("/" not in token and token.startswith(".")):
            continue
        extension = token.rsplit(".", 1)[-1] if "." in token.rsplit("/", 1)[-1] else ""
        if "/" in token or extension in _FILE_EXTENSIONS:
            tokens.append(token)
    return tokens


@cache
def _files_by_name() -> frozenset[str]:
    """Return the base name of every file in the repository, for a token that names a file without its folder."""
    names: set[str] = set()
    for _directory, subdirectories, files in os.walk(REPO_ROOT):
        subdirectories[:] = [name for name in subdirectories if name not in _PRUNED_DIRECTORIES]
        names.update(files)
    return frozenset(names)


def _exists(token: str) -> bool:
    """Return whether the repository holds what *token* names, under any of the search roots."""
    for alias, folder in _ALIASES.items():
        if token.startswith(alias + "/"):
            token = f"{folder}/{token.removeprefix(alias + '/')}"
    token = token.rstrip("/")
    if token in _CREATED_AT_RUN_TIME or token.rsplit("/", 1)[-1] in _CREATED_AT_RUN_TIME:
        return True
    if "/" not in token and "." in token:
        return token in _files_by_name()
    module, _, symbol = token.rpartition(".")
    if module and "/" in module and symbol.isidentifier() and symbol not in _FILE_EXTENSIONS:
        source = _module_file(module)
        return source is not None and symbol in source.read_text(encoding="utf-8")
    return _module_file(token) is not None or any((root / token).exists() for root in _SEARCH_ROOTS)


def _module_file(name: str) -> Path | None:
    """Return the file of the module *name* (a path without its extension, or with it) under the search roots, or ``None``."""
    for root in _SEARCH_ROOTS:
        for candidate in (root / name, *(Path(f"{root / name}{suffix}") for suffix in _MODULE_SUFFIXES)):
            if candidate.is_file():
                return candidate
    return None


@pytest.mark.parametrize("page", _PAGES, ids=lambda page: page.name)
def test_every_path_a_page_names_exists(page: Path) -> None:
    missing = sorted({token for token in _path_tokens(page.read_text(encoding="utf-8")) if not _exists(token)})

    assert not missing, f"{page.name} names paths that are not in the repository: {', '.join(missing)}"


def test_the_codemap_names_paths() -> None:
    """The check is not vacuous: the page with the most paths yields them, and a renamed one would be found missing."""
    tokens = _path_tokens((REPO_ROOT / "docs" / "developer" / "architecture.md").read_text(encoding="utf-8"))

    assert "backends/__init__.py" in tokens
    assert len(tokens) > 40
    assert _exists("backends/__init__.py")
    assert not _exists("backends/no_such_module.py")


@pytest.mark.parametrize(
    "span",
    [
        "tests/<area>/test_<module>.py",
        "output/*.png",
        "https://huggingface.co/owner/repo",
        "owner/repo@revision",
        "git switch main",
        "make frontend-build",
        "/api/jobs/log",
        "~/.ziv/presets/",
        "ziv.sqlite3",
        "ZIMAGE_MODEL",
        "--no-sharpen",
    ],
)
def test_what_is_not_a_path_is_skipped(span: str) -> None:
    assert _path_tokens(f"See `{span}` for more.") == []


def test_code_blocks_are_not_read() -> None:
    text = "Run:\n\n```bash\ncat docs/developer/missing_page.md\n```\n\nThen read `docs/developer/index.md`.\n"

    assert _path_tokens(text) == ["docs/developer/index.md"]
