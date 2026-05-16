from __future__ import annotations

from pathlib import Path
import re

from zvisiongenerator.web.workspace_contract import build_workflow_contract, canonicalize_workflow


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_make_check_enforces_frontend_docs_and_packaged_spa_gates() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    match = re.search(r"^check:\s+(?P<deps>.+?)\s+##", makefile, flags=re.MULTILINE)

    assert match is not None
    deps = match.group("deps").split()

    assert "frontend-test" in deps
    assert "frontend-static-check" in deps
    assert "docs-check" in deps


def test_make_install_installs_python_and_frontend_dependencies() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    install_match = re.search(r"^install:\s+(?P<deps>[^#\n]+)##[^\n]*(?P<body>(?:\n\t.+)+)", makefile, flags=re.MULTILINE)
    frontend_match = re.search(r"^frontend-install:[^#\n]*##[^\n]*(?P<body>(?:\n\t.+)+)", makefile, flags=re.MULTILINE)

    assert install_match is not None
    assert frontend_match is not None
    assert "frontend-install" in install_match.group("deps").split()
    assert "uv sync" in install_match.group("body")
    assert "pnpm --dir frontend install --frozen-lockfile" in frontend_match.group("body")


def test_release_workflow_validates_packaged_spa_before_build() -> None:
    release_workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    validation_index = release_workflow.index("make frontend-static-check")
    build_index = release_workflow.index("make build")

    assert validation_index < build_index


def test_workflow_contract_exposes_canonical_values_only() -> None:
    contract = build_workflow_contract()

    assert "legacy_aliases" not in contract
    assert contract["values"] == ["txt2img", "img2img", "txt2vid", "img2vid"]


def test_workflow_contract_accepts_only_canonical_values() -> None:
    assert canonicalize_workflow("txt2img") == "txt2img"
    assert canonicalize_workflow("img2img") == "img2img"
    assert canonicalize_workflow("txt2vid") == "txt2vid"
    assert canonicalize_workflow("img2vid") == "img2vid"
    assert canonicalize_workflow("image") is None
    assert canonicalize_workflow("texttoimage") is None
    assert canonicalize_workflow("i2i") is None
    assert canonicalize_workflow("i2v") is None
