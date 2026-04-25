"""Validate end-user Web UI documentation coverage and navigation."""

from __future__ import annotations

from pathlib import Path
import re

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"
DOCS_INDEX_PATH = REPO_ROOT / "docs" / "index.md"
GETTING_STARTED_PATH = REPO_ROOT / "docs" / "getting-started.md"
WEB_UI_GUIDE_PATH = REPO_ROOT / "docs" / "guides" / "web-ui.md"
CLI_REFERENCE_PATH = REPO_ROOT / "docs" / "reference" / "cli.md"
MKDOCS_PATH = REPO_ROOT / "mkdocs.yml"
DEVELOPMENT_DOC_PATH = REPO_ROOT / "docs" / "development.md"
WEB_UI_ARCHITECTURE_PATH = REPO_ROOT / "docs" / "architecture" / "web-ui.md"
WEB_MAIN_PATH = REPO_ROOT / "zvisiongenerator" / "web" / "__init__.py"
WEB_SERVER_PATH = REPO_ROOT / "zvisiongenerator" / "web" / "server.py"
TOP_NAV_PATH = REPO_ROOT / "frontend" / "src" / "lib" / "components" / "organisms" / "TopNav.svelte"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_web_ui_docs_cover_launch_access_pages_and_workflows() -> None:
    """User-facing docs should explain how to launch, open, and use the Web UI."""
    readme = _read_text(README_PATH)
    getting_started = _read_text(GETTING_STARTED_PATH)
    web_guide = _read_text(WEB_UI_GUIDE_PATH)
    cli_reference = _read_text(CLI_REFERENCE_PATH)

    assert "ziv ui" in readme
    assert "Running `ziv` with no arguments opens the same Web UI" in readme
    assert "http://127.0.0.1:8080/" in readme
    assert "Workspace, Models, Gallery, and Configuration" in readme

    assert "ziv ui" in getting_started
    assert "ziv-ui" in web_guide
    assert "--no-browser" in getting_started
    assert "--host" in web_guide
    assert "--port" in web_guide
    assert "http://127.0.0.1:8080/app" in getting_started
    assert "The root URL (`/`) redirects to the app automatically." in getting_started

    for page_name in ("Workspace", "Models", "Gallery", "Configuration"):
        assert page_name in web_guide

    for workflow_name in ("Text to Image", "Image to Image", "Text to Video", "Image to Video"):
        assert workflow_name in web_guide

    assert "On Windows, the Web UI is currently image-only" in web_guide
    assert "CLI-Only" in web_guide
    assert "upscale_save_pre" in web_guide
    assert "scheduler" in web_guide
    assert "upscale" in web_guide
    assert "Running `ziv` with no subcommand starts the Web UI." in cli_reference
    assert "`ziv ui` / `ziv-ui`" in cli_reference


def test_mkdocs_nav_and_web_ui_doc_links_point_to_existing_pages() -> None:
    """MkDocs nav and in-repo quick links should reference real documentation pages."""
    mkdocs_config = yaml.safe_load(_read_text(MKDOCS_PATH))
    docs_index = _read_text(DOCS_INDEX_PATH)

    nav = mkdocs_config["nav"]
    assert {"Home": "index.md"} in nav
    assert {"Getting Started": "getting-started.md"} in nav

    guides_entry = next(item for item in nav if isinstance(item, dict) and "Guides" in item)
    guide_targets = [next(iter(entry.values())) for entry in guides_entry["Guides"]]
    assert "guides/web-ui.md" in guide_targets
    assert "guides/image.md" in guide_targets
    assert "guides/video.md" in guide_targets

    reference_entry = next(item for item in nav if isinstance(item, dict) and "Reference" in item)
    reference_targets = [next(iter(entry.values())) for entry in reference_entry["Reference"]]
    assert "reference/cli.md" in reference_targets

    architecture_entry = next(item for item in nav if isinstance(item, dict) and "Architecture" in item)
    architecture_targets = [next(iter(entry.values())) for entry in architecture_entry["Architecture"]]
    assert "architecture/web-ui.md" in architecture_targets

    for target in [
        REPO_ROOT / "docs" / "index.md",
        REPO_ROOT / "docs" / "getting-started.md",
        REPO_ROOT / "docs" / "guides" / "web-ui.md",
        REPO_ROOT / "docs" / "reference" / "cli.md",
        REPO_ROOT / "docs" / "architecture" / "web-ui.md",
    ]:
        assert target.is_file()

    for relative_link in (
        "getting-started.md",
        "guides/web-ui.md",
        "guides/image.md",
        "guides/video.md",
        "reference/cli.md",
        "development.md",
    ):
        assert f"]({relative_link})" in docs_index


def test_development_docs_link_to_the_web_ui_contract_note() -> None:
    """Developer-facing docs should link to the maintained web contract note."""
    development_doc = _read_text(DEVELOPMENT_DOC_PATH)
    web_architecture_doc = _read_text(WEB_UI_ARCHITECTURE_PATH)

    assert "architecture/web-ui.md" in development_doc
    assert "`/api/workspace`" in web_architecture_doc
    assert "`draft.hydrateFromContext(ctx, preferredModel)`" in web_architecture_doc


def test_web_ui_docs_match_supported_runtime_surface_and_cli_behavior() -> None:
    """Docs should align with the current launcher behavior and shipped user routes."""
    web_main = _read_text(WEB_MAIN_PATH)
    web_server = _read_text(WEB_SERVER_PATH)
    web_guide = _read_text(WEB_UI_GUIDE_PATH)
    cli_reference = _read_text(CLI_REFERENCE_PATH)

    assert 'default="127.0.0.1"' in web_main
    assert "default=8080" in web_main
    assert '"--no-browser"' in web_main
    assert "if not argv:" in (REPO_ROOT / "zvisiongenerator" / "cli.py").read_text(encoding="utf-8")
    assert "run_server()" in (REPO_ROOT / "zvisiongenerator" / "cli.py").read_text(encoding="utf-8")
    assert 'RedirectResponse(url="/app"' in web_server
    assert '@app.get("/app")' in web_server

    assert "By default, the app starts on `http://127.0.0.1:8080/`" in web_guide
    assert "Open `http://127.0.0.1:8080/` or `http://127.0.0.1:8080/app`" in web_guide
    assert "The root URL (`/`) redirects to the app automatically." in _read_text(GETTING_STARTED_PATH)
    assert "Preferred local port; the app chooses the next available port if this one is busy" in cli_reference


def test_user_facing_web_ui_docs_avoid_internal_runtime_implementation_terms() -> None:
    """End-user docs should stay task-oriented instead of exposing internal runtime details."""
    user_facing_docs = {
        "README": _read_text(README_PATH),
        "docs/index.md": _read_text(DOCS_INDEX_PATH),
        "docs/getting-started.md": _read_text(GETTING_STARTED_PATH),
        "docs/guides/web-ui.md": _read_text(WEB_UI_GUIDE_PATH),
    }

    forbidden_patterns = [
        r"\bFastAPI\b",
        r"\bSvelte\b",
        r"\bJinja\b",
        r"\bHTMX\b",
        r"/api/",
        r"/jobs/",
        r"workflow_contract",
        r"JSON/SSE",
    ]

    for name, content in user_facing_docs.items():
        for pattern in forbidden_patterns:
            assert re.search(pattern, content) is None, f"{name} should not expose internal runtime details matching {pattern!r}"


def test_web_ui_doc_page_names_remain_consistent_with_visible_navigation() -> None:
    """The guide should describe the same user-visible destinations exposed by the SPA navigation."""
    top_nav = _read_text(TOP_NAV_PATH)
    web_guide = _read_text(WEB_UI_GUIDE_PATH)

    for route in ("workspace", "models", "gallery", "config"):
        assert f"id: '{route}'" in top_nav

    assert "/app#/workspace" in web_guide
    assert "/app#/models" in web_guide
    assert "/app#/gallery" in web_guide
    assert "/app#/config" in web_guide
