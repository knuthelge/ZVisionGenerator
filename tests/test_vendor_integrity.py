"""Verify vendored ltx_core_mlx and ltx_pipelines_mlx packages are present and importable."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent


class TestVendoredPackagesExist:
    """Vendored directories exist on disk with expected structure."""

    def test_ltx_core_mlx_directory_exists(self) -> None:
        assert (ROOT / "packages" / "ltx_core_mlx").is_dir()

    def test_ltx_pipelines_mlx_directory_exists(self) -> None:
        assert (ROOT / "packages" / "ltx_pipelines_mlx").is_dir()

    def test_ltx_core_mlx_has_init(self) -> None:
        assert (ROOT / "packages" / "ltx_core_mlx" / "__init__.py").is_file()

    def test_ltx_pipelines_mlx_has_init(self) -> None:
        assert (ROOT / "packages" / "ltx_pipelines_mlx" / "__init__.py").is_file()

    def test_cli_and_main_removed(self) -> None:
        """cli.py and __main__.py should not exist in vendored ltx_pipelines_mlx."""
        assert not (ROOT / "packages" / "ltx_pipelines_mlx" / "cli.py").exists()
        assert not (ROOT / "packages" / "ltx_pipelines_mlx" / "__main__.py").exists()


class TestVendoredPackagesImportable:
    """Vendored packages can be imported as top-level modules."""

    @pytest.mark.skipif(sys.platform != "darwin", reason="MLX only available on macOS")
    def test_import_ltx_core_mlx(self) -> None:
        mod = importlib.import_module("ltx_core_mlx")
        assert mod is not None

    @pytest.mark.skipif(sys.platform != "darwin", reason="MLX only available on macOS")
    def test_import_ltx_pipelines_mlx(self) -> None:
        mod = importlib.import_module("ltx_pipelines_mlx")
        assert mod is not None

    @pytest.mark.skipif(sys.platform != "darwin", reason="MLX only available on macOS")
    def test_import_ltx_core_mlx_submodule(self) -> None:
        mod = importlib.import_module("ltx_core_mlx.utils")
        assert mod is not None


class TestPyprojectToml:
    """pyproject.toml has correct vendoring configuration."""

    @pytest.fixture
    def pyproject_text(self) -> str:
        return (ROOT / "pyproject.toml").read_text()

    def test_no_ltx_pipelines_dependency(self, pyproject_text: str) -> None:
        """ltx-pipelines-mlx must NOT appear in [project] dependencies."""
        lines = pyproject_text.split("\n")
        in_deps = False
        for line in lines:
            if line.strip().startswith("dependencies"):
                in_deps = True
                continue
            if in_deps and line.strip() == "]":
                in_deps = False
            if in_deps:
                assert "ltx-pipelines-mlx" not in line, "ltx-pipelines-mlx found in dependencies"

    def test_no_ltx_pipelines_in_uv_sources(self, pyproject_text: str) -> None:
        """ltx-pipelines-mlx must NOT appear in [tool.uv.sources]."""
        lines = pyproject_text.split("\n")
        in_uv_sources = False
        for line in lines:
            if "[tool.uv.sources]" in line:
                in_uv_sources = True
                continue
            if in_uv_sources and line.startswith("["):
                in_uv_sources = False
            if in_uv_sources:
                assert "ltx-pipelines-mlx" not in line, "ltx-pipelines-mlx found in uv sources"

    def test_wheel_packages_include_vendored(self, pyproject_text: str) -> None:
        assert '"packages/ltx_core_mlx"' in pyproject_text
        assert '"packages/ltx_pipelines_mlx"' in pyproject_text

    def test_ruff_excludes_vendored(self, pyproject_text: str) -> None:
        assert '"packages"' in pyproject_text
