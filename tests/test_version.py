"""Tests for dynamic version resolution via hatch-vcs."""

from __future__ import annotations

import re


class TestVersion:
    """Verify __version__ is correctly exposed and well-formed."""

    def test_version_is_importable(self) -> None:
        from zvisiongenerator import __version__

        assert isinstance(__version__, str)
        assert __version__ != ""

    def test_version_is_not_unknown_fallback(self) -> None:
        from zvisiongenerator import __version__

        assert __version__ != "0.0.0+unknown", "_version.py was not generated; hatch-vcs may not be installed or the package is not in a git repo"

    def test_version_matches_pep440(self) -> None:
        from zvisiongenerator import __version__

        # PEP 440 loose pattern — covers release, pre, post, dev, and local segments
        pep440 = re.compile(r"^(\d+!)?\d+(\.\d+)*(\.post\d+)?(\.dev\d+)?(\+[a-zA-Z0-9._]+)?$")
        assert pep440.match(__version__), f"Version {__version__!r} does not look PEP 440-compliant"

    def test_version_accessible_from_internal_module(self) -> None:
        from zvisiongenerator._version import __version__

        assert isinstance(__version__, str)
        assert __version__ != ""
