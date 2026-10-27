"""Shared fixtures for the prompt, CLI and config test modules.

A test module that wants a fixture from here binds it at module level, for example::

    import _prompt_helpers

    packaged_config = _prompt_helpers.packaged_config
    packaged_config_shared = _prompt_helpers.packaged_config_shared
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from zvisiongenerator.utils import config as config_module


@pytest.fixture(scope="session")
def packaged_config_shared(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Load the packaged config once per session with no user override; read-only (use ``packaged_config`` to change it)."""
    data_dir = tmp_path_factory.mktemp("packaged-config") / ".ziv"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(config_module, "get_ziv_data_dir", lambda: data_dir)
        return config_module.load_config()


@pytest.fixture
def packaged_config(packaged_config_shared: dict[str, Any]) -> dict[str, Any]:
    """Return a private deep copy of the packaged config with no user override."""
    return copy.deepcopy(packaged_config_shared)
