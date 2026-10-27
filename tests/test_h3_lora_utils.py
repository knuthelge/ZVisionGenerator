"""Tests for the platform-neutral H3 LoRA reader: it needs no peft, no backend and no torch at import."""

from __future__ import annotations

import ast
from pathlib import Path

from zvisiongenerator.utils import h3_lora


def test_torch_and_diffusers_are_imported_lazily_and_no_backend_is_imported() -> None:
    tree = ast.parse(Path(h3_lora.__file__).read_text(encoding="utf-8"))
    top_level = [node.module or "" for node in tree.body if isinstance(node, ast.ImportFrom)] + [alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names]
    assert not [name for name in top_level if name.split(".")[0] in {"torch", "diffusers", "safetensors", "peft"}]
    # Backends are forbidden at any nesting level; heavy libraries only at module level (they are imported inside functions).
    anywhere = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)] + [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not [name for name in anywhere if name.startswith("zvisiongenerator.backends")]
