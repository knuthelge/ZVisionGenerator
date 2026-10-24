"""Tests for the platform-neutral H3 LoRA reader: it needs no peft, no backend and no torch at import."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
from safetensors.torch import save_file  # noqa: E402

from zvisiongenerator.backends.cuda.video.h3 import lora as cuda_lora  # noqa: E402
from zvisiongenerator.utils import h3_lora  # noqa: E402


def _write(path: Path, metadata: dict[str, str] | None = None) -> Path:
    tensors = {"diffusion_model.blocks.0.attn.out_proj.lora_A.weight": torch.ones(4, 8), "diffusion_model.blocks.0.attn.out_proj.lora_B.weight": torch.ones(8, 4)}
    save_file(tensors, str(path), metadata=metadata)
    return path


def test_loading_needs_neither_peft_nor_a_backend(tmp_path: Path) -> None:
    script = (
        "import importlib.util, sys\n"
        "real = importlib.util.find_spec\n"
        "importlib.util.find_spec = lambda name, *a, **k: None if name == 'peft' else real(name, *a, **k)\n"
        "from zvisiongenerator.utils.h3_lora import load_lora, pairs_from_state\n"
        f"f = load_lora({str(_write(tmp_path / 'a.safetensors', {'alpha': '2'}))!r})\n"
        "assert set(pairs_from_state(f.state)) == {'transformer_blocks.0.attn.to_out.0'}\n"
        "assert f.scales == {'transformer_blocks.0.attn.to_out.0': 0.5}\n"
        "assert not [m for m in sys.modules if m == 'peft' or m.startswith('zvisiongenerator.backends')], 'imported peft or a backend'\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_torch_and_diffusers_are_imported_lazily() -> None:
    tree = ast.parse(Path(h3_lora.__file__).read_text(encoding="utf-8"))
    names = [node.module or "" for node in tree.body if isinstance(node, ast.ImportFrom)] + [alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names]
    assert not [name for name in names if name.split(".")[0] in {"torch", "diffusers", "safetensors", "peft"}]


def test_the_cuda_module_re_exports_the_same_objects() -> None:
    for name in h3_lora.__all__:
        if name != "fetch_lora_state":
            assert getattr(cuda_lora, name) is getattr(h3_lora, name)
