"""The CUDA allocator options are set before any module that can start CUDA is imported."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from unittest.mock import MagicMock

import pytest

from tests._paths import REPO_ROOT
from zvisiongenerator import backends
from zvisiongenerator.backends.cuda import memory as memory_module

# Replaces the libraries that can start CUDA (PyTorch itself, and ``diffusers``, whose import pulls in ``bitsandbytes``,
# which queries the device while it is imported) with stubs and records the allocator variables each time one is first imported, so no real library loads.
_PROBE = """
import importlib.abc, importlib.machinery, json, os, sys, types
from unittest.mock import MagicMock

HEAVY = {"torch", "bitsandbytes", "diffusers", "transformers", "mflux", "mlx"}
seen = []


class Stub(types.ModuleType):
    __path__ = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return MagicMock()


class Probe(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] not in HEAVY:
            return None
        seen.append([name, os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), os.environ.get("PYTORCH_ALLOC_CONF")])
        return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        return Stub(spec.name)

    def exec_module(self, module):
        return None


sys.meta_path.insert(0, Probe())
try:
    importlib.import_module(sys.argv[1])
except Exception:
    pass
print(json.dumps(seen))
"""

_ENTRY_POINTS = [
    "zvisiongenerator.backends.cuda.quant",
    "zvisiongenerator.backends.cuda.image.backend",
    "zvisiongenerator.backends.cuda.video.backend",
    "zvisiongenerator.backends.cuda.video.h3.backend",
    "zvisiongenerator.backends.cuda.seedvr2",
    "zvisiongenerator.backends.cuda.prompt_enhancer",
    "zvisiongenerator.backends.audio.yue2",
    "zvisiongenerator.backends.audio.qwen3_tts",
    "zvisiongenerator.backends.audio.sa3",
    "zvisiongenerator.audio_runner",
    "zvisiongenerator.web",
    "zvisiongenerator.cli",
]


def _heavy_imports_seen(module: str) -> list[list[str | None]]:
    env = {k: v for k, v in os.environ.items() if k not in {"PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"}}
    result = subprocess.run([sys.executable, "-c", _PROBE, module], cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestAllocatorBeforeHeavyImports:
    @pytest.mark.parametrize("module", _ENTRY_POINTS)
    def test_the_options_are_set_when_the_first_heavy_library_is_imported(self, module):
        seen = _heavy_imports_seen(module)

        for name, cuda_conf, alloc_conf in seen:
            if sys.platform == "darwin":
                assert cuda_conf is None and alloc_conf is None, name
            else:
                assert cuda_conf is not None, name
                assert ("expandable_segments:True" in cuda_conf) is (sys.platform != "win32"), name
                assert alloc_conf == cuda_conf, name

    @pytest.mark.skipif(sys.platform == "darwin", reason="the CUDA backend is not imported on macOS")
    def test_the_probe_sees_the_libraries_the_image_backend_imports(self):
        """Guards the probe itself: without these imports seen, the test above would prove nothing."""
        names = {name.split(".")[0] for name, _, _ in _heavy_imports_seen("zvisiongenerator.backends.cuda.image.backend")}

        assert {"torch", "diffusers"} <= names


class TestConfigureAcceleratorEnvironment:
    @pytest.mark.parametrize(("platform", "expected_calls"), [("linux", 1), ("win32", 1), ("darwin", 0)])
    def test_only_windows_and_linux_set_the_allocator(self, monkeypatch, platform, expected_calls):
        configure = MagicMock()
        monkeypatch.setattr(sys, "platform", platform)
        monkeypatch.setattr(memory_module, "configure_allocator", configure)

        backends._configure_accelerator_environment()

        assert configure.call_count == expected_calls

    def test_an_unsupported_platform_is_left_alone(self, monkeypatch):
        configure = MagicMock()
        monkeypatch.setattr(sys, "platform", "freebsd")
        monkeypatch.setattr(memory_module, "configure_allocator", configure)

        backends._configure_accelerator_environment()

        configure.assert_not_called()


class TestWarnWhenCudaStartedFirst:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)

    def test_warns_when_cuda_is_already_initialised(self, monkeypatch):
        torch = MagicMock()
        torch.cuda.is_initialized.return_value = True
        monkeypatch.setitem(sys.modules, "torch", torch)

        with pytest.warns(UserWarning, match="before the allocator options were set"):
            memory_module.configure_allocator()

    def test_stays_quiet_when_cuda_has_not_started(self, monkeypatch):
        torch = MagicMock()
        torch.cuda.is_initialized.return_value = False
        monkeypatch.setitem(sys.modules, "torch", torch)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            memory_module.configure_allocator()

    def test_does_not_import_torch_to_find_out(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        monkeypatch.setitem(sys.modules, "torch", None)  # an import would now raise

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            memory_module.configure_allocator()
