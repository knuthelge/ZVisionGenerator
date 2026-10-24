"""Tests for the CUDA allocator configuration."""

from __future__ import annotations

import os

import pytest

from zvisiongenerator.backends.cuda.memory import PIN_ROUND_SETTING, configure_allocator, merged_alloc_conf

_DEFAULTS = "expandable_segments:True,garbage_collection_threshold:0.8"


class TestMergedAllocConf:
    def test_nothing_set_yields_only_the_added_option(self):
        assert merged_alloc_conf(None, None, PIN_ROUND_SETTING) == PIN_ROUND_SETTING

    def test_defaults_gain_the_added_option_last(self):
        assert merged_alloc_conf(_DEFAULTS, None, PIN_ROUND_SETTING) == f"{_DEFAULTS},{PIN_ROUND_SETTING}"

    def test_custom_options_are_kept(self):
        assert merged_alloc_conf("max_split_size_mb:128", None, PIN_ROUND_SETTING) == f"max_split_size_mb:128,{PIN_ROUND_SETTING}"

    def test_a_user_pin_value_is_kept(self):
        assert merged_alloc_conf(None, "pinned_max_round_threshold_mb:64", PIN_ROUND_SETTING) == "pinned_max_round_threshold_mb:64"
        assert merged_alloc_conf("pinned_max_round_threshold_mb:8,a:1", None, PIN_ROUND_SETTING) == "pinned_max_round_threshold_mb:8,a:1"

    def test_alloc_conf_overrides_cuda_conf_in_place(self):
        merged = merged_alloc_conf("a:1,b:2", "b:3,c:4", PIN_ROUND_SETTING)

        assert merged == f"a:1,b:3,c:4,{PIN_ROUND_SETTING}"

    def test_blank_pairs_and_whitespace_are_dropped(self):
        assert merged_alloc_conf(" a:1 ,, ", "", PIN_ROUND_SETTING) == f"a:1,{PIN_ROUND_SETTING}"

    def test_merging_its_own_result_changes_nothing(self):
        once = merged_alloc_conf(_DEFAULTS, None, PIN_ROUND_SETTING)

        assert merged_alloc_conf(_DEFAULTS, once, PIN_ROUND_SETTING) == once


class TestConfigureAllocator:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)

    def test_sets_the_same_string_in_both_variables(self):
        configure_allocator()

        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == f"{_DEFAULTS},{PIN_ROUND_SETTING}"
        assert os.environ["PYTORCH_ALLOC_CONF"] == os.environ["PYTORCH_CUDA_ALLOC_CONF"]

    def test_a_user_alloc_conf_alone_is_not_shadowed_by_the_defaults(self, monkeypatch):
        monkeypatch.setenv("PYTORCH_ALLOC_CONF", "max_split_size_mb:64")

        configure_allocator()

        expected = f"max_split_size_mb:64,{PIN_ROUND_SETTING}"
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == expected
        assert os.environ["PYTORCH_ALLOC_CONF"] == expected

    def test_a_user_pin_value_wins_in_both_variables(self, monkeypatch):
        monkeypatch.setenv("PYTORCH_ALLOC_CONF", "pinned_max_round_threshold_mb:32")

        configure_allocator()

        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "pinned_max_round_threshold_mb:32"
        assert os.environ["PYTORCH_ALLOC_CONF"] == "pinned_max_round_threshold_mb:32"

    def test_a_user_cuda_conf_is_left_as_written(self, monkeypatch):
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "custom:value")

        configure_allocator()

        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "custom:value"
        assert os.environ["PYTORCH_ALLOC_CONF"] == f"custom:value,{PIN_ROUND_SETTING}"

    def test_is_idempotent(self):
        configure_allocator()
        first = (os.environ["PYTORCH_CUDA_ALLOC_CONF"], os.environ["PYTORCH_ALLOC_CONF"])
        configure_allocator()

        assert (os.environ["PYTORCH_CUDA_ALLOC_CONF"], os.environ["PYTORCH_ALLOC_CONF"]) == first

    def test_torch_sees_the_pin_setting(self):
        """A subprocess with CUDA unavailable skips; otherwise the allocator's own settings include the pin."""
        import subprocess
        import sys

        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        code = (
            "from zvisiongenerator.backends.cuda.memory import configure_allocator\n"
            "configure_allocator()\n"
            "import torch\n"
            "torch.zeros(1, device='cuda')\n"
            "print(torch._C._accelerator_getAllocatorSettings())\n"
        )
        env = {k: v for k, v in os.environ.items() if k not in {"PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"}}
        result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120, check=True)

        assert "pinned_max_round_threshold_mb" in result.stdout
