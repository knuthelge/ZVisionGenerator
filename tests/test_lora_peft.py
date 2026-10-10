"""Tests for zvisiongenerator.backends.lora_peft."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from zvisiongenerator.backends.lora_peft import is_unmatched_adapter_error, restore_cpu_offload, warn_unmatched_lora


class TestIsUnmatchedAdapterError:
    def test_matches_peft_no_matching_module_error(self, fake_peft_no_match):
        assert is_unmatched_adapter_error(fake_peft_no_match("no match"))

    def test_other_errors_do_not_match(self, fake_peft_no_match):
        assert not is_unmatched_adapter_error(ValueError("leftover keys"))

    def test_without_peft_loaded_nothing_matches(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "peft", raising=False)

        assert not is_unmatched_adapter_error(ValueError("no match"))

    def test_peft_without_the_error_class_matches_nothing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "peft", MagicMock(spec=[]))

        assert not is_unmatched_adapter_error(ValueError("no match"))


def test_warn_unmatched_lora_names_the_file():
    with pytest.warns(UserWarning, match=r"LoRA style\.safetensors: none of its LoRA tensors match this model"):
        warn_unmatched_lora("/loras/style.safetensors")


class TestRestoreCpuOffload:
    @staticmethod
    def _pipeline(hook):
        pipeline = MagicMock()
        pipeline.transformer._hf_hook = hook
        return pipeline

    def test_enables_offload_when_the_hook_is_gone(self):
        pipeline = self._pipeline(None)

        restore_cpu_offload(pipeline, enabled=True)

        pipeline.enable_model_cpu_offload.assert_called_once_with()

    def test_keeps_a_hook_that_is_present(self):
        pipeline = self._pipeline(object())

        restore_cpu_offload(pipeline, enabled=True)

        pipeline.enable_model_cpu_offload.assert_not_called()

    def test_does_nothing_when_offload_was_not_enabled(self):
        pipeline = self._pipeline(None)

        restore_cpu_offload(pipeline, enabled=False)

        pipeline.enable_model_cpu_offload.assert_not_called()

    def test_pipeline_without_a_transformer_is_left_alone(self):
        pipeline = MagicMock(spec=["enable_model_cpu_offload"])

        restore_cpu_offload(pipeline, enabled=True)

        pipeline.enable_model_cpu_offload.assert_not_called()
