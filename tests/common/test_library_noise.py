"""Tests for the silencing of known library noise: only the named messages go, everything else stays visible."""

from __future__ import annotations

import sys
import warnings
from types import SimpleNamespace

import pytest

from zvisiongenerator.utils import library_noise
from zvisiongenerator.utils.library_noise import MUTED_ERROR_TEXT, mute_library_warnings, quiet_flash_attn_probe, silence_autotune_report


class TestMutedWarnings:
    def test_the_weight_norm_deprecation_is_ignored_and_other_future_warnings_are_not(self):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            mute_library_warnings()
            warnings.warn("`torch.nn.utils.weight_norm` is deprecated in favor of parametrizations.", FutureWarning, stacklevel=1)
            warnings.warn("`torch.jit.script` is not supported", FutureWarning, stacklevel=1)

        assert [str(warning.message) for warning in seen] == ["`torch.jit.script` is not supported"]

    def test_the_known_log_line_is_named(self):
        assert any("use_return_dict" in text for text in MUTED_ERROR_TEXT)


class TestFlashAttnProbe:
    def test_the_probe_lines_are_dropped_and_other_output_is_shown(self, capsys):
        with quiet_flash_attn_probe():
            print("No module named 'flash_attn'")
            print("flash_attn not installed, disabling Flash Attention")
            print("No module named 'flash_attn'")
            print("flash_attn varlen/bert_padding not available, disabling varlen attention")
            print("a real message")

        assert capsys.readouterr().out == "a real message\n"

    def test_a_failure_inside_the_block_still_shows_what_was_printed_before_it(self, capsys):
        with pytest.raises(ImportError, match="boom"), quiet_flash_attn_probe():
            print("progress of the import")
            raise ImportError("boom")

        assert capsys.readouterr().out == "progress of the import\n"

    def test_standard_output_is_restored(self):
        before = sys.stdout

        with quiet_flash_attn_probe():
            assert sys.stdout is not before

        assert sys.stdout is before


class TestAutotuneReport:
    def test_the_tables_are_turned_off_in_the_inductor_config(self, monkeypatch):
        config = SimpleNamespace(max_autotune_report_choices_stats=True, autotune_num_choices_displayed=10)
        monkeypatch.setitem(sys.modules, "torch._inductor.config", config)

        silence_autotune_report()

        assert config.max_autotune_report_choices_stats is False and config.autotune_num_choices_displayed == 0

    def test_a_setting_this_pytorch_lacks_is_not_invented(self, monkeypatch):
        config = SimpleNamespace(max_autotune_report_choices_stats=True)
        monkeypatch.setitem(sys.modules, "torch._inductor.config", config)

        silence_autotune_report()

        assert not hasattr(config, "autotune_num_choices_displayed")

    def test_a_pytorch_without_the_inductor_is_left_alone(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch._inductor.config", None)

        silence_autotune_report()

    def test_the_module_loads_no_heavy_library(self):
        assert "torch" not in vars(library_noise)
