"""Tests for the silencing of known library noise: only the named messages go, everything else stays visible."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import warnings
from types import SimpleNamespace

import pytest

from zvisiongenerator.utils import library_noise
from zvisiongenerator.utils.library_noise import MUTED_ERROR_TEXT, mute_library_warnings, quiet_flash_attn_probe, quiet_qwen_graph_capture, quiet_qwen_import, silence_autotune_report


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


class TestQwenGraphCapture:
    def test_the_capture_lines_are_dropped_and_other_output_is_shown(self, capsys):
        with quiet_qwen_graph_capture():
            print("Warming up predictor (3 runs)...")
            print("Capturing CUDA graph for predictor...")
            print("CUDA graph captured!")
            print("Warming up talker graph (3 runs)...")
            print("Capturing CUDA graph for talker decode...")
            print("Talker CUDA graph captured!")
            print("a real message")

        assert capsys.readouterr().out == "a real message\n"


def _import_like_qwen_tts() -> None:
    """Do what importing ``qwen_tts`` does: ``sox`` probes with ``os.popen`` and warns when nothing comes back, and the flash_attn banner is printed."""
    sox_logger = logging.getLogger("sox")
    sox_logger.setLevel(logging.WARNING)  # what importing sox does to its own logger
    if not os.popen("sox -h").readlines():
        sox_logger.warning("SoX could not be found!")
    print("\n********\nWarning: flash-attn is not installed. Will only run the manual PyTorch version. Please install flash-attn for faster inference.\n********\n ")


@pytest.fixture
def no_sox(monkeypatch):
    """Make this machine one without the ``sox`` program, and make a real ``os.popen`` fail the test instead of starting a shell."""
    monkeypatch.setattr(shutil, "which", lambda name, *args, **kwargs: None)

    def popen(command, *args, **kwargs):
        raise AssertionError(f"a shell was started for {command!r}")

    monkeypatch.setattr(os, "popen", popen)
    return popen


class TestQwenImport:
    def test_the_known_messages_are_dropped(self, capfd, no_sox):
        with quiet_qwen_import():
            _import_like_qwen_tts()

        seen = capfd.readouterr()
        assert seen.out == "" and seen.err == ""

    def test_a_missing_sox_program_starts_no_shell_and_reads_as_nothing_printed(self, no_sox):
        with quiet_qwen_import():
            pipe = os.popen("sox -h")
            assert pipe.readlines() == []
            pipe.close()

    def test_an_installed_sox_program_is_probed_as_usual(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name, *args, **kwargs: "/usr/bin/sox")
        probed: list[str] = []
        monkeypatch.setattr(os, "popen", lambda command, *args, **kwargs: probed.append(command))

        with quiet_qwen_import():
            os.popen("sox -h")

        assert probed == ["sox -h"]

    def test_other_writes_to_the_error_stream_pass_through_untouched(self, capfd, no_sox):
        with quiet_qwen_import():
            _import_like_qwen_tts()
            os.write(2, b"CUDA error: something real\n")
            print("a warning", file=sys.stderr)

        seen = capfd.readouterr()
        assert seen.err == "CUDA error: something real\na warning\n"

    def test_an_unexpected_line_on_standard_output_is_shown_after_the_block(self, capfd, no_sox):
        with quiet_qwen_import():
            _import_like_qwen_tts()
            print("loading tokenizer")

        assert capfd.readouterr().out == "loading tokenizer\n"

    def test_a_failure_inside_the_block_restores_popen_and_shows_what_was_printed_before_it(self, capfd, no_sox):
        with pytest.raises(ImportError, match="boom"), quiet_qwen_import():
            print("half way")
            raise ImportError("boom")

        assert os.popen is no_sox
        assert capfd.readouterr().out == "half way\n"

    def test_popen_and_the_sox_logger_are_restored(self, no_sox):
        logger = logging.getLogger("sox")
        filters = list(logger.filters)

        with quiet_qwen_import():
            assert os.popen is not no_sox
            assert len(logger.filters) == len(filters) + 1

        assert os.popen is no_sox
        assert logger.filters == filters

    def test_a_sox_error_is_still_logged(self, no_sox):
        records: list[logging.LogRecord] = []

        class Collect(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = Collect()
        logger = logging.getLogger("sox")
        logger.addHandler(handler)
        try:
            with quiet_qwen_import():
                logger.warning("muted")
                logger.error("kept")
        finally:
            logger.removeHandler(handler)

        assert [record.getMessage() for record in records] == ["kept"]


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
