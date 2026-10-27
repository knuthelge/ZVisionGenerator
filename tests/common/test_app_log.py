"""Tests for the per-command error log."""

from __future__ import annotations

import logging
import sys
import threading
import warnings

import pytest

from zvisiongenerator.utils import app_log


@pytest.fixture
def fresh_logging(monkeypatch, tmp_path):
    """Run setup_logging against a temp data dir, then restore global logging state and hooks."""
    root = logging.getLogger()
    saved = (list(root.handlers), root.level, sys.excepthook, threading.excepthook)
    monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("ZIV_LOG", raising=False)
    monkeypatch.setattr(app_log, "_configured", False)
    monkeypatch.setattr(app_log, "_log_path", None)
    yield tmp_path
    for handler in root.handlers:
        if handler not in saved[0]:
            handler.close()
    root.handlers[:] = saved[0]
    root.setLevel(saved[1])
    sys.excepthook, threading.excepthook = saved[2], saved[3]
    logging.captureWarnings(False)


def _flush() -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()


def test_log_file_is_cleared_on_start(fresh_logging):
    log_file = fresh_logging / "data" / "logs" / "ui.log"
    log_file.parent.mkdir(parents=True)
    log_file.write_text("old run\n")

    assert app_log.setup_logging("ui") == log_file
    assert log_file.read_text() == ""


def test_errors_and_warnings_are_logged_but_info_is_not(fresh_logging, capsys):
    path = app_log.setup_logging("image")
    logging.getLogger("zvisiongenerator.test").info("routine detail")
    warnings.warn("something looks off", UserWarning, stacklevel=1)
    try:
        raise PermissionError("denied")
    except PermissionError:
        logging.getLogger("zvisiongenerator.test").exception("Failed to load model")
    _flush()

    text = path.read_text()
    assert "routine detail" not in text
    assert "something looks off" in text
    assert "Failed to load model" in text and "PermissionError: denied" in text
    err = capsys.readouterr().err
    assert "ERROR: Failed to load model" in err
    assert "Traceback" not in err


def test_uncaught_exception_traceback_goes_to_file(fresh_logging, monkeypatch):
    printed = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: printed.append(exc))
    path = app_log.setup_logging("video")
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())
    _flush()

    assert "RuntimeError: boom" in path.read_text()
    assert printed  # the original hook still runs


def test_falls_back_to_temp_dir_when_data_dir_is_not_writable(fresh_logging, monkeypatch):
    blocker = fresh_logging / "data"
    blocker.write_text("not a directory")
    monkeypatch.setattr(app_log.tempfile, "gettempdir", lambda: str(fresh_logging / "tmp"))

    path = app_log.setup_logging("ui")

    assert path is not None and path.parent.parent == fresh_logging / "tmp"


def test_can_be_disabled(fresh_logging, monkeypatch):
    monkeypatch.setenv("ZIV_LOG", "off")
    assert app_log.setup_logging("ui") is None
    assert not (fresh_logging / "data" / "logs").exists()


def test_uvicorn_config_writes_server_errors_to_log_file(fresh_logging):
    path = app_log.setup_logging("ui")
    config = app_log.uvicorn_log_config()

    assert config["handlers"]["file"]["filename"] == str(path)
    assert config["handlers"]["file"]["level"] == "WARNING"
    assert "file" in config["loggers"]["uvicorn"]["handlers"]
    assert "file" not in config["loggers"]["uvicorn.access"]["handlers"]
