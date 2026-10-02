"""Error log: warnings, errors, and tracebacks written to a file that is cleared on every start.

Each command writes ``<data dir>/logs/<command>.log`` (e.g. ``~/.ziv/logs/ui.log``), so the file always
describes the latest run. Set ``ZIV_LOG=off`` to disable it. Logging must never stop the tool: if no log
file can be opened, problems are still shown on the terminal.
"""

from __future__ import annotations

import copy
import getpass
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

_FILE_FORMAT = "%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s"

_log_path: Path | None = None
_configured = False


def setup_logging(command: str) -> Path | None:
    """Start the error log for *command*; later calls in the same process are no-ops.

    Args:
        command: Short command name used for the file name, e.g. ``"ui"`` or ``"image"``.

    Returns:
        The log file path, or ``None`` when file logging is disabled or no log directory is writable.
    """
    global _configured, _log_path
    if _configured:
        return _log_path
    _configured = True
    if os.environ.get("ZIV_LOG", "").strip().lower() in {"off", "0", "false", "no"}:
        return None

    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(_ConsoleFormatter())
    root.addHandler(console)
    logging.captureWarnings(True)

    _log_path, open_errors = _open_log_file(command)
    if _log_path is not None:
        file_handler = logging.FileHandler(_log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        root.addHandler(file_handler)
    for error in open_errors:
        logging.getLogger(__name__).warning("Could not open log file %s", error)

    _install_exception_hooks()
    return _log_path


class _ConsoleFormatter(logging.Formatter):
    """One line per problem on the terminal; tracebacks go only to the log file."""

    def format(self, record: logging.LogRecord) -> str:
        return f"{record.levelname}: {record.getMessage()}"


def log_file_path() -> Path | None:
    """Return the current process's log file, if one is open."""
    return _log_path


def uvicorn_log_config() -> dict[str, Any]:
    """Return uvicorn's default logging config, with its server warnings and errors also written to the log file."""
    from uvicorn.config import LOGGING_CONFIG

    config = copy.deepcopy(LOGGING_CONFIG)
    if _log_path is None:
        return config
    config["formatters"]["file"] = {"format": _FILE_FORMAT}
    # Append mode: the file was truncated at startup and is shared with the root handler.
    config["handlers"]["file"] = {"class": "logging.FileHandler", "filename": str(_log_path), "mode": "a", "encoding": "utf-8", "formatter": "file", "level": "WARNING"}
    config["loggers"]["uvicorn"]["handlers"] = [*config["loggers"]["uvicorn"]["handlers"], "file"]
    return config


def _open_log_file(command: str) -> tuple[Path | None, list[str]]:
    """Create and truncate the log file in the first writable directory: the data dir, else the temp dir."""
    env = os.environ.get("ZIV_DATA_DIR", "").strip()
    data_dir = Path(env).expanduser() if env else Path.home() / ".ziv"
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 - only used to name the fallback directory
        user = "user"
    errors: list[str] = []
    for directory in (data_dir / "logs", Path(tempfile.gettempdir()) / f"ziv-logs-{user}"):
        path = directory / f"{command}.log"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        except OSError as exc:
            errors.append(f"{path}: {exc}")
            continue
        return path, errors
    return None, errors


def _install_exception_hooks() -> None:
    """Log uncaught exceptions with tracebacks, then defer to the previous hooks (which print them)."""
    log = logging.getLogger("zvisiongenerator.crash")
    previous_excepthook = sys.excepthook
    previous_thread_hook = threading.excepthook

    def _excepthook(exc_type, exc, tb) -> None:
        if not issubclass(exc_type, KeyboardInterrupt):
            # Straight to the file: the previous hook already prints the traceback on the terminal.
            _log_to_file_only(log, "Uncaught exception", (exc_type, exc, tb))
            if _log_path is not None:
                print(f"Details were written to {_log_path}", file=sys.stderr)
        previous_excepthook(exc_type, exc, tb)

    def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is not SystemExit:
            thread_name = args.thread.name if args.thread is not None else "unknown"
            _log_to_file_only(log, f"Uncaught exception in thread {thread_name}", (args.exc_type, args.exc_value, args.exc_traceback))
        previous_thread_hook(args)

    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook


def _log_to_file_only(log: logging.Logger, message: str, exc_info: Any) -> None:
    record = log.makeRecord(log.name, logging.CRITICAL, __file__, 0, message, (), exc_info)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.handle(record)
