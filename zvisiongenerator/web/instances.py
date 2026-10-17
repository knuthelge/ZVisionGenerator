"""Record running Web UI servers so ``ziv ui --background`` and ``--background-stop`` can find them.

Every Web UI server, in the foreground or the background, writes ``<data dir>/run/ui-<pid>.json`` while it
runs and removes it on exit. A record only counts while its process is alive and has the recorded start time,
so a process ID the system has reused for another program is never touched. Records of servers that died
without cleaning up are deleted when the records are read.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import warnings

import psutil

from zvisiongenerator.utils.atomic_write import write_text_atomic

# On Linux a process's start time is derived from the boot time, which shifts when the system clock is corrected.
# A reused process ID still never matches: it would also have to start within these seconds of the server.
_START_TIME_TOLERANCE = 5.0


@dataclass(frozen=True)
class UiInstance:
    """A running Web UI server as recorded in its instance file."""

    pid: int
    started: float
    host: str
    port: int
    path: Path
    log: str | None = None
    launch_id: str | None = None


def instances_dir(data_dir: Path) -> Path:
    """Return the directory holding the instance records under *data_dir*."""
    return data_dir / "run"


def register_instance(directory: Path, *, host: str, port: int, log: Path | None = None, launch_id: str | None = None) -> Path | None:
    """Record the current process as a Web UI server listening on *host*:*port*.

    Returns:
        The record's path, or ``None`` when it could not be written (the server still runs, it just cannot be
        found by ``--background-stop``).
    """
    pid = os.getpid()
    path = directory / f"ui-{pid}.json"
    record = {
        "pid": pid,
        "started": psutil.Process(pid).create_time(),
        "host": host,
        "port": port,
        "log": str(log) if log is not None else None,
        "launch_id": launch_id,
    }
    try:
        directory.mkdir(parents=True, exist_ok=True)
        write_text_atomic(path, json.dumps(record))
    except OSError as exc:
        warnings.warn(f"Could not record the Web UI in {path}, so --background-stop will not find it: {exc}", stacklevel=2)
        return None
    return path


def unregister_instance(path: Path | None) -> None:
    """Remove an instance record; a missing record is fine."""
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def list_instances(directory: Path) -> list[UiInstance]:
    """Return the Web UI servers that are still running, deleting records of servers that are gone."""
    if not directory.is_dir():
        return []
    running: list[UiInstance] = []
    for path in sorted(directory.glob("ui-*.json")):
        instance = _read_instance(path)
        if instance is not None and _process_of(instance) is not None:
            running.append(instance)
        else:
            unregister_instance(path)
    return running


def stop_instances(instances: list[UiInstance], *, timeout: float = 10.0) -> list[UiInstance]:
    """Stop the given servers and everything they started; return the ones that were stopped.

    Each server is asked to shut down first. Whatever is still running after *timeout* seconds, including
    processes the servers started, is killed.
    """
    stopped: list[UiInstance] = []
    servers: list[psutil.Process] = []
    started_by_servers: list[psutil.Process] = []
    for instance in instances:
        process = _process_of(instance)
        if process is None:
            unregister_instance(instance.path)
            continue
        try:
            started_by_servers.extend(process.children(recursive=True))
            process.terminate()
        except psutil.NoSuchProcess:
            pass
        except psutil.Error as exc:
            warnings.warn(f"Could not stop the Web UI with process ID {instance.pid}: {exc}", stacklevel=2)
            continue
        servers.append(process)
        stopped.append(instance)

    _, alive = psutil.wait_procs(servers, timeout=timeout)
    leftovers = [process for process in [*alive, *started_by_servers] if process.is_running()]
    for process in leftovers:
        try:
            process.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(leftovers, timeout=5.0)

    for instance in stopped:
        unregister_instance(instance.path)
    return stopped


def _read_instance(path: Path) -> UiInstance | None:
    """Parse an instance record, or return ``None`` when it is unreadable or malformed."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        return UiInstance(
            pid=int(record["pid"]),
            started=float(record["started"]),
            host=str(record["host"]),
            port=int(record["port"]),
            path=path,
            log=record.get("log"),
            launch_id=record.get("launch_id"),
        )
    except OSError, ValueError, TypeError, KeyError:
        return None


def _process_of(instance: UiInstance) -> psutil.Process | None:
    """Return the live process behind *instance*, or ``None`` when it exited or its ID now belongs to another program."""
    try:
        process = psutil.Process(instance.pid)
        if abs(process.create_time() - instance.started) > _START_TIME_TOLERANCE:
            return None
        if process.status() == psutil.STATUS_ZOMBIE:
            return None
    except psutil.Error:
        return None
    return process


__all__ = ["UiInstance", "instances_dir", "list_instances", "register_instance", "stop_instances", "unregister_instance"]
