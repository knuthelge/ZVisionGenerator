"""Measure one job in this process: the platform's peak, on CUDA the driver's view of the card, on Linux the peaks of the process's memory counters, on Windows its private working set and on a Mac the process's lifetime peak footprint."""

from __future__ import annotations

import contextlib
import io
import logging
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from bench.memory.storage import DOWNLOAD_THRESHOLD_BYTES, model_storage_bytes
from bench.memory.windows import WindowsSampler

GIB = 1024**3
_RSS_KEYS = ("RssAnon", "RssFile", "RssShmem")
_ANON_SHMEM = "anon+shmem"
_ANON_SHMEM_FILE = "anon+shmem+file"


def platform_label() -> str:
    """Return ``"mac"`` on macOS and ``"cuda"`` elsewhere: the two platforms the estimate has tables for."""
    return "mac" if sys.platform == "darwin" else "cuda"


def rss_fields(status_text: str) -> dict[str, int]:
    """Return ``RssAnon``, ``RssFile`` and ``RssShmem`` in bytes from the text of ``/proc/<pid>/status``; a counter the kernel does not report is left out."""
    found: dict[str, int] = {}
    for line in status_text.splitlines():
        key, _, rest = line.partition(":")
        if key in _RSS_KEYS and rest.split():
            found[key] = int(rest.split()[0]) * 1024
    return found


def vm_hwm(status_text: str) -> int | None:
    """Return ``VmHWM`` (the process's resident-memory high-water mark) in bytes from the text of ``/proc/<pid>/status``, or ``None`` when it is not there."""
    for line in status_text.splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    return None


def read_status() -> str:
    """Return the text of ``/proc/self/status``, or an empty string where there is none (macOS, Windows)."""
    try:
        return Path("/proc/self/status").read_text(encoding="utf-8")
    except OSError:
        return ""


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


class ProcSampler:
    """Poll this process's ``RssAnon``, ``RssFile`` and ``RssShmem`` on a daemon thread, keeping the peaks.

    Besides each counter's own peak it keeps the peak of the counters' sum within one reading (``rss_anon_shmem_peak_bytes``, and with the file-backed
    pages ``rss_anon_shmem_file_peak_bytes``). The counters peak at different moments (a text encoder's anonymous memory, then pinned streamed weights),
    so the sum of their separate peaks overstates what the process held at once; the per-reading sum is what the RAM figure uses.

    The sampler does nothing where ``/proc`` has no such counters (macOS, Windows; :class:`~bench.memory.windows.WindowsSampler` covers Windows).
    """

    def __init__(self, interval: float = 0.1, read: Callable[[str], str | None] = _read_text) -> None:
        self._interval = interval
        self._read = read
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._peaks: dict[str, int] = {}

    def start(self) -> None:
        """Take the first reading and start polling."""
        self._sample()
        self._thread.start()

    def stop(self) -> dict[str, int | None]:
        """Stop polling, take a last reading, and return the peaks.

        Returns:
            ``rss_anon_peak_bytes``, ``rss_file_peak_bytes``, ``rss_shmem_peak_bytes``, ``rss_anon_shmem_peak_bytes`` (the highest anon + shared in one reading),
            and ``rss_anon_shmem_file_peak_bytes`` (the same with the file-backed pages), each ``None`` when unread.
        """
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        self._sample()
        return {f"{_snake(key)}_peak_bytes": self._peaks.get(key) for key in _RSS_KEYS} | {
            "rss_anon_shmem_peak_bytes": self._peaks.get(_ANON_SHMEM),
            "rss_anon_shmem_file_peak_bytes": self._peaks.get(_ANON_SHMEM_FILE),
        }

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample()

    def _sample(self) -> None:
        status = self._read("/proc/self/status")
        fields = rss_fields(status) if status else {}
        for key, value in fields.items():
            self._peaks[key] = max(self._peaks.get(key, 0), value)
        if "RssAnon" in fields:
            anon_shmem = fields["RssAnon"] + fields.get("RssShmem", 0)
            for key, value in ((_ANON_SHMEM, anon_shmem), (_ANON_SHMEM_FILE, anon_shmem + fields.get("RssFile", 0))):
                self._peaks[key] = max(self._peaks.get(key, 0), value)


_RUSAGE_INFO_V4 = 4
_RUSAGE_V4_COUNTERS = (
    "user_time",
    "system_time",
    "pkg_idle_wkups",
    "interrupt_wkups",
    "pageins",
    "wired_size",
    "resident_size",
    "phys_footprint",
    "proc_start_abstime",
    "proc_exit_abstime",
    "child_user_time",
    "child_system_time",
    "child_pkg_idle_wkups",
    "child_interrupt_wkups",
    "child_pageins",
    "child_elapsed_abstime",
    "diskio_bytesread",
    "diskio_byteswritten",
    "cpu_time_qos_default",
    "cpu_time_qos_maintenance",
    "cpu_time_qos_background",
    "cpu_time_qos_utility",
    "cpu_time_qos_legacy",
    "cpu_time_qos_user_initiated",
    "cpu_time_qos_user_interactive",
    "billed_system_time",
    "serviced_system_time",
    "logical_writes",
    "lifetime_max_phys_footprint",
    "instructions",
    "cycles",
    "billed_energy",
    "serviced_energy",
    "interval_max_phys_footprint",
    "runnable_time",
)


def _rusage_v4_struct() -> Any:
    """Return the kernel's ``rusage_info_v4`` struct type, as ``proc_pid_rusage`` fills it in."""
    import ctypes

    class RusageInfoV4(ctypes.Structure):
        _fields_ = [("ri_uuid", ctypes.c_uint8 * 16), *((f"ri_{name}", ctypes.c_uint64) for name in _RUSAGE_V4_COUNTERS)]

    return RusageInfoV4


def _rusage_v4() -> Any | None:
    """Return this process's ``rusage_info_v4`` as the kernel reports it, or ``None`` off macOS or when the kernel call fails."""
    if sys.platform != "darwin":
        return None
    import ctypes
    import ctypes.util
    import os

    try:
        info = _rusage_v4_struct()()
        if ctypes.CDLL(ctypes.util.find_library("System")).proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
            return None
    except OSError, AttributeError:
        return None
    return info


def lifetime_max_footprint() -> int | None:
    """Return the highest ``phys_footprint`` this process has reached, in bytes, or ``None`` off macOS or when the kernel call fails.

    The kernel keeps the process's lifetime maximum (``ri_lifetime_max_phys_footprint`` of ``RUSAGE_INFO_V4``), so one reading at the end of a case, which
    runs in a process of its own, is its exact peak: no polling, no missed spike. ``phys_footprint`` is the figure Activity Monitor shows as Memory.
    """
    info = _rusage_v4()
    return None if info is None else int(info.ri_lifetime_max_phys_footprint)


def current_footprint() -> int | None:
    """Return this process's ``phys_footprint`` right now, in bytes, or ``None`` off macOS or when the kernel call fails.

    The lifetime maximum cannot be reset, so a step in a longer-running process samples this value and keeps the highest reading.
    """
    info = _rusage_v4()
    return None if info is None else int(info.ri_phys_footprint)


def smi_memory(run: Callable[[list[str]], str] | None = None) -> tuple[int, int] | None:
    """Return ``(used, total)`` bytes of the card CUDA uses as ``cuda:0`` as ``nvidia-smi`` reports them, or ``None`` without it or without a card.

    The card is the one at the PCI bus id the CUDA driver names (:func:`zvisiongenerator.backends.cuda.nvml.card_bus_id`), which starts no CUDA context here.
    ``used`` counts every process on the card, so read before this job starts CUDA it is what the others hold: the baseline. Without *run* the real ``nvidia-smi`` runs.
    """
    from zvisiongenerator.backends.cuda import nvml

    def _run(command: list[str]) -> str:
        return subprocess.run(command, check=True, capture_output=True, text=True, timeout=10).stdout

    bus_id = nvml.card_bus_id()
    if bus_id is None:
        return None
    try:
        line = (run or _run)(["nvidia-smi", "-i", bus_id, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]).strip().splitlines()[0]
        used, total = (int(float(part)) for part in line.split(","))
    except OSError, subprocess.SubprocessError, ValueError, IndexError:
        return None
    return used * 1024 * 1024, total * 1024 * 1024


def _driver_memory() -> tuple[int, int] | None:
    """Return ``(free, total)`` from the CUDA driver while this process has CUDA initialised, else ``None``; PyTorch is never imported here."""
    torch = sys.modules.get("torch")
    try:
        if torch is None or not torch.cuda.is_initialized():
            return None
        free, total = torch.cuda.mem_get_info(0)
    except Exception:  # noqa: BLE001 -- torch still importing on the main thread, or a driver error: no sample
        return None
    return int(free), int(total)


class GpuSampler:
    """Poll the driver's free GPU memory on a daemon thread and keep the lowest reading.

    The driver's view counts what the allocator holds, the CUDA context, the driver's own reservation and every other
    process, so ``total - lowest free`` is what the card had to give at the job's tightest moment.
    """

    def __init__(self, interval: float = 0.02, read: Callable[[], tuple[int, int] | None] = _driver_memory) -> None:
        self._interval = interval
        self._read = read
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._lowest_free: int | None = None
        self._total: int | None = None

    def start(self) -> None:
        """Start polling."""
        self._thread.start()

    def stop(self) -> dict[str, int | None]:
        """Stop polling, take a last reading, and return ``gpu_free_min_bytes`` and ``gpu_driver_total_bytes`` (``None`` when CUDA never started)."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        self._sample()
        return {"gpu_free_min_bytes": self._lowest_free, "gpu_driver_total_bytes": self._total}

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample()

    def _sample(self) -> None:
        reading = self._read()
        if reading is None:
            return
        free, self._total = reading
        self._lowest_free = free if self._lowest_free is None else min(self._lowest_free, free)


def _snake(key: str) -> str:
    return "rss_" + key.removeprefix("Rss").lower()


def platform_peak() -> dict[str, int | None]:
    """Return this process's accelerator peak: CUDA ``max_memory_reserved`` and ``max_memory_allocated``, or MLX ``get_peak_memory`` on macOS.

    On CUDA ``platform_allocated_peak_bytes`` is the memory tensors held at the peak, which the estimate follows (the
    picture size and the references move it); ``platform_peak_bytes`` is the reserved peak, which also holds what the
    allocator keeps for reuse and the fragmentation between blocks. On a card shared with other programs the reserved
    peak runs into the card's limit and says little about the job. A framework the process never imported reads as ``None``;
    it is not imported here.
    """
    if sys.platform == "darwin":
        mx = sys.modules.get("mlx.core")
        return {"platform_peak_bytes": int(mx.get_peak_memory()) if mx is not None else None, "platform_allocated_peak_bytes": None, "gpu_total_bytes": None}
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return {"platform_peak_bytes": None, "platform_allocated_peak_bytes": None, "gpu_total_bytes": None}
    return {
        "platform_peak_bytes": int(torch.cuda.max_memory_reserved()),
        "platform_allocated_peak_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
    }


class _Tee(io.TextIOBase):
    """A text stream that passes what is written to *stream* and keeps a copy in *copy*."""

    def __init__(self, stream: Any, copy: io.StringIO) -> None:
        self._stream = stream
        self._copy = copy

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self._copy.write(text)
        return self._stream.write(text)

    def flush(self) -> None:
        self._stream.flush()


class _Records(logging.Handler):
    """A logging handler that keeps each record it gets as text, traceback included."""

    def __init__(self, copy: io.StringIO) -> None:
        super().__init__(level=logging.ERROR)
        self._copy = copy

    def emit(self, record: logging.LogRecord) -> None:
        self._copy.write(self.format(record) + "\n")


@contextlib.contextmanager
def captured_errors() -> Iterator[io.StringIO]:
    """Keep what this process writes to stderr and logs at error level while the block runs, and still show it.

    The video and audio CLIs report a failure (an out of memory error included) through their logger and exit with a code, so the
    exception never reaches the caller; its text is only in what they print.
    """
    copy = io.StringIO()
    handler = _Records(copy)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        with contextlib.redirect_stderr(_Tee(sys.stderr, copy)):
            yield copy
    finally:
        root.removeHandler(handler)


_OUT_OF_MEMORY = re.compile(r"out of (?:gpu |cuda |device )?memory|OutOfMemoryError", re.IGNORECASE)


def says_out_of_memory(text: str) -> bool:
    """Return whether *text* reports a device that ran out of memory: PyTorch's error (``CUDA out of memory``) or the app's own wording (``ran out of GPU memory``)."""
    return _OUT_OF_MEMORY.search(text) is not None


def _out_of_memory_line(text: str) -> str | None:
    """Return the first line of *text* that says a device ran out of memory, or ``None``."""
    return next((line.strip() for line in text.splitlines() if says_out_of_memory(line)), None)


def call_cli(run_cli: Callable[[list[str]], None], argv: list[str]) -> str | None:
    """Run the CLI with *argv* through *run_cli* and return why it failed, or ``None`` when it finished.

    A non-zero exit, and any exception (out of memory included), is a reason: a bench records every failure and moves on. A CLI that
    catches an out of memory error, reports it on stderr or in its log and exits non-zero (the video and audio CLIs do) gives that line
    in its reason, so the row reads as an out of memory failure and not as a setup failure.
    """
    with captured_errors() as captured:
        try:
            run_cli(argv)
        except SystemExit as exit_:
            if exit_.code not in (None, 0):
                reason = f"the CLI exited with {exit_.code!r}"
                line = _out_of_memory_line(captured.getvalue())
                return reason if line is None else f"{reason}: {line}"
        except Exception as exc:  # noqa: BLE001 -- a bench records every failure (out of memory included) and moves on
            return f"{type(exc).__name__}: {exc}"
    return None


def run_case(
    case: Mapping[str, Any],
    argv: list[str],
    run_cli: Callable[[list[str]], None],
    out_dir: Path,
    *,
    estimate: Callable[[Mapping[str, Any]], dict[str, Any] | None],
    outputs: frozenset[str],
    probes: Callable[[], dict[str, Any]] | None = None,
    hold: Callable[[], dict[str, Any]] | None = None,
    storage: Callable[[], int] = model_storage_bytes,
) -> dict[str, Any]:
    """Run the CLI with *argv* through *run_cli* in this process and return the case's measurement row.

    The row is a failure, with the reason, when the CLI exits non-zero, raises, or writes no file of the medium. It records ``downloaded``:
    whether the model folders (the Hugging Face cache and the app's models folder) grew while the case ran, which means it fetched or built
    a model inside the measured process, so its peak and time overstate a run with the model in place.

    Args:
        case: The expanded case, with its size set by the medium's ``prepare``.
        argv: The CLI arguments from the medium's ``case_argv``.
        run_cli: Runs the CLI with those arguments; the bench passes a wrapper around the medium's ``main``.
        out_dir: Where the case writes its file.
        estimate: The medium's verdict for the case (:func:`bench.memory.image.estimate` and its siblings); it may return ``None``.
        outputs: The file suffixes the medium writes.
        probes: Reports what the bench's overrides saw (the loader's placement and warning); its fields join the row.
        hold: Takes the in-process GPU hold (:func:`bench.memory.holds.hold_in_process`) and returns its fields (``hold_charged_bytes``) for the row. It runs after the driver's baseline
            reading, so the baseline does not include the hold, and before the samplers start. A hold that cannot be taken is a setup failure and the job does not run.
        storage: Returns the bytes the model folders hold now.
    """
    try:
        estimated = estimate(case)
    except Exception:  # noqa: BLE001 -- a missing or unreadable model is a gap in the row, not a failed measurement
        estimated = None
    baseline = smi_memory()
    stored = storage()
    held: dict[str, Any] = {}
    error = None
    if hold is not None:
        try:
            held = hold()
        except Exception as exc:  # noqa: BLE001 -- a hold that cannot be taken is a failed row, not a crash
            # The text never reads as an out-of-memory failure, which `fit` takes for a bracket's edge: a hold that could not be taken is a setup failure.
            error = f"the in-process hold could not be taken: {_OUT_OF_MEMORY.sub('no room', f'{type(exc).__name__}: {exc}')}"
    sampler, gpu_sampler = ProcSampler(), GpuSampler()
    windows = WindowsSampler() if sys.platform == "win32" else None
    sampler.start()
    gpu_sampler.start()
    if windows is not None:
        windows.start()
    started = time.monotonic()
    if error is None:
        error = call_cli(run_cli, argv)
    seconds = round(time.monotonic() - started, 1)
    proc = sampler.stop() | ({} if windows is None else windows.stop())
    downloaded = storage() - stored > DOWNLOAD_THRESHOLD_BYTES
    footprint = lifetime_max_footprint()
    high_water = vm_hwm(read_status())
    gpu = gpu_sampler.stop() | {"gpu_baseline_used_bytes": None if baseline is None else baseline[0], "gpu_smi_total_bytes": None if baseline is None else baseline[1]}
    if error is None and not any(path.suffix.lower() in outputs for path in out_dir.glob("*")):
        error = "no output file was written"
    if error is None and case.get("placement") and probes is not None and probes()["loaded_placement"] is None:
        error = f"the loader never chose a placement, so {case['placement']!r} was not forced"
    return (
        {key: value for key, value in case.items() if key != "prompt"}
        | {"platform": platform_label(), "os": sys.platform, "status": "failed" if error else "ok", "error": error, "seconds": seconds, "downloaded": downloaded, "estimate": estimated}
        | platform_peak()
        | proc
        | {"vm_hwm_bytes": high_water}
        | held
        | gpu
        | ({} if footprint is None else {"footprint_peak_bytes": footprint})
        | ({} if probes is None else probes())
    )
