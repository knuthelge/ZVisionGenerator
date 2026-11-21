"""Tests for the NVML card-memory reader, with the library faked (no real NVML)."""

from __future__ import annotations

import ctypes

import pytest

from zvisiongenerator.backends.cuda import nvml


class _FakeLibrary:
    """Stand in for the NVML shared library: records the calls and fills ``nvmlMemory_t`` with fixed values."""

    def __init__(self, *, init: int = 0, handle: int = 0, memory: int = 0) -> None:
        self.codes = {"init": init, "handle": handle, "memory": memory}
        self.bus_id: bytes | None = None

    def nvmlInit_v2(self) -> int:  # noqa: N802 - NVML's own name
        return self.codes["init"]

    def nvmlDeviceGetHandleByPciBusId_v2(self, bus_id: bytes, _out) -> int:  # noqa: N802
        self.bus_id = bus_id
        return self.codes["handle"]

    def nvmlDeviceGetMemoryInfo(self, _handle, out) -> int:  # noqa: N802
        info = out._obj
        info.total, info.free, info.used = 10_000, 3_000, 7_000
        return self.codes["memory"]


class _FakeDriver:
    """Stand in for the CUDA driver library: cards by PCI bus id, in CUDA's order, with the codes each call returns."""

    def __init__(self, bus_ids: list[str], *, init: int = 0) -> None:
        self.bus_ids = bus_ids
        self.codes = {"init": init}
        self.calls: list[str] = []

    def cuInit(self, flags: int) -> int:  # noqa: N802 - the driver's own name
        self.calls.append("cuInit")
        return self.codes["init"]

    def cuDeviceGetCount(self, out) -> int:  # noqa: N802
        out._obj.value = len(self.bus_ids)
        return 0

    def cuDeviceGet(self, out, ordinal: int) -> int:  # noqa: N802
        out._obj.value = ordinal
        return 0 if ordinal < len(self.bus_ids) else 101

    def cuDeviceGetPCIBusId(self, buffer, length: int, device) -> int:  # noqa: N802
        buffer.value = self.bus_ids[device.value].encode()
        return 0


@pytest.fixture
def driver(monkeypatch):
    """Forget what an earlier test loaded, so each test starts the CUDA driver lookup and NVML again, with CUDA_VISIBLE_DEVICES unset."""
    monkeypatch.setattr(nvml, "_started", False)
    monkeypatch.setattr(nvml, "_library", None)
    monkeypatch.setattr(nvml, "_bus_id_known", False)
    monkeypatch.setattr(nvml, "_bus_id", None)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return monkeypatch


@pytest.fixture
def fresh_nvml(driver):
    """As ``driver``, with CUDA's card at 0000:01:00.0 unless a test says otherwise."""
    driver.setattr(nvml, "_ask_driver_for_bus_id", lambda: "0000:01:00.0")
    return driver


def test_a_working_library_gives_the_cards_total_and_free(fresh_nvml):
    library = _FakeLibrary()
    fresh_nvml.setattr(nvml, "_open_library", lambda: library)

    assert nvml.card_memory() == nvml.CardMemory(total=10_000, free=3_000)
    assert library.bus_id == b"0000:01:00.0"


def test_the_card_is_the_one_at_the_bus_id_cuda_names_cuda_0(fresh_nvml):
    fresh_nvml.setattr(nvml, "_ask_driver_for_bus_id", lambda: "0000:41:00.0")
    library = _FakeLibrary()
    fresh_nvml.setattr(nvml, "_open_library", lambda: library)

    assert nvml.card_memory() == nvml.CardMemory(total=10_000, free=3_000)
    assert library.bus_id == b"0000:41:00.0"


def test_without_a_card_for_cuda_nvml_is_not_opened(fresh_nvml):
    fresh_nvml.setattr(nvml, "_ask_driver_for_bus_id", lambda: None)
    fresh_nvml.setattr(nvml, "_open_library", lambda: pytest.fail("opened NVML"))

    assert nvml.card_memory() is None


def test_the_bus_id_is_what_the_cuda_driver_gives_its_first_card(driver):
    """CUDA's first card is the one at 0000:41:00.0 here, though NVML numbers it second: the order CUDA uses is not NVML's."""
    cuda = _FakeDriver(["0000:41:00.0", "0000:01:00.0"])
    driver.setattr(nvml.ctypes, "CDLL", lambda name: cuda)

    assert nvml.card_bus_id() == "0000:41:00.0"
    assert cuda.calls == ["cuInit"]


def test_the_driver_is_asked_once(driver):
    cuda = _FakeDriver(["0000:01:00.0"])
    driver.setattr(nvml.ctypes, "CDLL", lambda name: cuda)

    assert (nvml.card_bus_id(), nvml.card_bus_id()) == ("0000:01:00.0", "0000:01:00.0")
    assert cuda.calls == ["cuInit"]


@pytest.mark.parametrize("visible", ["", "  "])
def test_an_empty_cuda_visible_devices_is_no_card_and_the_driver_is_not_asked(driver, visible):
    driver.setenv("CUDA_VISIBLE_DEVICES", visible)
    driver.setattr(nvml.ctypes, "CDLL", lambda name: pytest.fail("loaded the CUDA driver"))
    driver.setattr(nvml, "_open_library", lambda: pytest.fail("opened NVML"))

    assert nvml.card_bus_id() is None
    assert nvml.card_memory() is None


def test_a_driver_that_sees_no_card_gives_none(driver):
    driver.setenv("CUDA_VISIBLE_DEVICES", "7")
    driver.setattr(nvml.ctypes, "CDLL", lambda name: _FakeDriver([]))

    assert nvml.card_bus_id() is None


def test_a_driver_that_fails_to_start_gives_none(driver):
    driver.setattr(nvml.ctypes, "CDLL", lambda name: _FakeDriver(["0000:01:00.0"], init=100))

    assert nvml.card_bus_id() is None


def test_a_machine_without_the_cuda_driver_gives_none(driver):
    def missing(name):
        raise OSError(name)

    driver.setattr(nvml.ctypes, "CDLL", missing)

    assert nvml.card_bus_id() is None


def test_a_card_nvml_does_not_know_gives_none(fresh_nvml):
    fresh_nvml.setattr(nvml, "_open_library", lambda: _FakeLibrary(handle=2))

    assert nvml.card_memory() is None


def test_no_library_gives_none(fresh_nvml):
    fresh_nvml.setattr(nvml, "_open_library", lambda: None)

    assert nvml.card_memory() is None


@pytest.mark.parametrize("codes", [{"init": 1}, {"handle": 2}, {"memory": 3}])
def test_a_failing_call_gives_none(fresh_nvml, codes):
    fresh_nvml.setattr(nvml, "_open_library", lambda: _FakeLibrary(**codes))

    assert nvml.card_memory() is None


def test_a_library_without_the_entry_points_gives_none(fresh_nvml):
    fresh_nvml.setattr(nvml, "_open_library", lambda: object())

    assert nvml.card_memory() is None


def test_the_library_is_started_once_even_when_it_failed(fresh_nvml):
    opened: list[int] = []
    fresh_nvml.setattr(nvml, "_open_library", lambda: opened.append(1))

    nvml.card_memory()
    nvml.card_memory()

    assert opened == [1]


def test_the_memory_struct_matches_nvml_memory_t():
    assert ctypes.sizeof(nvml._NvmlMemory) == 24
