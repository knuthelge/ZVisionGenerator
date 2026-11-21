"""Tests for the memory sampler; the sampling thread is synchronised with events, never sleeps."""

from __future__ import annotations

import threading

from zvisiongenerator.core.memory_sampler import MemorySampler, MemoryUsage

_LONG_INTERVAL = 3600.0  # the sampling thread never reads a second time while a test runs
_TIMEOUT = 5.0


class TestStartAndStop:
    def test_start_publishes_one_reading_and_stop_publishes_nothing(self):
        readings = iter([MemoryUsage(100), MemoryUsage(300)])
        published: list[MemoryUsage] = []
        sampler = MemorySampler(lambda: next(readings), published.append, interval=_LONG_INTERVAL)

        sampler.start()
        assert published == [MemoryUsage(100)]
        sampler.stop()

        assert published == [MemoryUsage(100)]

    def test_stop_takes_no_reading(self):
        reads: list[int] = []

        def read() -> MemoryUsage:
            reads.append(1)
            return MemoryUsage(len(reads))

        sampler = MemorySampler(read, lambda usage: None, interval=_LONG_INTERVAL)
        sampler.start()
        reads_after_start = len(reads)

        sampler.stop()
        sampler.stop()

        assert len(reads) == reads_after_start

    def test_stop_before_start_keeps_the_sampler_from_publishing(self):
        published: list[MemoryUsage] = []
        sampler = MemorySampler(lambda: MemoryUsage(1), published.append, interval=_LONG_INTERVAL)

        sampler.stop()
        sampler.start()

        assert published == []

    def test_a_reader_with_nothing_to_report_publishes_nothing(self):
        published: list[MemoryUsage] = []
        sampler = MemorySampler(lambda: None, published.append, interval=_LONG_INTERVAL)

        sampler.start()
        sampler.stop()

        assert published == []


class TestSamplingThread:
    def test_it_publishes_each_reading_in_order_until_the_reader_fails(self):
        """Reads after the fifth raise, so the thread ends by itself and every later outcome is fixed."""
        fifth_published = threading.Event()
        calls: list[int] = []
        published: list[MemoryUsage] = []

        def read() -> MemoryUsage:
            calls.append(1)
            if len(calls) > 5:
                raise RuntimeError("done")
            return MemoryUsage(len(calls) * 10)

        def publish(usage: MemoryUsage) -> None:
            published.append(usage)
            if len(published) == 5:
                fifth_published.set()

        sampler = MemorySampler(read, publish, interval=0)
        sampler.start()

        assert fifth_published.wait(_TIMEOUT)
        sampler.stop()

        assert published == [MemoryUsage(n * 10) for n in range(1, 6)]

    def test_a_reader_that_signals_on_its_nth_call_lets_the_test_stop_the_loop(self):
        third_read = threading.Event()
        calls: list[int] = []

        def read() -> MemoryUsage:
            calls.append(1)
            if len(calls) == 3:
                third_read.set()
            return MemoryUsage(7)

        sampler = MemorySampler(read, lambda usage: None, interval=0)
        sampler.start()

        assert third_read.wait(_TIMEOUT)
        sampler.stop()

    def test_a_failing_publish_ends_sampling(self):
        failed = threading.Event()
        published: list[MemoryUsage] = []

        def read() -> MemoryUsage:
            return MemoryUsage(len(published) + 1)

        def publish(usage: MemoryUsage) -> None:
            published.append(usage)
            if len(published) == 2:
                failed.set()
                raise RuntimeError("subscriber gone")

        sampler = MemorySampler(read, publish, interval=0)
        sampler.start()
        assert failed.wait(_TIMEOUT)

        sampler.stop()

        assert published == [MemoryUsage(1), MemoryUsage(2)]

    def test_a_reader_that_raises_at_start_ends_sampling_quietly(self):
        def read() -> MemoryUsage:
            raise RuntimeError("no counter")

        sampler = MemorySampler(read, lambda usage: None, interval=_LONG_INTERVAL)

        sampler.start()
        sampler.stop()
