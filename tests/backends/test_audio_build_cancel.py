"""A Stop reaches the middle of a model build: every layer the library creates checks it, in the loading thread only."""

from __future__ import annotations

import threading

import pytest
from torch import nn

from zvisiongenerator.backends.audio.build_cancel import stop_while_building
from zvisiongenerator.core.audio_backend import AudioCancelled


class TestStopWhileBuilding:
    def test_a_stop_ends_the_build_at_the_next_layer(self):
        stopped = [False]
        built: list[str] = []

        def build() -> None:
            model = nn.Module()
            model.first = nn.Linear(1, 1)
            built.append("first")
            stopped[0] = True
            model.second = nn.Linear(1, 1)
            built.append("second")

        with pytest.raises(AudioCancelled), stop_while_building(lambda: stopped[0]):
            build()

        assert built == ["first"]

    def test_a_stop_is_seen_when_a_parameter_is_registered(self):
        module, weight = nn.Module(), nn.Parameter(nn.Linear(1, 1).weight.data)

        with pytest.raises(AudioCancelled), stop_while_building(lambda: True):
            module.register_parameter("weight", weight)

    def test_a_build_runs_through_while_nothing_is_asked(self):
        with stop_while_building(lambda: False):
            model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

        assert len(model) == 2

    def test_without_a_stop_check_nothing_is_hooked(self):
        with stop_while_building(None):
            nn.Linear(1, 1)

    def test_another_thread_building_a_module_is_never_stopped(self):
        failures: list[BaseException] = []

        def build_elsewhere() -> None:
            try:
                nn.Sequential(nn.Linear(1, 1))
            except BaseException as error:  # noqa: BLE001 - the test records whatever the thread raises
                failures.append(error)

        with stop_while_building(lambda: True):
            thread = threading.Thread(target=build_elsewhere)
            thread.start()
            thread.join()

        assert failures == []

    def test_the_hooks_are_removed_after_a_stop_too(self):
        parent, child = nn.Module(), nn.Linear(1, 1)

        with pytest.raises(AudioCancelled), stop_while_building(lambda: True):
            parent.add_module("child", child)

        nn.Sequential(nn.Linear(1, 1))  # would raise if a hook were left behind
