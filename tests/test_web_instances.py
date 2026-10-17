"""Tests for the Web UI instance records and the background launcher."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import psutil
import pytest

from zvisiongenerator import web
from zvisiongenerator.web.instances import UiInstance, list_instances, register_instance, stop_instances, unregister_instance


def _write_record(directory, *, pid: int, started: float, port: int = 8080, launch_id: str | None = None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"ui-{pid}.json"
    path.write_text(json.dumps({"pid": pid, "started": started, "host": "127.0.0.1", "port": port, "log": None, "launch_id": launch_id}))
    return path


@pytest.fixture()
def sleeper():
    """Start a short-lived child process standing in for a Web UI server."""
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    yield process
    if process.poll() is None:
        process.kill()
    process.wait()


class TestInstanceRecords:
    def test_register_records_current_process_and_unregister_removes_it(self, tmp_path):
        path = register_instance(tmp_path, host="0.0.0.0", port=8123, launch_id="abc")

        [instance] = list_instances(tmp_path)
        assert (instance.pid, instance.host, instance.port, instance.launch_id) == (os.getpid(), "0.0.0.0", 8123, "abc")

        unregister_instance(path)
        assert list_instances(tmp_path) == []

    def test_register_warns_and_returns_none_when_directory_is_unwritable(self, tmp_path):
        blocker = tmp_path / "run"
        blocker.write_text("not a directory")

        with pytest.warns(UserWarning):
            assert register_instance(blocker, host="127.0.0.1", port=8080) is None

    def test_list_removes_records_of_exited_processes(self, tmp_path, sleeper):
        started = psutil.Process(sleeper.pid).create_time()
        path = _write_record(tmp_path, pid=sleeper.pid, started=started)
        sleeper.kill()
        sleeper.wait()

        assert list_instances(tmp_path) == []
        assert not path.exists()

    def test_list_ignores_reused_process_ids(self, tmp_path):
        path = _write_record(tmp_path, pid=os.getpid(), started=psutil.Process().create_time() - 100)

        assert list_instances(tmp_path) == []
        assert not path.exists()

    def test_list_removes_malformed_records(self, tmp_path):
        tmp_path.joinpath("ui-1.json").write_text("{not json")

        assert list_instances(tmp_path) == []
        assert not tmp_path.joinpath("ui-1.json").exists()

    def test_list_without_directory_is_empty(self, tmp_path):
        assert list_instances(tmp_path / "missing") == []


class TestStopInstances:
    def test_stops_running_server_and_removes_record(self, tmp_path, sleeper):
        _write_record(tmp_path, pid=sleeper.pid, started=psutil.Process(sleeper.pid).create_time())

        stopped = stop_instances(list_instances(tmp_path), timeout=5.0)

        assert [instance.pid for instance in stopped] == [sleeper.pid]
        assert sleeper.wait(timeout=5) is not None
        assert list_instances(tmp_path) == []

    def test_kills_server_that_ignores_the_shutdown_request(self, tmp_path):
        stubborn = subprocess.Popen(
            [sys.executable, "-c", "import signal, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(60)"],
            stdout=subprocess.PIPE,
        )
        try:
            stubborn.stdout.readline()
            _write_record(tmp_path, pid=stubborn.pid, started=psutil.Process(stubborn.pid).create_time())

            stopped = stop_instances(list_instances(tmp_path), timeout=0.5)

            assert [instance.pid for instance in stopped] == [stubborn.pid]
            assert stubborn.wait(timeout=5) is not None
        finally:
            if stubborn.poll() is None:
                stubborn.kill()
            stubborn.wait()
            stubborn.stdout.close()

    def test_skips_process_that_is_no_longer_the_server(self, tmp_path):
        path = _write_record(tmp_path, pid=os.getpid(), started=0.0)
        instance = UiInstance(pid=os.getpid(), started=0.0, host="127.0.0.1", port=8080, path=path)

        assert stop_instances([instance], timeout=0.1) == []
        assert not path.exists()


class TestRunServer:
    def test_records_server_while_running_and_bounds_shutdown(self, tmp_path):
        """The server is recorded while uvicorn runs, removed afterwards, and does not wait forever on open streams."""
        seen = []
        uvicorn = MagicMock()
        uvicorn.run.side_effect = lambda *args, **kwargs: seen.extend(list_instances(tmp_path / "run"))

        with (
            patch.dict(sys.modules, {"uvicorn": uvicorn}),
            patch.object(web, "_ensure_web_runtime_dependencies"),
            patch.object(web, "get_ziv_data_dir", return_value=tmp_path),
            patch.object(web, "_find_available_port", return_value=8099),
            patch.object(web, "_lan_address", return_value=None),
            patch.object(web, "uvicorn_log_config", return_value={}),
        ):
            web.run_server(port=8099, open_browser=False)

        assert [(instance.pid, instance.port) for instance in seen] == [(os.getpid(), 8099)]
        assert list_instances(tmp_path / "run") == []
        assert uvicorn.run.call_args.kwargs["timeout_graceful_shutdown"] > 0


class TestBackgroundLauncher:
    def test_spawns_detached_server_and_reports_address(self, tmp_path, capsys):
        process = MagicMock()
        process.poll.return_value = None
        instance = UiInstance(pid=4321, started=1.0, host="0.0.0.0", port=8081, path=tmp_path / "ui-4321.json", log="ui.log")

        with (
            patch.object(web, "get_ziv_data_dir", return_value=tmp_path),
            patch.object(web.subprocess, "Popen", return_value=process) as popen,
            patch.object(web, "_wait_for_background_server", return_value=instance),
            patch.object(web, "_lan_address", return_value=None),
            patch.object(web, "_open_browser") as open_browser,
        ):
            assert web.run_background(host="0.0.0.0", port=8080, open_browser=True) == 0

        command = popen.call_args.args[0]
        assert command[0] == sys.executable
        assert command[-5:] == ["--host", "0.0.0.0", "--port", "8080", "--no-browser"]
        options = popen.call_args.kwargs
        assert options["stdin"] == options["stdout"] == options["stderr"] == subprocess.DEVNULL
        assert options["env"][web._LAUNCH_ID_ENV]
        if sys.platform != "win32":
            assert options["start_new_session"] is True
        open_browser.assert_called_once_with("http://0.0.0.0:8081", "0.0.0.0", 8081)
        assert "http://0.0.0.0:8081" in capsys.readouterr().out

    def test_windows_server_gets_a_hidden_console(self, monkeypatch):
        """On Windows the background server has its own hidden console, so the programs it starts open no windows."""
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
        monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200, raising=False)
        monkeypatch.setattr(subprocess, "DETACHED_PROCESS", 0x00000008, raising=False)

        flags = web._detach_options()["creationflags"]

        assert flags & subprocess.CREATE_NO_WINDOW
        assert flags & subprocess.CREATE_NEW_PROCESS_GROUP
        assert not flags & subprocess.DETACHED_PROCESS

    def test_reports_failure_and_cleans_up_when_server_does_not_start(self, tmp_path):
        process = MagicMock()
        process.poll.return_value = None

        with (
            patch.object(web, "get_ziv_data_dir", return_value=tmp_path),
            patch.object(web.subprocess, "Popen", return_value=process),
            patch.object(web, "_wait_for_background_server", return_value=None),
            patch.object(web, "_open_browser") as open_browser,
        ):
            assert web.run_background(open_browser=True) == 1

        process.kill.assert_called_once_with()
        open_browser.assert_not_called()

    def test_wait_returns_matching_instance_once_reachable(self, tmp_path):
        process = MagicMock()
        process.poll.return_value = None
        mine = UiInstance(pid=1, started=1.0, host="127.0.0.1", port=8085, path=tmp_path / "a", launch_id="mine")
        other = UiInstance(pid=2, started=1.0, host="127.0.0.1", port=8084, path=tmp_path / "b", launch_id="other")

        with patch.object(web, "list_instances", return_value=[other, mine]), patch.object(web, "_wait_for_server", return_value=True) as wait:
            assert web._wait_for_background_server(process, tmp_path, "mine", timeout=1.0, delay=0.01) == mine

        wait.assert_called_once_with("127.0.0.1", 8085, attempts=1, delay=0.01)

    def test_wait_gives_up_when_server_process_exits(self, tmp_path):
        process = MagicMock()
        process.poll.return_value = 1

        with patch.object(web, "list_instances") as list_instances_mock:
            assert web._wait_for_background_server(process, tmp_path, "mine", timeout=1.0, delay=0.01) is None

        list_instances_mock.assert_not_called()


class TestStopAllServers:
    def test_succeeds_when_nothing_is_running(self, tmp_path):
        with patch.object(web, "get_ziv_data_dir", return_value=tmp_path), patch.object(web, "stop_instances") as stop:
            assert web.stop_all_servers() == 0

        stop.assert_not_called()

    def test_fails_when_a_server_could_not_be_stopped(self, tmp_path):
        first = UiInstance(pid=1, started=1.0, host="127.0.0.1", port=8080, path=tmp_path / "a")
        second = UiInstance(pid=2, started=1.0, host="127.0.0.1", port=8081, path=tmp_path / "b")

        with (
            patch.object(web, "get_ziv_data_dir", return_value=tmp_path),
            patch.object(web, "list_instances", return_value=[first, second]),
            patch.object(web, "stop_instances", return_value=[first]),
        ):
            assert web.stop_all_servers() == 1
