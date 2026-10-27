"""Tests for the LTX checkpoint conversion routes and converter subprocesses."""

from __future__ import annotations

import contextlib
import subprocess
import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest


from zvisiongenerator.web import server as web_server
from zvisiongenerator.utils.model_memory import MemoryBudget


class TestLtxConversion:
    """LTX checkpoints convert as a background job that holds the quantize slot until it ends."""

    def _hold(self, monkeypatch):
        """Record the job instead of running it; return the started (args, slot) pairs."""
        started: list[tuple[list[str], object]] = []
        monkeypatch.setattr("zvisiongenerator.web.video_conversion.start_conversion", lambda args, name, slot: started.append((args, slot)))
        monkeypatch.setattr(web_server, "_run_model_management_command", lambda args: pytest.fail("LTX never runs as a blocking convert"))
        return started

    def test_the_conversion_starts_in_the_background_with_its_level_and_distill(self, monkeypatch, tmp_path):
        checkpoint = tmp_path / "ft.safetensors"
        checkpoint.write_bytes(b"x")
        started = self._hold(monkeypatch)

        with TestClient(web_server.app) as client:
            response = client.post("/api/models/convert", json={"input_path": str(checkpoint), "model_type": "ltx-2.3", "name": "myft", "quantize": "8", "distill": True})

        assert response.status_code == 200
        assert response.json()["background"] is True
        assert started[0][0] == ["model", "--input", str(checkpoint.absolute()), "--model-type", "ltx-2.3", "--name", "myft", "--quantize", "8", "--distill"]
        with pytest.raises(web_server.JobConflictError):
            web_server._reject_while_busy()  # the slot stays held while the job runs
        started[0][1].close()
        web_server._reject_while_busy()

    def test_a_quantize_level_is_required(self, monkeypatch, tmp_path):
        checkpoint = tmp_path / "ft.safetensors"
        checkpoint.write_bytes(b"x")
        started = self._hold(monkeypatch)

        with TestClient(web_server.app) as client:
            response = client.post("/api/models/convert", json={"input_path": str(checkpoint), "model_type": "ltx-2.3", "quantize": ""})

        assert response.status_code == 400
        assert started == []

    def test_it_waits_for_running_jobs(self, monkeypatch, tmp_path):
        checkpoint = tmp_path / "ft.safetensors"
        checkpoint.write_bytes(b"x")
        started = self._hold(monkeypatch)
        monkeypatch.setattr(web_server.web_runner, "_find_active_exclusive_job_id", lambda: "busy")

        with TestClient(web_server.app) as client:
            response = client.post("/api/models/convert", json={"input_path": str(checkpoint), "model_type": "ltx-2.3", "quantize": "4"})

        assert response.status_code == 409
        assert started == []

    def test_the_job_releases_the_slot_and_reports_the_converters_error(self, monkeypatch):
        from contextlib import ExitStack

        from zvisiongenerator.web import video_conversion

        class _Process:
            stdout = iter(["Converting: Writing the transformer (3/10)\n", "Error: A model named 'myft' already exists.\n"])

            def wait(self):
                return 1

        monkeypatch.setattr(video_conversion.subprocess, "Popen", lambda *a, **k: _Process())
        released: list[bool] = []
        slot = ExitStack()
        slot.callback(lambda: released.append(True))

        video_conversion.start_conversion(["model"], "myft", slot)
        for _ in range(100):
            if released:
                break
            threading.Event().wait(0.01)

        status = video_conversion.conversion_status()
        assert released == [True]
        assert status["state"] == "failed"
        assert status["message"] == "A model named 'myft' already exists."
        assert (status["stage"], status["done"], status["total"]) == ("Writing the transformer", 3, 10)

    def test_a_killed_conversion_says_how_it_stopped_not_its_last_progress_line(self):
        from zvisiongenerator.web import video_conversion

        progress = ["Converting: Writing the transformer (1200/4186)"]

        assert "signal 9" in video_conversion._failure_message(progress, -9)
        assert video_conversion._failure_message([*progress, "RuntimeError: CUDA error"], 1) == "RuntimeError: CUDA error"

    @pytest.mark.usefixtures("hub_offline")
    def test_inspect_returns_what_the_form_preselects(self, monkeypatch, tmp_path):
        from zvisiongenerator.converters.ltx_convert import LtxInspection, distill_advice

        checkpoint = tmp_path / "ft.safetensors"
        checkpoint.write_bytes(b"x")
        found = LtxInspection(checkpoint=SimpleNamespace(version="2.5", precision={"float8_e4m3fn": 440}, bundles_components=True), distillation="dev", advice="Use --distill.")
        monkeypatch.setattr("zvisiongenerator.web.video_conversion.inspect_ltx", lambda path, model_type: found)
        monkeypatch.setattr("zvisiongenerator.web.video_conversion.recommended_ltx_quantize", lambda config: 8)

        with TestClient(web_server.app) as client:
            response = client.post("/api/models/convert/inspect", json={"input_path": str(checkpoint), "model_type": "ltx-2.5"})

        assert response.status_code == 200
        answer = response.json()
        assert answer["model_type"] == "ltx-2.5"
        assert answer["advice"] == distill_advice("dev", "Distill on", "2.5")
        assert (answer["distillation"], answer["recommend_distill"]) == ("dev", True)
        assert answer["recommended_quantize"] == 4  # LTX-2.5 starts at q4 while its q8 headroom is unmeasured, whatever fits LTX-2.3

    @pytest.mark.usefixtures("hub_offline")
    def test_inspect_answers_for_h3_with_its_turbo_hints(self, monkeypatch, tmp_path):
        from zvisiongenerator.converters.h3_convert import H3Inspection

        checkpoint = tmp_path / "ft.safetensors"
        checkpoint.write_bytes(b"x")
        found = H3Inspection(checkpoint=SimpleNamespace(layout="comfyui", curve_rank=8, precision={}, turbo_hints=("the name says 'turbo'",)), advice="Maybe distilled.")
        monkeypatch.setattr("zvisiongenerator.web.video_conversion.inspect_h3_for_conversion", lambda path, turbo: found)

        with TestClient(web_server.app) as client:
            response = client.post("/api/models/convert/inspect", json={"input_path": str(checkpoint), "model_type": "h3"})

        assert response.status_code == 200
        assert response.json()["turbo_hints"] == ["the name says 'turbo'"]
        assert response.json()["tone"] == "warning"

    def test_an_h3_conversion_passes_already_step_distilled_as_turbo_and_no_level(self, monkeypatch, tmp_path):
        checkpoint = tmp_path / "ft.safetensors"
        checkpoint.write_bytes(b"x")
        started = self._hold(monkeypatch)

        with TestClient(web_server.app) as client:
            response = client.post("/api/models/convert", json={"input_path": str(checkpoint), "model_type": "h3", "name": "myh3", "turbo": True})

        assert response.status_code == 200
        assert started[0][0] == ["model", "--input", str(checkpoint.absolute()), "--model-type", "h3", "--name", "myh3", "--turbo"]
        started[0][1].close()

    @pytest.mark.usefixtures("hub_offline")
    def test_inspect_refuses_a_file_that_is_not_ltx(self, monkeypatch, tmp_path):
        checkpoint = tmp_path / "ft.safetensors"
        checkpoint.write_bytes(b"x")

        with TestClient(web_server.app) as client:
            response = client.post("/api/models/convert/inspect", json={"input_path": str(checkpoint)})

        assert response.status_code == 400


@pytest.mark.parametrize(("unified_gib", "expected"), [(32, 8), (64, 8), (24, 4)])
def test_a_mac_converts_ltx_at_q8_from_32_gb_of_unified_memory(monkeypatch, unified_gib, expected):
    from zvisiongenerator.web import video_conversion

    monkeypatch.setattr("zvisiongenerator.backends.accelerator_memory_budget_or_none", lambda: MemoryBudget(unified_gib * 1024**3 * 3 // 4, unified_bytes=unified_gib * 1024**3))

    assert video_conversion.recommended_ltx_quantize({}) == expected


def test_converter_subprocesses_print_in_utf8(monkeypatch):
    from zvisiongenerator.web import video_conversion

    seen: list[dict] = []

    class _Process:
        stdout = iter([])

        def wait(self):
            return 0

    def _popen(*_args, **kwargs):
        seen.append(kwargs)
        return _Process()

    def _run(*_args, **kwargs):
        seen.append(kwargs)
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    monkeypatch.setattr(video_conversion.subprocess, "Popen", _popen)
    monkeypatch.setattr(web_server.subprocess, "run", _run)
    video_conversion._run(["model"], contextlib.nullcontext())
    web_server._run_model_management_command(["model"])

    assert [(kwargs["encoding"], kwargs["env"]["PYTHONIOENCODING"]) for kwargs in seen] == [("utf-8", "utf-8")] * 2
