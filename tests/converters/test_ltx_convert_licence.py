"""Tests for the licence notice shown before the converter downloads Lightricks' distilled LoRA."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._factories import load_packaged_config
from zvisiongenerator.converters import ltx_convert
from zvisiongenerator.downloads.target import DownloadTarget


@pytest.fixture
def downloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict]:
    """Record the files the converter asks the Hub for instead of fetching them."""
    calls: list[dict] = []

    def fake_download(**kwargs) -> str:
        calls.append(kwargs)
        return str(tmp_path / kwargs["filename"])

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    return calls


def _fetch(version: str | None, label: str) -> None:
    ltx_convert._distilled_lora(load_packaged_config(), version, label)


@pytest.mark.parametrize(("version", "label", "repo"), [(None, "2.3", "Lightricks/LTX-2.3"), ("ltx25", "2.5", "lxxxy6/LTX-2.5")])
def test_the_notice_comes_before_the_download_message_and_the_download(capsys, downloads, monkeypatch, version, label, repo):
    monkeypatch.setattr("zvisiongenerator.downloads.presence.target_present_offline", lambda target: False)

    _fetch(version, label)

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "By downloading you agree to these terms:"
    assert lines[1].endswith("(Lightricks): Free under $10M revenue")
    assert lines[-1].startswith("Getting Lightricks' LTX-")
    assert [call["repo_id"] for call in downloads] == [repo]


def test_each_version_names_only_its_own_licence(capsys, downloads, monkeypatch):
    monkeypatch.setattr("zvisiongenerator.downloads.presence.target_present_offline", lambda target: False)

    _fetch(None, "2.3")
    out_23 = capsys.readouterr().out
    _fetch("ltx25", "2.5")
    out_25 = capsys.readouterr().out

    assert "LTX-2 Community License Agreement" in out_23 and "LTX-2.x" not in out_23 and "Gemma" not in out_23
    assert "LTX-2.x Community License Agreement" in out_25 and "LTX-2 Community" not in out_25 and "Gemma" not in out_25


def test_nothing_is_said_when_the_lora_is_already_downloaded(capsys, downloads, monkeypatch):
    seen: list[DownloadTarget] = []
    monkeypatch.setattr("zvisiongenerator.downloads.presence.target_present_offline", lambda target: seen.append(target) or True)

    _fetch(None, "2.3")

    out = capsys.readouterr().out
    assert "By downloading" not in out
    assert [(target.repo_id, target.allow_patterns) for target in seen] == [("Lightricks/LTX-2.3", ("ltx-2.3-22b-distilled-lora-384-1.1.safetensors",))]
