"""Test what an edit records in the config payload of its saved picture."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from zvisiongenerator.core.image_types import EditWork, ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.utils.provenance import build_image_config_payload


def _payload(*, painted: bool = False, extend: tuple[str, int] | None = None, size: tuple[int, int] = (640, 480), alpha: np.ndarray | None = None, **overrides):
    request = ImageGenerationRequest(
        backend=MagicMock(),
        model=object(),
        prompt="a red door",
        edit_source="/pictures/door.png",
        edit_paint=np.ones((4, 4), dtype=bool) if painted else None,
        edit_extend=extend,
        sharpen=True,
        contrast=True,
        saturation=True,
        transparent=True,
        sharpen_amount_override=0.9,
        quantize=8,
        width=832,
        height=1216,
        **overrides,
    )
    artifacts = ImageWorkingArtifacts(image=Image.new("RGB", size))
    artifacts.metadata["edit_source"] = {"path": "/pictures/door.png", "width": 512, "height": 480}
    artifacts.generation_time = 3.14
    artifacts.edit = EditWork(canvas=Image.new("RGB", (512, 480)), alpha=alpha)
    return build_image_config_payload(request, artifacts)


def test_an_edit_records_the_edit_workflow_and_the_size_of_the_saved_picture():
    payload = _payload()

    assert payload["workflow"] == "edit"
    assert (payload["width"], payload["height"]) == (640, 480)
    assert (payload["ratio"], payload["size"], payload["image_path"]) == (None, None, None)


def test_the_source_block_has_the_path_and_size_and_no_workflow():
    assert _payload()["source"] == {"path": "/pictures/door.png", "width": 512, "height": 480}


@pytest.mark.parametrize(
    ("painted", "extend", "expected"),
    [
        (False, None, {"mode": "whole"}),
        (True, None, {"mode": "paint"}),
        (False, ("top", 256), {"mode": "extend", "extend": {"side": "top", "amount": 256}}),
        (True, ("left", 128), {"mode": "both", "extend": {"side": "left", "amount": 128}}),
    ],
)
def test_the_edit_block_records_the_mode_and_only_a_set_extend(painted, extend, expected):
    assert _payload(painted=painted, extend=extend)["edit"] == expected


def test_the_painted_area_is_never_recorded():
    payload = _payload(painted=True)

    assert "mask" not in str(payload).lower()


def test_an_edit_records_no_post_processing():
    assert _payload()["generation"] == {"time": 3.1, "quantize": 8}


def test_an_edit_of_a_cut_out_records_transparent():
    cutout = _payload(alpha=np.zeros((480, 512), dtype=np.uint8))

    assert cutout["generation"] == {"time": 3.1, "quantize": 8, "transparent": True}


def test_other_workflows_keep_their_recorded_settings():
    request = ImageGenerationRequest(backend=MagicMock(), model=object(), prompt="a cat", sharpen=True, quantize=8)

    payload = build_image_config_payload(request, ImageWorkingArtifacts())

    assert payload["workflow"] == "txt2img"
    assert "sharpen" in payload["generation"]
    assert "edit" not in payload and "source" not in payload
