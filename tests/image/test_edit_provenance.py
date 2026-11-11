"""Test what an edit records in the config payload of its saved picture."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image

from tests._factories import make_edit_payload
from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.utils.provenance import build_image_config_payload


# Everything an edit leaves off its record, requested anyway.
_POST_PROCESSING = dict(sharpen=True, contrast=True, saturation=True, transparent=True, sharpen_amount_override=0.9, quantize=8)


def _payload(**kwargs):
    return make_edit_payload(**_POST_PROCESSING, **kwargs)


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
    ],
)
def test_the_edit_block_records_the_mode_and_only_a_set_extend(painted, extend, expected):
    assert _payload(painted=painted, extend=extend)["edit"] == expected


@pytest.mark.parametrize(("cutout", "recorded"), [(None, {"time": 3.1, "quantize": 8}), (Image.new("RGBA", (512, 480)), {"time": 3.1, "quantize": 8, "transparent": True})], ids=["opaque", "cut-out"])
def test_an_edit_records_no_post_processing_and_transparent_only_for_a_cut_out(cutout, recorded):
    assert _payload(cutout=cutout)["generation"] == recorded


def test_other_workflows_keep_their_recorded_settings():
    request = ImageGenerationRequest(backend=MagicMock(), model=object(), prompt="a cat", sharpen=True, quantize=8)

    payload = build_image_config_payload(request, ImageWorkingArtifacts())

    assert payload["workflow"] == "txt2img"
    assert "sharpen" in payload["generation"]
    assert "edit" not in payload and "source" not in payload
