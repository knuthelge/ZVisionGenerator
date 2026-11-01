"""The matte stage: it cuts the image out through the job's chain and reports one step."""

from __future__ import annotations

from unittest.mock import MagicMock

from PIL import Image

from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.workflows.image_stages import matte_stage


def _chain(*, loaded: bool, cut_out: Image.Image | None = None) -> MagicMock:
    chain = MagicMock()
    chain.loaded = loaded
    chain.cut_out.return_value = cut_out or Image.new("RGBA", (8, 8), (1, 2, 3, 0))
    return chain


def _request(**overrides) -> ImageGenerationRequest:
    return ImageGenerationRequest(backend=MagicMock(), model=None, prompt="p", transparent=True, **overrides)


class TestMatteStage:
    def test_the_image_becomes_the_chains_cut_out(self):
        cut = Image.new("RGBA", (8, 8), (9, 9, 9, 0))
        chain = _chain(loaded=True, cut_out=cut)
        artifacts = ImageWorkingArtifacts(image=Image.new("RGBA", (8, 8), (200, 0, 0, 255)))

        outcome = matte_stage(_request(matte=chain), artifacts)

        assert outcome is StageOutcome.success
        assert artifacts.image is cut
        sent = chain.cut_out.call_args.args[0]
        assert sent.mode == "RGB"

    def test_room_is_made_only_while_the_chain_is_not_loaded(self):
        make_room = MagicMock()

        matte_stage(_request(matte=_chain(loaded=False), make_room=make_room), ImageWorkingArtifacts(image=Image.new("RGB", (8, 6))))
        make_room.assert_called_once_with(48)

        make_room.reset_mock()
        matte_stage(_request(matte=_chain(loaded=True), make_room=make_room), ImageWorkingArtifacts(image=Image.new("RGB", (8, 8))))
        make_room.assert_not_called()

    def test_room_is_made_before_the_cut_out(self):
        chain = _chain(loaded=False)
        order = MagicMock()
        order.attach_mock(chain.cut_out, "cut_out")
        make_room = MagicMock()
        order.attach_mock(make_room, "make_room")

        matte_stage(_request(matte=chain, make_room=make_room), ImageWorkingArtifacts(image=Image.new("RGB", (8, 8))))

        assert [c[0] for c in order.mock_calls] == ["make_room", "cut_out"]

    def test_a_missing_image_is_skipped_without_loading_the_chain(self):
        chain = _chain(loaded=False)
        make_room = MagicMock()

        outcome = matte_stage(_request(matte=chain, make_room=make_room), ImageWorkingArtifacts(image=None))

        assert outcome is StageOutcome.success
        chain.cut_out.assert_not_called()
        make_room.assert_not_called()

    def test_without_a_chain_the_image_is_left_alone(self):
        image = Image.new("RGB", (8, 8))
        artifacts = ImageWorkingArtifacts(image=image)

        assert matte_stage(_request(), artifacts) is StageOutcome.success
        assert artifacts.image is image

    def test_the_step_callback_gets_the_matte_phase(self):
        events: list[dict] = []

        matte_stage(_request(matte=_chain(loaded=True), step_callback=events.append), ImageWorkingArtifacts(image=Image.new("RGB", (8, 8))))

        assert [(e["phase"], e["current_step"], e["total_steps"]) for e in events] == [("image_matte", 1, 1)]
