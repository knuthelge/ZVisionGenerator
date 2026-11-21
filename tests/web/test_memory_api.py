"""Tests for the memory answer of the Web UI: ``POST /api/memory``, the reference job of each listed model and the rows each medium gets."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tests._audio_presets import audio_preset, use_audio_models
from tests.web._helpers import make_web_config
from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.processing.edit_region import edit_model_size
from zvisiongenerator.utils import memory_rows, memory_verdict, model_memory
from zvisiongenerator.utils.memory_verdict import MemoryWeights
from zvisiongenerator.utils.model_memory import GIB, LtxCudaMemoryEstimate, MemoryBudget, WeightTotals
from zvisiongenerator.utils.weight_totals import ResolvedWeights
from zvisiongenerator.web import audio_models as audio_models_module
from zvisiongenerator.web import model_status as model_status_module
from zvisiongenerator.web import extend_api, memory_api, request_fields, workspace_api
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web.memory_api import audio_memory, memory_response, reference_image_memory, reference_video_memory
from zvisiongenerator.web.lora_fit import LoraSize
from zvisiongenerator.web.model_inventory import ImageInventoryEntry, VideoInventoryEntry
from zvisiongenerator.web.video_clip import VideoClip

_CARD = MemoryBudget(10 * GIB, 30 * GIB)
_MAC = MemoryBudget(24 * GIB, unified_bytes=32 * GIB)
_ROW_KEYS = {"key", "status", "resource", "gpu_gb", "ram_gb", "word", "label", "sentence", "details"}
_ANSWER_KEYS = {"kind", "budget", "control", "current", "rows", "title", "control_title", "table_heading", "judged", "download_size_label", "expected_seconds"}
_JUDGED_KEYS = {"frame", "width", "height", "seconds", "quantize", "low_memory", "references", "loras"}
_STATUSES = {"fits", "tight", "too_large"}
_WORDS = {"fits": "Fits", "tight": "Tight", "too_large": "Too large"}


def _elements(gib: float) -> int:
    return round(gib * GIB)


def _klein_4b(*, stored_bits: int | None = None) -> ResolvedWeights:
    """Return the resolved weights of a Klein 4B-sized image model."""
    components = {
        "text_encoder": WeightTotals(float_matrix_elements=_elements(3.7)),
        "transformer": WeightTotals(float_matrix_elements=_elements(3.6)),
        "vae": WeightTotals(float_other_elements=_elements(0.08)),
    }
    return ResolvedWeights(MemoryWeights(components=components, stored_bits=stored_bits), family="flux2_klein", model_size="4b")


def _ltx_weights() -> ResolvedWeights:
    system = LtxCudaMemoryEstimate(text_encoder_bytes=11 * GIB, transformer_bytes=10 * GIB, resident_bytes=GIB, quantize=4)
    return ResolvedWeights(MemoryWeights(ltx_cuda=system), family="ltx")


@pytest.fixture
def jobs(monkeypatch):
    """Record every job the engine judges, and let the engine answer it."""
    judged: list[memory_verdict.MemoryJob] = []
    real = memory_rows.judge

    def spy(job, machine, weights, **kwargs):
        judged.append(job)
        return real(job, machine, weights, **kwargs)

    monkeypatch.setattr(memory_rows, "judge", spy)
    return judged


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point the LoRA folder at an empty one, so a bare LoRA name is not found."""
    monkeypatch.setattr("zvisiongenerator.utils.paths.get_ziv_data_dir", lambda: tmp_path)
    (tmp_path / "loras").mkdir()
    return tmp_path


@pytest.fixture
def image_model(monkeypatch):
    """Resolve every image model to a Klein 4B-sized one; the list holds what the resolver was asked for."""
    asked: list[dict[str, object]] = []
    resolved = [_klein_4b()]

    def resolve(name, path, **kwargs):
        asked.append({"name": name, "path": path, **kwargs})
        return resolved[0]

    monkeypatch.setattr(memory_api, "resolve_image_weights", resolve)
    monkeypatch.setattr(memory_api, "resolve_model_path", lambda model, **_: f"/resolved/{model}")
    monkeypatch.setattr(memory_api, "model_download_size_label", lambda name, platform: "17 GB")
    return SimpleNamespace(asked=asked, resolved=resolved)


def _image(payload=None, *, budget=_CARD, present=True, **kwargs):
    """Answer an image request on a model that is on disk (or not, when *present* is false)."""
    config = make_web_config()
    body = {"media": "image", "model": "zit", **(payload or {})}
    return memory_response(body, config, budget=budget, find_local_dir=lambda reference: "/models/zit" if present else None, **kwargs)


# --- the shape of an image answer -----------------------------------------------------------------------------------------


def test_an_image_answer_has_one_row_per_quantize_level_and_the_documented_shape(image_model):
    answer = _image({"ratio": "2:3", "size": "l"})

    assert set(answer) == _ANSWER_KEYS
    assert answer["kind"] == "discrete"
    assert answer["budget"] == {"gpu_gb": 10.0, "ram_gb": 30.0}
    assert (answer["control"], answer["current"]) == ("quantize", "none")
    assert [row["key"] for row in answer["rows"]] == ["none", "8", "4"]
    for row in answer["rows"]:
        assert set(row) == _ROW_KEYS
        assert row["status"] in _STATUSES
        assert row["resource"] in {"gpu", "ram"}
        assert isinstance(row["gpu_gb"], float)
        assert isinstance(row["ram_gb"], float)
    assert answer["judged"] == {"frame": "image", "width": 1024, "height": 1536, "seconds": None, "quantize": None, "low_memory": None, "references": 0, "loras": 0}
    assert (answer["download_size_label"], answer["expected_seconds"]) == (None, None)


def test_every_row_of_an_image_answer_carries_its_words_and_the_answer_its_headings(image_model):
    answer = _image({"ratio": "2:3", "size": "l", "quantize": 4, "reference_sizes": [[512, 512]], "lora": ""})

    assert [row["label"] for row in answer["rows"]] == ["None", "q8", "q4"]
    assert [row["word"] for row in answer["rows"]] == [_WORDS[row["status"]] for row in answer["rows"]]
    assert all(row["sentence"] for row in answer["rows"])
    # Each row's details are what was judged at its own level, so the client names the level the form holds without asking again.
    assert [("q" in row["details"], "q8" in row["details"], "q4" in row["details"]) for row in answer["rows"]] == [(False, False, False), (True, True, False), (True, False, True)]
    assert all(row["details"].startswith("1024×1536") and "1 reference" in row["details"] for row in answer["rows"])
    assert answer["title"] and answer["control_title"] == "Quant" and answer["table_heading"]


def test_an_ltx_clip_names_its_low_memory_levels_and_a_single_row_has_no_level_words(video_model):
    answer = _video({"model": "ltx-8", "low_memory": False})

    assert [row["label"] for row in answer["rows"]] == ["On", "Off"]
    assert [("Low memory on" in row["details"], "Low memory off" in row["details"]) for row in answer["rows"]] == [(True, False), (False, True)]
    assert (answer["control_title"], answer["table_heading"] is not None) == ("Low memory", True)
    video_model.family = "h3"
    video_model.clip = VideoClip(steps=8, width=768, height=448, num_frames=124)
    single = _video({})
    assert (single["control_title"], single["table_heading"], single["rows"][0]["label"]) == (None, None, "")


def test_the_requested_level_is_the_current_row_and_is_judged(image_model, jobs):
    answer = _image({"quantize": 4})

    assert (answer["current"], answer["judged"]["quantize"]) == ("4", 4)
    assert [job.quantize for job in jobs] == [None, 8, 4]


def test_a_quantize_option_the_engine_cannot_judge_has_no_row_and_the_answer_is_about_the_first_judged_level(image_model):
    answer = reference_image_memory("zit", "/resolved/zit", quantize=6, levels=(8, 6, 4), budget=_CARD, downloaded=None)

    assert [row["key"] for row in answer["rows"]] == ["none", "8", "4"]
    assert (answer["current"], answer["judged"]["quantize"]) == ("none", None)


def test_a_stored_quant_model_has_one_row_at_its_own_level(image_model, jobs):
    image_model.resolved[0] = _klein_4b(stored_bits=4)

    answer = _image({"quantize": 8})

    assert (answer["control"], answer["current"], [row["key"] for row in answer["rows"]]) == (None, "4", ["4"])
    assert answer["judged"]["quantize"] == 4
    assert [job.quantize for job in jobs] == [4]


def test_a_model_that_cannot_be_quantized_has_one_unquantized_row(image_model, monkeypatch):
    monkeypatch.setattr(memory_rows, "model_capabilities", lambda family: {"supports_quantize": False})

    answer = _image()

    assert (answer["control"], answer["current"], [row["key"] for row in answer["rows"]]) == (None, "none", ["none"])


def test_a_model_that_is_not_downloaded_is_judged_from_the_table_and_reports_its_download(image_model):

    answer = _image(present=False)

    assert answer["download_size_label"] == "17 GB"
    assert [row["key"] for row in answer["rows"]] == ["none", "8", "4"]


def test_an_unbuilt_model_whose_sources_are_all_on_disk_reports_no_download(image_model, monkeypatch):
    monkeypatch.setattr(model_status_module, "built_model_plan", lambda path, **_: ())  # the model a first job builds: nothing left to fetch

    answer = _image(present=False)

    assert answer["download_size_label"] is None


def test_an_unbuilt_model_with_a_source_still_to_fetch_reports_its_download(image_model, monkeypatch):
    monkeypatch.setattr(model_status_module, "built_model_plan", lambda path, **_: (DownloadTarget("not-a/cached-repo"),))

    answer = _image(present=False)

    assert answer["download_size_label"] == "17 GB"


def test_the_weights_resolve_from_the_models_path_without_the_hub(image_model):
    _image()

    assert [call["path"] for call in image_model.asked] == ["/resolved/zit"]
    assert not any(call.get("hub") for call in image_model.asked)


def test_the_answer_is_null_without_a_budget_or_weights(image_model):
    assert _image(budget=None) is None
    assert _image(budget=MemoryBudget(0, 0)) is None
    image_model.resolved[0] = None
    assert _image() is None


def test_a_mac_has_unified_rows_with_no_ram(image_model):
    answer = _image(budget=_MAC)

    assert (answer["kind"], answer["budget"]) == ("unified", {"gpu_gb": 32.0, "ram_gb": None})  # the total memory, which a Mac is graded against
    assert {(row["resource"], row["ram_gb"]) for row in answer["rows"]} == {("unified", None)}


# --- image jobs ------------------------------------------------------------------------------------------------------------


def test_a_new_image_judges_its_references_and_loras(image_model, jobs, monkeypatch):
    sized = []
    monkeypatch.setattr(memory_api, "lora_size_for_fit", lambda lora, downloading: sized.append((lora, dict(downloading))) or LoraSize(700_000_000))
    monkeypatch.setattr(memory_api, "parse_lora_arg", lambda text: [("style", 0.8), ("detail", 1.0)])

    answer = _image({"width": 1536, "reference_sizes": [[512, 512], [2000, 1500]], "lora": "style:0.8,detail"}, downloading_loras={"detail": 5})

    assert sized == [("style:0.8,detail", {"detail": 5})]
    assert (answer["judged"]["width"], answer["judged"]["height"]) == (1536, 1216)
    assert (answer["judged"]["references"], answer["judged"]["loras"]) == (2, 2)
    assert jobs[0].reference_megapixels == (0.262144, 3.0)
    assert jobs[0].lora_bytes == 700_000_000


def test_a_reference_file_on_this_machine_is_judged_at_the_size_its_header_gives(image_model, jobs, tmp_path):
    from PIL import Image

    Image.new("RGB", (800, 600)).save(tmp_path / "a.png")

    answer = _image({"reference_paths": [str(tmp_path / "a.png")], "reference_sizes": [[512, 512]]})

    assert jobs[0].reference_megapixels == (0.262144, 0.48)
    assert answer["judged"]["references"] == 2


def test_a_reference_path_the_file_browser_would_refuse_counts_nothing_and_does_not_stop_the_verdict(image_model, jobs, tmp_path):
    (tmp_path / "notes.txt").write_text("not a picture")
    (tmp_path / "broken.png").write_bytes(b"not a picture")

    answer = _image({"reference_paths": [str(tmp_path / "notes.txt"), str(tmp_path / "broken.png"), str(tmp_path / "gone.png"), "relative/a.png"]})

    assert jobs[0].reference_megapixels == ()
    assert answer["rows"]


@pytest.mark.parametrize("payload", [{"reference_paths": "a.png"}, {"reference_paths": [1]}, {"reference_paths": [["a.png"]]}])
def test_a_bad_reference_paths_field_is_a_value_error(image_model, payload):
    with pytest.raises(ValueError):
        _image(payload)


def test_an_image_counts_a_lora_that_is_not_there_as_nothing_and_still_judges_it_and_says_so(image_model, jobs, tmp_path):
    answer = _image({"lora": f"{tmp_path / 'gone.safetensors'}:0.8"})

    assert jobs[0].lora_bytes == 0
    assert answer["judged"]["loras"] == 1
    assert answer["notes"] == ["LoRA gone.safetensors isn't available, so it isn't counted."]


def test_a_remote_lora_reference_that_is_not_downloaded_is_not_counted_and_does_not_stop_the_verdict(image_model, jobs, data_dir):
    answer = _image({"lora": "someone/style-lora:0.8"})

    assert jobs[0].lora_bytes == 0
    assert answer["rows"]
    assert answer["notes"] == ["LoRA style-lora isn't available, so it isn't counted."]


def test_an_edit_judges_the_size_the_model_draws_with_the_picture_as_its_one_reference_but_counts_none_of_the_users(image_model, jobs):
    answer = _image({"kind": "edit", "source_width": 3000, "source_height": 2000, "masked": False, "extend": "right=256", "reference_sizes": [[512, 512]] * 3})

    expected = edit_model_size(3000, 2000, masked=False, extend=("right", 256))
    assert answer["judged"]["frame"] == "edit"
    assert (answer["judged"]["width"], answer["judged"]["height"]) == expected
    assert answer["judged"]["references"] == 0
    assert jobs[0].reference_megapixels == (6.0,)


@pytest.mark.parametrize(
    ("extra", "from_picture"), [({}, False), ({"upscale": 2}, True), ({"upscale": 4}, True), ({"image": True}, True), ({"image": False}, False), ({"upscale": 2, "image": True}, True)]
)
def test_a_new_image_starts_from_a_picture_with_an_upscale_or_a_start_image_at_the_size_it_draws(image_model, jobs, extra, from_picture):
    answer = _image({"ratio": "2:3", "size": "l", "width": 1600, "height": 1600, **extra})

    assert {job.from_picture for job in jobs} == {from_picture}
    assert (answer["judged"]["width"], answer["judged"]["height"]) == (jobs[0].width, jobs[0].height)


@pytest.mark.parametrize(("payload", "from_picture"), [({"masked": True}, True), ({"extend": "right=256"}, True), ({}, False)])
def test_an_edit_starts_from_a_picture_only_when_painted_or_extended(image_model, jobs, payload, from_picture):
    _image({"kind": "edit", "source_width": 1024, "source_height": 1024, **payload})

    assert {job.from_picture for job in jobs} == {from_picture}


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "edit"},
        {"kind": "edit", "source_width": 3000},
        {"kind": "edit", "source_width": 3000, "source_height": 2000, "extend": "up=256"},
        {"quantize": 5},
        {"upscale": 3},
        {"image": "yes"},
        {"quantize": "4"},
        {"ratio": "9:21"},
        {"reference_sizes": [{"width": 512, "height": 512}]},
        {"reference_sizes": [[512]]},
        {"reference_sizes": [[512, 0]]},
        {"reference_sizes": [[512, 1.5]]},
        {"reference_sizes": [[True, 512]]},
        {"reference_sizes": 2},
        {"model": ""},
        {"kind": "clip"},
        {"kind": "reference"},
    ],
)
def test_a_bad_image_request_is_a_value_error(image_model, payload):
    with pytest.raises(ValueError):
        _image(payload)


def test_a_body_that_is_not_an_object_or_names_no_medium_is_a_value_error(image_model):
    config = make_web_config()
    for body in ([], "image", {"media": "text"}, {"media": "audio", "model": "sa3"}, {}):
        with pytest.raises(ValueError):
            memory_response(body, config, budget=_CARD)


# --- video ----------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def video_model(monkeypatch):
    """Serve an LTX or MiniMax H3 model with a fixed clip, and a CUDA LTX weights stand-in."""
    state = SimpleNamespace(family="ltx", fps=24, clip=VideoClip(steps=8, width=736, height=448, num_frames=121))
    monkeypatch.setattr(memory_api, "resolve_video_model", lambda config, name: (f"/resolved/{name}", SimpleNamespace(family=state.family, default_fps=state.fps)))
    monkeypatch.setattr(memory_api, "resolve_video_clip", lambda *args, **kwargs: state.clip)
    monkeypatch.setattr(memory_api, "resolve_ltx_weights", lambda *args, **kwargs: _ltx_weights())
    monkeypatch.setattr(memory_api, "model_downloaded", lambda *args, **kwargs: True)
    monkeypatch.setattr(request_fields, "video_family", lambda config, family: None)
    return state


def _video(payload=None, *, budget=_CARD, **kwargs):
    return memory_response({"media": "video", **(payload or {})}, make_web_config(), budget=budget, find_local_dir=lambda reference: None, **kwargs)


def test_an_ltx_clip_has_a_row_for_low_memory_on_and_off(video_model, jobs):
    answer = _video({"model": "ltx-8", "frames": 121, "upscale": 2})

    assert set(answer) == _ANSWER_KEYS
    assert (answer["control"], answer["current"]) == ("low_memory", "on")
    assert [row["key"] for row in answer["rows"]] == ["on", "off"]
    assert [job.low_memory for job in jobs] == [True, False]
    assert answer["judged"] == {"frame": "clip", "width": 736, "height": 448, "seconds": 5.0, "quantize": None, "low_memory": True, "references": 0, "loras": 0}
    assert (jobs[0].frames, jobs[0].upscale, jobs[0].mode, jobs[0].family) == (121, True, "t2v", "ltx")
    assert answer["expected_seconds"] is None


def test_low_memory_off_makes_the_off_row_current(video_model):
    answer = _video({"low_memory": False})

    assert (answer["current"], answer["judged"]["low_memory"]) == ("off", False)
    assert answer["rows"][1]["key"] == "off"


def test_the_reference_clip_a_listing_judges_has_low_memory_on(video_model, jobs):
    answer = reference_video_memory("ltx-8", "/models/ltx", "ltx", budget=_CARD, downloaded=True)

    assert (answer["current"], answer["judged"]["low_memory"], answer["judged"]["frame"]) == ("on", True, "reference")
    assert (answer["judged"]["width"], answer["judged"]["height"], answer["judged"]["seconds"]) == (768, 448, 5.0)
    assert [(job.width, job.height, job.frames, job.mode, job.upscale) for job in jobs] == [(768, 448, 121, "t2v", False)] * 2


def test_an_h3_clip_has_one_row_and_an_expected_time(video_model, jobs):
    video_model.family = "h3"
    video_model.clip = VideoClip(steps=8, width=768, height=448, num_frames=124)

    answer = _video({"image": True, "references": 2, "lora": ""})

    assert (answer["control"], answer["current"], [row["key"] for row in answer["rows"]]) == (None, "only", ["only"])
    assert isinstance(answer["expected_seconds"], int) and answer["expected_seconds"] > 0
    assert (jobs[0].family, jobs[0].mode, jobs[0].references) == ("h3", "i2v", 2)
    assert (answer["judged"]["low_memory"], answer["judged"]["references"]) == (None, 2)


def test_the_h3_reference_clip_is_124_frames(video_model, jobs):
    answer = reference_video_memory("h3", "/models/h3", "h3", budget=_CARD, downloaded=True)

    assert (jobs[0].frames, jobs[0].width, jobs[0].height) == (124, 768, 448)
    assert answer["expected_seconds"] is not None


def test_an_h3_clip_counts_its_loras_that_are_still_downloading(video_model, jobs, data_dir):
    video_model.family = "h3"
    video_model.clip = VideoClip(steps=8, width=768, height=448, num_frames=124)

    answer = _video({"lora": "style"}, downloading_loras={"style": 2 * GIB})

    assert jobs[0].lora_bytes == 2 * GIB
    assert answer["judged"]["loras"] == 1


def test_an_ltx_clip_counts_its_loras_but_does_not_need_them_to_exist(video_model, jobs, data_dir):
    answer = _video({"lora": "never_installed:0.8,another:1"})

    assert [job.lora_bytes for job in jobs] == [0, 0]
    assert answer["judged"]["loras"] == 2
    assert answer["notes"] == ["LoRAs never_installed, another aren't available, so they aren't counted."]


def test_an_ltx_clip_sizes_the_files_of_its_loras_and_the_downloading_ones(video_model, jobs, tmp_path):
    lora = tmp_path / "style.safetensors"
    lora.write_bytes(b"\0" * 4_000)

    _video({"lora": f"{lora}:0.7"})
    assert [job.lora_bytes for job in jobs] == [4_000, 4_000]

    jobs.clear()
    _video({"lora": "pending"}, downloading_loras={"pending": 900})
    assert [job.lora_bytes for job in jobs] == [900, 900]


def test_an_ltx_clip_with_loras_needs_more_memory(video_model):
    plain = _video({})
    loaded = _video({"lora": "style"}, downloading_loras={"style": 5 * GIB})

    # The text encoder (11 GiB) sets the system memory until the transformer (10 GiB) and its adapters (5 GiB) outgrow it; the connectors (1 GiB) and the runtime are held throughout.
    runtime = model_memory._LTX_CUDA_RUNTIME_BYTES[4] / GIB
    assert [row["ram_gb"] for row in plain["rows"]] == [round(12 + runtime, 1)] * 2
    assert [row["ram_gb"] for row in loaded["rows"]] == [round(16 + runtime, 1)] * 2
    assert loaded["rows"][1]["gpu_gb"] == plain["rows"][1]["gpu_gb"] + 5.0  # Low memory off: the adapters sit on the GPU with the transformer
    assert loaded["rows"][0]["gpu_gb"] == plain["rows"][0]["gpu_gb"]  # Low memory on: the blocks stream, the adapters with them


def test_an_h3_clip_with_a_lora_that_is_not_there_is_judged_without_it_and_says_so(video_model, jobs, tmp_path):
    video_model.family = "h3"
    video_model.clip = VideoClip(steps=8, width=768, height=448, num_frames=124)

    answer = _video({"lora": str(tmp_path / "gone.safetensors")})

    assert jobs[0].lora_bytes == 0
    assert answer["notes"] == ["LoRA gone.safetensors isn't available, so it isn't counted."]


def test_an_h3_clip_sizes_the_files_of_its_loras(video_model, jobs, tmp_path):
    video_model.family = "h3"
    video_model.clip = VideoClip(steps=8, width=768, height=448, num_frames=124)
    lora = tmp_path / "style.safetensors"
    lora.write_bytes(b"\0" * 4_000)

    _video({"lora": f"{lora}:0.7"})

    assert jobs[0].lora_bytes == 4_000


def test_a_clip_is_judged_on_the_steps_it_resolves_to(video_model, jobs):
    video_model.clip = VideoClip(steps=6, width=736, height=448, num_frames=121)

    _video({"steps": 6})

    assert jobs[0].forwards == 6


def test_an_ltx_model_without_resolvable_weights_has_no_answer(video_model, monkeypatch):
    monkeypatch.setattr(memory_api, "resolve_ltx_weights", lambda *args, **kwargs: None)

    assert _video() is None


def test_a_family_without_an_estimate_has_no_answer(video_model):
    video_model.family = "other"

    assert _video() is None
    assert reference_video_memory("x", "/models/x", "other", budget=_CARD, downloaded=True) is None


def test_a_mac_ltx_clip_is_judged_on_unified_memory(video_model, monkeypatch):
    from zvisiongenerator.utils.model_memory import LtxMlxWeights

    mlx = LtxMlxWeights(gemma=7 * GIB, connector=GIB, transformer=10 * GIB, decoders=GIB, upsampler=None)
    monkeypatch.setattr(memory_api, "resolve_ltx_weights", lambda *args, **kwargs: ResolvedWeights(MemoryWeights(ltx_mlx=mlx), family="ltx"))

    answer = _video(budget=_MAC)

    assert (answer["kind"], {row["resource"] for row in answer["rows"]}, {row["ram_gb"] for row in answer["rows"]}) == ("unified", {"unified"}, {None})


def test_an_extension_judges_the_window_the_model_generates(video_model, jobs, monkeypatch):
    monkeypatch.setattr(extend_api, "load_source_video", lambda path, key: SimpleNamespace(width=1280, height=704, fps=24.0))
    monkeypatch.setattr(memory_api, "video_family_on_platform", lambda config, family: None)

    answer = _video({"kind": "extension", "source": "/clips/a.mp4", "extend_frames": 48, "model": "ltx-8"})

    assert (jobs[0].width, jobs[0].height, jobs[0].frames, jobs[0].mode) == (1280, 704, 25 + 48, "extend")
    assert answer["judged"]["frame"] == "extension"
    assert answer["judged"]["seconds"] == round((25 + 48) / 24, 1)


def test_an_h3_extension_is_judged_in_extend_mode(video_model, jobs, monkeypatch):
    video_model.family = "h3"
    monkeypatch.setattr(extend_api, "load_source_video", lambda path, key: SimpleNamespace(width=768, height=448, fps=24.0))
    monkeypatch.setattr(memory_api, "video_family_on_platform", lambda config, family: None)

    answer = _video({"kind": "extension", "source": "/clips/a.mp4"})

    assert (jobs[0].family, jobs[0].mode) == ("h3", "extend")
    assert answer["expected_seconds"] is not None


def test_a_mac_h3_extension_is_judged_on_the_window_the_mac_renders(video_model, jobs, monkeypatch):
    import zvisiongenerator.backends as backends_module
    from zvisiongenerator.utils.config import load_config

    video_model.family = "h3"
    monkeypatch.setattr(backends_module.sys, "platform", "darwin")
    monkeypatch.setattr(extend_api, "load_source_video", lambda path, key: SimpleNamespace(width=768, height=448, fps=24.0))
    config = make_web_config()
    config.app_config = load_config()

    memory_response({"media": "video", "kind": "extension", "source": "/clips/a.mp4", "extend_frames": 51, "model": "h3"}, config, budget=_MAC, find_local_dir=lambda reference: None)

    assert jobs[0].frames == 56


def test_an_extension_length_off_the_models_step_is_judged_at_the_nearest_length(video_model, jobs, monkeypatch):
    monkeypatch.setattr(extend_api, "load_source_video", lambda path, key: SimpleNamespace(width=1280, height=704, fps=24.0))
    monkeypatch.setattr(memory_api, "video_family_on_platform", lambda config, family: None)

    _video({"kind": "extension", "source": "/clips/a.mp4", "extend_frames": 50, "model": "ltx-8"})

    assert jobs[0].frames == 25 + 48


@pytest.mark.parametrize(
    "payload", [{"low_memory": "yes"}, {"upscale": 3}, {"frames": 0}, {"kind": "track"}, {"kind": "reference"}, {"kind": "extension"}, {"ratio": "3:1"}, {"size": "huge"}, {"steps": 0}]
)
def test_a_bad_video_request_is_a_value_error(video_model, payload):
    with pytest.raises(ValueError):
        _video(payload)


# --- audio -------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def audio(monkeypatch, tmp_path):
    use_audio_models(monkeypatch, tmp_path)
    return audio_preset("sa3").on_platform("cuda")


def test_an_audio_model_has_one_row_judged_on_its_measured_peak(audio):
    answer = audio_memory(audio, platform_key="cuda", budget=_CARD, downloaded=True)

    assert set(answer) == _ANSWER_KEYS
    assert (answer["control"], answer["current"]) == (None, "only")
    assert [{name: row[name] for name in ("key", "status", "resource", "gpu_gb", "ram_gb")} for row in answer["rows"]] == [
        {"key": "only", "status": "fits", "resource": "gpu", "gpu_gb": 5.7, "ram_gb": None}
    ]  # the 5.27 GiB peak and the CUDA offset
    assert answer["judged"] == {"frame": "reference", "width": None, "height": None, "seconds": None, "quantize": None, "low_memory": None, "references": 0, "loras": 0}
    assert (answer["control_title"], answer["table_heading"], answer["rows"][0]["label"]) == (None, None, "")


def test_an_audio_model_on_a_mac_is_judged_on_unified_memory(audio):
    answer = audio_memory(audio, platform_key="mps", budget=_MAC, downloaded=True)

    assert (answer["kind"], answer["rows"][0]["resource"]) == ("unified", "unified")


def test_an_audio_model_that_is_too_large_says_so_and_reports_its_download(audio):
    answer = audio_memory(audio, platform_key="cuda", budget=MemoryBudget(4 * GIB, 16 * GIB), downloaded=False)

    assert (answer["rows"][0]["status"], answer["rows"][0]["resource"]) == ("too_large", "gpu")
    assert answer["budget"] == {"gpu_gb": 4.0, "ram_gb": 16.0}


def test_an_audio_model_without_a_budget_or_a_peak_has_no_answer(audio):
    assert audio_memory(audio, platform_key="cuda", budget=None, downloaded=True) is None
    assert audio_memory(audio, platform_key="unmeasured", budget=_CARD, downloaded=True) is None


# --- the lists ----------------------------------------------------------------------------------------------------------------


def test_a_listed_image_model_is_judged_on_the_reference_image_with_a_row_per_level(image_model, jobs):
    answer = reference_image_memory("zit", "/models/zit", levels=(4, 8), budget=_CARD, downloaded=True)

    assert (answer["judged"]["frame"], answer["current"], [row["key"] for row in answer["rows"]]) == ("reference", "none", ["none", "8", "4"])
    assert {(job.width, job.height) for job in jobs} == {(1024, 1024)}


def test_a_listed_model_row_carries_memory(image_model, video_model, monkeypatch):
    monkeypatch.setattr(workspace_api, "memory_budget", lambda: _CARD)
    monkeypatch.setattr(workspace_api, "model_downloaded", lambda *args, **kwargs: False)
    monkeypatch.setattr(model_status_module, "built_model_plan", lambda *args, **kwargs: None)
    config = make_web_config()
    image = ImageInventoryEntry(name="zit", family="flux2_klein", size="4b", source="alias", resolved_path="/resolved/zit")
    video = VideoInventoryEntry(name="ltx-8", family="ltx", supports_i2v=True, source="alias", resolved_path="/resolved/ltx-8")
    status = workspace_api._status_resolver(config, {}, lambda reference: None)

    image_row, video_row = status(image, "image"), status(video, "video")

    assert image_row["memory"]["judged"]["frame"] == "reference"
    assert image_row["memory"]["download_size_label"] == "17 GB"
    assert video_row["memory"]["current"] == "on"
    assert status(None, "image")["memory"] is None


def test_a_listed_model_that_cannot_be_judged_warns_and_has_no_memory(image_model, monkeypatch):
    monkeypatch.setattr(workspace_api, "memory_budget", lambda: _CARD)
    monkeypatch.setattr(workspace_api, "model_downloaded", lambda *args, **kwargs: True)
    monkeypatch.setattr(memory_api, "resolve_image_weights", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("unreadable")))
    image = ImageInventoryEntry(name="zit", family="flux2_klein", size="4b", source="alias", resolved_path="/resolved/zit")
    status = workspace_api._status_resolver(make_web_config(), {}, lambda reference: None)

    with pytest.warns(UserWarning, match="zit"):
        row = status(image, "image")

    assert (row["downloaded"], row["memory"]) == (True, None)


def test_an_audio_model_row_carries_memory(audio, monkeypatch):
    present = lambda target: True  # noqa: E731
    row = audio_models_module._status(audio, "cuda", _CARD, present)

    assert row["memory"]["rows"][0]["key"] == "only"
    assert audio_models_module._status(audio, "cuda", None, present)["memory"] is None


# --- the route ------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def route(monkeypatch, image_model):
    config = make_web_config()
    monkeypatch.setattr(web_server, "load_web_config", lambda: config)
    monkeypatch.setattr(web_server, "memory_budget", lambda: _CARD)
    monkeypatch.setattr(memory_api, "model_downloaded", lambda *args, **kwargs: True)
    return config


def _post(body=None, *, raw=None):
    with TestClient(web_server.app) as client:
        return client.post("/api/memory", json=body) if raw is None else client.post("/api/memory", content=raw, headers={"content-type": "application/json"})


def test_the_route_answers_with_the_shape(route):
    response = _post({"media": "image", "model": "zit", "quantize": 4, "ratio": "2:3", "size": "m"})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == _ANSWER_KEYS
    assert (body["current"], body["judged"]["width"], body["judged"]["height"]) == ("4", 832, 1216)


def test_the_route_answers_null_without_a_budget(route, monkeypatch):
    monkeypatch.setattr(web_server, "memory_budget", lambda: None)

    response = _post({"media": "image", "model": "zit"})

    assert (response.status_code, response.json()) == (200, None)


@pytest.mark.parametrize("body", [{"media": "text"}, {"media": "image"}, {"media": "image", "model": "zit", "quantize": 5}, {"media": "image", "model": "zit", "lora": "a,,b"}])
def test_the_route_refuses_a_bad_request_with_a_sentence(route, body):
    response = _post(body)

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], str) and response.json()["detail"]


def test_the_route_refuses_a_body_that_is_not_json(route):
    assert _post(raw=b"{nope").status_code == 422


# --- what the request path may not do, and the jobs it must build as a run does ---------------------------------------------


def test_a_listing_never_lets_the_hub_be_read(image_model, monkeypatch):
    monkeypatch.setattr(workspace_api, "memory_budget", lambda: _CARD)
    monkeypatch.setattr(workspace_api, "model_downloaded", lambda *args, **kwargs: False)
    monkeypatch.setattr(model_status_module, "built_model_plan", lambda *args, **kwargs: None)
    image = ImageInventoryEntry(name="zit", family="flux2_klein", size="4b", source="alias", resolved_path="/resolved/zit")

    workspace_api._status_resolver(make_web_config(), {}, lambda reference: None)(image, "image")

    assert len(image_model.asked) == 1 and not image_model.asked[0].get("hub")


def test_the_route_never_lets_the_hub_be_read(route, image_model):
    _post({"media": "image", "model": "zit"})

    assert len(image_model.asked) == 1 and not image_model.asked[0].get("hub")


def test_a_masked_edit_is_judged_at_the_canvas_the_model_draws_for_a_mask(image_model, jobs):
    answer = _image({"kind": "edit", "source_width": 3000, "source_height": 2000, "masked": True})

    expected = edit_model_size(3000, 2000, masked=True, extend=None)
    assert (answer["judged"]["width"], answer["judged"]["height"]) == expected
    assert expected != edit_model_size(3000, 2000, masked=False, extend=None)
    assert jobs[0].reference_megapixels == (6.0,)


def test_an_h3_clip_with_an_upscale_that_is_reviewed_is_judged_as_the_run_is(video_model, jobs):
    video_model.family = "h3"
    video_model.clip = VideoClip(steps=8, width=768, height=448, num_frames=124)

    _video({"upscale": 2, "review": True})
    _video({"upscale": 2})
    _video({"review": True})

    assert [(job.upscale, job.review) for job in jobs] == [(True, True), (True, False), (False, False)]
