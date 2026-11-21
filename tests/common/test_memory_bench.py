"""Tests for the memory bench: the case lists, the CLI arguments, the /proc readers, the peaks row, the calibration rule and the size terms."""

from __future__ import annotations

import argparse
import getpass
import json
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import bench.memory.__main__ as bench_main
from bench.memory import audio, video
from bench.memory.cases import MEDIA, expand_cases, load_cases
from bench.memory.fit import (
    card_the_job_had,
    format_report,
    measured_ram_bytes,
    sampler_gap_bytes,
    GIB,
    measured_gpu_bytes,
    with_brackets,
    tolerance_bytes,
    format_probes,
    format_rule,
    allocated_rows,
    driver_view,
    edit_checks,
    hold_brackets,
    model_terms,
    reestimate,
    rule_checks,
    size_terms,
    upscale_checks,
)
from bench.memory.image import case_argv, case_canvas, ensure_sources, estimate
from bench.memory.measure import GpuSampler, ProcSampler, call_cli, current_footprint, lifetime_max_footprint, rss_fields, run_case, says_out_of_memory, smi_memory
from bench.memory import probes
from bench.memory.probes import watch_placement
from zvisiongenerator.edit_cli import refuse_edit_options
from zvisiongenerator.image_cli import _build_parser
from zvisiongenerator.utils.memory_offsets import CUDA_OFFSET_BYTES, CUDA_OFFSET_LINUX_BYTES, CUDA_OFFSET_WINDOWS_BYTES
from zvisiongenerator.utils.model_memory import MemoryBudget

_SIZES = {"1:1": {"m": {"width": 1024, "height": 1024}, "xl": {"width": 1600, "height": 1600}}}
_STATUS = "Name:\tpython\nRssAnon:\t  2048 kB\nRssFile:\t  1024 kB\nRssShmem:\t   512 kB\nThreads:\t3\n"


def _row(case_id: str, pixels: int, peak_gib: float, *, model: str = "klein9b", family: str = "flux2_klein", references: int = 0, **extra: object) -> dict:
    return {
        "id": case_id,
        "family": family,
        "platform": "cuda",
        "model": model,
        "quantize": 8,
        "pixels": pixels,
        "references": references,
        "status": "ok",
        "platform_peak_bytes": int(peak_gib * GIB),
    } | extra


# --- /proc readers -----------------------------------------------------------------------------------------------


def test_rss_fields_reads_the_three_counters_in_bytes() -> None:
    assert rss_fields(_STATUS) == {"RssAnon": 2048 * 1024, "RssFile": 1024 * 1024, "RssShmem": 512 * 1024}


def test_rss_fields_leaves_out_counters_the_kernel_does_not_report() -> None:
    assert rss_fields("Name:\tpython\nRssAnon:\t 4 kB\n") == {"RssAnon": 4096}
    assert rss_fields("") == {}


def _sampler_over(readings: list[str]) -> ProcSampler:
    """Return a sampler that reads each status text in turn; the first reading is the start, the last the stop."""
    statuses = iter(readings)
    return ProcSampler(interval=3600, read=lambda _path: next(statuses))


def test_proc_sampler_keeps_the_peak_of_each_counter() -> None:
    sampler = _sampler_over(
        [
            "RssAnon: 100 kB\nRssFile: 10 kB\n",  # start
            "RssAnon: 300 kB\nRssFile: 5 kB\nRssShmem: 7 kB\n",  # stop
        ]
    )
    sampler.start()
    peaks = sampler.stop()
    assert peaks == {
        "rss_anon_peak_bytes": 300 * 1024,
        "rss_file_peak_bytes": 10 * 1024,
        "rss_shmem_peak_bytes": 7 * 1024,
        "rss_anon_shmem_peak_bytes": 307 * 1024,
        "rss_anon_shmem_file_peak_bytes": 312 * 1024,
    }


def test_proc_sampler_keeps_the_highest_sum_of_one_reading_not_the_sum_of_the_separate_peaks() -> None:
    sampler = _sampler_over(
        [
            "RssAnon: 100 kB\nRssFile: 90 kB\nRssShmem: 0 kB\n",  # start
            "RssAnon: 900 kB\nRssFile: 10 kB\nRssShmem: 50 kB\n",  # the anonymous peak
            "RssAnon: 200 kB\nRssFile: 20 kB\nRssShmem: 600 kB\n",  # the shared peak
            "RssAnon: 150 kB\nRssFile: 300 kB\nRssShmem: 10 kB\n",  # the file-backed peak
        ]
    )
    sampler.start()
    sampler._sample()  # noqa: SLF001 - readings without waiting on the poll interval
    sampler._sample()  # noqa: SLF001
    peaks = sampler.stop()

    assert (peaks["rss_anon_peak_bytes"], peaks["rss_shmem_peak_bytes"], peaks["rss_file_peak_bytes"]) == (900 * 1024, 600 * 1024, 300 * 1024)
    assert peaks["rss_anon_shmem_peak_bytes"] == 950 * 1024  # not 900 + 600
    assert peaks["rss_anon_shmem_file_peak_bytes"] == 960 * 1024  # not 900 + 600 + 300


def test_proc_sampler_reports_none_where_proc_has_nothing() -> None:
    sampler = ProcSampler(interval=3600, read=lambda _path: None)
    sampler.start()
    assert sampler.stop() == dict.fromkeys(("rss_anon_peak_bytes", "rss_file_peak_bytes", "rss_shmem_peak_bytes", "rss_anon_shmem_peak_bytes", "rss_anon_shmem_file_peak_bytes"))


def test_the_rusage_v4_struct_places_the_lifetime_max_footprint_after_the_uuid_and_twenty_eight_counters() -> None:
    from bench.memory.measure import _rusage_v4_struct

    fields = [name for name, *_ in _rusage_v4_struct()._fields_]

    assert fields.index("ri_lifetime_max_phys_footprint") == 1 + 28 and fields.index("ri_phys_footprint") == 1 + 7
    assert len(fields) == 1 + 35


def test_the_lifetime_max_footprint_is_read_on_a_mac_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bench.memory.measure.sys.platform", "linux")

    assert lifetime_max_footprint() is None
    assert current_footprint() is None


def test_the_current_footprint_is_the_kernels_present_value_not_its_lifetime_maximum(monkeypatch: pytest.MonkeyPatch) -> None:
    info = SimpleNamespace(ri_phys_footprint=3 * GIB, ri_lifetime_max_phys_footprint=9 * GIB)
    monkeypatch.setattr("bench.memory.measure._rusage_v4", lambda: info)

    assert current_footprint() == 3 * GIB and lifetime_max_footprint() == 9 * GIB


def test_a_row_records_the_footprint_peak_only_when_it_could_be_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, measured: None) -> None:
    def fake_cli(argv: list[str]) -> None:
        (tmp_path / "picture.png").write_bytes(b"x")

    monkeypatch.setattr("bench.memory.measure.lifetime_max_footprint", lambda: 7 * GIB)
    assert _run([], fake_cli, tmp_path)["footprint_peak_bytes"] == 7 * GIB

    monkeypatch.setattr("bench.memory.measure.lifetime_max_footprint", lambda: None)
    assert "footprint_peak_bytes" not in _run([], fake_cli, tmp_path)


def test_gpu_sampler_keeps_the_lowest_free_memory_and_the_driver_total() -> None:
    readings = iter([(8 * GIB, 10 * GIB), None, (2 * GIB, 10 * GIB), (5 * GIB, 10 * GIB)])
    sampler = GpuSampler(interval=3600, read=lambda: next(readings))
    sampler.start()  # the thread waits out its interval, so only stop() reads
    sampler._sample()
    sampler._sample()
    sampler._sample()

    assert sampler.stop() == {"gpu_free_min_bytes": 2 * GIB, "gpu_driver_total_bytes": 10 * GIB}


def test_gpu_sampler_reports_none_before_cuda_starts() -> None:
    sampler = GpuSampler(interval=3600, read=lambda: None)
    sampler.start()

    assert sampler.stop() == {"gpu_free_min_bytes": None, "gpu_driver_total_bytes": None}


def test_smi_memory_reads_used_and_total_of_the_card_cuda_uses_in_bytes_or_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from zvisiongenerator.backends.cuda import nvml

    monkeypatch.setattr(nvml, "card_bus_id", lambda: "0000:41:00.0")
    asked: list[list[str]] = []
    assert smi_memory(lambda command: asked.append(command) or "682, 10240\n") == (682 * 1024**2, 10240 * 1024**2)
    assert asked[0][:3] == ["nvidia-smi", "-i", "0000:41:00.0"]  # the card CUDA names cuda:0, not nvidia-smi's first
    assert smi_memory(lambda _command: "") is None

    def missing(_command: list[str]) -> str:
        raise FileNotFoundError("nvidia-smi")

    assert smi_memory(missing) is None

    monkeypatch.setattr(nvml, "card_bus_id", lambda: None)
    assert smi_memory(lambda _command: pytest.fail("asked nvidia-smi without a card")) is None


def test_driver_view_separates_what_the_card_took_besides_the_tensors() -> None:
    row = {
        "gpu_free_min_bytes": GIB // 10,
        "gpu_driver_total_bytes": 10 * GIB,
        "gpu_baseline_used_bytes": GIB,
        "gpu_smi_total_bytes": 10 * GIB + GIB // 2,
        "platform_allocated_peak_bytes": 7 * GIB,
    }

    assert driver_view(row) == {"taken": 10 * GIB - GIB // 10, "others": GIB, "beside_tensors": 10 * GIB - GIB // 10 - 8 * GIB, "driver_reserved": GIB // 2}
    assert driver_view(row | {"gpu_free_min_bytes": None}) is None


# --- cases and arguments ------------------------------------------------------------------------------------------


def test_expand_cases_fills_shared_settings_and_filters_by_platform() -> None:
    cases = load_cases("image")
    cuda = expand_cases(cases, media="image", platform="cuda")
    mac = expand_cases(cases, media="image", platform="mac")
    assert {case["id"] for case in cuda if case["family"] == "moss"} == {
        "moss-m",
        "moss-xl",
        "moss-2048",
        "moss-none-m",
        "moss-none-xl",
        "moss-none-2048",
        "moss-q4-m",
        "moss-q4-xl",
        "moss-q4-2048",
        "moss-sft-q4-m",
        "moss-sft-q4-m-lora",
    }
    assert {case["id"] for case in mac if case["family"] == "moss"} == {"moss-m", "moss-xl"}
    assert all(case["seed"] == cases["defaults"]["seed"] for case in cuda)
    assert all(case["quantize"] == cases["defaults"]["quantize"] for case in cuda if "-q4-" not in case["id"] and "-none-" not in case["id"] and not case["id"].startswith("klein4b-q4"))
    assert len({case["id"] for case in mac}) == len(mac)


@pytest.mark.parametrize("media", MEDIA)
def test_no_case_that_runs_on_a_mac_sets_a_card_budget(media: str) -> None:
    mac = expand_cases(load_cases(media), media=media, platform="mac")
    assert [case["id"] for case in mac if case.get("budget_gib")] == []  # a Mac is simulated by total_gib; a budget raises
    assert {"klein4b-q4-m-turns", "klein4b-q4-xl-turns"} <= {case["id"] for case in expand_cases(load_cases("image"), media="image", platform="cuda")}


def test_expand_cases_selects_families_and_ids_and_rejects_unknown_ones() -> None:
    cases = load_cases("image")
    assert {case["family"] for case in expand_cases(cases, media="image", platform="cuda", families=["zimage"])} == {"zimage"}
    assert [case["id"] for case in expand_cases(cases, media="image", platform="cuda", ids=["zit-m"])] == ["zit-m"]
    with pytest.raises(ValueError, match="Unknown family"):
        expand_cases(cases, media="image", platform="cuda", families=["nope"])
    with pytest.raises(ValueError, match="No case"):
        expand_cases(cases, media="image", platform="mac", ids=["moss-q4-m"])


def test_the_case_file_covers_the_cuda_pass() -> None:
    cases = {case["id"]: case for case in expand_cases(load_cases("image"), media="image", platform="cuda")}
    assert {case["quantize"] for case in cases.values()} == {None, 4, 8}
    assert {case["quantize"] for case in cases.values() if "-q4-" in case["id"]} == {4}
    assert {case["quantize"] for case in cases.values() if "-none-" in case["id"]} == {None}
    # Every level the estimate has a term for is measured at 1 MP and at 1600x1600, and q4 with references too.
    assert {"krea2-q4-m", "krea2-q4-xl", "ideo-q4-m", "ideo-q4-xl", "klein9b-q4-m-ref3", "ideo-none-m", "klein9b-none-m", "zit-q4-m"} <= set(cases)
    assert {cases[name]["references"] for name in ("klein9b-m", "klein9b-m-ref1", "klein9b-m-ref3")} == {0, 1, 3}
    assert cases["klein9b-edit-2048"]["edit"] == "edit-2048"
    assert cases["klein9b-extend-right512"]["extend"] == "right=512"
    q4 = ("klein9b-q4-m", "klein9b-q4-m-ref1", "klein9b-q4-xl", "klein9b-q4-edit-2048", "klein9b-q4-extend-right512")
    assert {cases[name]["model"] for name in q4} == {"klein9b"} and {cases[name]["references"] for name in q4[:2]} == {0, 1}
    assert cases["klein9b-q4-edit-2048"]["edit"] == "edit-2048" and cases["klein9b-q4-extend-right512"]["extend"] == "right=512"
    assert cases["zit-xl-up2"]["upscale"] == 2 and cases["zit-xl-up2"]["size"] == cases["zit-xl"]["size"]


def test_case_canvas_is_the_output_size_the_model_draws() -> None:
    pictures = {"edit-2048": {"width": 2048, "height": 2048}, "extend-1024": {"width": 1024, "height": 1024}}
    plain = {"ratio": "1:1", "size": "xl"}
    assert case_canvas(plain, _SIZES, pictures) == (1600, 1600)
    assert case_canvas(plain | {"upscale": 2}, _SIZES, pictures) == (1600, 1600)
    assert case_canvas({"edit": "edit-2048"}, _SIZES, pictures) == (2048, 2048)
    width, height = case_canvas({"edit": "extend-1024", "extend": "right=512"}, _SIZES, pictures)
    assert height < 1024 < width < 1024 + 512


def _args_for(case: dict, run_dir: Path) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(case_argv(case, run_dir))
    refuse_edit_options(parser, args)
    return args


@pytest.mark.parametrize("platform_name", ["cuda", "mac"])
def test_every_case_argument_list_is_accepted_by_the_real_ziv_image_parser(platform_name: str, tmp_path: Path) -> None:
    for case in expand_cases(load_cases("image"), media="image", platform=platform_name):
        args = _args_for(case, tmp_path)
        assert args.model == case["model"] and args.quantize == case["quantize"] and args.runs == 1
        assert args.no_enhance and args.seed == case["seed"] and args.output == str(tmp_path / "out" / case["id"])


def test_case_argv_carries_size_references_upscale_and_edit_options(tmp_path: Path) -> None:
    cases = {case["id"]: case for case in expand_cases(load_cases("image"), media="image", platform="cuda")}
    refs = _args_for(cases["klein9b-m-ref3"], tmp_path)
    assert (refs.ratio, refs.size) == ("1:1", "m") and len(refs.reference_paths) == 3 and len(set(refs.reference_paths)) == 3
    assert _args_for(cases["zit-xl-up2"], tmp_path).upscale == 2
    extend = _args_for(cases["klein9b-extend-right512"], tmp_path)
    assert extend.edit_extend == ("right", 512) and extend.ratio is None and extend.edit_source == str(tmp_path / "sources" / "extend-1024.png")
    assert _args_for(cases["klein9b-edit-2048"], tmp_path).edit_extend is None


def test_ensure_sources_writes_the_synthetic_pictures_once(tmp_path: Path) -> None:
    from PIL import Image

    cases = load_cases("image")
    case = next(case for case in expand_cases(cases, media="image", platform="cuda") if case["id"] == "klein9b-m-ref3")
    ensure_sources(case, tmp_path, cases)
    files = sorted((tmp_path / "sources").glob("*.png"))
    assert [file.name for file in files] == ["reference-1.png", "reference-2.png", "reference-3.png"]
    assert Image.open(files[0]).size == (cases["reference"]["width"], cases["reference"]["height"])
    assert files[0].read_bytes() != files[1].read_bytes()
    before = files[0].stat().st_mtime_ns
    ensure_sources(case, tmp_path, cases)
    assert files[0].stat().st_mtime_ns == before


# --- measuring one case ----------------------------------------------------------------------------------------------


@pytest.fixture
def measured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bench.memory.measure.platform_peak", lambda: {"platform_peak_bytes": 9 * GIB, "gpu_total_bytes": 10 * GIB})
    monkeypatch.setattr("bench.memory.measure.smi_memory", lambda: (GIB, 10 * GIB))
    monkeypatch.setattr("bench.memory.measure._driver_memory", lambda: None)


def _case() -> dict:
    return {"id": "c", "family": "zimage", "model": "zit", "quantize": 8, "ratio": "1:1", "size": "m", "references": 0, "steps": 2, "seed": 1, "width": 1024, "height": 1024, "pixels": 1024 * 1024}


def _estimate_environment(monkeypatch: pytest.MonkeyPatch, *, card_gib: float = 10.0) -> None:
    """Stand in for the model lookup and the weights: a Z-Image-like q4 model on a card of *card_gib*."""
    from zvisiongenerator.utils.memory_verdict import MemoryWeights
    from zvisiongenerator.utils.model_memory import MemoryBudget, WeightTotals
    from zvisiongenerator.utils.weight_totals import ResolvedWeights

    components = {
        "text_encoder": WeightTotals(float_matrix_elements=int(2.5 * 2**30 / 0.51625)),
        "transformer": WeightTotals(float_matrix_elements=int(2.9 * 2**30 / 0.51625)),
        "vae": WeightTotals(float_other_elements=int(0.078 * 2**30)),
    }
    monkeypatch.setattr("bench.memory.machine.platform_label", lambda: "cuda")
    monkeypatch.setattr("zvisiongenerator.backends.accelerator_memory_budget_or_none", lambda: MemoryBudget(int(card_gib * GIB), 32 * GIB))
    monkeypatch.setattr("zvisiongenerator.utils.config.load_config", lambda: {})
    monkeypatch.setattr("zvisiongenerator.utils.model_metadata.alias_map", lambda config: {})
    monkeypatch.setattr("zvisiongenerator.utils.paths.canonical_model_name", lambda name, **kwargs: name)
    monkeypatch.setattr("zvisiongenerator.utils.paths.resolve_model_path", lambda name, **kwargs: name)
    resolved = ResolvedWeights(MemoryWeights(components=components), family="zimage")
    monkeypatch.setattr("zvisiongenerator.utils.weight_totals.resolve_image_weights", lambda name, resolved_path, **kwargs: resolved)


def test_estimate_is_the_verdict_with_the_cuda_offset_and_the_card_it_was_judged_against(monkeypatch: pytest.MonkeyPatch) -> None:
    from zvisiongenerator.utils.memory_offsets import CUDA_OFFSET_BYTES

    _estimate_environment(monkeypatch)
    row = estimate(_case() | {"quantize": 4})
    assert row is not None and row["placement"] == "resident" and row["status"] == "fits" and row["resource"] == "gpu"
    assert row["gpu_bytes"] == estimate(_case() | {"quantize": 4, "placement": "resident"})["gpu_bytes"]
    assert row["gpu_bytes"] > CUDA_OFFSET_BYTES + 5 * GIB  # the weights, the working memory and the offset
    assert row["budget_bytes"] == 10 * GIB


def test_estimate_reads_a_card_of_the_cases_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    _estimate_environment(monkeypatch)
    row = estimate(_case() | {"quantize": 4, "budget_gib": 6})
    assert row is not None and row["placement"] == "turns"


def test_estimate_is_none_without_a_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    _estimate_environment(monkeypatch)
    monkeypatch.setattr("zvisiongenerator.backends.accelerator_memory_budget_or_none", lambda: None)
    assert estimate(_case()) is None


def test_the_image_estimate_judges_the_job_the_apps_builders_make(monkeypatch: pytest.MonkeyPatch) -> None:
    from zvisiongenerator.processing.edit_region import edit_model_size
    from zvisiongenerator.utils.memory_verdict import MemoryJob, MemoryVerdict

    _estimate_environment(monkeypatch)
    judged: list[MemoryJob] = []
    monkeypatch.setattr("zvisiongenerator.utils.memory_verdict.judge", lambda job, machine, weights: judged.append(job) or MemoryVerdict(1, 2, "streamed", "fits", "gpu"))

    estimate(_case() | {"references": 2, "upscale": 2, "lora_bytes": 777})
    estimate(_case() | {"edit": "edit-2048", "width": 2048, "height": 2048})
    estimate(_case() | {"edit": "extend-1024", "extend": "right=512"})

    refined, whole, extension = judged
    assert (refined.family, refined.quantize, refined.reference_megapixels, refined.from_picture, refined.lora_bytes) == ("zimage", 8, (1024 * 1024 / 1e6,) * 2, True, 777)
    assert (whole.width, whole.height, whole.reference_megapixels, whole.from_picture) == (2048, 2048, (2048 * 2048 / 1e6,), False)  # a whole-picture edit holds its own picture, at its own size
    assert (extension.width, extension.height) == edit_model_size(1024, 1024, masked=False, extend=("right", 512))
    assert extension.from_picture and extension.reference_megapixels == (1024 * 1024 / 1e6,)


def test_a_forced_placement_is_judged_by_the_apps_own_engine_and_the_engine_is_left_as_it_was(monkeypatch: pytest.MonkeyPatch) -> None:
    from zvisiongenerator.utils import memory_verdict

    _estimate_environment(monkeypatch)
    real = memory_verdict.image_placement_candidates
    case = _case() | {"quantize": 4}

    resident, turns = (estimate(case | {"placement": placement}) for placement in ("resident", "turns"))

    assert (resident["placement"], turns["placement"]) == ("resident", "turns")
    assert turns["gpu_bytes"] < resident["gpu_bytes"]
    assert estimate(case | {"loaded_placement": "turns"})["gpu_bytes"] == turns["gpu_bytes"]  # the placement the loader took is judged as one that was forced
    assert estimate(case)["placement"] == "resident" and memory_verdict.image_placement_candidates is real


def test_the_video_estimate_judges_the_clip_job_the_apps_builder_makes(monkeypatch: pytest.MonkeyPatch) -> None:
    from zvisiongenerator.utils.memory_verdict import MemoryJob

    seen: list[MemoryJob] = []
    monkeypatch.setattr("zvisiongenerator.backends.accelerator_memory_budget_or_none", lambda: MemoryBudget(24 * GIB, 64 * GIB))
    monkeypatch.setattr(
        "zvisiongenerator.utils.memory_verdict.judge", lambda job, machine, weights: seen.append(job) or SimpleNamespace(gpu_bytes=1, ram_bytes=1, status="fits", resource=None, placement=None)
    )
    clip = {"family": "h3", "model": "h3", "width": 768, "height": 448, "frames": 22, "steps": 5}

    video.estimate(clip | {"start_image": True, "upscale": 2, "references": 1})
    video.estimate(clip | {"extend_source": "extend-xs"})

    first, second = seen
    assert (first.mode, first.upscale, first.references, first.forwards, first.low_memory) == ("i2v", True, 1, 5, True)  # the steps are the transformer forwards, as the CLI counts them
    assert (second.mode, second.upscale) == ("extend", False)


def test_the_estimates_of_the_three_media_share_one_machine_and_one_row_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    from bench.memory.machine import case_machine

    monkeypatch.setattr("zvisiongenerator.backends.accelerator_memory_budget_or_none", lambda: MemoryBudget(10 * GIB, 32 * GIB))
    assert case_machine({"platform": "cuda"}).budget.gpu_bytes == 10 * GIB
    assert case_machine({"platform": "cuda", "budget_gib": 6}).budget.gpu_bytes == 6 * GIB
    assert case_machine({"platform": "mac"}) is None  # this machine is a card, not a Mac
    foreign = case_machine({"platform": "mac", "mac_unified_bytes": 16 * GIB, "total_gib": 8})
    assert (foreign.mac, foreign.foreign_mac, foreign.budget.unified_bytes) == (True, True, 8 * GIB)


def test_a_mac_row_is_judged_as_a_mac_with_the_budget_it_is_given_on_any_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    _estimate_environment(monkeypatch)
    case = _case() | {"platform": "mac", "mac_unified_bytes": 32 * GIB}
    plain = estimate(case)
    assert estimate(case | {"mac_unified_bytes": 1 * GIB})["status"] == "too_large"
    refined = estimate(case | {"upscale": 2, "width": 1600, "height": 1600})
    assert plain is not None and refined is not None
    assert (plain["placement"], plain["resource"], plain["system_bytes"]) == ("unified", "unified", None)
    assert refined["gpu_bytes"] > plain["gpu_bytes"]
    assert estimate(_case() | {"platform": "mac"}) is None


def test_a_mac_case_under_a_total_is_judged_as_the_smaller_mac(monkeypatch: pytest.MonkeyPatch) -> None:
    _estimate_environment(monkeypatch)
    monkeypatch.setattr("bench.memory.machine.platform_label", lambda: "mac")
    monkeypatch.setattr("zvisiongenerator.backends.accelerator_memory_budget_or_none", lambda: MemoryBudget(26 * GIB, None, 32 * GIB))
    case = _case() | {"platform": "mac"}

    assert estimate(case)["unified_bytes"] == 32 * GIB
    smaller = estimate(case | {"total_gib": 4})
    assert smaller["unified_bytes"] == 4 * GIB and smaller["status"] == "too_large"
    assert estimate(case | {"budget_gib": 4.0})["unified_bytes"] == 32 * GIB  # a card's budget means nothing on a Mac


@pytest.fixture
def fresh_budget():
    from zvisiongenerator import backends

    real = backends.get_accelerator_memory_budget
    yield backends
    backends.get_accelerator_memory_budget = real
    backends.accelerator_memory_budget_or_none.cache_clear()


_DECODE_BUDGET_ENV = "LTX2_VAE_DECODE_BUDGET_GB"


@pytest.fixture
def mac_probe_state(monkeypatch: pytest.MonkeyPatch):
    """Restore what ``show_mac_of`` patches: the size readers of the Mac memory module and of the Mac video backend, and the decode budget variable."""
    import sys
    import types

    from zvisiongenerator.backends.mac import memory as mac_memory

    video = types.ModuleType("zvisiongenerator.backends.mac.video")
    video.memory_size_bytes = lambda: 32 * GIB
    monkeypatch.setitem(sys.modules, "zvisiongenerator.backends.mac.video", video)
    monkeypatch.setattr(mac_memory, "memory_size_bytes", lambda: 32 * GIB)
    monkeypatch.setenv(_DECODE_BUDGET_ENV, "unset")
    monkeypatch.delenv(_DECODE_BUDGET_ENV)
    return mac_memory, video


def test_show_mac_of_makes_the_app_read_a_mac_with_less_memory(monkeypatch: pytest.MonkeyPatch, fresh_budget, mac_probe_state) -> None:
    monkeypatch.setattr(fresh_budget, "get_accelerator_memory_budget", lambda: MemoryBudget(26 * GIB, None, 32 * GIB))
    fresh_budget.accelerator_memory_budget_or_none.cache_clear()

    probes.show_mac_of(16)

    assert fresh_budget.accelerator_memory_budget_or_none() == MemoryBudget(26 * GIB, None, 16 * GIB)


def test_show_mac_of_also_drives_the_size_dependent_paths_the_ltx_loader_reads(monkeypatch: pytest.MonkeyPatch, fresh_budget, mac_probe_state) -> None:
    import os

    from zvisiongenerator.utils.model_memory import ltx_decode_budget_bytes

    mac_memory, video = mac_probe_state
    monkeypatch.setattr(fresh_budget, "get_accelerator_memory_budget", lambda: MemoryBudget(26 * GIB, None, 32 * GIB))

    probes.show_mac_of(16)

    assert mac_memory.memory_size_bytes() == 16 * GIB and video.memory_size_bytes() == 16 * GIB  # the loader's 24 GB streaming test reads this
    assert ltx_decode_budget_bytes(16 * GIB, os.environ) == 8 * GIB  # half of the simulated total


def test_show_mac_of_keeps_a_decode_budget_the_user_set(monkeypatch: pytest.MonkeyPatch, fresh_budget, mac_probe_state) -> None:
    import os

    monkeypatch.setattr(fresh_budget, "get_accelerator_memory_budget", lambda: MemoryBudget(26 * GIB, None, 32 * GIB))
    monkeypatch.setenv(_DECODE_BUDGET_ENV, "5")

    probes.show_mac_of(16)

    assert os.environ[_DECODE_BUDGET_ENV] == "5"


def test_call_cli_reads_an_out_of_memory_error_a_cli_only_logs_before_it_exits() -> None:
    import logging

    def video_like_cli(_argv: list[str]) -> None:
        try:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB.")
        except RuntimeError:
            logging.getLogger("zvisiongenerator.video_cli").exception("Failed to load video model ltx-8")
        raise SystemExit(2)

    reason = call_cli(video_like_cli, [])

    assert reason is not None and reason.startswith("the CLI exited with 2") and "out of memory" in reason.lower()


def test_call_cli_reads_an_out_of_memory_line_written_to_stderr() -> None:
    import sys

    def cli(_argv: list[str]) -> None:
        print("error: the GPU ran out of memory while decoding", file=sys.stderr)
        raise SystemExit(1)

    assert "out of memory" in (call_cli(cli, []) or "")


def test_the_apps_own_out_of_gpu_memory_wording_is_read_as_out_of_memory(tmp_path: Path) -> None:
    def cli(_argv: list[str]) -> None:
        import logging

        logging.getLogger("zvisiongenerator.video_cli").error("MiniMax H3 ran out of GPU memory at 512x320, 124 frames. Try a shorter clip.")
        raise SystemExit(2)

    reason = call_cli(cli, [])
    row = {"id": "h3", "status": "failed", "error": reason, "estimate": {"status": "too_large", "gpu_bytes": 12 * GIB}, "platform": "cuda"}
    checks, left_out = rule_checks([row])

    assert [check["check"] for check in checks] == ["out of memory reads Too large"] and not left_out


def test_call_cli_keeps_the_plain_exit_reason_when_the_failure_is_not_about_memory() -> None:
    def cli(_argv: list[str]) -> None:
        raise SystemExit(2)

    assert call_cli(cli, []) == "the CLI exited with 2"
    assert call_cli(lambda _argv: None, []) is None


def test_show_machine_of_picks_the_mac_or_the_card_by_platform(monkeypatch: pytest.MonkeyPatch, fresh_budget) -> None:
    from types import SimpleNamespace

    calls: list[tuple] = []
    monkeypatch.setattr(probes, "show_mac_of", lambda total: calls.append(("mac", total)))
    monkeypatch.setattr(probes, "show_card_of", lambda gib: calls.append(("card", gib)))
    monkeypatch.setattr(probes, "sys", SimpleNamespace(platform="darwin", modules={}))
    probes.show_machine_of(total_gib=16)
    with pytest.raises(ValueError, match="total memory only"):
        probes.show_machine_of(6)
    monkeypatch.setattr(probes, "sys", SimpleNamespace(platform="linux", modules={}))
    probes.show_machine_of(6)

    assert calls == [("mac", 16), ("card", 6)]
    with pytest.raises(ValueError, match="CUDA card"):
        probes.show_machine_of(6, 16)


def test_ui_applies_the_budget_then_starts_the_web_ui_with_the_other_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[tuple] = []
    monkeypatch.setattr(bench_main, "show_machine_of", lambda gib, total=None: started.append(("machine", gib, total)))
    monkeypatch.setattr("zvisiongenerator.web.main", lambda argv, prog: started.append(("ui", argv, prog)))

    assert bench_main.main(["ui", "--total-gib", "16", "--host", "local", "--port", "9000"]) == 0
    assert started == [("machine", None, 16.0), ("ui", ["--host", "local", "--port", "9000"], "ziv ui")]
    with pytest.raises(SystemExit):
        bench_main.main(["ui", "--budget-gib", "10.67", "--background"])
    with pytest.raises(SystemExit):
        bench_main.main(["fit", "results.jsonl", "--host", "local"])


def test_run_total_gib_is_recorded_on_a_mac_and_the_budget_on_a_card(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spawned: list[dict] = []
    monkeypatch.setattr(bench_main, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(bench_main, "_spawn", lambda case, run_dir, timeout: spawned.append(case) or {"id": case["id"], "status": "ok", "seconds": 1})
    monkeypatch.setattr(bench_main, "platform_label", lambda: "mac")
    argv = ["run", "--media", "image", "--case", "zit-q4-m", "--total-gib", "16", "--label", "t"]
    assert bench_main.main(argv) == 0
    assert [case.get("total_gib") for case in spawned] == [16.0]
    assert bench_main._key({"id": "a"}) != bench_main._key({"id": "a", "total_gib": 16.0})
    with pytest.raises(SystemExit):
        bench_main.main([*argv, "--budget-gib", "10.67"])  # a Mac has no card to size
    monkeypatch.setattr(bench_main, "platform_label", lambda: "cuda")
    with pytest.raises(SystemExit):
        bench_main.main(argv)


def test_fit_judges_mac_rows_against_the_total_memory_it_is_given(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[dict] = []
    results = tmp_path / "results.jsonl"
    results.write_text(json.dumps({"id": "m", "platform": "mac", "status": "ok"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(bench_main, "_estimate_for", lambda row: seen.append(row) or None)

    bench_main.main(["fit", str(results), "--total-gib", "16"])

    assert [row["mac_unified_bytes"] for row in seen] == [16 * GIB]


def test_fit_json_fits_the_allocated_peak_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows = [
        _row("m", _MP1, 9.0, quantize=4, platform_allocated_peak_bytes=int(5.0 * GIB)),
        _row("xl", _XL, 9.0, quantize=4, platform_allocated_peak_bytes=int(8.0 * GIB)),
        _row("reserved-only", _XL, 12.0, quantize=4),
    ]
    results = tmp_path / "results.jsonl"
    results.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert bench_main.main(["fit", str(results), "--stored", "--json"]) == 0

    terms = json.loads(capsys.readouterr().out)["flux2_klein/cuda/q4"]
    assert terms["per_megapixel"] == round(3 * GIB / ((_XL - _MP1) / 1e6))
    with pytest.raises(SystemExit):
        bench_main.main(["fit", str(results), "--json", "--allocated"])  # the allocated peak is the only one fitted


def test_the_terms_report_fits_the_allocated_peak_only() -> None:
    rows = [_row("m", _MP1, 9.0, quantize=4, platform_allocated_peak_bytes=int(5.0 * GIB)), _row("xl", _XL, 9.0, quantize=4, platform_allocated_peak_bytes=int(8.0 * GIB))]

    report = format_report(rows)

    assert "from the allocated peak" in report and "reserved peak" not in report.replace("max_memory_reserved", "")
    assert f"per_megapixel={round(3 * GIB / ((_XL - _MP1) / 1e6))}" in report


_ESTIMATE = {"gpu_bytes": 5 * GIB, "system_bytes": 7 * GIB, "placement": "streamed"}


def _run(argv: list[str], cli, out_dir: Path, case: dict | None = None, probes=None, hold=None) -> dict:
    return run_case(case or _case(), argv, cli, out_dir, estimate=lambda _case: _ESTIMATE, outputs=frozenset({".png"}), probes=probes, hold=hold, storage=lambda: 0)


def test_run_case_records_the_peaks_and_the_estimate(tmp_path: Path, measured: None) -> None:
    def fake_cli(argv: list[str]) -> None:
        assert argv == ["--go"]
        (tmp_path / "picture.png").write_bytes(b"x")

    row = _run(["--go"], fake_cli, tmp_path)
    assert (row["status"], row["error"]) == ("ok", None)
    assert row["platform_peak_bytes"] == 9 * GIB and row["estimate"] == _ESTIMATE
    assert row["pixels"] == 1024 * 1024 and row["model"] == "zit" and "rss_anon_peak_bytes" in row and "prompt" not in row


def test_run_case_records_the_kernels_resident_high_water_mark(tmp_path: Path, measured: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bench.memory.measure.read_status", lambda: "Name:\tpython\nVmHWM:\t 3145728 kB\nVmRSS:\t 100 kB\n")

    def fake_cli(_argv: list[str]) -> None:
        (tmp_path / "picture.png").write_bytes(b"x")

    assert _run([], fake_cli, tmp_path)["vm_hwm_bytes"] == 3 * GIB


def test_run_case_leaves_the_high_water_mark_empty_where_the_system_has_none(tmp_path: Path, measured: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bench.memory.measure.read_status", lambda: "")

    def fake_cli(_argv: list[str]) -> None:
        (tmp_path / "picture.png").write_bytes(b"x")

    assert _run([], fake_cli, tmp_path)["vm_hwm_bytes"] is None


@pytest.mark.parametrize(
    ("behaviour", "message"),
    [
        (lambda: (_ for _ in ()).throw(SystemExit(2)), "exited with 2"),
        (lambda: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")), "CUDA out of memory"),
        (lambda: None, "no output file"),
    ],
)
def test_run_case_records_a_failure(tmp_path: Path, measured: None, behaviour, message: str) -> None:
    row = _run([], lambda _argv: behaviour(), tmp_path)
    assert row["status"] == "failed" and message in row["error"]


def test_run_case_treats_a_zero_exit_as_success(tmp_path: Path, measured: None) -> None:
    def fake_cli(_argv: list[str]) -> None:
        (tmp_path / "picture.png").write_bytes(b"x")
        raise SystemExit(0)

    assert _run([], fake_cli, tmp_path)["status"] == "ok"


def test_a_process_that_dies_without_a_row_is_a_failed_row(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: subprocess.CompletedProcess([], -9))
    row = bench_main._spawn(_case(), tmp_path, timeout=5)
    assert row["status"] == "failed" and "-9" in row["error"] and row["id"] == "c"
    assert row["os"] == sys.platform  # a row that names its system is never read as another one's


def test_meta_records_no_host_user_or_device_serial(tmp_path: Path) -> None:
    bench_main._write_meta(tmp_path, argparse.Namespace(label="cuda", family=None, func=lambda: None))
    text = (tmp_path / "meta.json").read_text(encoding="utf-8")
    assert platform.node() not in text and getpass.getuser() not in text
    assert set(json.loads(text)) == {"commit", "platform", "system", "machine", "python", "versions", "args"}


def test_a_dry_run_writes_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(bench_main, "RUNS_DIR", tmp_path)
    assert bench_main.main(["run", "--dry-run", "--family", "zimage", "--label", "dry"]) == 0
    assert "zit-xl-up2: 1600x1600" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("media", MEDIA)
def test_every_medium_has_a_case_file_whose_cases_a_dry_run_lists(media: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(bench_main, "RUNS_DIR", tmp_path)
    cases = expand_cases(load_cases(media), media=media, platform="cuda")
    assert cases and all(case["media"] == media for case in cases) and len({case["id"] for case in cases}) == len(cases)

    assert bench_main.main(["run", "--media", media, "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert all(f"{case['id']}: " in printed and f"ziv-{media}" in printed for case in cases)
    assert list(tmp_path.iterdir()) == []


def test_the_video_and_audio_cli_arguments_are_accepted_by_the_real_parsers(tmp_path: Path) -> None:
    from zvisiongenerator.audio_cli import _build_audio_parser
    from zvisiongenerator.video_cli import _build_video_parser

    from zvisiongenerator.utils.config import load_config

    cases = load_cases("video")
    for case in expand_cases(cases, media="video", platform="cuda"):
        video.prepare(case, cases, load_config())
        args = _build_video_parser().parse_args(video.case_argv(case, tmp_path))
        assert args.model == case["model"] and args.no_enhance and args.low_memory == case.get("low_memory", True)
        assert args.frames == (None if case.get("extend_source") else case["frames"])
    for case in expand_cases(load_cases("audio"), media="audio", platform="cuda"):
        args = _build_audio_parser().parse_args(audio.case_argv(case, tmp_path))
        assert args.model == case["model"] and args.kind == case["kind"] and args.prompt == " ".join([case["prompt"]] * case.get("prompt_repeat", 1))


def test_the_case_file_has_the_cases_the_overnight_run_names() -> None:
    cases = {case["id"]: case for case in expand_cases(load_cases("image"), media="image", platform="cuda")}
    assert {"zit-q4-m", "zit-q4-xl", "klein4b-q4-m", "ideo-q4-xl", "klein4b-q4-m-ref1", "zit-q4-m-resident"} <= set(cases)
    assert cases["zit-q4-m-resident"]["placement"] == "resident" and cases["zit-q4-m-resident"]["quantize"] == 4
    assert {case["id"] for case in expand_cases(load_cases("image"), media="image", platform="mac")}.isdisjoint({"zit-q4-m-resident"})
    assert cases["klein4b-q4-m-ref1"]["references"] == 1


def test_a_budget_or_a_hold_makes_a_result_its_own() -> None:
    plain = {"id": "zit-q4-m"}
    assert bench_main._key(plain) != bench_main._key(plain | {"hold_gib": 1.5}) != bench_main._key(plain | {"budget_gib": 6.0})


def test_run_budget_gib_is_recorded_on_every_case(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spawned: list[dict] = []
    monkeypatch.setattr(bench_main, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(bench_main, "_spawn", lambda case, run_dir, timeout: spawned.append(case) or {"id": case["id"], "status": "ok", "seconds": 1})
    assert bench_main.main(["run", "--media", "image", "--case", "zit-q4-m", "--budget-gib", "6", "--label", "t"]) == 0
    assert [(case["id"], case["budget_gib"]) for case in spawned] == [("zit-q4-m", 6.0)]
    assert "zit-q4-m" in (tmp_path / "image" / "t" / "results.jsonl").read_text(encoding="utf-8")


def test_a_video_lora_case_runs_with_the_file_it_is_given_and_records_only_its_size(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    lora = tmp_path / "adapter.safetensors"
    lora.write_bytes(b"x" * 1234)
    spawned: list[dict] = []
    monkeypatch.setattr(bench_main, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(bench_main, "_spawn", lambda case, run_dir, timeout: spawned.append(case) or {k: v for k, v in case.items() if k != "prompt"} | {"status": "ok", "seconds": 1})

    assert bench_main.main(["run", "--media", "video", "--case", "ltx8-s-lora", "--lora", str(lora), "--label", "t"]) == 0

    assert spawned[0]["lora_path"] == str(lora) and spawned[0]["lora_bytes"] == 1234
    assert video.case_argv(spawned[0], tmp_path)[-2:] == ["--lora", str(lora)]
    recorded = (tmp_path / "runs" / "video" / "t" / "results.jsonl").read_text(encoding="utf-8")
    assert str(lora) not in recorded and json.loads(recorded)["lora_bytes"] == 1234


def test_a_video_lora_case_reads_its_file_from_the_environment_and_is_skipped_without_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    lora = tmp_path / "adapter.safetensors"
    lora.write_bytes(b"x" * 10)
    spawned: list[dict] = []
    monkeypatch.setattr(bench_main, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(bench_main, "_spawn", lambda case, run_dir, timeout: spawned.append(case) or {"id": case["id"], "status": "ok", "seconds": 1})

    monkeypatch.delenv(bench_main.LORA_ENV, raising=False)
    assert bench_main.main(["run", "--media", "video", "--case", "ltx8-s-lora", "--label", "t"]) == 0
    assert spawned == [] and "needs a LoRA file" in capsys.readouterr().out

    monkeypatch.setenv(bench_main.LORA_ENV, str(lora))
    assert bench_main.main(["run", "--media", "video", "--case", "ltx8-s-lora", "--label", "t"]) == 0
    assert [case["lora_bytes"] for case in spawned] == [10]


def test_a_missing_lora_file_is_an_error_not_a_silent_skip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bench_main, "RUNS_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="LoRA file"):
        bench_main.main(["run", "--media", "video", "--case", "ltx8-s-lora", "--lora", str(tmp_path / "gone.safetensors"), "--label", "t"])


def test_an_image_lora_case_runs_with_the_file_it_is_given_and_records_only_its_size(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    lora = tmp_path / "adapter.safetensors"
    lora.write_bytes(b"x" * 1234)
    spawned: list[dict] = []
    monkeypatch.setattr(bench_main, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(bench_main, "_spawn", lambda case, run_dir, timeout: spawned.append(case) or {k: v for k, v in case.items() if k != "prompt"} | {"status": "ok", "seconds": 1})

    assert bench_main.main(["run", "--media", "image", "--case", "klein9b-q4-m-lora", "--case", "klein9b-q4-m", "--lora", str(lora), "--label", "t"]) == 0

    with_lora, plain = {case["id"]: case for case in spawned}["klein9b-q4-m-lora"], {case["id"]: case for case in spawned}["klein9b-q4-m"]
    assert with_lora["lora_bytes"] == 1234 and "lora_path" not in plain
    argv = case_argv(with_lora, tmp_path)
    assert argv[argv.index("--lora") + 1] == str(lora)
    assert "--lora" not in case_argv(plain, tmp_path)
    assert _build_parser().parse_args(argv).lora == str(lora)
    recorded = (tmp_path / "runs" / "image" / "t" / "results.jsonl").read_text(encoding="utf-8")
    assert str(lora) not in recorded
    assert {row["id"]: row.get("lora_bytes") for row in map(json.loads, recorded.splitlines())} == {"klein9b-q4-m": None, "klein9b-q4-m-lora": 1234}


def test_an_image_lora_case_is_skipped_without_a_file_and_every_one_has_a_plain_twin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    spawned: list[dict] = []
    monkeypatch.setattr(bench_main, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(bench_main, "_spawn", lambda case, run_dir, timeout: spawned.append(case) or {"id": case["id"], "status": "ok", "seconds": 1})
    monkeypatch.delenv(bench_main.LORA_ENV, raising=False)

    assert bench_main.main(["run", "--media", "image", "--case", "klein9b-m-lora", "--label", "t"]) == 0

    assert spawned == [] and "needs a LoRA file" in capsys.readouterr().out
    cases = {case["id"]: case for case in expand_cases(load_cases("image"), media="image", platform="cuda")}
    for case_id, case in cases.items():
        if case.get("lora"):
            twin = cases[case_id.rsplit("-", 1)[0]]
            assert (twin["model"], twin["quantize"], twin["size"], twin.get("placement")) == (case["model"], case["quantize"], case["size"], case.get("placement")) and not twin.get("lora")


def test_a_dry_run_lists_a_lora_case_with_a_placeholder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(bench_main, "RUNS_DIR", tmp_path)
    monkeypatch.delenv(bench_main.LORA_ENV, raising=False)

    assert bench_main.main(["run", "--media", "video", "--case", "ltx8-s-lora", "--dry-run"]) == 0

    assert "--lora <lora file>" in capsys.readouterr().out


def test_the_video_estimate_counts_the_loras_size(monkeypatch: pytest.MonkeyPatch) -> None:
    from zvisiongenerator.utils.memory_verdict import MemoryJob, MemoryVerdict
    from zvisiongenerator.utils.model_memory import MemoryBudget

    judged: list[MemoryJob] = []
    monkeypatch.setattr("zvisiongenerator.backends.accelerator_memory_budget_or_none", lambda: MemoryBudget(24 * GIB, 64 * GIB))
    monkeypatch.setattr("zvisiongenerator.utils.memory_verdict.judge", lambda job, machine, weights: judged.append(job) or MemoryVerdict(1, 2, "streamed", "fits", "gpu"))

    video.estimate({"family": "h3", "model": "h3", "width": 768, "height": 448, "frames": 22, "lora_bytes": 777})

    assert [job.lora_bytes for job in judged] == [777]


def test_a_forced_placement_the_loader_never_chose_is_a_failed_row(tmp_path: Path, measured: None) -> None:
    def fake_cli(_argv: list[str]) -> None:
        (tmp_path / "picture.png").write_bytes(b"x")

    case = _case() | {"placement": "resident"}
    row = _run([], fake_cli, tmp_path, case, probes=lambda: {"loaded_placement": None})
    assert row["status"] == "failed" and "resident" in row["error"]
    assert _run([], fake_cli, tmp_path, case, probes=lambda: {"loaded_placement": "resident", "loader_free_bytes": 9 * GIB})["status"] == "ok"


def test_watch_placement_forces_a_placement_and_records_what_the_loader_took(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    from zvisiongenerator.backends.cuda.image import backend
    from zvisiongenerator.backends.cuda.image.placement import PlacementChoice

    monkeypatch.setattr(backend, "choose_placement", lambda *a, **k: PlacementChoice("turns", "stepped down"))
    report = watch_placement(None)
    assert backend.choose_placement(free_bytes=7 * GIB).placement == "turns"
    assert report() == {"loaded_placement": "turns", "placement_warning": "stepped down", "forced_placement": None, "loader_free_bytes": 7 * GIB}

    forced = watch_placement("resident")
    assert backend.choose_placement(free_bytes=7 * GIB) == PlacementChoice("resident")
    assert forced()["forced_placement"] == "resident" and forced()["loader_free_bytes"] == 7 * GIB
    with pytest.raises(ValueError, match="Unknown placement"):
        watch_placement("elsewhere")


@pytest.fixture
def mlx_decode(monkeypatch: pytest.MonkeyPatch):
    """Stub ``mlx.core`` and mflux's ``VAEUtil`` (neither is installed off a Mac); return the stub MLX, the call log and the stub class."""
    import sys
    import types

    log: list[str] = []
    mx = types.ModuleType("mlx.core")
    mx.cache = 3 * GIB
    mx.get_cache_memory = lambda: mx.cache

    class VAEUtil:
        @staticmethod
        def decode(vae: object, latent: object, tiling_config: object = None) -> str:
            log.append(f"decode {vae} {latent} {tiling_config}")
            return "decoded"

    modules = {"mlx": types.ModuleType("mlx"), "mlx.core": mx}
    for name in ("mflux", "mflux.models", "mflux.models.common", "mflux.models.common.vae"):
        modules[name] = types.ModuleType(name)
    modules["mflux.models.common.vae.vae_util"] = types.SimpleNamespace(VAEUtil=VAEUtil)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return mx, log, VAEUtil


def test_a_mac_case_traces_the_job_and_a_cuda_case_watches_the_loader(monkeypatch: pytest.MonkeyPatch, mlx_job, mlx_decode) -> None:
    from bench.memory import image

    monkeypatch.setattr(image, "platform_label", lambda: "mac")
    assert set(image.install_probes({"id": "a"})()) == {"trace", "step_seconds"}

    monkeypatch.setattr(image, "platform_label", lambda: "cuda")
    monkeypatch.setattr(image, "watch_placement", lambda forced: lambda: {"forced_placement": forced})
    assert image.install_probes({"id": "c", "placement": "turns"})() == {"forced_placement": "turns"}


@pytest.fixture
def mlx_job(monkeypatch: pytest.MonkeyPatch, mlx_decode):
    """Stub the Mac image backend's ``_reporting`` and its MLX counters on top of ``mlx_decode``; return the model, the call log and a function that reads a counter set."""
    import contextlib
    import sys
    import types

    mx, log, _vae_util = mlx_decode
    mx.active = 5 * GIB
    mx.get_active_memory = lambda: mx.active
    mx.array = type("array", (), {})
    mx.eval = lambda value: log.append(f"eval {value}")
    monkeypatch.setattr("bench.memory.measure.current_footprint", lambda: 7 * GIB)

    class Registry:
        def __init__(self) -> None:
            self.callbacks: list[object] = []

        def register(self, callback: object) -> None:
            self.callbacks.append(callback)

    model = SimpleNamespace(callbacks=Registry())
    image = types.ModuleType("zvisiongenerator.backends.mac.image")

    @contextlib.contextmanager
    def reporting(model: object, *args: object, **kwargs: object):
        log.append("reporting start")
        yield
        log.append("reporting end")

    image._reporting = reporting
    image._unregister_callback = lambda model, callback: model.callbacks.callbacks.remove(callback)
    monkeypatch.setitem(sys.modules, "zvisiongenerator.backends.mac.image", image)
    monkeypatch.setattr(sys.modules["zvisiongenerator.backends.mac"], "image", image, raising=False)
    return model, image, log


def test_trace_mlx_job_samples_every_step_and_around_the_decode(mlx_job, mlx_decode) -> None:
    model, image, log = mlx_job
    mx, _log, vae_util = mlx_decode
    report = probes.trace_mlx_job()

    with image._reporting(model, 2, "zimage", None, None):
        (tracer,) = model.callbacks.callbacks  # the trace is registered for the generation and nothing else
        tracer.call_before_loop(1, "p", "latents", object())
        mx.active, mx.cache = 6 * GIB, 2 * GIB
        tracer.call_in_loop(0, 1, "p", "latents-1", object(), [])
        mx.cache = 4 * GIB
        tracer.call_in_loop(1, 1, "p", "latents-2", object(), [])
        vae_util.decode("v", "z")
        vae_util.decode("v", "z")  # a second decode is not recorded
    assert model.callbacks.callbacks == []  # removed again, as the app removes its own

    result = report()
    assert result["trace"] == [
        [0, 7 * GIB, 5 * GIB, 3 * GIB],
        [1, 7 * GIB, 6 * GIB, 2 * GIB],
        [2, 7 * GIB, 6 * GIB, 4 * GIB],
        ["before_decode", 7 * GIB, 6 * GIB, 4 * GIB],
        ["after_decode", 7 * GIB, 6 * GIB, 4 * GIB],
    ]
    assert len(result["step_seconds"]) == 2 and all(seconds >= 0 for seconds in result["step_seconds"])
    assert "eval latents-1" in log and log[0] == "reporting start" and "reporting end" in log  # the app's own reporting still runs, around the trace


def test_trace_mlx_job_evaluates_a_lazy_decode_before_the_reading_after_it(mlx_job, mlx_decode) -> None:
    _model, _image, log = mlx_job
    mx, _log, vae_util = mlx_decode
    lazy = mx.array()
    vae_util.decode = staticmethod(lambda *args, **kwargs: lazy)
    report = probes.trace_mlx_job()

    assert vae_util.decode("v", "z") is lazy
    assert [row[0] for row in report()["trace"]] == ["before_decode", "after_decode"] and any(entry.startswith("eval ") for entry in log)


def test_the_fit_leaves_a_probe_row_of_a_removed_experiment_out_of_the_rule_and_the_size_terms() -> None:
    for flag in ({"empty_cache": True}, {"clear_mlx_cache_before_decode": True}, {"mlx_cache_limit_gib": 0.0}):
        row = _ran("zit-xl", status="fits", estimate_gib=1.0, measured_gib=8.0, **flag)
        checks, left_out = rule_checks([row])
        assert checks == [] and [item["id"] for item in left_out] == ["zit-xl"]
        plain_row = _row("zit-xl", 1_000_000, 10.0, model="zit", family="zimage")
        probed_row = _row("zit-xl", 2_560_000, 20.0, model="zit", family="zimage", **flag)
        assert size_terms([plain_row, probed_row]) == size_terms([plain_row])


# --- the calibration rule ------------------------------------------------------------------------------------------------


def _ran(case_id: str, *, status: str, estimate_gib: float, measured_gib: float, ram_estimate: float | None = None, ram: float | None = None, **extra: object) -> dict:
    row = {
        "id": case_id,
        "media": "image",
        "platform": "cuda",
        "os": "linux",
        "status": "ok",
        "estimate": {"status": status, "gpu_bytes": int(estimate_gib * GIB), "system_bytes": None if ram_estimate is None else int(ram_estimate * GIB)},
        "platform_allocated_peak_bytes": int(measured_gib * GIB),
        "platform_peak_bytes": int((measured_gib + 1) * GIB),
        "rss_anon_peak_bytes": None if ram is None else int(ram * GIB),
        "rss_anon_shmem_peak_bytes": None if ram is None else int(ram * GIB),
    }
    return row | extra


def _failures(rows: list[dict]) -> list[str]:
    checks, _ = rule_checks(rows)
    return [f"{check['id']}: {check['check']}" for check in checks if not check["ok"]]


def test_the_rule_passes_a_job_that_ran_within_the_tolerance_and_the_ram_estimate() -> None:
    assert _failures([_ran("a", status="tight", estimate_gib=7.9, measured_gib=7.98, ram_estimate=2.2, ram=2.1)]) == []


def test_the_rule_fails_an_estimate_further_than_the_tolerance_from_the_peak_on_either_side() -> None:
    assert tolerance_bytes(5 * GIB) == GIB and tolerance_bytes(20 * GIB) == 2 * GIB  # the larger of 1 GiB and 10 %
    assert _failures([_ran("a", status="fits", estimate_gib=5.0, measured_gib=7.5)]) == ["a: GPU estimate matches the peak"]  # under: the peak is 7.5 + the offset
    assert _failures([_ran("a", status="fits", estimate_gib=9.5, measured_gib=7.5)]) == ["a: GPU estimate matches the peak"]  # over
    assert _failures([_ran("a", status="fits", estimate_gib=7.8, measured_gib=7.5)]) == []
    assert _failures([_ran("a", status="fits", estimate_gib=7.0, measured_gib=7.1, ram_estimate=2.0, ram=4.0)]) == ["a: RAM estimate matches the peak"]
    assert _failures([_ran("a", status="fits", estimate_gib=7.0, measured_gib=7.1, ram_estimate=6.0, ram=4.0)]) == ["a: RAM estimate matches the peak"]


def test_the_rule_fails_a_job_that_ran_but_reads_too_large() -> None:
    assert _failures([_ran("a", status="too_large", estimate_gib=10.2, measured_gib=9.9)]) == ["a: ran reads below Too large"]


def test_a_mac_job_that_ran_beyond_the_whole_memory_may_read_too_large() -> None:
    row = _ran("m", status="too_large", estimate_gib=33.0, measured_gib=0.0, platform="mac", footprint_peak_bytes=33 * GIB, platform_peak_bytes=30 * GIB)
    assert _failures([row | {"estimate": row["estimate"] | {"unified_bytes": 32 * GIB}}]) == []
    assert _failures([row | {"estimate": row["estimate"] | {"unified_bytes": 64 * GIB}}]) == ["m: ran reads below Too large"]


def test_the_rule_wants_an_out_of_memory_job_to_read_too_large_against_the_card_less_what_others_held() -> None:
    oom = {"id": "b", "media": "image", "platform": "cuda", "status": "failed", "error": "OutOfMemoryError: CUDA out of memory"}
    assert _failures([oom | {"estimate": {"status": "too_large", "gpu_bytes": 11 * GIB}}]) == []
    assert _failures([oom | {"estimate": {"status": "tight", "gpu_bytes": 9.5 * GIB, "budget_bytes": 10 * GIB}}]) == ["b: out of memory reads Too large"]
    # Another program held 1 GiB of the card: the job had 9 GiB, and 9.5 of it is too large; a program that held nothing leaves it tight.
    assert _failures([oom | {"gpu_baseline_used_bytes": GIB, "estimate": {"status": "tight", "gpu_bytes": 9.5 * GIB, "budget_bytes": 10 * GIB}}]) == []
    assert _failures([oom | {"gpu_baseline_used_bytes": 0, "estimate": {"status": "tight", "gpu_bytes": 9.5 * GIB, "budget_bytes": 10 * GIB}}]) == ["b: out of memory reads Too large"]
    # A hold is another program too, and a run under a card of its own size judges a card the job did not have.
    assert _failures([oom | {"hold_gib": 1.5, "estimate": {"status": "tight", "gpu_bytes": 9 * GIB}}]) == []


def test_on_windows_an_out_of_memory_job_must_read_too_large_as_on_linux() -> None:
    oom = {
        "id": "w",
        "media": "image",
        "platform": "cuda",
        "status": "failed",
        "error": "OutOfMemoryError: CUDA out of memory",
        "estimate": {"status": "tight", "gpu_bytes": 9 * GIB, "budget_bytes": 10 * GIB},
    }

    for system in ("win32", "linux"):
        assert _failures([oom | {"os": system}]) == ["w: out of memory reads Too large"]
        assert _failures([oom | {"os": system, "estimate": {"status": "fits", "gpu_bytes": 5 * GIB, "budget_bytes": 10 * GIB}}]) == ["w: out of memory reads Too large"]
        # Others' memory is subtracted on both: 1.5 GiB held by other programs leaves 8.5 GiB, and 9 GiB of it reads Too large.
        assert _failures([oom | {"os": system, "gpu_baseline_used_bytes": int(1.5 * GIB)}]) == []


def test_a_windows_row_is_measured_at_its_peak_plus_the_windows_offset_and_a_linux_row_at_the_linux_one() -> None:
    row = _ran("o", status="fits", estimate_gib=7.0, measured_gib=6.0)
    assert measured_gpu_bytes(row | {"os": "win32"}) == 6 * GIB + CUDA_OFFSET_WINDOWS_BYTES
    assert measured_gpu_bytes(row | {"os": "linux"}) == 6 * GIB + CUDA_OFFSET_LINUX_BYTES
    assert measured_gpu_bytes(row) == 6 * GIB + CUDA_OFFSET_LINUX_BYTES  # a row without os is a Linux row
    assert (CUDA_OFFSET_LINUX_BYTES, CUDA_OFFSET_WINDOWS_BYTES) == (round(0.40 * GIB), GIB)


def test_the_probe_lines_name_the_offset_of_each_probed_cases_system() -> None:
    rows = [row | {"os": system} for system in ("linux", "win32") for row in (_held("p", 0.0, outcome="ran", card_gib=9.0), _held("p", 1.5, outcome="failed", card_gib=7.5))]
    text = format_probes(hold_brackets(rows))
    assert "(CUDA offset on linux 0.40)" in text and "(CUDA offset on win32 1.00)" in text


def test_the_card_the_job_had_counts_from_the_budget_the_run_graded_against_where_that_is_smaller() -> None:
    row = {"gpu_driver_total_bytes": 10 * GIB, "gpu_baseline_used_bytes": GIB // 2, "hold_charged_bytes": GIB // 4}
    assert card_the_job_had(row) == int(9.25 * GIB)
    assert card_the_job_had(row | {"run_budget_bytes": 9 * GIB}) == int(8.25 * GIB)
    assert card_the_job_had(row | {"run_budget_bytes": 12 * GIB}) == int(9.25 * GIB)  # never above the driver's total
    assert card_the_job_had(row | {"run_budget_bytes": None}) == int(9.25 * GIB)


def test_fit_saves_the_budget_the_run_graded_against_before_the_estimates_are_replaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    stored = {"status": "fits", "gpu_bytes": 8 * GIB, "budget_bytes": 9 * GIB}
    rows = [
        _held("probed", 0.0, outcome="ran", card_gib=9.0) | {"os": "linux", "estimate": stored},
        _held("probed", 1.5, outcome="failed", card_gib=7.5) | {"os": "linux", "estimate": stored},
    ]
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    # The fresh estimate this machine gives has a larger budget; the cards are still counted from the run's.
    monkeypatch.setattr(bench_main, "_estimate_for", lambda row: stored | {"budget_bytes": 10 * GIB})
    bench_main.main(["fit", str(path)])
    assert "out of memory on 6.50, ran on 8.00" in capsys.readouterr().out


def test_fit_asks_the_engine_only_for_rows_of_this_operating_system_and_says_so_once_for_the_others(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    stored = {"status": "fits", "gpu_bytes": 8 * GIB, "budget_bytes": 10 * GIB}
    rows = [
        _ran("here", status="fits", estimate_gib=8.0, measured_gib=6.6) | {"os": "linux", "estimate": stored},
        _ran("there", status="fits", estimate_gib=8.0, measured_gib=6.6) | {"os": "win32", "estimate": stored},
    ]
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    asked: list[str] = []
    monkeypatch.setattr(bench_main, "_estimate_for", lambda row: asked.append(row["id"]) or stored)
    bench_main.main(["fit", str(path)])
    captured = capsys.readouterr()
    assert asked == ["here"]
    assert captured.err.count("judged on their stored estimates") == 1 and "1 rows measured on another operating system (win32)" in captured.err


def test_the_card_of_a_row_run_under_a_simulated_budget_counts_from_the_driver_total() -> None:
    row = {"gpu_driver_total_bytes": 10 * GIB, "gpu_baseline_used_bytes": GIB // 2, "run_budget_bytes": 6 * GIB}
    assert card_the_job_had(row) == int(5.5 * GIB)
    assert card_the_job_had(row | {"budget_gib": 6}) == int(9.5 * GIB)


def test_a_case_whose_outcomes_overlap_is_reported_with_both_cards_and_gets_no_bracket(capsys: pytest.CaptureFixture[str]) -> None:
    rows = [_held("overlap", 0.0, outcome="ran", card_gib=7.0), _held("overlap", 1.5, outcome="failed", card_gib=8.0)]
    assert [row.get("bracket_bytes") for row in with_brackets(rows)] == [None, None]
    output = capsys.readouterr().out
    assert output.count("NO BRACKET overlap (linux)") == 1 and "out of memory on 8.00 GiB but ran on 7.00 GiB" in output
    with_brackets(rows, report=False)
    assert capsys.readouterr().out == ""


def test_the_measured_ram_on_windows_is_the_private_working_set_peak() -> None:
    """Pinned memory raises the private working set and GPU shared usage both, so the shared usage is not added."""
    row = {"os": "win32", "private_working_set_peak_bytes": 6 * GIB, "gpu_shared_peak_bytes": 3 * GIB, "private_commit_peak_bytes": 20 * GIB}

    assert measured_ram_bytes(row) == 6 * GIB
    assert measured_ram_bytes(row | {"private_working_set_peak_bytes": None}) is None


def test_a_windows_row_is_judged_on_its_ram_estimate() -> None:
    row = _ran("w", status="fits", estimate_gib=8.0, measured_gib=7.1, ram_estimate=8.0, os="win32", private_working_set_peak_bytes=8 * GIB, gpu_shared_peak_bytes=4 * GIB)

    assert _failures([row]) == []
    assert _failures([row | {"private_working_set_peak_bytes": 6 * GIB}]) == ["w: RAM estimate matches the peak"]


def test_the_rule_leaves_out_setup_failures_and_rows_without_an_estimate() -> None:
    rows = [
        {"id": "setup", "status": "failed", "error": "the CLI exited with 2", "estimate": {"status": "fits", "gpu_bytes": GIB}},
        _ran("blind", status="fits", estimate_gib=1.0, measured_gib=8.0) | {"estimate": None},
        _ran("reserved-only", status="fits", estimate_gib=1.0, measured_gib=8.0) | {"platform_allocated_peak_bytes": None},
    ]
    checks, left_out = rule_checks(rows)
    assert checks == []
    assert {item["id"]: item["reason"] for item in left_out} == {
        "setup": "a setup failure",
        "blind": "no estimate",
        "reserved-only": "no peak to compare",
    }


def test_video_is_compared_with_the_allocated_peak_plus_the_cuda_offset() -> None:
    row = _ran("v", status="fits", estimate_gib=6.9, measured_gib=6.0, media="video")
    assert row["platform_peak_bytes"] == 7 * GIB  # the reserved peak is not what the estimate follows
    assert measured_gpu_bytes(row) == 6 * GIB + CUDA_OFFSET_BYTES
    assert _failures([row]) == []
    assert _failures([row | {"estimate": row["estimate"] | {"gpu_bytes": 5 * GIB}}]) == ["v: GPU estimate matches the peak"]


def test_audio_on_cuda_is_compared_with_the_allocated_peak_plus_the_cuda_offset() -> None:
    row = _ran("a", status="fits", estimate_gib=6.5, measured_gib=6.0, media="audio")
    assert _failures([row]) == []
    assert _failures([row | {"estimate": row["estimate"] | {"gpu_bytes": 5 * GIB}}]) == ["a: GPU estimate matches the peak"]
    assert measured_gpu_bytes(row) == 6 * GIB + CUDA_OFFSET_BYTES


def test_a_mac_is_compared_with_its_peak_footprint_and_a_row_without_one_is_left_out() -> None:
    with_footprint = _ran("m", status="fits", estimate_gib=12.0, measured_gib=0.0, platform="mac", platform_peak_bytes=9 * GIB, footprint_peak_bytes=12 * GIB)
    without = _ran("n", status="fits", estimate_gib=9.0, measured_gib=0.0, platform="mac", platform_peak_bytes=9 * GIB)
    assert (measured_gpu_bytes(with_footprint), measured_gpu_bytes(without)) == (12 * GIB, None)
    assert measured_gpu_bytes(with_footprint | {"platform_peak_bytes": 13 * GIB}) == 12 * GIB  # MLX's peak above the footprint is memory the OS does not charge
    assert _failures([with_footprint, without]) == []
    assert _failures([with_footprint | {"estimate": with_footprint["estimate"] | {"gpu_bytes": 9 * GIB}}]) == ["m: GPU estimate matches the peak"]
    checks, left_out = rule_checks([with_footprint, without])
    assert [(item["id"], item["reason"]) for item in left_out] == [("n", "no peak to compare")]
    assert "*" not in format_rule([with_footprint, without], checks, left_out).split("GiB. est")[0]


def test_reestimate_replaces_the_estimate_and_keeps_the_stored_one_when_none_can_be_made() -> None:
    rows = [{"id": "a", "estimate": {"status": "fits"}}, {"id": "b", "estimate": {"status": "fits"}}]
    fresh = reestimate(rows, lambda row: {"status": "tight"} if row["id"] == "a" else None)
    assert [row["estimate"]["status"] for row in fresh] == ["tight", "fits"]
    assert reestimate(rows, lambda _row: (_ for _ in ()).throw(RuntimeError("gone")))[0]["estimate"] == {"status": "fits"}


def test_fit_exits_non_zero_when_the_rule_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    results = tmp_path / "results.jsonl"
    good = _ran("good", status="tight", estimate_gib=7.98, measured_gib=7.98)
    bad = _ran("bad", status="fits", estimate_gib=5.0, measured_gib=7.0)
    results.write_text(json.dumps(good) + "\n", encoding="utf-8")
    assert bench_main.main(["fit", "--stored", str(results)]) == 0
    results.write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n", encoding="utf-8")
    assert bench_main.main(["fit", "--stored", str(results)]) == 1
    out = capsys.readouterr().out
    assert "FAIL bad: GPU estimate matches the peak" in out and "The calibration rule fails" in out


def test_fit_judges_only_the_last_row_of_a_case_within_a_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    results = tmp_path / "results.jsonl"
    old = _ran("rerun", status="fits", estimate_gib=5.0, measured_gib=7.0)
    new = _ran("rerun", status="tight", estimate_gib=7.0, measured_gib=7.0)
    held = _ran("rerun", status="tight", estimate_gib=5.0, measured_gib=7.0, hold_gib=1.0)
    results.write_text("".join(json.dumps(row) + "\n" for row in (old, new)), encoding="utf-8")
    assert bench_main.main(["fit", "--stored", str(results)]) == 0
    assert "FAIL" not in capsys.readouterr().out
    # A case under another hold is its own result, so it is kept beside the last plain row.
    results.write_text("".join(json.dumps(row) + "\n" for row in (new, held)), encoding="utf-8")
    assert bench_main.main(["fit", "--stored", str(results)]) == 1
    assert bench_main._latest_rows([old, new, held]) == [new, held]


def _held(case_id: str, hold: float, *, outcome: str, card_gib: float, peak_gib: float = 7.0, **extra: object) -> dict:
    """A row of a case run under a hold: the driver's total is 10 GiB and others (the hold and its context) held what is left of *card_gib*."""
    row = _ran(case_id, status="fits", estimate_gib=7.4, measured_gib=peak_gib, hold_gib=hold, gpu_driver_total_bytes=10 * GIB, gpu_baseline_used_bytes=int((10 - card_gib) * GIB), **extra)
    return row if outcome == "ran" else row | {"status": "failed", "error": "OutOfMemoryError: CUDA out of memory"}


def test_a_probed_case_is_measured_at_the_middle_of_the_bracket_between_the_card_it_failed_on_and_the_card_it_ran_on() -> None:
    rows = [
        _held("probed", 0.0, outcome="ran", card_gib=9.0),
        _held("probed", 1.0, outcome="ran", card_gib=8.0),
        _held("probed", 1.5, outcome="failed", card_gib=7.5),
        _held("probed", 2.0, outcome="failed", card_gib=7.0),
        _held("unprobed", 0.0, outcome="ran", card_gib=9.0),
    ]
    rows = with_brackets(rows)
    assert {row["id"]: row.get("bracket_bytes") for row in rows} == {"probed": (int(7.5 * GIB), 8 * GIB), "unprobed": None}
    assert [measured_gpu_bytes(row) for row in rows] == [int(7.75 * GIB)] * 4 + [7 * GIB + CUDA_OFFSET_BYTES]
    # The probed case is judged against its bracket and not against its free-card peak; the case without probes against its peak plus the offset.
    assert _failures(rows) == []
    assert _failures([row | {"estimate": row["estimate"] | {"gpu_bytes": 9 * GIB}} for row in rows if row["id"] == "probed"]) == ["probed: GPU estimate matches the peak"] * 2


def _twin_rows(**plain: object) -> list[dict]:
    """A probed case (bracket middle 7.75 GiB, free-card peak 7.0 GiB) and a plain row of the case *plain* describes, which defaults to the identical job."""
    job = {"model": "klein4b", "quantize": 4, "width": 1024, "height": 1024, "references": 0, "loaded_placement": "resident"}
    probed = [
        _held("probed-resident", 0.0, outcome="ran", card_gib=9.0, **job),
        _held("probed-resident", 1.0, outcome="ran", card_gib=8.0, **job),
        _held("probed-resident", 1.5, outcome="failed", card_gib=7.5, **job),
    ]
    return [*probed, _ran("plain", status="fits", estimate_gib=7.4, measured_gib=7.0, **(job | plain))]


def test_a_plain_row_of_the_identical_job_is_judged_by_the_probed_cases_bracket() -> None:
    rows = with_brackets(_twin_rows())
    plain = rows[-1]
    assert plain["twin_case"] == "probed-resident" and measured_gpu_bytes(plain) == int(7.75 * GIB)
    # Its free-card peak plus the offset is 7.0 + CUDA_OFFSET; the estimate is within the tolerance of the bracket's middle either way, and the check says what judged it.
    checks, _ = rule_checks(rows)
    assert [check["check"] for check in checks if check["id"] == "plain" and check["check"].startswith("GPU")] == ["GPU estimate matches the peak (judged by the probes of probed-resident)"]
    far = [row | {"estimate": row["estimate"] | {"gpu_bytes": 9 * GIB}} if row["id"] == "plain" else row for row in rows]
    assert "plain: GPU estimate matches the peak (judged by the probes of probed-resident)" in _failures(far)
    assert "twin bracket" in format_rule(rows, *rule_checks(rows))
    assert [probe["id"] for probe in hold_brackets(rows)] == ["probed-resident"]


@pytest.mark.parametrize(
    "different",
    [
        {"width": 1600, "height": 1600},
        {"quantize": 8},
        {"model": "klein9b"},
        {"references": 1},
        {"edit": "pic"},
        {"upscale": 2},
        {"loaded_placement": "streamed"},
        {"lora_bytes": 150_000_000},
        {"budget_gib": 6},
    ],
)
def test_a_plain_row_that_is_not_the_identical_job_keeps_its_own_peak_plus_the_offset(different: dict) -> None:
    plain = with_brackets(_twin_rows(**different))[-1]
    assert "twin_case" not in plain and measured_gpu_bytes(plain) == 7 * GIB + CUDA_OFFSET_BYTES


def test_a_lora_row_never_borrows_the_plain_cases_bracket_and_a_plain_row_never_borrows_the_loras() -> None:
    rows = with_brackets(_twin_rows(lora_bytes=150_000_000))
    assert "twin_case" not in rows[-1] and measured_gpu_bytes(rows[-1]) == 7 * GIB + CUDA_OFFSET_BYTES
    assert _failures(rows) == []

    job = {"model": "klein4b", "quantize": 4, "width": 1024, "height": 1024, "references": 0, "loaded_placement": "resident", "lora_bytes": 150_000_000}
    probed = [
        _held("probed-lora", 0.0, outcome="ran", card_gib=9.0, **job),
        _held("probed-lora", 1.0, outcome="ran", card_gib=8.0, **job),
        _held("probed-lora", 1.5, outcome="failed", card_gib=7.5, **job),
    ]
    plain = _ran("plain", status="fits", estimate_gib=7.4, measured_gib=7.0, **(job | {"lora_bytes": 0}))
    assert "twin_case" not in with_brackets([*probed, plain])[-1]
    assert with_brackets([*probed, plain | {"lora_bytes": 150_000_000}])[-1]["twin_case"] == "probed-lora"
    assert "twin_case" in with_brackets(_twin_rows(lora_bytes=0))[-1]  # a LoRA of no size is no LoRA


def _budgeted(case_id: str, hold: float, *, outcome: str, card_gib: float, budget_gib: float | None, **extra: object) -> dict:
    return _held(case_id, hold, outcome=outcome, card_gib=card_gib, **({"budget_gib": budget_gib} if budget_gib else {}), **extra)


def test_a_row_run_under_a_budget_neither_takes_nor_spoils_the_bracket_of_the_full_card() -> None:
    full = [
        _budgeted("probed", 0.0, outcome="ran", card_gib=9.0, budget_gib=None),
        _budgeted("probed", 1.0, outcome="ran", card_gib=8.0, budget_gib=None),
        _budgeted("probed", 1.5, outcome="failed", card_gib=7.5, budget_gib=None),
    ]
    smaller = _budgeted("probed", 0.0, outcome="ran", card_gib=9.0, budget_gib=6.0)
    rows = with_brackets([*full, smaller])
    assert [row.get("bracket_bytes") for row in rows] == [(int(7.5 * GIB), 8 * GIB)] * 3 + [None]  # the budget row has no bracket of its own and takes none
    assert measured_gpu_bytes(rows[-1]) == 7 * GIB + CUDA_OFFSET_BYTES
    assert [probe["budget_gib"] for probe in hold_brackets(rows)] == [None]

    # A budget row that ran out of memory on a card the full-card case ran on is no outcome of the full-card case.
    spoiler = _budgeted("probed", 0.5, outcome="failed", card_gib=9.0, budget_gib=6.0)
    assert [row.get("bracket_bytes") for row in with_brackets([*full, spoiler])][:3] == [(int(7.5 * GIB), 8 * GIB)] * 3


def test_the_probes_of_a_case_under_a_budget_are_its_own_bracket_and_are_named_with_the_budget() -> None:
    rows = [
        _budgeted("probed", 0.0, outcome="ran", card_gib=9.0, budget_gib=None),
        _budgeted("probed", 0.0, outcome="ran", card_gib=9.0, budget_gib=6.0),
        _budgeted("probed", 1.0, outcome="ran", card_gib=8.0, budget_gib=6.0),
        _budgeted("probed", 2.0, outcome="failed", card_gib=7.0, budget_gib=6.0),
    ]
    rows = with_brackets(rows)
    assert [row.get("bracket_bytes") for row in rows] == [None, (7 * GIB, 8 * GIB), (7 * GIB, 8 * GIB), (7 * GIB, 8 * GIB)]
    probes = hold_brackets(rows)
    assert [(probe["id"], probe["budget_gib"]) for probe in probes] == [("probed", 6.0)]
    assert "probed (budget 6)" in format_probes(probes)


@pytest.mark.parametrize("media", ["video", "audio"])
def test_only_image_rows_are_twins(media: str) -> None:
    rows = [row | {"media": media} for row in _twin_rows()]
    assert [row["twin_case"] for row in with_brackets(rows) if "twin_case" in row] == [] and measured_gpu_bytes(with_brackets(rows)[-1]) == 7 * GIB + CUDA_OFFSET_BYTES


def test_a_row_that_names_no_model_is_never_a_twin() -> None:
    rows = _twin_rows()
    rows = [{k: v for k, v in row.items() if k != "model"} for row in rows]
    assert "twin_case" not in with_brackets(rows)[-1]


def test_a_windows_row_is_not_judged_by_a_linux_bracket() -> None:
    linux = [
        _held("probed", 0.0, outcome="ran", card_gib=9.0, os="linux"),
        _held("probed", 1.0, outcome="ran", card_gib=8.0, os="linux"),
        _held("probed", 1.5, outcome="failed", card_gib=7.5, os="linux"),
    ]
    windows = [_held("probed", 0.0, outcome="ran", card_gib=9.0, os="win32")]
    rows = with_brackets([*linux, *windows])
    assert [row.get("bracket_bytes") for row in rows] == [(int(7.5 * GIB), 8 * GIB)] * 3 + [None]
    assert measured_gpu_bytes(rows[-1]) == 7 * GIB + CUDA_OFFSET_WINDOWS_BYTES
    assert [probe["os"] for probe in hold_brackets([*linux, *windows])] == ["linux"]


def test_the_outcomes_of_one_case_on_two_systems_are_not_merged_into_a_bracket() -> None:
    # Linux only ran and Windows only ran out of memory: merged they would read as a bracket, apart they have none.
    rows = [_held("probed", 0.0, outcome="ran", card_gib=9.0, os="linux"), _held("probed", 1.5, outcome="failed", card_gib=7.5, os="win32")]
    assert [row.get("bracket_bytes") for row in with_brackets(rows)] == [None, None]
    # Each system's own bracket stands, and a conflicting outcome on one system does not drop the other's.
    both = [
        _held("probed", 0.0, outcome="ran", card_gib=9.0, os="linux"),
        _held("probed", 1.5, outcome="failed", card_gib=7.5, os="linux"),
        _held("probed", 0.0, outcome="ran", card_gib=7.0, os="win32"),
        _held("probed", 1.5, outcome="failed", card_gib=8.0, os="win32"),
    ]
    assert [row.get("bracket_bytes") for row in with_brackets(both)] == [(int(7.5 * GIB), 9 * GIB), (int(7.5 * GIB), 9 * GIB), None, None]


def test_a_plain_row_is_a_twin_only_of_a_probed_case_on_its_own_system() -> None:
    rows = [row | {"os": "linux"} for row in _twin_rows()]
    plain = with_brackets([*rows[:-1], rows[-1] | {"os": "win32"}])[-1]
    assert "twin_case" not in plain and measured_gpu_bytes(plain) == 7 * GIB + CUDA_OFFSET_WINDOWS_BYTES
    assert with_brackets(rows)[-1]["twin_case"] == "probed-resident"


def test_a_case_that_only_ran_or_only_failed_has_no_bracket() -> None:
    rows = [
        _held("ran", 0.5, outcome="ran", card_gib=8.5),
        _held("failed", 0.5, outcome="failed", card_gib=8.5),
        _held("odd", 1.0, outcome="ran", card_gib=7.0),
        _held("odd", 0.5, outcome="failed", card_gib=8.0),
    ]
    assert [row.get("bracket_bytes") for row in with_brackets(rows)] == [None, None, None, None]


def test_the_hold_brackets_report_each_probed_case_with_the_offset_its_cards_leave_above_the_free_card_peak() -> None:
    rows = [
        _held("probed", 0.0, outcome="ran", card_gib=9.0),
        _held("probed", 1.0, outcome="ran", card_gib=8.0),
        _held("probed", 1.5, outcome="failed", card_gib=7.5),
        _held("plain", 0.0, outcome="ran", card_gib=9.0),
    ]
    [probe] = hold_brackets(rows)
    assert probe["id"] == "probed"
    assert (probe["low_bytes"], probe["high_bytes"]) == (int(7.5 * GIB), 8 * GIB)
    assert (round(probe["offset_low_bytes"] / GIB, 2), round(probe["offset_high_bytes"] / GIB, 2)) == (0.5, 1.0)
    assert "probed: out of memory on 7.50, ran on 8.00, figure 7.75" in format_probes(hold_brackets(rows))


# --- the size terms -----------------------------------------------------------------------------------------------------

_MP1, _XL = 1024 * 1024, 1600 * 1600


def test_size_terms_turn_a_peak_that_moved_over_1_gib_into_bytes_per_megapixel() -> None:
    rows = [_row("m", _MP1, 10.0), _row("xl", _XL, 12.0)]
    term = size_terms(rows)[("flux2_klein", "cuda", 8)]
    assert term["per_megapixel"] == round(2 * GIB / ((_XL - _MP1) / 1e6))
    assert term["per_reference"] == 0


def test_size_terms_are_zero_when_the_slope_adds_under_1_gib_at_the_largest_picture_a_job_reaches() -> None:
    def term(xl_peak: float) -> int:
        return size_terms([_row("m", _MP1, 10.0), _row("xl", _XL, xl_peak)])[("flux2_klein", "cuda", 8)]["per_megapixel"]

    # 2048x2048 is 4.19 MP: a 1.56 MP climb of 0.49 GiB is 1 GiB there.
    assert term(10.4) == 0
    assert term(9.0) == 0
    assert term(10.5) > 0 and term(11.0) > 0


def test_size_terms_take_the_larger_slope_when_models_share_a_family() -> None:
    rows = [_row("a-m", _MP1, 6.0, model="klein4b"), _row("a-xl", _XL, 8.0, model="klein4b"), _row("b-m", _MP1, 10.0), _row("b-xl", _XL, 13.0)]
    terms = size_terms(rows)[("flux2_klein", "cuda", 8)]
    assert terms["per_megapixel"] == round(3 * GIB / ((_XL - _MP1) / 1e6))
    assert {model["model"] for model in model_terms(rows)} == {"klein4b", "klein9b"}


def test_size_terms_keep_families_and_platforms_apart() -> None:
    rows = [
        _row("m", _MP1, 10.0),
        _row("xl", _XL, 13.0),
        _row("z-m", _MP1, 8.0, family="zimage", model="zit"),
        _row("z-xl", _XL, 8.2, family="zimage", model="zit"),
        _row("mac-m", _MP1, 9.0, platform="mac"),
        _row("mac-xl", _XL, 9.5, platform="mac"),
    ]
    terms = size_terms(rows)
    assert set(terms) == {("flux2_klein", "cuda", 8), ("zimage", "cuda", 8), ("flux2_klein", "mac", 8)}
    assert terms[("zimage", "cuda", 8)]["per_megapixel"] == 0 and terms[("flux2_klein", "cuda", 8)]["per_megapixel"] > 0


def test_size_terms_keep_quantize_levels_apart() -> None:
    rows = [
        _row("m8", _MP1, 4.0),
        _row("xl8", _XL, 4.2),
        _row("m4", _MP1, 6.0, quantize=4),
        _row("xl4", _XL, 9.0, quantize=4),
        _row("r4", _MP1, 9.0, quantize=4, references=3),
    ]
    terms = size_terms(rows)
    assert terms[("flux2_klein", "cuda", 8)] == {"per_megapixel": 0, "per_reference": 0}
    assert terms[("flux2_klein", "cuda", 4)]["per_megapixel"] == round(3 * GIB / ((_XL - _MP1) / 1e6))
    assert terms[("flux2_klein", "cuda", 4)]["per_reference"] == GIB


def test_allocated_rows_fit_the_allocated_peak_and_skip_rows_without_one() -> None:
    rows = [
        _row("m", _MP1, 9.0, quantize=4, platform_allocated_peak_bytes=int(5.0 * GIB)),
        _row("xl", _XL, 9.0, quantize=4, platform_allocated_peak_bytes=int(8.0 * GIB)),
        _row("old", _XL, 9.1, quantize=4),
    ]

    assert size_terms(rows)[("flux2_klein", "cuda", 4)]["per_megapixel"] == 0  # the reserved peak barely moves
    terms = size_terms(allocated_rows(rows))[("flux2_klein", "cuda", 4)]
    assert terms["per_megapixel"] == round(3 * GIB / ((_XL - _MP1) / 1e6))
    assert [row["id"] for row in allocated_rows(rows)] == ["m", "xl"]


def test_size_terms_derive_the_reference_term_from_the_smallest_size() -> None:
    rows = [_row("m", _MP1, 10.0), _row("xl", _XL, 10.0), _row("r1", _MP1, 11.0, references=1), _row("r3", _MP1, 13.4, references=3)]
    term = size_terms(rows)[("flux2_klein", "cuda", 8)]
    assert term["per_megapixel"] == 0 and term["per_reference"] == pytest.approx(3.4 * GIB / 3, abs=2)


def test_size_terms_ignore_failed_rows_rows_without_a_peak_and_edit_and_upscale_rows() -> None:
    rows = [
        _row("m", _MP1, 10.0),
        _row("xl", _XL, 20.0, status="failed"),
        _row("xl2", _XL, 20.0, platform_peak_bytes=None),
        _row("edit", 4 * _MP1, 30.0, edit="p"),
        _row("up", _XL, 30.0, upscale=2),
    ]
    assert size_terms(rows)[("flux2_klein", "cuda", 8)] == {"per_megapixel": 0, "per_reference": 0}


def test_the_largest_of_repeated_runs_counts() -> None:
    rows = [_row("m", _MP1, 10.0), _row("m-again", _MP1, 10.1), _row("xl", _XL, 12.0), _row("xl-again", _XL, 12.3)]
    model = model_terms(rows)[0]
    assert model["small_peak_bytes"] == int(10.1 * GIB) and model["large_peak_bytes"] == int(12.3 * GIB)


def test_upscale_checks_flag_an_upscaled_job_more_than_1_gib_above_the_plain_one() -> None:
    plain = _row("xl", _XL, 10.0, family="zimage", model="zit")
    assert upscale_checks([plain, _row("up", _XL, 10.5, family="zimage", model="zit", upscale=2)])[0]["escalate"] is False
    check = upscale_checks([plain, _row("up", _XL, 11.5, family="zimage", model="zit", upscale=2)])[0]
    assert check["escalate"] is True and check["delta_bytes"] == int(11.5 * GIB) - int(10.0 * GIB)


def test_edit_checks_compare_an_edit_with_the_prediction_from_the_terms() -> None:
    rows = [_row("m", _MP1, 10.0), _row("xl", _XL, 13.0), _row("r3", _MP1, 10.0, references=3), _row("edit", 2 * _XL, 17.0, edit="p")]
    check = edit_checks(rows, size_terms(rows))[0]
    assert check["id"] == "edit" and check["observed_bytes"] == int(17.0 * GIB)
    assert check["predicted_bytes"] == int(10.0 * GIB) + size_terms(rows)[("flux2_klein", "cuda", 8)]["per_megapixel"] * (2 * _XL - _MP1) // 1_000_000


# --- the long cases and the footprint column -----------------------------------------------------------------------------


def test_a_video_case_with_its_own_size_and_references_runs_at_that_size_with_synthetic_pictures(tmp_path: Path) -> None:
    cases = load_cases("video")
    case = {case["id"]: case for case in expand_cases(cases, media="video", platform="cuda")}["h3-5s-refs"]

    video.prepare(case, cases, {})
    video.ensure_sources(case, tmp_path, cases)
    argv = video.case_argv(case, tmp_path)

    assert (case["width"], case["height"], case["frames"], case["pixels"]) == (512, 320, 124, 512 * 320)
    assert argv[argv.index("--width") + 1 :][:3] == ["512", "--height", "320"] and "--size" not in argv and "--ratio" not in argv
    references = argv[argv.index("--reference") + 1].split(",")
    assert len(references) == 2 and all(Path(path).is_file() for path in references)
    from PIL import Image

    assert Image.open(references[0]).size == (512, 320)
    assert "<Picture 1>" in case["prompt"] and "<Picture 2>" in case["prompt"]


def test_the_plain_long_h3_case_has_no_references(tmp_path: Path) -> None:
    cases = load_cases("video")
    case = {case["id"]: case for case in expand_cases(cases, media="video", platform="mac")}["h3-5s"]

    video.prepare(case, cases, {})

    assert "--reference" not in video.case_argv(case, tmp_path) and (case["width"], case["frames"]) == (512, 124)


def test_the_video_estimate_counts_the_references(monkeypatch: pytest.MonkeyPatch) -> None:
    from zvisiongenerator.utils.memory_verdict import MemoryJob

    seen: list[MemoryJob] = []
    monkeypatch.setattr("zvisiongenerator.backends.accelerator_memory_budget_or_none", lambda: MemoryBudget(10 * GIB, 32 * GIB))
    monkeypatch.setattr(
        "zvisiongenerator.utils.memory_verdict.judge", lambda job, machine, weights: seen.append(job) or SimpleNamespace(gpu_bytes=1, ram_bytes=1, status="fits", resource=None, placement=None)
    )

    video.estimate({"family": "h3", "width": 512, "height": 320, "frames": 124, "references": 2})

    assert seen[0].references == 2


def test_the_audio_case_file_runs_the_longest_takes_the_app_allows() -> None:
    from zvisiongenerator.utils.audio_request import find_audio_preset
    from zvisiongenerator.utils.config import load_config

    cases = {case["id"]: case for case in expand_cases(load_cases("audio"), media="audio", platform="cuda")}
    limits = find_audio_preset(load_config(), "sa3", missing="no model").limits

    assert {cases["sa3-sound-max"]["seconds"], cases["sa3-music-max"]["seconds"]} == {limits.max_seconds}
    assert {(cases["sa3-sound-max"]["kind"]), cases["sa3-music-max"]["kind"]} == {"sound", "music"}
    assert cases["yue2-song-max"]["lyrics"].count("[") >= 14
    assert len(" ".join([cases["qwen3-speech-long"]["prompt"]] * cases["qwen3-speech-long"]["prompt_repeat"])) >= 3000
    assert cases["qwen3-speech-clone"]["voice_text"]


def test_a_long_text_case_reads_its_prompt_repeated(tmp_path: Path) -> None:
    case = {"id": "t", "model": "qwen3_tts", "kind": "speech", "seed": 1, "prompt": "One.", "prompt_repeat": 3}

    argv = audio.case_argv(case, tmp_path)

    assert argv[argv.index("--prompt") + 1] == "One. One. One."


def test_a_cloned_voice_case_follows_a_recording_the_bench_makes_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    made: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> None:
        made.append(command)
        (Path(command[command.index("--output") + 1]) / "take.wav").write_bytes(b"x")

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    case = {"id": "c", "model": "qwen3_tts", "kind": "speech", "seed": 1, "prompt": "Hi.", "voice_text": "What the recording says."}

    audio.ensure_sources(case, tmp_path, {})
    audio.ensure_sources(case, tmp_path, {})
    argv = audio.case_argv(case, tmp_path)

    assert len(made) == 1 and made[0][made[0].index("--prompt") + 1] == "What the recording says." and "--voice" not in made[0]
    assert argv[argv.index("--voice") + 1] == str(tmp_path / "sources" / "voice.wav") and (tmp_path / "sources" / "voice.wav").is_file()
    assert argv[argv.index("--voice-text") + 1] == "What the recording says." and "--voice-consent" not in argv
    assert [path.name for path in (tmp_path / "sources").iterdir()] == ["voice.wav"]


def test_a_case_without_a_voice_text_makes_no_recording(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(audio.subprocess, "run", lambda *args, **kwargs: pytest.fail("no recording is needed"))

    audio.ensure_sources({"id": "c", "model": "sa3", "kind": "sound", "seed": 1, "prompt": "x"}, tmp_path, {})


def test_the_rule_table_shows_a_macs_footprint_peak() -> None:
    row = _ran("m", status="fits", estimate_gib=8.0, measured_gib=6.0, media="video", footprint_peak_bytes=int(12.5 * GIB))
    checks, left_out = rule_checks([row])

    assert "12.50" in format_rule([row], checks, left_out)


def test_the_measured_ram_counts_file_backed_pages_only_where_the_estimate_counts_mapped_weights() -> None:
    row = {"rss_anon_shmem_peak_bytes": 5 * GIB, "rss_anon_shmem_file_peak_bytes": 12 * GIB}

    assert measured_ram_bytes(row) == 5 * GIB
    assert measured_ram_bytes(row | {"estimate": {"ram_mapped": True}}) == 12 * GIB
    assert measured_ram_bytes({"rss_file_peak_bytes": GIB}) is None


def test_the_sampler_gap_is_the_kernels_high_water_mark_above_the_simultaneous_anon_shmem_file_peak() -> None:
    row = {"vm_hwm_bytes": 12 * GIB, "rss_anon_shmem_peak_bytes": 9 * GIB, "rss_anon_shmem_file_peak_bytes": 11 * GIB}
    assert sampler_gap_bytes(row) == GIB  # like for like: against the sum with the file-backed pages, not the figure that leaves them out
    assert sampler_gap_bytes({"rss_anon_shmem_file_peak_bytes": 11 * GIB}) is None and sampler_gap_bytes({"vm_hwm_bytes": GIB}) is None


def test_the_reports_show_the_simultaneous_peaks_and_the_high_water_mark_and_name_a_sampler_that_missed_a_peak() -> None:
    near = _ran("near", status="fits", estimate_gib=7.4, measured_gib=7.0, ram_estimate=2.2, ram=2.1, vm_hwm_bytes=int(2.3 * GIB), rss_anon_shmem_file_peak_bytes=int(2.2 * GIB))
    missed = _ran("missed", status="fits", estimate_gib=7.4, measured_gib=7.0, ram_estimate=2.2, ram=2.1, vm_hwm_bytes=4 * GIB, rss_anon_shmem_file_peak_bytes=2 * GIB)
    report = format_report([near | {"family": "zimage", "model": "zit", "quantize": 8, "pixels": 1024 * 1024, "references": 0}])
    assert "anon+shm" in report.splitlines()[0] and "vmHWM" in report.splitlines()[0]
    assert "2.10" in report and "2.30" in report and "2.20" in report
    rule = format_rule([near, missed], *rule_checks([near, missed]))
    assert "vmHWM" in rule.splitlines()[0] and "hwm gap" in rule.splitlines()[0]
    assert "SAMPLER MISSED A PEAK missed: VmHWM is 2.00 GiB above" in rule and "SAMPLER MISSED A PEAK near" not in rule


def test_the_measured_ram_is_the_simultaneous_peak_and_a_row_without_it_has_none() -> None:
    separate = {"rss_anon_peak_bytes": 16 * GIB, "rss_shmem_peak_bytes": 10 * GIB, "rss_file_peak_bytes": 2 * GIB}

    assert measured_ram_bytes(separate) is None  # the separate peaks are not summed
    assert measured_ram_bytes(separate | {"rss_anon_shmem_peak_bytes": 20 * GIB}) == 20 * GIB


def _in_process(case_id: str, hold: float, *, outcome: str, card_gib: float, charged_gib: float | None = None, **extra: object) -> dict:
    """A row of a case run under an in-process hold: the driver's baseline (read before the hold) is the 1 GiB other programs held, so the card the job had is *card_gib* once the hold's charge is subtracted."""
    charged = int((hold if charged_gib is None else charged_gib) * GIB)
    row = _ran(
        case_id,
        status="fits",
        estimate_gib=7.4,
        measured_gib=7.0,
        hold_in_process_gib=hold,
        hold_charged_bytes=charged,
        gpu_driver_total_bytes=10 * GIB,
        gpu_baseline_used_bytes=int(10 * GIB - card_gib * GIB - charged),
        **extra,
    )
    return row if outcome == "ran" else row | {"status": "failed", "error": "OutOfMemoryError: CUDA out of memory"}


def test_an_in_process_hold_is_subtracted_from_the_card_the_job_had() -> None:
    row = {"gpu_driver_total_bytes": 10 * GIB, "gpu_baseline_used_bytes": GIB}
    assert card_the_job_had(row) == 9 * GIB
    assert card_the_job_had(row | {"hold_charged_bytes": GIB // 2}) == 9 * GIB - GIB // 2
    assert card_the_job_had(row | {"hold_charged_bytes": None}) == 9 * GIB
    assert card_the_job_had({"hold_charged_bytes": GIB}) is None


def test_in_process_holds_make_a_bracket_like_holds_in_another_process() -> None:
    rows = with_brackets(
        [
            _in_process("probed", 0.0, outcome="ran", card_gib=9.0),
            _in_process("probed", 0.75, outcome="ran", card_gib=8.0, charged_gib=0.77),
            _in_process("probed", 1.0, outcome="failed", card_gib=7.5, charged_gib=1.0),
        ]
    )
    assert {row.get("hold_in_process_gib"): row["bracket_bytes"] for row in rows} == dict.fromkeys((0.0, 0.75, 1.0), (int(7.5 * GIB), 8 * GIB))
    assert measured_gpu_bytes(rows[0]) == int(7.75 * GIB)


def test_a_row_under_an_in_process_hold_is_a_judged_card_and_shows_its_hold() -> None:
    oom = _in_process("a", 1.0, outcome="failed", card_gib=7.0) | {"estimate": {"status": "tight", "gpu_bytes": 9 * GIB}}
    assert _failures([oom]) == []  # a hold makes the verdict judge a card the job did not have: no check on the status
    ran = _in_process("a", 0.75, outcome="ran", card_gib=8.0)
    checks, left_out = rule_checks([ran])
    assert {check["hold_gib"] for check in checks} == {0.75}
    assert " 0.75 " in format_rule([ran], checks, left_out)


def test_an_in_process_hold_makes_a_result_its_own_and_is_not_a_cold_row() -> None:
    plain = {"id": "zit-q4-m"}
    assert len({bench_main._key(plain), bench_main._key(plain | {"hold_gib": 1.0}), bench_main._key(plain | {"hold_in_process_gib": 1.0}), bench_main._key(plain | {"hold_in_process_gib": 0.5})}) == 4


def test_a_case_that_could_not_take_its_hold_is_a_failed_row_that_never_ran(tmp_path: Path, measured: None) -> None:
    def refuse() -> dict:
        raise RuntimeError("cuMemAlloc failed with CUDA driver error 2")

    ran: list[str] = []
    row = _run([], lambda _argv: ran.append("cli"), tmp_path, hold=refuse)
    assert ran == [] and row["status"] == "failed" and "hold could not be taken" in row["error"] and "hold_charged_bytes" not in row
    assert not says_out_of_memory(row["error"])  # a hold that could not be taken is a setup failure, never a bracket's out-of-memory edge


def test_a_hold_failure_that_says_out_of_memory_is_reworded_so_it_is_never_a_bracket_edge(tmp_path: Path, measured: None) -> None:
    def refuse() -> dict:
        raise RuntimeError("torch.OutOfMemoryError: CUDA out of memory while taking the hold")

    row = _run([], lambda _argv: None, tmp_path, hold=refuse)

    assert row["status"] == "failed" and "hold could not be taken" in row["error"] and not says_out_of_memory(row["error"])
    assert with_brackets([row | {"gpu_driver_total_bytes": 10 * GIB, "gpu_baseline_used_bytes": GIB}])[0].get("bracket_bytes") is None


def test_the_hold_is_taken_after_the_baseline_reading_and_its_charge_joins_the_row(tmp_path: Path, measured: None, monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    monkeypatch.setattr("bench.memory.measure.smi_memory", lambda: order.append("baseline") or (GIB, 10 * GIB))

    def fake_cli(_argv: list[str]) -> None:
        order.append("job")
        (tmp_path / "picture.png").write_bytes(b"x")

    row = _run([], fake_cli, tmp_path, hold=lambda: order.append("hold") or {"hold_charged_bytes": GIB // 2})
    assert order == ["baseline", "hold", "job"] and row["status"] == "ok"
    assert (row["hold_charged_bytes"], row["gpu_baseline_used_bytes"]) == (GIB // 2, GIB)
