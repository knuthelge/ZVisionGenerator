"""Tests for the weight resolver: its three sources in order, the shipped table's composition and the Hugging Face path (mocked)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests._factories import metadata_copy, use_model_metadata
from tests.common._helpers import write_safetensors
from zvisiongenerator.utils import weight_totals
from zvisiongenerator.utils.memory_verdict import MemoryWeights
from zvisiongenerator.utils.model_memory import WeightTotals
from zvisiongenerator.utils.weight_totals import (
    ResolvedWeights,
    build_row,
    _fetch_hub_row,
    resolve_image_weights,
    resolve_ltx_weights,
    _totals_from_row,
    _totals_to_row,
    _weight_table,
)

_REPO = "owner/typed-in"
_PIN = "c" * 40
_TOTALS = {"float_matrix_elements": 100, "float_other_elements": 7}
_ROW = {"family": "zimage", "components": {"text_encoder": _TOTALS, "transformer": _TOTALS, "vae": _TOTALS}}
_NO_FILES = lambda _reference: None  # noqa: E731 - nothing is downloaded


def _serve_table(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, table: dict[str, Any]) -> None:
    data = metadata_copy()
    data["weight_totals"] = table
    use_model_metadata(monkeypatch, tmp_path, data)


def _local_model(root: Path) -> Path:
    """Return a Z-Image-like model folder whose headers differ from ``_TOTALS``."""
    (root).mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "ZImagePipeline", "transformer": ["diffusers", "ZImageTransformer2DModel"], "vae": ["diffusers", "AutoencoderKL"]}), encoding="utf-8"
    )
    write_safetensors(root / "transformer" / "diffusion_pytorch_model.safetensors", {"w": ("BF16", [3, 4])})
    write_safetensors(root / "vae" / "diffusion_pytorch_model.safetensors", {"b": ("F32", [5])})
    return root


class TestRows:
    def test_totals_round_trip_through_a_row_without_their_defaults(self) -> None:
        totals = WeightTotals(float_matrix_elements=5, packed_bytes=9, prequantized=True)

        entry = _totals_to_row(totals)

        assert entry == {"float_matrix_elements": 5, "packed_bytes": 9, "prequantized": True}
        assert _totals_from_row(entry) == totals


class TestImageResolverOrder:
    def test_the_local_headers_come_before_the_table(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        model_dir = _local_model(tmp_path / "zit")
        _serve_table(monkeypatch, tmp_path, {"Tongyi-MAI/Z-Image-Turbo": _ROW})

        resolved = resolve_image_weights("zit", "Tongyi-MAI/Z-Image-Turbo", platform_key="linux", find_local_dir=lambda _reference: model_dir)

        assert resolved is not None and resolved.family == "zimage"
        assert resolved.weights.components["transformer"] == WeightTotals(float_matrix_elements=12)
        assert resolved.weights.components["vae"] == WeightTotals(float_other_elements=5)

    def test_the_table_answers_a_model_that_is_not_downloaded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _serve_table(monkeypatch, tmp_path, {"Tongyi-MAI/Z-Image-Turbo": _ROW})

        resolved = resolve_image_weights("zit", "Tongyi-MAI/Z-Image-Turbo", platform_key="linux", find_local_dir=_NO_FILES)

        assert resolved is not None and resolved.family == "zimage"
        assert resolved.weights.components["transformer"] == _totals_from_row(_TOTALS) and resolved.stored_bits is None

    def test_a_model_folder_with_unreadable_weights_falls_through_to_the_table(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        broken = tmp_path / "broken"
        (broken / "transformer").mkdir(parents=True)
        (broken / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"not safetensors")
        _serve_table(monkeypatch, tmp_path, {"Tongyi-MAI/Z-Image-Turbo": _ROW})

        resolved = resolve_image_weights("zit", "Tongyi-MAI/Z-Image-Turbo", platform_key="linux", find_local_dir=lambda _reference: broken)

        assert resolved is not None and resolved.weights.components["transformer"] == _totals_from_row(_TOTALS)

    def test_the_hub_is_read_only_when_allowed_and_only_for_a_repo_the_table_lacks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _serve_table(monkeypatch, tmp_path, {f"{_REPO}@{_PIN}": _ROW})
        fetch = MagicMock(return_value={"family": "flux2_klein", "size": "9b", "components": {"transformer": _TOTALS}})
        monkeypatch.setattr(weight_totals, "_fetch_hub_row", fetch)

        in_table = resolve_image_weights(None, f"{_REPO}@{_PIN}", platform_key="linux", hub=True, find_local_dir=_NO_FILES)
        unknown = resolve_image_weights(None, "owner/other", platform_key="linux", find_local_dir=_NO_FILES)

        assert in_table is not None and in_table.family == "zimage"
        assert unknown is None
        fetch.assert_not_called()

    def test_the_hub_answers_a_repo_nothing_else_knows(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _serve_table(monkeypatch, tmp_path, {})
        fetch = MagicMock(return_value={"family": "flux2_klein", "size": "9b", "components": {"transformer": _TOTALS}})
        monkeypatch.setattr(weight_totals, "_fetch_hub_row", fetch)

        resolved = resolve_image_weights(None, "owner/other@" + _PIN, platform_key="linux", hub=True, find_local_dir=_NO_FILES)

        assert resolved is not None and (resolved.family, resolved.model_size) == ("flux2_klein", "9b")
        fetch.assert_called_once_with("owner/other", _PIN, ())

    def test_a_hub_that_cannot_answer_gives_no_weights(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _serve_table(monkeypatch, tmp_path, {})
        monkeypatch.setattr(weight_totals, "_fetch_hub_row", lambda *_args: None)

        resolved = resolve_image_weights(None, "owner/other", platform_key="linux", hub=True, find_local_dir=_NO_FILES)

        assert resolved is None

    def test_a_local_path_that_is_no_repo_has_no_table_or_hub_answer(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _serve_table(monkeypatch, tmp_path, {})
        fetch = MagicMock()
        monkeypatch.setattr(weight_totals, "_fetch_hub_row", fetch)

        assert resolve_image_weights(None, "/models/mine", platform_key="linux", hub=True, find_local_dir=_NO_FILES) is None
        fetch.assert_not_called()


class TestBuiltImageModels:
    """Ideogram 4 Instant and MOSS-Image are built from several repos; the shipped table holds each."""

    def test_ideogram_takes_its_transformer_from_the_file_on_cuda_and_the_fp8_repo_on_mac(self) -> None:
        cuda = resolve_image_weights("ideo", "ideo", platform_key="linux", find_local_dir=_NO_FILES)
        mac = resolve_image_weights("ideo", "ideo", platform_key="darwin", find_local_dir=_NO_FILES)

        assert cuda is not None and mac is not None and cuda.family == mac.family == "ideogram4"
        assert cuda.weights.components["transformer"].fp8_matrix_elements == 0 and mac.weights.components["transformer"].fp8_matrix_elements > 0
        assert set(cuda.weights.components) == set(mac.weights.components) == {"text_encoder", "vae", "transformer"}

    def test_moss_turbo_carries_its_lora_and_sft_does_not(self) -> None:
        turbo = resolve_image_weights("moss", "moss", platform_key="linux", find_local_dir=_NO_FILES)
        sft = resolve_image_weights("moss-sft", "moss-sft", platform_key="linux", find_local_dir=_NO_FILES)

        assert turbo is not None and sft is not None and turbo.family == sft.family == "moss"
        assert turbo.weights.turbo_lora_bytes > 0 and sft.weights.turbo_lora_bytes == 0
        assert turbo.weights.components == sft.weights.components

    def test_moss_on_a_mac_loads_its_text_encoder_without_the_vision_tower(self) -> None:
        cuda = resolve_image_weights("moss", "moss", platform_key="linux", find_local_dir=_NO_FILES)
        mac = resolve_image_weights("moss", "moss", platform_key="darwin", find_local_dir=_NO_FILES)

        assert cuda is not None and mac is not None
        assert mac.weights.components["text_encoder"].bfloat16_bytes < cuda.weights.components["text_encoder"].bfloat16_bytes
        assert mac.weights.components["transformer"] == cuda.weights.components["transformer"]

    def test_klein_carries_its_size(self) -> None:
        resolved = resolve_image_weights("klein9b", "unsloth/FLUX.2-klein-9B", platform_key="linux", find_local_dir=_NO_FILES)

        assert resolved is not None and (resolved.family, resolved.model_size) == ("flux2_klein", "9b")


class TestLtxResolver:
    def test_the_table_gives_the_cuda_figures_at_the_models_level(self) -> None:
        four = resolve_ltx_weights("ltx-4", "unused", platform_key="linux", find_local_dir=_NO_FILES)
        eight = resolve_ltx_weights("ltx-8", "unused", platform_key="linux", find_local_dir=_NO_FILES)

        assert four is not None and eight is not None and four.family == "ltx"
        assert (four.weights.ltx_cuda.quantize, eight.weights.ltx_cuda.quantize) == (4, 8)
        assert four.weights.ltx_cuda.transformer_bytes < eight.weights.ltx_cuda.transformer_bytes

    def test_the_table_gives_a_mac_pack_its_gemma_from_the_gemma_repo_or_its_own_files(self) -> None:
        gemma_repo = resolve_ltx_weights("ltx-4", "unused", platform_key="darwin", find_local_dir=_NO_FILES)
        bundled = resolve_ltx_weights("ltx25-4", "unused", platform_key="darwin", find_local_dir=_NO_FILES)

        assert gemma_repo is not None and bundled is not None
        assert gemma_repo.weights.ltx_mlx.upsampler > 0 and bundled.weights.ltx_mlx.gemma > gemma_repo.weights.ltx_mlx.gemma

    def test_local_files_come_before_the_table(self, monkeypatch: pytest.MonkeyPatch) -> None:
        local = MemoryWeights(ltx_cuda=weight_totals.estimate_ltx_cuda_system_memory(block_matrices=1, transformer_rest=1, connectors=1, text_encoder=1, quantize=4))
        monkeypatch.setattr(weight_totals, "_local_ltx", lambda *_args: local)

        resolved = resolve_ltx_weights("ltx-4", "unused", platform_key="linux")

        assert resolved == ResolvedWeights(local, family="ltx")

    def test_a_model_the_metadata_does_not_know_has_no_weights(self) -> None:
        assert resolve_ltx_weights("not-a-model", "owner/repo", platform_key="linux", find_local_dir=_NO_FILES) is None

    def test_a_table_without_the_rows_has_no_weights(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _serve_table(monkeypatch, tmp_path, {})

        assert resolve_ltx_weights("ltx-4", "unused", platform_key="linux", find_local_dir=_NO_FILES) is None
        assert resolve_ltx_weights("ltx-4", "unused", platform_key="darwin", find_local_dir=_NO_FILES) is None


def test_the_shipped_table_is_keyed_by_ref() -> None:
    assert all(key.count("@") <= 1 for key in _weight_table())


# --- the Hugging Face path (mocked) ----------------------------------------------------------------------------------


def _tensor(dtype: str, shape: list[int]) -> SimpleNamespace:
    return SimpleNamespace(dtype=dtype, shape=shape, data_offsets=[0, 8])


class _FakeApi:
    """A ``HfApi`` for one diffusers repo: a text encoder (with an LM head) and a transformer."""

    def __init__(self, *, sha: str = _PIN) -> None:
        self.sha = sha
        self.calls: list[str] = []
        self.tensors = {
            "text_encoder/model.safetensors": {"model.embed_tokens.weight": _tensor("BF16", [10, 4]), "lm_head.weight": _tensor("BF16", [10, 4]), "norm.weight": _tensor("BF16", [4])},
            "transformer/diffusion_pytorch_model.safetensors": {"blocks.0.weight": _tensor("BF16", [8, 8]), "blocks.0.bias": _tensor("BF16", [8])},
            "single.safetensors": {"w": _tensor("BF16", [2, 3])},
        }

    def model_info(self, repo_id: str, *, revision: str | None = None, timeout: float | None = None) -> SimpleNamespace:
        self.calls.append(f"model_info {repo_id}")
        return SimpleNamespace(sha=self.sha, siblings=[SimpleNamespace(rfilename=name) for name in ["model_index.json", *self.tensors]])

    def parse_safetensors_file_metadata(self, repo_id: str, name: str, *, revision: str | None = None, timeout: float | None = None) -> SimpleNamespace:
        self.calls.append(f"header {name}")
        return SimpleNamespace(tensors=self.tensors[name])


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> Callable[[Any], Any]:
    """Return a function that makes the Hugging Face client the one ``_fetch_hub_row`` reads through *api*, and gives *api* back."""

    def serve(api: Any) -> Any:
        monkeypatch.setattr("huggingface_hub.HfApi", lambda *args, **kwargs: api)
        return api

    return serve


@pytest.fixture
def served_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve ``model_index.json`` of a Klein-like repo for every download the family detection and the reader make."""
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model_index.json").write_text(
        json.dumps({"_class_name": "Flux2KleinPipeline", "text_encoder": ["transformers", "Qwen3Model"], "transformer": ["diffusers", "Flux2Transformer2DModel"]}), encoding="utf-8"
    )
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *args, **kwargs: str(snapshot / "model_index.json"))


class TestHubRow:
    def test_a_diffusers_repo_gives_its_components_as_the_local_files_would(self, served_index: None, hub: Callable[[Any], Any]) -> None:
        hub(_FakeApi())
        row = _fetch_hub_row("black-forest-labs/FLUX.2-klein-4B", None)

        assert row is not None
        assert (row["revision"], row["family"], row["size"]) == (_PIN, "flux2_klein", "4b")
        assert _totals_from_row(row["components"]["text_encoder"]) == WeightTotals(float_matrix_elements=40, float_other_elements=4, embedding_elements=40)  # the LM head is not counted
        assert _totals_from_row(row["components"]["transformer"]) == WeightTotals(float_matrix_elements=64, float_other_elements=8)

    def test_a_pinned_repo_records_no_revision(self, served_index: None, hub: Callable[[Any], Any]) -> None:
        hub(_FakeApi())
        row = _fetch_hub_row("black-forest-labs/FLUX.2-klein-4B", _PIN)

        assert row is not None and "revision" not in row

    def test_named_files_give_their_totals_and_the_size_of_their_data(self, hub: Callable[[Any], Any]) -> None:
        hub(_FakeApi())
        row = _fetch_hub_row(_REPO, _PIN, ("single.safetensors",))

        assert row == {"files": {"single.safetensors": {"float_matrix_elements": 6, "bytes": 8}}}

    def test_a_missing_file_gives_no_row(self, hub: Callable[[Any], Any]) -> None:
        hub(_FakeApi())
        assert _fetch_hub_row(_REPO, _PIN, ("absent.safetensors",)) is None

    def test_a_repo_that_is_not_a_diffusers_model_gives_no_row(self, hub: Callable[[Any], Any]) -> None:
        api = _FakeApi()
        api.model_info = lambda *_args, **_kwargs: SimpleNamespace(sha=_PIN, siblings=[SimpleNamespace(rfilename="model.safetensors")])  # type: ignore[method-assign]

        hub(api)
        assert _fetch_hub_row(_REPO, _PIN) is None

    @pytest.mark.parametrize("error", [TimeoutError("timed out"), ConnectionError("offline"), RuntimeError("401")])
    def test_a_failure_gives_no_row(self, error: Exception, hub: Callable[[Any], Any]) -> None:
        api = MagicMock()
        api.model_info.side_effect = error

        hub(api)
        assert _fetch_hub_row(_REPO, _PIN) is None

    def test_a_refusal_gives_no_row(self, hub: Callable[[Any], Any]) -> None:
        from huggingface_hub.errors import GatedRepoError

        api = MagicMock()
        api.model_info.side_effect = GatedRepoError("gated", response=MagicMock(status_code=403, headers={}))

        hub(api)
        assert _fetch_hub_row(_REPO, _PIN) is None

    def test_a_read_that_takes_longer_than_the_timeout_gives_no_row(self, monkeypatch: pytest.MonkeyPatch, hub: Callable[[Any], Any]) -> None:
        monkeypatch.setattr(weight_totals, "_HUB_TIMEOUT_SECONDS", 0.05)
        api = MagicMock()
        api.model_info.side_effect = lambda *_args, **_kwargs: time.sleep(0.5)

        started = time.monotonic()
        hub(api)
        row = _fetch_hub_row(_REPO, _PIN)

        assert row is None and time.monotonic() - started < 0.4

    def test_every_request_is_bounded_by_the_deadline(self) -> None:
        api = _FakeApi()
        api.model_info = MagicMock(side_effect=api.model_info)  # type: ignore[method-assign]
        build_row(weight_totals.HubReader(api, deadline=time.monotonic() + 5), _REPO, _PIN, ("single.safetensors",))

        assert 0 < api.model_info.call_args.kwargs["timeout"] <= weight_totals._HUB_TIMEOUT_SECONDS
