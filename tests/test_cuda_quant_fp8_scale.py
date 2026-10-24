"""Tests for reading scaled-FP8 safetensors files into bfloat16 weights (``dequantized_tensors``), on tiny real tensors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import save_file  # noqa: E402

from zvisiongenerator.backends.cuda.quant import dequantized_tensors  # noqa: E402


def _fp8(values: list[list[float]]) -> torch.Tensor:
    return torch.tensor(values).to(torch.float8_e4m3fn)


def _quant_format(name: str) -> torch.Tensor:
    return torch.tensor(list(json.dumps({"format": name}).encode()), dtype=torch.uint8)


def _save(path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    save_file(tensors, str(path))
    return path


def test_a_per_tensor_scale_is_multiplied_out_and_the_metadata_is_dropped(tmp_path: Path) -> None:
    path = _save(
        tmp_path / "a.safetensors",
        {
            "x.weight": _fp8([[1.0, 2.0], [4.0, -8.0]]),
            "x.weight_scale": torch.tensor(0.5),
            "x.input_scale": torch.tensor(3.0),
            "x.comfy_quant": _quant_format("float8_e4m3fn"),
            "x.bias": torch.ones(2, dtype=torch.bfloat16),
        },
    )

    out = dict(dequantized_tensors([path]))

    assert set(out) == {"x.weight", "x.bias"}
    assert out["x.weight"].dtype == torch.bfloat16
    assert torch.equal(out["x.weight"], torch.tensor([[0.5, 1.0], [2.0, -4.0]], dtype=torch.bfloat16))
    assert torch.equal(out["x.bias"], torch.ones(2, dtype=torch.bfloat16))


def test_the_older_scale_weight_name_and_a_scale_in_another_shard_are_found(tmp_path: Path) -> None:
    weights = _save(tmp_path / "w.safetensors", {"y.weight": _fp8([[2.0, 4.0]])})
    scales = _save(tmp_path / "s.safetensors", {"y.scale_weight": torch.tensor([0.25])})

    out = dict(dequantized_tensors([weights, scales]))

    assert out.keys() == {"y.weight"}
    assert torch.equal(out["y.weight"], torch.tensor([[0.5, 1.0]], dtype=torch.bfloat16))


def test_a_per_row_scale_scales_each_output_row(tmp_path: Path) -> None:
    path = _save(tmp_path / "r.safetensors", {"z.weight": _fp8([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]), "z.weight_scale": torch.tensor([[2.0], [4.0]])})

    out = dict(dequantized_tensors([path]))

    assert torch.equal(out["z.weight"], torch.tensor([[2.0] * 3, [4.0] * 3], dtype=torch.bfloat16))


def test_an_fp8_weight_without_a_scale_is_cast_and_other_dtypes_are_untouched(tmp_path: Path) -> None:
    path = _save(tmp_path / "u.safetensors", {"a.weight": _fp8([[1.5, 2.0]]), "b.weight": torch.tensor([[1.0, 2.0]]), "c.weight": torch.tensor([[1.0]], dtype=torch.float16)})

    out = dict(dequantized_tensors([path]))

    assert out["a.weight"].dtype == torch.bfloat16
    assert torch.equal(out["a.weight"], torch.tensor([[1.5, 2.0]], dtype=torch.bfloat16))
    assert out["b.weight"].dtype == torch.float32
    assert out["c.weight"].dtype == torch.float16


def test_wanted_filters_by_the_weight_name_and_still_finds_its_scale(tmp_path: Path) -> None:
    path = _save(tmp_path / "f.safetensors", {"p.weight": _fp8([[2.0]]), "p.weight_scale": torch.tensor(2.0), "q.weight": _fp8([[2.0]]), "q.weight_scale": torch.tensor(8.0)})

    out = dict(dequantized_tensors([path], lambda name: name.startswith("p.")))

    assert list(out) == ["p.weight"]
    assert out["p.weight"].item() == 4.0


def test_another_quantization_format_is_refused(tmp_path: Path) -> None:
    path = _save(tmp_path / "i.safetensors", {"x.weight": torch.zeros(2, 2, dtype=torch.uint8), "x.comfy_quant": _quant_format("int4")})

    with pytest.raises(ValueError, match="int4"):
        list(dequantized_tensors([path]))


def test_a_block_wise_scale_is_refused(tmp_path: Path) -> None:
    path = _save(tmp_path / "b.safetensors", {"x.weight": _fp8([[1.0] * 4] * 2), "x.weight_scale": torch.ones(2, 2)})

    with pytest.raises(ValueError, match="per-tensor and per-row"):
        list(dequantized_tensors([path]))


def test_a_scale_on_a_weight_that_is_not_fp8_is_refused(tmp_path: Path) -> None:
    path = _save(tmp_path / "n.safetensors", {"x.weight": torch.ones(2, 2, dtype=torch.bfloat16), "x.weight_scale": torch.tensor(2.0)})

    with pytest.raises(ValueError, match="not FP8"):
        list(dequantized_tensors([path]))
