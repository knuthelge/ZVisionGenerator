"""Tests for decoding quantized checkpoint weights (scaled FP8, INT8, ConvRot INT8, W4A4, W4A8, W6A8, NVFP4 and MXFP8)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from zvisiongenerator.utils.quantized_weights import decode_weight, decoded_tensors, quantization_formats


def _record(**fields) -> torch.Tensor:
    return torch.tensor(list(json.dumps(fields).encode()), dtype=torch.uint8)


def _hadamard(size: int) -> torch.Tensor:
    base = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    matrix = base
    while matrix.shape[0] < size:
        matrix = torch.kron(matrix, base)
    return matrix / math.sqrt(size)


def _rotate(weight: torch.Tensor, group: int) -> torch.Tensor:
    rows, columns = weight.shape
    return (weight.reshape(rows, columns // group, group) @ _hadamard(group).T).reshape(rows, columns)


def _weight(rows: int = 8, columns: int = 64) -> torch.Tensor:
    return torch.randn(rows, columns, generator=torch.Generator().manual_seed(0)) * 0.05


def _int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = weight.abs().amax(dim=1, keepdim=True) / 127
    return (weight / scale).round().clamp(-127, 127).to(torch.int8), scale


def _rel(decoded: torch.Tensor, weight: torch.Tensor) -> float:
    return ((decoded.float() - weight).norm() / weight.norm()).item()


def test_int8_with_a_per_row_scale_decodes_to_the_weight():
    weight = _weight()
    codes, scale = _int8(weight)

    decoded = decode_weight("w", codes, {"format": "int8_tensorwise"}, {".weight_scale": scale})

    assert decoded.dtype == torch.bfloat16
    assert _rel(decoded, weight) < 0.01


def test_convrot_int8_is_rotated_back_and_reads_wrong_without_it():
    weight = _weight()
    codes, scale = _int8(_rotate(weight, 16))

    rotated = decode_weight("w", codes, {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 16}, {".weight_scale": scale})
    plain = decode_weight("w", codes, {"format": "int8_tensorwise"}, {".weight_scale": scale})

    assert _rel(rotated, weight) < 0.01
    assert _rel(plain, weight) > 0.5


def test_convrot_settings_may_sit_under_params():
    weight = _weight()
    codes, scale = _int8(_rotate(weight, 16))

    decoded = decode_weight("w", codes, {"format": "int8_tensorwise", "params": {"convrot": True, "convrot_groupsize": 16}}, {".weight_scale": scale})

    assert _rel(decoded, weight) < 0.01


def test_w4a4_signed_nibbles_low_first_decode_per_row():
    codes = torch.tensor([[-8, 7, 1, -1] * 4], dtype=torch.int32)
    packed = ((codes[:, 0::2] & 0xF) | ((codes[:, 1::2] & 0xF) << 4)).to(torch.uint8).view(torch.int8)
    rotated_back = _rotate(codes.float() * 0.5, 16)  # what the stored rotated codes decode to

    decoded = decode_weight("w", packed, {"format": "convrot_w4a4", "convrot_groupsize": 16}, {".weight_scale": torch.tensor([0.5])})

    assert torch.allclose(decoded.float(), rotated_back, atol=1e-2)


@pytest.mark.parametrize("codebook", [True, False])
def test_w4a8_codes_go_through_the_codebook_group_and_row_scales(codebook):
    rows, columns, group = 2, 32, 16
    codes = torch.arange(columns).remainder(16).repeat(rows, 1)
    packed = ((codes[:, 0::2] & 0xF) | ((codes[:, 1::2] & 0xF) << 4)).to(torch.uint8).view(torch.int8)
    table = torch.linspace(-1.0, 1.0, 16)
    s_rel = torch.full((rows, columns // group), 100.0).to(torch.float8_e4m3fn)
    s_channel = torch.tensor([0.01, 0.02])
    parts = {".weight_s_rel": s_rel, ".weight_s_channel": s_channel}
    if codebook:
        parts[".weight_codebook"] = table
    values = table[codes] if codebook else codes.float() - 8
    expected = (values * 96.0).round().clamp(-127, 127) * s_channel.reshape(-1, 1)  # 100 is 96 in e4m3

    decoded = decode_weight("w", packed, {"format": "asym_w4a8_int8", "group_size": group, "convrot_groupsize": 16}, parts)

    assert torch.allclose(decoded.float(), _rotate(expected, 16), atol=1e-2)


def test_w6a8_reads_the_top_bits_plane():
    rows, columns = 1, 16
    codes = torch.arange(columns) * 4 % 64  # uses the top two bits
    low = ((codes[0::2] & 0xF) | ((codes[1::2] & 0xF) << 4)).to(torch.uint8)
    high = torch.zeros(columns // 4, dtype=torch.uint8)
    for column in range(columns):
        high[column // 4] |= ((codes[column] >> 4) & 0x3) << (2 * (column % 4))
    packed = torch.cat([low, high]).reshape(rows, -1).view(torch.int8)
    parts = {".weight_s_rel": torch.ones(rows, 1).to(torch.float8_e4m3fn), ".weight_s_channel": torch.ones(rows)}
    expected = (codes.float() - 32).reshape(rows, columns)

    decoded = decode_weight("w", packed, {"format": "w6a8_int8", "group_size": 16, "convrot_groupsize": 16}, parts)

    assert torch.allclose(decoded.float(), _rotate(expected, 16), atol=0.2)


def _to_blocked(matrix: torch.Tensor) -> torch.Tensor:
    """Lay *matrix* out in cuBLAS's tiled block-scale layout (128-row by 4-column tiles, zero padded)."""
    rows, columns = matrix.shape
    row_tiles, column_tiles = -(-rows // 128), -(-columns // 4)
    padded = torch.zeros(row_tiles * 128, column_tiles * 4, dtype=matrix.dtype)
    padded[:rows, :columns] = matrix
    tiles = padded.view(row_tiles, 128, column_tiles, 4).permute(0, 2, 1, 3)
    return tiles.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(row_tiles * 128, column_tiles * 4)


def test_nvfp4_reads_high_nibble_first_with_tiled_block_scales():
    rows, columns = 4, 32
    codes = torch.arange(rows * columns).remainder(16).reshape(rows, columns)
    packed = ((codes[:, 0::2] << 4) | codes[:, 1::2]).to(torch.uint8)
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])[codes]
    block = torch.tensor([[1.0, 2.0], [0.5, 4.0], [1.0, 1.0], [2.0, 0.25]])
    expected = (values.view(rows, 2, 16) * block.unsqueeze(-1)).view(rows, columns) * 0.5

    decoded = decode_weight("w", packed, {"format": "nvfp4"}, {".weight_scale": _to_blocked(block).to(torch.float8_e4m3fn), ".weight_scale_2": torch.tensor(0.5)})

    assert torch.equal(decoded.float(), expected)


def test_mxfp8_applies_power_of_two_tiled_block_scales():
    rows, columns = 2, 64
    weight = torch.ones(rows, columns).to(torch.float8_e4m3fn)
    exponents = torch.tensor([[127, 128], [126, 130]], dtype=torch.uint8)  # 1, 2, 0.5, 8

    decoded = decode_weight("w", weight, {"format": "mxfp8"}, {".weight_scale": _to_blocked(exponents)})

    assert decoded.float()[:, ::32].tolist() == [[1.0, 2.0], [0.5, 8.0]]


def test_a_pre_quant_scale_multiplies_the_columns():
    codes, scale = _int8(torch.ones(2, 4))

    decoded = decode_weight("w", codes, {"format": "int8_tensorwise"}, {".weight_scale": scale, ".pre_quant_scale": torch.tensor([1.0, 2.0, 3.0, 4.0])})

    assert decoded.float()[0].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_scaled_fp8_with_and_without_a_record(tmp_path: Path):
    path = tmp_path / "fp8.safetensors"
    save_file(
        {
            "a.weight": torch.full((2, 2), 2.0).to(torch.float8_e4m3fn),
            "a.weight_scale": torch.tensor(0.5),
            "a.comfy_quant": _record(format="float8_e4m3fn"),
            "b.weight": torch.full((2, 2), 2.0).to(torch.float8_e4m3fn),
            "b.scale_weight": torch.tensor([0.25, 0.5]),
            "c.weight": torch.full((2, 2), 3.0).to(torch.float8_e4m3fn),
            "d.weight": torch.ones(2, 2, dtype=torch.float32),
        },
        str(path),
    )

    out = dict(decoded_tensors([path]))

    assert out.keys() == {"a.weight", "b.weight", "c.weight", "d.weight"}
    assert out["a.weight"].float().unique().tolist() == [1.0]
    assert out["b.weight"].float()[:, 0].tolist() == [0.5, 1.0]
    assert out["c.weight"].dtype == torch.bfloat16
    assert out["d.weight"].dtype == torch.float32


def test_a_scale_in_another_shard_is_found(tmp_path: Path):
    weight = _weight()
    codes, scale = _int8(weight)
    save_file({"a.weight": codes, "a.comfy_quant": _record(format="int8_tensorwise")}, str(tmp_path / "one.safetensors"))
    save_file({"a.weight_scale": scale}, str(tmp_path / "two.safetensors"))

    out = dict(decoded_tensors([tmp_path / "one.safetensors", tmp_path / "two.safetensors"]))

    assert list(out) == ["a.weight"]
    assert _rel(out["a.weight"], weight) < 0.01


def test_each_quantized_weight_gets_its_format(tmp_path: Path):
    path = tmp_path / "mixed.safetensors"
    codes, scale = _int8(_weight())
    save_file(
        {
            "a.weight": codes,
            "a.weight_scale": scale,
            "a.comfy_quant": _record(format="int8_tensorwise", convrot=True),
            "b.weight": torch.ones(2, 2).to(torch.float8_e4m3fn),
            "b.weight_scale": torch.tensor(1.0),
        },
        str(path),
    )

    assert {name: record["format"] for name, record in quantization_formats([path]).items()} == {"a.weight": "int8_tensorwise", "b.weight": "float8_e4m3fn"}


@pytest.mark.parametrize("quant_format", ["svdquant_w4a4", "awq_w4a16", "int4"])
def test_unreadable_formats_are_refused_by_name(tmp_path: Path, quant_format):
    path = tmp_path / "q.safetensors"
    save_file({"a.weight": torch.ones(2, 2, dtype=torch.uint8), "a.weight_scale": torch.ones(1), "a.comfy_quant": _record(format=quant_format)}, str(path))

    with pytest.raises(ValueError):
        quantization_formats([path])


def test_int8_without_a_record_is_refused(tmp_path: Path):
    path = tmp_path / "q.safetensors"
    save_file({"a.weight": torch.ones(2, 2, dtype=torch.int8), "a.weight_scale": torch.ones(1)}, str(path))

    with pytest.raises(ValueError):
        quantization_formats([path])


def test_a_missing_group_scale_is_refused(tmp_path: Path):
    path = tmp_path / "q.safetensors"
    save_file({"a.weight": torch.ones(2, 8, dtype=torch.int8), "a.weight_s_channel": torch.ones(2), "a.comfy_quant": _record(format="asym_w4a8_int8")}, str(path))

    with pytest.raises(ValueError):
        quantization_formats([path])


def _companions(quant_format: str, *, fit: bool) -> dict[str, torch.Tensor]:
    """Return a weight of *quant_format* whose companion tensors fit it, or are one row or block short."""
    short = 0 if fit else 1
    if quant_format == "int8_tensorwise":
        return {"a.weight": torch.ones(8, 64, dtype=torch.int8), "a.weight_scale": torch.ones(8 - short)}
    if quant_format == "asym_w4a8_int8":
        return {"a.weight": torch.ones(2, 16, dtype=torch.uint8), "a.weight_s_rel": torch.ones(2, 2 - short), "a.weight_s_channel": torch.ones(2)}
    if quant_format == "nvfp4":
        return {"a.weight": torch.ones(8, 16, dtype=torch.uint8), "a.weight_scale": torch.ones(512 - short), "a.weight_scale_2": torch.tensor(1.0)}
    return {"a.weight": torch.ones(8, 64).to(torch.float8_e4m3fn), "a.weight_scale": torch.ones(512 - short, dtype=torch.uint8)}


@pytest.mark.parametrize("quant_format", ["int8_tensorwise", "asym_w4a8_int8", "nvfp4", "mxfp8"])
def test_companions_that_do_not_fit_are_refused_before_reading(tmp_path: Path, quant_format):
    good, bad = tmp_path / "good.safetensors", tmp_path / "bad.safetensors"
    save_file({**_companions(quant_format, fit=True), "a.comfy_quant": _record(format=quant_format)}, str(good))
    save_file({**_companions(quant_format, fit=False), "a.comfy_quant": _record(format=quant_format)}, str(bad))

    assert quantization_formats([good])["a.weight"]["format"] == quant_format
    with pytest.raises(ValueError):
        quantization_formats([bad])


def _fp8(values: list[list[float]]) -> torch.Tensor:
    return torch.tensor(values).to(torch.float8_e4m3fn)


def _save(path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    save_file(tensors, str(path))
    return path


def test_a_per_tensor_fp8_scale_is_multiplied_out_and_the_metadata_is_dropped(tmp_path: Path) -> None:
    path = _save(
        tmp_path / "a.safetensors",
        {
            "x.weight": _fp8([[1.0, 2.0], [4.0, -8.0]]),
            "x.weight_scale": torch.tensor(0.5),
            "x.input_scale": torch.tensor(3.0),
            "x.comfy_quant": _record(format="float8_e4m3fn"),
            "x.bias": torch.ones(2, dtype=torch.bfloat16),
        },
    )

    out = dict(decoded_tensors([path]))

    assert set(out) == {"x.weight", "x.bias"}
    assert out["x.weight"].dtype == torch.bfloat16
    assert torch.equal(out["x.weight"], torch.tensor([[0.5, 1.0], [2.0, -4.0]], dtype=torch.bfloat16))
    assert torch.equal(out["x.bias"], torch.ones(2, dtype=torch.bfloat16))


def test_the_older_scale_weight_name_in_another_shard_is_found_for_fp8(tmp_path: Path) -> None:
    weights = _save(tmp_path / "w.safetensors", {"y.weight": _fp8([[2.0, 4.0]])})
    scales = _save(tmp_path / "s.safetensors", {"y.scale_weight": torch.tensor([0.25])})

    out = dict(decoded_tensors([weights, scales]))

    assert out.keys() == {"y.weight"}
    assert torch.equal(out["y.weight"], torch.tensor([[0.5, 1.0]], dtype=torch.bfloat16))


def test_an_fp8_per_row_scale_scales_each_output_row(tmp_path: Path) -> None:
    path = _save(tmp_path / "r.safetensors", {"z.weight": _fp8([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]), "z.weight_scale": torch.tensor([[2.0], [4.0]])})

    out = dict(decoded_tensors([path]))

    assert torch.equal(out["z.weight"], torch.tensor([[2.0] * 3, [4.0] * 3], dtype=torch.bfloat16))


def test_wanted_filters_by_the_weight_name_and_still_finds_its_scale(tmp_path: Path) -> None:
    path = _save(tmp_path / "f.safetensors", {"p.weight": _fp8([[2.0]]), "p.weight_scale": torch.tensor(2.0), "q.weight": _fp8([[2.0]]), "q.weight_scale": torch.tensor(8.0)})

    out = dict(decoded_tensors([path], lambda name: name.startswith("p.")))

    assert list(out) == ["p.weight"]
    assert out["p.weight"].item() == 4.0


def test_decoded_tensors_refuses_another_quantization_format(tmp_path: Path) -> None:
    path = _save(tmp_path / "i.safetensors", {"x.weight": torch.zeros(2, 2, dtype=torch.uint8), "x.comfy_quant": _record(format="int4")})

    with pytest.raises(ValueError, match="int4"):
        list(decoded_tensors([path]))


def test_a_block_wise_fp8_scale_is_refused(tmp_path: Path) -> None:
    path = _save(tmp_path / "b.safetensors", {"x.weight": _fp8([[1.0] * 4] * 2), "x.weight_scale": torch.ones(2, 2)})

    with pytest.raises(ValueError):
        list(decoded_tensors([path]))


def test_a_scale_on_a_weight_that_is_not_fp8_is_refused(tmp_path: Path) -> None:
    path = _save(tmp_path / "n.safetensors", {"x.weight": torch.ones(2, 2, dtype=torch.bfloat16), "x.weight_scale": torch.tensor(2.0)})

    with pytest.raises(ValueError):
        list(decoded_tensors([path]))
