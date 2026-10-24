"""Tests for the NF4 pre-pack in ``backends/cuda/quant``: real torch on the CPU, a fake ``bitsandbytes``."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")

from zvisiongenerator.backends.cuda import quant  # noqa: E402

_PARTS = ("absmax", "quant_map", "nested_absmax", "nested_quant_map", "quant_state.bitsandbytes__nf4")


class _FakeLinear4bit(torch.nn.Module):
    """Stands in for ``bitsandbytes.nn.Linear4bit``: records how it was built."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args, self.kwargs = args, kwargs


@pytest.fixture
def fake_bitsandbytes(monkeypatch):
    """Enter fake ``bitsandbytes.nn`` (and its parents) with a recording ``Linear4bit`` and ``Params4bit``; return the ``Params4bit`` calls."""
    calls: list[dict] = []

    def from_prequantized(**kwargs):
        calls.append({**kwargs, "quantized_stats": dict(kwargs["quantized_stats"])})
        return torch.nn.Parameter(kwargs["data"], requires_grad=False)

    nn = ModuleType("bitsandbytes.nn")
    nn.Linear4bit = _FakeLinear4bit
    nn.Params4bit = type("Params4bit", (), {"from_prequantized": staticmethod(from_prequantized)})
    package = ModuleType("bitsandbytes")
    package.nn = nn
    monkeypatch.setitem(sys.modules, "bitsandbytes", package)
    monkeypatch.setitem(sys.modules, "bitsandbytes.nn", nn)
    return calls


def _model(*names: str) -> torch.nn.Module:
    """Return a module tree with a ``Linear(4, 4)`` at each dotted name in *names*."""
    root = torch.nn.Module()
    for name in names:
        parent = root
        *path, leaf = name.split(".")
        for part in path:
            if not hasattr(parent, part):
                setattr(parent, part, torch.nn.Module())
            parent = getattr(parent, part)
        setattr(parent, leaf, torch.nn.Linear(4, 4))
    return root


class TestNf4LinearNames:
    def test_only_linears_under_the_prefix_are_named_in_module_order(self):
        model = _model("blocks.0.attn", "blocks.1.attn", "proj_in", "other.0.attn")

        assert quant.nf4_linear_names(model, prefix="blocks.") == ["blocks.0.attn", "blocks.1.attn"]
        assert quant.nf4_linear_names(model, prefix="other.") == ["other.0.attn"]
        assert quant.nf4_linear_names(model, prefix="missing.") == []

    def test_a_layer_with_an_excluded_name_part_is_left_out_but_a_longer_name_is_kept(self):
        model = _model("transformer_blocks.0.attn", "transformer_blocks.0.adaln_proj.linear", "transformer_blocks.0.adaln_projection", "transformer_blocks.1.attn")

        names = quant.nf4_linear_names(model, prefix="transformer_blocks.", exclude={"adaln_proj"})

        assert names == ["transformer_blocks.0.attn", "transformer_blocks.0.adaln_projection", "transformer_blocks.1.attn"]
        assert quant.nf4_linear_names(model, prefix="transformer_blocks.", exclude=()) == [
            "transformer_blocks.0.attn",
            "transformer_blocks.0.adaln_proj.linear",
            "transformer_blocks.0.adaln_projection",
            "transformer_blocks.1.attn",
        ]

    def test_a_non_linear_layer_is_never_named(self):
        model = torch.nn.Module()
        model.blocks = torch.nn.Module()
        model.blocks.norm = torch.nn.LayerNorm(4)

        assert quant.nf4_linear_names(model, prefix="blocks.") == []


class TestSwapInLinear4bit:
    def test_exactly_the_given_names_are_replaced_and_returned_by_weight_name(self, fake_bitsandbytes):
        model = _model("blocks.a", "blocks.b", "blocks.c")
        model.blocks.b.bias = None

        layers = quant.swap_in_linear4bit(model, ["blocks.a", "blocks.b"], torch.bfloat16)

        assert list(layers) == ["blocks.a.weight", "blocks.b.weight"]
        assert model.blocks.a is layers["blocks.a.weight"]
        assert model.blocks.b is layers["blocks.b.weight"]
        assert type(model.blocks.c) is torch.nn.Linear
        assert layers["blocks.a.weight"].args == (4, 4)
        assert layers["blocks.a.weight"].kwargs == {"bias": True, "compute_dtype": torch.bfloat16, "compress_statistics": True, "quant_type": "nf4", "quant_storage": torch.uint8, "device": "meta"}
        assert layers["blocks.b.weight"].kwargs["bias"] is False


class TestPlaceNf4:
    def _parts(self, owner):
        return {f"{owner}.{part}": torch.full((1,), index, dtype=torch.uint8) for index, part in enumerate(_PARTS)}

    def test_other_tensors_are_yielded_in_order_and_interleaved_parts_are_gathered(self, fake_bitsandbytes):
        layers = {"blocks.a.weight": _FakeLinear4bit(), "blocks.b.weight": _FakeLinear4bit()}
        data_a, data_b = torch.tensor([1], dtype=torch.uint8), torch.tensor([2], dtype=torch.uint8)
        parts_a, parts_b = self._parts("blocks.a.weight"), self._parts("blocks.b.weight")
        plain = [("first", torch.zeros(1)), ("blocks.a.bias", torch.zeros(2)), ("last", torch.zeros(3))]
        stream = [
            plain[0],
            ("blocks.a.weight", data_a),
            *list(parts_a.items())[:2],
            ("blocks.b.weight", data_b),
            *list(parts_b.items())[:3],
            plain[1],
            *list(parts_a.items())[2:],
            *list(parts_b.items())[3:],
            plain[2],
        ]

        yielded = list(quant.place_nf4(stream, layers))

        assert [name for name, _ in yielded] == ["first", "blocks.a.bias", "last"]
        assert [call["module"] for call in fake_bitsandbytes] == [layers["blocks.a.weight"], layers["blocks.b.weight"]]
        for call, data, parts, owner in zip(fake_bitsandbytes, (data_a, data_b), (parts_a, parts_b), layers, strict=True):
            assert (call["requires_grad"], call["device"]) == (False, "cpu")
            assert call["data"] is data
            assert call["quantized_stats"].keys() == set(_PARTS)
            assert all(torch.equal(call["quantized_stats"][part], parts[f"{owner}.{part}"]) for part in _PARTS)
            assert torch.equal(layers[owner].weight, data)

    def test_a_weight_with_a_missing_part_is_refused(self, fake_bitsandbytes):
        layers = {"blocks.a.weight": _FakeLinear4bit()}
        parts = self._parts("blocks.a.weight")
        del parts["blocks.a.weight.quant_map"]

        with pytest.raises(RuntimeError, match="NF4 weight blocks.a.weight is incomplete in the prepared copy"):
            list(quant.place_nf4([("blocks.a.weight", torch.zeros(1, dtype=torch.uint8)), *parts.items()], layers))


class TestNf4StorageFormat:
    def test_the_format_names_bitsandbytes_major_and_minor(self, monkeypatch):
        monkeypatch.setattr("importlib.metadata.version", lambda name: "0.99.0" if name == "bitsandbytes" else pytest.fail(name))

        assert quant.nf4_storage_format() == "nf4-bitsandbytes-0.99"
