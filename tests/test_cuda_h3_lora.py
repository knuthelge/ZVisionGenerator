"""Tests for the H3 LoRA module: turbo merge, user-LoRA routing, refiner merge. Synthetic files only, no peft, no models."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
from safetensors.torch import save_file  # noqa: E402

from tests._h3_helpers import lora_factors, write_lora  # noqa: E402

from zvisiongenerator.backends.cuda.video.h3.lora import (  # noqa: E402
    LoraMerger,
    bake_scales,
    load_lora,
    load_pairs,
    merge,
    merge_refiner,
    pairs_from_state,
    peft_state,
    split_user_lora,
)

TURBO_SCALE = 0.0625
HIDDEN = 8
FF = 6
# module name in the original checkpoint -> (output rows, input columns)
SHAPES = {
    "blocks.0.attn.qkv_proj": (3 * HIDDEN, HIDDEN),
    "blocks.0.attn.out_proj": (HIDDEN, HIDDEN),
    "blocks.0.mlp.fc1": (2 * FF, HIDDEN),
    "blocks.0.mlp.fc2": (HIDDEN, FF),
    "blocks.0.adaln_proj.linear": (6 * HIDDEN, HIDDEN),
    "blocks.1.adaln_proj.linear": (6 * HIDDEN, HIDDEN),
    "token_refiner.blocks.0.attn.out_proj": (HIDDEN, HIDDEN),
    "token_refiner.blocks.0.mlp.fc2": (HIDDEN, FF),
    "final_layer.adaln_proj.linear": (2 * HIDDEN, HIDDEN),
}
TIME = {"time_embedder.proj_in": (HIDDEN, 4)}


def _flatten(module: str) -> str:
    return "lora_unet_" + module.replace(".", "_")


def _write_kohya(path: Path, factors: dict[str, tuple[torch.Tensor, torch.Tensor]], alpha: float | None = None) -> Path:
    """Write a musubi/kohya file (flattened names, ``lora_down`` / ``lora_up``, optional per-module ``.alpha``)."""
    tensors = {}
    for module, (down, up) in factors.items():
        base = _flatten(module)
        tensors[f"{base}.lora_down.weight"] = down.contiguous()
        tensors[f"{base}.lora_up.weight"] = up.contiguous()
        if alpha is not None:
            tensors[f"{base}.alpha"] = torch.tensor(float(alpha))
    save_file(tensors, str(path))
    return path


def _qkv_module(index: int = 0) -> str:
    return f"transformer_blocks.{index}.attn.to_q"


def _delta(down: torch.Tensor, up: torch.Tensor, scale: float) -> torch.Tensor:
    return scale * (up.float() @ down.float())


class TestTurboMerge:
    def test_effective_factor_is_exactly_the_config_scale(self, tmp_path: Path) -> None:
        factors = lora_factors({"blocks.0.attn.out_proj": (HIDDEN, HIDDEN)}, 128)
        path = write_lora(tmp_path / "turbo.safetensors", factors, metadata={"alpha": "8"})  # alpha 8 over rank 128 must play no part
        pairs = load_pairs(path)
        down, up = factors["blocks.0.attn.out_proj"]
        weight = torch.randn(HIDDEN, HIDDEN)

        merged = LoraMerger(pairs, TURBO_SCALE).apply("transformer_blocks.0.attn.to_out.0", weight)

        assert torch.equal(merged, weight + _delta(down, up, TURBO_SCALE))
        # With a zero weight the merge is the product itself, scaled by a power of two: the factor is exactly 0.0625.
        product = up.float() @ down.float()
        zero = LoraMerger(pairs, TURBO_SCALE).apply("transformer_blocks.0.attn.to_out.0", torch.zeros(HIDDEN, HIDDEN))
        assert torch.equal(zero, product * 0.0625)

    def test_per_module_alpha_tensors_are_not_folded_into_the_turbolora_factors(self, tmp_path: Path) -> None:
        factors = lora_factors({"blocks.0.attn.out_proj": (HIDDEN, HIDDEN)}, 128)
        path = _write_kohya(tmp_path / "turbo.safetensors", factors, alpha=8)
        pairs = load_pairs(path)
        down, up = pairs["transformer_blocks.0.attn.to_out.0"]
        raw_down, raw_up = factors["blocks.0.attn.out_proj"]
        assert torch.equal(down, raw_down)
        assert torch.equal(up, raw_up)
        merged = LoraMerger(pairs, TURBO_SCALE).apply("transformer_blocks.0.attn.to_out.0", torch.zeros(HIDDEN, HIDDEN))
        assert torch.equal(merged, (raw_up.float() @ raw_down.float()) * 0.0625)

    def test_fused_qkv_and_fc1_are_converted_into_merge_targets(self, tmp_path: Path) -> None:
        factors = lora_factors({k: SHAPES[k] for k in ("blocks.0.attn.qkv_proj", "blocks.0.mlp.fc1")}, 4)
        pairs = load_pairs(write_lora(tmp_path / "turbo.safetensors", factors))
        assert set(pairs) == {
            "transformer_blocks.0.attn.to_q",
            "transformer_blocks.0.attn.to_k",
            "transformer_blocks.0.attn.to_v",
            "transformer_blocks.0.ff.net.0.proj",
        }
        merger = LoraMerger(pairs, TURBO_SCALE)
        for name in ("to_q", "to_k", "to_v"):
            assert merger.apply(f"transformer_blocks.0.attn.{name}", torch.zeros(HIDDEN, HIDDEN)).shape == (HIDDEN, HIDDEN)

    def test_untargeted_module_passes_through_and_unmerged_is_reported(self, tmp_path: Path) -> None:
        pairs = load_pairs(write_lora(tmp_path / "t.safetensors", lora_factors({"blocks.0.attn.out_proj": SHAPES["blocks.0.attn.out_proj"]}, 4)))
        merger = LoraMerger(pairs, TURBO_SCALE)
        other = torch.randn(3, 3)
        assert merger.apply("transformer_blocks.9.attn.to_out.0", other) is other
        assert "transformer_blocks.0.attn.to_out.0" in merger
        assert merger.unmerged() == ["transformer_blocks.0.attn.to_out.0"]
        merger.apply("transformer_blocks.0.attn.to_out.0", torch.zeros(HIDDEN, HIDDEN))
        assert merger.unmerged() == []

    def test_dtype_is_kept_and_maths_run_in_fp32(self) -> None:
        down, up = torch.randn(4, HIDDEN), torch.randn(HIDDEN, 4)
        weight = torch.randn(HIDDEN, HIDDEN).to(torch.bfloat16)
        merged = merge(weight, down.to(torch.bfloat16), up.to(torch.bfloat16), TURBO_SCALE)
        assert merged.dtype == torch.bfloat16
        expected = (weight.float() + TURBO_SCALE * (up.to(torch.bfloat16).float() @ down.to(torch.bfloat16).float())).to(torch.bfloat16)
        assert torch.equal(merged, expected)

    def test_shape_mismatch_names_the_module(self) -> None:
        merger = LoraMerger({"transformer_blocks.0.attn.to_q": (torch.zeros(2, 3), torch.zeros(4, 2))}, TURBO_SCALE)
        with pytest.raises(ValueError, match=r"transformer_blocks\.0\.attn\.to_q.*do not fit"):
            merger.apply("transformer_blocks.0.attn.to_q", torch.zeros(5, 5))

    def test_non_finite_scale_is_refused(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            LoraMerger({}, float("nan"))

    def test_turbo_with_a_timestep_target_is_refused_but_adaln_and_norm_out_merge(self, tmp_path: Path) -> None:
        ok = lora_factors({k: SHAPES[k] for k in ("blocks.0.adaln_proj.linear", "final_layer.adaln_proj.linear")}, 4)
        merger = LoraMerger(load_pairs(write_lora(tmp_path / "ok.safetensors", ok)), TURBO_SCALE)
        assert "transformer_blocks.0.adaln_proj.linear" in merger
        assert "norm_out.linear" in merger
        with pytest.raises(ValueError, match="timestep embedding"):
            LoraMerger({"time_embedder.linear_1": (torch.zeros(2, 2), torch.zeros(2, 2))}, TURBO_SCALE)


class TestLoading:
    def test_peft_and_kohya_files_convert_to_the_same_modules(self, tmp_path: Path) -> None:
        factors = lora_factors({k: SHAPES[k] for k in ("blocks.0.attn.qkv_proj", "blocks.0.mlp.fc2", "blocks.0.adaln_proj.linear")}, 4)
        peft = load_lora(write_lora(tmp_path / "a.safetensors", factors))
        kohya = load_lora(_write_kohya(tmp_path / "b.safetensors", factors))
        assert set(peft.state) == set(kohya.state)
        for key, value in peft.state.items():
            assert torch.equal(value, kohya.state[key])

    def test_metadata_alpha_sets_alpha_over_rank(self, tmp_path: Path) -> None:
        factors = lora_factors({"blocks.0.attn.out_proj": SHAPES["blocks.0.attn.out_proj"]}, 16)
        file = load_lora(write_lora(tmp_path / "a.safetensors", factors, metadata={"alpha": "4"}))
        assert file.scales == {"transformer_blocks.0.attn.to_out.0": 0.25}

    def test_alpha_less_file_has_unit_scale(self, tmp_path: Path) -> None:
        factors = lora_factors({"blocks.0.attn.out_proj": SHAPES["blocks.0.attn.out_proj"]}, 16)
        file = load_lora(write_lora(tmp_path / "a.safetensors", factors))
        assert set(file.scales.values()) == {1.0}

    def test_alpha_tensors_are_folded_into_thelora_factors(self, tmp_path: Path) -> None:
        factors = lora_factors({"blocks.0.attn.out_proj": SHAPES["blocks.0.attn.out_proj"]}, 16)
        file = load_lora(_write_kohya(tmp_path / "a.safetensors", factors, alpha=4))
        down, up = pairs_from_state(file.state)["transformer_blocks.0.attn.to_out.0"]
        raw_down, raw_up = factors["blocks.0.attn.out_proj"]
        assert torch.allclose(up @ down, (raw_up @ raw_down) * 0.25, rtol=1e-5, atol=1e-6)
        assert set(file.scales.values()) == {1.0}

    def test_pairs_from_state_refuses_stray_and_half_pairs(self) -> None:
        with pytest.raises(ValueError, match="not transformer lora_A/lora_B"):
            pairs_from_state({"transformer_ref.x.lora_A.weight": torch.zeros(1, 1)})
        with pytest.raises(ValueError, match="only one of"):
            pairs_from_state({"transformer.x.lora_A.weight": torch.zeros(1, 1)})


class TestSplit:
    def _file(self, tmp_path: Path, *, metadata: dict[str, str] | None = None, extra: dict | None = None):
        factors = lora_factors(SHAPES | (extra or {}), 4)
        return load_lora(write_lora(tmp_path / "user.safetensors", factors, metadata=metadata)), factors

    def test_routes_by_target(self, tmp_path: Path) -> None:
        file, _ = self._file(tmp_path)
        parts = split_user_lora(file.state, 1.0, scales=file.scales)

        assert sorted(parts.adaln) == [0, 1]
        assert sorted(parts.refiner) == [
            "token_refiner.refiner_blocks.0.attn.to_out.0",
            "token_refiner.refiner_blocks.0.ff.net.2",
        ]
        peft_modules = {key.removeprefix("transformer.").removesuffix(".lora_A.weight").removesuffix(".lora_B.weight") for key in parts.peft.state}
        assert peft_modules == {
            "transformer_blocks.0.attn.to_q",
            "transformer_blocks.0.attn.to_k",
            "transformer_blocks.0.attn.to_v",
            "transformer_blocks.0.attn.to_out.0",
            "transformer_blocks.0.ff.net.0.proj",
            "transformer_blocks.0.ff.net.2",
            "norm_out.linear",
        }
        assert set(parts.peft.scales) == peft_modules
        assert parts.refused == ()
        # Nothing is routed twice and nothing is lost.
        routed = len(parts.adaln) + len(parts.refiner) + len(peft_modules)
        assert routed == len(pairs_from_state(file.state))

    def test_adaln_factors_are_the_files_own(self, tmp_path: Path) -> None:
        file, factors = self._file(tmp_path)
        down, up, scale = split_user_lora(file.state, 1.0, scales=file.scales).adaln[1]
        raw_down, raw_up = factors["blocks.1.adaln_proj.linear"]
        assert torch.equal(down, raw_down)
        assert torch.equal(up, raw_up)
        assert scale == 1.0

    def test_user_scale_is_weight_times_alpha_over_rank(self, tmp_path: Path) -> None:
        file, _ = self._file(tmp_path, metadata={"alpha": "1"})  # rank 4 -> alpha/rank = 0.25
        parts = split_user_lora(file.state, 0.5, scales=file.scales)
        assert parts.adaln[0][2] == pytest.approx(0.125)
        assert all(scale == pytest.approx(0.125) for _, _, scale in parts.refiner.values())
        assert parts.peft.weight == 0.5
        assert set(parts.peft.scales.values()) == {0.25}

    def test_timestep_embedding_is_refused(self, tmp_path: Path) -> None:
        file, _ = self._file(tmp_path, extra=TIME)
        with pytest.raises(ValueError, match=r"targets the timestep embedding, which H3's folded AdaLN can't take") as raised:
            split_user_lora(file.state, 1.0)
        assert "time_embedder.linear_1" in str(raised.value)

    def test_non_strict_split_lists_the_refused_modules(self, tmp_path: Path) -> None:
        file, _ = self._file(tmp_path, extra=TIME)
        parts = split_user_lora(file.state, 1.0, strict=False)
        assert parts.refused == ("time_embedder.linear_1",)
        assert not any("time_embedder" in key for key in parts.peft.state)


class TestPeftScales:
    def test_bake_scales_multiplies_lora_b_only_where_needed(self) -> None:
        state = {
            "transformer.a.lora_A.weight": torch.ones(2, 3),
            "transformer.a.lora_B.weight": torch.ones(4, 2),
            "transformer.b.lora_A.weight": torch.ones(2, 3),
            "transformer.b.lora_B.weight": torch.ones(4, 2),
        }
        baked = bake_scales(state, {"a": 0.25, "b": 1.0})
        assert torch.equal(baked["transformer.a.lora_B.weight"], torch.full((4, 2), 0.25))
        assert torch.equal(baked["transformer.a.lora_A.weight"], state["transformer.a.lora_A.weight"])
        assert baked["transformer.b.lora_B.weight"] is state["transformer.b.lora_B.weight"]
        assert torch.equal(state["transformer.a.lora_B.weight"], torch.ones(4, 2))

    def test_peft_state_bakes_alpha_over_rank_but_not_the_weight(self, tmp_path: Path) -> None:
        factors = lora_factors({"blocks.0.attn.out_proj": SHAPES["blocks.0.attn.out_proj"]}, 16)
        file = load_lora(write_lora(tmp_path / "a.safetensors", factors, metadata={"alpha": "4"}))
        part = split_user_lora(file.state, 0.5, scales=file.scales).peft
        baked = peft_state(part)
        raw_down, raw_up = factors["blocks.0.attn.out_proj"]
        assert torch.equal(baked["transformer.transformer_blocks.0.attn.to_out.0.lora_B.weight"], raw_up * 0.25)
        assert part.weight == 0.5


class _Refiner(torch.nn.Module):
    def __init__(self, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.token_refiner = torch.nn.Module()
        self.token_refiner.refiner_blocks = torch.nn.ModuleList([torch.nn.Module()])
        block = self.token_refiner.refiner_blocks[0]
        block.attn = torch.nn.Module()
        block.attn.to_out = torch.nn.ModuleList([torch.nn.Linear(HIDDEN, HIDDEN, bias=False, dtype=dtype)])


class TestRefinerMerge:
    MODULE = "token_refiner.refiner_blocks.0.attn.to_out.0"

    def test_merges_in_place_in_bf16_with_fp32_maths(self) -> None:
        model = _Refiner()
        before = model.get_submodule(self.MODULE).weight.detach().clone()
        down, up = torch.randn(4, HIDDEN), torch.randn(HIDDEN, 4)

        assert merge_refiner(model, {self.MODULE: (down, up, 0.5)}) == [self.MODULE]

        after = model.get_submodule(self.MODULE).weight
        assert after.dtype == torch.bfloat16
        assert torch.equal(after.detach(), (before.float() + 0.5 * (up @ down)).to(torch.bfloat16))

    def test_end_to_end_from_a_user_file(self, tmp_path: Path) -> None:
        factors = lora_factors({"token_refiner.blocks.0.attn.out_proj": (HIDDEN, HIDDEN)}, 4)
        file = load_lora(write_lora(tmp_path / "a.safetensors", factors, metadata={"alpha": "2"}))
        parts = split_user_lora(file.state, 0.5, scales=file.scales)
        model = _Refiner()
        before = model.get_submodule(self.MODULE).weight.detach().clone()

        merge_refiner(model, parts.refiner)

        down, up = factors["token_refiner.blocks.0.attn.out_proj"]
        expected = (before.float() + 0.5 * 0.5 * (up @ down)).to(torch.bfloat16)
        assert torch.equal(model.get_submodule(self.MODULE).weight.detach(), expected)

    def test_unknown_module_quantized_weight_and_shape_mismatch_are_refused(self) -> None:
        model = _Refiner()
        down, up = torch.zeros(4, HIDDEN), torch.zeros(HIDDEN, 4)
        with pytest.raises(ValueError, match="not a module"):
            merge_refiner(model, {"token_refiner.refiner_blocks.7.attn.to_out.0": (down, up, 1.0)})
        with pytest.raises(ValueError, match="do not fit"):
            merge_refiner(model, {self.MODULE: (torch.zeros(4, HIDDEN + 1), up, 1.0)})
        model.get_submodule(self.MODULE).weight = torch.nn.Parameter(torch.zeros(HIDDEN, HIDDEN, dtype=torch.uint8), requires_grad=False)
        with pytest.raises(ValueError, match="floating-point"):
            merge_refiner(model, {self.MODULE: (down, up, 1.0)})
