"""Tests for LyCORIS LoKr adapters on the diffusers backend, with small CPU tensors and a stand-in pipeline."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from zvisiongenerator.backends import image_win_lokr as lokr  # noqa: E402


def _converter(state):
    """Mimic a diffusers LoRA converter: rename ``proj`` and split fused ``qkv`` rows into q, k and v."""
    converted = {}
    for key, value in state.items():
        layer, tensor = key.removeprefix("diffusion_model.").rsplit(".", 2)[0], ".".join(key.rsplit(".", 2)[1:])
        if layer.endswith(".qkv"):
            parts = torch.chunk(value, 3, dim=0) if tensor == "lora_B.weight" else (value, value, value)
            for name, part in zip(("to_q", "to_k", "to_v"), parts, strict=True):
                converted[f"transformer.{layer.removesuffix('.qkv')}.{name}.{tensor}"] = part
        else:
            converted[f"transformer.{layer.replace('proj', 'to_out')}.{tensor}"] = value
    return converted


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("to_q", "to_k", "to_v", "to_out"):
            setattr(self, name, torch.nn.Linear(6, 4, bias=False))


class _Pipeline:
    lora_state_dict = staticmethod(_converter)

    def __init__(self):
        self.transformer = torch.nn.Module()
        self.transformer.blocks = torch.nn.ModuleList([_Block()])


@pytest.fixture(autouse=True)
def _seeded():
    """Make every random input repeatable, so bfloat16 tolerances never meet an unlucky draw."""
    torch.manual_seed(0)


def _factors(seed: int, w1_shape=(2, 3), w2_shape=(2, 2)):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*w1_shape, generator=generator), torch.randn(*w2_shape, generator=generator)


class TestSplitLoraTensors:
    def test_lokr_layers_are_split_from_the_rest(self):
        t = torch.zeros(1)
        state = {
            "diffusion_model.a.lora_A.weight": t,
            "diffusion_model.a.lora_B.weight": t,
            "diffusion_model.a.alpha": t,
            "diffusion_model.b.lokr_w1": t,
            "diffusion_model.b.lokr_w2": t,
            "diffusion_model.b.alpha": t,
            "diffusion_model.c.diff": t,
        }

        parts = lokr.split_lora_tensors(state)

        assert set(parts.lokr) == {"diffusion_model.b"}
        assert set(parts.lokr["diffusion_model.b"]) == {"lokr_w1", "lokr_w2", "alpha"}
        assert set(parts.rest) == {"diffusion_model.a.lora_A.weight", "diffusion_model.a.lora_B.weight", "diffusion_model.a.alpha", "diffusion_model.c.diff"}
        assert set(parts.lora) == {"diffusion_model.a.lora_A.weight", "diffusion_model.a.lora_B.weight", "diffusion_model.a.alpha"}
        assert parts.unsupported == ("diffusion_model.c.diff",)

    def test_unsupported_tensors_next_to_a_lora_stay_in_the_rest(self):
        t = torch.zeros(1)

        parts = lokr.split_lora_tensors({"x.lora_A.weight": t, "x.lora_B.weight": t, "x.diff_b": t})

        assert set(parts.rest) == {"x.lora_A.weight", "x.lora_B.weight", "x.diff_b"}
        assert parts.unsupported == ("x.diff_b",)

    @pytest.mark.parametrize(("keys", "expected"), [(["a.lokr_w1", "a.lokr_w2"], True), (["a.lora_A.weight", "a.lora_B.weight", "a.alpha"], False), (["a.lokr_w2_a"], True)])
    def test_has_lokr_tensors(self, keys, expected):
        assert lokr.has_lokr_tensors(keys) is expected


class TestLokrFactors:
    def test_full_factors_ignore_alpha(self):
        w1, w2 = _factors(0)
        assert lokr._lokr_factors({"lokr_w1": w1, "lokr_w2": w2, "alpha": torch.tensor(8.0)})[2] == 1.0

    def test_decomposed_factor_scales_by_alpha_over_rank(self):
        w1, _ = _factors(0)
        w2_a, w2_b = torch.ones(2, 4), torch.ones(4, 2)
        _w1, w2, scale = lokr._lokr_factors({"lokr_w1": w1, "lokr_w2_a": w2_a, "lokr_w2_b": w2_b, "alpha": torch.tensor(2.0)})
        assert scale == 0.5
        assert torch.equal(w2, w2_a @ w2_b)

    def test_tucker_and_dora_layers_are_rejected(self):
        w1, w2 = _factors(0)
        with pytest.raises(ValueError):
            lokr._lokr_factors({"lokr_w1": w1, "lokr_w2": w2, "dora_scale": torch.ones(4)})


class TestLokrTerm:
    def test_matches_the_kronecker_product(self):
        w1, w2 = _factors(1)
        x = torch.randn(5, 6)

        out = lokr._LoKrTerm(w1, w2, 0.5)(x)

        assert torch.allclose(out, 0.5 * x @ torch.kron(w1, w2).T, atol=1e-5)


class TestApplyLokr:
    def test_adds_the_delta_to_a_mapped_layer_and_keeps_its_weight(self):
        pipeline = _Pipeline()
        base = pipeline.transformer.blocks[0].to_out
        weight_before = base.weight.clone()
        w1, w2 = _factors(2)
        x = torch.randn(3, 6)
        expected = base(x) + 0.8 * x @ torch.kron(w1, w2).T

        lokr.apply_lokr(pipeline, {"diffusion_model.blocks.0.proj": {"lokr_w1": w1, "lokr_w2": w2}}, 0.8, pin_memory=False)

        adapted = pipeline.transformer.blocks[0].to_out
        assert torch.equal(adapted.weight, weight_before)
        assert torch.allclose(adapted(x), expected, atol=2e-2)  # adapter terms compute in bfloat16

    def test_fused_layer_gives_each_split_target_its_rows(self):
        pipeline = _Pipeline()
        block = pipeline.transformer.blocks[0]
        w1, w2 = _factors(3, w1_shape=(6, 3))  # 12 output rows: q, k and v get 4 each
        delta = torch.kron(w1, w2)
        x = torch.randn(2, 6)
        expected = {name: getattr(block, name)(x) + x @ delta[i * 4 : (i + 1) * 4].T for i, name in enumerate(("to_q", "to_k", "to_v"))}

        lokr.apply_lokr(pipeline, {"diffusion_model.blocks.0.qkv": {"lokr_w1": w1, "lokr_w2": w2}}, 1.0, pin_memory=False)

        for name, value in expected.items():
            assert torch.allclose(getattr(block, name)(x), value, rtol=2e-2, atol=5e-2), name  # bfloat16 factors: about 0.5% error

    def test_two_lokr_files_on_one_layer_add_up(self):
        pipeline = _Pipeline()
        base = pipeline.transformer.blocks[0].to_out
        (a1, a2), (b1, b2) = _factors(4), _factors(5)
        x = torch.randn(1, 6)
        expected = base(x) + x @ torch.kron(a1, a2).T + 0.5 * x @ torch.kron(b1, b2).T

        lokr.apply_lokr(pipeline, {"diffusion_model.blocks.0.proj": {"lokr_w1": a1, "lokr_w2": a2}}, 1.0, pin_memory=False)
        lokr.apply_lokr(pipeline, {"diffusion_model.blocks.0.proj": {"lokr_w1": b1, "lokr_w2": b2}}, 0.5, pin_memory=False)

        adapted = pipeline.transformer.blocks[0].to_out
        assert len(adapted.terms) == 2
        assert torch.allclose(adapted(x), expected, atol=5e-2)

    def test_adapted_layers_hold_no_adapter_parameters_or_buffers(self):
        """Model offloading tracks a layer's parameters and buffers; adapter tensors must stay outside them."""
        pipeline = _Pipeline()
        tracked_before = {name for name, _ in pipeline.transformer.named_parameters()} | {name for name, _ in pipeline.transformer.named_buffers()}
        w1, w2 = _factors(7)

        lokr.apply_lokr(pipeline, {"diffusion_model.blocks.0.proj": {"lokr_w1": w1, "lokr_w2": w2}}, 1.0, pin_memory=False)

        tracked_after = {name.replace(".base.", ".") for name, _ in pipeline.transformer.named_parameters()}
        tracked_after |= {name.replace(".base.", ".") for name, _ in pipeline.transformer.named_buffers()}
        assert tracked_after == tracked_before

    def test_layers_the_converter_rejects_are_skipped_and_the_rest_applied(self):
        def strict_converter(state):
            if any("text_encoder_layer" in key for key in state):
                raise ValueError("unknown layer")
            return _converter(state)

        class _Strict(_Pipeline):
            lora_state_dict = staticmethod(strict_converter)

        pipeline = _Strict()
        (a1, a2), (b1, b2) = _factors(8), _factors(9)

        skipped = lokr.apply_lokr(
            pipeline,
            {"diffusion_model.blocks.0.proj": {"lokr_w1": a1, "lokr_w2": a2}, "diffusion_model.text_encoder_layer": {"lokr_w1": b1, "lokr_w2": b2}},
            1.0,
            pin_memory=False,
        )

        assert skipped == ("diffusion_model.text_encoder_layer",)
        assert len(pipeline.transformer.blocks[0].to_out.terms) == 1

    def test_factors_that_do_not_fit_the_layer_raise_before_anything_is_applied(self):
        pipeline = _Pipeline()
        good1, good2 = _factors(10)
        wide1, wide2 = _factors(11, w1_shape=(2, 4))  # 4 × 8: the layer is 4 × 6

        with pytest.raises(ValueError, match="different model"):
            lokr.apply_lokr(
                pipeline,
                {"diffusion_model.blocks.0.qkv": {"lokr_w1": torch.cat([good1] * 3), "lokr_w2": good2}, "diffusion_model.blocks.0.proj": {"lokr_w1": wide1, "lokr_w2": wide2}},
                1.0,
                pin_memory=False,
            )

        assert isinstance(pipeline.transformer.blocks[0].to_q, torch.nn.Linear)  # nothing was wrapped

    def test_kohya_named_layers_reach_the_kohya_converter(self):
        """diffusers converts kohya names only from lora_down/lora_up keys, as a kohya file holds them."""

        def kohya_only(state):
            return {
                key.replace("lora_unet_blocks_0_proj.lora_up.weight", "transformer.blocks.0.to_out.lora_B.weight").replace(
                    "lora_unet_blocks_0_proj.lora_down.weight", "transformer.blocks.0.to_out.lora_A.weight"
                ): value
                for key, value in state.items()
                if ".lora_up." in key or ".lora_down." in key
            }

        class _Kohya(_Pipeline):
            lora_state_dict = staticmethod(kohya_only)

        pipeline = _Kohya()
        w1, w2 = _factors(13)

        skipped = lokr.apply_lokr(pipeline, {"lora_unet_blocks_0_proj": {"lokr_w1": w1, "lokr_w2": w2}}, 1.0, pin_memory=False)

        assert skipped == ()
        assert len(pipeline.transformer.blocks[0].to_out.terms) == 1

    def test_a_lora_for_another_model_skips_every_layer(self):
        class _NothingMaps(_Pipeline):
            lora_state_dict = staticmethod(lambda state: {})

        pipeline = _NothingMaps()
        w1, w2 = _factors(6)

        skipped = lokr.apply_lokr(pipeline, {"diffusion_model.blocks.0.proj": {"lokr_w1": w1, "lokr_w2": w2}}, 1.0, pin_memory=False)

        assert skipped == ("diffusion_model.blocks.0.proj",)
        assert isinstance(pipeline.transformer.blocks[0].to_out, torch.nn.Linear)

    def test_a_target_that_is_not_a_linear_layer_raises(self):
        class _NormTarget(_Pipeline):
            lora_state_dict = staticmethod(lambda state: {key.replace("diffusion_model.norm", "transformer.norm"): value for key, value in state.items()})

        pipeline = _NormTarget()
        pipeline.transformer.norm = torch.nn.LayerNorm(6)
        w1, w2 = _factors(12)

        with pytest.raises(ValueError, match="not a linear layer"):
            lokr.apply_lokr(pipeline, {"diffusion_model.norm": {"lokr_w1": w1, "lokr_w2": w2}}, 1.0, pin_memory=False)
