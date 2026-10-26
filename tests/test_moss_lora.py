"""Tests for unmerged LoRA and LoKr adapters on the MOSS transformer, with toy CPU modules and synthetic tensors."""

from __future__ import annotations

import copy
import json
import warnings

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from zvisiongenerator.backends.cuda.image import lokr  # noqa: E402
from zvisiongenerator.backends.cuda.image.moss_lora import LoraFile, install_loras, moss_targets  # noqa: E402

DIM = 4
FFN = 6
BLOCKS = 2


class _Proj(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(DIM, DIM, bias=False)
        self.k = torch.nn.Linear(DIM, DIM, bias=False)
        self.v = torch.nn.Linear(DIM, DIM, bias=False)
        self.o = torch.nn.Linear(DIM, DIM, bias=False)
        self.ffn_up_gate = torch.nn.Linear(DIM, 2 * FFN, bias=False)
        self.ffn_down = torch.nn.Linear(FFN, DIM, bias=False)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = _Proj()


class _Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Block() for _ in range(BLOCKS)])
        self.text_embedder = torch.nn.Module()
        self.text_embedder.blocks = torch.nn.ModuleList([torch.nn.Module()])
        self.text_embedder.blocks[0].qkv = torch.nn.Linear(DIM, 3 * DIM, bias=False)
        self.final_linear_video = torch.nn.Linear(DIM, 3, bias=False)


@pytest.fixture(autouse=True)
def _seeded():
    torch.manual_seed(0)


@pytest.fixture
def model():
    return _Transformer()


def _lora(seed: int, out: int, inn: int, rank: int = 4):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rank, inn, generator=generator), torch.randn(out, rank, generator=generator)


def _dyadic(seed: int, *shape: int):
    """Return values that bfloat16 holds exactly, so LoKr results can be compared at fp32 precision."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-2, 3, shape, generator=generator).float() / 2


def _save(tmp_path, state, metadata=None, name="adapter.safetensors"):
    path = str(tmp_path / name)
    safetensors_torch.save_file({k: v.clone().contiguous() for k, v in state.items()}, path, metadata=metadata)
    return path


def _install(model, path, strength=1.0, dtype=torch.float32):
    install_loras(model, [LoraFile(path, strength, dtype)], device="cpu")


def _no_warnings():
    context = warnings.catch_warnings()
    context.__enter__()
    warnings.simplefilter("error")
    return context


def _assert_close(actual, expected):
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def _delta(down, up, x):
    return (x @ down.T) @ up.T


class TestMossTargets:
    NAMES = {
        "layers_0_proj_q": "layers.0.proj.q",
        "layers_0_proj_qkv": "layers.0.proj.qkv",
        "text_embedder_blocks_0_qkv": "text_embedder.blocks.0.qkv",
        "final_linear_video": "final_linear_video",
    }

    @pytest.mark.parametrize("prefix", ["diffusion_model.", "transformer.", "model.diffusion_model.", "base_model.model."])
    def test_dotted_names_are_their_own_module_path(self, prefix):
        assert moss_targets(f"{prefix}layers.0.proj.q", self.NAMES, DIM) == (("layers.0.proj.q", None),)

    def test_underscore_names_are_looked_up(self):
        assert moss_targets("lora_unet_layers_0_proj_q", self.NAMES, DIM) == (("layers.0.proj.q", None),)
        assert moss_targets("lora_unet_text_embedder_blocks_0_qkv", self.NAMES, DIM) == (("text_embedder.blocks.0.qkv", None),)
        assert moss_targets("lora_unet_final_linear_video", self.NAMES, DIM) == (("final_linear_video", None),)

    @pytest.mark.parametrize("layer", ["diffusion_model.layers.0.proj.qkv", "lora_unet_layers_0_proj_qkv"])
    def test_fused_qkv_is_split_into_thirds_in_q_k_v_order(self, layer):
        assert moss_targets(layer, self.NAMES, DIM) == (
            ("layers.0.proj.q", slice(0, DIM)),
            ("layers.0.proj.k", slice(DIM, 2 * DIM)),
            ("layers.0.proj.v", slice(2 * DIM, 3 * DIM)),
        )

    @pytest.mark.parametrize("layer", ["diffusion_model.layers.9.proj.q", "lora_unet_layers_9_proj_q", "lora_unet_nothing", "unknown.layer"])
    def test_unknown_layers_have_no_targets(self, layer):
        assert moss_targets(layer, self.NAMES, DIM) == ()


class TestLoraFormats:
    def test_l1_kohya_dotted_with_alpha(self, model, tmp_path):
        reference = copy.deepcopy(model)
        down, up = _lora(1, DIM, DIM)
        path = _save(
            tmp_path,
            {
                "diffusion_model.layers.1.proj.q.lora_down.weight": down,
                "diffusion_model.layers.1.proj.q.lora_up.weight": up,
                "diffusion_model.layers.1.proj.q.alpha": torch.tensor(2.0),
            },
        )
        x = torch.randn(5, DIM)

        _install(model, path, strength=0.5)

        scale = (2.0 / 4) * 0.5
        _assert_close(model.layers[1].proj.q(x), reference.layers[1].proj.q(x) + scale * _delta(down, up, x))
        assert not isinstance(model.layers[0].proj.q, lokr.AdaptedLinear)

    def test_l2_kohya_underscore_covers_every_layer_kind(self, model, tmp_path):
        reference = copy.deepcopy(model)
        shapes = {
            "layers_0_proj_o": (DIM, DIM, "layers.0.proj.o"),
            "layers_1_proj_ffn_up_gate": (2 * FFN, DIM, "layers.1.proj.ffn_up_gate"),
            "layers_1_proj_ffn_down": (DIM, FFN, "layers.1.proj.ffn_down"),
            "text_embedder_blocks_0_qkv": (3 * DIM, DIM, "text_embedder.blocks.0.qkv"),
            "final_linear_video": (3, DIM, "final_linear_video"),
        }
        state, factors = {}, {}
        for seed, (name, (out, inn, _)) in enumerate(shapes.items()):
            factors[name] = _lora(seed, out, inn, rank=2)
            state[f"lora_unet_{name}.lora_down.weight"] = factors[name][0]
            state[f"lora_unet_{name}.lora_up.weight"] = factors[name][1]
            state[f"lora_unet_{name}.alpha"] = torch.tensor(1.0)
        path = _save(tmp_path, state)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _install(model, path)

        for name, (_, inn, module_path) in shapes.items():
            x = torch.randn(3, inn)
            down, up = factors[name]
            _assert_close(model.get_submodule(module_path)(x), reference.get_submodule(module_path)(x) + 0.5 * _delta(down, up, x))

    def test_l3_lora_a_b_without_alpha_has_scale_one(self, model, tmp_path):
        reference = copy.deepcopy(model)
        down, up = _lora(2, DIM, DIM)
        path = _save(tmp_path, {"diffusion_model.layers.0.proj.o.lora_A.weight": down, "diffusion_model.layers.0.proj.o.lora_B.weight": up})
        x = torch.randn(5, DIM)

        _install(model, path, strength=0.75)

        _assert_close(model.layers[0].proj.o(x), reference.layers[0].proj.o(x) + 0.75 * _delta(down, up, x))

    @pytest.mark.parametrize("prefix", ["base_model.model.", "model.diffusion_model."])
    def test_trainer_prefixes_apply_with_nothing_skipped(self, model, tmp_path, prefix):
        reference = copy.deepcopy(model)
        down, up = _lora(7, DIM, DIM)
        path = _save(tmp_path, {f"{prefix}layers.0.proj.q.lora_A.weight": down, f"{prefix}layers.0.proj.q.lora_B.weight": up})
        x = torch.randn(5, DIM)

        context = _no_warnings()
        try:
            _install(model, path)
        finally:
            context.__exit__(None, None, None)

        _assert_close(model.layers[0].proj.q(x), reference.layers[0].proj.q(x) + _delta(down, up, x))

    def test_l3_fused_qkv_gives_each_target_its_rows(self, model, tmp_path):
        reference = copy.deepcopy(model)
        down, up = _lora(3, 3 * DIM, DIM)
        path = _save(tmp_path, {"diffusion_model.layers.0.proj.qkv.lora_A.weight": down, "diffusion_model.layers.0.proj.qkv.lora_B.weight": up})
        x = torch.randn(5, DIM)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _install(model, path)

        for i, name in enumerate(("q", "k", "v")):
            rows = up[i * DIM : (i + 1) * DIM]
            _assert_close(getattr(model.layers[0].proj, name)(x), getattr(reference.layers[0].proj, name)(x) + _delta(down, rows, x))

    def test_fused_qkv_equals_separate_q_k_v(self, model, tmp_path):
        separate = copy.deepcopy(model)
        down, up = _lora(4, 3 * DIM, DIM)
        fused_path = _save(tmp_path, {"lora_unet_layers_0_proj_qkv.lora_down.weight": down, "lora_unet_layers_0_proj_qkv.lora_up.weight": up}, name="fused.safetensors")
        state = {}
        for i, name in enumerate(("q", "k", "v")):
            state[f"diffusion_model.layers.0.proj.{name}.lora_A.weight"] = down
            state[f"diffusion_model.layers.0.proj.{name}.lora_B.weight"] = up[i * DIM : (i + 1) * DIM]
        separate_path = _save(tmp_path, state, name="separate.safetensors")
        x = torch.randn(5, DIM)

        _install(model, fused_path)
        _install(separate, separate_path)

        for name in ("q", "k", "v"):
            _assert_close(getattr(model.layers[0].proj, name)(x), getattr(separate.layers[0].proj, name)(x))

    def test_l4_peft_takes_alpha_from_the_file_metadata(self, model, tmp_path):
        reference = copy.deepcopy(model)
        down, up = _lora(5, DIM, DIM, rank=4)
        path = _save(
            tmp_path,
            {"transformer.layers.0.proj.k.lora_A.weight": down, "transformer.layers.0.proj.k.lora_B.weight": up},
            metadata={"lora_adapter_metadata": json.dumps({"r": 4, "lora_alpha": 2})},
        )
        x = torch.randn(5, DIM)

        _install(model, path)

        _assert_close(model.layers[0].proj.k(x), reference.layers[0].proj.k(x) + 0.5 * _delta(down, up, x))

    def test_l4_metadata_keys_may_carry_a_component_prefix(self, model, tmp_path):
        reference = copy.deepcopy(model)
        down, up = _lora(6, DIM, DIM, rank=4)
        path = _save(
            tmp_path,
            {"transformer.layers.0.proj.k.lora_A.weight": down, "transformer.layers.0.proj.k.lora_B.weight": up},
            metadata={"lora_adapter_metadata": json.dumps({"transformer.r": 4, "transformer.lora_alpha": 8})},
        )
        x = torch.randn(5, DIM)

        _install(model, path)

        _assert_close(model.layers[0].proj.k(x), reference.layers[0].proj.k(x) + 2.0 * _delta(down, up, x))

    def test_l5_lokr_fused_qkv_with_orphan_alphas_is_silent(self, model, tmp_path):
        reference = copy.deepcopy(model)
        w1, w2 = _dyadic(1, 3, 2), _dyadic(2, 4, 2)  # kron is 12 x 4: q, k and v rows
        o1, o2 = _dyadic(3, 2, 2), _dyadic(4, 2, 2)
        path = _save(
            tmp_path,
            {
                "diffusion_model.layers.0.proj.qkv.lokr_w1": w1,
                "diffusion_model.layers.0.proj.qkv.lokr_w2": w2,
                "diffusion_model.layers.0.proj.qkv.alpha": torch.tensor(1.0),
                "diffusion_model.layers.0.proj.q.alpha": torch.tensor(1.0),
                "diffusion_model.layers.0.proj.k.alpha": torch.tensor(1.0),
                "diffusion_model.layers.0.proj.v.alpha": torch.tensor(1.0),
                "diffusion_model.layers.1.proj.o.lokr_w1": o1,
                "diffusion_model.layers.1.proj.o.lokr_w2": o2,
            },
        )
        x = _dyadic(5, 5, DIM)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _install(model, path, strength=0.5)

        delta = torch.kron(w1, w2)
        for i, name in enumerate(("q", "k", "v")):
            expected = getattr(reference.layers[0].proj, name)(x) + 0.5 * x @ delta[i * DIM : (i + 1) * DIM].T
            _assert_close(getattr(model.layers[0].proj, name)(x), expected)
        _assert_close(model.layers[1].proj.o(x), reference.layers[1].proj.o(x) + 0.5 * x @ torch.kron(o1, o2).T)

    def test_a_layer_wrapped_twice_keeps_one_wrapper(self, model, tmp_path):
        down, up = _lora(7, DIM, DIM)
        state = {"diffusion_model.layers.0.proj.o.lora_A.weight": down, "diffusion_model.layers.0.proj.o.lora_B.weight": up}
        first, second = _save(tmp_path, state, name="a.safetensors"), _save(tmp_path, state, name="b.safetensors")

        install_loras(model, [LoraFile(first, 1.0, torch.float32), LoraFile(second, 1.0, torch.float32)], device="cpu")

        wrapper = model.layers[0].proj.o
        assert isinstance(wrapper, lokr.AdaptedLinear) and not isinstance(wrapper.base, lokr.AdaptedLinear)
        assert len(wrapper.terms) == 2


class TestSkippedAndInvalid:
    def test_loha_and_diff_layers_warn_once_and_the_rest_applies(self, model, tmp_path):
        reference = copy.deepcopy(model)
        down, up = _lora(8, DIM, DIM)
        t = torch.zeros(2, 2)
        path = _save(
            tmp_path,
            {
                "diffusion_model.layers.0.proj.o.lora_A.weight": down,
                "diffusion_model.layers.0.proj.o.lora_B.weight": up,
                "diffusion_model.layers.0.proj.q.hada_w1_a": t,
                "diffusion_model.layers.0.proj.q.hada_w1_b": t,
                "diffusion_model.layers.0.proj.q.hada_w2_a": t,
                "diffusion_model.layers.0.proj.q.hada_w2_b": t,
                "diffusion_model.layers.0.proj.q.alpha": torch.tensor(1.0),
                "diffusion_model.final_norm.diff": torch.zeros(DIM),
            },
        )
        x = torch.randn(5, DIM)

        with pytest.warns(UserWarning, match="skipped") as caught:
            _install(model, path)

        assert len(caught) == 1
        _assert_close(model.layers[0].proj.o(x), reference.layers[0].proj.o(x) + _delta(down, up, x))
        assert not isinstance(model.layers[0].proj.q, lokr.AdaptedLinear)

    def test_alpha_only_layers_are_ignored_silently(self, model, tmp_path):
        down, up = _lora(9, DIM, DIM)
        path = _save(
            tmp_path,
            {
                "diffusion_model.layers.0.proj.o.lora_A.weight": down,
                "diffusion_model.layers.0.proj.o.lora_B.weight": up,
                "diffusion_model.layers.0.proj.q.alpha": torch.tensor(1.0),
            },
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _install(model, path)

    def test_unknown_layers_are_counted_in_one_warning_and_known_ones_apply(self, model, tmp_path):
        down, up = _lora(10, DIM, DIM)
        state = {"diffusion_model.layers.0.proj.o.lora_A.weight": down, "diffusion_model.layers.0.proj.o.lora_B.weight": up}
        for name in ("a", "b"):
            state[f"diffusion_model.extra.{name}.lora_A.weight"] = down
            state[f"diffusion_model.extra.{name}.lora_B.weight"] = up
        path = _save(tmp_path, state)

        with pytest.warns(UserWarning, match="2 layer") as caught:
            _install(model, path)

        assert len(caught) == 1
        assert isinstance(model.layers[0].proj.o, lokr.AdaptedLinear)

    def test_a_file_matching_nothing_warns(self, model, tmp_path):
        down, up = _lora(11, DIM, DIM)
        path = _save(tmp_path, {"diffusion_model.other.layer.lora_A.weight": down, "diffusion_model.other.layer.lora_B.weight": up})

        with pytest.warns(UserWarning, match="none of its LoRA tensors match"):
            _install(model, path)

        assert not any(isinstance(m, lokr.AdaptedLinear) for m in model.modules())

    @pytest.mark.parametrize("variant", ["lokr_t2", "dora_scale"])
    def test_tucker_and_dora_layers_are_skipped(self, model, tmp_path, variant):
        path = _save(
            tmp_path,
            {
                "diffusion_model.layers.0.proj.o.lokr_w1": _dyadic(1, 2, 2),
                "diffusion_model.layers.0.proj.o.lokr_w2": _dyadic(2, 2, 2),
                f"diffusion_model.layers.0.proj.o.{variant}": torch.ones(2),
            },
        )

        with pytest.warns(UserWarning, match="skipped"):
            _install(model, path)

        assert not isinstance(model.layers[0].proj.o, lokr.AdaptedLinear)

    def test_a_misfit_raises_before_anything_is_applied(self, model, tmp_path):
        good_down, good_up = _lora(12, DIM, DIM)
        bad_down, bad_up = _lora(13, DIM, DIM + 1)
        path = _save(
            tmp_path,
            {
                "diffusion_model.layers.0.proj.o.lora_A.weight": good_down,
                "diffusion_model.layers.0.proj.o.lora_B.weight": good_up,
                "diffusion_model.layers.1.proj.o.lora_A.weight": bad_down,
                "diffusion_model.layers.1.proj.o.lora_B.weight": bad_up,
            },
        )

        with pytest.raises(ValueError, match="does not fit layers.1.proj.o"):
            _install(model, path)

        assert not any(isinstance(m, lokr.AdaptedLinear) for m in model.modules())

    def test_a_fused_layer_of_the_wrong_height_raises(self, model, tmp_path):
        down, up = _lora(14, 2 * DIM, DIM)
        path = _save(tmp_path, {"diffusion_model.layers.0.proj.qkv.lora_A.weight": down, "diffusion_model.layers.0.proj.qkv.lora_B.weight": up})

        with pytest.raises(ValueError, match="does not fit"):
            _install(model, path)


class TestChunking:
    @pytest.mark.parametrize("shape", [(10_000, DIM), (2, 5_000, DIM)])
    def test_ten_thousand_tokens_match_the_direct_product(self, model, tmp_path, shape):
        reference = copy.deepcopy(model)
        down, up = _lora(15, DIM, DIM)
        path = _save(tmp_path, {"diffusion_model.layers.0.proj.o.lora_A.weight": down, "diffusion_model.layers.0.proj.o.lora_B.weight": up})
        x = torch.randn(*shape)

        _install(model, path)

        out = model.layers[0].proj.o(x)
        assert out.shape == x.shape
        _assert_close(out, reference.layers[0].proj.o(x) + _delta(down, up, x))

    def test_the_term_returns_the_input_dtype(self, model, tmp_path):
        down, up = _lora(16, DIM, DIM)
        path = _save(tmp_path, {"diffusion_model.layers.0.proj.o.lora_A.weight": down, "diffusion_model.layers.0.proj.o.lora_B.weight": up})
        model.layers[0].proj.o.to(torch.bfloat16)

        _install(model, path, dtype=torch.float32)

        assert model.layers[0].proj.o(torch.randn(3, DIM, dtype=torch.bfloat16)).dtype == torch.bfloat16


class TestStacking:
    """A Turbo-style kohya file plus a user file with a fused ``proj_qkv`` must apply together."""

    def _files(self, tmp_path):
        turbo, user = {}, {}
        factors = {}
        for i, name in enumerate(("q", "k", "v", "o")):
            factors[f"turbo_{name}"] = _lora(20 + i, DIM, DIM, rank=2)
            turbo[f"diffusion_model.layers.0.proj.{name}.lora_down.weight"] = factors[f"turbo_{name}"][0]
            turbo[f"diffusion_model.layers.0.proj.{name}.lora_up.weight"] = factors[f"turbo_{name}"][1]
            turbo[f"diffusion_model.layers.0.proj.{name}.alpha"] = torch.tensor(2.0)
        factors["user_qkv"] = _lora(30, 3 * DIM, DIM, rank=2)
        user["lora_unet_layers_0_proj_qkv.lora_down.weight"] = factors["user_qkv"][0]
        user["lora_unet_layers_0_proj_qkv.lora_up.weight"] = factors["user_qkv"][1]
        user["lora_unet_layers_0_proj_qkv.alpha"] = torch.tensor(1.0)
        factors["user_o"] = _lora(31, DIM, DIM, rank=2)
        user["lora_unet_layers_0_proj_o.lora_down.weight"] = factors["user_o"][0]
        user["lora_unet_layers_0_proj_o.lora_up.weight"] = factors["user_o"][1]
        user["lora_unet_layers_0_proj_o.alpha"] = torch.tensor(1.0)
        return (_save(tmp_path, turbo, name="turbo.safetensors"), _save(tmp_path, user, name="user.safetensors")), factors

    def _check(self, model, reference, factors):
        x = torch.randn(5, DIM)
        for i, name in enumerate(("q", "k", "v")):
            turbo_down, turbo_up = factors[f"turbo_{name}"]
            user_down, user_up = factors["user_qkv"]
            expected = getattr(reference.layers[0].proj, name)(x) + (2.0 / 2) * _delta(turbo_down, turbo_up, x) + (1.0 / 2) * _delta(user_down, user_up[i * DIM : (i + 1) * DIM], x)
            _assert_close(getattr(model.layers[0].proj, name)(x), expected)
        turbo_down, turbo_up = factors["turbo_o"]
        user_down, user_up = factors["user_o"]
        expected = reference.layers[0].proj.o(x) + _delta(turbo_down, turbo_up, x) + 0.5 * _delta(user_down, user_up, x)
        _assert_close(model.layers[0].proj.o(x), expected)

    def test_both_apply_in_one_call(self, model, tmp_path):
        reference = copy.deepcopy(model)
        (turbo, user), factors = self._files(tmp_path)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            install_loras(model, [LoraFile(turbo, 1.0, torch.float32), LoraFile(user, 1.0, torch.float32)], device="cpu")

        self._check(model, reference, factors)

    def test_both_apply_in_two_calls(self, model, tmp_path):
        reference = copy.deepcopy(model)
        (turbo, user), factors = self._files(tmp_path)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            install_loras(model, [LoraFile(turbo, 1.0, torch.float32)], device="cpu")
            install_loras(model, [LoraFile(user, 1.0, torch.float32)], device="cpu")

        self._check(model, reference, factors)
        assert all(not isinstance(m.base, lokr.AdaptedLinear) for m in model.modules() if isinstance(m, lokr.AdaptedLinear))
