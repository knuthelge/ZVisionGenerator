"""Tests for the H3 AdaLN fold, on tiny random-init transformers (no real weights).

The fold must reproduce the stock modulation bit for bit, so the transformer tests compare outputs with ``torch.equal``.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("safetensors")

from zvisiongenerator.backends.cuda.video.h3 import adaln as adaln_module  # noqa: E402
from zvisiongenerator.backends.cuda.video.h3.adaln import (  # noqa: E402
    FoldedAdaLN,
    FoldState,
    build_tables,
    compute_temb,
    fold_keys,
    install_fold,
    load_time_embedder,
    lora_table_delta,
    read_tables,
    step_keys,
    table_keys,
    unique_keys,
    write_tables,
)
from zvisiongenerator.backends.cuda.video.h3.schedule import (  # noqa: E402
    AUDIO_COND_T,
    KEYFRAME_T,
    ROW_SETS,
    build_schedulers,
    row_timesteps,
)
from zvisiongenerator.utils.h3_preset import H3Schedule  # noqa: E402

_SCHEDULE_8 = H3Schedule(8, 6.0, 3.0)
_SCHEDULE_4 = H3Schedule(4, 6.0, 3.0)
_VIDEO_TAG, _TEXT_TAG, _AUDIO_TAG = 0, 1, 2
_PATCH = (1, 2, 2)
_HIDDEN = 64
_TIME_EMBED_DIM = 32
_LAYERS = 2
_CONDITIONING = {"plain": (0, 0), "keyframe": (1, 0), "extend": (1, 2)}  # leading (video, audio) conditioning rows
_KEYFRAME_F32 = float(torch.tensor(KEYFRAME_T, dtype=torch.float32))
_AUDIO_COND_F32 = float(torch.tensor(AUDIO_COND_T, dtype=torch.float32))


def _expected_keys(video, audio, row_set: str) -> list[tuple[float, ...]]:
    """Return the keys of one row set straight from the sigma lists: the generated levels plus the pinned conditioning levels."""
    pinned = {"plain": set(), "keyframe": {_KEYFRAME_F32}, "extend": {_KEYFRAME_F32, _AUDIO_COND_F32}}[row_set]
    return [tuple(sorted({float(t), float(a), *pinned})) for t, a in zip(video.timesteps, audio.timesteps, strict=True)]


# --- keys ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("schedule", [_SCHEDULE_8, _SCHEDULE_4], ids=["8-step", "4-step"])
@pytest.mark.parametrize("row_set", ROW_SETS)
def test_step_keys_match_the_sigma_lists(schedule, row_set):
    video, audio = build_schedulers(schedule)
    assert step_keys(video, audio, row_set) == _expected_keys(video, audio, row_set)


@pytest.mark.parametrize(("row_set", "first", "later"), [("plain", 1, 2), ("keyframe", 2, 3), ("extend", 3, 4)])
def test_key_arity_per_forward(row_set, first, later):
    video, audio = build_schedulers(_SCHEDULE_8)
    keys = step_keys(video, audio, row_set)
    # The first forward has t = 0 for video and audio, so the two generated levels collapse into one.
    assert [len(key) for key in keys] == [first] + [later] * 7
    assert all(list(key) == sorted(key) for key in keys)


def test_all_keys_are_24_and_the_four_step_pairs_are_a_subset():
    keys_8 = unique_keys(key for row_set in ROW_SETS for key in step_keys(*build_schedulers(_SCHEDULE_8), row_set))
    keys_4 = unique_keys(key for row_set in ROW_SETS for key in step_keys(*build_schedulers(_SCHEDULE_4), row_set))
    assert len(keys_8) == 24
    assert len(keys_4) == 12
    assert set(keys_4) <= set(keys_8)
    assert fold_keys({8: _SCHEDULE_8, 4: _SCHEDULE_4}) == keys_8
    assert fold_keys({4: _SCHEDULE_4, 8: _SCHEDULE_8}) == keys_8
    assert len(fold_keys({8: _SCHEDULE_8})) == 24


def test_keys_do_not_depend_on_the_layout():
    # The dummy layouts of step_keys must give what the real packed layout gives for the same conditioning rows.
    video, audio = build_schedulers(_SCHEDULE_8)
    layout = _layout()
    for row_set in ROW_SETS:
        real = []
        for t, audio_t in zip(video.timesteps, audio.timesteps, strict=True):
            unique, _ = _row_timesteps(layout, float(t), float(audio_t), row_set)
            real.append(tuple(float(value) for value in unique))
        assert real == step_keys(video, audio, row_set)


def test_a_schedule_with_keys_outside_the_longest_is_refused_by_name():
    with pytest.raises(ValueError, match="5-step"):
        fold_keys({8: _SCHEDULE_8, 5: H3Schedule(5, 6.0, 3.0)})


def test_fold_keys_needs_a_schedule():
    with pytest.raises(ValueError, match="At least one schedule"):
        fold_keys({})


def test_step_keys_refuses_an_unknown_row_set_and_uneven_schedules():
    video, audio = build_schedulers(_SCHEDULE_8)
    with pytest.raises(ValueError, match="Unknown row set"):
        step_keys(video, audio, "reference")
    with pytest.raises(ValueError, match="differ in length"):
        step_keys(video, SimpleNamespace(timesteps=audio.timesteps[:3]), "plain")


def test_unique_keys_keeps_first_appearance_order():
    assert unique_keys([(0.5,), (0.1,), (0.5,), (0.1, 0.5)]) == [(0.5,), (0.1,), (0.1, 0.5)]


# --- tiny transformer ------------------------------------------------------------------------------------------------


def _config() -> dict:
    return {
        "num_attention_heads": 2,
        "attention_head_dim": 32,
        "hidden_size": _HIDDEN,
        "num_layers": _LAYERS,
        "num_refiner_layers": 1,
        "ffn_dim": 2 * _HIDDEN,
        "in_channels": 24,
        "audio_in_channels": 32,
        "patch_size": _PATCH,
        "text_dim": 48,
        "freq_dim": 32,
        "time_embed_hidden_dim": _HIDDEN,
        "time_embed_dim": _TIME_EMBED_DIM,
        "rope_freq_dim": 4,
    }


def _transformer(seed: int = 0):
    """Return a random-init transformer with the checkpoint's dtype mix: bf16 blocks, fp32 timestep MLP, patch projections and heads."""
    from diffusers import MiniMaxH3Transformer3DModel

    torch.manual_seed(seed)
    model = MiniMaxH3Transformer3DModel(**_config())
    keep = set(model._keep_in_fp32_modules)
    for name, module in model.named_modules():
        if not any(part in keep for part in name.split(".")):
            for parameter in module.parameters(recurse=False):
                parameter.data = parameter.data.to(torch.bfloat16)
    return model.eval()


def _layout() -> SimpleNamespace:
    """Return a tiny packed layout: 4 text rows, 6 audio rows and 4 video rows, with the structural tensors the forward takes."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep

    text_tokens = 4
    tags = torch.full((text_tokens,), _TEXT_TAG, dtype=torch.long)
    position_ids, token_tags, video_indices, audio_indices, text_indices, _, _ = MiniMaxH3PrepareLayoutStep.build_packed_sequence(tags, 2, 2, 4, 3, _PATCH, 2, _AUDIO_TAG, _VIDEO_TAG, ())
    return SimpleNamespace(
        text_tokens=text_tokens,
        video_indices=video_indices,
        audio_indices=audio_indices,
        kwargs={"position_ids": position_ids, "token_tags": token_tags, "video_indices": video_indices, "audio_indices": audio_indices, "text_indices": text_indices},
    )


def _row_timesteps(layout, t: float, audio_t: float, row_set: str):
    condition_video, condition_audio = _CONDITIONING[row_set]
    return row_timesteps(
        layout.video_indices,
        layout.audio_indices,
        layout.text_tokens,
        t,
        audio_t,
        condition_video_rows=condition_video,
        condition_audio_rows=condition_audio,
    )


def _inputs(layout, seed: int = 1) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {
        "hidden_states": torch.randn(1, len(layout.video_indices), 96, generator=generator),
        "audio_hidden_states": torch.randn(1, len(layout.audio_indices), 32, generator=generator),
        "encoder_hidden_states": torch.randn(1, layout.text_tokens, 48, generator=generator),
    }


def _tables_for(model, keys):
    temb = {key: compute_temb(model.time_proj, model.time_embedder, key, "cpu") for key in keys}
    return [build_tables(block.adaln_proj, temb) for block in model.transformer_blocks], temb


def test_compute_temb_is_the_transformers_own():
    model = _transformer()
    key = (0.0, 0.25, 0.999)
    stock = model.time_embedder(model.time_proj(torch.tensor(key)))
    assert torch.equal(compute_temb(model.time_proj, model.time_embedder, key, "cpu"), stock)


def test_load_time_embedder_reproduces_the_transformers_temb():
    model = _transformer()
    state = {name.removeprefix("time_embedder."): value for name, value in model.state_dict().items() if name.startswith("time_embedder.")}
    time_proj, embedder = load_time_embedder(_config(), state, "cpu")
    key = (0.1, 0.5)
    assert torch.equal(compute_temb(time_proj, embedder, key, "cpu"), compute_temb(model.time_proj, model.time_embedder, key, "cpu"))
    assert next(embedder.parameters()).dtype == torch.float32


def test_build_tables_equal_the_stock_module_bitwise():
    model = _transformer()
    video, audio = build_schedulers(_SCHEDULE_8)
    keys = unique_keys(key for row_set in ROW_SETS for key in step_keys(video, audio, row_set))
    tables, temb = _tables_for(model, keys)
    for block, block_tables in zip(model.transformer_blocks, tables, strict=True):
        for key in keys:
            stock = block.adaln_proj(temb[key])
            folded = block_tables[key].view(-1, 6 * _HIDDEN).chunk(6, dim=-1)
            assert all(torch.equal(a, b) for a, b in zip(stock, folded, strict=True))
            assert block_tables[key].dtype == torch.bfloat16


@pytest.mark.parametrize("row_set", ROW_SETS)
def test_folded_transformer_equals_stock_bitwise(row_set):
    stock = _transformer()
    folded = copy.deepcopy(stock)
    video, audio = build_schedulers(_SCHEDULE_8)
    keys = step_keys(video, audio, row_set)
    tables, _ = _tables_for(stock, unique_keys(keys))
    handle = install_fold(folded, tables)
    layout = _layout()
    inputs = _inputs(layout)
    with torch.no_grad():
        for t, audio_t in zip(video.timesteps, audio.timesteps, strict=True):
            unique, indices = _row_timesteps(layout, float(t), float(audio_t), row_set)
            call = {**inputs, "timestep": unique, "timestep_indices": indices, "return_dict": False, **layout.kwargs}
            want_video, want_audio = stock(**call)
            got_video, got_audio = folded(**call)
            assert torch.equal(got_video, want_video)
            assert torch.equal(got_audio, want_audio)
    assert len(handle.state.hits) == 8
    assert [handle.state.keys[hit] for hit in handle.state.hits] == keys


def test_installed_fold_drops_the_projection_weights_and_keeps_state_dicts_clean():
    model = _transformer()
    tables, _ = _tables_for(model, [(0.25, 0.5)])
    install_fold(model, tables)
    assert all(isinstance(block.adaln_proj, FoldedAdaLN) for block in model.transformer_blocks)
    assert not any("adaln_proj" in name for name in model.state_dict())


def test_the_pre_hook_reads_a_positional_timestep_too():
    model = _transformer()
    tables, _ = _tables_for(model, [(0.25, 0.5)])
    handle = install_fold(model, tables)
    layout = _layout()
    inputs = _inputs(layout)
    indices = torch.zeros(len(layout.kwargs["token_tags"]), dtype=torch.long)
    indices[layout.audio_indices] = 1
    with torch.no_grad():
        model(
            inputs["hidden_states"],
            inputs["audio_hidden_states"],
            inputs["encoder_hidden_states"],
            torch.tensor([0.25, 0.5]),
            indices,
            **{k: layout.kwargs[k] for k in ("token_tags", "position_ids", "video_indices", "audio_indices", "text_indices")},
        )
    assert handle.state.hits == [0]


def test_an_unfolded_timestep_key_raises_with_the_rebuild_hint():
    state = FoldState([(0.25, 0.5)])
    with pytest.raises(KeyError, match="rebuild the H3 copy"):
        state.select(torch.tensor([0.25, 0.75]))
    assert state.select(torch.tensor([0.25, 0.5])) == 0


def test_a_forward_with_an_unbuilt_key_is_refused_not_computed_stock():
    model = _transformer()
    tables, _ = _tables_for(model, [(0.25, 0.5)])
    install_fold(model, tables)
    layout = _layout()
    indices = torch.zeros(len(layout.kwargs["token_tags"]), dtype=torch.long)
    with pytest.raises(KeyError, match="rebuild the H3 copy"):
        model(**_inputs(layout), timestep=torch.tensor([0.3]), timestep_indices=indices, return_dict=False, **layout.kwargs)


def test_folded_projection_outside_a_forward_and_with_wrong_rows():
    state = FoldState([(0.25, 0.5)])
    table = torch.zeros(2, 6 * 3 * 4)  # two timestep rows of six modulation parameters for three modalities, hidden size 4
    folded = FoldedAdaLN(state, [table], 4)
    with pytest.raises(RuntimeError, match="outside a transformer forward"):
        folded(torch.zeros(2, 8))
    state.select(torch.tensor([0.25, 0.5]))
    with pytest.raises(ValueError, match="2 timestep rows, the forward has 3"):
        folded(torch.zeros(3, 8))


def test_install_fold_validates_the_tables():
    model = _transformer()
    tables, _ = _tables_for(model, [(0.25, 0.5), (0.5,)])
    with pytest.raises(ValueError, match="tables for 1 blocks"):
        install_fold(model, tables[:1])
    reordered = [tables[0], dict(reversed(list(tables[1].items())))]
    with pytest.raises(ValueError, match="same keys in the same order"):
        install_fold(model, reordered)
    assert not any(isinstance(block.adaln_proj, FoldedAdaLN) for block in model.transformer_blocks)


def test_removing_the_handle_stops_the_key_selection():
    model = _transformer()
    tables, _ = _tables_for(model, [(0.25, 0.5)])
    handle = install_fold(model, tables)
    handle.remove()
    assert not model._forward_pre_hooks


# --- table files -----------------------------------------------------------------------------------------------------


def test_tables_round_trip_bitwise_and_only_the_wanted_keys_are_read(tmp_path, monkeypatch):
    from zvisiongenerator.utils import safetensors_stream

    model = _transformer()
    keys = [(0.5,), (0.1, 0.5), (0.0, 0.25, 0.999)]
    tables, _ = _tables_for(model, keys)
    path = tmp_path / "adaln" / "tables.safetensors"
    info = write_tables(path, tables)
    assert info["keys"] == 3 and info["blocks"] == _LAYERS and info["bytes"] == path.stat().st_size
    assert table_keys(path) == sorted(keys)
    assert all(isinstance(key, list) for key in json.loads(safetensors_stream.read_metadata(path)["keys"]))

    reads: list[str] = []
    original = safetensors_stream.TensorFile.read

    def recording(self, name):
        reads.append(name)
        return original(self, name)

    monkeypatch.setattr(safetensors_stream.TensorFile, "read", recording)
    wanted = [(0.5,), (0.0, 0.25, 0.999)]
    back = read_tables(path, wanted, _LAYERS)
    assert len(reads) == len(wanted) * _LAYERS
    for block in range(_LAYERS):
        assert list(back[block]) == wanted
        for key in wanted:
            assert torch.equal(back[block][key], tables[block][key])
            assert back[block][key].dtype == tables[block][key].dtype


def test_read_tables_refuses_a_missing_key_and_a_block_mismatch(tmp_path):
    model = _transformer()
    tables, _ = _tables_for(model, [(0.5,)])
    path = tmp_path / "tables.safetensors"
    write_tables(path, tables)
    with pytest.raises(KeyError, match="rebuild the H3 copy"):
        read_tables(path, [(0.5,), (0.7,)], _LAYERS)
    with pytest.raises(ValueError, match="2 blocks, the transformer has 3"):
        read_tables(path, [(0.5,)], 3)


def test_write_tables_refuses_uneven_or_empty_blocks(tmp_path):
    table = torch.zeros(3, 4)
    with pytest.raises(ValueError, match="same keys"):
        write_tables(tmp_path / "t.safetensors", [{(0.5,): table}, {(0.6,): table}])
    with pytest.raises(ValueError, match="At least one block"):
        write_tables(tmp_path / "t.safetensors", [])


def test_a_restored_fold_equals_the_in_memory_fold(tmp_path):
    stock = _transformer()
    video, audio = build_schedulers(_SCHEDULE_8)
    keys = unique_keys(step_keys(video, audio, "keyframe"))
    tables, _ = _tables_for(stock, keys)
    path = tmp_path / "tables.safetensors"
    write_tables(path, tables)
    folded = copy.deepcopy(stock)
    install_fold(folded, read_tables(path, keys, _LAYERS))
    layout = _layout()
    unique, indices = _row_timesteps(layout, float(video.timesteps[3]), float(audio.timesteps[3]), "keyframe")
    call = {**_inputs(layout), "timestep": unique, "timestep_indices": indices, "return_dict": False, **layout.kwargs}
    with torch.no_grad():
        assert all(torch.equal(a, b) for a, b in zip(stock(**call), folded(**call), strict=True))


# --- exact LoRA delta ------------------------------------------------------------------------------------------------


def _linear_holder(weight: torch.Tensor, bias: torch.Tensor) -> SimpleNamespace:
    linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=True)
    linear.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    linear.bias = torch.nn.Parameter(bias.clone(), requires_grad=False)
    return SimpleNamespace(linear=linear)


def _lora_case(dtype: torch.dtype):
    generator = torch.Generator().manual_seed(7)
    out = 6 * _HIDDEN * 3
    weight = torch.randn(out, _TIME_EMBED_DIM, generator=generator) * 0.1
    bias = torch.randn(out, generator=generator) * 0.1
    down = torch.randn(4, _TIME_EMBED_DIM, generator=generator) * 0.1
    up = torch.randn(out, 4, generator=generator) * 0.1
    keys = [(0.5,), (0.1, 0.5), (0.0, 0.25, 0.999)]
    temb = {key: torch.randn(len(key), _TIME_EMBED_DIM, generator=generator) for key in keys}
    return weight.to(dtype), bias.to(dtype), down, up, temb


def test_lora_delta_equals_tables_from_the_merged_weight_in_fp32():
    weight, bias, down, up, temb = _lora_case(torch.float32)
    scale = 0.7
    base = build_tables(_linear_holder(weight, bias), temb)
    folded = lora_table_delta(base, temb, down, up, scale)
    merged = build_tables(_linear_holder(weight + scale * (up @ down), bias), temb)
    assert set(folded) == set(merged)
    for key in merged:
        torch.testing.assert_close(folded[key], merged[key], rtol=1e-4, atol=1e-5)
        assert not torch.equal(folded[key], base[key])


def test_lora_delta_on_bf16_tables_stays_within_bf16_rounding_of_the_merged_tables():
    weight, bias, down, up, temb = _lora_case(torch.bfloat16)
    scale = 0.7
    base = build_tables(_linear_holder(weight, bias), temb)
    folded = lora_table_delta(base, temb, down, up, scale)
    merged = build_tables(_linear_holder((weight.float() + scale * (up @ down)).to(torch.bfloat16), bias), temb)
    for key in merged:
        assert folded[key].dtype == torch.bfloat16
        tolerance = 0.02 * float(merged[key].float().abs().max())
        assert float((folded[key].float() - merged[key].float()).abs().max()) <= tolerance


def test_lora_delta_leaves_the_inputs_alone_and_a_zero_scale_changes_nothing():
    weight, bias, down, up, temb = _lora_case(torch.bfloat16)
    base = build_tables(_linear_holder(weight, bias), temb)
    snapshot = {key: table.clone() for key, table in base.items()}
    lora_table_delta(base, temb, down, up, 0.7)
    assert all(torch.equal(base[key], snapshot[key]) for key in base)
    same = lora_table_delta(base, temb, down, up, 0.0)
    assert all(torch.equal(same[key], base[key]) for key in base)


def test_module_surface():
    assert set(adaln_module.__all__) >= {"install_fold", "fold_keys", "step_keys", "lora_table_delta", "FoldedAdaLN", "FoldState", "FoldHandle"}
    assert all(hasattr(adaln_module, name) for name in adaln_module.__all__)


def test_the_bitwise_comparison_detects_a_wrong_table():
    # Control for the equality tests above: one changed table entry must change the output.
    stock = _transformer()
    folded = copy.deepcopy(stock)
    layout = _layout()
    unique, indices = _row_timesteps(layout, 0.25, 0.5, "plain")
    key = tuple(float(value) for value in unique)
    tables, _ = _tables_for(stock, [key])
    tables[0][key] = tables[0][key] + torch.tensor(0.5, dtype=torch.bfloat16)
    install_fold(folded, tables)
    call = {**_inputs(layout), "timestep": unique, "timestep_indices": indices, "return_dict": False, **layout.kwargs}
    with torch.no_grad():
        assert not torch.equal(stock(**call)[0], folded(**call)[0])
