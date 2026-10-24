"""Tests for the stored 4-bit H3 copy: manifest, one-pass write, ensure, and load order.

Everything runs on a tiny random-init transformer written to ``tmp_path``. ``pack_nf4`` is replaced by a stand-in that keeps the
weight under its own name, so the copy's layout and what gets packed are checked without bitsandbytes; one test packs for real
when bitsandbytes and a GPU are present.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("safetensors")
from safetensors.torch import load_file, save_file  # noqa: E402

from zvisiongenerator.backends.cuda.video.h3 import adaln as adaln_module  # noqa: E402
from zvisiongenerator.backends.cuda.video.h3 import lora as h3_lora  # noqa: E402
from zvisiongenerator.backends.cuda.video.h3 import stored_copy  # noqa: E402
from zvisiongenerator.backends.cuda.video.h3.schedule import ROW_SETS, build_schedulers  # noqa: E402
from zvisiongenerator.backends.cuda.video.h3.stored_copy import COPY_NAME, copy_manifest, ensure_copy, load_copy, write_copy  # noqa: E402
from zvisiongenerator.utils.h3_preset import H3Schedule  # noqa: E402
from zvisiongenerator.utils.stored_quant import MANIFEST_NAME  # noqa: E402

_SCHEDULE_8 = H3Schedule(8, 6.0, 3.0)
_SCHEDULE_4 = H3Schedule(4, 6.0, 3.0)
_SCALE = 0.0625
_PATCH = (1, 2, 2)
_HIDDEN = 64
_LAYERS = 2
_TEXT_TAG, _AUDIO_TAG, _VIDEO_TAG = 1, 2, 0
_NF4_FORMAT = "nf4-test"
_ALL_KEYS = adaln_module.fold_keys({8: _SCHEDULE_8, 4: _SCHEDULE_4})
_BLOCK_LINEARS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")


# --- tiny model and files ---------------------------------------------------------------------------------------------


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
        "time_embed_dim": 32,
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


def _inputs(layout, seed: int = 1) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {
        "hidden_states": torch.randn(1, len(layout.video_indices), 96, generator=generator),
        "audio_hidden_states": torch.randn(1, len(layout.audio_indices), 32, generator=generator),
        "encoder_hidden_states": torch.randn(1, layout.text_tokens, 48, generator=generator),
    }


def _write_lora(path: Path, factors: dict[str, tuple[torch.Tensor, torch.Tensor]], metadata: dict[str, str] | None = None) -> Path:
    """Write an ai-toolkit style LoRA file (checkpoint module names under ``diffusion_model.``)."""
    tensors = {}
    for module, (down, up) in factors.items():
        tensors[f"diffusion_model.{module}.lora_A.weight"] = down.contiguous()
        tensors[f"diffusion_model.{module}.lora_B.weight"] = up.contiguous()
    save_file(tensors, str(path), metadata=metadata)
    return path


def _factors(shapes: dict[str, tuple[int, int]], rank: int, seed: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    return {module: (torch.randn(rank, columns, generator=generator), torch.randn(rows, rank, generator=generator)) for module, (rows, columns) in shapes.items()}


_TURBO_SHAPES = {
    "blocks.0.attn.out_proj": (_HIDDEN, _HIDDEN),
    "blocks.1.adaln_proj.linear": (6 * 3 * _HIDDEN, 32),
    "token_refiner.blocks.0.attn.out_proj": (_HIDDEN, _HIDDEN),
}
_USER_SHAPES = {
    "blocks.0.attn.out_proj": (_HIDDEN, _HIDDEN),
    "blocks.0.adaln_proj.linear": (6 * 3 * _HIDDEN, 32),
    "blocks.1.adaln_proj.linear": (6 * 3 * _HIDDEN, 32),
    "token_refiner.blocks.0.attn.out_proj": (_HIDDEN, _HIDDEN),
}


class _Sources:
    """A tiny checkpoint in two shards, its config and a turbo LoRA that hits a block linear, an AdaLN projection and the refiner."""

    def __init__(self, root: Path) -> None:
        self.model = _transformer()
        self.config = _config()
        state = {name: tensor.detach().contiguous() for name, tensor in self.model.state_dict().items()}
        names = sorted(state)
        half = len(names) // 2
        self.shards = []
        for number, chunk in enumerate((names[:half], names[half:])):
            path = root / f"shard-{number}.safetensors"
            save_file({name: state[name] for name in chunk}, str(path))
            self.shards.append(path)
        self.turbo_factors = _factors(_TURBO_SHAPES, 4, seed=3)
        self.turbo = _write_lora(root / "turbo.safetensors", self.turbo_factors, metadata={"alpha": "1"})
        self.root = root


@pytest.fixture
def sources(tmp_path: Path) -> _Sources:
    return _Sources(tmp_path)


class _Recorder:
    """Calls recorded by the stand-ins of the packing and streaming steps."""

    def __init__(self) -> None:
        self.packed: list[str] = []
        self.nf4_args: list[dict] = []
        self.events: list[str] = []


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    """Replace the bitsandbytes steps and the streaming with stand-ins that record their use."""
    record = _Recorder()
    real_names = stored_copy.nf4_linear_names

    def fake_pack(key, weight, device):
        record.packed.append(key)
        return {key: weight}

    def spy_names(model, *, prefix, exclude=()):
        record.nf4_args.append({"prefix": prefix, "exclude": exclude})
        return real_names(model, prefix=prefix, exclude=exclude)

    def swap(model, names, dtype):
        record.events.append("swap")
        return {}

    real_fill = stored_copy.fill_weights

    def fill(model, tensors, source, skipped=()):
        record.events.append("fill")
        return real_fill(model, tensors, source, skipped)

    monkeypatch.setattr(stored_copy, "nf4_storage_format", lambda: _NF4_FORMAT)
    monkeypatch.setattr(stored_copy, "pack_nf4", fake_pack)
    monkeypatch.setattr(stored_copy, "nf4_linear_names", spy_names)
    monkeypatch.setattr(stored_copy, "swap_in_linear4bit", swap)
    monkeypatch.setattr(stored_copy, "fill_weights", fill)
    monkeypatch.setattr(stored_copy, "stream_blocks", lambda blocks, device: record.events.append("stream"))
    monkeypatch.setattr(stored_copy, "move_outside", lambda model, blocks, device: record.events.append("move_outside"))
    return record


def _write(sources: _Sources, target: Path, keys=_ALL_KEYS, scale: float = _SCALE) -> Path:
    write_copy(sources.shards, sources.config, sources.turbo, scale, keys, target, "cpu")
    return target


# --- manifest ---------------------------------------------------------------------------------------------------------


def test_manifest_is_json_native_with_the_24_key_union(sources: _Sources, recorder: _Recorder) -> None:
    manifest = copy_manifest(sources.shards, sources.turbo, _SCALE, _ALL_KEYS)

    assert len(_ALL_KEYS) == 24
    assert json.loads(json.dumps(manifest)) == manifest
    assert manifest["layout"] == 1
    assert manifest["family"] == "h3"
    assert manifest["bits"] == 4
    assert manifest["format"] == _NF4_FORMAT
    assert manifest["lora_scale"] == _SCALE
    assert manifest["fold_keys"] == [list(key) for key in _ALL_KEYS]
    assert [source["name"] for source in manifest["sources"]] == [path.name for path in (*sources.shards, sources.turbo)]
    assert all(source["bytes"] > 0 for source in manifest["sources"])


# --- ensure ---------------------------------------------------------------------------------------------------------


def _spy_writes(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    calls: list[Path] = []
    real = stored_copy.write_copy

    def spy(shards, config, lora_file, scale, keys, target, device):
        calls.append(target)
        real(shards, config, lora_file, scale, keys, target, device)

    monkeypatch.setattr(stored_copy, "write_copy", spy)
    return calls


def _ensure(sources: _Sources, models: Path, keys=_ALL_KEYS, scale: float = _SCALE) -> Path:
    return ensure_copy(models, sources.shards, sources.config, sources.turbo, scale, keys, "cpu")


def test_ensure_writes_h3_q4_with_its_manifest_and_a_second_ensure_writes_nothing(sources: _Sources, recorder: _Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy_writes(monkeypatch)
    models = sources.root / "models"

    first = _ensure(sources, models)
    snapshot = {path.name: path.stat().st_mtime_ns for path in first.iterdir()}
    second = _ensure(sources, models)

    assert first == second == models / COPY_NAME == models / "h3@q4"
    assert len(calls) == 1
    assert {path.name: path.stat().st_mtime_ns for path in second.iterdir()} == snapshot
    assert json.loads((first / MANIFEST_NAME).read_text(encoding="utf-8")) == copy_manifest(sources.shards, sources.turbo, _SCALE, _ALL_KEYS)
    assert not list(models.glob(".*.partial"))


def _fine_tune(sources: _Sources, root: Path) -> list[Path]:
    """Return shards with the names, shapes, dtypes and byte sizes of *sources*' but other values (a same-layout fine-tune)."""
    root.mkdir()
    shards = []
    for shard in sources.shards:
        tensors = load_file(str(shard))
        path = root / shard.name
        save_file({name: tensor + 1 if tensor.is_floating_point() else tensor for name, tensor in tensors.items()}, str(path))
        assert path.stat().st_size == shard.stat().st_size
        shards.append(path)
    return shards


def test_a_local_fine_tune_with_the_stock_names_and_sizes_rebuilds_the_copy(sources: _Sources, recorder: _Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy_writes(monkeypatch)
    models = sources.root / "models"
    tuned = _fine_tune(sources, sources.root / "tuned")

    _ensure(sources, models)
    ensure_copy(models, tuned, sources.config, sources.turbo, _SCALE, _ALL_KEYS, "cpu")
    assert len(calls) == 2
    name = "transformer_blocks.1.attn.to_q.weight"
    tuned_weight = next(tensors[name] for shard in tuned if name in (tensors := load_file(str(shard))))
    stored = _stored(models / COPY_NAME)[name]
    assert torch.equal(stored, tuned_weight.to(stored.dtype))
    assert not torch.equal(stored, sources.model.state_dict()[name].to(stored.dtype))

    ensure_copy(models, tuned, sources.config, sources.turbo, _SCALE, _ALL_KEYS, "cpu")
    assert len(calls) == 2


def test_a_copy_made_from_hub_cache_files_stays_current_under_its_old_manifest(sources: _Sources, recorder: _Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy_writes(monkeypatch)
    cache = sources.root / "hub"
    (cache / "blobs").mkdir(parents=True)
    snapshot = cache / "models--org--h3" / "snapshots" / "rev"
    snapshot.mkdir(parents=True)
    shards, legacy_sources = [], []
    for number, shard in enumerate((*sources.shards, sources.turbo)):
        blob = cache / "blobs" / f"{number:x}".rjust(64, "a")
        blob.write_bytes(shard.read_bytes())
        link = snapshot / shard.name
        link.symlink_to(blob)
        shards.append(link)
        legacy_sources.append({"name": blob.name, "bytes": blob.stat().st_size})
    *model_shards, turbo = shards
    models = sources.root / "models"
    legacy = {
        "layout": 1,
        "family": "h3",
        "bits": 4,
        "format": _NF4_FORMAT,
        "sources": legacy_sources,
        "lora_scale": _SCALE,
        "fold_keys": [list(key) for key in _ALL_KEYS],
    }
    (models / COPY_NAME).mkdir(parents=True)
    (models / COPY_NAME / MANIFEST_NAME).write_text(json.dumps(legacy, indent=2) + "\n", encoding="utf-8")

    assert copy_manifest(model_shards, turbo, _SCALE, _ALL_KEYS) == legacy
    ensure_copy(models, model_shards, sources.config, turbo, _SCALE, _ALL_KEYS, "cpu")
    assert calls == []


def test_a_scaled_fp8_weight_is_stored_multiplied_by_its_scale(sources: _Sources, recorder: _Recorder) -> None:
    name = "transformer_blocks.1.attn.to_q.weight"
    shape = tuple(sources.model.state_dict()[name].shape)
    shard = next(path for path in sources.shards if name in load_file(str(path)))
    tensors = load_file(str(shard))
    tensors[name] = torch.full(shape, 2.0).to(torch.float8_e4m3fn)
    tensors[f"{name}_scale"] = torch.tensor(0.5)
    save_file(tensors, str(shard))

    stored = _stored(_write(sources, sources.root / "copy"))

    assert torch.equal(stored[name], torch.ones(shape, dtype=torch.bfloat16))
    assert f"{name}_scale" not in stored


def test_a_changed_key_union_or_scale_rebuilds(sources: _Sources, recorder: _Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy_writes(monkeypatch)
    models = sources.root / "models"
    subset = _ALL_KEYS[:12]

    _ensure(sources, models, subset)
    _ensure(sources, models, subset)
    assert len(calls) == 1
    target = _ensure(sources, models, _ALL_KEYS)
    assert len(calls) == 2
    assert adaln_module.table_keys(target / "adaln_tables.safetensors") == sorted(_ALL_KEYS)
    _ensure(sources, models, _ALL_KEYS, scale=0.125)
    assert len(calls) == 3


def test_a_failed_write_leaves_no_copy(sources: _Sources, recorder: _Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(key, weight, device):
        raise RuntimeError("pack failed")

    monkeypatch.setattr(stored_copy, "pack_nf4", broken)
    models = sources.root / "models"
    with pytest.raises(RuntimeError, match="pack failed"):
        _ensure(sources, models)
    assert not (models / COPY_NAME).exists()
    assert not list(models.glob(".*.partial"))


# --- write ---------------------------------------------------------------------------------------------------------


def _stored(copy_dir: Path) -> dict[str, torch.Tensor]:
    index = json.loads((copy_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    tensors: dict[str, torch.Tensor] = {}
    for name in sorted(set(index["weight_map"].values())):
        tensors.update(load_file(str(copy_dir / name)))
    return tensors


def test_nf4_packs_only_the_block_linears_and_the_refiner_stays_bf16(sources: _Sources, recorder: _Recorder) -> None:
    copy_dir = _write(sources, sources.root / "copy")

    assert recorder.nf4_args == [{"prefix": "transformer_blocks.", "exclude": {"adaln_proj"}}]
    assert isinstance(recorder.nf4_args[0]["exclude"], set)
    expected = {f"transformer_blocks.{block}.{name}.weight" for block in range(_LAYERS) for name in _BLOCK_LINEARS}
    assert set(recorder.packed) == expected
    assert all(name.startswith("transformer_blocks.") for name in recorder.packed)
    assert not any("adaln_proj" in name or "token_refiner" in name for name in recorder.packed)
    stored = _stored(copy_dir)
    refiner = {name: tensor for name, tensor in stored.items() if name.startswith("token_refiner.")}
    assert refiner
    assert all(tensor.dtype == torch.bfloat16 for name, tensor in refiner.items())
    assert stored["time_embedder.linear_1.weight"].dtype == torch.float32
    assert stored["proj_in.weight"].dtype == torch.float32


def test_the_copy_holds_tables_not_adaln_weights_and_the_config(sources: _Sources, recorder: _Recorder) -> None:
    copy_dir = _write(sources, sources.root / "copy")

    assert not any("adaln_proj" in name for name in _stored(copy_dir))
    assert adaln_module.table_keys(copy_dir / "adaln_tables.safetensors") == sorted(_ALL_KEYS)
    assert json.loads((copy_dir / "config.json").read_text(encoding="utf-8"))["num_layers"] == _LAYERS
    tables = adaln_module.read_tables(copy_dir / "adaln_tables.safetensors", _ALL_KEYS, _LAYERS)
    assert [list(block) for block in tables] == [_ALL_KEYS] * _LAYERS


def test_the_turbo_lora_is_merged_at_the_config_scale_before_the_fold(sources: _Sources, recorder: _Recorder) -> None:
    copy_dir = _write(sources, sources.root / "copy")
    stored = _stored(copy_dir)
    pairs = h3_lora.load_pairs(sources.turbo)
    source = sources.model.state_dict()

    # The file's metadata alpha (1 over rank 4) plays no part: the factor is the config scale alone.
    down, up = pairs["transformer_blocks.0.attn.to_out.0"]
    merged = h3_lora.merge(source["transformer_blocks.0.attn.to_out.0.weight"], down, up, _SCALE)
    assert torch.equal(stored["transformer_blocks.0.attn.to_out.0.weight"], merged)
    assert not torch.equal(merged, source["transformer_blocks.0.attn.to_out.0.weight"])
    down, up = pairs["token_refiner.refiner_blocks.0.attn.to_out.0"]
    assert torch.equal(stored["token_refiner.refiner_blocks.0.attn.to_out.0.weight"], h3_lora.merge(source["token_refiner.refiner_blocks.0.attn.to_out.0.weight"], down, up, _SCALE))
    assert torch.equal(stored["transformer_blocks.1.attn.to_q.weight"], source["transformer_blocks.1.attn.to_q.weight"])

    # Block 1's AdaLN projection is merged, then folded: its tables equal the stock module's output with the merged weight.
    reference = copy.deepcopy(sources.model)
    down, up = pairs["transformer_blocks.1.adaln_proj.linear"]
    with torch.no_grad():
        reference.transformer_blocks[1].adaln_proj.linear.weight.copy_(h3_lora.merge(source["transformer_blocks.1.adaln_proj.linear.weight"], down, up, _SCALE))
    tables = adaln_module.read_tables(copy_dir / "adaln_tables.safetensors", _ALL_KEYS[:3], _LAYERS)
    for key in _ALL_KEYS[:3]:
        temb = adaln_module.compute_temb(reference.time_proj, reference.time_embedder, key, "cpu")
        want = adaln_module.build_tables(reference.transformer_blocks[1].adaln_proj, {key: temb})[key]
        assert torch.equal(tables[1][key], want)
        unmerged = adaln_module.build_tables(sources.model.transformer_blocks[1].adaln_proj, {key: temb})[key]
        assert not torch.equal(tables[1][key], unmerged)
        assert torch.equal(tables[0][key], adaln_module.build_tables(sources.model.transformer_blocks[0].adaln_proj, {key: temb})[key])


def test_a_turbo_lora_for_another_layout_is_refused(sources: _Sources, recorder: _Recorder, tmp_path: Path) -> None:
    other = _write_lora(tmp_path / "other.safetensors", _factors({"blocks.7.attn.out_proj": (_HIDDEN, _HIDDEN)}, 4, seed=1))
    with pytest.raises(RuntimeError, match="targets 1 modules the transformer does not have"):
        write_copy(sources.shards, sources.config, other, _SCALE, _ALL_KEYS[:2], tmp_path / "copy", "cpu")


def test_shards_that_do_not_cover_the_model_are_refused(sources: _Sources, recorder: _Recorder, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="do not cover the model"):
        write_copy(sources.shards[:1], sources.config, sources.turbo, _SCALE, _ALL_KEYS[:2], tmp_path / "copy", "cpu")


# --- load ---------------------------------------------------------------------------------------------------------


@pytest.fixture
def copy_dir(sources: _Sources, recorder: _Recorder) -> Path:
    return _write(sources, sources.root / "copy")


def _reference_with_turbo(sources: _Sources):
    """Return the source transformer with the turbo LoRA merged into the weights, run by its stock modulation."""
    reference = copy.deepcopy(sources.model)
    pairs = h3_lora.load_pairs(sources.turbo)
    with torch.no_grad():
        for module, (down, up) in pairs.items():
            weight = reference.get_submodule(module).weight
            weight.copy_(h3_lora.merge(weight, down, up, _SCALE))
    return reference


@pytest.mark.parametrize("row_set", ROW_SETS)
def test_the_loaded_copy_runs_like_the_stock_model_with_the_turbo_merged(sources: _Sources, copy_dir: Path, recorder: _Recorder, row_set: str) -> None:
    from zvisiongenerator.backends.cuda.video.h3.adaln import step_keys

    video, audio = build_schedulers(_SCHEDULE_8)
    keys = adaln_module.unique_keys(step_keys(video, audio, row_set))
    reference = _reference_with_turbo(sources)
    loaded = load_copy(copy_dir, keys, "cpu")
    layout = _layout()
    inputs = _inputs(layout)
    counts = {"plain": (0, 0), "keyframe": (1, 0), "extend": (1, 2)}[row_set]

    from zvisiongenerator.backends.cuda.video.h3.schedule import row_timesteps

    with torch.no_grad():
        for t, audio_t in zip(video.timesteps, audio.timesteps, strict=True):
            unique, indices = row_timesteps(layout.video_indices, layout.audio_indices, layout.text_tokens, float(t), float(audio_t), condition_video_rows=counts[0], condition_audio_rows=counts[1])
            call = {**inputs, "timestep": unique, "timestep_indices": indices, "return_dict": False, **layout.kwargs}
            want = reference(**call)
            got = loaded(**call)
            assert torch.equal(got[0], want[0])
            assert torch.equal(got[1], want[1])
    assert not loaded.training


def test_a_subset_of_keys_reads_only_those_tables(sources: _Sources, copy_dir: Path, recorder: _Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    subset = _ALL_KEYS[3:6]
    seen: dict[str, object] = {}
    real_read, real_install = adaln_module.read_tables, adaln_module.install_fold

    def read(path, wanted, blocks):
        seen["wanted"] = list(wanted)
        return real_read(path, wanted, blocks)

    def install(transformer, tables):
        seen["installed"] = [list(block) for block in tables]
        return real_install(transformer, tables)

    monkeypatch.setattr(adaln_module, "read_tables", read)
    monkeypatch.setattr(adaln_module, "install_fold", install)

    model = load_copy(copy_dir, subset, "cpu")

    assert seen["wanted"] == subset
    assert seen["installed"] == [subset] * _LAYERS
    assert model.transformer_blocks[0].adaln_proj.count == 3


def test_a_key_the_copy_lacks_asks_for_a_rebuild(sources: _Sources, recorder: _Recorder, tmp_path: Path) -> None:
    built = _write(sources, tmp_path / "small", keys=_ALL_KEYS[:4])
    with pytest.raises(KeyError, match="rebuild the H3 copy"):
        load_copy(built, _ALL_KEYS[:5], "cpu")


def test_without_user_loras_the_order_is_fold_swap_fill_eval_stream(sources: _Sources, copy_dir: Path, recorder: _Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    events = recorder.events
    for name in ("read_tables", "install_fold"):
        real = getattr(adaln_module, name)
        monkeypatch.setattr(adaln_module, name, lambda *args, _real=real, _name=name, **kwargs: (events.append(_name), _real(*args, **kwargs))[1])
    monkeypatch.setattr(h3_lora, "merge_refiner", lambda *args: events.append("merge_refiner"))

    load_copy(copy_dir, _ALL_KEYS[:2], "cpu")

    assert events == ["read_tables", "install_fold", "swap", "fill", "stream", "move_outside"]


def _user_lora(path: Path, seed: int) -> Path:
    return _write_lora(path, _factors(_USER_SHAPES, 4, seed), metadata={"alpha": "2"})


@pytest.fixture
def ordered(monkeypatch: pytest.MonkeyPatch, recorder: _Recorder):
    """Record the user-LoRA steps in order, running the real refiner merge and table delta; PEFT and ``set_adapters`` are stand-ins."""
    from diffusers import MiniMaxH3Transformer3DModel

    events = recorder.events
    real_install, real_read = adaln_module.install_fold, adaln_module.read_tables
    real_merge, real_delta = h3_lora.merge_refiner, adaln_module.lora_table_delta
    adapters: dict[str, dict] = {}
    activated: list[tuple] = []
    real_eval = MiniMaxH3Transformer3DModel.eval

    def merge_refiner(model, refiner):
        events.append("merge_refiner")
        return real_merge(model, refiner)

    def delta(*args, **kwargs):
        events.append("adaln_delta")
        return real_delta(*args, **kwargs)

    def load_adapter(self, state, prefix=None, adapter_name=None, **kwargs):
        events.append(f"peft:{adapter_name}")
        adapters[adapter_name] = {"keys": sorted(state), "prefix": prefix}

    def set_adapters(self, names, weights=None):
        events.append("set_adapters")
        activated.append((list(names), list(weights)))

    def evaluate(self, *args, **kwargs):
        events.append("eval")
        return real_eval(self, *args, **kwargs)

    monkeypatch.setattr(adaln_module, "read_tables", lambda *args: (events.append("read_tables"), real_read(*args))[1])
    monkeypatch.setattr(adaln_module, "install_fold", lambda *args: (events.append("install_fold"), real_install(*args))[1])
    monkeypatch.setattr(adaln_module, "lora_table_delta", delta)
    monkeypatch.setattr(h3_lora, "merge_refiner", merge_refiner)
    monkeypatch.setattr(MiniMaxH3Transformer3DModel, "load_lora_adapter", load_adapter)
    monkeypatch.setattr(MiniMaxH3Transformer3DModel, "set_adapters", set_adapters)
    monkeypatch.setattr(MiniMaxH3Transformer3DModel, "eval", evaluate)
    return SimpleNamespace(events=events, adapters=adapters, activated=activated)


def test_user_loras_apply_in_the_documented_order(sources: _Sources, copy_dir: Path, recorder: _Recorder, ordered, tmp_path: Path) -> None:
    first = _user_lora(tmp_path / "first.safetensors", 11)
    second = _user_lora(tmp_path / "second.safetensors", 12)

    load_copy(copy_dir, _ALL_KEYS[:2], "cpu", loras=[(first, 0.75), (second, 1.5)], lora_names=["style", "detail"])

    one = ["merge_refiner", "adaln_delta", "adaln_delta"]  # two blocks carry an AdaLN delta
    assert ordered.events == ["read_tables", "install_fold", "swap", "fill", *one, "peft:style", *one, "peft:detail", "set_adapters", "eval", "stream", "move_outside"]
    assert ordered.activated == [(["style", "detail"], [0.75, 1.5])]
    assert ordered.adapters["style"]["prefix"] == "transformer"
    assert all(key.startswith("transformer.") for key in ordered.adapters["style"]["keys"])
    assert all("adaln_proj" not in key and "token_refiner" not in key for key in ordered.adapters["style"]["keys"])


def test_adapter_names_default_by_position_and_must_match(sources: _Sources, copy_dir: Path, recorder: _Recorder, ordered, tmp_path: Path) -> None:
    user = _user_lora(tmp_path / "user.safetensors", 11)

    load_copy(copy_dir, _ALL_KEYS[:2], "cpu", loras=[(user, 1.0)])
    assert ordered.activated == [(["lora_0"], [1.0])]
    with pytest.raises(ValueError, match="2 LoRA names for 1 LoRAs"):
        load_copy(copy_dir, _ALL_KEYS[:2], "cpu", loras=[(user, 1.0)], lora_names=["a", "b"])


def test_a_lora_without_a_peft_part_adds_no_adapter(sources: _Sources, copy_dir: Path, recorder: _Recorder, ordered, tmp_path: Path) -> None:
    only_adaln = _write_lora(tmp_path / "adaln.safetensors", _factors({"blocks.0.adaln_proj.linear": _USER_SHAPES["blocks.0.adaln_proj.linear"]}, 4, seed=5))

    load_copy(copy_dir, _ALL_KEYS[:2], "cpu", loras=[(only_adaln, 1.0)])

    assert "set_adapters" not in ordered.events
    assert not any(event.startswith("peft:") for event in ordered.events)
    assert ordered.events.count("adaln_delta") == 1


def test_the_refiner_and_the_adaln_tables_take_the_user_lora(sources: _Sources, copy_dir: Path, recorder: _Recorder, ordered, tmp_path: Path) -> None:
    user = _user_lora(tmp_path / "user.safetensors", 11)
    keys = _ALL_KEYS[:3]
    plain = load_copy(copy_dir, keys, "cpu")
    loaded = load_copy(copy_dir, keys, "cpu", loras=[(user, 0.5)])

    file = h3_lora.load_lora(user)
    parts = h3_lora.split_user_lora(file.state, 0.5, scales=file.scales)
    module = "token_refiner.refiner_blocks.0.attn.to_out.0"
    down, up, scale = parts.refiner[module]
    assert scale == 0.5 * 0.5  # weight times alpha / rank (2 / 4)
    assert torch.equal(loaded.get_submodule(module).weight, h3_lora.merge(plain.get_submodule(module).weight, down, up, scale))
    temb = {key: adaln_module.compute_temb(plain.time_proj, plain.time_embedder, key, "cpu") for key in keys}
    for block, (down, up, scale) in parts.adaln.items():
        base = {key: getattr(plain.transformer_blocks[block].adaln_proj, f"table{position}") for position, key in enumerate(keys)}
        want = adaln_module.lora_table_delta(base, temb, down, up, scale)
        for position, key in enumerate(keys):
            got = getattr(loaded.transformer_blocks[block].adaln_proj, f"table{position}")
            assert torch.equal(got, want[key])
            assert not torch.equal(got, base[key])


def test_an_adaln_target_outside_the_blocks_is_refused(sources: _Sources, copy_dir: Path, recorder: _Recorder, ordered, tmp_path: Path) -> None:
    far = _write_lora(tmp_path / "far.safetensors", _factors({"blocks.9.adaln_proj.linear": _USER_SHAPES["blocks.0.adaln_proj.linear"]}, 4, seed=5))
    with pytest.raises(ValueError, match="not a block of this model"):
        load_copy(copy_dir, _ALL_KEYS[:2], "cpu", loras=[(far, 1.0)])


def test_a_lora_that_matches_nothing_warns_and_is_not_activated(sources: _Sources, copy_dir: Path, recorder: _Recorder, ordered, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    peft = pytest.importorskip("peft")
    from diffusers import MiniMaxH3Transformer3DModel

    def unmatched(self, state, prefix=None, adapter_name=None, **kwargs):
        raise peft.NoMatchingPeftModuleError("no match")

    monkeypatch.setattr(MiniMaxH3Transformer3DModel, "load_lora_adapter", unmatched)
    user = _user_lora(tmp_path / "user.safetensors", 11)

    with pytest.warns(UserWarning, match=r"user\.safetensors: none of its LoRA tensors match"):
        load_copy(copy_dir, _ALL_KEYS[:2], "cpu", loras=[(user, 1.0)])

    assert "set_adapters" not in ordered.events


def test_another_peft_failure_names_the_file(sources: _Sources, copy_dir: Path, recorder: _Recorder, ordered, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from diffusers import MiniMaxH3Transformer3DModel

    def broken(self, state, prefix=None, adapter_name=None, **kwargs):
        raise ValueError("rank mismatch")

    monkeypatch.setattr(MiniMaxH3Transformer3DModel, "load_lora_adapter", broken)
    user = _user_lora(tmp_path / "user.safetensors", 11)
    with pytest.raises(RuntimeError, match=r"Failed to load LoRA .*user\.safetensors.*rank mismatch"):
        load_copy(copy_dir, _ALL_KEYS[:2], "cpu", loras=[(user, 1.0)])


# --- bitsandbytes ---------------------------------------------------------------------------------------------------


def test_the_packed_copy_loads_back_into_linear4bit_layers(sources: _Sources, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bnb = pytest.importorskip("bitsandbytes")
    if not torch.cuda.is_available():
        pytest.skip("bitsandbytes packs NF4 weights on a GPU")
    monkeypatch.setattr(stored_copy, "stream_blocks", lambda blocks, device: [])
    monkeypatch.setattr(stored_copy, "move_outside", lambda model, blocks, device: None)

    target = tmp_path / "packed"
    write_copy(sources.shards, sources.config, sources.turbo, _SCALE, _ALL_KEYS[:3], target, "cuda")
    loaded = load_copy(target, _ALL_KEYS[:3], "cpu")

    for block in loaded.transformer_blocks:
        for name in _BLOCK_LINEARS:
            layer = block.get_submodule(name)
            assert isinstance(layer, bnb.nn.Linear4bit)
            assert layer.weight.dtype == torch.uint8
            assert layer.weight.quant_state is not None
    refiner = loaded.token_refiner.refiner_blocks[0].attn.to_q
    assert type(refiner) is torch.nn.Linear
    assert refiner.weight.dtype == torch.bfloat16
    reference = _reference_with_turbo(sources)
    layer = loaded.transformer_blocks[0].attn.to_q
    state = layer.weight.quant_state
    state.to("cuda")  # in place
    dequantized = bnb.functional.dequantize_4bit(layer.weight.data.cuda(), state).cpu()
    wanted = reference.transformer_blocks[0].attn.to_q.weight.float()
    assert (dequantized.float() - wanted).abs().max() < 0.5 * wanted.abs().max()


def test_loading_needs_no_heavy_import_at_module_level() -> None:
    import subprocess
    import sys

    script = "import sys; import zvisiongenerator.backends.cuda.video.h3.stored_copy; sys.exit(int('bitsandbytes' in sys.modules or 'diffusers' in sys.modules))"
    assert subprocess.run([sys.executable, "-c", script], check=False).returncode == 0


def test_public_names_are_exported() -> None:
    assert sorted(stored_copy.__all__) == ["COPY_NAME", "copy_manifest", "ensure_copy", "load_copy", "write_copy"]
