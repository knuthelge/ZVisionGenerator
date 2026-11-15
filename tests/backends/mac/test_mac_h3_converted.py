"""Tests for writing converted MiniMax H3 folders for macOS, on a tiny config with a stand-in quantizer (no MLX)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from tests._h3_states import _TINY

from zvisiongenerator.backends.mac import h3_converted
from zvisiongenerator.backends.mac.h3_converted import native_tensors, write_converted
from zvisiongenerator.converters import h3_checkpoint
from zvisiongenerator.converters.h3_checkpoint import diffusers_tensors, inspect_h3
from zvisiongenerator.downloads.target import HubFiles
from zvisiongenerator.utils.h3_converted import read_converted_h3
from zvisiongenerator.utils.h3_preset import h3_turbo_lora
from zvisiongenerator.utils.safetensors_stream import iter_tensors, shard_files
from zvisiongenerator.utils.stored_quant import MANIFEST_NAME


@pytest.fixture(autouse=True)
def _tiny_config(monkeypatch):
    monkeypatch.setattr(h3_checkpoint, "STOCK_CONFIG", _TINY)
    monkeypatch.setattr(h3_converted, "STOCK_CONFIG", _TINY)


def _state(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {name: torch.randn(shape, generator=generator).to(torch.bfloat16) for name, shape in h3_checkpoint._expected_shapes(None).items()}


def _fake_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stand in for ``mx.quantize`` at 8 bits, groups of 32 rounded down to the tiny widths: the shapes and dtypes it returns."""
    rows, columns = weight.shape
    return torch.zeros(rows, max(columns // 4, 1), dtype=torch.uint32), torch.ones(rows, 1, dtype=torch.bfloat16), torch.zeros(rows, 1, dtype=torch.bfloat16)


def test_native_names_round_trip_through_the_reader(tmp_path):
    state = _state(0)
    native = dict(native_tensors(state.items()))
    save_file(native, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file({"rope.inv_freq": torch.ones(2)}, str(tmp_path / "model-00002-of-00002.safetensors"))
    weight_map = {**dict.fromkeys(native, "model-00001-of-00002.safetensors"), "rope.inv_freq": "model-00002-of-00002.safetensors"}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    checkpoint = inspect_h3(tmp_path / "model-00001-of-00002.safetensors")
    found = dict(diffusers_tensors(checkpoint))

    assert checkpoint.layout == "native"  # a sharded native set: QKV interleaved per head, as the MLX port reads it
    assert found.keys() == state.keys()
    assert all(torch.equal(found[name], state[name]) for name in state)


def _components(tmp_path: Path) -> Path:
    folder = tmp_path / "stock" / "FL2VA"
    (folder / "vae").mkdir(parents=True)
    (folder / "text_encoder").mkdir()
    (folder / "model_index.json").write_text("{}")
    (folder / "vae" / "weights.safetensors").write_bytes(b"vae")
    (folder / "text_encoder" / "model-00001-of-00014.safetensors").write_bytes(b"te")
    return folder


def _turbo(tmp_path: Path, module: str, rows: int, columns: int) -> Path:
    path = tmp_path / "turbo.safetensors"
    save_file({f"transformer.{module}.lora_A.weight": torch.ones(1, columns), f"transformer.{module}.lora_B.weight": torch.ones(rows, 1), f"transformer.{module}.alpha": torch.tensor(99.0)}, str(path))
    return path


@pytest.mark.parametrize("turbo", [True, False])
def test_the_folder_holds_an_int8_transformer_the_port_loads_and_linked_components(tmp_path, turbo):
    state = _state(1)
    source = tmp_path / "ft.safetensors"
    save_file(state, str(source))
    lora = None if turbo else _turbo(tmp_path, "transformer_blocks.0.attn.to_out.0", 8, 8)
    target = tmp_path / "out"

    write_converted(inspect_h3(source), target, components=_components(tmp_path), lora_file=lora, lora_scale=0.5, time_embedder=None, quantize=_fake_quantize, report=lambda *_: None)

    transformer = target / "transformer"
    stored = {name: tensor for path in shard_files(transformer, "model.safetensors.index.json") for name, tensor in iter_tensors(path)}
    assert {"blocks.0.attn.qkv_proj.weight", "blocks.0.attn.qkv_proj.scales", "blocks.1.adaln_proj.linear.biases", "token_refiner.blocks.1.mlp.fc1.scales"} <= stored.keys()
    assert stored["blocks.0.attn.qkv_proj.weight"].dtype == torch.uint32
    assert "final_layer.adaln_proj.linear.scales" not in stored  # the output AdaLN stays unquantized, as the recipe keeps it
    assert stored["final_layer.adaln_proj.linear.weight"].dtype == torch.bfloat16
    assert stored["time_embedder.proj_in.weight"].dtype == torch.float32
    assert stored["video_patch_proj.weight"].dtype == torch.float32
    quant = json.loads((transformer / "quant_config.json").read_text())
    assert (quant["bits"], quant["group_size"], quant["quantize_adaln"]) == (8, 32, True)
    assert json.loads((transformer / "config.json").read_text())["token_refiner_num_layers"] == 2
    assert (target / "FL2VA" / "vae" / "weights.safetensors").read_bytes() == b"vae"
    assert not (target / "FL2VA" / "text_encoder").exists()  # read from the stock download, so stock's copy is shared
    manifest = read_converted_h3(target)
    assert manifest is not None
    assert (manifest["storage"], manifest["turbo"]) == ("mlx-int8", "distilled" if turbo else "merged")


def test_the_turbo_lora_is_merged_at_the_config_scale_replacing_alpha(tmp_path):
    state = _state(2)
    source = tmp_path / "ft.safetensors"
    save_file(state, str(source))
    lora = _turbo(tmp_path, "context_embedder", 8, 4)  # an unquantized module, so the merged weight can be read back
    out = tmp_path / "out"

    write_converted(inspect_h3(source), out, components=_components(tmp_path), lora_file=lora, lora_scale=0.5, time_embedder=None, quantize=_fake_quantize, report=lambda *_: None)

    stored = {name: tensor for path in shard_files(out / "transformer", "model.safetensors.index.json") for name, tensor in iter_tensors(path)}
    expected = (state["context_embedder.weight"].float() + 0.5).to(torch.bfloat16)
    assert torch.equal(stored["condition_proj.weight"], expected)


def test_a_turbo_lora_for_another_model_is_refused(tmp_path):
    source = tmp_path / "ft.safetensors"
    save_file(_state(3), str(source))
    lora = _turbo(tmp_path, "transformer_blocks.9.attn.to_out.0", 8, 8)

    with pytest.raises(RuntimeError):
        write_converted(inspect_h3(source), tmp_path / "out", components=_components(tmp_path), lora_file=lora, lora_scale=0.5, time_embedder=None, quantize=_fake_quantize, report=lambda *_: None)


def test_stocks_time_embedder_is_range_read_from_the_one_int8_shard_holding_it(tmp_path):
    from zvisiongenerator.converters.h3_checkpoint import stock_time_embedder

    tensors = {f"time_embedder.{name}": torch.full((2,), float(index)) for index, name in enumerate(("proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias"))}
    save_file({**tensors, "blocks.0.attn.qkv_proj.weight": torch.zeros(2, dtype=torch.uint32)}, str(tmp_path / "model-00001-of-00001.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": dict.fromkeys([*tensors, "blocks.0.attn.qkv_proj.weight"], "model-00001-of-00001.safetensors")}))

    fetched: list[str] = []
    opened: list[str] = []

    def download_file(*, repo_id, filename, revision):
        fetched.append(filename)
        return str(tmp_path / filename)

    def open_remote(repo_id, path, revision):
        opened.append(path)
        return open(tmp_path / path, "rb"), (tmp_path / path).stat().st_size

    state = stock_time_embedder("org/int8", "abc", index="model.safetensors.index.json", native=True, download_file=download_file, open_remote=open_remote)

    assert {name: tensor[0].item() for name, tensor in state.items()} == {"linear_1.weight": 0.0, "linear_1.bias": 1.0, "linear_2.weight": 2.0, "linear_2.bias": 3.0}
    assert fetched == ["model.safetensors.index.json"]  # only the index is downloaded
    assert opened == ["model-00001-of-00001.safetensors"]  # the one shard holding it is read by range


class _Writes:
    """The shared writer replaced: records the LoRA file and scale each folder was written with."""

    def __init__(self, monkeypatch, tmp_path: Path) -> None:
        self.models = tmp_path / "models"
        self.calls: list[tuple[Path | None, float]] = []
        self.flushed: list[bool] = []
        self.transformer = tmp_path / "ref2va.safetensors"
        self.transformer.write_bytes(b"transformer")
        self.lora = tmp_path / "ref2v-turbo.safetensors"
        self.lora.write_bytes(b"lora")
        self.fl2va = tmp_path / "fl2va-turbo.safetensors"
        self.fl2va.write_bytes(b"fl2va")
        self.preset = SimpleNamespace(reference_transformer=HubFiles("org/ref", "r2", ("ref2va.safetensors",)), reference_lora=HubFiles("org/turbo", "r3", ("ref2v-turbo.safetensors",)))
        files = {"ref2va.safetensors": self.transformer, "ref2v-turbo.safetensors": self.lora, h3_turbo_lora("darwin")[0].files[0]: self.fl2va}
        monkeypatch.setattr(h3_converted, "snapshot_file", lambda hub, download: files[hub.files[0]])
        monkeypatch.setattr(h3_converted, "inspect_h3", lambda path: SimpleNamespace(path=path))
        monkeypatch.setattr(h3_converted, "_write", self._write)
        real = h3_converted.ensure_stored_copy
        monkeypatch.setattr(h3_converted, "ensure_stored_copy", lambda *args, flush=False, **kwargs: (self.flushed.append(flush), real(*args, flush=flush, **kwargs))[1])

    def _write(self, checkpoint, target, preset, lora_file, lora_scale, report) -> None:
        self.calls.append((lora_file, lora_scale))
        (target / "weights").write_text("x")

    def ensure(self, scale: float = 0.0625) -> Path:
        return h3_converted.ensure_ref2va_copy(self.models, self.preset, scale)


@pytest.fixture
def writes(monkeypatch, tmp_path):
    return _Writes(monkeypatch, tmp_path)


def test_the_ref2va_copy_is_built_once_with_its_own_lora_and_flushed(writes):
    copy = writes.ensure()

    assert copy == writes.models / "h3@ref2va-q8"
    assert writes.calls == [(writes.lora, 0.0625)]
    assert (copy / "weights").is_file() and (copy / MANIFEST_NAME).is_file()
    assert writes.flushed == [True]


def test_a_current_ref2va_copy_is_reused(writes):
    first = writes.ensure()

    assert writes.ensure() == first
    assert len(writes.calls) == 1


def test_a_changed_lora_or_scale_rebuilds_the_ref2va_copy(writes):
    writes.ensure()
    writes.lora.write_bytes(b"a different lora")
    writes.ensure()
    writes.ensure(scale=0.125)

    assert [scale for _lora, scale in writes.calls] == [0.0625, 0.0625, 0.125]


def test_an_interrupted_ref2va_write_leaves_no_copy(writes, monkeypatch):
    def fail(checkpoint, target, preset, lora_file, lora_scale, report):
        (target / "weights").write_text("half")
        raise RuntimeError("out of disk")

    monkeypatch.setattr(h3_converted, "_write", fail)

    with pytest.raises(RuntimeError, match="out of disk"):
        writes.ensure()

    assert not (writes.models / "h3@ref2va-q8").exists()


@pytest.mark.parametrize("turbo", [True, False])
def test_convert_still_merges_the_fl2va_lora_unless_the_checkpoint_is_distilled(writes, tmp_path, turbo):
    (tmp_path / "out").mkdir()

    h3_converted.convert(SimpleNamespace(), tmp_path / "out", turbo=turbo, report=lambda *_: None)

    assert writes.calls == [(None, 0.0) if turbo else (writes.fl2va, h3_turbo_lora("darwin")[1])]
