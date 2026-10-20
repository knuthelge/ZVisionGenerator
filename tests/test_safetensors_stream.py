"""Tests for reading and writing safetensors one tensor at a time."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from safetensors.torch import load_file, save_file  # noqa: E402

from zvisiongenerator.utils import safetensors_stream  # noqa: E402
from zvisiongenerator.utils.safetensors_stream import ShardWriter, SingleFileWriter, iter_tensors, read_metadata, shard_files  # noqa: E402


def _tensors():
    return {
        "a.weight": torch.randn(4, 3).to(torch.bfloat16),
        "b.bias": torch.randn(5),
        "c.packed": torch.arange(6, dtype=torch.uint8),
        "d.fp8": torch.randn(2, 2).to(torch.float8_e4m3fn),
        "e.index": torch.tensor([1, 2, 3], dtype=torch.int64),
    }


class TestIterTensors:
    def test_every_tensor_reads_back_equal_in_dtype_and_shape(self, tmp_path):
        tensors = _tensors()
        save_file(tensors, str(tmp_path / "w.safetensors"))

        read = dict(iter_tensors(tmp_path / "w.safetensors"))

        assert read.keys() == tensors.keys()
        for name, tensor in tensors.items():
            assert read[name].dtype == tensor.dtype
            assert read[name].shape == tensor.shape
            assert torch.equal(read[name].view(torch.uint8), tensor.view(torch.uint8))

    def test_only_wanted_tensors_are_read(self, tmp_path):
        save_file(_tensors(), str(tmp_path / "w.safetensors"))

        names = [name for name, _ in iter_tensors(tmp_path / "w.safetensors", lambda name: name.endswith("weight"))]

        assert names == ["a.weight"]

    def test_read_tensors_are_writable_and_independent_of_the_file(self, tmp_path):
        path = tmp_path / "w.safetensors"
        save_file({"x": torch.zeros(3)}, str(path))

        (_, tensor), *_ = iter_tensors(path)
        path.unlink()
        tensor += 1

        assert torch.equal(tensor, torch.ones(3))

    @pytest.mark.skipif(not hasattr(os, "posix_fadvise"), reason="page-cache hints are Linux only")
    def test_each_read_range_leaves_the_page_cache(self, tmp_path):
        save_file(_tensors(), str(tmp_path / "w.safetensors"))

        with patch.object(safetensors_stream.os, "posix_fadvise") as fadvise:
            list(iter_tensors(tmp_path / "w.safetensors"))

        assert fadvise.call_count == len(_tensors())
        assert all(call.args[3] == os.POSIX_FADV_DONTNEED for call in fadvise.call_args_list)


class TestShardWriter:
    def test_shards_stay_under_the_limit_and_the_index_maps_every_tensor(self, tmp_path):
        writer = ShardWriter(tmp_path, "model", "model.safetensors.index.json", max_shard_bytes=100)
        for number in range(5):
            writer.add({f"t{number}": torch.zeros(10)})  # 40 bytes each
        writer.close()

        files = shard_files(tmp_path, "model.safetensors.index.json")
        assert [path.name for path in files] == [f"model-0000{n}-of-00003.safetensors" for n in (1, 2, 3)]
        index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
        assert sorted(index["weight_map"]) == [f"t{n}" for n in range(5)]
        assert index["metadata"]["total_size"] == 200
        assert all(len(load_file(str(path))) <= 2 for path in files)

    def test_a_group_is_never_split_across_shards(self, tmp_path):
        writer = ShardWriter(tmp_path, "model", "index.json", max_shard_bytes=100)
        writer.add({"first": torch.zeros(20)})  # 80 bytes
        writer.add({"w": torch.zeros(10), "w.state": torch.zeros(10)})  # 80 bytes together
        writer.close()

        index = json.loads((tmp_path / "index.json").read_text())["weight_map"]
        assert index["w"] == index["w.state"] != index["first"]

    def test_written_shards_read_back_through_the_index(self, tmp_path):
        tensors = _tensors()
        writer = ShardWriter(tmp_path, "model", "index.json", max_shard_bytes=64)
        for name, tensor in tensors.items():
            writer.add({name: tensor})
        writer.close()

        read = {name: tensor for path in shard_files(tmp_path, "index.json") for name, tensor in iter_tensors(path)}

        assert read.keys() == tensors.keys()
        assert all(torch.equal(read[name].view(torch.uint8), tensor.view(torch.uint8)) for name, tensor in tensors.items())


class TestSingleFileWriter:
    def _entries(self):
        return [("w", "BF16", [2, 3]), ("s", "F32", []), ("q", "U32", [2, 1])]

    def test_the_header_matches_the_data_streamed_after_it(self, tmp_path):
        path = tmp_path / "one.safetensors"
        w, s, q = torch.randn(2, 3).to(torch.bfloat16), torch.tensor(1.5), torch.tensor([[7], [9]], dtype=torch.int32)
        with SingleFileWriter(path, self._entries()) as writer:
            writer.write("w", w.view(torch.uint8).numpy())
            writer.write("s", s.reshape(1).view(torch.uint8).numpy())
            writer.write("q", q.numpy())

        read = load_file(str(path))
        assert torch.equal(read["w"], w)
        assert read["s"].item() == 1.5 and read["s"].shape == ()
        assert read["q"].view(torch.int32).tolist() == [[7], [9]]

    def test_tensors_must_come_in_the_declared_order_and_size(self, tmp_path):
        writer = SingleFileWriter(tmp_path / "one.safetensors", self._entries())

        with pytest.raises(ValueError, match="where w comes next"):
            writer.write("s", bytes(4))
        with pytest.raises(ValueError, match="its header says 12"):
            writer.write("w", bytes(4))
        with pytest.raises(ValueError, match="missing w"):
            writer.close()

    def test_closing_before_every_tensor_is_written_fails(self, tmp_path):
        writer = SingleFileWriter(tmp_path / "one.safetensors", self._entries())
        writer.write("w", bytes(12))

        with pytest.raises(ValueError, match="missing s"):
            writer.close()


def test_metadata_reads_back_and_is_empty_when_absent(tmp_path):
    save_file({"a": torch.zeros(1)}, str(tmp_path / "m.safetensors"), metadata={"config": "{}"})
    save_file({"a": torch.zeros(1)}, str(tmp_path / "n.safetensors"))

    assert read_metadata(tmp_path / "m.safetensors") == {"config": "{}"}
    assert read_metadata(tmp_path / "n.safetensors") == {}
