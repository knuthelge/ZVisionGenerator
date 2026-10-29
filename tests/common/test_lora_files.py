"""Tests for handing a loader a LoRA file under the tensor names its model family reads."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from zvisiongenerator.utils import lora_files  # noqa: E402
from zvisiongenerator.utils.lora_files import normalised_lora_state, readable_lora_files  # noqa: E402
from zvisiongenerator.utils.safetensors_stream import read_header, read_metadata  # noqa: E402


def _write(path, tensors, metadata=None):
    save_file(tensors, str(path), metadata=metadata)
    return path


def _peft(module="layers.0.proj.q", adapter="default", rank=4):
    return {f"{module}.lora_A.{adapter}.weight": torch.randn(rank, 8), f"{module}.lora_B.{adapter}.weight": torch.randn(8, rank)}


def _kohya_krea2(alpha=8.0, rank=4):
    return {
        "lora_unet_blocks_0_attn_gate.lora_down.weight": torch.randn(rank, 8),
        "lora_unet_blocks_0_attn_gate.lora_up.weight": torch.randn(8, rank),
        "lora_unet_blocks_0_attn_gate.alpha": torch.tensor(alpha),
    }


class TestNormalisedState:
    def test_a_file_whose_names_are_read_as_they_are_gives_none(self, tmp_path):
        path = _write(tmp_path / "a.safetensors", {"layers.0.proj.q.lora_A.weight": torch.zeros(4, 8), "layers.0.proj.q.lora_B.weight": torch.zeros(8, 4)})

        assert normalised_lora_state(path, "moss", "moss") is None

    def test_the_adapter_name_is_dropped_and_the_tensors_are_kept(self, tmp_path):
        tensors = _peft()
        path = _write(tmp_path / "a.safetensors", tensors)

        state = normalised_lora_state(path, "moss", "moss")

        assert set(state) == {"layers.0.proj.q.lora_A.weight", "layers.0.proj.q.lora_B.weight"}
        assert torch.equal(state["layers.0.proj.q.lora_A.weight"], tensors["layers.0.proj.q.lora_A.default.weight"])

    def test_kohya_alpha_is_folded_into_lora_a_for_a_family_whose_converter_cannot_read_it(self, tmp_path):
        tensors = _kohya_krea2(alpha=2.0, rank=4)
        path = _write(tmp_path / "k.safetensors", tensors)

        state = normalised_lora_state(path, "krea2")

        assert set(state) == {"diffusion_model.blocks.0.attn.gate.lora_A.weight", "diffusion_model.blocks.0.attn.gate.lora_B.weight"}
        assert torch.allclose(state["diffusion_model.blocks.0.attn.gate.lora_A.weight"], tensors["lora_unet_blocks_0_attn_gate.lora_down.weight"] * 0.5)
        assert torch.equal(state["diffusion_model.blocks.0.attn.gate.lora_B.weight"], tensors["lora_unet_blocks_0_attn_gate.lora_up.weight"])

    def test_alpha_is_folded_even_when_no_name_changes(self, tmp_path):
        tensors = {
            "diffusion_model.blocks.0.attn.wq.lora_A.weight": torch.ones(4, 8),
            "diffusion_model.blocks.0.attn.wq.lora_B.weight": torch.ones(8, 4),
            "diffusion_model.blocks.0.attn.wq.alpha": torch.tensor(1.0),
        }
        path = _write(tmp_path / "a.safetensors", tensors)

        state = normalised_lora_state(path, "krea2")

        assert "diffusion_model.blocks.0.attn.wq.alpha" not in state
        assert torch.allclose(state["diffusion_model.blocks.0.attn.wq.lora_A.weight"], torch.full((4, 8), 0.25))

    def test_alpha_stays_for_a_family_whose_converter_reads_it(self, tmp_path):
        tensors = {**_peft(), "layers.0.proj.q.alpha": torch.tensor(2.0)}
        path = _write(tmp_path / "a.safetensors", tensors)

        state = normalised_lora_state(path, "moss", "moss")

        assert state["layers.0.proj.q.alpha"].item() == 2.0

    def test_an_alpha_without_a_lora_pair_stays(self, tmp_path):
        path = _write(tmp_path / "a.safetensors", {"diffusion_model.blocks.0.attn.wq.lokr_w1": torch.zeros(2, 2), "diffusion_model.blocks.0.attn.wq.alpha": torch.tensor(1.0)})

        assert normalised_lora_state(path, "krea2")["diffusion_model.blocks.0.attn.wq.alpha"].item() == 1.0

    @pytest.mark.parametrize("name", ["style.ckpt", "style.bin", "missing.safetensors"])
    def test_a_file_that_is_not_a_readable_safetensors_file_gives_none(self, tmp_path, name):
        (tmp_path / "style.ckpt").write_bytes(b"x")
        (tmp_path / "style.bin").write_bytes(b"x")

        assert normalised_lora_state(tmp_path / name, "krea2") is None

    def test_a_safetensors_file_that_is_corrupt_gives_none(self, tmp_path):
        path = tmp_path / "bad.safetensors"
        path.write_bytes(b"not a header")

        assert normalised_lora_state(path, "krea2") is None


class TestReadableFiles:
    def test_a_file_whose_names_are_read_as_they_are_is_handed_over_as_it_is(self, tmp_path):
        path = _write(
            tmp_path / "a.safetensors", {"diffusion_model.layers.0.attention.to_q.lora_A.weight": torch.zeros(4, 8), "diffusion_model.layers.0.attention.to_q.lora_B.weight": torch.zeros(8, 4)}
        )

        with readable_lora_files([path], "zimage", "mflux", tmp_path / "cache") as readable:
            assert readable == [str(path)]
        assert not (tmp_path / "cache").exists()

    def test_a_path_that_is_not_a_safetensors_file_is_handed_over_as_it_is(self, tmp_path):
        with readable_lora_files(["lora.safetensors", tmp_path], "ltx", "mlx", tmp_path / "cache") as readable:
            assert readable == ["lora.safetensors", str(tmp_path)]

    def test_no_files_need_no_folder_and_no_copy(self, tmp_path):
        with readable_lora_files([], "ltx", "mlx", tmp_path / "cache") as readable:
            assert readable == []
        assert not (tmp_path / "cache").exists()

    def test_the_paths_come_back_in_order_with_only_the_files_that_need_it_copied(self, tmp_path):
        plain = _write(tmp_path / "plain.safetensors", {"transformer_blocks.0.attn1.to_q.lora_A.weight": torch.zeros(4, 8), "transformer_blocks.0.attn1.to_q.lora_B.weight": torch.zeros(8, 4)})
        peft = _write(tmp_path / "peft.safetensors", _peft("diffusion_model.transformer_blocks.0.attn1.to_q"))

        with readable_lora_files([peft, plain, peft], "ltx", "mlx", tmp_path / "cache") as readable:
            assert readable[1] == str(plain)
            assert readable[0] != str(peft) != readable[2]
            assert readable[0] != readable[2]

    def test_the_copy_has_the_new_names_and_the_same_tensors_metadata_and_alpha(self, tmp_path):
        tensors = {**_peft("diffusion_model.transformer_blocks.0.attn1.to_q", rank=4), "diffusion_model.transformer_blocks.0.attn1.to_q.alpha": torch.tensor(3.0)}
        source = _write(tmp_path / "a.safetensors", tensors, metadata={"ss_note": "keep me"})

        with readable_lora_files([source], "ltx", "mlx", tmp_path / "cache") as [copy]:
            assert copy != str(source)
            renamed = load_file(copy)
            assert set(renamed) == {
                "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight",
                "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight",
                "diffusion_model.transformer_blocks.0.attn1.to_q.alpha",
            }
            assert torch.equal(renamed["diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight"], tensors["diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.default.weight"])
            assert renamed["diffusion_model.transformer_blocks.0.attn1.to_q.alpha"].item() == 3.0
            with safe_open(copy, framework="pt") as handle:
                assert handle.metadata()["ss_note"] == "keep me"
            assert read_metadata(Path(copy)) == {"ss_note": "keep me"}

    def test_the_copy_keeps_each_tensors_dtype(self, tmp_path):
        tensors = {"layers.0.proj.q.lora_A.default.weight": torch.randn(4, 8).to(torch.bfloat16), "layers.0.proj.q.lora_B.default.weight": torch.randn(8, 4).to(torch.float16)}
        source = _write(tmp_path / "a.safetensors", tensors)

        with readable_lora_files([source], "ltx", "mlx", tmp_path / "cache") as [copy]:
            renamed = load_file(copy)

        assert [renamed[name].dtype for name in sorted(renamed)] == [torch.bfloat16, torch.float16]

    def test_the_copy_header_is_valid_and_aligned(self, tmp_path):
        source = _write(tmp_path / "a.safetensors", _peft())

        with readable_lora_files([source], "ltx", "mlx", tmp_path / "cache") as [copy]:
            data_start, header = read_header(Path(copy))

        assert data_start % 8 == 0
        assert all(entry["data_offsets"][1] >= entry["data_offsets"][0] for entry in header.values())

    def test_the_copy_goes_to_the_data_folder_by_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path / "data"))
        source = _write(tmp_path / "a.safetensors", _peft())

        with readable_lora_files([source], "ltx", "mlx") as [copy]:
            assert copy.startswith(str(tmp_path / "data" / "cache" / "loras"))

    def test_the_copies_are_removed_when_the_block_ends(self, tmp_path):
        source = _write(tmp_path / "a.safetensors", _peft())
        cache = tmp_path / "cache"

        with readable_lora_files([source, source], "ltx", "mlx", cache) as copies:
            assert all(Path(copy).is_file() for copy in copies)

        assert list(cache.iterdir()) == []
        assert source.is_file()

    def test_the_copies_are_removed_when_the_block_raises(self, tmp_path):
        source = _write(tmp_path / "a.safetensors", _peft())
        cache = tmp_path / "cache"

        with pytest.raises(RuntimeError, match="load failed"), readable_lora_files([source], "ltx", "mlx", cache):
            raise RuntimeError("load failed")

        assert list(cache.iterdir()) == []

    def test_the_copy_keeps_the_lora_file_name(self, tmp_path):
        source = _write(tmp_path / "my_style.safetensors", _peft())

        with readable_lora_files([source], "ltx", "mlx", tmp_path / "cache") as [copy]:
            assert Path(copy).name == "my_style.safetensors"
            assert Path(copy) != source

    def test_two_jobs_at_once_keep_separate_copies(self, tmp_path):
        source = _write(tmp_path / "a.safetensors", _peft())
        cache = tmp_path / "cache"

        with readable_lora_files([source], "ltx", "mlx", cache) as [first], readable_lora_files([source], "ltx", "mlx", cache) as [second]:
            assert first != second
            assert Path(first).is_file()
            assert Path(second).is_file()


@pytest.mark.skipif(lora_files.fcntl is None, reason="copies are swept with flock, which this platform lacks")
class TestSweep:
    def test_a_copy_left_by_a_crashed_job_is_removed_by_the_next_one(self, tmp_path):
        cache = tmp_path / "cache"
        (cache / "crashed").mkdir(parents=True)
        (cache / "crashed" / "style.safetensors").write_bytes(b"x")
        (cache / "half").mkdir()
        (cache / "half" / ".style.safetensors.tmp").write_bytes(b"x")
        source = _write(tmp_path / "a.safetensors", _peft())

        with readable_lora_files([source], "ltx", "mlx", cache) as [copy]:
            assert [path.name for path in cache.iterdir()] == [Path(copy).parent.name]

    def test_the_sweep_runs_for_a_job_whose_files_need_no_copy(self, tmp_path):
        cache = tmp_path / "cache"
        (cache / "crashed").mkdir(parents=True)
        (cache / "crashed" / "style.safetensors").write_bytes(b"x")
        plain = _write(tmp_path / "a.safetensors", {"transformer_blocks.0.attn1.to_q.lora_A.weight": torch.zeros(4, 8), "transformer_blocks.0.attn1.to_q.lora_B.weight": torch.zeros(8, 4)})

        with readable_lora_files([plain], "ltx", "mlx", cache):
            pass

        assert list(cache.iterdir()) == []

    def test_a_copy_held_by_a_process_that_is_gone_is_removed_and_one_still_held_is_not(self, tmp_path):
        cache = tmp_path / "cache"
        (cache / "running").mkdir(parents=True)
        held = cache / "running" / "held.safetensors"
        held.write_bytes(b"x")
        source = _write(tmp_path / "a.safetensors", _peft())

        with open(held, "rb") as owner:
            lora_files.fcntl.flock(owner, lora_files.fcntl.LOCK_EX | lora_files.fcntl.LOCK_NB)
            with readable_lora_files([source], "ltx", "mlx", cache):
                assert held.is_file()

        with readable_lora_files([source], "ltx", "mlx", cache):
            assert not held.exists()
