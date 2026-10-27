"""Tests for the macOS MiniMax H3 backend, against a fake ``minimax_h3_mlx`` and a fake ``mlx``: no Apple hardware, no models."""

from __future__ import annotations

import ast
import dataclasses
import gc
import json
import re
import sys
import types
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from tests._factories import make_extend_source  # noqa: E402
from tests._media import REAL, ffmpeg, numbered_frames  # noqa: E402
from tests._paths import REPO_ROOT  # noqa: E402

from zvisiongenerator.backends.mac import h3_video  # noqa: E402
from zvisiongenerator.backends.mac.h3_video import (  # noqa: E402
    H3_MLX_COMMIT,
    H3MlxVideoBackend,
    combine_loras,
    ensure_text_encoder_copy,
    resolve_mlx_files,
)
from zvisiongenerator.core.video_backend import VideoBackend  # noqa: E402
from zvisiongenerator.core.video_family import FrameRule  # noqa: E402
from zvisiongenerator.core.video_types import ExtendSegment  # noqa: E402
from zvisiongenerator.utils import hf_snapshot  # noqa: E402
from zvisiongenerator.utils.h3_preset import H3MlxFiles, H3MlxPreset, H3Schedule, HubFiles  # noqa: E402
from zvisiongenerator.utils.video_extend import build_extend_source  # noqa: E402
from zvisiongenerator.utils.video_probe import probe_video  # noqa: E402
from zvisiongenerator.utils.video_splice import splice_clips  # noqa: E402

HIDDEN = 8
FF = 6
# projection -> (output rows, input columns)
TARGET_SHAPES = {
    "attn.to_q": (HIDDEN, HIDDEN),
    "attn.to_k": (HIDDEN, HIDDEN),
    "attn.to_v": (HIDDEN, HIDDEN),
    "attn.to_out.0": (HIDDEN, HIDDEN),
    "ff.net.0.proj": (2 * FF, HIDDEN),
    "ff.net.2": (HIDDEN, FF),
}
TURBO_RANK = 4
PRESET = H3MlxPreset(
    transformer=HubFiles("int8/repo", "rev-int8", ("config.json",)),
    components=HubFiles("main/repo", "rev-main", ("FL2VA/model_index.json",)),
    text_encoder_bits=4,
    text_encoder_group_size=64,
    text_encoder_layers=50,
    frame_rule=FrameRule(17, 5, "up"),
    schedules={4: H3Schedule(4, 6.0, 3.0), 8: H3Schedule(8, 6.0, 3.0)},
)
TURBO_SCALE = 0.0625
LORA = HubFiles("lora/repo", "rev-lora", ("turbo.safetensors",))
MODEL_REF = "MiniMaxAI/MiniMax-H3@rev-main"


# ---- Helpers ---------------------------------------------------------------------------------------------------------


def _bf16_exact(tensor: torch.Tensor) -> torch.Tensor:
    """Return *tensor* rounded to bf16, so scaled sums of products in fp32 stay exact."""
    return tensor.to(torch.bfloat16)


def _write_turbo(path: Path, *, rank: int = TURBO_RANK, blocks: int = 2, refiners: int = 1, seed: int = 0, metadata: dict[str, str] | None = None) -> Path:
    """Write a turbo-style adapter: PEFT diffusers keys for every projection of every block and refiner block."""
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for family, count in (("transformer_blocks", blocks), ("token_refiner.refiner_blocks", refiners)):
        for index in range(count):
            for target, (rows, columns) in TARGET_SHAPES.items():
                prefix = f"base_model.model.{family}.{index}.{target}"
                tensors[f"{prefix}.lora_A.default.weight"] = _bf16_exact(torch.randn(rank, columns, generator=generator))
                tensors[f"{prefix}.lora_B.default.weight"] = _bf16_exact(torch.randn(rows, rank, generator=generator))
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata or {"alpha": "8"})
    return path


def _write_kohya(path: Path, modules: dict[str, tuple[int, int]], rank: int, alpha: float, seed: int = 1) -> Path:
    """Write a musubi/kohya file (original module names, flattened; a per-module ``.alpha`` tensor) covering only *modules*."""
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for module, (rows, columns) in modules.items():
        base = "lora_unet_" + module.replace(".", "_")
        tensors[f"{base}.lora_down.weight"] = torch.randn(rank, columns, generator=generator)
        tensors[f"{base}.lora_up.weight"] = torch.randn(rows, rank, generator=generator)
        tensors[f"{base}.alpha"] = torch.tensor(float(alpha))
    save_file(tensors, str(path))
    return path


def _write_peft(path: Path, modules: dict[str, tuple[int, int]], rank: int, metadata: dict[str, str], seed: int = 2) -> Path:
    """Write an ai-toolkit style file whose factors are bf16-exact."""
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for module, (rows, columns) in modules.items():
        tensors[f"diffusion_model.{module}.lora_A.weight"] = _bf16_exact(torch.randn(rank, columns, generator=generator)).float()
        tensors[f"diffusion_model.{module}.lora_B.weight"] = _bf16_exact(torch.randn(rows, rank, generator=generator)).float()
    save_file(tensors, str(path), metadata=metadata)
    return path


def _delta(tensors: dict[str, torch.Tensor], module: str) -> torch.Tensor:
    return tensors[f"{module}.lora_B.default.weight"].float() @ tensors[f"{module}.lora_A.default.weight"].float()


def _raw_delta(tensors: dict[str, torch.Tensor], module: str, scale: float) -> torch.Tensor:
    """Return ``scale * B @ A`` of the turbo file's factors (keys carry the file's ``base_model.model.`` prefix)."""
    return scale * (tensors[f"base_model.model.{module}.lora_B.default.weight"].float() @ tensors[f"base_model.model.{module}.lora_A.default.weight"].float())


def _provider_key_pattern() -> re.Pattern[str]:
    """Return the key regex of the vendored ``TurboLoRAProvider``, read from its source without importing mlx."""
    tree = ast.parse((REPO_ROOT / "packages" / "minimax_h3_mlx" / "turbo_lora.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_KEY" for t in n.targets))
    return re.compile(ast.literal_eval(node.value.args[0]))


# ---- Fake minimax_h3_mlx and mlx -------------------------------------------------------------------------------------


def _grid(count: int) -> int:
    """Round *count* up to the 17 * n + 5 grid, as the fork does."""
    return max(0, -(-(count - 5) // 17)) * 17 + 5


def _clip(env: SimpleNamespace, frames: int, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a fake clip: frame *k* is filled with ``k % 256`` and the audio is a ramp of ``k / 100000``."""
    video = np.stack([np.full((height, width, 3), index % 256, dtype=np.uint8) for index in range(frames)])
    samples = round(frames / 24 * env.sample_rate)
    audio = np.tile((np.arange(samples, dtype=np.float32) / 100000)[None], (2, 1))
    return video, audio


class _FakePipeline:
    """Stands in for the fork's low-memory ``MiniMaxH3Pipeline``: records its call and returns a clip."""

    def __init__(self, env: SimpleNamespace) -> None:
        self.env = env

    def __call__(self, prompt: str, **kwargs: Any) -> SimpleNamespace:
        env = self.env
        env.calls.append({"prompt": prompt, **kwargs})
        if env.fail_in_call:
            raise RuntimeError("out of memory")
        video, audio = _clip(env, _grid(round(kwargs["duration_seconds"] * 24)), kwargs["height"], kwargs["width"])
        return SimpleNamespace(video=video, audio=audio, sample_rate=env.sample_rate)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Fake ``minimax_h3_mlx`` (pipeline and converter) and ``mlx.core`` in ``sys.modules``, plus the files a load resolves."""
    state = SimpleNamespace(
        calls=[],
        staged=[],
        loads=[],
        converts=[],
        limits=[],
        released=0,
        sample_rate=32000,
        fail_in_call=False,
        tmp=tmp_path,
    )

    class Pipeline:
        @classmethod
        def from_pretrained(cls, **kwargs: Any) -> _FakePipeline:
            state.loads.append(kwargs)
            return _FakePipeline(state)

    def convert(source: Path, output: Path, *, bits: int, group_size: int, num_layers: int) -> dict:
        state.converts.append((source, output, bits, group_size, num_layers))
        shard = "model-00001-of-00014.safetensors"
        save_file({"model.language_model.layers.0.w": torch.ones(2, 2)}, str(output / shard))
        (output / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 16}, "weight_map": {"model.language_model.layers.0.w": shard}}), encoding="utf-8")
        (output / "quant_config.json").write_text("{}", encoding="utf-8")
        (output / "config.json").write_text("{}", encoding="utf-8")
        return {}

    mx = types.ModuleType("mlx.core")
    mx.set_cache_limit = lambda value: state.limits.append(("cache", value)) or 111
    mx.set_memory_limit = lambda value: state.limits.append(("memory", value)) or 222
    mlx = types.ModuleType("mlx")
    mlx.core = mx
    package = types.ModuleType("minimax_h3_mlx")
    package.__path__ = []
    pipeline_module = types.ModuleType("minimax_h3_mlx.pipeline")
    pipeline_module.MiniMaxH3Pipeline = Pipeline
    quantize_module = types.ModuleType("minimax_h3_mlx.quantize_text_encoder")
    quantize_module.convert = convert
    for name, module in (("mlx", mlx), ("mlx.core", mx), ("minimax_h3_mlx", package), ("minimax_h3_mlx.pipeline", pipeline_module), ("minimax_h3_mlx.quantize_text_encoder", quantize_module)):
        monkeypatch.setitem(sys.modules, name, module)

    def generate_conditioned(**kwargs: Any) -> tuple[np.ndarray, np.ndarray, int]:
        state.staged.append(kwargs)
        if state.fail_in_call:
            raise RuntimeError("out of memory")
        total = kwargs["schedule"].forwards
        for done in range(1, total + 1):
            kwargs["step_callback"](done, total)
        video, audio = _clip(state, _grid(kwargs["frames"]), kwargs["height"], kwargs["width"])
        return video, audio, state.sample_rate

    monkeypatch.setattr(h3_video, "generate_conditioned", generate_conditioned)
    state.files = _make_files(tmp_path)
    monkeypatch.setattr(h3_video, "resolve_mlx_files", lambda *_a, **_k: state.files)
    monkeypatch.setattr(h3_video, "release_memory", lambda: setattr(state, "released", state.released + 1))
    return state


def _make_files(root: Path) -> H3MlxFiles:
    """Build a tiny text-encoder release (language shard 1, vision shard 14), a turbo LoRA and an empty transformer folder."""
    fl2va = root / "release" / "FL2VA"
    encoder = fl2va / "text_encoder"
    encoder.mkdir(parents=True)
    (fl2va / "model_index.json").write_text("{}", encoding="utf-8")
    (encoder / "config.json").write_text("{}", encoding="utf-8")
    save_file({"model.language_model.layers.0.w": torch.ones(2, 2)}, str(encoder / "model-00001-of-00014.safetensors"))
    save_file(
        {"model.visual.blocks.0.w": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3), "model.visual.merger.b": torch.ones(3, dtype=torch.bfloat16)},
        str(encoder / "model-00014-of-00014.safetensors"),
    )
    weight_map = {
        "model.language_model.layers.0.w": "model-00001-of-00014.safetensors",
        "model.visual.blocks.0.w": "model-00014-of-00014.safetensors",
        "model.visual.merger.b": "model-00014-of-00014.safetensors",
    }
    (encoder / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 100}, "weight_map": weight_map}), encoding="utf-8")
    transformer = root / "int8"
    transformer.mkdir()
    return H3MlxFiles(fl2va_dir=fl2va, transformer_dir=transformer, text_encoder_source=encoder, lora_file=_write_turbo(root / "lora" / "turbo.safetensors"))


def _load(env: SimpleNamespace, mode: str = "t2v", **extra: Any) -> Any:
    model, _ = H3MlxVideoBackend().load_model(MODEL_REF, **{"mode": mode, "preset": PRESET, "lora": LORA, "lora_scale": TURBO_SCALE, "models_dir": env.tmp / "models", **extra})
    return model


def _image(path: Path, size: tuple[int, int], color: tuple[int, int, int] = (200, 30, 30)) -> str:
    Image.new("RGB", size, color).save(path)
    return str(path)


@pytest.fixture
def written(monkeypatch):
    """Replace the ffmpeg writers: the encoded frames and the WAV settings are recorded, and the output file is a stub."""
    record = SimpleNamespace(frames=None, wav=None, commands=[], muxed=[])

    def write_frames(frames, command):
        record.frames = np.array(list(frames))
        record.commands.append(command)
        Path(command[-1]).write_bytes(b"video")

    def mux(video, wav, output):
        record.muxed.append((Path(video), Path(wav), Path(output)))
        with wave.open(str(wav)) as reader:
            record.wav = (reader.getnchannels(), reader.getframerate(), reader.getnframes())
        Path(output).write_bytes(b"muxed")

    monkeypatch.setattr(h3_video, "write_segment_frames", write_frames)
    monkeypatch.setattr(h3_video, "mux_audio", mux)
    return record


# ---- Constants, protocol ---------------------------------------------------------------------------------------------


def test_backend_satisfies_the_video_protocol() -> None:
    backend = H3MlxVideoBackend()
    assert backend.name == "h3"
    assert isinstance(backend, VideoBackend)
    assert callable(backend.extend_video)


def test_mlx_the_fork_and_torch_are_imported_lazily() -> None:
    tree = ast.parse(Path(h3_video.__file__).read_text(encoding="utf-8"))
    modules = [node.module or "" for node in tree.body if isinstance(node, ast.ImportFrom)] + [alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names]
    assert not [name for name in modules if name.split(".")[0] in {"mlx", "minimax_h3_mlx", "torch", "safetensors", "diffusers", "scipy", "PIL", "numpy"}]


# ---- resolve_mlx_files -----------------------------------------------------------------------------------------------


class TestResolveFiles:
    def test_a_hub_reference_downloads_the_three_targets_into_their_snapshots(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hf_snapshot, "cached_snapshot_dirs", lambda *_a: [])
        calls = []

        def download(**kwargs):
            calls.append(kwargs)
            return str(tmp_path / kwargs["repo_id"].replace("/", "_"))

        (tmp_path / "lora_repo").mkdir()
        (tmp_path / "lora_repo" / "turbo.safetensors").write_bytes(b"x")

        files = resolve_mlx_files(MODEL_REF, PRESET, lora=LORA, snapshot_download=download)

        assert {call["repo_id"] for call in calls} == {"main/repo", "int8/repo", "lora/repo"}
        assert next(call for call in calls if call["repo_id"] == "main/repo") == {"repo_id": "main/repo", "revision": "rev-main", "allow_patterns": ["FL2VA/model_index.json"]}
        assert files.fl2va_dir == tmp_path / "main_repo" / "FL2VA"
        assert files.text_encoder_source == files.fl2va_dir / "text_encoder"
        assert files.transformer_dir == tmp_path / "int8_repo"
        assert files.lora_file == tmp_path / "lora_repo" / "turbo.safetensors"

    def test_a_cached_snapshot_with_every_file_is_not_downloaded_again(self, tmp_path, monkeypatch):
        snapshots = {"main/repo": tmp_path / "m", "int8/repo": tmp_path / "t", "lora/repo": tmp_path / "l"}
        for snapshot, name in ((snapshots["main/repo"], "FL2VA/model_index.json"), (snapshots["int8/repo"], "config.json"), (snapshots["lora/repo"], "turbo.safetensors")):
            (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
            (snapshot / name).write_bytes(b"x")
        monkeypatch.setattr(hf_snapshot, "cached_snapshot_dirs", lambda repo_id, _revision: [snapshots[repo_id]])

        def no_download(**_kwargs):
            raise AssertionError("downloaded")

        files = resolve_mlx_files(MODEL_REF, PRESET, lora=LORA, snapshot_download=no_download)
        assert files.transformer_dir == snapshots["int8/repo"]

    def test_a_local_folder_supplies_the_release_and_the_others_still_resolve(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hf_snapshot, "cached_snapshot_dirs", lambda *_a: [])
        (tmp_path / "h3" / "FL2VA").mkdir(parents=True)
        (tmp_path / "h3" / "FL2VA" / "model_index.json").write_text("{}", encoding="utf-8")
        (tmp_path / "lora").mkdir()
        (tmp_path / "lora" / "turbo.safetensors").write_bytes(b"x")
        downloads = []

        def download(**kwargs):
            downloads.append(kwargs["repo_id"])
            return str(tmp_path / ("lora" if kwargs["repo_id"] == "lora/repo" else "int8"))

        files = resolve_mlx_files(str(tmp_path / "h3"), PRESET, lora=LORA, snapshot_download=download)
        assert files.fl2va_dir == tmp_path / "h3" / "FL2VA"
        assert "main/repo" not in downloads

    @staticmethod
    def _local_h3(tmp_path: Path) -> Path:
        folder = tmp_path / "h3"
        (folder / "FL2VA").mkdir(parents=True)
        (folder / "FL2VA" / "model_index.json").write_text("{}", encoding="utf-8")
        (tmp_path / "lora").mkdir(exist_ok=True)
        (tmp_path / "lora" / "turbo.safetensors").write_bytes(b"x")
        return folder

    @staticmethod
    def _int8(folder: Path, *, shards: tuple[str, ...] = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text("{}", encoding="utf-8")
        (folder / "quant_config.json").write_text('{"bits": 8, "group_size": 32}', encoding="utf-8")
        (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {f"w{n}": name for n, name in enumerate(shards)}}), encoding="utf-8")
        for name in shards:
            (folder / name).write_bytes(b"x")
        return folder

    def _resolve(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, folder: Path) -> tuple[H3MlxFiles, list[str]]:
        monkeypatch.setattr(hf_snapshot, "cached_snapshot_dirs", lambda *_a: [])
        downloads: list[str] = []

        def download(**kwargs):
            downloads.append(kwargs["repo_id"])
            return str(tmp_path / ("lora" if kwargs["repo_id"] == "lora/repo" else "int8"))

        return resolve_mlx_files(str(folder), PRESET, lora=LORA, snapshot_download=download), downloads

    @pytest.mark.parametrize("where", ["FL2VA/transformer", "transformer"])
    def test_a_local_int8_transformer_is_used_and_not_downloaded(self, tmp_path, monkeypatch, where):
        folder = self._local_h3(tmp_path)
        transformer = self._int8(folder / where)

        files, downloads = self._resolve(tmp_path, monkeypatch, folder)

        assert files.transformer_dir == transformer
        assert downloads == ["lora/repo"]

    @pytest.mark.parametrize("where", ["FL2VA/transformer", "transformer"])
    def test_a_local_transformer_in_another_layout_is_refused_not_replaced_by_the_stock_one(self, tmp_path, monkeypatch, where):
        folder = self._local_h3(tmp_path)
        (folder / where).mkdir(parents=True)
        (folder / where / "config.json").write_text("{}", encoding="utf-8")
        (folder / where / "diffusion_pytorch_model-00001-of-00014.safetensors").write_bytes(b"x")

        with pytest.raises(ValueError):
            self._resolve(tmp_path, monkeypatch, folder)

    def test_a_bf16_transformer_beside_an_int8_one_does_not_hide_the_int8_one(self, tmp_path, monkeypatch):
        folder = self._local_h3(tmp_path)
        (folder / "FL2VA" / "transformer").mkdir()
        (folder / "FL2VA" / "transformer" / "model-00001-of-00013.safetensors").write_bytes(b"x")
        transformer = self._int8(folder / "transformer")

        files, _downloads = self._resolve(tmp_path, monkeypatch, folder)

        assert files.transformer_dir == transformer

    def test_a_local_int8_transformer_missing_a_shard_is_refused(self, tmp_path, monkeypatch):
        folder = self._local_h3(tmp_path)
        transformer = self._int8(folder / "FL2VA" / "transformer")
        (transformer / "model-00002-of-00002.safetensors").unlink()

        with pytest.raises(FileNotFoundError, match="model-00002-of-00002"):
            self._resolve(tmp_path, monkeypatch, folder)

    def test_a_local_folder_without_a_transformer_takes_the_presets_int8_one(self, tmp_path, monkeypatch):
        folder = self._local_h3(tmp_path)
        (folder / "transformer").mkdir()  # an empty folder is no transformer

        files, downloads = self._resolve(tmp_path, monkeypatch, folder)

        assert files.transformer_dir == tmp_path / "int8"
        assert sorted(downloads) == ["int8/repo", "lora/repo"]

    def test_the_download_plan_agrees_with_the_loader_for_a_local_folder(self, tmp_path, monkeypatch):
        from zvisiongenerator.backends.mac.download_plan import h3_mlx_download_plan

        monkeypatch.setattr(hf_snapshot, "cached_snapshot_dirs", lambda *_a: [])
        bare = self._local_h3(tmp_path)
        with_int8 = self._local_h3(tmp_path / "a")
        self._int8(with_int8 / "FL2VA" / "transformer")
        with_bf16 = self._local_h3(tmp_path / "b")
        (with_bf16 / "transformer").mkdir()
        (with_bf16 / "transformer" / "model-00001-of-00014.safetensors").write_bytes(b"x")

        for folder, loader_downloads in ((bare, {"int8/repo", "lora/repo"}), (with_int8, {"lora/repo"})):
            _files, downloads = self._resolve(tmp_path, monkeypatch, folder)
            assert {target.repo_id for target in h3_mlx_download_plan(str(folder), PRESET, LORA)} == set(downloads) == loader_downloads
        # The loader refuses a bf16 transformer, so the plan offers no stock transformer for it.
        assert {target.repo_id for target in h3_mlx_download_plan(str(with_bf16), PRESET, LORA)} == {"lora/repo"}
        with pytest.raises(ValueError, match="INT8 layout"):
            self._resolve(tmp_path, monkeypatch, with_bf16)

    def test_a_local_folder_without_the_release_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="FL2VA/model_index.json"):
            resolve_mlx_files(str(tmp_path), PRESET, lora=LORA, snapshot_download=lambda **_k: str(tmp_path))

    def test_the_lora_must_be_one_file(self):
        with pytest.raises(ValueError, match="one file"):
            resolve_mlx_files(MODEL_REF, PRESET, lora=HubFiles("a/b", "r", ("x", "y")), snapshot_download=lambda **_k: "/nowhere")


# ---- The text-encoder copy -------------------------------------------------------------------------------------------


class TestTextEncoderCopy:
    def test_converts_once_and_adds_the_vision_tensors_and_the_index(self, env):
        models = env.tmp / "models"
        copy = ensure_text_encoder_copy(models, "h3", env.files, PRESET)

        assert copy == models / "h3@mlx-te4"
        assert len(env.converts) == 1
        assert env.converts[0][2:] == (4, 64, 50)
        vision = load_file(str(copy / "model-visual.safetensors"))
        assert set(vision) == {"model.visual.blocks.0.w", "model.visual.merger.b"}
        assert all(tensor.dtype == torch.bfloat16 for tensor in vision.values())
        index = json.loads((copy / "model.safetensors.index.json").read_text(encoding="utf-8"))
        assert index["weight_map"]["model.visual.blocks.0.w"] == "model-visual.safetensors"
        assert index["weight_map"]["model.language_model.layers.0.w"] == "model-00001-of-00014.safetensors"
        assert index["metadata"]["total_size"] == 16 + sum(tensor.nbytes for tensor in vision.values())

    def test_the_manifest_is_json_native_and_names_the_fork(self, env):
        copy = ensure_text_encoder_copy(env.tmp / "models", "h3", env.files, PRESET)
        text = (copy / "ziv-quant.json").read_text(encoding="utf-8")
        manifest = json.loads(text)
        assert manifest["fork"] == H3_MLX_COMMIT
        assert manifest["bits"] == 4
        assert manifest["family"] == "h3"
        assert all(isinstance(entry, dict) and set(entry) == {"name", "bytes", "mtime_ns", "inode"} for entry in manifest["sources"])
        assert json.dumps(manifest, indent=2) + "\n" == text

    def test_a_second_call_writes_nothing(self, env):
        first = ensure_text_encoder_copy(env.tmp / "models", "h3", env.files, PRESET)
        marker = first / "marker"
        marker.write_text("kept", encoding="utf-8")
        assert ensure_text_encoder_copy(env.tmp / "models", "h3", env.files, PRESET) == first
        assert marker.exists()
        assert len(env.converts) == 1

    def test_a_changed_source_rebuilds_the_copy(self, env):
        ensure_text_encoder_copy(env.tmp / "models", "h3", env.files, PRESET)
        shard = env.files.text_encoder_source / "model-00001-of-00014.safetensors"
        save_file({"model.language_model.layers.0.w": torch.ones(4, 4)}, str(shard))
        ensure_text_encoder_copy(env.tmp / "models", "h3", env.files, PRESET)
        assert len(env.converts) == 2

    def test_a_release_without_vision_tensors_leaves_no_copy(self, env):
        index = env.files.text_encoder_source / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"model.language_model.layers.0.w": "model-00001-of-00014.safetensors"}}), encoding="utf-8")
        with pytest.raises(ValueError, match="model.visual"):
            ensure_text_encoder_copy(env.tmp / "models", "h3", env.files, PRESET)
        assert not (env.tmp / "models" / "h3@mlx-te4").exists()

    def test_a_missing_index_is_reported(self, env):
        (env.files.text_encoder_source / "model.safetensors.index.json").unlink()
        with pytest.raises(FileNotFoundError, match="index"):
            ensure_text_encoder_copy(env.tmp / "models", "h3", env.files, PRESET)


# ---- load_model -------------------------------------------------------------------------------------------------------


class TestLoadModel:
    def test_returns_the_h3_model_info(self, env):
        _, info = H3MlxVideoBackend().load_model(MODEL_REF, preset=PRESET, lora=LORA, lora_scale=TURBO_SCALE, models_dir=env.tmp / "models")
        assert (info.family, info.backend, info.supports_i2v, info.default_fps, info.frame_alignment, info.resolution_alignment) == ("h3", "h3", True, 24, 17, 32)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"quantize": 4}, "quantize"),
            ({"upscale": True}, "upscale"),
            ({"mode": "fl2va"}, "mode"),
            ({"preset": None}, "preset"),
            ({"lora_scale": None}, "lora_scale"),
        ],
    )
    def test_refuses_what_it_cannot_do(self, env, kwargs, message):
        options = {"preset": PRESET, "lora": LORA, "lora_scale": TURBO_SCALE, "models_dir": env.tmp / "models", **kwargs}
        with pytest.raises(ValueError, match=message):
            H3MlxVideoBackend().load_model(MODEL_REF, **options)


class TestDirectTurbo:
    def test_the_turbo_file_goes_through_with_alpha_sixteenth_of_its_rank_times_one(self, env, written, tmp_path):
        model = _load(env)
        H3MlxVideoBackend().text_to_video(model, "a fox", 64, 32, 22, 7, 8, str(tmp_path / "out.mp4"))

        kwargs = env.loads[0]
        assert kwargs["turbo_lora_path"] == env.files.lora_file
        assert kwargs["turbo_lora_alpha"] == 0.0625 * TURBO_RANK
        assert kwargs["turbo_lora_scale"] == 1.0
        # What the fork's provider computes: scale * alpha / rank.
        assert kwargs["turbo_lora_scale"] * kwargs["turbo_lora_alpha"] / TURBO_RANK == 0.0625

    def test_the_files_own_alpha_is_not_used(self, env, written, tmp_path):
        _write_turbo(env.files.lora_file, rank=8, metadata={"alpha": "8"})
        H3MlxVideoBackend().text_to_video(_load(env), "a fox", 64, 32, 22, 7, 8, str(tmp_path / "out.mp4"))
        assert env.loads[0]["turbo_lora_alpha"] == 0.0625 * 8

    def test_the_scale_can_be_given(self, env, written, tmp_path):
        H3MlxVideoBackend().text_to_video(_load(env, lora_scale=0.125), "a fox", 64, 32, 22, 7, 8, str(tmp_path / "out.mp4"))
        assert env.loads[0]["turbo_lora_alpha"] == 0.125 * TURBO_RANK


class TestConvertedFolder:
    @staticmethod
    def _converted(tmp_path: Path, storage: str = "mlx-int8") -> str:
        from zvisiongenerator.utils.h3_converted import write_converted_h3

        folder = tmp_path / "myh3"
        folder.mkdir()
        write_converted_h3(folder, {"storage": storage, "turbo": "merged"})
        return str(folder)

    def test_gets_no_turbo_lora_and_shares_stocks_text_encoder_copy(self, env, written, tmp_path, monkeypatch):
        stock = _make_files(tmp_path / "stock")
        monkeypatch.setattr(h3_video, "snapshot_with", lambda files, download: stock.fl2va_dir.parent)
        model, _ = H3MlxVideoBackend().load_model(self._converted(tmp_path), preset=PRESET, lora=LORA, lora_scale=TURBO_SCALE, models_dir=env.tmp / "models", model_name="myh3")
        H3MlxVideoBackend().text_to_video(model, "a fox", 64, 32, 22, 7, 8, str(tmp_path / "out.mp4"))

        assert env.loads[0]["turbo_lora_path"] is None
        assert model.text_encoder_dir.name == "h3@mlx-te4"
        assert model.files.text_encoder_source == stock.text_encoder_source  # stock's files, so stock's copy stays current

    def test_the_staged_image_path_passes_no_adapter(self, env, written, tmp_path, monkeypatch):
        stock = _make_files(tmp_path / "stock")
        monkeypatch.setattr(h3_video, "snapshot_with", lambda files, download: stock.fl2va_dir.parent)
        model, _ = H3MlxVideoBackend().load_model(self._converted(tmp_path), mode="i2v", preset=PRESET, lora=LORA, lora_scale=TURBO_SCALE, models_dir=env.tmp / "models")
        H3MlxVideoBackend().image_to_video(model, _image(tmp_path / "a.png", (64, 32)), "a fox", 64, 32, 22, 7, 8, str(tmp_path / "out.mp4"))

        call = env.staged[0]
        assert (call["turbo_lora_path"], call["turbo_lora_alpha"], call["turbo_lora_scale"]) == (None, None, 1.0)
        assert call["text_encoder_dir"].name == "h3@mlx-te4"

    def test_a_folder_converted_on_windows_or_linux_is_refused(self, env, tmp_path):
        with pytest.raises(RuntimeError):
            H3MlxVideoBackend().load_model(self._converted(tmp_path, "nf4-folded"), preset=PRESET, lora=LORA, lora_scale=TURBO_SCALE, models_dir=env.tmp / "models")


class TestTextToVideo:
    def test_verbose_defaults_to_the_cli_setting_and_can_be_turned_off(self, env, written, tmp_path):
        H3MlxVideoBackend().text_to_video(_load(env), "a fox", 64, 32, 22, 7, 8, str(tmp_path / "out.mp4"))
        H3MlxVideoBackend().text_to_video(_load(env, verbose=False), "a fox", 64, 32, 22, 7, 8, str(tmp_path / "out.mp4"))
        assert [kwargs["verbose"] for kwargs in env.loads] == [True, False]

    def test_calls_the_pipeline_with_the_clip_geometry(self, env, written, tmp_path):
        H3MlxVideoBackend().text_to_video(_load(env), "a fox", 64, 32, 124, 7, 8, str(tmp_path / "out.mp4"))

        call = env.calls[0]
        assert call["prompt"] == "a fox"
        assert call["duration_seconds"] == pytest.approx(124 / 24)
        assert call["num_inference_steps"] == 9
        assert (call["seed"], call["height"], call["width"]) == (7, 32, 64)

    def test_writes_the_frames_and_muxes_the_audio_at_the_models_rate(self, env, written, tmp_path):
        output = tmp_path / "videos" / "out.mp4"
        result = H3MlxVideoBackend().text_to_video(_load(env), "a fox", 64, 32, 22, 7, 8, str(output))

        assert result == output
        assert output.read_bytes() == b"muxed"
        assert written.frames.shape == (22, 32, 64, 3)
        assert "24/1" in written.commands[0] and "64x32" in written.commands[0]
        assert written.wav == (2, 32000, round(22 / 24 * 32000))
        assert [p.name for p in output.parent.iterdir()] == ["out.mp4"]

    def test_progress_fires_at_the_start_and_the_end(self, env, written, tmp_path):
        events = []
        H3MlxVideoBackend().text_to_video(_load(env), "a fox", 64, 32, 22, 7, 8, str(tmp_path / "out.mp4"), step_callback=events.append)
        assert events == [{"phase": "video_generate", "current_step": 0, "total_steps": 1}, {"phase": "video_generate", "current_step": 1, "total_steps": 1}]

    def test_the_cache_and_memory_limits_are_set_for_the_run_and_restored(self, env, written, tmp_path):
        H3MlxVideoBackend().text_to_video(_load(env), "a fox", 64, 32, 22, 7, 8, str(tmp_path / "out.mp4"))
        assert [limit for limit in env.limits if limit[0] in ("cache", "memory")] == [("cache", 0), ("memory", 16_000_000_000), ("cache", 111), ("memory", 222)]

    @pytest.mark.parametrize("option", ["stage1_steps", "review"])
    def test_upscale_is_refused(self, env, written, tmp_path, option):
        with pytest.raises(ValueError, match="upscale"):
            H3MlxVideoBackend().text_to_video(_load(env), "a fox", 64, 32, 22, 7, 8, str(tmp_path / "out.mp4"), **{option: 4 if option == "stage1_steps" else print})
        assert env.loads == []

    def test_releases_memory_in_finally_and_still_raises(self, env, written, tmp_path):
        env.fail_in_call = True
        events = []
        with pytest.raises(RuntimeError, match="out of memory"):
            H3MlxVideoBackend().text_to_video(_load(env), "a fox", 64, 32, 22, 7, 8, str(tmp_path / "out.mp4"), step_callback=events.append)
        assert env.released == 1
        assert [event["current_step"] for event in events] == [0]
        assert env.limits[-2:] == [("cache", 111), ("memory", 222)]
        assert not (tmp_path / "out.mp4").exists()


class TestImageToVideo:
    def test_runs_the_staged_path_with_the_models_files_and_the_clip_settings(self, env, written, tmp_path):
        model = _load(env, mode="i2v")
        H3MlxVideoBackend().image_to_video(model, _image(tmp_path / "a.png", (100, 50)), "a fox", 64, 32, 22, 7, 8, str(tmp_path / "out.mp4"))

        assert env.loads == []
        call = env.staged[0]
        assert call["fl2va_dir"] == env.files.fl2va_dir
        assert call["transformer_dir"] == env.files.transformer_dir
        assert call["text_encoder_dir"] == env.tmp / "models" / "h3@mlx-te4"
        assert (call["turbo_lora_path"], call["turbo_lora_alpha"], call["turbo_lora_scale"]) == (env.files.lora_file, TURBO_SCALE * TURBO_RANK, 1.0)
        assert call["schedule"] == PRESET.schedules[8]
        assert (call["prompt"], call["width"], call["height"], call["frames"], call["seed"], call["verbose"]) == ("a fox", 64, 32, 22, 7, True)

    def test_passes_one_fitted_first_frame_with_its_anchor(self, env, written, tmp_path):
        model = _load(env, mode="i2v")
        H3MlxVideoBackend().image_to_video(model, _image(tmp_path / "a.png", (100, 50)), "a fox", 64, 32, 22, 7, 8, str(tmp_path / "out.mp4"))

        call = env.staged[0]
        assert [image.size for image in call["images"]] == [(64, 32)]
        assert list(call["anchors"]) == ["first"]

    def test_the_image_is_centre_cropped_not_stretched(self, env, written, tmp_path):
        wide = Image.new("RGB", (128, 32), (0, 0, 255))
        wide.paste((255, 0, 0), (48, 0, 80, 32))  # a red centre column, 32 px wide
        wide.save(tmp_path / "wide.png")
        H3MlxVideoBackend().image_to_video(_load(env, mode="i2v"), str(tmp_path / "wide.png"), "a fox", 32, 32, 22, 7, 8, str(tmp_path / "out.mp4"))

        fitted = np.asarray(env.staged[0]["images"][0])
        assert fitted.shape == (32, 32, 3)
        assert (fitted[:, 4:28, 0] > 200).all() and (fitted[:, 4:28, 2] < 60).all()

    def test_an_end_image_adds_a_last_anchor(self, env, written, tmp_path):
        model = _load(env, mode="i2v")
        H3MlxVideoBackend().image_to_video(
            model, _image(tmp_path / "a.png", (64, 32)), "a fox", 64, 32, 22, 7, 8, str(tmp_path / "out.mp4"), end_image=_image(tmp_path / "b.png", (32, 64), (0, 0, 255))
        )

        call = env.staged[0]
        assert list(call["anchors"]) == ["first", "last"]
        assert [image.size for image in call["images"]] == [(64, 32), (64, 32)]

    def test_progress_runs_from_zero_to_the_forward_count_in_every_image_mode_step(self, env, written, tmp_path):
        events = []
        H3MlxVideoBackend().image_to_video(_load(env, mode="i2v"), _image(tmp_path / "a.png", (64, 32)), "a fox", 64, 32, 22, 7, 4, str(tmp_path / "out.mp4"), step_callback=events.append)

        assert events == [{"phase": "video_generate", "current_step": step, "total_steps": 4} for step in range(5)]

    def test_a_model_loaded_for_text_to_video_refuses_images(self, env, written, tmp_path):
        with pytest.raises(ValueError, match="text-to-video"):
            H3MlxVideoBackend().image_to_video(_load(env, mode="t2v"), _image(tmp_path / "a.png", (64, 32)), "a fox", 64, 32, 22, 7, 8, str(tmp_path / "out.mp4"))
        assert env.loads == [] and env.staged == []

    def test_upscale_is_refused(self, env, written, tmp_path):
        with pytest.raises(ValueError, match="upscale"):
            H3MlxVideoBackend().image_to_video(_load(env, mode="i2v"), _image(tmp_path / "a.png", (64, 32)), "a fox", 64, 32, 22, 7, 8, str(tmp_path / "out.mp4"), stage1_steps=4)

    def test_a_failure_restores_the_limits_releases_memory_and_still_raises(self, env, written, tmp_path):
        env.fail_in_call = True
        with pytest.raises(RuntimeError, match="out of memory"):
            H3MlxVideoBackend().image_to_video(_load(env, mode="i2v"), _image(tmp_path / "a.png", (64, 32)), "a fox", 64, 32, 22, 7, 8, str(tmp_path / "out.mp4"))
        assert env.limits[-2:] == [("cache", 111), ("memory", 222)]
        assert env.released == 1


# ---- Extend ----------------------------------------------------------------------------------------------------------


def _extend(env, tmp_path, written, monkeypatch, *, extend_frames=51, audio_output=True, source=None, mode="extend", callback=None):
    source = source or make_extend_source(path=tmp_path / "clip.mp4", width=64, height=32, frame_count=49)
    tail = np.zeros((25, 32, 64, 3), dtype=np.uint8)
    tail[-1] = (10, 120, 230)
    monkeypatch.setattr(h3_video, "read_source_tail", lambda _source: tail)
    segment_dir = tmp_path / "segment"
    segment_dir.mkdir(exist_ok=True)
    segment = H3MlxVideoBackend().extend_video(
        _load(env, mode=mode), source, "the fox stops", extend_frames=extend_frames, seed=3, steps=8, segment_dir=segment_dir, audio_output=audio_output, step_callback=callback
    )
    return segment, segment_dir, tail


class TestExtend:
    def test_returns_exactly_the_extend_frames_after_the_anchor_frame(self, env, written, tmp_path, monkeypatch):
        segment, segment_dir, _ = _extend(env, tmp_path, written, monkeypatch, extend_frames=51)

        # 52 frames are asked for: the next clip length on the 17n + 5 grid is 56; frame 0 repeats the anchor and is dropped.
        assert env.staged[0]["frames"] == 56
        assert written.frames.shape == (51, 32, 64, 3)
        assert [int(frame[0, 0, 0]) for frame in written.frames] == list(range(1, 52))
        assert segment == ExtendSegment(video_path=segment_dir / "segment.mp4", audio_path=segment_dir / "segment.wav", lead_samples=0)

    @pytest.mark.parametrize(("extend", "window"), [(17, 22), (34, 39), (51, 56), (68, 73)])
    def test_the_window_is_the_next_grid_length_above_extend_plus_one(self, env, written, tmp_path, monkeypatch, extend, window):
        _extend(env, tmp_path, written, monkeypatch, extend_frames=extend)
        assert env.staged[0]["frames"] == window
        assert written.frames.shape[0] == extend

    def test_pins_the_last_source_frame_as_the_first_anchor_on_the_staged_path(self, env, written, tmp_path, monkeypatch):
        _, _, tail = _extend(env, tmp_path, written, monkeypatch)

        call = env.staged[0]
        assert list(call["anchors"]) == ["first"]
        assert (call["height"], call["width"], call["seed"]) == (32, 64, 3)
        assert np.array_equal(np.asarray(call["images"][0]), tail[-1])
        assert env.loads == []

    def test_the_audio_is_cut_to_the_new_frames_and_written_at_the_joins_rate(self, env, written, tmp_path, monkeypatch):
        _, segment_dir, _ = _extend(env, tmp_path, written, monkeypatch, extend_frames=51)

        with wave.open(str(segment_dir / "segment.wav")) as reader:
            assert (reader.getnchannels(), reader.getframerate(), reader.getnframes()) == (2, 48000, 51 * 2000)
            samples = np.frombuffer(reader.readframes(reader.getnframes()), dtype=np.int16).reshape(-1, 2).astype(np.float32) / 32767
        # The model's audio is a ramp of k / 100000 at 32 kHz; the wav starts at frame 1 (sample 1333) and runs at 48 kHz.
        position = 1000
        assert samples[position, 0] == pytest.approx((round(32000 / 24) + position * 32000 / 48000) / 100000, abs=2e-4)

    def test_no_wav_without_audio_output_or_for_a_silent_source(self, env, written, tmp_path, monkeypatch):
        segment, segment_dir, _ = _extend(env, tmp_path, written, monkeypatch, audio_output=False)
        assert segment.audio_path is None and not (segment_dir / "segment.wav").exists()
        silent = make_extend_source(path=tmp_path / "silent.mp4", width=64, height=32, has_audio=False)
        segment, _, _ = _extend(env, tmp_path, written, monkeypatch, source=silent)
        assert segment.audio_path is None

    def test_the_segment_video_takes_the_sources_size_rate_and_colour_tags(self, env, written, tmp_path, monkeypatch):
        source = make_extend_source(path=tmp_path / "clip.mp4", width=64, height=32, fps_text="24/1", color_space="bt709", color_primaries="bt709", color_trc="bt709")
        _extend(env, tmp_path, written, monkeypatch, source=source)
        command = written.commands[0]
        assert "64x32" in command and "24/1" in command
        assert any("setparams=colorspace=bt709" in part for part in command)

    def test_progress_fires_at_the_start_and_after_every_step(self, env, written, tmp_path, monkeypatch):
        events = []
        _extend(env, tmp_path, written, monkeypatch, callback=events.append)
        assert [(event["current_step"], event["total_steps"]) for event in events] == [(step, 8) for step in range(9)]

    @pytest.mark.parametrize("extend", [0, 18])
    def test_extend_frames_must_be_a_positive_multiple_of_17(self, env, written, tmp_path, monkeypatch, extend):
        with pytest.raises(ValueError, match="multiple of 17"):
            _extend(env, tmp_path, written, monkeypatch, extend_frames=extend)

    def test_the_source_must_be_24_fps_and_32_aligned(self, env, written, tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="24 fps"):
            _extend(env, tmp_path, written, monkeypatch, source=make_extend_source(path=tmp_path / "a.mp4", width=64, height=32, fps=30.0, fps_text="30/1"))
        with pytest.raises(ValueError, match="multiple of 32"):
            _extend(env, tmp_path, written, monkeypatch, source=make_extend_source(path=tmp_path / "b.mp4", width=100, height=32))

    def test_a_model_loaded_for_text_to_video_cannot_extend(self, env, written, tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="text-to-video"):
            _extend(env, tmp_path, written, monkeypatch, mode="t2v")

    def test_a_short_pipeline_result_is_an_error(self, env, written, tmp_path, monkeypatch):
        monkeypatch.setattr(h3_video, "align_frames", lambda _count, _rule: 22)
        with pytest.raises(RuntimeError, match="expected at least 52"):
            _extend(env, tmp_path, written, monkeypatch, extend_frames=51)


def _make_real_source(path: Path, frames: int = 33) -> Any:
    """Encode a 64x32, 24 fps clip with a sine track and return its extend source."""
    clip = numbered_frames(frames, 64, 32)
    ffmpeg(
        *("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=1.4"),
        *("-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "64x32", "-r", "24", "-i", "-"),
        *("-map", "1:v", "-map", "0:a", "-c:a", "aac", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"),
        str(path),
        data=clip.tobytes(),
    )
    return build_extend_source(path, probe_video(path), None)


@REAL
class TestExtendJoin:
    def test_the_segment_passes_the_join_check_and_adds_exactly_the_extend_frames(self, env, tmp_path):
        source = _make_real_source(tmp_path / "source.mp4")
        segment_dir = tmp_path / "segment"
        segment_dir.mkdir()

        segment = H3MlxVideoBackend().extend_video(_load(env, mode="extend"), source, "the fox stops", extend_frames=17, seed=3, steps=8, segment_dir=segment_dir, audio_output=True)

        assert probe_video(segment.video_path).frame_count == 17
        output = splice_clips(source, segment, tmp_path / "joined.mp4", extend_frames=17, include_audio=True)
        probe = probe_video(output)
        assert (probe.frame_count, probe.width, probe.height, probe.has_audio) == (33 + 17, 64, 32, True)

    def test_a_text_to_video_clip_is_a_real_mp4_with_both_streams(self, env, tmp_path):
        output = tmp_path / "out.mp4"
        H3MlxVideoBackend().text_to_video(_load(env), "a fox", 64, 32, 22, 7, 8, str(output))
        probe = probe_video(output)
        assert (probe.frame_count, probe.width, probe.height, probe.has_audio) == (22, 64, 32, True)
        assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".h3-")] == []


# ---- LoRAs -----------------------------------------------------------------------------------------------------------

# Original-checkpoint names of the user files; diffusers converts them to the block and refiner projections.
USER_MODULES = {
    "blocks.0.attn.out_proj": (HIDDEN, HIDDEN),  # -> transformer_blocks.0.attn.to_out.0
    "blocks.0.mlp.fc2": (HIDDEN, FF),  # -> transformer_blocks.0.ff.net.2
    "token_refiner.blocks.0.attn.out_proj": (HIDDEN, HIDDEN),  # -> token_refiner.refiner_blocks.0.attn.to_out.0
}
USER_SLOTS = ["transformer_blocks.0.attn.to_out.0", "transformer_blocks.0.ff.net.2", "token_refiner.refiner_blocks.0.attn.to_out.0"]


class TestCombineLoras:
    def test_a_kohya_file_covering_some_blocks_is_rank_concatenated_with_the_turbo(self, tmp_path):
        from zvisiongenerator.utils.h3_lora import load_lora, pairs_from_state

        turbo = _write_turbo(tmp_path / "turbo.safetensors")
        user = _write_kohya(tmp_path / "user.safetensors", USER_MODULES, rank=2, alpha=1.0)
        converted = pairs_from_state(load_lora(user).state)
        turbo_raw = load_file(str(turbo))

        combined = combine_loras(turbo, 0.0625, [(user, 0.5)], tmp_path / "out" / "adapter.safetensors")

        tensors = load_file(str(combined))
        # The turbo's slots, none added or lost.
        assert {key.removeprefix("base_model.model.") for key in turbo_raw} == set(tensors)
        total = TURBO_RANK + 2
        for slot in sorted({key.rsplit(".lora_", 1)[0] for key in tensors}):
            rows, columns = TARGET_SHAPES[next(target for target in TARGET_SHAPES if slot.endswith(target))]
            assert tensors[f"{slot}.lora_A.default.weight"].shape == (total, columns)
            assert tensors[f"{slot}.lora_B.default.weight"].shape == (rows, total)
            expected = _raw_delta(turbo_raw, slot, 0.0625)
            if slot in USER_SLOTS:
                down, up = converted[slot]
                expected = expected + 0.5 * (up.float() @ down.float())
                torch.testing.assert_close(_delta(tensors, slot), expected, rtol=0, atol=0.02 * float(expected.abs().max()))
            else:
                # Not covered by the user file: the turbo alone, with the user's columns zero-padded.
                torch.testing.assert_close(_delta(tensors, slot), expected, rtol=1e-5, atol=1e-6)
                assert not tensors[f"{slot}.lora_A.default.weight"][TURBO_RANK:].any()
                assert not tensors[f"{slot}.lora_B.default.weight"][:, TURBO_RANK:].any()

    def test_the_metadata_alpha_is_the_total_rank_so_the_provider_multiplies_by_one(self, tmp_path):
        turbo = _write_turbo(tmp_path / "turbo.safetensors")
        user = _write_peft(tmp_path / "user.safetensors", {"blocks.0.attn.out_proj": (HIDDEN, HIDDEN)}, rank=2, metadata={"alpha": "2"})
        combined = combine_loras(turbo, 0.0625, [(user, 1.0)], tmp_path / "adapter.safetensors")

        with safe_open(str(combined), framework="pt") as reader:
            alpha = float(reader.metadata()["alpha"])
            rank = reader.get_slice("transformer_blocks.0.attn.to_q.lora_A.default.weight").get_shape()[0]
        assert alpha == rank == TURBO_RANK + 2
        assert 1.0 * alpha / rank == 1.0  # the provider's multiplier with turbo_lora_scale=1.0

    def test_the_turbo_factor_is_exactly_the_scale_and_ignores_the_files_alpha(self, tmp_path):
        for alpha in ("8", "128"):
            turbo = _write_turbo(tmp_path / f"turbo-{alpha}.safetensors", metadata={"alpha": alpha})
            combined = combine_loras(turbo, 0.0625, [], tmp_path / f"adapter-{alpha}.safetensors")
            tensors, raw = load_file(str(combined)), load_file(str(turbo))
            slot = "transformer_blocks.0.attn.to_q"
            assert tensors[f"{slot}.lora_B.default.weight"].float().equal(raw[f"base_model.model.{slot}.lora_B.default.weight"].float() * 0.0625)
            assert tensors[f"{slot}.lora_A.default.weight"].shape[0] == TURBO_RANK

    def test_a_user_files_scale_is_weight_times_alpha_over_rank_and_the_deltas_add_exactly(self, tmp_path):
        turbo = _write_turbo(tmp_path / "turbo.safetensors")
        first = _write_peft(tmp_path / "first.safetensors", {"blocks.0.attn.out_proj": (HIDDEN, HIDDEN)}, rank=4, metadata={"alpha": "2"})  # alpha / rank = 0.5
        second = _write_peft(
            tmp_path / "second.safetensors", {"blocks.0.attn.out_proj": (HIDDEN, HIDDEN), "blocks.0.mlp.fc2": (HIDDEN, FF)}, rank=2, metadata={"alpha": "2"}, seed=9
        )  # alpha / rank = 1
        slot = "transformer_blocks.0.attn.to_out.0"

        combined = load_file(str(combine_loras(turbo, 0.0625, [(first, 0.5), (second, 0.25)], tmp_path / "adapter.safetensors")))

        assert combined[f"{slot}.lora_A.default.weight"].shape[0] == TURBO_RANK + 4 + 2
        raw_first, raw_second = load_file(str(first)), load_file(str(second))
        turbo_part = _raw_delta(load_file(str(turbo)), slot, 0.0625)
        first_part = 0.5 * 0.5 * (raw_first["diffusion_model.blocks.0.attn.out_proj.lora_B.weight"] @ raw_first["diffusion_model.blocks.0.attn.out_proj.lora_A.weight"])
        second_part = 0.25 * 1.0 * (raw_second["diffusion_model.blocks.0.attn.out_proj.lora_B.weight"] @ raw_second["diffusion_model.blocks.0.attn.out_proj.lora_A.weight"])
        torch.testing.assert_close(_delta(combined, slot), turbo_part + first_part + second_part, rtol=1e-5, atol=1e-5)
        # The second file alone covers the feed-forward: the first's columns there stay zero.
        ff_slot = "transformer_blocks.0.ff.net.2"
        assert not combined[f"{ff_slot}.lora_B.default.weight"][:, TURBO_RANK : TURBO_RANK + 4].any()

    def test_every_key_is_one_the_vendored_provider_accepts(self, tmp_path):
        turbo = _write_turbo(tmp_path / "turbo.safetensors")
        user = _write_kohya(tmp_path / "user.safetensors", USER_MODULES, rank=2, alpha=1.0)
        pattern = _provider_key_pattern()
        for key in load_file(str(combine_loras(turbo, 0.0625, [(user, 1.0)], tmp_path / "adapter.safetensors"))):
            assert pattern.fullmatch(key), key

    @pytest.mark.parametrize("module", ["blocks.1.adaln_proj.linear", "final_layer.adaln_proj.linear", "time_embedder.proj_in"])
    def test_other_targets_are_refused(self, tmp_path, module):
        turbo = _write_turbo(tmp_path / "turbo.safetensors")
        shapes = {"blocks.0.attn.out_proj": (HIDDEN, HIDDEN), module: (6 * HIDDEN, HIDDEN) if "adaln" in module else (HIDDEN, 4)}
        user = _write_kohya(tmp_path / "user.safetensors", shapes, rank=2, alpha=1.0)
        with pytest.raises(ValueError):
            combine_loras(turbo, 0.0625, [(user, 1.0)], tmp_path / "adapter.safetensors")
        assert not (tmp_path / "adapter.safetensors").exists()

    def test_a_block_the_turbo_does_not_have_is_refused(self, tmp_path):
        turbo = _write_turbo(tmp_path / "turbo.safetensors", blocks=2)
        user = _write_peft(tmp_path / "user.safetensors", {"blocks.7.attn.out_proj": (HIDDEN, HIDDEN)}, rank=2, metadata={"alpha": "2"})
        with pytest.raises(ValueError, match="transformer_blocks.7.attn.to_out.0.*does not have"):
            combine_loras(turbo, 0.0625, [(user, 1.0)], tmp_path / "adapter.safetensors")

    def test_factors_of_the_wrong_size_are_refused(self, tmp_path):
        turbo = _write_turbo(tmp_path / "turbo.safetensors")
        user = _write_peft(tmp_path / "user.safetensors", {"blocks.0.attn.out_proj": (HIDDEN + 8, HIDDEN)}, rank=2, metadata={"alpha": "2"})
        with pytest.raises(ValueError, match="do not fit"):
            combine_loras(turbo, 0.0625, [(user, 1.0)], tmp_path / "adapter.safetensors")


def _tiny_slot_shapes() -> dict:
    """Return the slots of a two-block, one-refiner model with the test's projection shapes, as ``(inputs, outputs)``."""
    counts = {"transformer_blocks": 2, "token_refiner.refiner_blocks": 1}
    return {(family, index, target): (columns, rows) for family, count in counts.items() for index in range(count) for target, (rows, columns) in TARGET_SHAPES.items()}


class TestCombineWithoutTurbo:
    def test_covers_every_projection_at_the_users_rank(self, tmp_path, monkeypatch):
        from zvisiongenerator.utils.h3_lora import load_lora, pairs_from_state

        monkeypatch.setattr(h3_video, "_slot_shapes", _tiny_slot_shapes)
        user = _write_kohya(tmp_path / "user.safetensors", USER_MODULES, rank=2, alpha=1.0)
        converted = pairs_from_state(load_lora(user).state)

        tensors = load_file(str(combine_loras(None, 0.0625, [(user, 0.5)], tmp_path / "adapter.safetensors")))

        assert {key.rsplit(".lora_", 1)[0] for key in tensors} == {f"{family}.{index}.{target}" for family, index, target in _tiny_slot_shapes()}
        for slot in {key.rsplit(".lora_", 1)[0] for key in tensors}:
            assert tensors[f"{slot}.lora_A.default.weight"].shape[0] == 2
            if slot in USER_SLOTS:
                down, up = converted[slot]
                expected = 0.5 * (up.float() @ down.float())
                torch.testing.assert_close(_delta(tensors, slot), expected, rtol=0, atol=0.02 * float(expected.abs().max()))
            else:
                assert not tensors[f"{slot}.lora_A.default.weight"].any()

    def test_a_converted_folder_with_user_loras_gets_an_adapter_without_the_turbo(self, env, written, tmp_path, monkeypatch):
        monkeypatch.setattr(h3_video, "_slot_shapes", _tiny_slot_shapes)
        stock = _make_files(tmp_path / "stock")
        monkeypatch.setattr(h3_video, "snapshot_with", lambda files, download: stock.fl2va_dir.parent)
        user = _write_kohya(tmp_path / "user.safetensors", USER_MODULES, rank=2, alpha=1.0)
        folder = TestConvertedFolder._converted(tmp_path)

        model, _ = H3MlxVideoBackend().load_model(folder, preset=PRESET, lora=LORA, lora_scale=TURBO_SCALE, models_dir=env.tmp / "models", model_name="myh3", loras=[(str(user), 1.0)])
        H3MlxVideoBackend().text_to_video(model, "a fox", 64, 32, 22, 7, 8, str(tmp_path / "out.mp4"))

        adapter = load_file(str(env.loads[0]["turbo_lora_path"]))
        assert all(tensor.shape[0] == 2 for key, tensor in adapter.items() if key.endswith("lora_A.default.weight"))


class TestLoadWithUserLoras:
    def test_the_combined_adapter_is_passed_with_unit_scale_and_no_alpha_override(self, env, written, tmp_path):
        user = _write_kohya(tmp_path / "user.safetensors", USER_MODULES, rank=2, alpha=1.0)
        model = _load(env, loras=[(str(user), 0.5)])
        H3MlxVideoBackend().text_to_video(model, "a fox", 64, 32, 22, 7, 8, str(tmp_path / "out.mp4"))

        kwargs = env.loads[0]
        assert kwargs["turbo_lora_path"] != env.files.lora_file
        assert kwargs["turbo_lora_path"].is_file()
        assert kwargs["turbo_lora_alpha"] is None
        assert kwargs["turbo_lora_scale"] == 1.0

    def test_the_adapter_lives_as_long_as_the_model(self, env, tmp_path):
        user = _write_kohya(tmp_path / "user.safetensors", USER_MODULES, rank=2, alpha=1.0)
        model = _load(env, loras=[(str(user), 1.0)])
        adapter = model.turbo_lora_path
        assert adapter.is_file()
        del model
        gc.collect()
        assert not adapter.exists()

    def test_a_refused_lora_fails_before_the_encoder_is_converted_and_leaves_no_temporary_folder(self, env, tmp_path, monkeypatch):
        made = []
        real = h3_video.tempfile.mkdtemp
        monkeypatch.setattr(h3_video.tempfile, "mkdtemp", lambda **kwargs: made.append(real(**kwargs)) or made[-1])
        user = _write_kohya(tmp_path / "user.safetensors", {"blocks.1.adaln_proj.linear": (6 * HIDDEN, HIDDEN)}, rank=2, alpha=1.0)
        with pytest.raises(ValueError, match="transformer blocks and the token refiner only"):
            _load(env, loras=[(str(user), 1.0)])
        assert env.converts == []
        assert len(made) == 1 and not Path(made[0]).exists()


# ---- Settings from the preset ----------------------------------------------------------------------------------------

# What the pipeline got before the settings moved into the config: the fork's fps, shifts and LoRA strength, 17n + 5 frames.
SHIPPED_LOAD_ARGUMENTS = {"sigma_shift_video": 6.0, "sigma_shift_audio": 3.0, "turbo_lora_scale": 1.0, "stream_blocks": True, "memory_limit_gb": 16.0}


def _shipped_schedules() -> dict[int, H3Schedule]:
    from zvisiongenerator.utils.config import load_config
    from zvisiongenerator.utils.h3_preset import h3_mlx_load_options

    return h3_mlx_load_options(load_config(), "h3")["preset"].schedules


SHIPPED_SCHEDULES = _shipped_schedules()


def _shipped_load(env: SimpleNamespace, mode: str) -> Any:
    """Load the model from the options the packaged config produces on macOS, as a real run does."""
    from zvisiongenerator.utils.config import load_config
    from zvisiongenerator.utils.h3_preset import h3_mlx_load_options

    options = h3_mlx_load_options(load_config(), "h3")
    model, _ = H3MlxVideoBackend().load_model(MODEL_REF, mode=mode, models_dir=env.tmp / "models", **options)
    return model


class TestSettingsFromThePreset:
    def test_text_to_video_gets_the_shipped_arguments(self, env, written, tmp_path):
        H3MlxVideoBackend().text_to_video(_shipped_load(env, "t2v"), "a fox", 64, 32, 124, 7, 8, str(tmp_path / "out.mp4"))

        loaded = env.loads[0]
        assert loaded["checkpoint_dir"] == env.files.fl2va_dir
        assert loaded["transformer_dir"] == env.files.transformer_dir
        assert loaded["text_encoder_dir"] == env.tmp / "models" / "h3@mlx-te4"
        assert {key: loaded[key] for key in SHIPPED_LOAD_ARGUMENTS} == SHIPPED_LOAD_ARGUMENTS
        assert loaded["turbo_lora_alpha"] == 0.0625 * TURBO_RANK
        assert (loaded["low_memory"], loaded["load_vision"]) == (True, False)
        assert env.calls[0] == {
            "prompt": "a fox",
            "duration_seconds": 124 / 24,
            "num_inference_steps": 9,
            "seed": 7,
            "height": 32,
            "width": 64,
            "verbose": True,
        }

    def test_image_to_video_gets_the_shipped_arguments(self, env, written, tmp_path):
        H3MlxVideoBackend().image_to_video(_shipped_load(env, "i2v"), _image(tmp_path / "a.png", (64, 32)), "a fox", 64, 32, 124, 7, 4, str(tmp_path / "out.mp4"))

        call = env.staged[0]
        assert (call["frames"], call["schedule"], call["turbo_lora_scale"], list(call["anchors"])) == (124, SHIPPED_SCHEDULES[4], 1.0, ["first"])
        assert [image.size for image in call["images"]] == [(64, 32)]

    def test_extend_gets_the_shipped_arguments(self, env, written, tmp_path, monkeypatch):
        source = make_extend_source(path=tmp_path / "clip.mp4", width=64, height=32, frame_count=49)
        monkeypatch.setattr(h3_video, "read_source_tail", lambda _source: np.zeros((25, 32, 64, 3), dtype=np.uint8))
        segment_dir = tmp_path / "segment"
        segment_dir.mkdir()

        H3MlxVideoBackend().extend_video(_shipped_load(env, "extend"), source, "the fox stops", extend_frames=51, seed=3, steps=8, segment_dir=segment_dir, audio_output=False)

        call = env.staged[0]
        assert (call["frames"], call["schedule"], call["turbo_lora_scale"], list(call["anchors"])) == (56, SHIPPED_SCHEDULES[8], 1.0, ["first"])
        assert (call["seed"], call["height"], call["width"]) == (3, 32, 64)

    def test_a_changed_shift_reaches_the_pipeline(self, env, written, tmp_path):
        schedules = {4: H3Schedule(4, 6.0, 3.0), 8: H3Schedule(8, 5.0, 2.5)}
        model = _load(env, preset=dataclasses.replace(PRESET, schedules=schedules))

        H3MlxVideoBackend().text_to_video(model, "a fox", 64, 32, 22, 7, 8, str(tmp_path / "out.mp4"))
        H3MlxVideoBackend().text_to_video(model, "a fox", 64, 32, 22, 7, 4, str(tmp_path / "out.mp4"))

        assert [(kwargs["sigma_shift_video"], kwargs["sigma_shift_audio"]) for kwargs in env.loads] == [(5.0, 2.5), (6.0, 3.0)]

    def test_a_changed_forward_count_sets_the_sigma_grid(self, env, written, tmp_path):
        model = _load(env, preset=dataclasses.replace(PRESET, schedules={6: H3Schedule(10, 6.0, 3.0)}))

        H3MlxVideoBackend().text_to_video(model, "a fox", 64, 32, 22, 7, 6, str(tmp_path / "out.mp4"))

        assert env.calls[0]["num_inference_steps"] == 11

    def test_a_changed_frame_rule_changes_the_extension_window(self, env, written, tmp_path, monkeypatch):
        source = make_extend_source(path=tmp_path / "clip.mp4", width=64, height=32, frame_count=49)
        monkeypatch.setattr(h3_video, "read_source_tail", lambda _source: np.zeros((25, 32, 64, 3), dtype=np.uint8))
        segment_dir = tmp_path / "segment"
        segment_dir.mkdir()

        # 18 frames are asked for: the shipped rule (17k + 5, up) gives 22, 8k + 1 gives 25.
        model = _load(env, mode="extend", preset=dataclasses.replace(PRESET, frame_rule=FrameRule(8, 1, "up")))
        H3MlxVideoBackend().extend_video(model, source, "p", extend_frames=17, seed=3, steps=8, segment_dir=segment_dir, audio_output=False)

        assert env.staged[0]["frames"] == 25

    @pytest.mark.parametrize("steps", [0, 1, 6, 12])
    def test_steps_outside_the_schedules_are_refused_before_a_pipeline_is_built(self, env, written, tmp_path, steps):
        with pytest.raises(ValueError, match=rf"MiniMax H3 runs 4 or 8 steps, got {steps}\."):
            H3MlxVideoBackend().text_to_video(_load(env), "a fox", 64, 32, 22, 7, steps, str(tmp_path / "out.mp4"))
        assert env.loads == []

    def test_steps_outside_the_schedules_are_refused_for_images_and_extension(self, env, written, tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="runs 4 or 8 steps, got 6"):
            H3MlxVideoBackend().image_to_video(_load(env, mode="i2v"), _image(tmp_path / "a.png", (64, 32)), "a fox", 64, 32, 22, 7, 6, str(tmp_path / "out.mp4"))
        source = make_extend_source(path=tmp_path / "clip.mp4", width=64, height=32, frame_count=49)
        read = []
        monkeypatch.setattr(h3_video, "read_source_tail", lambda _source: read.append(True))
        with pytest.raises(ValueError, match="runs 4 or 8 steps, got 6"):
            H3MlxVideoBackend().extend_video(_load(env, mode="extend"), source, "p", extend_frames=17, seed=3, steps=6, segment_dir=tmp_path, audio_output=False)
        assert env.loads == [] and read == []
