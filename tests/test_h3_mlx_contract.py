"""Pin the vendored MiniMax H3 MLX fork that the macOS H3 backend calls.

The fork needs MLX, which Linux does not have, so almost everything here reads the vendored source as a syntax tree
without importing it. A failure after ``make update-h3-mlx`` means the fork changed a name, a signature or a behaviour
the backend relies on. The import checks at the end run only where MLX is installed.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packages" / "minimax_h3_mlx"
MAKEFILE = ROOT / "Makefile"
BACKEND = ROOT / "zvisiongenerator" / "backends" / "mac" / "h3_video.py"

# The commit that ``make update-h3-mlx`` vendors. Until the macOS backend defines ``H3_MLX_COMMIT`` this value is pinned
# here, so bumping the Makefile alone fails the test.
PINNED_COMMIT = "3fcc9f8c557cf353400b1acf737b985e524a509f"

REMOVED_ENTRY_POINTS = ("generate_cli", "doctor", "cleanroom_smoke")


def _tree(name: str) -> ast.Module:
    return ast.parse((PACKAGE / name).read_text(encoding="utf-8"))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _function(scope: ast.Module | ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(node for node in scope.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _parameters(function: ast.FunctionDef) -> dict[str, ast.expr | None]:
    """Map every parameter name (``self`` and ``cls`` excluded) to its default node, or ``None`` when it has none."""
    args = function.args
    positional = [*args.posonlyargs, *args.args]
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    result = {arg.arg: default for arg, default in zip(positional, defaults, strict=True)}
    result.update({arg.arg: default for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True)})
    result.pop("self", None)
    result.pop("cls", None)
    return result


def _default(function: ast.FunctionDef, name: str) -> object:
    node = _parameters(function)[name]
    assert node is not None, f"{name} has no default"
    return ast.literal_eval(node)


def _assigned(tree: ast.Module, name: str) -> ast.expr:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return node.value
    raise AssertionError(f"{name} is not assigned at module level")


def _makefile_commit() -> str:
    match = re.search(r"^H3_MLX_COMMIT\s*\?=\s*(\S+)\s*$", MAKEFILE.read_text(encoding="utf-8"), re.MULTILINE)
    assert match is not None, "Makefile defines no H3_MLX_COMMIT"
    return match.group(1)


def _pipeline_class() -> ast.ClassDef:
    return _class(_tree("pipeline.py"), "MiniMaxH3Pipeline")


# ---- Pipeline surface ------------------------------------------------------------------------------------------------


def test_from_pretrained_takes_the_arguments_the_backend_passes() -> None:
    parameters = _parameters(_function(_pipeline_class(), "from_pretrained"))
    expected = {
        "checkpoint_dir",
        "transformer_dir",
        "text_encoder_dir",
        "turbo_lora_path",
        "turbo_lora_alpha",
        "turbo_lora_scale",
        "sigma_shift_video",
        "sigma_shift_audio",
        "stream_blocks",
        "low_memory",
        "load_vision",
        "memory_limit_gb",
        "verbose",
    }
    assert expected <= set(parameters)
    assert parameters["checkpoint_dir"] is None, "checkpoint_dir is required"


def test_from_pretrained_defaults() -> None:
    function = _function(_pipeline_class(), "from_pretrained")
    assert _default(function, "memory_limit_gb") == 16.0
    assert _default(function, "turbo_lora_scale") == 1.0
    assert _default(function, "low_memory") is False
    assert _default(function, "load_vision") is False
    assert _default(function, "stream_blocks") is False
    assert _default(function, "transformer_dir") is None
    assert _default(function, "text_encoder_dir") is None


def test_from_pretrained_is_a_classmethod() -> None:
    function = _function(_pipeline_class(), "from_pretrained")
    assert [ast.unparse(decorator) for decorator in function.decorator_list] == ["classmethod"]


def test_call_takes_the_arguments_the_backend_passes() -> None:
    parameters = _parameters(_function(_pipeline_class(), "__call__"))
    expected = {"prompt", "duration_seconds", "num_inference_steps", "seed", "height", "width", "verbose"}
    assert expected <= set(parameters)
    assert parameters["prompt"] is None, "prompt is required"


def test_low_memory_mode_refuses_images() -> None:
    """The reason image modes run through the staged path in ``h3_staged`` instead of the low-memory pipeline."""
    call = _function(_pipeline_class(), "__call__")
    guards = [
        node
        for node in ast.walk(call)
        if isinstance(node, ast.If)
        and any(
            isinstance(statement, ast.Raise) and isinstance(statement.exc, ast.Call) and isinstance(statement.exc.func, ast.Name) and statement.exc.func.id == "NotImplementedError"
            for statement in node.body
        )
    ]
    assert len(guards) == 1
    guard = ast.unparse(guards[0].test)
    assert "self._low_memory" in guard
    assert "images" in guard


def test_generation_result_fields() -> None:
    result = _class(_tree("pipeline.py"), "GenerationResult")
    fields = [node.target.id for node in result.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)]
    assert fields == ["video", "audio", "sample_rate", "fps", "seconds_per_step", "total_seconds", "block_cache_stats"]


# ---- Staged image path -----------------------------------------------------------------------------------------------
# ``h3_staged`` composes these pieces of the fork; each pin is a name, a signature or a ``self`` read it depends on.


def _self_reads(function: ast.FunctionDef) -> set[str]:
    """Return the attributes *function* reads from ``self``."""
    return {node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self" and isinstance(node.ctx, ast.Load)}


def _self_writes(function: ast.FunctionDef) -> set[str]:
    return {node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self" and isinstance(node.ctx, ast.Store)}


@pytest.mark.parametrize(
    ("name", "reads"),
    [
        ("_row_timestep_plan", set()),
        ("_encode_keyframes", {"_dit_config", "_video_config", "video_vae"}),
        ("_decode_video", {"_dit_config", "_video_config", "video_vae"}),
        ("_decode_audio", {"_audio_config", "audio_vae"}),
    ],
)
def test_the_pipeline_methods_the_staged_path_calls_read_only_the_attributes_it_supplies(name: str, reads: set[str]) -> None:
    assert _self_reads(_function(_pipeline_class(), name)) == reads


@pytest.mark.parametrize(
    ("name", "parameters"),
    [
        ("_row_timestep_plan", ["layout", "video_timesteps", "audio_timesteps"]),
        ("_encode_keyframes", ["images", "height", "width"]),
        ("_decode_video", ["rows", "num_latent_frames", "latent_height", "latent_width"]),
        ("_decode_audio", ["rows", "num_audio_latents"]),
    ],
)
def test_the_pipeline_methods_the_staged_path_calls_keep_their_signatures(name: str, parameters: list[str]) -> None:
    assert list(_parameters(_function(_pipeline_class(), name))) == parameters


def test_the_keyframe_constants() -> None:
    packing = _tree("packing.py")
    assert ast.literal_eval(_assigned(packing, "KEYFRAME_ENCODE_SEED")) == 42
    assert ast.literal_eval(_assigned(packing, "KEYFRAME_NOISE_AUG")) == 0.999


def _module_function(module: str, name: str) -> ast.FunctionDef:
    return _function(_tree(module), name)


@pytest.mark.parametrize(
    ("module", "name", "parameters"),
    [
        ("packing.py", "build_packed_sequence", ["text_token_tags", "num_latent_frames", "latent_height", "latent_width", "num_audio_latents", "patch_size", "keyframe_anchors"]),
        ("packing.py", "patchify_video_latents", ["latents", "patch_size"]),
        ("packing.py", "align_num_frames", ["num_frames"]),
        ("packing.py", "video_latent_num_frames", ["num_frames"]),
        ("packing.py", "audio_latent_num_frames", ["num_frames"]),
        ("load.py", "load_video_vae", ["model_dir", "strict"]),
        ("load.py", "load_audio_vae", ["model_dir", "strict"]),
        ("load.py", "read_video_vae_config", ["model_dir"]),
        ("load.py", "read_audio_vae_config", ["model_dir"]),
        (
            "streaming.py",
            "load_streaming_dit",
            ["model_dir", "turbo_lora_path", "turbo_lora_alpha", "turbo_lora_scale", "block_load_mode", "stream_block_group_size", "verbose"],
        ),
    ],
)
def test_the_functions_the_staged_path_calls_keep_their_signatures(module: str, name: str, parameters: list[str]) -> None:
    assert list(_parameters(_module_function(module, name))) == parameters


def test_the_modulation_cache_builds_from_a_block_provider() -> None:
    cache = _class(_tree("adaln.py"), "ModulationCache")
    assert list(_parameters(_function(cache, "build_streaming"))) == ["dit", "block_provider", "timesteps", "dtype"]


def test_the_scheduler_surface() -> None:
    scheduler = _class(_tree("scheduler.py"), "MiniMaxH3Scheduler")
    assert list(_parameters(_function(scheduler, "__init__"))) == ["shift"]
    assert list(_parameters(_function(scheduler, "set_timesteps"))) == ["num_inference_steps", "sigmas"]
    assert list(_parameters(_function(scheduler, "scale_noise"))) == ["sample", "timestep", "noise"]
    assert list(_parameters(_function(scheduler, "step"))) == ["model_output", "timestep", "sample"]


def test_the_transformer_call_takes_the_packed_arguments_in_order() -> None:
    call = _function(_class(_tree("dit.py"), "MiniMaxH3DiT"), "__call__")
    parameters = list(_parameters(call))
    assert parameters[:10] == ["video_latents", "audio_latents", "text_embeds", "timestep", "timestep_indices", "token_tags", "position_ids", "video_indices", "audio_indices", "text_indices"]
    assert {"modulation_cache", "block_provider"} <= set(parameters)


def test_the_text_encoder_surface() -> None:
    encoder = _class(_tree("text_encoder.py"), "MiniMaxH3TextEncoder")
    assert {"model_dir", "load_vision", "tokenizer_dir", "processor_dir", "stream_layers"} <= set(_parameters(_function(encoder, "__init__")))
    assert list(_parameters(_function(encoder, "build_request"))) == ["prompt", "images"]
    assert list(_parameters(_function(encoder, "_hidden_states"))) == ["input_ids", "position_ids", "inputs_embeds", "visual_pos_masks", "deepstack_visual_embeds"]
    assert {"language", "image_token_id", "_processor"} <= _self_writes(_function(encoder, "__init__"))


def test_the_text_encoder_uses_a_preset_processor_instead_of_loading_its_own() -> None:
    """The staged path seeds ``_processor`` with the PIL image processor, since the default loader needs torchvision."""
    getter = _function(_class(_tree("text_encoder.py"), "MiniMaxH3TextEncoder"), "processor")
    source = ast.unparse(getter)
    assert "if self._processor is None" in source and source.rstrip().endswith("return self._processor")


def test_build_request_reads_the_processors_image_processor() -> None:
    request = ast.unparse(_function(_class(_tree("text_encoder.py"), "MiniMaxH3TextEncoder"), "build_request"))
    assert "self.processor.image_processor(images=images, return_tensors='np')" in request
    assert "self.processor.image_processor.merge_size" in request


# ---- Turbo LoRA ------------------------------------------------------------------------------------------------------


def test_turbo_lora_targets() -> None:
    targets = ast.literal_eval(_assigned(_tree("turbo_lora.py"), "TARGETS"))
    assert targets == ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")


def _turbo_key_pattern() -> re.Pattern[str]:
    call = _assigned(_tree("turbo_lora.py"), "_KEY")
    assert isinstance(call, ast.Call)
    return re.compile(ast.literal_eval(call.args[0]))


@pytest.mark.parametrize(
    "key",
    [
        "transformer_blocks.3.attn.to_q.lora_A.default.weight",
        "base_model.model.transformer_blocks.0.ff.net.2.lora_B.default.weight",
        "token_refiner.refiner_blocks.1.attn.to_out.0.lora_A.default.weight",
        "diffusion_model.token_refiner.refiner_blocks.0.ff.net.0.proj.lora_B.default.weight",
    ],
)
def test_turbo_lora_accepts_block_and_refiner_keys(key: str) -> None:
    assert _turbo_key_pattern().fullmatch(key) is not None


@pytest.mark.parametrize(
    "key",
    [
        "single_blocks.0.attn.to_q.lora_A.default.weight",
        "transformer_blocks.0.attn.to_q.lora_A.weight",
        "transformer_blocks.0.attn.add_q_proj.lora_A.default.weight",
        "transformer_blocks.0.norm1.lora_B.default.weight",
        "token_refiner.proj_in.lora_A.default.weight",
    ],
)
def test_turbo_lora_rejects_other_keys(key: str) -> None:
    assert _turbo_key_pattern().fullmatch(key) is None


def test_turbo_lora_provider_signature() -> None:
    provider = _class(_tree("turbo_lora.py"), "TurboLoRAProvider")
    parameters = _parameters(_function(provider, "__init__"))
    assert list(parameters) == ["path", "num_blocks", "num_refiner_blocks", "hidden_size", "inner_dim", "ffn_hidden_size", "alpha", "scale"]
    assert _default(_function(provider, "__init__"), "scale") == 1.0
    assert _default(_function(provider, "__init__"), "alpha") is None


def test_turbo_lora_default_alpha() -> None:
    assert ast.literal_eval(_assigned(_tree("turbo_lora.py"), "DEFAULT_TURBO_LORA_ALPHA")) == 8.0


# ---- Text encoder converter ------------------------------------------------------------------------------------------


def test_quantize_text_encoder_convert_signature() -> None:
    function = _function(_tree("quantize_text_encoder.py"), "convert")
    assert list(_parameters(function)) == ["source", "output", "bits", "group_size", "num_layers"]
    assert [arg.arg for arg in function.args.args] == ["source", "output"]
    assert _default(function, "bits") == 4
    assert _default(function, "group_size") == 64
    assert _default(function, "num_layers") == 50


# ---- Layout ----------------------------------------------------------------------------------------------------------


def test_vendored_layout() -> None:
    for name in ("__init__.py", "pipeline.py", "turbo_lora.py", "quantize_text_encoder.py", "release_assets.json", "LICENSE"):
        assert (PACKAGE / name).is_file(), name
    for name in REMOVED_ENTRY_POINTS:
        assert not (PACKAGE / f"{name}.py").exists(), name
    assert "Apache License" in (PACKAGE / "LICENSE").read_text(encoding="utf-8")


def _imports_of_the_package() -> list[tuple[str, str, int]]:
    """List ``(file, module, level)`` for each import of the package's own modules."""
    found: list[tuple[str, str, int]] = []
    for path in sorted(PACKAGE.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                if node.level == 1 and node.module:
                    found.append((path.name, node.module, 1))
                elif node.level == 0 and node.module and node.module.split(".")[0] == "minimax_h3_mlx":
                    found.append((path.name, node.module.removeprefix("minimax_h3_mlx.") if "." in node.module else "", 0))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "minimax_h3_mlx":
                        found.append((path.name, alias.name.removeprefix("minimax_h3_mlx.") if "." in alias.name else "", 0))
    return found


def test_package_imports_resolve_inside_the_vendored_copy() -> None:
    imports = _imports_of_the_package()
    assert imports, "the fork imports its own modules"
    for file, module, _level in imports:
        if not module:
            continue
        head = module.split(".")[0]
        assert (PACKAGE / f"{head}.py").is_file() or (PACKAGE / head).is_dir(), f"{file} imports {module}"
        assert head not in REMOVED_ENTRY_POINTS, f"{file} imports the removed {head}"


def test_package_imports_itself_under_the_vendored_name() -> None:
    """The fork's absolute imports use ``minimax_h3_mlx``, so the directory must keep that name."""
    absolute = [module for _file, module, level in _imports_of_the_package() if level == 0]
    assert absolute
    assert PACKAGE.name == "minimax_h3_mlx"


# ---- Pin and packaging -----------------------------------------------------------------------------------------------


def _backend_commit() -> str | None:
    if not BACKEND.is_file():
        return None
    for node in ast.parse(BACKEND.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "H3_MLX_COMMIT" and node.value is not None:
            return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "H3_MLX_COMMIT" for target in node.targets):
            return ast.literal_eval(node.value)
    return None


def test_makefile_commit_matches_the_pin() -> None:
    commit = _makefile_commit()
    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    backend = _backend_commit()
    assert commit == (backend if backend is not None else PINNED_COMMIT)


def test_makefile_has_the_vendoring_target() -> None:
    assert re.search(r"^update-h3-mlx:", MAKEFILE.read_text(encoding="utf-8"), re.MULTILINE)


def test_packaging_declares_the_vendored_package_and_its_dependency() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "packages/minimax_h3_mlx" in pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert "mlx-vlm==0.6.10; sys_platform == 'darwin'" in pyproject["project"]["dependencies"]


# ---- Import checks (macOS only) --------------------------------------------------------------------------------------


def test_imported_pipeline_matches_the_syntax_tree() -> None:
    pytest.importorskip("mlx.core")
    import inspect

    from minimax_h3_mlx.pipeline import GenerationResult, MiniMaxH3Pipeline

    function = _function(_pipeline_class(), "from_pretrained")
    assert list(inspect.signature(MiniMaxH3Pipeline.from_pretrained).parameters) == list(_parameters(function))
    assert list(inspect.signature(MiniMaxH3Pipeline.__call__).parameters)[1:] == list(_parameters(_function(_pipeline_class(), "__call__")))
    assert list(GenerationResult.__dataclass_fields__) == ["video", "audio", "sample_rate", "fps", "seconds_per_step", "total_seconds", "block_cache_stats"]


def test_imported_turbo_lora_targets() -> None:
    pytest.importorskip("mlx.core")
    import inspect

    from minimax_h3_mlx.turbo_lora import TARGETS, TurboLoRAProvider

    assert len(TARGETS) == 6
    parameters = list(inspect.signature(TurboLoRAProvider.__init__).parameters)[1:]
    assert parameters == list(_parameters(_function(_class(_tree("turbo_lora.py"), "TurboLoRAProvider"), "__init__")))


def test_imported_converter_signature() -> None:
    pytest.importorskip("mlx.core")
    import inspect

    from minimax_h3_mlx.quantize_text_encoder import convert

    assert list(inspect.signature(convert).parameters) == ["source", "output", "bits", "group_size", "num_layers"]


def test_imported_staged_path_dependencies() -> None:
    pytest.importorskip("mlx.core")
    import inspect

    from minimax_h3_mlx.load import load_audio_vae, load_video_vae, read_audio_vae_config, read_video_vae_config
    from minimax_h3_mlx.packing import KEYFRAME_ENCODE_SEED, KEYFRAME_NOISE_AUG, build_packed_sequence
    from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline
    from minimax_h3_mlx.streaming import load_streaming_dit

    assert (KEYFRAME_ENCODE_SEED, KEYFRAME_NOISE_AUG) == (42, 0.999)
    for function, module, name in (
        (load_video_vae, "load.py", "load_video_vae"),
        (load_audio_vae, "load.py", "load_audio_vae"),
        (read_video_vae_config, "load.py", "read_video_vae_config"),
        (read_audio_vae_config, "load.py", "read_audio_vae_config"),
        (build_packed_sequence, "packing.py", "build_packed_sequence"),
        (load_streaming_dit, "streaming.py", "load_streaming_dit"),
    ):
        assert list(inspect.signature(function).parameters) == list(_parameters(_module_function(module, name)))
    for name in ("_row_timestep_plan", "_encode_keyframes", "_decode_video", "_decode_audio"):
        assert list(inspect.signature(getattr(MiniMaxH3Pipeline, name)).parameters)[1:] == list(_parameters(_function(_pipeline_class(), name)))


def test_mlx_vlm_vision_sanitize_moves_the_patch_embedding_to_the_mlx_layout_and_keeps_that_layout() -> None:
    """The text-encoder copy holds the checkpoint's PyTorch conv layout; ``VisionModel.sanitize`` converts it at load time."""
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_vlm")
    import inspect

    from mlx_vlm.models.qwen3_vl.config import VisionConfig
    from mlx_vlm.models.qwen3_vl.vision import VisionModel

    assert list(inspect.signature(VisionModel.sanitize).parameters) == ["self", "weights"]
    assert isinstance(inspect.getattr_static(VisionConfig, "from_dict"), classmethod)
    # sanitize reads no model state.
    torch_layout = mx.zeros((8, 3, 2, 16, 16))
    key = "patch_embed.proj.weight"
    (moved,) = VisionModel.sanitize(None, {key: torch_layout}).values()
    assert moved.shape == (8, 2, 16, 16, 3)
    (kept,) = VisionModel.sanitize(None, {key: moved}).values()
    assert kept.shape == (8, 2, 16, 16, 3)


def test_mlx_vlm_rope_index_and_vision_call_signatures() -> None:
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_vlm")
    import inspect

    from mlx_vlm.models.qwen3_vl.language import LanguageModel
    from mlx_vlm.models.qwen3_vl.vision import VisionModel

    assert list(inspect.signature(LanguageModel.get_rope_index).parameters) == ["self", "input_ids", "image_grid_thw", "video_grid_thw", "attention_mask"]
    assert list(inspect.signature(VisionModel.__call__).parameters)[:3] == ["self", "hidden_states", "grid_thw"]


def test_the_pil_image_processor_loads_without_torchvision_and_reads_keyframes(tmp_path: Path) -> None:
    """The release's processor config names the fast class, which needs torchvision; ``backend="pil"`` picks the PIL class."""
    pytest.importorskip("transformers")
    pytest.importorskip("PIL")
    import json

    from PIL import Image
    from transformers import AutoImageProcessor

    config = {
        "size": {"longest_edge": 16777216, "shortest_edge": 65536},
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "processor_class": "Qwen3VLProcessor",
        "image_processor_type": "Qwen2VLImageProcessorFast",
    }
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(config), encoding="utf-8")

    processor = AutoImageProcessor.from_pretrained(str(tmp_path), backend="pil")
    processed = processor(images=[Image.new("RGB", (576, 320), (10, 20, 30))], return_tensors="np")

    assert processor.merge_size == 2
    assert processed["image_grid_thw"].tolist() == [[1, 20, 36]]
    assert processed["pixel_values"].shape == (20 * 36, 3 * 2 * 16 * 16)
