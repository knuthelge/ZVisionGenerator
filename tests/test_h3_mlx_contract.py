"""Pin the vendored MiniMax H3 MLX fork that the macOS H3 backend will call.

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
    """Map every parameter name (``self`` excluded) to its default node, or ``None`` when it has none."""
    args = function.args
    positional = [*args.posonlyargs, *args.args]
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    result = {arg.arg: default for arg, default in zip(positional, defaults, strict=True)}
    result.update({arg.arg: default for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True)})
    result.pop("self", None)
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
    expected = {"prompt", "duration_seconds", "num_inference_steps", "seed", "images", "keyframe_anchors", "height", "width", "verbose"}
    assert expected <= set(parameters)
    assert parameters["prompt"] is None, "prompt is required"


def test_call_reads_the_text_encoder_exactly_once() -> None:
    """The one-shot encoder proxy relies on a single read of ``self.text_encoder`` inside ``__call__``."""
    call = _function(_pipeline_class(), "__call__")
    reads = [
        node
        for node in ast.walk(call)
        if isinstance(node, ast.Attribute) and node.attr == "text_encoder" and isinstance(node.value, ast.Name) and node.value.id == "self" and isinstance(node.ctx, ast.Load)
    ]
    assert len(reads) == 1


def test_low_memory_mode_refuses_images() -> None:
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
    assert list(inspect.signature(MiniMaxH3Pipeline.from_pretrained).parameters) == ["cls", *_parameters(function)]
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
