"""Pin the vendored YuE2 inference kit that the audio backend calls, including the private hooks its CUDA weight swap uses.

The kit is read as a syntax tree, so a failure after ``make update-yue2`` names the class, signature or attribute that
moved. The weight swap reaches into private parts of the kit (``_model``, ``_load_model``, ``CachedNAR``, ``Chunk``,
the token constants and the layer attribute names), which is why they are listed here. One import check at the end
proves the package loads against the installed torch and transformers.
"""

from __future__ import annotations

import ast
import importlib
import re

from tests._paths import REPO_ROOT

ROOT = REPO_ROOT
PACKAGE = ROOT / "packages" / "yue2"
MAKEFILE = ROOT / "Makefile"

EXCLUDED_FILES = ("cli.py", "fast.py")
VENDORED_NOTICES = ("LICENSE", "THIRD_PARTY_NOTICES.md", "licenses")


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


def _keyword_only(function: ast.FunctionDef) -> list[str]:
    return [arg.arg for arg in function.args.kwonlyargs]


def _default(function: ast.FunctionDef, name: str) -> object:
    node = _parameters(function)[name]
    assert node is not None, f"{name} has no default"
    return ast.literal_eval(node)


def _assigned(tree: ast.Module, names: tuple[str, ...]) -> dict[str, object]:
    """Return the literal values assigned to ``names`` at module level, unpacking tuple assignments."""
    found: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in names:
                found[target.id] = ast.literal_eval(node.value)
            elif isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                for element, value in zip(target.elts, node.value.elts, strict=True):
                    if isinstance(element, ast.Name) and element.id in names:
                        found[element.id] = ast.literal_eval(value)
    return found


def _self_stores(scope: ast.ClassDef) -> set[str]:
    """Return the attributes ``__init__`` assigns on ``self``."""
    init = _function(scope, "__init__")
    return {node.attr for node in ast.walk(init) if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self" and isinstance(node.ctx, ast.Store)}


def _calls(function: ast.FunctionDef) -> list[ast.Call]:
    return [node for node in ast.walk(function) if isinstance(node, ast.Call)]


def _call_sources(function: ast.FunctionDef) -> set[str]:
    return {ast.unparse(call.func) for call in _calls(function)}


def _calls_to(function: ast.FunctionDef, callee: str) -> list[ast.Call]:
    return [call for call in _calls(function) if ast.unparse(call.func) == callee]


def _makefile_commit() -> str:
    match = re.search(r"^YUE2_COMMIT\s*\?=\s*(\S+)\s*$", MAKEFILE.read_text(encoding="utf-8"), re.MULTILINE)
    assert match is not None, "Makefile defines no YUE2_COMMIT"
    return match.group(1)


# ---- The vendored tree -----------------------------------------------------------------------------------------------


def test_pinned_commit_matches_the_makefile() -> None:
    commit = _makefile_commit()
    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    assert (PACKAGE / "UPSTREAM_COMMIT").read_text(encoding="utf-8").strip() == commit


def test_command_line_and_server_modules_are_not_vendored() -> None:
    for name in EXCLUDED_FILES:
        assert not (PACKAGE / name).exists()


def test_licence_and_notices_are_vendored() -> None:
    for name in VENDORED_NOTICES:
        assert (PACKAGE / name).exists()
    assert "Apache License" in (PACKAGE / "LICENSE").read_text(encoding="utf-8")


def test_only_the_vllm_backend_reaches_the_excluded_fast_module() -> None:
    """``pipeline`` imports ``fast`` lazily and only for ``backend="vllm"``, which the audio backend never selects."""
    pipeline = _class(_tree("pipeline.py"), "YuE2Pipeline")
    for name in ("_generate", "synthesize", "close"):
        function = _function(pipeline, name)
        imports = [node for node in ast.walk(function) if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module == "fast"]
        assert imports, f"{name} no longer imports fast; recheck that cli.py and fast.py can stay excluded"
        guarded = [node for node in ast.walk(function) if isinstance(node, ast.If) and "vllm" in ast.unparse(node.test)]
        assert guarded, f"{name} imports fast without a vllm guard"
    for module in ("__init__.py", "nar.py", "protocol.py", "sampling.py", "modeling_yue2.py", "modeling_vae.py", "storage.py", "progress.py", "tokenization_yue2.py"):
        source = (PACKAGE / module).read_text(encoding="utf-8")
        assert not re.search(r"from \.(fast|cli)\b|import (fast|cli)\b", source), module


# ---- Public entry points ---------------------------------------------------------------------------------------------


def test_from_pretrained_takes_the_arguments_the_backend_passes() -> None:
    """``device``, ``memory_budget_gib``, ``offload_ar`` and ``vae_core_frames`` travel through ``**kwargs`` to ``__init__``."""
    pipeline = _class(_tree("pipeline.py"), "YuE2Pipeline")
    from_pretrained = _function(pipeline, "from_pretrained")
    assert any(isinstance(d, ast.Name) and d.id == "classmethod" for d in from_pretrained.decorator_list)
    assert from_pretrained.args.kwarg is not None
    explicit = _parameters(from_pretrained)
    assert {"model", "vae", "progress"} <= set(explicit)
    assert _default(from_pretrained, "vae") == "m-a-p/YuE2-Vae"
    assert _default(from_pretrained, "progress") is True
    forwarded = _calls_to(from_pretrained, "cls")
    assert len(forwarded) == 1
    assert [keyword.arg for keyword in forwarded[0].keywords] == ["progress", None], "from_pretrained must forward progress and **kwargs to the constructor"

    init = _function(pipeline, "__init__")
    assert {"device", "memory_budget_gib", "offload_ar", "vae_core_frames", "progress"} <= set(_keyword_only(init))
    assert _default(init, "device") == "auto"
    assert _default(init, "offload_ar") is False
    assert _default(init, "vae_core_frames") is None


def test_call_signature() -> None:
    """``seed`` and ``cot`` travel through ``**kwargs`` into ``SongRequest``."""
    call = _function(_class(_tree("pipeline.py"), "YuE2Pipeline"), "__call__")
    assert list(_parameters(call))[:2] == ["style", "lyrics"]
    assert {"semantic_sampling", "cancelled", "on_token", "abc_sampling"} <= set(_keyword_only(call))
    assert call.args.kwarg is not None
    request = _class(_tree("protocol.py"), "SongRequest")
    fields = {node.target.id: node for node in request.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
    assert {"style", "lyrics", "cot", "seed"} <= set(fields)
    assert ast.literal_eval(fields["cot"].value) == "full"
    assert isinstance(fields["seed"].value, ast.Constant) and isinstance(fields["seed"].value.value, int)


def test_cot_values() -> None:
    instructions = next(node.value for node in _tree("protocol.py").body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "INSTRUCTIONS" for t in node.targets))
    assert isinstance(instructions, ast.Dict)
    assert {ast.literal_eval(key) for key in instructions.keys if key is not None} == {"off", "melody", "full"}


def test_close_releases_the_model() -> None:
    close = _function(_class(_tree("pipeline.py"), "YuE2Pipeline"), "close")
    assert list(_parameters(close)) == []
    stored = [ast.unparse(target) for node in ast.walk(close) if isinstance(node, ast.Assign) for target in node.targets]
    assert any("self._model" in target for target in stored)


# ---- Private hooks the CUDA weight swap relies on --------------------------------------------------------------------


def test_the_pipeline_stages_the_backend_runs_one_by_one_exist() -> None:
    pipeline = _class(_tree("pipeline.py"), "YuE2Pipeline")
    plan = _function(pipeline, "plan")
    assert list(_parameters(plan))[:2] == ["style", "lyrics"]
    assert {"abc_sampling", "cancelled", "on_token"} <= set(_parameters(plan))
    generate_semantic = _function(pipeline, "generate_semantic")
    assert list(_parameters(generate_semantic)) == ["plan", "sampling", "cancelled", "on_token"]
    assert list(_parameters(_function(pipeline, "synthesize"))) == ["semantic", "cancelled"]
    decode = _function(pipeline, "decode")
    assert list(_parameters(decode)) == ["latents", "full", "vae"]
    assert list(_parameters(_function(pipeline, "_load_model"))) == ["for_nar"]


def test_the_model_is_held_in_the_private_model_attribute_and_loaded_through_load_model() -> None:
    """The swap replaces ``pipe._model`` and ``pipe._load_model``; every stage must obtain the model through them."""
    pipeline = _class(_tree("pipeline.py"), "YuE2Pipeline")
    assert "_model" in _self_stores(pipeline)
    for stage in ("_generate", "synthesize"):
        assert "self._load_model" in _call_sources(_function(pipeline, stage)), stage
    load_model = _function(pipeline, "_load_model")
    assert "YuE2ForCausalLM.from_pretrained" in _call_sources(load_model)
    from_pretrained = _calls_to(load_model, "YuE2ForCausalLM.from_pretrained")[0]
    assert {"local_files_only", "torch_dtype", "low_cpu_mem_usage"} <= {keyword.arg for keyword in from_pretrained.keywords}


def test_synthesize_passes_the_steps_and_offload_options_from_the_pipeline() -> None:
    synthesize = _function(_class(_tree("pipeline.py"), "YuE2Pipeline"), "synthesize")
    call = _calls_to(synthesize, "synthesize")[0]
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
    assert keywords["steps"] == "self.generation_config.ode_steps"
    assert keywords["context"] == "self.generation_config.context"
    assert keywords["offload_ar"] == "self.offload_ar"
    assert keywords["cancelled"] == "cancelled"


def test_the_pipeline_and_stage_results_expose_what_the_backend_reads() -> None:
    """The backend reads ``model_dir`` and ``generation_config`` off the pipeline, and the plan's and semantic result's fields."""
    tree = _tree("pipeline.py")
    assert {"model_dir", "generation_config", "_model"} <= _self_stores(_class(tree, "YuE2Pipeline"))

    def fields(name: str) -> list[str]:
        return [node.target.id for node in _class(tree, name).body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)]

    assert {"request", "abc_ids", "prefix", "truncated"} <= set(fields("SymbolicPlan"))
    assert {"plan", "tokens", "truncated"} <= set(fields("SemanticResult"))


def test_decode_moves_the_model_to_the_cpu() -> None:
    """The swap's GPU-resident tensors go to the CPU here, so the backend puts them back at the start of every take."""
    decode = _function(_class(_tree("pipeline.py"), "YuE2Pipeline"), "decode")
    moves = [ast.unparse(call) for call in _calls_to(decode, "self._model.to")]
    assert 'self._model.to("cpu")' in moves or "self._model.to('cpu')" in moves


def test_the_process_memory_fraction_is_capped_in_the_constructor() -> None:
    """The constructor caps the whole process; the backend sets the fraction back to 1.0 after construction and on close."""
    init = _function(_class(_tree("pipeline.py"), "YuE2Pipeline"), "__init__")
    caps = _calls_to(init, "torch.cuda.set_per_process_memory_fraction")
    assert len(caps) == 1
    assert ast.unparse(caps[0].args[0]) == "min(budget / total, 1)"
    budget = [ast.unparse(node.value) for node in ast.walk(init) if isinstance(node, ast.Assign) and any(ast.unparse(t) == "budget" for t in node.targets)]
    assert budget == ["min((self.memory_budget_gib - 2) * 2 ** 30, total - 2 * 2 ** 30)"]


def test_the_constructor_gets_the_weight_identity_from_the_module_level_model_identity() -> None:
    """The backend replaces ``pipeline.model_identity`` with a size check while the pipeline is built, so the constructor must look it up there."""
    tree = _tree("pipeline.py")
    init = _function(_class(tree, "YuE2Pipeline"), "__init__")
    assert len(_calls_to(init, "model_identity")) == 2
    imported = [alias.name for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "storage" for alias in node.names]
    assert "model_identity" in imported


def test_the_weight_identity_has_the_shape_the_size_check_returns() -> None:
    """The size check stands in for ``storage.model_identity``: the same folder in, the same record out, with the manifest's ``files`` and ``bytes``."""
    identity = _function(_tree("storage.py"), "model_identity")
    assert list(_parameters(identity)) == ["path", "verify"]
    returned = next(node.value for node in ast.walk(identity) if isinstance(node, ast.Return))
    assert isinstance(returned, ast.Dict)
    assert [ast.literal_eval(key) for key in returned.keys if key is not None] == ["files", "config_sha256"]
    sources = ast.unparse(identity)
    assert "weights_manifest.json" in sources
    assert "'bytes'" in sources
    assert "'sha256'" in sources


def test_chunk_fields() -> None:
    chunk = _class(_tree("nar.py"), "Chunk")
    fields = [node.target.id for node in chunk.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)]
    assert fields == ["ar_tokens", "noise", "nar_cond_end"]
    decorators = [ast.unparse(d) for d in chunk.decorator_list]
    assert decorators == ["dataclass"]


def test_cached_nar_interface() -> None:
    cached = _class(_tree("nar.py"), "CachedNAR")
    init = _function(cached, "__init__")
    assert list(_parameters(init)) == ["model", "chunk", "attention", "query_chunk_size"]
    assert _default(init, "attention") == "sdpa"
    assert _default(init, "query_chunk_size") is None
    solve = _function(cached, "solve")
    assert list(_parameters(solve)) == ["steps", "cancelled", "on_progress"]
    assert _default(solve, "steps") == 32
    assert list(_parameters(_function(cached, "close"))) == []
    assert "self._prefill" in _call_sources(init), "CachedNAR must still prefill in its constructor"


def test_cached_nar_reads_the_model_attributes_the_swap_groups_name() -> None:
    cached = _class(_tree("nar.py"), "CachedNAR")
    sources = {ast.unparse(node) for node in ast.walk(cached) if isinstance(node, ast.Attribute)}
    for needed in ("model.vae2llm", "model.latent_pos_embed", "model.llm2vae", "model.model.rotary_emb", "model.model.norm", "layer.nar_self_attn", "layer.nar_mlp"):
        assert needed in sources, needed


def test_synthesize_splits_the_way_the_backend_reproduces_it() -> None:
    """The backend rebuilds each chunk and runs prefill and solve separately; the kit's own loop must stay the same shape."""
    nar = _tree("nar.py")
    synthesize = _function(nar, "synthesize")
    assert list(_parameters(synthesize)) == [
        "model",
        "prefix",
        "codec",
        "seed",
        "steps",
        "context",
        "attention",
        "offload_ar",
        "cancelled",
        "query_chunk_size",
        "on_progress",
    ]
    assert "song_chunks" in _call_sources(synthesize)
    engine = _calls_to(synthesize, "CachedNAR")
    assert [ast.unparse(arg) for arg in engine[0].args] == ["model", "chunk", "attention", "query_chunk_size"]
    solve = _calls_to(synthesize, "engine.solve")
    assert [ast.unparse(arg) for arg in solve[0].args] == ["steps", "cancelled"]
    assert _calls_to(synthesize, "engine.close")
    assert _calls_to(synthesize, "torch.cat")


def test_chunk_construction_and_noise_draw_match_the_backend() -> None:
    """``song_chunks`` draws one seeded CPU noise tensor, then builds ``Chunk(prefix + codec + CODEC_OFFSET + [MUSIC_END], noise)``."""
    song_chunks = _function(_tree("nar.py"), "song_chunks")
    assert list(_parameters(song_chunks)) == ["prefix", "codec", "seed", "context"]
    generator = next(node for node in ast.walk(song_chunks) if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "generator")
    assert ast.unparse(generator.value) == "torch.Generator(device='cpu').manual_seed(int(seed))"
    draw = next(node for node in ast.walk(song_chunks) if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "noise")
    assert ast.unparse(draw.value) == "torch.randn((len(codec), 64), dtype=torch.float32, device='cpu', generator=generator)"
    chunk = _calls_to(song_chunks, "Chunk")
    assert len(chunk) == 1
    assert [ast.unparse(arg) for arg in chunk[0].args] == ["prefix + [value + CODEC_OFFSET for value in codec[a:b]] + [MUSIC_END]", "noise[a:b]"]
    assert "chunk_ranges" in _call_sources(song_chunks)


def test_token_constants() -> None:
    values = _assigned(_tree("protocol.py"), ("CODEC_OFFSET", "CODEC_SIZE", "MUSIC_END", "MUSIC_START", "CONTEXT", "EOD"))
    assert values == {"EOD": 151643, "MUSIC_START": 151851, "MUSIC_END": 151852, "CODEC_OFFSET": 151853, "CODEC_SIZE": 32768, "CONTEXT": 24576}


def test_ode_steps_default() -> None:
    config = _class(_tree("protocol.py"), "GenerationConfig")
    fields = {node.target.id: node for node in config.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
    assert ast.literal_eval(fields["ode_steps"].value) == 32
    assert ast.literal_eval(fields["ode_method"].value) == "midpoint"
    assert ast.unparse(fields["context"].value) == "CONTEXT"


def test_sampling_fields_the_backend_overrides() -> None:
    """The backend passes ``max_tokens`` through the ``abc_sampling`` and ``sampling`` dictionaries."""
    sampling = _class(_tree("protocol.py"), "Sampling")
    fields = [node.target.id for node in sampling.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)]
    assert "max_tokens" in fields
    resolve = _function(_tree("protocol.py"), "resolve_sampling")
    assert list(_parameters(resolve)) == ["value", "default"]


def test_model_layer_attributes_the_swap_groups_name() -> None:
    """AR group: ``embed_tokens``, ``lm_head``, per-layer ``self_attn`` and ``mlp``; NAR group: ``nar_self_attn`` and ``nar_mlp``."""
    tree = _tree("modeling_yue2.py")
    layer = _self_stores(_class(tree, "DecoderLayer"))
    assert {"input_layernorm", "self_attn", "post_attention_layernorm", "mlp", "nar_input_layernorm", "nar_self_attn", "nar_pre_mlp_layernorm", "nar_mlp"} <= layer
    backbone = _self_stores(_class(tree, "Backbone"))
    assert {"embed_tokens", "layers", "norm", "rotary_emb"} <= backbone
    causal = _self_stores(_class(tree, "YuE2ForCausalLM"))
    assert {"model", "lm_head", "llm2vae", "vae2llm", "time_embedder", "latent_pos_embed"} <= causal


def test_the_offload_helper_moves_the_ar_modules_the_swap_also_names() -> None:
    offload = _function(_tree("nar.py"), "_offload_ar")
    sources = {ast.unparse(node) for node in ast.walk(offload) if isinstance(node, ast.Attribute)}
    assert {"model.model.embed_tokens", "model.lm_head", "model.model.layers", "layer.self_attn", "layer.mlp"} <= sources


# ---- Import check ----------------------------------------------------------------------------------------------------


def test_the_package_imports_against_the_installed_libraries() -> None:
    package = importlib.import_module("yue2")
    pipeline = importlib.import_module("yue2.pipeline")
    nar = importlib.import_module("yue2.nar")
    protocol = importlib.import_module("yue2.protocol")
    importlib.import_module("yue2.modeling_yue2")
    importlib.import_module("yue2.modeling_vae")
    assert package.YuE2Pipeline is pipeline.YuE2Pipeline
    assert nar.CachedNAR and nar.Chunk and protocol.CODEC_OFFSET and protocol.MUSIC_END and protocol.CONTEXT


def test_attention_calls_the_module_level_sdpa_the_mps_fold_replaces() -> None:
    """``yue2_mps.folded_decode_attention`` swaps ``modeling_yue2.sdpa``; the layers must look it up by that name."""
    modeling = _tree("modeling_yue2.py")
    sdpa = _function(modeling, "sdpa")
    assert list(_parameters(sdpa)) == ["query", "key", "value", "attn_mask", "is_causal"]
    assert _keyword_only(sdpa) == ["attn_mask", "is_causal"]
    forward = _function(_class(modeling, "Attention"), "forward")
    assert "sdpa" in _call_sources(forward)


def test_the_causal_lm_computes_its_logits_through_the_lm_head_attribute() -> None:
    """``yue2_mps.semantic_head`` replaces ``model.lm_head`` for the song tokens."""
    forward = _function(_class(_tree("modeling_yue2.py"), "YuE2ForCausalLM"), "forward")
    assert "self.lm_head" in _call_sources(forward)


def test_the_song_phase_samples_only_the_rows_the_mps_head_computes() -> None:
    """``yue2_mps.semantic_head`` computes codec tokens and the end of music only; the kit's song phase must allow exactly those."""
    import torch
    from yue2.protocol import CODEC_OFFSET, CODEC_SIZE, MUSIC_END, Sampling
    from yue2.sampling import distribution

    sampling = Sampling(temperature=0, top_p=1.0, top_k=10**6, repetition_penalty=1.0, penalty_window=50, min_tokens=0, max_tokens=1)
    scores = distribution(torch.zeros(1, CODEC_OFFSET + CODEC_SIZE), sampling, [], 10_000, "semantic")

    allowed = torch.isfinite(scores[0]).nonzero().flatten().tolist()
    assert allowed == [MUSIC_END, *range(CODEC_OFFSET, CODEC_OFFSET + CODEC_SIZE)]
