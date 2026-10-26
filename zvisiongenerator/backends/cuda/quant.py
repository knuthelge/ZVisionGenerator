"""Quantized weights for the diffusers image backend: NF4 at q4 and FP8 weight storage at q8.

q4 quantizes the transformer and text encoder to NF4 with bitsandbytes. Their weights live on the GPU, so the
NF4 transformer has to fit in VRAM.

q8 keeps the transformer's and text encoder's layer weights in FP8 and computes in bfloat16 (diffusers'
layerwise casting). The weights still stream to the GPU block by block, like unquantized ones, so every model
that runs unquantized also runs at q8, with half the system memory and half the data to move per step.

Both levels can be stored for reuse (see ``utils/stored_quant.py``): NF4 through diffusers' own pre-quantized
save, FP8 by casting one component at a time and saving it.

The FP8 transformer is built empty and filled one saved tensor at a time, each cast to its storage dtype as it
is read (see :func:`_stream_fp8_transformer`), so it is never held in memory at full precision.

A model whose weights are written one tensor at a time (the video transformers) pre-packs its NF4 linear layers
instead: :func:`pack_nf4` packs each weight on the GPU as it streams past, the copy is saved in the layout
``Linear4bit`` saves itself in, and :func:`swap_in_linear4bit` with :func:`place_nf4` load it back into
``Linear4bit`` layers on the CPU.
"""

from __future__ import annotations

import importlib.metadata
import json
import re
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.backends.cuda.memory import release_memory
from zvisiongenerator.utils.file_links import link_tree
from zvisiongenerator.utils.page_cache import drop_cached_files
from zvisiongenerator.utils.safetensors_stream import iter_tensors

if TYPE_CHECKING:
    import torch

__all__ = [
    "cast_to_fp8",
    "empty_diffusers_model",
    "fill_weights",
    "kept_in_compute_dtype",
    "load_fp8_components",
    "load_fp8_transformer",
    "load_nf4_components",
    "load_nf4_text_encoder",
    "load_nf4_transformer",
    "nf4_linear_names",
    "nf4_storage_format",
    "pack_nf4",
    "place_nf4",
    "save_components",
    "swap_in_linear4bit",
    "write_fp8_copy",
]

# Tensors assigned to an empty model at a time (bounds what is held beyond the model itself).
_FILL_BATCH_BYTES = 512 * 1024**2
_NF4_BLOCKSIZE = 64
# What one saved NF4 weight consists of, by name suffix: the packed data and its double-quantized state.
_NF4_PARTS = frozenset({"", "absmax", "quant_map", "nested_absmax", "nested_quant_map", "quant_state.bitsandbytes__nf4"})
# Text-encoder layers kept in the compute dtype: norms and embeddings (diffusers' defaults cover the rest).
_TEXT_ENCODER_SKIP_PATTERNS = ("pos_embed", "patch_embed", "norm", "embed", "^proj_in$", "^proj_out$", "lm_head")


def kept_in_compute_dtype(tensor_name: str) -> bool:
    """Return whether :func:`cast_to_fp8` keeps the text-encoder tensor *tensor_name* in the compute dtype."""
    module = tensor_name.rpartition(".")[0]
    return any(re.search(pattern, module) for pattern in _TEXT_ENCODER_SKIP_PATTERNS)


def cast_to_fp8(component: Any, compute_dtype: torch.dtype, keep_patterns: tuple[str, ...] = ()) -> Any:
    """Store *component*'s layer weights in FP8 and compute in *compute_dtype*, in place; return *component*.

    Weights already in FP8 stay as they are, so this also prepares a component loaded from a stored FP8 copy.
    Each FP8 layer also declares *compute_dtype* (see :func:`_declare_compute_dtype`) so LoRAs load on top of it.

    Args:
        component: The diffusers model or transformers text encoder to cast.
        compute_dtype: The dtype the layers compute in.
        keep_patterns: Regular expressions for module names of a diffusers model that stay in *compute_dtype*
            on top of diffusers' defaults, for layers the model reads for their dtype (see ``model_class`` of
            :func:`load_fp8_transformer`). Ignored for a text encoder, which has its own list.
    """
    import torch

    if hasattr(component, "enable_layerwise_casting"):
        # A diffusers model knows which of its layers must stay in the compute dtype.
        if keep_patterns:
            from diffusers.hooks.layerwise_casting import DEFAULT_SKIP_MODULES_PATTERN

            # Giving patterns replaces diffusers' defaults, so they are passed along.
            component.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=compute_dtype, skip_modules_pattern=(*DEFAULT_SKIP_MODULES_PATTERN, *keep_patterns))
        else:
            component.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=compute_dtype)
    else:
        from diffusers.hooks import apply_layerwise_casting

        apply_layerwise_casting(component, storage_dtype=torch.float8_e4m3fn, compute_dtype=compute_dtype, skip_modules_pattern=_TEXT_ENCODER_SKIP_PATTERNS)
    _declare_compute_dtype(component, torch.float8_e4m3fn, compute_dtype)
    return component


def _declare_compute_dtype(component: Any, storage_dtype: torch.dtype, compute_dtype: torch.dtype) -> None:
    """Set ``compute_dtype`` on every layer whose weight is stored in *storage_dtype*.

    peft creates a LoRA's adapter layers in the base layer's ``compute_dtype`` when it has one (as bitsandbytes
    layers do), else in its weight dtype. FP8 adapters could neither run (CUDA has no FP8 matmul) nor hold
    typical LoRA values, which underflow in FP8.
    """
    for module in component.modules():
        weight = getattr(module, "weight", None)
        if weight is not None and weight.dtype == storage_dtype:
            module.compute_dtype = compute_dtype


def load_fp8_components(model_path: str, compute_dtype: torch.dtype) -> dict[str, Any]:
    """Load the text encoder and transformer of *model_path*, or of a stored FP8 copy of it, with FP8 weight storage.

    The text encoder loads through transformers in *compute_dtype* and is cast right after (lossless for a stored
    copy). It loads first, so its full-precision weights are released before the transformer loads.
    """
    text_encoder = cast_to_fp8(_load_text_encoder(model_path, compute_dtype), compute_dtype)
    return {"text_encoder": text_encoder, "transformer": load_fp8_transformer(model_path, compute_dtype)}


def load_fp8_transformer(model_path: str, compute_dtype: torch.dtype, *, model_class: type[Any] | None = None, keep_patterns: tuple[str, ...] = ()) -> Any:
    """Load the transformer of *model_path*, or of a stored FP8 copy of it, with FP8 weight storage, streamed from its files.

    Args:
        model_path: The model directory or repo id.
        compute_dtype: The dtype the layers compute in.
        model_class: The class of a transformer diffusers' ``AutoModel`` cannot resolve by name (a custom class).
        keep_patterns: Regular expressions for layers that stay in *compute_dtype* on top of diffusers' defaults,
            for layers the model reads for their dtype.
    """
    return _stream_fp8_transformer(_component_dir(model_path, "transformer"), compute_dtype, model_class, keep_patterns)


def write_fp8_copy(source: Path, target: Path, compute_dtype: torch.dtype, cancelled: Callable[[], bool] | None = None) -> None:
    """Write an FP8 copy of the diffusers model at *source* into the new directory *target*.

    The text encoder and transformer are loaded, cast and saved one at a time; after each, its memory is
    released and its source and new files leave the page cache. A component saved already quantized (its config
    carries a ``quantization_config``, such as a pre-quantized NF4 text encoder) is linked as it is. ``model_index.json`` and the other components it
    lists are hard-linked (or copied, across file systems); other files in *source* are left out.
    When *cancelled* turns true the write stops after the current component, leaving an incomplete folder for the
    caller to discard.
    """
    target.mkdir(parents=True, exist_ok=True)
    # The components FP8 applies to; the VAE, tokenizer, scheduler and any second text encoder are linked as they are.
    loaders = {
        "text_encoder": lambda: cast_to_fp8(_load_text_encoder(str(source), compute_dtype), compute_dtype),
        "transformer": lambda: _stream_fp8_transformer(source / "transformer", compute_dtype),
    }
    loaders = {name: load for name, load in loaders.items() if not _is_prequantized(source / name)}
    if not save_components(source, target, loaders, cancelled):
        return
    link_tree(source / "model_index.json", target / "model_index.json")
    for name in _listed_components(source):
        if name not in loaders and (source / name).exists():
            link_tree(source / name, target / name)


def save_components(source: Path, target: Path, loaders: Mapping[str, Callable[[], Any]], cancelled: Callable[[], bool] | None = None) -> bool:
    """Load each component with its loader and save it into *target*, one at a time; return False if stopped.

    After each save the component's memory is released and its source and new files leave the page cache. *cancelled*
    is checked before each component; when it turns true the loop stops, leaving an incomplete folder for the caller
    to discard.

    Args:
        source: The model directory the components were loaded from (its files leave the page cache).
        target: The directory to save into; each component goes into ``target / name``.
        loaders: Maps a component's name to a function that returns it, quantized as wanted.
        cancelled: Returns true to stop.
    """
    for name, load in loaders.items():
        if cancelled is not None and cancelled():
            return False
        component = load()
        component.save_pretrained(str(target / name))
        del component
        release_memory()
        drop_cached_files(source / name)
        drop_cached_files(target / name)
    return True


def _is_prequantized(component_dir: Path) -> bool:
    """Return whether the component in *component_dir* is saved already quantized (its config has a ``quantization_config``)."""
    config = component_dir / "config.json"
    return config.is_file() and "quantization_config" in json.loads(config.read_text(encoding="utf-8"))


def _listed_components(model_dir: Path) -> list[str]:
    """Return the components *model_dir*'s ``model_index.json`` lists (entries with a library and class)."""
    index = json.loads((model_dir / "model_index.json").read_text(encoding="utf-8"))
    return [name for name, value in index.items() if isinstance(value, list) and value and value[0] is not None]


def load_nf4_components(model_path: str, compute_dtype: torch.dtype, *, prequantized: bool, text_encoder_to_cpu: bool) -> dict[str, Any]:
    """Load the text encoder and transformer of *model_path* quantized to NF4 on the GPU.

    Args:
        model_path: The model directory or repo id.
        compute_dtype: The dtype NF4 layers compute in.
        prequantized: Whether *model_path* is a stored NF4 copy, whose configs carry the quantization.
        text_encoder_to_cpu: Move the text encoder off the GPU before the transformer loads, for models
            whose NF4 text encoder and transformer do not fit a 10-12 GB card together.
    """
    text_encoder = load_nf4_text_encoder(model_path, compute_dtype, prequantized=prequantized)
    if text_encoder_to_cpu:
        text_encoder.to("cpu")
    return {"text_encoder": text_encoder, "transformer": load_nf4_transformer(model_path, compute_dtype, prequantized=prequantized)}


def load_nf4_text_encoder(model_path: str, compute_dtype: torch.dtype, *, prequantized: bool) -> Any:
    """Load the text encoder of *model_path* quantized to NF4 on the GPU (*prequantized*: a stored NF4 copy, loaded as saved)."""
    from transformers import AutoModel as HFAutoModel

    # A stored copy's configs carry its quantization; transformers rejects an explicit None config.
    options = {} if prequantized else {"quantization_config": _nf4_configs(compute_dtype)[0]}
    return HFAutoModel.from_pretrained(model_path, subfolder="text_encoder", dtype=compute_dtype, **options)


def load_nf4_transformer(model_path: str, compute_dtype: torch.dtype, *, prequantized: bool, model_class: type[Any] | None = None, keep_modules: tuple[str, ...] = ()) -> Any:
    """Load the transformer of *model_path* quantized to NF4 on the GPU (*prequantized*: a stored NF4 copy, loaded as saved).

    Args:
        model_path: The model directory or repo id.
        compute_dtype: The dtype NF4 layers compute in.
        prequantized: Whether *model_path* is a stored NF4 copy, whose config carries the quantization.
        model_class: The class of a transformer diffusers' ``AutoModel`` cannot resolve by name (a custom class).
        keep_modules: Names of modules left unquantized, for layers the model reads for their dtype. Only
            applied when quantizing at load; a stored copy keeps what it was saved with.
    """
    from diffusers import AutoModel as DiffusersAutoModel

    options: dict[str, Any] = {}
    if not prequantized:
        options["quantization_config"] = _nf4_configs(compute_dtype, keep_modules)[1]
    return (model_class or DiffusersAutoModel).from_pretrained(model_path, subfolder="transformer", torch_dtype=compute_dtype, **options)


def _nf4_configs(compute_dtype: torch.dtype, keep_modules: tuple[str, ...] = ()) -> tuple[Any, Any]:
    """Return matching NF4 quantization configs for the transformers text encoder and the diffusers transformer.

    *keep_modules* names modules the transformer config leaves unquantized.
    """
    from diffusers import BitsAndBytesConfig as DiffusersBnBConfig
    from transformers import BitsAndBytesConfig as TransformersBnBConfig

    options = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": compute_dtype}
    transformer_options = {**options, "llm_int8_skip_modules": list(keep_modules)} if keep_modules else options
    return TransformersBnBConfig(**options), DiffusersBnBConfig(**transformer_options)


def _load_text_encoder(model_path: str, dtype: torch.dtype) -> Any:
    """Load the text encoder of *model_path* on the CPU in *dtype*."""
    from transformers import AutoModel as HFAutoModel

    return HFAutoModel.from_pretrained(model_path, subfolder="text_encoder", dtype=dtype)


def _component_dir(model_path: str, name: str) -> Path:
    """Return the local folder of component *name* of *model_path*, downloading it for a Hugging Face repo id."""
    local = Path(model_path).expanduser() / name
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_path, allow_patterns=[f"{name}/*"])) / name


def _stream_fp8_transformer(component_dir: Path, compute_dtype: torch.dtype, model_class: type[Any] | None = None, keep_patterns: tuple[str, ...] = ()) -> Any:
    """Build the diffusers model in *component_dir* with FP8 weight storage, reading one tensor at a time.

    The model is created empty in *compute_dtype* and prepared with :func:`cast_to_fp8`, which fixes each
    parameter's storage dtype; every saved tensor is then cast to that dtype as it is read and put in place.
    Each tensor leaves the page cache once it is read. Saved bfloat16 (a source model) and saved FP8 (a stored copy) load the same way.
    *model_class* and *keep_patterns* are as for :func:`load_fp8_transformer`.

    Raises:
        RuntimeError: When the saved tensors do not cover the model (a damaged or partial copy).
    """
    from diffusers import AutoModel as DiffusersAutoModel

    files = _weight_files(component_dir)
    if not files:
        # Weights in another format (e.g. an old .bin checkpoint) cannot stream: load them in full, then cast.
        return cast_to_fp8((model_class or DiffusersAutoModel).from_pretrained(str(component_dir), torch_dtype=compute_dtype), compute_dtype, keep_patterns)
    # AutoModel.from_config reads a path as a pipeline folder; a component's own config is passed as a dict.
    config = json.loads((component_dir / "config.json").read_text(encoding="utf-8"))
    model = cast_to_fp8(empty_diffusers_model(config, compute_dtype, model_class), compute_dtype, keep_patterns)
    fill_weights(model, (item for path in files for item in iter_tensors(path)), str(component_dir))
    return model.eval()


def empty_diffusers_model(config: dict[str, Any], compute_dtype: torch.dtype, model_class: type[Any] | None = None) -> Any:
    """Return the diffusers model *config* describes, without weights (on the meta device), in *compute_dtype*.

    Modules the model keeps in float32 stay in float32. Non-persistent buffers (the rotary frequencies) keep the values
    and dtype the constructor gave them, as they do under ``from_pretrained(torch_dtype=...)``: no state dict holds them, so
    a cast to *compute_dtype* would round them for good. *model_class* builds models diffusers' ``AutoModel`` cannot
    resolve by name (pipeline-specific ones).
    """
    import torch
    from accelerate import init_empty_weights
    from diffusers import AutoModel as DiffusersAutoModel

    with init_empty_weights():
        model = (model_class or DiffusersAutoModel).from_config(config)
    derived = _derived_buffers(model)
    model.to(compute_dtype)
    for name, module in model.named_modules():
        if any(part in (model._keep_in_fp32_modules or ()) for part in name.split(".")):
            module.to(torch.float32)
    for module, leaf, buffer in derived:
        module._buffers[leaf] = buffer
    return model


def _derived_buffers(model: Any) -> list[tuple[Any, str, torch.Tensor]]:
    """Return ``(module, name, buffer)`` of every floating-point non-persistent buffer of *model*, as constructed."""
    return [
        (module, leaf, buffer)
        for module in model.modules()
        for leaf in module._non_persistent_buffers_set
        if (buffer := module._buffers.get(leaf)) is not None and buffer.is_floating_point() and not buffer.is_meta
    ]


def fill_weights(model: Any, tensors: Iterable[tuple[str, torch.Tensor]], source: str, skipped: Collection[str] = ()) -> None:
    """Put each ``(name, tensor)`` of *tensors* in place in the empty *model*, cast to the dtype the model stores it in.

    Tensors the model has no parameter or buffer for are ignored. They are assigned in small batches, so only a
    few are held at once beyond the model itself.

    Args:
        model: A model built without weights, its storage dtypes already set.
        tensors: The saved tensors, under the model's own names.
        source: Where the tensors come from, for the error message.
        skipped: Names the model has but the source need not provide (e.g. tied weights).

    Raises:
        RuntimeError: When the tensors do not cover the model (a damaged or partial copy).
    """
    storage = {key: tensor.dtype for key, tensor in model.state_dict().items()}
    loaded: set[str] = set()
    batch: dict[str, torch.Tensor] = {}
    batch_bytes = 0
    for name, tensor in tensors:
        if name not in storage:
            continue
        batch[name] = _owned(tensor, storage[name])
        batch_bytes += tensor.numel() * tensor.element_size()
        if batch_bytes >= _FILL_BATCH_BYTES:
            model.load_state_dict(batch, strict=False, assign=True)
            loaded |= batch.keys()
            batch, batch_bytes = {}, 0
    model.load_state_dict(batch, strict=False, assign=True)
    loaded |= batch.keys()
    missing = storage.keys() - loaded - set(skipped)
    if missing:
        raise RuntimeError(f"The weights in {source} do not cover the model ({len(missing)} tensors missing, e.g. {sorted(missing)[0]}).")


def _owned(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Return *tensor* in *dtype*, converting only when its dtype differs (it is already in memory of its own)."""
    return tensor.to(dtype) if tensor.dtype != dtype else tensor


def _weight_files(component_dir: Path) -> list[Path]:
    """Return the safetensors files diffusers would load from *component_dir* (the sharded index, else the single file)."""
    index = component_dir / "diffusion_pytorch_model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return [component_dir / name for name in sorted(set(weight_map.values()))]
    single = component_dir / "diffusion_pytorch_model.safetensors"
    return [single] if single.is_file() else sorted(component_dir.glob("*.safetensors"))


def nf4_storage_format() -> str:
    """Return the storage format of a pre-packed NF4 copy, ``nf4-bitsandbytes-<major>.<minor>``: it depends on bitsandbytes' serialization."""
    major_minor = ".".join(importlib.metadata.version("bitsandbytes").split(".")[:2])
    return f"nf4-bitsandbytes-{major_minor}"


def nf4_linear_names(model: Any, *, prefix: str, exclude: Collection[str] = ()) -> list[str]:
    """Return the names of *model*'s linear layers NF4 packs: those under *prefix*, less any with a name part in *exclude*.

    Args:
        model: The model (it may be empty).
        prefix: What a layer's name starts with to be packed.
        exclude: Name parts (the dot-separated pieces of a name) that keep a layer out; whole parts, not substrings.
    """
    import torch

    return [name for name, module in model.named_modules() if name.startswith(prefix) and isinstance(module, torch.nn.Linear) and not any(part in exclude for part in name.split("."))]


def pack_nf4(key: str, weight: torch.Tensor, device: Any) -> dict[str, torch.Tensor]:
    """Pack *weight* to NF4 on *device* and return it with its quantization state, named as ``Linear4bit`` saves them."""
    import torch
    from bitsandbytes.functional import quantize_4bit

    packed, state = quantize_4bit(weight.to(device=device, dtype=torch.bfloat16), blocksize=_NF4_BLOCKSIZE, compress_statistics=True, quant_type="nf4", quant_storage=torch.uint8)
    return {key: packed.cpu(), **{f"{key}.{name}": value.cpu() for name, value in state.as_dict(packed=True).items()}}


def swap_in_linear4bit(model: Any, names: Iterable[str], compute_dtype: torch.dtype) -> dict[str, Any]:
    """Replace the linears *names* of the empty *model* with empty ``Linear4bit`` layers; return them by weight name."""
    import torch
    from bitsandbytes.nn import Linear4bit

    layers: dict[str, Any] = {}
    for name in names:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        linear = getattr(parent, child)
        layer = Linear4bit(
            linear.in_features, linear.out_features, bias=linear.bias is not None, compute_dtype=compute_dtype, compress_statistics=True, quant_type="nf4", quant_storage=torch.uint8, device="meta"
        )
        setattr(parent, child, layer)
        layers[f"{name}.weight"] = layer
    return layers


def place_nf4(tensors: Iterable[tuple[str, torch.Tensor]], layers: Mapping[str, Any]) -> Iterator[tuple[str, torch.Tensor]]:
    """Put each NF4 weight of *tensors* into its ``Linear4bit`` layer in *layers*, and yield every other tensor.

    A weight's parts are saved in one shard but not next to each other, so they are gathered until complete.

    Raises:
        RuntimeError: When a weight's parts are incomplete at the end.
    """
    from bitsandbytes.nn import Params4bit

    parts: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in tensors:
        owner = _nf4_owner(name, layers)
        if owner is None:
            yield name, tensor
            continue
        found = parts.setdefault(owner, {})
        found[name[len(owner) :].lstrip(".")] = tensor
        if found.keys() == _NF4_PARTS:
            data = found.pop("")
            layer = layers[owner]
            layer.weight = Params4bit.from_prequantized(data=data, quantized_stats=found, requires_grad=False, device="cpu", module=layer)
            del parts[owner]
    if parts:
        raise RuntimeError(f"The NF4 weight {sorted(parts)[0]} is incomplete in the prepared copy.")


def _nf4_owner(name: str, layers: Mapping[str, Any]) -> str | None:
    """Return the NF4 weight *name* belongs to (the weight itself or part of its quantization state), else ``None``."""
    if name in layers:
        return name
    head, separator, _ = name.partition(".weight.")
    owner = f"{head}.weight"
    return owner if separator and owner in layers else None
