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
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.backends.memory_cuda import drop_cached_file, drop_cached_files, release_memory

if TYPE_CHECKING:
    import torch

__all__ = [
    "cast_to_fp8",
    "load_fp8_components",
    "load_nf4_components",
    "write_fp8_copy",
]

# Text-encoder layers kept in the compute dtype: norms and embeddings (diffusers' defaults cover the rest).
_TEXT_ENCODER_SKIP_PATTERNS = ("pos_embed", "patch_embed", "norm", "embed", "^proj_in$", "^proj_out$", "lm_head")


def cast_to_fp8(component: Any, compute_dtype: torch.dtype) -> Any:
    """Store *component*'s layer weights in FP8 and compute in *compute_dtype*, in place; return *component*.

    Weights already in FP8 stay as they are, so this also prepares a component loaded from a stored FP8 copy.
    Each FP8 layer also declares *compute_dtype* (see :func:`_declare_compute_dtype`) so LoRAs load on top of it.
    """
    import torch

    if hasattr(component, "enable_layerwise_casting"):
        # A diffusers model knows which of its layers must stay in the compute dtype.
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
    transformer = _stream_fp8_transformer(_component_dir(model_path, "transformer"), compute_dtype)
    return {"text_encoder": text_encoder, "transformer": transformer}


def write_fp8_copy(source: Path, target: Path, compute_dtype: torch.dtype, cancelled: Callable[[], bool] | None = None) -> None:
    """Write an FP8 copy of the diffusers model at *source* into the new directory *target*.

    The text encoder and transformer are loaded, cast and saved one at a time; after each, its memory is
    released and its source and new files leave the page cache. ``model_index.json`` and the other components it
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
    for name, load in loaders.items():
        if cancelled is not None and cancelled():
            return
        component = load()
        component.save_pretrained(str(target / name))
        del component
        release_memory()
        drop_cached_files(source / name)
        drop_cached_files(target / name)
    _link_tree(source / "model_index.json", target / "model_index.json")
    for name in _listed_components(source):
        if name not in loaders and (source / name).exists():
            _link_tree(source / name, target / name)


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
    from diffusers import AutoModel as DiffusersAutoModel
    from transformers import AutoModel as HFAutoModel

    # A stored copy's configs carry its quantization; transformers rejects an explicit None config.
    te_options, tx_options = ({}, {}) if prequantized else ({"quantization_config": config} for config in _nf4_configs(compute_dtype))
    text_encoder = HFAutoModel.from_pretrained(model_path, subfolder="text_encoder", dtype=compute_dtype, **te_options)
    if text_encoder_to_cpu:
        text_encoder.to("cpu")
    transformer = DiffusersAutoModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=compute_dtype, **tx_options)
    return {"text_encoder": text_encoder, "transformer": transformer}


def _nf4_configs(compute_dtype: torch.dtype) -> tuple[Any, Any]:
    """Return matching NF4 quantization configs for the transformers text encoder and the diffusers transformer."""
    from diffusers import BitsAndBytesConfig as DiffusersBnBConfig
    from transformers import BitsAndBytesConfig as TransformersBnBConfig

    options = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": compute_dtype}
    return TransformersBnBConfig(**options), DiffusersBnBConfig(**options)


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


def _stream_fp8_transformer(component_dir: Path, compute_dtype: torch.dtype) -> Any:
    """Build the diffusers model in *component_dir* with FP8 weight storage, reading one tensor at a time.

    The model is created empty in *compute_dtype* and prepared with :func:`cast_to_fp8`, which fixes each
    parameter's storage dtype; every saved tensor is then cast to that dtype as it is read and put in place.
    Each weight file leaves the page cache once it is read. Saved bfloat16 (a source model) and saved FP8 (a stored copy) load the same way.

    Raises:
        RuntimeError: When the saved tensors do not cover the model (a damaged or partial copy).
    """
    import torch
    from accelerate import init_empty_weights
    from diffusers import AutoModel as DiffusersAutoModel
    from safetensors import safe_open

    files = _weight_files(component_dir)
    if not files:
        # Weights in another format (e.g. an old .bin checkpoint) cannot stream: load them in full, then cast.
        return cast_to_fp8(DiffusersAutoModel.from_pretrained(str(component_dir), torch_dtype=compute_dtype), compute_dtype)
    # AutoModel.from_config reads a path as a pipeline folder; a component's own config is passed as a dict.
    config = json.loads((component_dir / "config.json").read_text(encoding="utf-8"))
    with init_empty_weights():
        model = DiffusersAutoModel.from_config(config)
    model.to(compute_dtype)
    for name, module in model.named_modules():
        if any(part in (model._keep_in_fp32_modules or ()) for part in name.split(".")):
            module.to(torch.float32)
    cast_to_fp8(model, compute_dtype)
    storage = {key: tensor.dtype for key, tensor in model.state_dict().items()}
    loaded: set[str] = set()
    for path in files:
        with safe_open(str(path), framework="pt") as handle:
            tensors = {key: _owned(handle.get_tensor(key), storage[key]) for key in handle.keys() if key in storage}
        model.load_state_dict(tensors, strict=False, assign=True)
        loaded |= tensors.keys()
        del tensors
        drop_cached_file(path)  # files can add up to more than system memory
    missing = storage.keys() - loaded
    if missing:
        raise RuntimeError(f"The weights in {component_dir} do not cover the model ({len(missing)} tensors missing, e.g. {sorted(missing)[0]}).")
    return model.eval()


def _owned(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Return *tensor* in *dtype* in memory of its own.

    ``safe_open`` tensors map the file. One already in *dtype* (FP8 from a stored copy) would stay mapped, and
    under memory pressure its pages are dropped and read from disk again at every step.
    """
    return tensor.to(dtype) if tensor.dtype != dtype else tensor.clone()


def _weight_files(component_dir: Path) -> list[Path]:
    """Return the safetensors files diffusers would load from *component_dir* (the sharded index, else the single file)."""
    index = component_dir / "diffusion_pytorch_model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return [component_dir / name for name in sorted(set(weight_map.values()))]
    single = component_dir / "diffusion_pytorch_model.safetensors"
    return [single] if single.is_file() else sorted(component_dir.glob("*.safetensors"))


def _link_tree(source: Path, target: Path) -> None:
    """Hard-link *source* (a file or directory, following symlinks) to *target*, copying where linking fails."""
    if source.is_dir():
        shutil.copytree(source, target, copy_function=_link_or_copy, dirs_exist_ok=True)
    else:
        _link_or_copy(source, target)


def _link_or_copy(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
    """Hard-link the file *source* resolves to as *target*, or copy it when a link is impossible."""
    try:
        os.link(os.path.realpath(source), target)
    except OSError:
        shutil.copy2(source, target)
