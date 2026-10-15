"""Keep a model's layers in system memory and stream them to the GPU as they run.

Two mechanisms, chosen by how the weights are stored:

- :func:`group_offload` uses diffusers' group offloading, which prefetches the next block on a side stream.
  It moves raw tensor data, so it suits plain and FP8-stored weights.
- :func:`stream_blocks` swaps each block's parameters for GPU copies around its forward pass. It moves whole
  parameter objects, so it also carries bitsandbytes NF4 weights, whose quantization state diffusers' group
  offloading leaves behind on the CPU.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

__all__ = ["group_offload", "load_when_used", "move_outside", "offload_type", "stream_blocks", "unload_before"]


def offload_type(module: Any) -> str:
    """Return how to group-offload *module*: block by block when its layers form a top-level list, else leaf by leaf.

    Block-level offloading only splits a top-level layer list. Models that nest their layers deeper (Qwen3-VL's
    ``language_model``, Gemma 3's ``model.language_model``) would otherwise move to the GPU whole.
    """
    has_block_list = any(type(child).__name__ in ("ModuleList", "Sequential") for child in module.children())
    return "block_level" if has_block_list else "leaf_level"


def group_offload(module: Any, device: Any, kind: str | None = None) -> None:
    """Stream *module* to *device* with diffusers' group offloading, one block (or leaf) at a time.

    Args:
        module: The model to stream.
        device: The GPU the layers run on.
        kind: ``block_level`` or ``leaf_level``; by default chosen with :func:`offload_type`.
    """
    from diffusers.hooks import apply_group_offloading

    options = {"onload_device": device, "num_blocks_per_group": 1, "use_stream": True, "record_stream": True, "non_blocking": True, "low_cpu_mem_usage": True}
    apply_group_offloading(module, offload_type=kind or offload_type(module), **options)


def stream_blocks(blocks: Iterable[Any], device: Any) -> list[Any]:
    """Keep each module of *blocks* on the CPU and give it GPU copies of its parameters only while it runs.

    The CPU parameters are pinned so the copies can run asynchronously. Before a block's forward pass every
    parameter is replaced by ``param.to(device)``; after it the CPU originals are put back and the copies are freed.
    A bitsandbytes NF4 parameter moves its small quantization state to *device* in place on its first copy, and it
    stays there. Buffers are small and move to *device* for good.

    Returns:
        The hook handles, so the streaming can be removed.
    """
    handles: list[Any] = []
    for block in blocks:
        _pin_parameters(block)
        _move_buffers(block, device)
        saved: list[tuple[Any, str, Any]] = []

        def _onload(module: Any, _args: Any, saved: list[tuple[Any, str, Any]] = saved) -> None:
            for owner in module.modules():
                for name, param in list(owner._parameters.items()):
                    if param is not None:
                        saved.append((owner, name, param))
                        owner._parameters[name] = param.to(device, non_blocking=True)

        def _offload(_module: Any, _args: Any, _output: Any, saved: list[tuple[Any, str, Any]] = saved) -> None:
            for owner, name, param in saved:
                owner._parameters[name] = param
            saved.clear()

        handles.append(block.register_forward_pre_hook(_onload))
        handles.append(block.register_forward_hook(_offload, always_call=True))  # also after a failed pass (out of memory)
    return handles


def load_when_used(component: Any, entry_points: Iterable[Any], device: Any) -> list[Any]:
    """Keep *component* where it is until one of its *entry_points* runs, then move all of it to *device*.

    For parts that run only briefly (a VAE that decodes at the end): they stay off the GPU while something larger
    uses it. Pair with :func:`unload_before` to send them back.

    Returns:
        The hook handles.
    """

    def _onload(_module: Any, _args: Any) -> None:
        if not _is_on(component, device):
            component.to(device)

    return [module.register_forward_pre_hook(_onload) for module in entry_points]


def unload_before(module: Any, components: Iterable[Any]) -> Any:
    """Move each of *components* back to the CPU, and free the GPU memory they held, before *module* runs.

    Returns:
        The hook handle.
    """
    import torch

    parts = list(components)

    def _offload(_module: Any, _args: Any) -> None:
        moved = False
        for part in parts:
            if not _is_on(part, torch.device("cpu")):
                part.to("cpu")
                moved = True
        if moved:
            torch.cuda.empty_cache()

    return module.register_forward_pre_hook(_offload)


def _is_on(module: Any, device: Any) -> bool:
    """Return whether *module*'s first parameter is on a device of *device*'s type (a module without parameters always is)."""
    import torch

    first = next(module.parameters(), None)
    return first is None or first.device.type == torch.device(device).type


def move_outside(model: Any, blocks: Any, device: Any) -> None:
    """Move every parameter and buffer of *model* that is not inside *blocks* to *device*, in place.

    Used with :func:`stream_blocks`: the streamed blocks stay on the CPU, the rest of the model (embeddings, norms,
    output projections) lives on the GPU.
    """
    streamed = {id(module) for module in blocks.modules()}
    for module in model.modules():
        if id(module) not in streamed:
            module._apply(lambda tensor: tensor.to(device), recurse=False)  # its own parameters and buffers only


def _move_buffers(module: Any, device: Any) -> None:
    """Move every buffer of *module* to *device* in place."""
    for owner in module.modules():
        for name, buffer in list(owner._buffers.items()):
            if buffer is not None:
                owner._buffers[name] = buffer.to(device)


def _pin_parameters(module: Any) -> None:
    """Move *module*'s CPU parameters into pinned memory in place, keeping each parameter object (and its type)."""
    for param in module.parameters():
        if param.device.type == "cpu" and not param.data.is_pinned():
            param.data = param.data.pin_memory()
