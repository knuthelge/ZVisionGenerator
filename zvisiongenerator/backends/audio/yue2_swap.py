"""Keep YuE2's big weights in pinned CPU memory and put only the running phase's group on the GPU.

The model is one set of layers used twice: the autoregressive (AR) path plans and writes song tokens, and the acoustic
(NAR) path turns them into audio latents. Each phase needs only its own group, so the other group waits in CPU memory.
Everything small (norms, embedders, projections, buffers) stays on the GPU the whole time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor, nn

__all__ = ["WeightSwap"]


class WeightSwap:
    """Own a ``YuE2ForCausalLM`` loaded on the CPU and move its weight groups to the GPU phase by phase.

    The AR group is ``embed_tokens``, ``lm_head`` and each layer's ``self_attn`` and ``mlp``. The NAR group is each layer's
    ``nar_self_attn`` and ``nar_mlp``. The kit's ``decode()`` moves the whole model to the CPU, which takes the resident
    tensors with it, so :meth:`restore_resident` puts them back and runs at the start of every take.
    """

    def __init__(self, model: nn.Module, device: str) -> None:
        """Pin the groups in CPU memory and move everything else to *device*.

        Args:
            model: The model, loaded on the CPU in bf16.
            device: The torch device that holds the running phase, such as ``cuda``.
        """
        self.model, self.device = model, device
        layers = model.model.layers
        self._embed: list[nn.Module] = [model.model.embed_tokens]
        self._head: list[nn.Module] = [model.lm_head]
        self._ar_layers: list[nn.Module] = [module for layer in layers for module in (layer.self_attn, layer.mlp)]
        self._nar_layers: list[nn.Module] = [module for layer in layers for module in (layer.nar_self_attn, layer.nar_mlp)]
        self._groups = self._embed + self._head + self._ar_layers + self._nar_layers
        self._pinned: dict[int, Tensor] = {}
        grouped = {id(parameter) for module in self._groups for parameter in module.parameters()}
        for module in self._groups:
            for parameter in module.parameters():
                parameter.data = parameter.data.cpu().pin_memory()
                self._pinned[id(parameter)] = parameter.data
        self._resident = [parameter for parameter in model.parameters() if id(parameter) not in grouped]
        self._on_device = {id(module): False for module in self._groups}
        self.restore_resident()

    def restore_resident(self) -> None:
        """Move the resident parameters and the buffers back to the device, wherever they are now."""
        for parameter in self._resident:
            if parameter.device.type != self.device:
                parameter.data = parameter.data.to(self.device)
        for buffer in self.model.buffers():
            if buffer.device.type != self.device:
                buffer.data = buffer.data.to(self.device)

    def phase(self, name: str) -> None:
        """Put the group *name* needs on the device and every other group in CPU memory.

        The phases of a take, in order: ``ar`` (AR tokens) needs the AR group, ``prefill`` (the NAR prefill) the AR group
        without ``lm_head``, ``ode`` (the NAR ODE solve) the NAR group and ``none`` (the VAE decode) no group.
        """
        import torch

        wanted = {
            "ar": self._embed + self._head + self._ar_layers,
            "prefill": self._embed + self._ar_layers,
            "ode": self._nar_layers,
            "none": [],
        }[name]
        keep = {id(module) for module in wanted}
        self._place([module for module in self._groups if id(module) not in keep], on_device=False)
        torch.cuda.empty_cache()
        self._place(wanted, on_device=True)
        torch.cuda.synchronize()

    def release(self) -> None:
        """Drop the pinned copies. The caller drops the model too."""
        self._pinned.clear()
        self._resident = []
        self._groups = []
        self._on_device.clear()

    def _place(self, modules: list[nn.Module], *, on_device: bool) -> None:
        for module in modules:
            if self._on_device[id(module)] == on_device:
                continue
            for parameter in module.parameters():
                pinned = self._pinned[id(parameter)]
                parameter.data = pinned.to(self.device, non_blocking=True) if on_device else pinned
            self._on_device[id(module)] = on_device
