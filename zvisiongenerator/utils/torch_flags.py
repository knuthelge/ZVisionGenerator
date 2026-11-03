"""Snapshot and restore the process-wide PyTorch numerics flags that audio model code changes.

The vendored audio kits switch these flags when a model is built (deterministic cuDNN, TF32 off, the strictest float32
matmul precision). They are global, so a backend that is closed would leave every later image and video job slower.
A backend takes a :class:`TorchFlags` snapshot before it loads and restores it when it closes.

PyTorch keeps the TF32 and matmul-precision settings in per-backend ``fp32_precision`` values; the older ``allow_tf32``
flags and ``set_float32_matmul_precision`` write to the same values. Code that mixes the two leaves them out of step, so
``torch.get_float32_matmul_precision()`` raises. The snapshot holds every ``fp32_precision`` value as well, so a mixed
state is restored exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["TorchFlags", "capture_torch_flags", "restore_torch_flags"]


@dataclass(frozen=True)
class TorchFlags:
    """The global PyTorch flags an audio model may change.

    Attributes:
        cudnn_deterministic: ``torch.backends.cudnn.deterministic``.
        cudnn_benchmark: ``torch.backends.cudnn.benchmark``.
        matmul_tf32: ``torch.backends.cuda.matmul.allow_tf32``.
        cudnn_tf32: ``torch.backends.cudnn.allow_tf32``.
        matmul_fp16_reduced_reduction: ``torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction``.
        float32_matmul_precision: ``torch.get_float32_matmul_precision()``, or ``None`` when PyTorch reports the
            matmul backends as out of step (it raises then).
        cuda_matmul_precision: ``torch.backends.cuda.matmul.fp32_precision``.
        mkldnn_matmul_precision: ``torch.backends.mkldnn.matmul.fp32_precision``.
        cudnn_precision: ``torch.backends.cudnn.fp32_precision``.
        cudnn_conv_precision: ``torch.backends.cudnn.conv.fp32_precision``.
        cudnn_rnn_precision: ``torch.backends.cudnn.rnn.fp32_precision``.
    """

    cudnn_deterministic: bool
    cudnn_benchmark: bool
    matmul_tf32: bool
    cudnn_tf32: bool
    matmul_fp16_reduced_reduction: bool
    float32_matmul_precision: str | None
    cuda_matmul_precision: str
    mkldnn_matmul_precision: str
    cudnn_precision: str
    cudnn_conv_precision: str
    cudnn_rnn_precision: str


def capture_torch_flags() -> TorchFlags:
    """Return the current values of the flags in :class:`TorchFlags`."""
    import torch

    backends = torch.backends
    try:
        matmul_precision: str | None = str(torch.get_float32_matmul_precision())
    except RuntimeError:  # the legacy and the new settings disagree
        matmul_precision = None
    return TorchFlags(
        cudnn_deterministic=bool(backends.cudnn.deterministic),
        cudnn_benchmark=bool(backends.cudnn.benchmark),
        matmul_tf32=bool(backends.cuda.matmul.allow_tf32),
        cudnn_tf32=bool(backends.cudnn.allow_tf32),
        matmul_fp16_reduced_reduction=bool(backends.cuda.matmul.allow_fp16_reduced_precision_reduction),
        float32_matmul_precision=matmul_precision,
        cuda_matmul_precision=str(backends.cuda.matmul.fp32_precision),
        mkldnn_matmul_precision=str(backends.mkldnn.matmul.fp32_precision),
        cudnn_precision=str(backends.cudnn.fp32_precision),
        cudnn_conv_precision=str(backends.cudnn.conv.fp32_precision),
        cudnn_rnn_precision=str(backends.cudnn.rnn.fp32_precision),
    )


def restore_torch_flags(flags: TorchFlags) -> None:
    """Set the global flags back to the values in *flags*.

    The coarse settings go first and the per-backend precisions last, so the per-backend values are what stays.
    """
    import torch

    backends = torch.backends
    backends.cudnn.deterministic = flags.cudnn_deterministic
    backends.cudnn.benchmark = flags.cudnn_benchmark
    backends.cuda.matmul.allow_tf32 = flags.matmul_tf32
    backends.cudnn.allow_tf32 = flags.cudnn_tf32
    backends.cuda.matmul.allow_fp16_reduced_precision_reduction = flags.matmul_fp16_reduced_reduction
    if flags.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(flags.float32_matmul_precision)
    backends.cudnn.fp32_precision = flags.cudnn_precision
    backends.cudnn.conv.fp32_precision = flags.cudnn_conv_precision
    backends.cudnn.rnn.fp32_precision = flags.cudnn_rnn_precision
    backends.cuda.matmul.fp32_precision = flags.cuda_matmul_precision
    backends.mkldnn.matmul.fp32_precision = flags.mkldnn_matmul_precision
