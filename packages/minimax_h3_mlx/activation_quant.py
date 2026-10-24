"""Activation-aware affine INT8 quantization using the native MLX ABI."""

from __future__ import annotations

import contextvars
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map_with_path


_ACTIVE_RECORDER: contextvars.ContextVar["ActivationRecorder | None"] = contextvars.ContextVar(
    "minimax_h3_activation_recorder",
    default=None,
)


@dataclass(frozen=True)
class ActivationAwareResult:
    weight: mx.array
    scales: mx.array
    biases: mx.array
    baseline_error: float
    candidate_error: float
    improved_group_fraction: float


class ActivationDataset:
    """Calibration and holdout activation rows keyed by exact module path."""

    def __init__(self, manifest: dict[str, Any], rows: dict[str, dict[str, np.ndarray]]) -> None:
        self.manifest = manifest
        self.rows = rows

    def get(self, split: str, path: str) -> np.ndarray:
        try:
            return self.rows[split][path]
        except KeyError as exc:
            raise KeyError(f"activation dataset has no {split!r} rows for {path!r}") from exc

    def paths(self, split: str) -> set[str]:
        return set(self.rows.get(split, {}))

    def save(self, path: str | Path) -> None:
        arrays: dict[str, np.ndarray] = {
            "__manifest__": np.asarray(json.dumps(self.manifest, sort_keys=True))
        }
        for split, split_rows in self.rows.items():
            for module_path, values in split_rows.items():
                arrays[f"{split}::{module_path}"] = np.asarray(values, dtype=np.float32)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "ActivationDataset":
        rows: dict[str, dict[str, np.ndarray]] = {}
        with np.load(path, allow_pickle=False) as archive:
            manifest = json.loads(str(archive["__manifest__"].item()))
            for key in archive.files:
                if key == "__manifest__":
                    continue
                split, module_path = key.split("::", 1)
                rows.setdefault(split, {})[module_path] = np.asarray(
                    archive[key], dtype=np.float32
                )
        return cls(manifest, rows)


class ActivationRecorder:
    """Deterministically retain bounded real activation rows for calibration."""

    def __init__(self, max_rows_per_call: int = 8, max_rows_per_layer: int = 64) -> None:
        if max_rows_per_call <= 0 or max_rows_per_layer <= 0:
            raise ValueError("activation row limits must be positive")
        self.max_rows_per_call = int(max_rows_per_call)
        self.max_rows_per_layer = int(max_rows_per_layer)
        self.split = "calibration"
        self._rows: dict[str, dict[str, list[np.ndarray]]] = {}
        self._token: contextvars.Token | None = None
        self._path_predicate: Callable[[str], bool] | None = None

    def set_split(self, split: str) -> None:
        if not split:
            raise ValueError("activation split must be non-empty")
        self.split = split

    def set_path_predicate(self, predicate: Callable[[str], bool] | None) -> None:
        """Restrict subsequent recording to matching module paths."""

        self._path_predicate = predicate

    def __enter__(self) -> "ActivationRecorder":
        self._token = _ACTIVE_RECORDER.set(self)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._token is not None:
            _ACTIVE_RECORDER.reset(self._token)
            self._token = None

    def record(self, path: str, value: mx.array) -> None:
        if self._path_predicate is not None and not self._path_predicate(path):
            return
        flat = value.reshape(-1, value.shape[-1])
        row_count = min(int(flat.shape[0]), self.max_rows_per_call)
        if row_count == 0:
            return
        if int(flat.shape[0]) == row_count:
            selected = flat
        else:
            indices = np.linspace(0, int(flat.shape[0]) - 1, row_count, dtype=np.int32)
            selected = flat[mx.array(indices)]
        host = np.asarray(selected.astype(mx.float32))
        bucket = self._rows.setdefault(self.split, {}).setdefault(path, [])
        retained = sum(rows.shape[0] for rows in bucket)
        remaining = self.max_rows_per_layer - retained
        if remaining > 0:
            bucket.append(np.array(host[:remaining], copy=True))

    def dataset(self, manifest: dict[str, Any]) -> ActivationDataset:
        rows = {
            split: {
                path: np.concatenate(chunks, axis=0)
                for path, chunks in split_rows.items()
                if chunks
            }
            for split, split_rows in self._rows.items()
        }
        return ActivationDataset(manifest, rows)


def bind_activation_paths(model: nn.Module) -> None:
    """Attach exact tree paths so shared projection classes can report inputs."""

    def bind(path: str, module: nn.Module):
        module._activation_quant_path = path
        return module

    leaves = tree_map_with_path(bind, model.leaf_modules(), is_leaf=nn.Module.is_module)
    model.update_modules(leaves)


def capture_linear_input(layer: nn.Module, value: mx.array) -> None:
    recorder = _ACTIVE_RECORDER.get()
    path = getattr(layer, "_activation_quant_path", None)
    if recorder is not None and path is not None:
        recorder.record(path, value)


def project_linear(layer: nn.Module, value: mx.array) -> mx.array:
    capture_linear_input(layer, value)
    return layer(value)


def _round_to_dtype(values: np.ndarray, dtype: mx.Dtype) -> np.ndarray:
    rounded = mx.array(np.asarray(values, dtype=np.float32)).astype(dtype)
    return np.asarray(rounded.astype(mx.float32))


def _pack_uint32(codes: np.ndarray, bits: int) -> np.ndarray:
    if 32 % bits:
        raise ValueError(f"bits={bits} does not evenly pack into uint32")
    per_word = 32 // bits
    if codes.shape[-1] % per_word:
        raise ValueError(
            f"code width {codes.shape[-1]} is not divisible by {per_word} values per uint32"
        )
    grouped = codes.astype(np.uint32).reshape(*codes.shape[:-1], -1, per_word)
    shifts = (np.arange(per_word, dtype=np.uint32) * bits).reshape(
        *((1,) * (grouped.ndim - 1)), per_word
    )
    return np.bitwise_or.reduce(grouped << shifts, axis=-1)


def _affine_dequantized(
    codes: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
) -> np.ndarray:
    return codes.astype(np.float32) * scales[..., None] + biases[..., None]


def _weighted_error(
    weight: np.ndarray,
    reconstructed: np.ndarray,
    hessian_diagonal: np.ndarray,
) -> np.ndarray:
    delta = weight.astype(np.float64) - reconstructed.astype(np.float64)
    hessian = hessian_diagonal.astype(np.float64)
    result = np.zeros(delta.shape[:-1], dtype=np.float64)
    for index in range(delta.shape[-1]):
        result += np.square(delta[..., index]) * hessian[None, :, index]
    return result


def _hessian_diagonal(rows: np.ndarray) -> np.ndarray:
    result = np.zeros(rows.shape[1], dtype=np.float64)
    for row in rows:
        values = row.astype(np.float64)
        result += values * values
    return result / rows.shape[0]


def activation_aware_affine_quantize(
    weight: mx.array,
    activation_rows: np.ndarray,
    *,
    bits: int = 8,
    group_size: int = 32,
    iterations: int = 3,
) -> ActivationAwareResult:
    """Fit native affine codes to diagonal-Hessian layer-output error."""

    if bits != 8 or group_size != 32:
        raise ValueError("the calibrated MiniMax-H3 delivery requires INT8 group_size=32")
    if weight.ndim != 2 or int(weight.shape[-1]) % group_size:
        raise ValueError(f"invalid weight shape for group {group_size}: {weight.shape}")
    rows = np.asarray(activation_rows, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != int(weight.shape[-1]) or not rows.shape[0]:
        raise ValueError(
            f"activation rows {rows.shape} do not match weight input width {weight.shape[-1]}"
        )
    if not np.isfinite(rows).all():
        raise ValueError("calibration activations contain non-finite values")

    native_weight, native_scales_array, native_biases_array = mx.quantize(
        weight,
        group_size=group_size,
        bits=bits,
        mode="affine",
    )
    mx.eval(native_weight, native_scales_array, native_biases_array)
    auxiliary_dtype = native_scales_array.dtype
    dense = np.asarray(weight.astype(mx.float32))
    grouped = dense.reshape(dense.shape[0], -1, group_size)
    levels = (1 << bits) - 1
    hessian = _hessian_diagonal(rows)
    floor = max(float(np.mean(hessian)) * 1e-6, 1e-12)
    hessian = np.maximum(hessian, floor).astype(np.float32).reshape(-1, group_size)

    baseline_codes = (
        np.asarray(native_weight)
        .view(np.uint8)
        .reshape(dense.shape)
        .astype(np.uint16)
        .reshape(grouped.shape)
    )
    baseline_scales = np.asarray(native_scales_array.astype(mx.float32))
    baseline_biases = np.asarray(native_biases_array.astype(mx.float32))
    baseline_reconstructed = _affine_dequantized(
        baseline_codes, baseline_scales, baseline_biases
    )
    baseline_errors = _weighted_error(grouped, baseline_reconstructed, hessian)

    codes = baseline_codes.copy()
    scales = baseline_scales.copy()
    biases = baseline_biases.copy()
    h = hessian[None, :, :]
    sum_h = np.zeros(h.shape[:-1], dtype=np.float64)
    for index in range(group_size):
        sum_h += h[..., index].astype(np.float64)
    for _ in range(iterations):
        q = codes.astype(np.float32)
        sum_hq = np.zeros(q.shape[:-1], dtype=np.float64)
        sum_hqq = np.zeros(q.shape[:-1], dtype=np.float64)
        sum_hw = np.zeros(q.shape[:-1], dtype=np.float64)
        sum_hqw = np.zeros(q.shape[:-1], dtype=np.float64)
        for index in range(group_size):
            hi = h[..., index].astype(np.float64)
            qi = q[..., index].astype(np.float64)
            wi = grouped[..., index].astype(np.float64)
            sum_hq += hi * qi
            sum_hqq += hi * qi * qi
            sum_hw += hi * wi
            sum_hqw += hi * qi * wi
        determinant = sum_hqq * sum_h - sum_hq * sum_hq
        valid = determinant > np.finfo(np.float32).eps
        fitted_scales = np.where(
            valid,
            (sum_hqw * sum_h - sum_hq * sum_hw) / determinant,
            scales,
        )
        fitted_biases = np.where(
            valid,
            (sum_hqq * sum_hw - sum_hq * sum_hqw) / determinant,
            biases,
        )
        fitted_scales = np.where(
            np.abs(fitted_scales) > np.finfo(np.float32).tiny,
            fitted_scales,
            scales,
        )
        scales = _round_to_dtype(fitted_scales, auxiliary_dtype)
        biases = _round_to_dtype(fitted_biases, auxiliary_dtype)
        codes = np.clip(
            np.rint((grouped - biases[..., None]) / scales[..., None]),
            0,
            levels,
        ).astype(np.uint16)

    candidate_reconstructed = _affine_dequantized(codes, scales, biases)
    candidate_errors = _weighted_error(grouped, candidate_reconstructed, hessian)
    improved = candidate_errors < baseline_errors
    codes = np.where(improved[..., None], codes, baseline_codes)
    scales = np.where(improved, scales, baseline_scales)
    biases = np.where(improved, biases, baseline_biases)
    final_reconstructed = _affine_dequantized(codes, scales, biases)
    final_errors = _weighted_error(grouped, final_reconstructed, hessian)

    packed = _pack_uint32(codes.reshape(dense.shape), bits)
    return ActivationAwareResult(
        weight=mx.array(packed, dtype=mx.uint32),
        scales=mx.array(scales).astype(auxiliary_dtype),
        biases=mx.array(biases).astype(auxiliary_dtype),
        baseline_error=float(np.sum(baseline_errors)),
        candidate_error=float(np.sum(final_errors)),
        improved_group_fraction=float(np.mean(improved)),
    )


def diagonal_hessian_relative_error(
    weight: mx.array,
    activation_rows: np.ndarray,
    *,
    bits: int = 8,
    group_size: int = 32,
    quantizer: Callable[..., ActivationAwareResult] = activation_aware_affine_quantize,
) -> float:
    result = quantizer(
        weight,
        activation_rows,
        bits=bits,
        group_size=group_size,
        iterations=0,
    )
    dense = np.asarray(weight.astype(mx.float32))
    rows = np.asarray(activation_rows, dtype=np.float32)
    hessian = np.maximum(_hessian_diagonal(rows), 1e-12)
    denominator = float(np.sum(np.square(dense, dtype=np.float64) * hessian[None, :]))
    return float(np.sqrt(result.baseline_error / max(denominator, 1e-24)))


def quantized_parameter_paths(model: nn.Module, class_predicate) -> list[tuple[str, nn.Linear]]:
    return [
        (path, module)
        for path, module in tree_flatten(
            model.leaf_modules(),
            is_leaf=nn.Module.is_module,
        )
        if isinstance(module, nn.Linear) and class_predicate(path, module)
    ]


def quantize_activation_aware(
    model: nn.Module,
    class_predicate,
    calibration: ActivationDataset,
    bf16_layers: set[str],
    *,
    quantizer: Callable[..., ActivationAwareResult] = activation_aware_affine_quantize,
) -> dict[str, Any]:
    """Replace selected dense linears with calibrated MLX QuantizedLinear modules."""

    metrics: dict[str, dict[str, float | int]] = {}

    def replace(path: str, module: nn.Module):
        params = class_predicate(path, module)
        if not params or path in bf16_layers:
            return module
        rows = calibration.get("calibration", path)
        result = quantizer(
            module.weight,
            rows,
            bits=int(params["bits"]),
            group_size=int(params["group_size"]),
        )
        slot = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
        nn.Module.__init__(slot)
        slot.group_size = int(params["group_size"])
        slot.bits = int(params["bits"])
        slot.mode = str(params["mode"])
        slot.weight = result.weight
        slot.scales = result.scales
        slot.biases = result.biases
        if "bias" in module:
            slot.bias = module.bias
        slot.freeze()
        metrics[path] = {
            "parameters": int(module.weight.size),
            "calibration_rtn_diagonal_hessian_sse": result.baseline_error,
            "calibration_candidate_diagonal_hessian_sse": result.candidate_error,
            "improved_group_fraction": result.improved_group_fraction,
        }
        return slot

    leaves = tree_map_with_path(replace, model.leaf_modules(), is_leaf=nn.Module.is_module)
    model.update_modules(leaves)
    mx.eval(model.parameters())
    return metrics


__all__ = [
    "ActivationDataset",
    "ActivationRecorder",
    "activation_aware_affine_quantize",
    "bind_activation_paths",
    "capture_linear_input",
    "diagonal_hessian_relative_error",
    "project_linear",
    "quantize_activation_aware",
    "quantized_parameter_paths",
]
