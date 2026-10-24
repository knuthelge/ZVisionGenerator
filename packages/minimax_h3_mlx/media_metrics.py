"""Paired final-media metrics for MiniMax-H3 quality comparisons.

The functions in this module compare a candidate clip against a reference clip
produced for the same prompt, seed, resolution, duration, and schedule.  They
are intentionally lightweight: NumPy supplies the metric math and ffmpeg/ffprobe
are used only for decoding existing media files.  Nothing here starts model
generation or downloads weights.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from minimax_h3_mlx.media import _resolve_executable

SCHEMA_VERSION = 1
DEFAULT_AUDIO_SAMPLE_RATE_HZ = 16_000
DEFAULT_AUDIO_SYNC_MAX_LAG_SECONDS = 0.5
DEFAULT_AUDIO_ENVELOPE_FRAME_RATE_HZ = 100.0
DEFAULT_AUDIO_NONZERO_EPSILON = 1e-8
DEFAULT_AUDIO_MIN_RMS = 1e-6
DEFAULT_AUDIO_MIN_PEAK = 1e-5
DEFAULT_AUDIO_MIN_NONZERO_SAMPLES = 16


def _finite_float(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _require_video(video: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(video)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (frames, height, width, 3), got {array.shape}")
    if array.shape[0] <= 0 or array.shape[1] <= 0 or array.shape[2] <= 0:
        raise ValueError(f"{name} must contain at least one non-empty RGB frame, got {array.shape}")
    if not np.isfinite(array.astype(np.float64)).all():
        raise ValueError(f"{name} contains non-finite values")
    return array.astype(np.float64, copy=False)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _global_ssim(x: np.ndarray, y: np.ndarray, *, max_value: float) -> float:
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    c1 = (0.01 * max_value) ** 2
    c2 = (0.03 * max_value) ** 2
    mu_x = float(np.mean(x))
    mu_y = float(np.mean(y))
    var_x = float(np.mean((x - mu_x) ** 2))
    var_y = float(np.mean((y - mu_y) ** 2))
    cov_xy = float(np.mean((x - mu_x) * (y - mu_y)))
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    if denominator == 0.0:
        return 1.0 if np.array_equal(x, y) else 0.0
    return float(((2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)) / denominator)


def _sobel_edges(luma: np.ndarray) -> np.ndarray:
    padded = np.pad(luma, ((0, 0), (1, 1), (1, 1)), mode="edge")
    gx = (
        -padded[:, :-2, :-2]
        + padded[:, :-2, 2:]
        - 2.0 * padded[:, 1:-1, :-2]
        + 2.0 * padded[:, 1:-1, 2:]
        - padded[:, 2:, :-2]
        + padded[:, 2:, 2:]
    )
    gy = (
        padded[:, :-2, :-2]
        + 2.0 * padded[:, :-2, 1:-1]
        + padded[:, :-2, 2:]
        - padded[:, 2:, :-2]
        - 2.0 * padded[:, 2:, 1:-1]
        - padded[:, 2:, 2:]
    )
    return np.sqrt(gx * gx + gy * gy)


def _psnr_payload(prefix: str, mse: float, *, max_value: float) -> dict[str, Any]:
    mse = float(mse)
    if mse <= 0.0:
        return {f"{prefix}_mse": 0.0, f"{prefix}_psnr_db": None, f"{prefix}_psnr_is_infinite": True}
    return {
        f"{prefix}_mse": mse,
        f"{prefix}_psnr_db": _finite_float(20.0 * math.log10(max_value / math.sqrt(mse))),
        f"{prefix}_psnr_is_infinite": False,
    }


def _finite_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {
        "finite_count": int(finite.size),
        "nonfinite_count": int(values.size - finite.size),
        "mean": _finite_float(np.mean(finite)) if finite.size else None,
        "min": _finite_float(np.min(finite)) if finite.size else None,
        "max": _finite_float(np.max(finite)) if finite.size else None,
    }


def video_pair_metrics(
    reference_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    *,
    fps: float = 24.0,
    max_value: float = 255.0,
) -> dict[str, Any]:
    """Compute paired RGB-video metrics for already-decoded clips.

    The inputs must be frame-aligned arrays with shape ``(frames, height, width,
    3)``.  PSNR is reported as ``null`` with an explicit ``*_is_infinite`` flag
    for exact matches so the resulting JSON remains standards-compliant.
    """
    reference = _require_video(reference_rgb, "reference_rgb")
    candidate = _require_video(candidate_rgb, "candidate_rgb")
    if reference.shape != candidate.shape:
        raise ValueError(f"reference/candidate video shapes differ: {reference.shape} vs {candidate.shape}")

    diff = candidate - reference
    frame_mse = np.mean(diff * diff, axis=(1, 2, 3))
    finite_frame_psnr = np.array(
        [20.0 * math.log10(max_value / math.sqrt(float(value))) for value in frame_mse if value > 0.0],
        dtype=np.float64,
    )

    ref_luma = _luma(reference)
    cand_luma = _luma(candidate)
    ssim_values = np.array(
        [_global_ssim(ref_luma[index], cand_luma[index], max_value=max_value) for index in range(reference.shape[0])],
        dtype=np.float64,
    )

    ref_edges = _sobel_edges(ref_luma)
    cand_edges = _sobel_edges(cand_luma)
    edge_mse = float(np.mean((cand_edges - ref_edges) ** 2))

    temporal: dict[str, Any] = {"frame_delta_count": max(int(reference.shape[0] - 1), 0)}
    if reference.shape[0] > 1:
        ref_delta = np.diff(ref_luma, axis=0)
        cand_delta = np.diff(cand_luma, axis=0)
        temporal_mse = float(np.mean((cand_delta - ref_delta) ** 2))
        ref_flicker = np.mean(np.abs(ref_delta), axis=(1, 2))
        cand_flicker = np.mean(np.abs(cand_delta), axis=(1, 2))
        ref_flicker_mean = float(np.mean(ref_flicker))
        cand_flicker_mean = float(np.mean(cand_flicker))
        temporal.update(
            {
                **_psnr_payload("temporal_delta_luma", temporal_mse, max_value=max_value),
                "reference_luma_flicker_mean_abs": _finite_float(ref_flicker_mean),
                "candidate_luma_flicker_mean_abs": _finite_float(cand_flicker_mean),
                "luma_flicker_mean_abs_delta": _finite_float(abs(cand_flicker_mean - ref_flicker_mean)),
                "luma_flicker_ratio_candidate_over_reference": (
                    _finite_float(cand_flicker_mean / ref_flicker_mean) if ref_flicker_mean > 0.0 else None
                ),
            }
        )
    else:
        temporal.update(
            {
                "temporal_delta_luma_mse": None,
                "temporal_delta_luma_psnr_db": None,
                "temporal_delta_luma_psnr_is_infinite": None,
                "reference_luma_flicker_mean_abs": None,
                "candidate_luma_flicker_mean_abs": None,
                "luma_flicker_mean_abs_delta": None,
                "luma_flicker_ratio_candidate_over_reference": None,
            }
        )

    ref_channel_mean = np.mean(reference, axis=(0, 1, 2))
    cand_channel_mean = np.mean(candidate, axis=(0, 1, 2))
    channel_delta = cand_channel_mean - ref_channel_mean

    return {
        "schema_version": SCHEMA_VERSION,
        "metric_family": "paired_video_rgb",
        "frame_count": int(reference.shape[0]),
        "height": int(reference.shape[1]),
        "width": int(reference.shape[2]),
        "fps": _finite_float(fps),
        "pixel": {
            **_psnr_payload("rgb", float(np.mean(diff * diff)), max_value=max_value),
            "rgb_mae": _finite_float(np.mean(np.abs(diff))),
            "rgb_max_abs_error": _finite_float(np.max(np.abs(diff))),
            "per_frame_psnr_db": _finite_summary(finite_frame_psnr),
            "per_frame_psnr_infinite_count": int(np.count_nonzero(frame_mse <= 0.0)),
        },
        "ssim": {
            "luma_global_mean": _finite_float(np.mean(ssim_values)),
            "luma_global_min": _finite_float(np.min(ssim_values)),
            "luma_global_max": _finite_float(np.max(ssim_values)),
        },
        "color": {
            "reference_rgb_mean": [_finite_float(v) for v in ref_channel_mean],
            "candidate_rgb_mean": [_finite_float(v) for v in cand_channel_mean],
            "rgb_mean_delta_candidate_minus_reference": [_finite_float(v) for v in channel_delta],
            "rgb_mean_abs_delta": _finite_float(np.mean(np.abs(channel_delta))),
        },
        "temporal": temporal,
        "perceptual_proxy": {
            **_psnr_payload("edge_luma", edge_mse, max_value=4.0 * max_value),
            "edge_luma_mae": _finite_float(np.mean(np.abs(cand_edges - ref_edges))),
            "note": "Sobel-edge luma error is a lightweight perceptual proxy; it is not LPIPS/FVD.",
        },
    }


def _to_mono_float64(audio: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float64)
    if array.ndim == 1:
        mono = array
    elif array.ndim == 2:
        if array.shape[0] <= 8 and array.shape[1] > array.shape[0]:
            mono = np.mean(array, axis=0)
        elif array.shape[1] <= 8:
            mono = np.mean(array, axis=1)
        else:
            raise ValueError(f"{name} must be mono or have a small channel dimension, got {array.shape}")
    else:
        raise ValueError(f"{name} must be a mono or stereo-like array, got {array.shape}")
    if mono.size == 0:
        raise ValueError(f"{name} contains no audio samples")
    if not np.isfinite(mono).all():
        raise ValueError(f"{name} contains non-finite samples")
    return mono.astype(np.float64, copy=False)


def audio_activity_metrics(
    audio: np.ndarray,
    *,
    sample_rate_hz: int = DEFAULT_AUDIO_SAMPLE_RATE_HZ,
    nonzero_epsilon: float = DEFAULT_AUDIO_NONZERO_EPSILON,
    min_rms: float = DEFAULT_AUDIO_MIN_RMS,
    min_peak: float = DEFAULT_AUDIO_MIN_PEAK,
    min_nonzero_samples: int = DEFAULT_AUDIO_MIN_NONZERO_SAMPLES,
) -> dict[str, Any]:
    """Record finite/non-silent activity metrics for one decoded audio stream."""
    mono = _to_mono_float64(audio, "audio")
    abs_audio = np.abs(mono)
    nonzero_count = int(np.count_nonzero(abs_audio > nonzero_epsilon))
    peak = float(np.max(abs_audio))
    rms = float(math.sqrt(float(np.mean(mono * mono))))
    ok = nonzero_count >= min_nonzero_samples and peak >= min_peak and rms >= min_rms
    return {
        "sample_rate_hz": int(sample_rate_hz),
        "sample_count": int(mono.size),
        "duration_seconds": _finite_float(mono.size / sample_rate_hz),
        "finite_sample_count": int(mono.size),
        "nonzero_sample_count": nonzero_count,
        "max_abs_sample": _finite_float(peak),
        "rms": _finite_float(rms),
        "activity_ok": bool(ok),
        "activity_contract": {
            "nonzero_epsilon": nonzero_epsilon,
            "min_rms": min_rms,
            "min_peak": min_peak,
            "min_nonzero_samples": min_nonzero_samples,
        },
    }


def _activity_envelope(audio: np.ndarray, *, sample_rate_hz: int, frame_rate_hz: float) -> np.ndarray:
    hop = max(1, int(round(sample_rate_hz / frame_rate_hz)))
    usable = (audio.size // hop) * hop
    if usable == 0:
        return np.array([float(np.mean(np.abs(audio)))], dtype=np.float64)
    return np.mean(np.abs(audio[:usable]).reshape(-1, hop), axis=1).astype(np.float64)


def _normalized_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    a0 = a - float(np.mean(a))
    b0 = b - float(np.mean(b))
    denom = float(np.linalg.norm(a0) * np.linalg.norm(b0))
    if denom <= 0.0:
        return None
    return _finite_float(float(np.dot(a0, b0) / denom))


def _activity_sync_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    sample_rate_hz: int,
    max_lag_seconds: float,
    envelope_frame_rate_hz: float,
) -> dict[str, Any]:
    ref_env = _activity_envelope(reference, sample_rate_hz=sample_rate_hz, frame_rate_hz=envelope_frame_rate_hz)
    cand_env = _activity_envelope(candidate, sample_rate_hz=sample_rate_hz, frame_rate_hz=envelope_frame_rate_hz)
    count = min(ref_env.size, cand_env.size)
    ref_env = ref_env[:count]
    cand_env = cand_env[:count]
    max_lag_frames = max(0, int(round(max_lag_seconds * envelope_frame_rate_hz)))
    result: dict[str, Any] = {
        "method": "mean-absolute-audio-envelope cross-correlation",
        "envelope_frame_rate_hz": _finite_float(envelope_frame_rate_hz),
        "envelope_frame_count": int(count),
        "max_lag_seconds": _finite_float(max_lag_seconds),
        "max_lag_frames": int(max_lag_frames),
        "candidate_lag_seconds": None,
        "candidate_lag_frames": None,
        "max_envelope_correlation": None,
        "note": "Positive lag means the candidate audio activity lags the reference.",
    }
    if count == 0:
        result["sync_error"] = "no activity-envelope frames"
        return result
    if float(np.std(ref_env)) <= 0.0 or float(np.std(cand_env)) <= 0.0:
        result["sync_error"] = "one or both audio activity envelopes are constant"
        return result

    ref_zero = ref_env - float(np.mean(ref_env))
    cand_zero = cand_env - float(np.mean(cand_env))
    corr = np.correlate(cand_zero, ref_zero, mode="full")
    lags = np.arange(-count + 1, count, dtype=np.int64)
    window = np.abs(lags) <= max_lag_frames
    if not np.any(window):
        result["sync_error"] = "empty lag window"
        return result
    window_corr = corr[window]
    window_lags = lags[window]
    best_index = int(np.argmax(window_corr))
    best_lag = int(window_lags[best_index])
    denom = float(np.linalg.norm(ref_zero) * np.linalg.norm(cand_zero))
    result.update(
        {
            "candidate_lag_seconds": _finite_float(best_lag / envelope_frame_rate_hz),
            "candidate_lag_frames": best_lag,
            "max_envelope_correlation": _finite_float(float(window_corr[best_index]) / denom) if denom > 0.0 else None,
        }
    )
    return result


def audio_pair_metrics(
    reference_audio: np.ndarray,
    candidate_audio: np.ndarray,
    *,
    sample_rate_hz: int = DEFAULT_AUDIO_SAMPLE_RATE_HZ,
    max_lag_seconds: float = DEFAULT_AUDIO_SYNC_MAX_LAG_SECONDS,
    envelope_frame_rate_hz: float = DEFAULT_AUDIO_ENVELOPE_FRAME_RATE_HZ,
) -> dict[str, Any]:
    """Compute paired waveform/activity/sync metrics for decoded audio."""
    reference = _to_mono_float64(reference_audio, "reference_audio")
    candidate = _to_mono_float64(candidate_audio, "candidate_audio")
    common = min(reference.size, candidate.size)
    if common <= 0:
        raise ValueError("reference/candidate audio have no overlapping samples")
    ref_common = reference[:common]
    cand_common = candidate[:common]
    diff = cand_common - ref_common
    mse = float(np.mean(diff * diff))
    ref_power = float(np.mean(ref_common * ref_common))
    waveform_corr = _normalized_correlation(cand_common, ref_common)
    if mse <= 0.0:
        snr_db = None
        snr_is_infinite = True
    elif ref_power <= 0.0:
        snr_db = None
        snr_is_infinite = False
    else:
        snr_db = _finite_float(10.0 * math.log10(ref_power / mse))
        snr_is_infinite = False

    return {
        "schema_version": SCHEMA_VERSION,
        "metric_family": "paired_audio_mono_f32",
        "sample_rate_hz": int(sample_rate_hz),
        "reference_sample_count": int(reference.size),
        "candidate_sample_count": int(candidate.size),
        "common_sample_count": int(common),
        "duration_seconds_common": _finite_float(common / sample_rate_hz),
        "waveform": {
            "mse": _finite_float(mse),
            "mae": _finite_float(np.mean(np.abs(diff))),
            "max_abs_error": _finite_float(np.max(np.abs(diff))),
            "reference_power": _finite_float(ref_power),
            "snr_db": snr_db,
            "snr_is_infinite": snr_is_infinite,
            "zero_mean_correlation": waveform_corr,
        },
        "activity": {
            "reference": audio_activity_metrics(reference, sample_rate_hz=sample_rate_hz),
            "candidate": audio_activity_metrics(candidate, sample_rate_hz=sample_rate_hz),
        },
        "sync": _activity_sync_metrics(
            ref_common,
            cand_common,
            sample_rate_hz=sample_rate_hz,
            max_lag_seconds=max_lag_seconds,
            envelope_frame_rate_hz=envelope_frame_rate_hz,
        ),
    }


def probe_media(path: str | Path, *, ffprobe: str | Path | None = None) -> dict[str, Any]:
    """Return ffprobe JSON for an existing media file."""
    ffprobe_path = _resolve_executable("ffprobe", ffprobe, env_var="MINIMAX_H3_FFPROBE")
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {process.stderr[:500]}")
    payload = json.loads(process.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError(f"ffprobe returned a non-object payload for {path}")
    return payload


def _first_stream(info: dict[str, Any], codec_type: str) -> dict[str, Any]:
    for stream in info.get("streams", []):
        if stream.get("codec_type") == codec_type:
            return stream
    raise RuntimeError(f"ffprobe found no {codec_type} stream")


def _stream_fps(stream: dict[str, Any]) -> float | None:
    raw = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    if not isinstance(raw, str) or "/" not in raw:
        return None
    num_s, den_s = raw.split("/", 1)
    try:
        num = float(num_s)
        den = float(den_s)
    except ValueError:
        return None
    if den == 0.0:
        return None
    return _finite_float(num / den)


def decode_video_rgb(
    path: str | Path,
    *,
    ffmpeg: str | Path | None = None,
    ffprobe: str | Path | None = None,
    max_frames: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode an existing video stream to ``uint8`` RGB frames."""
    path = Path(path)
    info = probe_media(path, ffprobe=ffprobe)
    video_stream = _first_stream(info, "video")
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    ffmpeg_path = _resolve_executable("ffmpeg", ffmpeg, env_var="MINIMAX_H3_FFMPEG")
    cmd = [ffmpeg_path, "-hide_banner", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:v:0"]
    if max_frames is not None:
        cmd += ["-frames:v", str(int(max_frames))]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    process = subprocess.run(cmd, capture_output=True)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg video decode failed for {path}: {process.stderr.decode(errors='replace')[:500]}")
    frame_size = width * height * 3
    if frame_size <= 0 or len(process.stdout) % frame_size:
        raise RuntimeError(
            f"decoded video byte count {len(process.stdout)} is not divisible by frame size {frame_size}"
        )
    frames = np.frombuffer(process.stdout, dtype=np.uint8).reshape((-1, height, width, 3)).copy()
    if frames.shape[0] == 0:
        raise RuntimeError(f"ffmpeg decoded no video frames from {path}")
    metadata = {
        "path": str(path),
        "width": width,
        "height": height,
        "decoded_frame_count": int(frames.shape[0]),
        "fps": _stream_fps(video_stream),
        "codec_name": video_stream.get("codec_name"),
    }
    return frames, metadata


def decode_audio_mono_f32(
    path: str | Path,
    *,
    ffmpeg: str | Path | None = None,
    sample_rate_hz: int = DEFAULT_AUDIO_SAMPLE_RATE_HZ,
) -> np.ndarray:
    """Decode the first audio stream to mono little-endian f32 samples."""
    ffmpeg_path = _resolve_executable("ffmpeg", ffmpeg, env_var="MINIMAX_H3_FFMPEG")
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate_hz)),
        "-f",
        "f32le",
        "pipe:1",
    ]
    process = subprocess.run(cmd, capture_output=True)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg audio decode failed for {path}: {process.stderr.decode(errors='replace')[:500]}")
    if len(process.stdout) == 0:
        raise RuntimeError(f"ffmpeg decoded no audio samples from {path}")
    if len(process.stdout) % 4:
        raise RuntimeError(f"decoded f32le byte count is not divisible by 4 for {path}")
    return np.frombuffer(process.stdout, dtype="<f4").astype(np.float64, copy=True)


def paired_media_file_metrics(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    ffmpeg: str | Path | None = None,
    ffprobe: str | Path | None = None,
    sample_rate_hz: int = DEFAULT_AUDIO_SAMPLE_RATE_HZ,
    max_audio_lag_seconds: float = DEFAULT_AUDIO_SYNC_MAX_LAG_SECONDS,
    max_video_frames: int | None = None,
    reference_label: str = "reference",
    candidate_label: str = "candidate",
) -> dict[str, Any]:
    """Decode two existing media files and compute paired video/audio metrics."""
    reference_video, reference_video_meta = decode_video_rgb(
        reference_path, ffmpeg=ffmpeg, ffprobe=ffprobe, max_frames=max_video_frames
    )
    candidate_video, candidate_video_meta = decode_video_rgb(
        candidate_path, ffmpeg=ffmpeg, ffprobe=ffprobe, max_frames=max_video_frames
    )
    reference_audio = decode_audio_mono_f32(reference_path, ffmpeg=ffmpeg, sample_rate_hz=sample_rate_hz)
    candidate_audio = decode_audio_mono_f32(candidate_path, ffmpeg=ffmpeg, sample_rate_hz=sample_rate_hz)
    fps = reference_video_meta.get("fps") or candidate_video_meta.get("fps") or 24.0
    return {
        "schema_version": SCHEMA_VERSION,
        "metric_family": "paired_media_file",
        "pair": {
            "reference_path": str(reference_path),
            "candidate_path": str(candidate_path),
            "reference_label": reference_label,
            "candidate_label": candidate_label,
            "pairing_requirement": "same prompt, seed, resolution, duration, and decoding route; record each condition schedule separately",
        },
        "decode": {
            "reference_video": reference_video_meta,
            "candidate_video": candidate_video_meta,
            "audio_sample_rate_hz": int(sample_rate_hz),
            "max_video_frames": max_video_frames,
        },
        "video": video_pair_metrics(reference_video, candidate_video, fps=float(fps)),
        "audio": audio_pair_metrics(
            reference_audio,
            candidate_audio,
            sample_rate_hz=sample_rate_hz,
            max_lag_seconds=max_audio_lag_seconds,
        ),
    }


def write_metrics_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a strict JSON metrics payload."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path
