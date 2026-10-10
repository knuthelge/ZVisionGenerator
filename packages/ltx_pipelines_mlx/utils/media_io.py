"""Image / video / audio I/O — mirrors upstream ``ltx_pipelines.utils.media_io``.

Ports the public API surface of upstream's ``media_io.py`` so pipelines can
import the same names from the same path. The implementation uses ffmpeg
subprocess pipes (instead of upstream's PyAV) — same I/O behavior, no extra
runtime dependency.

Public names match upstream verbatim:

- ``DEFAULT_IMAGE_CRF`` — H.264 CRF of the pre-2.4 model generations (re-exported
  from :mod:`ltx_pipelines_mlx.utils.constants`, with ``LTX_2_4_IMAGE_CRF``).
- ``decode_image`` — load an image file as an oriented sRGB ``numpy.ndarray`` (HWC, uint8).
- ``encode_single_frame`` — encode one RGB frame to H.264 mp4 bytes.
- ``decode_single_frame`` — decode the first frame of a buffer back to RGB.
- ``preprocess`` — round-trip an image through libx264 (4:2:0) at a given CRF.
- ``resize_and_center_crop`` — aspect-preserving resize + center crop.
- ``to_vae_range`` / ``from_vae_range`` — ``[0, 1] ↔ [-1, 1]`` shifts.
- ``load_image_and_preprocess`` — full I2V image pipeline (decode → CRF
  round-trip → resize/crop → normalize → MLX tensor).

The legacy alias ``prepare_image_for_encoding`` (kept in
:mod:`ltx_core_mlx.utils.image`) now delegates here so call sites that
haven't migrated to the upstream-named API keep working.
"""

from __future__ import annotations

import logging
import math
import subprocess
from collections.abc import Iterable, Iterator
from io import BytesIO
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import ExifTags, Image, ImageCms, UnidentifiedImageError

from ltx_core_mlx.utils.ffmpeg import find_ffmpeg, probe_video_info

# Re-exported for upstream-iso import paths. ``DEFAULT_IMAGE_CRF`` is the CRF of the
# pre-2.4 model generations (``LTX_2_4_IMAGE_CRF`` from 2.4 on); the value a run uses
# comes from the checkpoint (``ImageConditioner.resolve_crf``), never from a default here.
# Round-tripping the input image through libx264 brings it close to the LTX-2 training
# distribution (real video frames carrying H.264 compression artefacts).
from ltx_pipelines_mlx.utils.constants import DEFAULT_IMAGE_CRF, LTX_2_4_IMAGE_CRF

logger = logging.getLogger(__name__)


def to_vae_range(x: mx.array) -> mx.array:
    """Shift ``[0, 1]`` pixels to the VAE's expected ``[-1, 1]`` range."""
    return x * 2.0 - 1.0


def from_vae_range(z: mx.array) -> mx.array:
    """Inverse of :func:`to_vae_range`: ``[-1, 1]`` → ``[0, 1]``."""
    return (z + 1.0) / 2.0


_ORIENTATION_EXIF_KEY = next(key for key, value in ExifTags.TAGS.items() if value == "Orientation")

_ORIENTATION_TO_ROTATION = {3: 180, 6: 270, 8: 90}

_SRGB_PROFILE = ImageCms.createProfile("sRGB")


def decode_image(image_path: str) -> np.ndarray:
    """Load an image as an oriented, sRGB, three-channel uint8 RGB array.

    Mirrors upstream ``decode_image``: the EXIF orientation is applied, an
    embedded ICC profile is converted to sRGB (falling back to a plain RGB
    conversion with a warning if the profile cannot be used), and the result
    is ``HWC, uint8``.

    Raises:
        ValueError: If the file cannot be decoded as an image.
    """
    try:
        with Image.open(image_path) as source_image:
            image = source_image
            orientation = image.getexif().get(_ORIENTATION_EXIF_KEY)
            if orientation in _ORIENTATION_TO_ROTATION:
                image = image.rotate(_ORIENTATION_TO_ROTATION[orientation], expand=True)

            icc_profile = image.info.get("icc_profile")

            if image.mode == "RGBA":
                image = image.convert("RGB")
            elif image.mode == "LA":
                image = image.convert("L")

            if icc_profile:
                try:
                    source_profile = ImageCms.ImageCmsProfile(BytesIO(icc_profile))
                    destination_profile = ImageCms.ImageCmsProfile(_SRGB_PROFILE)
                    image = ImageCms.profileToProfile(
                        image,
                        source_profile,
                        destination_profile,
                        outputMode="RGB",
                    )
                except (ImageCms.PyCMSError, OSError, ValueError) as error:
                    logger.warning("Failed to convert image to sRGB: %s", error)
                    image = image.convert("RGB")
            else:
                image = image.convert("RGB")

            return np.array(image, dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as err:
        raise ValueError(f"Cannot decode image file '{image_path}'.") from err


def encode_single_frame(
    output_file: BytesIO | str,
    image_array: np.ndarray,
    crf: float,
) -> None:
    """Encode a single RGB frame to a 1-frame H.264 mp4.

    Mirrors upstream's PyAV-based implementation with an ffmpeg subprocess
    pipeline. Output goes to ``output_file`` (``BytesIO`` for in-memory or a
    path string for disk). Like upstream, odd dimensions are cropped down to
    the nearest even size (the decoded frame comes back cropped), and the frame
    is encoded as 4:2:0 (``yuv420p``), the chroma subsampling of the video the
    model was trained on. The encoder settings match what PyAV hands libx264:
    bilinear RGB→YUV conversion (PyAV's ``reformat`` default) and slice
    threading (PyAV's default thread type). The bitstream is not bit-identical
    to upstream's: PyAV ships its own libx264 build, and the slice count follows
    the machine's core count on both sides.

    Args:
        output_file: Destination — either a ``BytesIO`` (preferred, in-memory)
            or a filesystem path string.
        image_array: ``HxWx3`` uint8 array.
        crf: H.264 CRF (0 = lossless, higher = more compression). Upstream
            default is 33.
    """
    if image_array.dtype != np.uint8:
        image_array = image_array.astype(np.uint8)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError(f"encode_single_frame expects HxWx3 RGB, got {image_array.shape}")

    # Round down to a multiple of 2 for the 4:2:0 codec, as upstream does.
    height = image_array.shape[0] // 2 * 2
    width = image_array.shape[1] // 2 * 2
    image_array = np.ascontiguousarray(image_array[:height, :width])

    raw = image_array.tobytes()
    ffmpeg = find_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        "1",
        "-i",
        "pipe:0",
        "-sws_flags",
        "bilinear",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(int(crf)),
        "-thread_type",
        "slice",
        "-frames:v",
        "1",
    ]

    if isinstance(output_file, BytesIO):
        cmd += ["-f", "mp4", "-movflags", "frag_keyframe+empty_moov", "pipe:1"]
        proc = subprocess.run(cmd, input=raw, capture_output=True, timeout=60)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg encode_single_frame failed: {proc.stderr.decode(errors='ignore')}")
        output_file.write(proc.stdout)
    else:
        cmd += [str(output_file)]
        proc = subprocess.run(cmd, input=raw, capture_output=True, timeout=60)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg encode_single_frame failed: {proc.stderr.decode(errors='ignore')}")


def decode_single_frame(video_file: BytesIO | str) -> np.ndarray:
    """Decode the first frame of a video buffer/file back to ``HxWx3`` RGB.

    Companion to :func:`encode_single_frame`. The YUV→RGB conversion is
    bilinear, like PyAV's ``to_ndarray(format="rgb24")`` upstream.
    """
    ffmpeg = find_ffmpeg()
    if isinstance(video_file, BytesIO):
        in_data = video_file.getvalue()
        in_arg = "pipe:0"
    else:
        in_data = None
        in_arg = str(video_file)

    # Probe size by asking ffmpeg for raw RGB; we read frame data then deduce shape.
    # For pipe input we need to ask ffprobe-equivalent first; simpler: do a probe.
    if in_data is not None:
        # Probe via ffmpeg -i (writes metadata to stderr). Workaround: use
        # ffprobe through a temp file would be cleaner; here we trust the
        # caller and infer dimensions from the encoded raw buffer length
        # after a first decode pass.
        cmd_probe = [ffmpeg, "-i", in_arg, "-f", "null", "-"]
        probe = subprocess.run(cmd_probe, input=in_data, capture_output=True, timeout=30)
        # Parse "Stream ... NxM" from stderr.
        size = _parse_size_from_stderr(probe.stderr.decode(errors="ignore"))
    else:
        cmd_probe = [ffmpeg, "-i", in_arg, "-f", "null", "-"]
        probe = subprocess.run(cmd_probe, capture_output=True, timeout=30)
        size = _parse_size_from_stderr(probe.stderr.decode(errors="ignore"))

    if size is None:
        raise RuntimeError("decode_single_frame: could not determine frame size from ffmpeg probe")
    width, height = size

    cmd = [
        ffmpeg,
        "-i",
        in_arg,
        "-frames:v",
        "1",
        "-sws_flags",
        "bilinear",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, input=in_data, capture_output=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode_single_frame failed: {proc.stderr.decode(errors='ignore')}")

    arr = np.frombuffer(proc.stdout, dtype=np.uint8)
    return arr.reshape(height, width, 3).copy()


def _parse_size_from_stderr(stderr: str) -> tuple[int, int] | None:
    """Extract (width, height) from ffmpeg's ``-i ...`` stderr summary."""
    import re

    # e.g. "Stream #0:0(und): Video: h264 ..., yuv420p, 1280x720, ..."
    m = re.search(r",\s*(\d{2,5})x(\d{2,5})", stderr)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def preprocess(image: np.ndarray, crf: int | None) -> np.ndarray:
    """Round-trip an ``HxWx3`` uint8 RGB array through libx264 at the given CRF.

    Mirrors upstream verbatim: encode → decode → return decoded RGB.
    ``crf == 0`` is a passthrough. ``crf`` is required: the correct value is a
    property of the model generation, so a code-level default here would
    silently condition on the wrong compression. As upstream, an odd height or
    width comes back cropped to the nearest even size.

    Raises:
        ValueError: If ``crf`` is ``None`` — a conditioning skipped resolution
            (see ``ImageConditioner.resolve_crf``).
    """
    if crf is None:
        raise ValueError(
            "Image conditioning CRF is unresolved (crf=None). Resolve it against the checkpoint "
            "first -- ImageConditioner.resolve_crf(images), or detect_params(checkpoint_path).default_image_crf."
        )
    if crf == 0:
        return image
    if min(image.shape[0], image.shape[1]) < 2:
        return image

    with BytesIO() as buf:
        encode_single_frame(buf, image, crf)
        encoded_bytes = buf.getvalue()
    return decode_single_frame(BytesIO(encoded_bytes))


def _bilinear_resize_axis(x: np.ndarray, out_size: int, axis: int) -> np.ndarray:
    """Resize one axis like ``torch.nn.functional.interpolate(mode="bilinear", align_corners=False)``.

    Half-pixel centres, source coordinates clamped at 0, no antialiasing on
    downscale — the PyTorch semantics upstream relies on.
    """
    in_size = x.shape[axis]
    scale = np.float32(in_size / out_size)
    src = (np.arange(out_size, dtype=np.float32) + np.float32(0.5)) * scale - np.float32(0.5)
    src = np.maximum(src, np.float32(0.0))
    i0 = np.minimum(np.floor(src).astype(np.int64), in_size - 1)
    i1 = np.minimum(i0 + 1, in_size - 1)
    lam = (src - i0).astype(np.float32)
    shape = [1] * x.ndim
    shape[axis] = out_size
    lam = lam.reshape(shape)
    return np.take(x, i0, axis=axis) * (np.float32(1.0) - lam) + np.take(x, i1, axis=axis) * lam


def resize_and_center_crop(
    image: Image.Image | np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Aspect-preserving resize (filling the target), then center crop to ``(height, width)``.

    Mirrors upstream ``resize_and_center_crop``: the frame is resized on float
    values with PyTorch-style bilinear interpolation (``align_corners=False``,
    no antialiasing), with the resized size rounded up, and then center cropped.
    The values are not re-quantized to uint8.

    Args:
        image: ``HxWx3`` RGB array (any numeric dtype, values in ``[0, 255]``)
            or a PIL image.
        height: Target height.
        width: Target width.

    Returns:
        ``(height, width, 3)`` float32 array in ``[0, 255]``.
    """
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    image = np.asarray(image, dtype=np.float32)
    src_h, src_w = image.shape[:2]
    scale = max(height / src_h, width / src_w)
    # Ceil so float rounding never leaves the resized size below the target (negative crop offsets).
    new_h = math.ceil(src_h * scale)
    new_w = math.ceil(src_w * scale)
    resized = _bilinear_resize_axis(_bilinear_resize_axis(image, new_h, axis=0), new_w, axis=1)
    crop_top = (new_h - height) // 2
    crop_left = (new_w - width) // 2
    return resized[crop_top : crop_top + height, crop_left : crop_left + width]


def load_image_and_preprocess(
    image_path: str | Path,
    height: int,
    width: int,
    crf: int | None,
) -> mx.array:
    """Full I2V image pipeline (upstream-iso).

    Pipeline (upstream verbatim):
        1. :func:`decode_image` — load PNG/JPEG → oriented sRGB HxWx3 uint8.
        2. :func:`preprocess` — H.264 4:2:0 round-trip at ``crf``.
        3. :func:`resize_and_center_crop` — bilinear fit to target H/W, on floats.
        4. ``x / 127.5 - 1`` (upstream ``normalize_images``) + HWC→BCHW + bfloat16.

    Mirrors upstream's ``load_image_and_preprocess`` signature; the upstream
    ``dtype`` / ``device`` arguments are dropped (MLX uses bfloat16 + unified
    memory, no device choice). ``crf`` is required, as upstream: ``None``
    raises in :func:`preprocess`.

    Returns:
        ``mx.array`` of shape ``(1, 3, H, W)`` in ``[-1, 1]``, bfloat16.
    """
    if isinstance(image_path, Path):
        image_path = str(image_path)
    arr = decode_image(image_path)
    arr = preprocess(arr, crf=crf)
    image = resize_and_center_crop(arr, height, width)

    # [0, 255] float32 → [-1, 1] (upstream ``normalize_images``)
    f = image / np.float32(127.5) - np.float32(1.0)
    # HWC → CHW → BCHW
    tensor = mx.array(f).transpose(2, 0, 1)[None, ...]
    return tensor.astype(mx.bfloat16)


def decode_video_by_frame(
    path: str | Path,
    starting_frame: int = 0,
    frame_cap: int | None = None,
) -> Iterator[np.ndarray]:
    """Decode a video by sequential frame index, at its native size.

    Mirrors upstream ``decode_video_by_frame`` (PyAV): frames come out in
    decode order, the first ``starting_frame`` are skipped, at most
    ``frame_cap`` are yielded, and nothing is resized or rotated (PyAV does not
    apply the display-matrix rotation, hence ``-noautorotate``). The YUV→RGB
    conversion is bilinear, like PyAV's ``to_rgb()``.

    Args:
        path: Path to the video file.
        starting_frame: Number of leading frames to skip.
        frame_cap: Maximum number of frames to yield (``None`` = all).

    Yields:
        ``(H, W, 3)`` uint8 RGB frames.

    Raises:
        RuntimeError: If ffmpeg cannot decode the file.
    """
    info = probe_video_info(str(path))
    frame_bytes = info.width * info.height * 3
    cmd = [find_ffmpeg(), "-v", "error", "-noautorotate", "-i", str(path)]
    if starting_frame > 0:
        cmd += ["-vf", f"select=gte(n\\,{starting_frame})"]
    if frame_cap is not None:
        cmd += ["-frames:v", str(frame_cap)]
    cmd += ["-fps_mode", "passthrough", "-sws_flags", "bilinear", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(info.height, info.width, 3)
    finally:
        proc.stdout.close()
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode {path}: {stderr.decode(errors='ignore')}")


def video_preprocess(frames: Iterable[np.ndarray], height: int, width: int) -> mx.array:
    """Resize, center crop and normalize video frames for conditioning.

    Mirrors upstream ``video_preprocess``: every frame goes through
    :func:`resize_and_center_crop` (bilinear on floats, aspect-preserving fill)
    and ``x / 127.5 - 1``.

    Args:
        frames: ``(H, W, 3)`` uint8 RGB frames, e.g. from :func:`decode_video_by_frame`.
        height: Target height in pixels.
        width: Target width in pixels.

    Returns:
        ``mx.array`` of shape ``(1, 3, F, height, width)`` in ``[-1, 1]``, bfloat16.

    Raises:
        ValueError: If ``frames`` is empty.
    """
    processed = [resize_and_center_crop(frame, height, width) / np.float32(127.5) - np.float32(1.0) for frame in frames]
    if not processed:
        raise ValueError("video_preprocess received an empty frame generator; no frames were decoded from the source.")
    video = np.stack(processed)  # (F, H, W, 3)
    return mx.array(video).transpose(3, 0, 1, 2)[None].astype(mx.bfloat16)


__all__ = [
    "DEFAULT_IMAGE_CRF",
    "LTX_2_4_IMAGE_CRF",
    "decode_image",
    "decode_single_frame",
    "decode_video_by_frame",
    "encode_single_frame",
    "from_vae_range",
    "load_image_and_preprocess",
    "preprocess",
    "resize_and_center_crop",
    "to_vae_range",
    "video_preprocess",
]
