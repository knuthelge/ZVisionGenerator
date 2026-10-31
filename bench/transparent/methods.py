"""The matte methods under test. Each takes an RGB float array and returns ``(alpha, foreground)``.

Model-backed methods load lazily and are cached for the run: BiRefNet and BEN2 run as ONNX through rembg, the
BiRefNet matting checkpoints as community ONNX exports through onnxruntime directly, ViTMatte through Transformers, and foreground estimation through pymatting. They run in the bench's own environment (see
``bench/transparent/__main__.py``), not the app's.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from pathlib import Path

import numpy as np

from bench.transparent.alpha import chroma_key, ink_key, screen_key, trimap

Matte = Callable[[np.ndarray, np.ndarray | None], tuple[np.ndarray, np.ndarray]]
_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
_BEN2_REPO, _BEN2_FILE = "PramaLLC/BEN2", "BEN2_Base.onnx"
_VITMATTE_REPO = "hustvl/vitmatte-small-composition-1k"
_TRIMAP_RADIUS = 12
_VITMATTE_MAX = 1024
# HR-matting at 2048 px asks onnxruntime for more than a 10 GB card holds, so the bench runs it on the CPU.
_CPU_MATTES = frozenset({"birefnet-hr-matting"})
_IMAGENET_MEAN, _IMAGENET_STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
# name: (repo, file, input size, output is logits). Both are MIT-licensed exports of BiRefNet's matting checkpoints.
_ONNX_MATTES = {
    "birefnet-matting": ("emrikol/birefnet-matting-onnx", "birefnet-matting.onnx", 1024, True),
    "birefnet-hr-matting": ("PinkPixel/birefnet-hr-matting-onnx", "model_fp16.onnx", 2048, False),
}


@cache
def _rembg_session(name: str):  # noqa: ANN202
    import onnxruntime
    from rembg import new_session

    # onnxruntime-gpu finds the CUDA and cuDNN libraries of the nvidia-* wheels only when told to load them first.
    onnxruntime.preload_dlls(directory="")
    session = new_session("ben_custom", model_path=str(_ben2_path()), providers=_PROVIDERS) if name == "ben2" else new_session(name, providers=_PROVIDERS)
    if session.inner_session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"{name} fell back to the CPU; the bench times would be meaningless.")
    return session


def _ben2_path() -> Path:
    """Return BEN2's ONNX file inside rembg's models folder, the only place rembg loads a custom model from."""
    import shutil

    from huggingface_hub import hf_hub_download
    from rembg.sessions.ben_custom import BenCustomSession

    target = Path(BenCustomSession.rembg_home()) / "models" / "ben2" / _BEN2_FILE
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(hf_hub_download(_BEN2_REPO, _BEN2_FILE), target)
    return target


def _mask(session_name: str, rgb: np.ndarray) -> np.ndarray:
    from PIL import Image

    if session_name in _ONNX_MATTES:
        return _onnx_mask(session_name, rgb)
    image = Image.fromarray((rgb * 255).round().astype(np.uint8))
    mask = _rembg_session(session_name).predict(image)[0]
    return np.asarray(mask, dtype=np.float32) / 255.0


@cache
def _onnx_session(name: str):  # noqa: ANN202
    import onnxruntime
    from huggingface_hub import hf_hub_download

    repo, file, _, _ = _ONNX_MATTES[name]
    onnxruntime.preload_dlls(directory="")
    session = onnxruntime.InferenceSession(hf_hub_download(repo, file), providers=_PROVIDERS)
    if session.get_providers()[0] != "CUDAExecutionProvider" and name not in _CPU_MATTES:
        raise RuntimeError(f"{name} fell back to the CPU; the bench times would be meaningless.")
    return session


def _onnx_mask(name: str, rgb: np.ndarray) -> np.ndarray:
    from PIL import Image

    _, _, size, logits = _ONNX_MATTES[name]
    session = _onnx_session(name)
    image = Image.fromarray((rgb * 255).round().astype(np.uint8)).resize((size, size), Image.Resampling.BILINEAR)
    tensor = ((np.asarray(image, np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1)[None]
    out = session.run(None, {session.get_inputs()[0].name: tensor})[0][0, 0].astype(np.float32)
    alpha = 1.0 / (1.0 + np.exp(-out)) if logits else out
    height, width = rgb.shape[:2]
    resized = Image.fromarray(np.clip(alpha, 0.0, 1.0)).resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


@cache
def _vitmatte():  # noqa: ANN202
    import torch
    from transformers import VitMatteForImageMatting, VitMatteImageProcessor

    model = VitMatteForImageMatting.from_pretrained(_VITMATTE_REPO).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return VitMatteImageProcessor.from_pretrained(_VITMATTE_REPO), model


def _refine(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import torch

    import cv2

    processor, model = _vitmatte()
    tri = trimap(mask, radius=_TRIMAP_RADIUS)
    height, width = mask.shape
    # ViTMatte's global attention needs about 6 GB at 2048 px, so it runs at most at 1024 and its alpha is scaled back.
    scale = min(1.0, _VITMATTE_MAX / max(height, width))
    small = (round(width * scale), round(height * scale))
    small_rgb = cv2.resize(rgb, small, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    small_tri = cv2.resize(tri, small, interpolation=cv2.INTER_NEAREST) if scale < 1 else tri
    inputs = processor(images=small_rgb, trimaps=small_tri, do_rescale=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        alpha = model(**inputs).alphas[0, 0].float().cpu().numpy()[: small[1], : small[0]]
    alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR) if scale < 1 else alpha
    alpha = np.where(tri == 0.5, alpha, tri)
    return np.clip(alpha, 0.0, 1.0)


def foreground_for(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Return the subject colours under *alpha*, with the background's colour taken out of the edges (pymatting)."""
    return _foreground(rgb, alpha)


def _foreground(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    from pymatting import estimate_foreground_ml

    return np.clip(estimate_foreground_ml(rgb.astype(np.float64), alpha.astype(np.float64)), 0.0, 1.0).astype(np.float32)


def _model(session_name: str, *, refine: bool = False, decontaminate: bool = False) -> Matte:
    def run(rgb: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        alpha = _mask(session_name, rgb) if mask is None else mask
        if refine:
            alpha = _refine(rgb, alpha)
        return alpha, _foreground(rgb, alpha) if decontaminate else rgb

    return run


METHODS: dict[str, Matte] = {
    "birefnet": _model("birefnet-general"),
    "birefnet-massive": _model("birefnet-massive"),
    "ben2": _model("ben2"),
    "birefnet+fg": _model("birefnet-general", decontaminate=True),
    "ben2+fg": _model("ben2", decontaminate=True),
    "birefnet+vitmatte+fg": _model("birefnet-general", refine=True, decontaminate=True),
    "birefnet-matting+fg": _model("birefnet-matting", decontaminate=True),
    "birefnet-hr-matting+fg": _model("birefnet-hr-matting", decontaminate=True),
    "chroma": lambda rgb, mask=None: chroma_key(rgb),
    "ink": lambda rgb, mask=None: ink_key(rgb),
    "ink-solid": lambda rgb, mask=None: ink_key(rgb, keep_enclosed=True),
    "screen": lambda rgb, mask=None: screen_key(rgb),
}
"""Every method by name, called with the render and an optional precomputed mask to start from.

``chroma`` needs a key-colour render, ``ink`` a white one and ``screen`` a black one; ``+fg`` re-estimates edge colours.
"""

MASK_SOURCE = {"birefnet+vitmatte+fg": "birefnet"}
"""Methods that start from another method's saved mask: BiRefNet and ViTMatte don't fit on a 10 GB card together."""

KEY_METHODS = ("chroma", "ink", "ink-solid", "screen")
CPU_METHODS = ("birefnet-hr-matting+fg",)
"""Methods too large for a 10 GB card under onnxruntime; run them on the CPU with ``--methods`` (no GPU queue)."""
MODEL_METHODS = tuple(name for name in METHODS if name not in KEY_METHODS + CPU_METHODS)


def release_models() -> None:
    """Drop every loaded model and return its GPU memory, so the next method has the card to itself."""
    import gc

    _rembg_session.cache_clear()
    _onnx_session.cache_clear()
    _vitmatte.cache_clear()
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass
