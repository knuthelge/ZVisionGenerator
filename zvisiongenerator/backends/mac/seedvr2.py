"""macOS SeedVR2 upscaler using mflux and MLX."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from zvisiongenerator.backends.mac import memory as memory_mac

# SeedVR2 model name → the ``ModelConfig`` factory that describes its layout. mflux does not infer the
# variant from the weights, so the config is always passed explicitly.
_CONFIG_FACTORIES = {"seedvr2-3b": "seedvr2_3b", "seedvr2-7b": "seedvr2_7b"}


def _use_bfloat16_precision() -> None:
    """Set mflux's process-wide model precision to bfloat16, which an image load may have changed."""
    import mlx.core as mx
    from mflux.models.common.config.model_config import ModelConfig

    ModelConfig.precision = mx.bfloat16


class MfluxSeedVR2Upscaler:
    """mflux/MLX upscaler for macOS — implements the SeedVR2Upscaler Protocol."""

    def load_model(self, model: str, *, quantize: int | None, model_path: str) -> Any:
        """Load the SeedVR2 variant *model* from *model_path* and keep its weights resident.

        Args:
            model: "seedvr2-3b" or "seedvr2-7b".
            quantize: Quantization bits to apply while loading, or ``None`` for weights already quantized.
            model_path: Directory or Hugging Face repo that holds the weights.

        Returns:
            The loaded mflux model.

        Raises:
            ValueError: If *model* is not a known SeedVR2 variant.
        """
        factory = _CONFIG_FACTORIES.get(model)
        if factory is None:
            raise ValueError(f"Unknown SeedVR2 model '{model}'. Supported: {', '.join(_CONFIG_FACTORIES)}")
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.seedvr2 import SeedVR2

        _use_bfloat16_precision()
        loaded = SeedVR2(quantize=quantize, model_path=model_path, model_config=getattr(ModelConfig, factory)())
        memory_mac.materialize_weights(loaded)
        memory_mac.apply_buffer_cache_policy()
        return loaded

    def save_model(self, model: Any, path: Path) -> None:
        """Write the loaded model's transformer and VAE weights to *path*."""
        model.save_model(str(path))

    def upscale_image(self, model: Any, image_path: str, *, width: int, height: int, seed: int, softness: float) -> Image.Image:
        """Return *image_path* enlarged to *width* × *height*.

        mflux sizes the output from the shorter side, so the target's shorter side is passed as the resolution.
        """
        _use_bfloat16_precision()
        return model.generate_image(seed=seed, image_path=image_path, resolution=min(width, height), softness=softness).image
