"""
faster-qwen3-tts: Real-time Qwen3-TTS inference with CUDA graphs or GGML
"""
from .model import FasterQwen3TTS
from .ggml_backend import GGMLQwen3TTS

__version__ = "0.5.4"
__all__ = ["FasterQwen3TTS", "GGMLQwen3TTS"]
