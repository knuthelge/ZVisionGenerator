"""Make ``qwen_tts`` load on the transformers the app locks, without editing the library.

``qwen-tts-hf`` is written for transformers 5.15; on 5.20 and later three things break, each shown by leaving its part out:

1. ``MimiModel.upsample`` and ``.decoder_transformer`` are read-only properties, and the speech tokenizer's encoder assigns them.
2. The encoder is custom code, so transformers skips the built-in ``MimiModel`` weight renames and the encoder would load untrained.
3. Importing ``qwen_tts`` adds a ``"default"`` entry to transformers' global ``ROPE_INIT_FUNCTIONS`` that reads ``config.rope_theta``;
   the configs keep it inside ``rope_parameters``, so every default-RoPE model loaded afterwards in the process fails, the prompt
   enhancer's included.

The shim is applied once per process, after ``qwen_tts`` is imported and before the first ``from_pretrained``. It relies on a
private name of ``qwen_tts``, which is safe because ``qwen_tts`` is vendored at a pinned commit; the contract test fails if it moves.
"""

from __future__ import annotations

import sys

from zvisiongenerator.utils.library_noise import quiet_qwen_import

__all__ = ["apply_qwen_shim"]

_applied = False

_ROPE_MODULES = ("qwen_tts.core.models.modeling_qwen3_tts", "qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2")
"""The modules of ``qwen_tts`` that look up ``ROPE_INIT_FUNCTIONS``, each given a private copy that still has ``"default"``."""


def apply_qwen_shim() -> None:
    """Patch ``qwen_tts`` and transformers for the locked transformers version; do nothing when already applied.

    The checkpoint-conversion registration raises when called twice, so a second call returns before it.
    """
    global _applied
    if _applied:
        return
    with quiet_qwen_import():  # the first import of qwen_tts: it probes for sox and flash_attn and prints that neither exists
        from qwen_tts._transformers_compat import _default_rope_parameters
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Encoder
    from transformers.conversion_mapping import register_checkpoint_conversion_mapping
    from transformers.core_model_loading import WeightRenaming
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    # 1. Shadow the read-only Mimi properties the encoder assigns.
    Qwen3TTSTokenizerV2Encoder.upsample = None
    Qwen3TTSTokenizerV2Encoder.decoder_transformer = None
    # 2. Give the custom-code encoder MimiModel's six weight renames.
    register_checkpoint_conversion_mapping(
        "Qwen3TTSTokenizerV2Encoder",
        [
            WeightRenaming(source_patterns=r"^encoder\.layers", target_patterns="encoder.encoder.layers"),
            WeightRenaming(source_patterns=r"^encoder_transformer\.", target_patterns="encoder.encoder_transformer."),
            WeightRenaming(source_patterns=r"^downsample\.", target_patterns="encoder.downsample."),
            WeightRenaming(source_patterns=r"^upsample\.", target_patterns="decoder.upsample."),
            WeightRenaming(source_patterns=r"^decoder_transformer\.", target_patterns="decoder.decoder_transformer."),
            WeightRenaming(source_patterns=r"^decoder\.layers", target_patterns="decoder.decoder.layers"),
        ],
    )
    # 3. Take qwen_tts's "default" out of the global registry and keep it for qwen_tts's own rotary classes.
    if ROPE_INIT_FUNCTIONS.get("default") is _default_rope_parameters:
        del ROPE_INIT_FUNCTIONS["default"]
    private = {**ROPE_INIT_FUNCTIONS, "default": _default_rope_parameters}
    for name in _ROPE_MODULES:
        module = sys.modules[name]
        module.ROPE_INIT_FUNCTIONS = private
        for cls_name, cls in vars(module).items():
            if isinstance(cls, type) and cls.__module__ == name and ("Rotary" in cls_name or "Rotatory" in cls_name):
                cls.compute_default_rope_parameters = staticmethod(_default_rope_parameters)
    _applied = True
