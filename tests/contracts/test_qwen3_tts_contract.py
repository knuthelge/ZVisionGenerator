"""Pin the parts of ``faster-qwen3-tts`` and ``qwen-tts-hf`` the Qwen3-TTS backend calls, and the shim that makes them load on the locked transformers.

Both libraries are vendored under ``packages/`` at the commits ``make update-qwen3-tts`` pins, so a failure after an update names
what moved. The shim patches transformers' global state (the RoPE registry and the conversion mappings), so it is applied in a fresh
interpreter and the observations are read back as JSON; the test process itself stays untouched. Not run on macOS, where the speech model runs on mlx-audio.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="Qwen3-TTS runs on mlx-audio on macOS")

_SCRIPT = textwrap.dedent(
    """
    import inspect, json, sys
    import faster_qwen3_tts, qwen_tts
    from faster_qwen3_tts import FasterQwen3TTS
    from qwen_tts._transformers_compat import _default_rope_parameters
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Encoder
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    import transformers.conversion_mapping as conversion_mapping
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping
    from transformers.models.mimi.modeling_mimi import MimiModel
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    def params(function):
        return {name: parameter.default is inspect.Parameter.empty for name, parameter in inspect.signature(function).parameters.items()}

    def rules(mapping):
        return [[list(rule.source_patterns), list(rule.target_patterns)] for rule in mapping]

    out = {
        "signatures": {
            "from_pretrained": params(FasterQwen3TTS.from_pretrained),
            "generate_custom_voice": params(FasterQwen3TTS.generate_custom_voice),
            "generate_voice_clone": params(FasterQwen3TTS.generate_voice_clone),
            "warmup": params(FasterQwen3TTS.warmup),
            "create_voice_clone_prompt": params(Qwen3TTSModel.create_voice_clone_prompt),
        },
        "mimi_read_only": {name: isinstance(getattr(MimiModel, name, None), property) and getattr(MimiModel, name).fset is None for name in ("upsample", "decoder_transformer")},
        "default_before": "default" in ROPE_INIT_FUNCTIONS,
        "default_is_qwens_before": ROPE_INIT_FUNCTIONS.get("default") is _default_rope_parameters,
        "mimi_rules": rules(get_checkpoint_conversion_mapping("MimiModel")),
        "encoder_mapping_before": get_checkpoint_conversion_mapping("Qwen3TTSTokenizerV2Encoder") is not None,
    }

    registrations = []
    original = conversion_mapping.register_checkpoint_conversion_mapping

    def counting(*args, **kwargs):
        registrations.append(args[0])
        return original(*args, **kwargs)

    conversion_mapping.register_checkpoint_conversion_mapping = counting
    from zvisiongenerator.backends.audio.qwen3_tts_shim import apply_qwen_shim
    apply_qwen_shim()
    apply_qwen_shim()  # a second call must do nothing: the registration raises when repeated
    conversion_mapping.register_checkpoint_conversion_mapping = original

    out["registrations"] = registrations
    out["default_after"] = "default" in ROPE_INIT_FUNCTIONS
    out["encoder_upsample"] = repr(Qwen3TTSTokenizerV2Encoder.upsample)
    out["encoder_decoder_transformer"] = repr(Qwen3TTSTokenizerV2Encoder.decoder_transformer)
    out["encoder_rules"] = rules(get_checkpoint_conversion_mapping("Qwen3TTSTokenizerV2Encoder"))
    for name in ("qwen_tts.core.models.modeling_qwen3_tts", "qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2"):
        registry = sys.modules[name].ROPE_INIT_FUNCTIONS
        out["private_" + name.rsplit(".", 1)[1]] = registry.get("default") is _default_rope_parameters and registry is not ROPE_INIT_FUNCTIONS
    print("RESULT" + json.dumps(out))
    """
)


@pytest.fixture(scope="module")
def observed() -> dict:
    """Import the libraries and apply the shim twice in a fresh interpreter, and return what it saw before and after."""
    done = subprocess.run([sys.executable, "-I", "-c", _SCRIPT], capture_output=True, text=True, timeout=300, check=False)
    assert done.returncode == 0, done.stderr[-3000:]
    line = next(line for line in done.stdout.splitlines() if line.startswith("RESULT"))
    return json.loads(line.removeprefix("RESULT"))


class TestLibrarySignatures:
    def test_from_pretrained_takes_what_the_backend_passes(self, observed):
        assert {"model_name", "device", "dtype", "local_files_only"} <= set(observed["signatures"]["from_pretrained"])

    def test_generate_custom_voice_takes_the_speaker_language_instruction_and_limit(self, observed):
        signature = observed["signatures"]["generate_custom_voice"]

        assert {"text", "speaker", "language", "instruct", "max_new_tokens"} <= set(signature)
        assert signature["text"] and signature["speaker"] and signature["language"] and not signature["instruct"]

    def test_generate_voice_clone_takes_a_precomputed_prompt(self, observed):
        signature = observed["signatures"]["generate_voice_clone"]

        assert {"text", "language", "voice_clone_prompt", "max_new_tokens"} <= set(signature)
        assert not signature["voice_clone_prompt"]

    def test_the_model_can_warm_up_and_build_a_cloning_prompt_from_a_waveform_and_its_transcript(self, observed):
        assert "warmup" in observed["signatures"]
        assert {"ref_audio", "ref_text"} <= set(observed["signatures"]["create_voice_clone_prompt"])


class TestShimPremises:
    """Each part of the shim exists because of a fault of the installed libraries; these fail when the fault goes away or moves."""

    def test_mimi_properties_the_encoder_assigns_are_read_only(self, observed):
        assert observed["mimi_read_only"] == {"upsample": True, "decoder_transformer": True}

    def test_importing_qwen_tts_puts_its_default_rope_function_in_transformers_global_registry(self, observed):
        assert observed["default_before"] is True and observed["default_is_qwens_before"] is True

    def test_transformers_has_no_renames_for_the_custom_encoder_until_the_shim_registers_them(self, observed):
        assert observed["encoder_mapping_before"] is False

    def test_mimi_model_has_six_renames(self, observed):
        assert len(observed["mimi_rules"]) == 6


class TestShim:
    def test_the_default_rope_function_is_gone_from_the_global_registry(self, observed):
        assert observed["default_after"] is False

    def test_the_encoders_read_only_properties_are_shadowed(self, observed):
        assert (observed["encoder_upsample"], observed["encoder_decoder_transformer"]) == ("None", "None")

    def test_the_conversion_mapping_is_registered_once_though_the_shim_ran_twice(self, observed):
        assert observed["registrations"] == ["Qwen3TTSTokenizerV2Encoder"]

    def test_the_encoder_gets_exactly_the_renames_of_mimi_model(self, observed):
        assert observed["encoder_rules"] == observed["mimi_rules"]

    def test_qwens_own_modules_keep_a_private_registry_with_the_default_function(self, observed):
        assert observed["private_modeling_qwen3_tts"] is True
        assert observed["private_modeling_qwen3_tts_tokenizer_v2"] is True


def test_applying_the_shim_prints_nothing_about_sox_or_flash_attn():
    """The first import of ``qwen_tts`` probes for the sox program (a shell's "sox: not found", a SoX warning) and flash_attn; none of that reaches the console."""
    code = "from zvisiongenerator.backends.audio.qwen3_tts_shim import apply_qwen_shim; apply_qwen_shim()"
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=300, check=False)

    assert done.returncode == 0, done.stderr[-3000:]
    assert "sox" not in done.stderr.lower() and "flash-attn" not in done.stdout and done.stdout.strip() == ""
