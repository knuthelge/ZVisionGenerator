"""Pin the transformers codec internals the Higgs TTS 3 backend relies on.

The backend loads ``HiggsAudioV2TokenizerModel`` and, on close, clears the ``lru_cache`` on
``HiggsAudioV2TokenizerPreTrainedModel._get_conv1d_layers``, whose keys hold every codec that encoded audio. A transformers
upgrade that moves the class or drops the cache fails here instead of leaving a codec in memory.
"""

from __future__ import annotations

import importlib

from zvisiongenerator.backends.audio import higgs3


def _codec_module():
    return importlib.import_module(higgs3._CODEC_MODULE)


def test_the_codec_model_lives_in_the_module_the_backend_clears():
    from transformers import HiggsAudioV2TokenizerModel

    assert HiggsAudioV2TokenizerModel.__module__ == higgs3._CODEC_MODULE
    assert issubclass(HiggsAudioV2TokenizerModel, _codec_module().HiggsAudioV2TokenizerPreTrainedModel)


def test_the_conv_layer_lookup_caches_by_codec_and_can_be_cleared():
    from torch import nn

    codec_class = _codec_module().HiggsAudioV2TokenizerPreTrainedModel
    lookup = codec_class._get_conv1d_layers
    lookup.cache_clear()
    codec = object.__new__(codec_class)

    codec._get_conv1d_output_lengths(100, nn.Sequential(nn.Conv1d(1, 1, 3)))
    assert lookup.cache_info().currsize == 1

    lookup.cache_clear()
    assert lookup.cache_info().currsize == 0
