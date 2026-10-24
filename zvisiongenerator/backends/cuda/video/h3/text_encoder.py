"""The MiniMax H3 text encoder: a 50-layer Qwen3-VL in NF4, with its vision tower for keyframe conditioning.

H3 conditions on ``hidden_states[50]`` of Qwen3-VL-32B. A tower cut to 50 layers has the layer-50 output as its *last*
hidden state, which transformers replaces with the post-norm output; swapping the final norm for ``Identity`` makes
the last hidden state the raw layer-50 output, which is the conditioning the released weights were trained against.

The presentation is what diffusers' H3 text-encoder steps build: the prompt verbatim with no chat template and no
special tokens, preceded for each keyframe by a ``"<Picture i>: "`` label and a vision block whose rows count as video.
"""

from __future__ import annotations

import copy
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.backends.cuda import quant
from zvisiongenerator.backends.cuda.block_streaming import move_outside, stream_blocks
from zvisiongenerator.utils.safetensors_stream import iter_tensors, read_header

if TYPE_CHECKING:
    import torch

__all__ = ["TEXT_ENCODER_LAYERS", "TEXT_TAG", "VIDEO_TAG", "Presentation", "empty_text_encoder", "encode", "load_text_encoder", "tokenize_presentation", "wanted_tensor"]

TEXT_ENCODER_LAYERS = 50  # decoder layers kept: the conditioning is the output of the 50th
TEXT_TAG = 1  # modality tag of a text row of the packed sequence
VIDEO_TAG = 0  # modality tag of a video row; a keyframe's vision block carries it
_CHECKPOINT_PREFIX = "model."  # tensors of the Qwen3-VL checkpoint live under it, the bare ``Qwen3VLModel`` omits it
_TOWER_LAYERS = "language_model.layers."  # the decoder layers, as the bare model names them


@dataclass(frozen=True)
class Presentation:
    """One tokenized H3 prompt: what the text encoder reads.

    Attributes:
        token_ids: The token ids, keyframe labels and vision blocks first, then the prompt.
        token_tags: The modality tag of each row (``TEXT_TAG``, or ``VIDEO_TAG`` inside a vision block).
        mm_token_type_ids: Qwen3-VL's own per-token modality type (0 text, 1 image) that lays out its rotary positions.
        vision_inputs: The processed keyframes by the encoder's parameter names (``pixel_values``, ``image_grid_thw``);
            empty for a text-only prompt.
    """

    token_ids: tuple[int, ...]
    token_tags: tuple[int, ...]
    mm_token_type_ids: tuple[int, ...]
    vision_inputs: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        if not len(self.token_ids) == len(self.token_tags) == len(self.mm_token_type_ids):
            raise ValueError(f"A presentation needs one tag and one type per token, got {len(self.token_ids)} ids, {len(self.token_tags)} tags and {len(self.mm_token_type_ids)} types.")


def tokenize_presentation(prompt: str, tokenizer_dir: Path, processor_dir: Path, images: Sequence[Any]) -> Presentation:
    """Return the presentation of *prompt* with the keyframe *images* (PIL images, in packed order) in front of it.

    Without images the prompt is tokenized verbatim and *processor_dir* is not read. With images, each one adds a
    ``"<Picture i>: "`` label and a vision block (start, one pad per merged patch, end) whose rows are tagged as video.

    Args:
        prompt: The text prompt.
        tokenizer_dir: The directory of the Qwen2 tokenizer.
        processor_dir: The directory of the Qwen3-VL processor (image processor and tokenizer).
        images: The keyframes, already fitted to the target canvas.
    """
    from transformers import Qwen2TokenizerFast

    tokenizer = Qwen2TokenizerFast.from_pretrained(str(tokenizer_dir))
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not images:
        return Presentation(tuple(prompt_ids), (TEXT_TAG,) * len(prompt_ids), (0,) * len(prompt_ids), {})

    from transformers import Qwen3VLProcessor

    processor = Qwen3VLProcessor.from_pretrained(str(processor_dir))
    vision = processor.image_processor(images=list(images), return_tensors="pt")
    grids = vision["image_grid_thw"]
    merge_area = processor.image_processor.merge_size**2
    start, pad, end = (tokenizer.convert_tokens_to_ids(token) for token in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>"))
    token_ids: list[int] = []
    token_tags: list[int] = []
    for index in range(len(images)):
        label = tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
        block = [start] + [pad] * (int(grids[index].prod()) // merge_area) + [end]
        token_ids += label + block
        token_tags += [TEXT_TAG] * len(label) + [VIDEO_TAG] * len(block)
    token_ids += prompt_ids
    token_tags += [TEXT_TAG] * len(prompt_ids)
    mm_token_type_ids = processor.create_mm_token_type_ids([token_ids])[0]
    return Presentation(tuple(token_ids), tuple(token_tags), tuple(mm_token_type_ids), {"pixel_values": vision["pixel_values"], "image_grid_thw": grids})


def empty_text_encoder(config: Any, *, layers: int = TEXT_ENCODER_LAYERS, vision: bool = True, dtype: torch.dtype | None = None) -> Any:
    """Return a ``Qwen3VLModel`` without weights (on the meta device) cut to *layers* decoder layers, its final norm an ``Identity``.

    Args:
        config: The checkpoint's ``Qwen3VLConfig``; it is not modified.
        layers: How many decoder layers to keep.
        vision: Whether to keep the vision tower (without it, ``visual`` is ``None`` and images cannot be encoded).
        dtype: The dtype of the parameters; bfloat16 by default. The rotary buffers keep their own dtype.

    Raises:
        ValueError: If *layers* is not between 1 and the checkpoint's layer count.
    """
    import torch
    from accelerate import init_empty_weights
    from transformers import Qwen3VLModel

    available = config.text_config.num_hidden_layers
    if not 1 <= layers <= available:
        raise ValueError(f"The text encoder keeps between 1 and {available} layers, got {layers}.")
    config = copy.deepcopy(config)
    config.text_config.num_hidden_layers = layers
    if getattr(config.text_config, "layer_types", None):
        config.text_config.layer_types = list(config.text_config.layer_types)[:layers]
    with init_empty_weights():
        model = Qwen3VLModel(config)
    for param in model.parameters():
        param.data = param.data.to(dtype or torch.bfloat16)
    model.language_model.norm = torch.nn.Identity()
    if not vision:
        model.visual = None
    return model


def wanted_tensor(name: str, layers: int = TEXT_ENCODER_LAYERS, vision: bool = True) -> bool:
    """Return whether the checkpoint tensor *name* belongs to the cut model: the embeddings, the first *layers* decoder layers and, with *vision*, the vision tower.

    The final norm (replaced by ``Identity``), the layers beyond *layers* and ``lm_head`` are never read.
    """
    if not name.startswith(_CHECKPOINT_PREFIX):
        return False
    bare = name[len(_CHECKPOINT_PREFIX) :]
    if bare.startswith("visual."):
        return vision
    if bare.startswith(_TOWER_LAYERS):
        return int(bare[len(_TOWER_LAYERS) :].partition(".")[0]) < layers
    return bare.startswith("language_model.") and not bare.startswith("language_model.norm.")


def load_text_encoder(config_dir: Path, shards: Sequence[Path], device: Any, *, layers: int = TEXT_ENCODER_LAYERS, vision: bool = True) -> Any:
    """Build the cut Qwen3-VL from the NF4 *shards* and return it ready for :func:`encode`.

    The decoder layers stay in system memory and stream to *device* as they run; the embeddings, rotary buffers and
    vision tower live on *device*. The linears of the decoder layers are NF4 (``Linear4bit``); everything else is bfloat16.

    Args:
        config_dir: The directory holding the checkpoint's ``config.json``.
        shards: The safetensors files with the tensors (the NF4 build's, vision tower included).
        device: The GPU.
        layers: How many decoder layers to keep.
        vision: Whether to load the vision tower (needed to encode keyframes).

    Raises:
        RuntimeError: If the shards do not hold every tensor the model needs.
        ValueError: If *layers* is out of range.
    """
    import torch
    from transformers import AutoConfig

    model = empty_text_encoder(AutoConfig.from_pretrained(str(config_dir)), layers=layers, vision=vision, dtype=torch.bfloat16)
    linears = quant.swap_in_linear4bit(model, quant.nf4_linear_names(model, prefix=_TOWER_LAYERS), torch.bfloat16)
    _check_coverage(model, shards)
    renamed = ((name[len(_CHECKPOINT_PREFIX) :], tensor) for path in shards for name, tensor in iter_tensors(path, lambda name: wanted_tensor(name, layers, vision)))
    quant.fill_weights(model, quant.place_nf4(renamed, linears), ", ".join(path.name for path in shards), skipped=linears)
    _check_nf4_placed(linears)
    model.eval()
    stream_blocks(model.language_model.layers, device)
    move_outside(model, model.language_model.layers, device)
    return model


def encode(model: Any, presentation: Presentation, device: Any) -> torch.Tensor:
    """Return the ``[1, N, hidden]`` hidden state after the last decoder layer of *model* for *presentation* (the conditioning).

    Raises:
        ValueError: If the presentation has images but *model* was loaded without its vision tower.
    """
    import torch

    if presentation.vision_inputs and getattr(model, "visual", None) is None:
        raise ValueError("The presentation has keyframes but the text encoder was loaded without its vision tower.")
    ids = torch.tensor([presentation.token_ids], dtype=torch.long, device=device)
    types = torch.tensor([presentation.mm_token_type_ids], dtype=torch.long, device=device)
    vision = {name: value.to(device, model.dtype) if name.startswith("pixel_") else value.to(device) for name, value in presentation.vision_inputs.items()}
    with torch.no_grad():
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=types, use_cache=False, output_hidden_states=True, **vision)
    return output.hidden_states[-1]


def _check_coverage(model: Any, shards: Collection[Path]) -> None:
    """Raise unless the headers of *shards* together list every tensor the empty *model* needs (an NF4 linear is checked by its weight).

    Raises:
        RuntimeError: Naming how many tensors are missing, before any weight is read.
    """
    present: set[str] = set()
    for path in shards:
        present |= read_header(path)[1].keys()
    missing = sorted({f"{_CHECKPOINT_PREFIX}{name}" for name in model.state_dict()} - present)
    if missing:
        raise RuntimeError(f"The text encoder shards do not hold every tensor the model needs ({len(missing)} missing, e.g. {missing[0]}); check that all shards are listed.")


def _check_nf4_placed(linears: Mapping[str, Any]) -> None:
    """Raise unless every NF4 linear in *linears* has its packed weight and quantization state.

    Raises:
        RuntimeError: Naming the first linear left empty.
    """
    for name, layer in linears.items():
        if getattr(layer.weight, "quant_state", None) is None:
            raise RuntimeError(f"The NF4 weight {name} was not found in the text encoder shards.")
