"""Rename a LoRA file's tensors to the names each model family's loaders read.

Trainers disagree on how they name a LoRA's tensors, and the loaders each read only some spellings:

* PEFT writes the adapter name into the key (``….lora_A.default.weight``); diffusers' Krea 2, FLUX.2 and LTX-2
  converters read only the plain ``….lora_A.weight``, and MOSS-Image's loader reads the infix as part of the layer name.
* Kohya flattens a module path into one underscore name (``lora_unet_blocks_0_attn_wq``). Module names contain
  underscores themselves (``attn_gate``, ``to_q``, ``adaln_modulation``), so the dots are recovered by matching the
  whole name against the family's real module names (:data:`LORA_MODULES`), never by splitting on ``_``.
* A file may have no prefix at all, where a converter only runs for ``diffusion_model.`` or ``transformer.`` keys.

:func:`normalise_lora_keys` is the one place that knows this. It works on names alone, so it is pure and imports no
torch: the family detector reads header names through it, and every loader renames the tensors it is about to hand to
diffusers, mflux, MLX or the MOSS and MiniMax H3 loaders through the same call. With no family it gives the
family-agnostic view (adapter name dropped, Kohya names dotted) that fingerprinting needs; with a family and a loader
it renames only what that loader cannot read, so a file that loads already keeps its names (and, for the loaders that
take only a path, is never copied).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
import re

__all__ = ["LORA_MODULES", "folds_alpha", "normalise_lora_keys"]


def _blocks(prefix: str, modules: Iterable[str]) -> tuple[str, ...]:
    """Return the module templates of a numbered block list: ``<prefix>.{n}.<module>`` for each of *modules*."""
    return tuple(f"{prefix}.{{n}}.{module}" for module in modules)


_ZIMAGE_BLOCK = (
    "attention.to_q",
    "attention.to_k",
    "attention.to_v",
    "attention.to_out.0",
    "attention.qkv",
    "attention.out",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
    "adaLN_modulation.0",
)
_KREA2_BLOCK = ("attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.gate", "mlp.gate", "mlp.up", "mlp.down")
_H3_BLOCK = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
_FLUX_GLOBALS = (
    "img_in",
    "txt_in",
    "time_in.in_layer",
    "time_in.out_layer",
    "vector_in.in_layer",
    "vector_in.out_layer",
    "guidance_in.in_layer",
    "guidance_in.out_layer",
    "final_layer.linear",
    "final_layer.adaLN_modulation.1",
    "double_stream_modulation_img.lin",
    "double_stream_modulation_txt.lin",
    "single_stream_modulation.lin",
)

# The module names (original trainer namespace, no prefix) a Kohya name can flatten, by family. ``{n}`` is a block number.
# They come from the converters in diffusers and the weight mappings in mflux, and cover every Kohya name found in
# real LoRAs of these families. Every flattened spelling must be unique across the table, which the tests check.
LORA_MODULES: dict[str, tuple[str, ...]] = {
    "zimage": (
        *_blocks("layers", _ZIMAGE_BLOCK),
        *_blocks("noise_refiner", _ZIMAGE_BLOCK),
        *_blocks("context_refiner", _ZIMAGE_BLOCK),
        "x_embedder",
        "final_layer.linear",
        "final_layer.adaLN_modulation.1",
        "cap_embedder.1",
        "t_embedder.mlp.0",
        "t_embedder.mlp.2",
    ),
    "ideogram4": (
        *_blocks(
            "layers",
            ("attention.qkv", "attention.o", "attention.to_q", "attention.to_k", "attention.to_v", "attention.to_out.0", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3", "adaln_modulation"),
        ),
        "final_layer.adaln_modulation",
    ),
    "krea2": (
        *_blocks("blocks", _KREA2_BLOCK),
        *_blocks("txtfusion.layerwise_blocks", _KREA2_BLOCK),
        *_blocks("txtfusion.refiner_blocks", _KREA2_BLOCK),
        "first",
        "last.linear",
        "tmlp.0",
        "tmlp.2",
        "tproj.1",
        "txtmlp.1",
        "txtmlp.3",
        "txtfusion.projector",
    ),
    "h3": (
        *_blocks("blocks", (*_H3_BLOCK, "adaln_proj.linear")),
        *_blocks("token_refiner.blocks", _H3_BLOCK),
        "video_patch_proj",
        "audio_patch_proj",
        "condition_proj",
        "time_embedder.proj_in",
        "time_embedder.proj_out",
        "final_layer.adaln_proj.linear",
        "final_layer.video_out",
        "final_layer.audio_out",
    ),
    "moss": (
        *_blocks("layers", ("proj.q", "proj.k", "proj.v", "proj.o", "proj.qkv", "proj.ffn_up_gate", "proj.ffn_down")),
        *_blocks("text_embedder.blocks", ("qkv", "proj", "ffn_up", "ffn_gate", "ffn_down")),
        "text_embedder.proj",
        "video_embedder",
        "final_linear_video",
        "shared_block_modulation",
        "video_head_modulation",
        "time_embedding.0",
        "time_embedding.2",
    ),
    "flux2_klein": (
        *_blocks("double_blocks", ("img_attn.qkv", "img_attn.proj", "img_mlp.0", "img_mlp.2", "img_mod.lin", "txt_attn.qkv", "txt_attn.proj", "txt_mlp.0", "txt_mlp.2", "txt_mod.lin")),
        *_blocks("single_blocks", ("linear1", "linear2", "modulation.lin")),
        *_FLUX_GLOBALS,
    ),
}
# FLUX.1 and FLUX.2 Klein share the trainer namespace.
LORA_MODULES["flux1"] = LORA_MODULES["flux2_klein"]


@dataclass(frozen=True)
class _Policy:
    """What one loader needs, for one family, beyond the family-agnostic view.

    Attributes:
        strip_adapter: When to drop the PEFT adapter name: ``always``, ``never`` (the loader reads it) or only in a
            ``native`` file (the loader reads it on diffusers names only).
        keep_default_adapter: Leave the ``default`` adapter name in even when stripping. MiniMax H3's converter reads it
            as the mark of a DiffSynth file, whose fused qkv is laid out per head.
        kohya: Write Kohya names dotted. Off for a loader that reads them itself.
        down_up_as_a_b: Write ``lora_down``/``lora_up`` as ``lora_A``/``lora_B``, the only spelling the converter reads.
        indexed_to_out: Write an attention's ``to_out`` as ``to_out.0``, the output projection's place in the model's module list.
        fold_alpha: The converter cannot take an ``alpha`` tensor, so the loader multiplies ``alpha / rank`` into the
            weights (see :func:`~zvisiongenerator.utils.lora_files.normalised_lora_state`).
        bare_prefix: Give a key with no prefix the one the converter reads. Off for a loader that reads bare names.
        native_roots: First segments of the trainer namespace. They tell a file is in it, and in such a file a key with
            no prefix gets ``diffusion_model.``.
        diffusers_roots: First segments of the diffusers namespace. They tell a file is in it, and in such a file a key
            with no prefix, or under ``base_model.model.``, gets ``transformer.``.
        shared_roots: First segments that both namespaces use. They follow the namespace of the file.
        shared_namespace: The namespace of a file whose modules are all in *shared_roots*, which tell neither (``native``
            or ``diffusers``). None leaves such a file as it is.
        peft_to_native: Give a key under ``base_model.model.`` in a native file the native prefix too. Off for a
            converter that reads ``base_model.model.`` itself.
    """

    strip_adapter: str = "always"
    keep_default_adapter: bool = False
    kohya: bool = False
    down_up_as_a_b: bool = False
    indexed_to_out: bool = False
    fold_alpha: bool = False
    bare_prefix: bool = True
    native_roots: frozenset[str] = frozenset()
    diffusers_roots: frozenset[str] = frozenset()
    shared_roots: frozenset[str] = frozenset()
    shared_namespace: str | None = None
    peft_to_native: bool = False


_ZIMAGE_ROOTS = frozenset({"layers", "noise_refiner", "context_refiner", "x_embedder", "all_x_embedder", "final_layer", "all_final_layer", "cap_embedder", "t_embedder"})
_KLEIN_NATIVE = frozenset({"double_blocks", "single_blocks"})
_KLEIN_DIFFUSERS = frozenset({"transformer_blocks", "single_transformer_blocks"})
_KLEIN_SHARED = frozenset({"context_embedder", "x_embedder", "proj_out", "norm_out", "time_guidance_embed", "double_stream_modulation_img", "double_stream_modulation_txt", "single_stream_modulation"})
_KREA2_NATIVE = frozenset({"blocks", "txtfusion", "first", "last", "tmlp", "tproj", "txtmlp"})
_KREA2_DIFFUSERS = frozenset({"transformer_blocks", "text_fusion", "img_in", "final_layer", "time_embed", "txt_in", "time_mod_proj"})
# The first segments of the LTX-2 transformer's modules: the original trainer's (``ltx_core_mlx``'s model) and diffusers'
# (``LTX2VideoTransformer3DModel``'s), which diffusers' converter renames one by one.
_LTX_NATIVE = frozenset(
    {
        "patchify_proj",
        "audio_patchify_proj",
        "adaln_single",
        "audio_adaln_single",
        "prompt_adaln_single",
        "audio_prompt_adaln_single",
        "av_ca_video_scale_shift_adaln_single",
        "av_ca_audio_scale_shift_adaln_single",
        "av_ca_a2v_gate_adaln_single",
        "av_ca_v2a_gate_adaln_single",
    }
)
_LTX_DIFFUSERS = frozenset(
    {
        "proj_in",
        "audio_proj_in",
        "time_embed",
        "audio_time_embed",
        "prompt_adaln",
        "audio_prompt_adaln",
        "av_cross_attn_video_scale_shift",
        "av_cross_attn_audio_scale_shift",
        "av_cross_attn_video_a2v_gate",
        "av_cross_attn_audio_v2a_gate",
    }
)
_LTX_SHARED = frozenset({"transformer_blocks", "proj_out", "audio_proj_out", "caption_projection", "audio_caption_projection"})
_IDEOGRAM4_ROOTS = frozenset({"layers", "final_layer"})

# Keyed by (loader, family). The loaders are diffusers (CUDA image and video), mflux and MLX (macOS, which read only
# files) and the MOSS-Image and MiniMax H3 loaders. What is off for mflux and MLX is what they read as it is, so a
# file that loads there already is never copied.
_POLICIES: dict[tuple[str, str], _Policy] = {
    ("diffusers", "zimage"): _Policy(native_roots=_ZIMAGE_ROOTS),
    ("diffusers", "ideogram4"): _Policy(kohya=True, native_roots=_IDEOGRAM4_ROOTS),
    ("diffusers", "krea2"): _Policy(kohya=True, down_up_as_a_b=True, fold_alpha=True, native_roots=_KREA2_NATIVE, diffusers_roots=_KREA2_DIFFUSERS),
    ("diffusers", "flux2_klein"): _Policy(native_roots=_KLEIN_NATIVE, diffusers_roots=_KLEIN_DIFFUSERS, shared_roots=_KLEIN_SHARED),
    ("diffusers", "ltx"): _Policy(
        indexed_to_out=True,
        native_roots=_LTX_NATIVE,
        diffusers_roots=_LTX_DIFFUSERS,
        shared_roots=_LTX_SHARED,
        shared_namespace="native",
        peft_to_native=True,
    ),
    ("mflux", "zimage"): _Policy(strip_adapter="never", kohya=True, bare_prefix=False),
    ("mflux", "ideogram4"): _Policy(native_roots=_IDEOGRAM4_ROOTS),
    ("mflux", "krea2"): _Policy(strip_adapter="never", bare_prefix=False),
    ("mflux", "flux2_klein"): _Policy(strip_adapter="native", bare_prefix=False, native_roots=_KLEIN_NATIVE, diffusers_roots=_KLEIN_DIFFUSERS, shared_roots=_KLEIN_SHARED),
    ("mlx", "ltx"): _Policy(bare_prefix=False),
    ("moss", "moss"): _Policy(bare_prefix=False),
    ("h3", "h3"): _Policy(keep_default_adapter=True, bare_prefix=False),
}
# What the detector fingerprints: every family's names in one spelling, each key keeping its prefix.
_FINGERPRINT = _Policy(kohya=True, bare_prefix=False)

# The prefixes a trainer puts before the module path, longest first.
_PREFIXES = ("model.diffusion_model.", "base_model.model.", "diffusion_model.", "transformer.")
_NATIVE_PREFIX = "diffusion_model."
_DIFFUSERS_PREFIX = "transformer."
_PEFT_PREFIX = "base_model.model."
_KOHYA_PREFIX = "lora_unet_"
# The tensor a key names, after the module path: a LoRA weight (``lora_A.<adapter>.weight``), alpha, or another adapter tensor.
_TAIL = re.compile(r"^(?P<head>.+?)\.(?P<tail>lora_(?:A|B|down|up)(?:\.[^.]+)?\.weight|lora\.(?:down|up)\.weight|alpha|dora_scale|lokr_\w+|hada_\w+|diff_b|diff)$")
# LyCORIS tensors: a file with any is applied by the LoKr code, which reads its own names, so loaders leave its names alone.
_LYCORIS_TENSOR = re.compile(r"\.(?:lokr|hada)_\w+$")
_ADAPTER_INFIX = re.compile(r"^(?P<kind>lora_(?:A|B|down|up))\.(?P<name>[^.]+)\.weight$")
_DEFAULT_ADAPTER = "default"
_DOWN_UP = {"lora_down.weight": "lora_A.weight", "lora_up.weight": "lora_B.weight"}

_Parsed = tuple[str, str, str]  # (prefix with its dot, module path, tensor tail)


def normalise_lora_keys(names: Iterable[str], family: str | None = None, *, loader: str = "diffusers") -> dict[str, str]:
    """Return ``{name: normalised name}`` for each of *names* that changes; the others are not in the result.

    A rename that would make two tensors share a name is left out, so no tensor is ever lost to it.

    Args:
        names: The tensor names of one LoRA file.
        family: The family of the model the file is loaded on (``zimage``, ``krea2``, ...). None gives the
            family-agnostic view the detector fingerprints: the adapter name dropped and every family's Kohya names
            written dotted, with each key keeping its prefix.
        loader: What reads the file: ``diffusers`` (CUDA image and video), ``mflux`` or ``mlx`` (macOS), ``moss`` or
            ``h3``. Ignored without a family.

    Returns:
        The renames to apply. Nothing, for a loader or family this module has no policy for, and for a file with
        LyCORIS (LoKr, LoHa) tensors, which a loader reads by its own names.
    """
    names = list(names)
    if family is None:
        policy = _FINGERPRINT
    else:
        policy = _POLICIES.get((loader, family))
        if policy is None or any(_LYCORIS_TENSOR.search(name) for name in names):
            return {}
    parsed = {name: found for name in names if (found := _parse(name)) is not None}
    namespace = _namespace(parsed.values(), policy)
    renames = {name: new for name, found in parsed.items() if (new := _normalise(found, family, policy, namespace)) != name}
    taken = [name for name in names if name not in renames]
    claimed: dict[str, list[str]] = {}
    for name, new in renames.items():
        claimed.setdefault(new, []).append(name)
    clashing = {new for new, sources in claimed.items() if len(sources) > 1 or new in taken}
    return {name: new for name, new in renames.items() if new not in clashing}


def folds_alpha(family: str, *, loader: str = "diffusers") -> bool:
    """Return whether *loader* cannot take an ``alpha`` tensor for *family*, so its loader multiplies ``alpha / rank`` into the weights."""
    policy = _POLICIES.get((loader, family))
    return policy is not None and policy.fold_alpha


def _parse(name: str) -> _Parsed | None:
    """Split a tensor name into prefix, module path and tensor tail, or None when it is not a LoRA tensor's name."""
    match = _TAIL.match(name)
    if match is None:
        return None
    for prefix in _PREFIXES:
        if match.group("head").startswith(prefix):
            return prefix, match.group("head").removeprefix(prefix), match.group("tail")
    return "", match.group("head"), match.group("tail")


def _namespace(parsed: Iterable[_Parsed], policy: _Policy) -> str | None:
    """Return ``native`` or ``diffusers`` for a file whose modules are all in one namespace, else None.

    The namespace is a property of the file: a module whose name both namespaces use goes where the rest of the file goes.
    A file with only such modules is in the policy's ``shared_namespace``, if it has one.
    """
    roots = {module.partition(".")[0] for _, module, _ in parsed}
    native, diffusers = bool(roots & policy.native_roots), bool(roots & policy.diffusers_roots)
    if native and diffusers:
        return None
    if not native and not diffusers:
        return policy.shared_namespace if roots & policy.shared_roots else None
    return "native" if native else "diffusers"


def _normalise(parsed: _Parsed, family: str | None, policy: _Policy, namespace: str | None) -> str:
    """Return the name of the tensor *parsed* for *policy*, in a file in *namespace*."""
    prefix, module, tail = parsed
    if policy.strip_adapter == "always" or (policy.strip_adapter == "native" and namespace == "native"):
        tail = _plain_tail(tail, policy)
    if not prefix and module.startswith(_KOHYA_PREFIX):
        if policy.kohya:
            dotted = _unflatten(module.removeprefix(_KOHYA_PREFIX), None if family is None else (family,))
            if dotted is None:
                return f"{module}.{parsed[2]}"  # a flat name no table spells is of no use renamed
            prefix, module = _NATIVE_PREFIX, dotted
    else:
        prefix = _loader_prefix(prefix, module, policy, namespace)
    if policy.down_up_as_a_b:
        tail = _DOWN_UP.get(tail, tail)
    if policy.indexed_to_out and module.endswith(".to_out"):
        module += ".0"
    return f"{prefix}{module}.{tail}"


def _plain_tail(tail: str, policy: _Policy) -> str:
    """Return the tensor *tail* without a PEFT adapter name (the ``default`` one stays where *policy* keeps it)."""
    infix = _ADAPTER_INFIX.match(tail)
    if infix is None or (policy.keep_default_adapter and infix.group("name") == _DEFAULT_ADAPTER):
        return tail
    return f"{infix.group('kind')}.weight"


def _loader_prefix(prefix: str, module: str, policy: _Policy, namespace: str | None) -> str:
    """Return the prefix the loader reads for *module*, which has *prefix* in a file in *namespace*."""
    root = module.partition(".")[0]
    if namespace == "diffusers" and root in policy.diffusers_roots | policy.shared_roots:
        if prefix == _PEFT_PREFIX or (not prefix and policy.bare_prefix):
            return _DIFFUSERS_PREFIX
    elif namespace == "native" and root in policy.native_roots | policy.shared_roots:
        if (not prefix and policy.bare_prefix) or (prefix == _PEFT_PREFIX and policy.peft_to_native):
            return _NATIVE_PREFIX
    return prefix


def _unflatten(flat: str, families: tuple[str, ...] | None) -> str | None:
    """Return the dotted module path that the Kohya name *flat* spells, or None when it spells none, or more than one.

    Args:
        flat: The Kohya name without ``lora_unet_``.
        families: The families whose module names to match; None matches every family's.
    """
    tokens = flat.split("_")
    numbers = [position for position, token in enumerate(tokens) if token.isdecimal()]
    found: set[str] = set()
    for chosen in product((False, True), repeat=len(numbers)):
        marked = {position for position, is_block in zip(numbers, chosen, strict=True) if is_block}
        spelling = "_".join("{n}" if position in marked else token for position, token in enumerate(tokens))
        for template, owners in _flat_index().get(spelling, ()):
            if families is None or owners.intersection(families):
                counts = iter(tokens[position] for position in sorted(marked))
                found.add(re.sub(r"\{n\}", lambda _m: next(counts), template))
    return found.pop() if len(found) == 1 else None


_FLAT_INDEX: dict[str, list[tuple[str, frozenset[str]]]] = {}


def _flat_index() -> dict[str, list[tuple[str, frozenset[str]]]]:
    """Return the flattened spelling of each module template in :data:`LORA_MODULES` -> ``[(template, families that have it)]``."""
    if not _FLAT_INDEX:
        owners: dict[str, set[str]] = {}
        for family, templates in LORA_MODULES.items():
            for template in templates:
                owners.setdefault(template, set()).add(family)
        for template, families in owners.items():
            _FLAT_INDEX.setdefault(template.replace(".", "_"), []).append((template, frozenset(families)))
    return _FLAT_INDEX
