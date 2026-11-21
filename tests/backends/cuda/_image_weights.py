"""Synthetic component totals of the CUDA image models, whose sizes are the ones the memory bench measured."""

from __future__ import annotations

from zvisiongenerator.utils.model_memory import WeightTotals

_GIB = 1024**3
_NF4_PER_WEIGHT = 4.13 / 8

# GiB per model: (text encoder NF4, transformer NF4, text encoder FP8, transformer FP8, text encoder bf16, transformer bf16, VAE).
_COMPONENT_GIB = {
    "zit": (2.472, 2.961, 3.746, 5.733, 7.492, 11.464, 0.156),
    "klein4b": (2.472, 1.863, 3.746, 3.609, 7.492, 7.219, 0.157),
    "klein9b": (4.499, 4.365, 7.049, 8.455, 14.097, 16.910, 0.157),
}
FAMILIES = {"zit": ("zimage", None), "klein4b": ("flux2_klein", "4b"), "klein9b": ("flux2_klein", "9b")}


def _elements(gib: float) -> int:
    return round(gib * _GIB)


def _component(nf4: float, fp8: float, bf16: float, *, embedding: bool = False) -> WeightTotals:
    """Return totals whose NF4, FP8 and bfloat16 sizes are the given GiB: matrices, other floats and, for a text encoder, its embedding table."""
    matrices = bf16 - fp8
    other = (fp8 - matrices) / 2
    table = max(0.0, (nf4 - _NF4_PER_WEIGHT * matrices - 2 * other) / (2 - _NF4_PER_WEIGHT)) if embedding else 0.0
    return WeightTotals(float_matrix_elements=_elements(matrices), float_other_elements=_elements(other), embedding_elements=_elements(table))


def image_components(model: str) -> dict[str, WeightTotals]:
    """Return the component totals of *model* (``zit``, ``klein4b`` or ``klein9b``)."""
    te4, tr4, te8, tr8, te16, tr16, vae = _COMPONENT_GIB[model]
    return {"text_encoder": _component(te4, te8, te16, embedding=True), "transformer": _component(tr4, tr8, tr16), "vae": WeightTotals(float_other_elements=_elements(vae / 2))}
