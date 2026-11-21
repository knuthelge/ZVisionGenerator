"""One memory verdict per job: whether a model, as set up, fits this machine, and where its weights would sit.

:func:`judge` is a pure function of the job, the machine's memory and the model's weight totals. Every surface (the
tag, the popover rows, the CLI warnings and the loaders) reads its answer; the thresholds it grades with live in :mod:`~zvisiongenerator.utils.memory_grade`.

CUDA image jobs are estimated from allocated peaks (the memory tensors hold) per placement. MLX image jobs are the model's weights
and working memory at 1 MP plus size, reference and picture terms of their own, fitted to the process's peak footprint. Every CUDA figure carries one offset (:mod:`~zvisiongenerator.utils.memory_offsets`), added here once; a Mac
figure carries none. Nothing else is added: the status bands are the only margin.

The terms of the image estimates come from the memory bench (``bench/memory``). Each carries a mark: *measured*, or *not measured*
where no bench case isolates it. An unmeasured case reads the same model's own terms (those of its other level or size), else the generic estimate
(the weights in their placement plus a generic working memory); it is never read as 0.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace

from zvisiongenerator.utils.h3_memory import (
    estimate_h3_cuda_clip_memory,
    estimate_h3_mlx_clip_memory,
    estimate_h3_mlx_seconds,
    h3_expected_seconds,
    h3_nominal_references,
    h3_upscale_keeps_transformer,
)
from zvisiongenerator.utils.memory_grade import CUDA_GPU_RATIOS, CUDA_RAM_RATIOS, FITS, MAC_RATIOS, TIGHT, TOO_LARGE, grade
from zvisiongenerator.utils.memory_offsets import CUDA_OFFSET_BYTES
from zvisiongenerator.utils.model_memory import (
    GIB,
    LtxCudaMemoryEstimate,
    LtxMlxWeights,
    MemoryBudget,
    WeightTotals,
    estimate_ltx_cuda_clip_memory,
    image_component_bytes,
    ltx_mlx_clip_bytes,
    mlx_image_bytes,
    level_component_bytes,
)

__all__ = [
    "ImageTerms",
    "MemoryJob",
    "MemoryVerdict",
    "MemoryWeights",
    "RESIDENT",
    "STREAMED",
    "TURNS",
    "capped_reference_megapixels",
    "cuda_image_gpu_bytes",
    "cuda_image_placement",
    "image_placement_candidates",
    "judge",
]

_RANK = {FITS: 0, TIGHT: 1, TOO_LARGE: 2}

# MP is decimal. Every size term is anchored at 1024x1024, the size of the runs the constants were measured at.
_ANCHOR_MP = 1024 * 1024 / 1e6

_NF4_LEVEL = 4
_FP8_LEVEL = 8
_UNQUANTIZED_BITS = 16

# The placements of a CUDA image model's weights, as :func:`cuda_image_placement` names them.
RESIDENT, TURNS, STREAMED = "resident", "turns", "streamed"
_UNIFIED = "unified"
_KEPT = "kept"

# RAM working term above the weights held in RAM, per placement (GiB), each the middle of the span of its measured rows (the least and the most it needed), checked against the simultaneous peak of the process's
# anonymous plus shared pages (the highest sum within one sample) in bench runs ``cuda-r10``, ``cuda-r11`` and ``cuda-r11-turns`` (a 10 GB card, 30.3 GB of RAM). Turns: the working memory above the held
# weights was 1.94 (Klein 9B q4, 1 MP), 1.98 (Klein 4B q4, 1 MP), 2.09 (Klein 9B q4, 1 MP with three references), 3.01 (Krea 2 q4, 1 MP), 3.24 (Z-Image q4, 1 MP) and 3.71 GiB (Z-Image q4, 1600x1600), so the
# middle of the span is 2.83. Streamed (the generic figure of the q8 families; Krea 2 has its own below): 1.95 (Z-Image q8, 1 MP), 2.05 (Klein 4B q8, 1 MP), 2.39 (Klein 9B q8, 1 MP) and 2.42 (Z-Image q8,
# 1600x1600), so the middle is 2.19. Resident: 1.54 (Z-Image q4) and 1.66 and 1.68 (Klein 4B q4, plain and forced), all at 1 MP, so the middle is 1.61; no resident row at a larger size is measured. measured.
_RAM_WORKING_GIB = {STREAMED: 2.19, TURNS: 2.83, RESIDENT: 1.61}
# What a user LoRA file adds to the process's RAM, per byte of the file, per placement. Fitted through the origin to the pairs of bench run ``cuda-r15`` (a LoRA case against its plain case, the same process
# peak: anonymous plus shared pages in one sample): Klein 9B with a 0.15 GiB LoRA and a 1.02 GiB LoKr file, +0.02 and +1.07 GiB when it takes turns (q4, factor 1.03) and -0.02 and +1.02 when it streams (q8,
# factor 0.97; its plain row is ``klein9b-m`` of ``cuda-r12``), so both are 1.0: the adapter's tensors sit in system memory (the streamed ones pinned, so shared) beside the weights. A resident job loads them to the card and keeps a share: Z-Image q4 with a
# 0.33 GiB LoRA held +0.13 GiB (0.4; one pair, whose difference is small beside the run-to-run noise of about 0.1 GiB). measured. A file's size, not its rank, sets it. MOSS-Image keeps its LoRAs on the card
# (``moss_lora``) and held none: moss-sft q4 with a 1.1 GiB LoRA read -0.11 GiB, so :func:`_judge_moss` counts none.
_RAM_LORA_SHARE = {STREAMED: 1.0, TURNS: 1.0, RESIDENT: 0.4}
# Unquantized weights that stream are mapped from disk, so only part of them is resident at the peak. Klein 9B bf16 held 0.74 (1 MP) and 0.86
# (1600x1600) of its weights above the working term in the rows of bench run ``cuda-r9`` (0.62 and 0.80 in the run before): the resident share of a
# mapped file varies from run to run and with the picture size, so the middle of the two newest rows is the figure and one value cannot match both
# rows exactly. Klein 9B bf16 held 0.81 of its weights above the working term at 1 MP in bench run ``cuda-r10`` (mapped pages counted). measured, to be re-measured on a machine with more RAM than the weights.
_MAPPED_RESIDENT_SHARE = 0.80
# Krea 2 at q8 streams its text encoder as well as its transformer, and its process held 19.51 GiB at 1 MP in bench run ``cuda-r10`` (19.47 in ``cuda-te``; Windows read 20.94), 0.84 over the
# generic streamed figure that the other families fit: the family's own working term replaces the generic one. measured.
_STREAMED_Q8_RAM_WORKING_GIB = {"krea2": 3.03}

# Ideogram 4 streams its transformer at every level. The base is the least memory its denoise needs at 1 MP: 5.86 at q4 (its NF4 text encoder
# turn: encoder 4.88 + VAE 0.16 + encode 0.82), 5.71 at q8 and bf16 (the streamed transformer). Above 1 MP the denoise grows by the slope per extra
# megapixel. A level's slope is the line through its 1 MP row and the allocated memory its largest size had reached when it ran out of
# memory (the failed 2.39 GiB allocation included), which is the least the job needs: 1600x1600 stopped at 8.73 (q4), 8.75 (q8) and 8.99 GiB (bf16)
# in bench run ``cuda-r9``, so at 1600x1600 q4 and q8 read Too large on the 10 GB card. The 1408x1408 rows ran at 7.14 (q4), 7.43 (q8) and 7.90 GiB (bf16),
# below the lines. measured.
_IDEOGRAM_BASE_GIB = {4: 5.86, 8: 5.71, None: 5.71}
_IDEOGRAM_SLOPE_GIB = {4: 1.90, 8: 2.01, None: 2.17}
# Process RAM (GiB), fixed per level. q4 and q8 are the highest anonymous plus shared memory the process held in one sample; bf16 is that plus the mapped
# weights resident at the peak (7.87 + 17.83 GiB at 1 MP; the two larger sizes ran out of GPU memory early). q4 is the middle of its 1 MP and 1408x1408 rows
# in bench run ``cuda-r10`` (11.84 and 11.79 GiB); q8 held 15.82 GiB and bf16 25.70 GiB at 1 MP in the same run. measured.
_IDEOGRAM_RAM_GIB = {4: 11.8, 8: 15.8, None: 25.7}
# What a user LoRA file adds to the process's RAM, per byte of the file. Ideogram 4 streams or takes turns at every level and the adapter's tensors sit in system memory beside
# the weights, but only part of the file counts. Fitted through the origin to the pairs of bench run ``cuda-r16`` (a LoRA case against its plain case, cold, anonymous plus shared
# pages in one sample): a 1.05 GiB LoKr file added +0.70 GiB at q8 (ideo-m-lora, factor 0.67) and +0.95 GiB at q4 (ideo-q4-m-lora, factor 0.91), so 0.79; the two levels sit within
# the run-to-run noise of about 0.1 GiB of each other's line, and bf16 is not measured (it takes the same share). measured. The GPU figure takes nothing: the allocated peak
# was the same to the byte with and without the file (5.71 GiB at q8, 5.86 at q4), so the file adds nothing on the card at the job's peak.
_IDEOGRAM_RAM_LORA_SHARE = 0.79
# MOSS-Image holds the VAE and the Turbo LoRA on the GPU throughout; its largest phase above them is 5.48 (the text encode, 6.82 GiB allocated) at every
# size and level up to 2048x2048. On a free card the FP32 VAE decode of a 1600x1600 or 2048x2048 picture reaches 7.47-7.51 GiB (bench run ``cuda-r9``, the
# allocator's expandable segments active; 6.82 in ``cuda-r8`` without them), because the decode takes more memory when the card has memory free; with 1.5 GiB held the
# decode peaks at 6.07, under the encode. The figure is the least the job runs on: moss-q4-xl ran on a card of 7.29 GiB and ran out of memory on 7.04 (bench runs ``cuda-r9``
# and ``cuda-r9b``, holds of 1.5, 1.75 and 2.0 GiB), a figure of 7.16 against 7.22 here. measured.
_MOSS_WORKING_GIB = 5.48
# MOSS-Image process RAM (GiB) per level, the middle of each level's span of rows in the bench runs named (a 10 GB card, 30.3 GB of RAM). q4 and q8 are the simultaneous peak of anonymous plus shared
# pages in ``cuda-r11`` at 1 MP, 1600x1600 and 2048x2048: 11.30, 11.89 and 12.22 at q4 (middle 11.76) and 12.88, 13.27 and 13.68 at q8 (middle 13.28). Unquantized (None) maps its transformer from disk
# and reads the text encoder it quantizes at load from disk, so its figure counts the file-backed pages as well (``rss_anon_shmem_file_peak_bytes``), as Klein 9B's and Ideogram 4's unquantized figures
# do, and is never read above Tight. ``cuda-r15`` held 16.82, 16.61 and 16.95 GiB in one sample at 1 MP, 1600x1600 and 2048x2048 (VmHWM 16.85, 16.64 and 17.03), the middle of the span is the figure. The
# 1 MP and 2048x2048 cases had first read 18.22 and 14.18 (VmHWM 18.36 and 14.30), a miss that ran again once and was refitted: the share of the mapped files resident at the peak varies from run to run and not with
# the size, and the first pair, taken while other jobs were loading models on the machine, sat at the two ends of it. The GPU needs no term of its own: unquantized peaked at 6.82, 7.47 and 7.51 GiB allocated at
# those sizes, the figures of q4 and q8. measured.
_MOSS_RAM_GIB = {4: 11.76, 8: 13.28, None: 16.8}


@dataclass(frozen=True)
class ImageTerms:
    """The activation terms of one CUDA image model, in GiB: the phase constants and the per-megapixel and per-reference slopes.

    E, R, D and K are the encode, reference-encode, denoise and decode working memory at 1 MP above the weights of their
    phase; S is the streamed denoise at 1 MP. J and F are the resident and turn-taking denoise slopes per megapixel above
    1 MP and per reference megapixel; Js and Fs are the streamed ones. A model that takes no references has no R, F or Fs.
    Ks is the streamed job's decode peak above the VAE on a free card (the decode takes less under a hold, and no streamed row is probed under one);
    a model whose S already sits at or above it leaves Ks at 0.
    """

    encode: float  # E
    reference_encode: float  # R
    denoise: float  # D
    decode: float  # K
    streamed: float  # S
    per_megapixel: float  # J
    per_reference: float  # F
    streamed_per_megapixel: float  # Js
    streamed_per_reference: float  # Fs
    streamed_decode: float = 0.0  # Ks


# Each row is what the bench measured of that model (runs ``cuda-alloc``, ``cuda-alloc2``, ``r7-*``, ``cuda-r9``, ``cuda-r15`` and ``cuda-r17``, a 10 GB card). A term marked not measured is one no row isolates, because the phase that
# peaks hides it; it is left at 0 where it does not bind in any placement.
# K is the VAE decode's working memory at its floor: the decode takes less when the card is tight (the Klein 4B resident job's decode peaked 3.38 GiB above its weights on a free
# card and 2.38 with 0.5 GiB held; Z-Image's 2.38 and 2.00; per-phase peaks, bench probe of ``cuda-r9``), so K is read from the hold bracket of the resident job and not from its peak on a free card:
# the card it ran out of memory on and the one it ran on, 6.35 and 6.47 GiB (Klein 4B q4, ``cuda-r17``) and 7.29 and 7.79 (Z-Image q4), middle less the CUDA offset and the weights.
# Klein 9B is measured throughout; its VAE is the 4B's, and its K is the 4B's of ``cuda-r9`` (1.65, a bracket of 6.29..6.79 GiB), which ``cuda-r17`` narrowed for the 4B only. S is the streamed job's peak on a free card, which for Klein 9B is
# the decode too (not probed under a hold); the 4B's S is its denoise and its decode is Ks.
_KLEIN_9B = ImageTerms(0.78, 1.78, 0.91, 1.65, 3.40, 0.64, 0.67, 0.27, 0.09)
_IMAGE_TERMS: dict[str, ImageTerms] = {
    "flux2_klein:9b": _KLEIN_9B,
    # Klein 4B: every term is measured, in bench run ``cuda-r17`` (a 10 GB card) with the per-phase allocated peaks of a probe in the case's own process (``bench/memory/runs/image/cuda-r17/phases-klein4b.json``). E: the text encode peaked at 2.825 GiB above
    # 2.629 of weights (turns, 1 MP), 0.20. R: the reference encode peaked 1.78 above the resident weights (1 reference) and 1.80 above the encoder and the VAE (4 references); it does not grow with
    # the count. D: the denoise phase peaked 0.65 above the transformer's start at 1 MP (0.64 in ``cuda-r9``). K: the resident job's hold bracket at 1 MP, 6.35 and 6.47 GiB of card (middle 6.41), less
    # the CUDA offset and the weights 4.49, 1.52 (the earlier bracket of ``cuda-r9`` was 6.29..6.79, 1.65). J and F: five hold brackets, middle less the CUDA offset and the weights, above 0.64: resident 2048x2048
    # 7.04 (2.15 above its weights), resident 1 MP with four references 7.54 (2.65), resident 1600x1600 with four references 8.29 (3.40), turns 2048x2048 4.54 (2.12) and turns 1600x1600 with four references
    # 5.79 (3.37); least squares through them gives J 0.476 and F 0.481, and the probe's phase peaks give 0.49 and 0.49, so J and F are 0.48 (resident and turns share them). Streamed (q8 and unquantized): the decode
    # on a free card is the peak up to 1600x1600 and 2048x2048 alone, and at 1 MP with four references (3.55..3.57 GiB allocated, Ks 3.40 above the VAE's 0.157); the denoise outgrows it at 1600x1600 with four references
    # (3.98) and at 2048x2048 with four references (4.76). The two rows fix S, the denoise at 1 MP, at 1.08 each (Js and Fs are J and F: the activations are the same whatever holds the weights), so the check is that they agree.
    "flux2_klein:4b": ImageTerms(0.20, 1.78, 0.64, 1.52, 1.08, 0.48, 0.48, 0.48, 0.48, 3.40),
    # Z-Image: D and J are the two turn-taking rows (3.60 GiB at 1 MP, 4.28 at 1600x1600, bench run r7-turns) read as the transformer's turn; K and S are measured (resident
    # 7.29..7.79 GiB of card at the floor, streamed 3.55, flat with size). E is not measured: the text encoder's turn sits under the transformer's.
    "zimage": ImageTerms(0.0, 0.0, 0.48, 1.55, 3.39, 0.45, 0.0, 0.0, 0.0),
    # Krea 2: D, J, S and Js are measured (turns 7.16 and 7.92 GiB, streamed 2.44 and 3.21, at 1 MP and 1600x1600). E and K are measured from the per-phase allocated peaks of a taking-turns q4
    # job in bench run ``cuda-r15`` (the probe of ``cuda-r9``, ``bench/memory/runs/image/cuda-r15/phases-krea2.json``): the text encode peaked at 3.165 GiB above 2.910 of weights (E 0.26 with the
    # NF4 encoder and the VAE as the weights), and the VAE decode, run with the transformer already off the card, at 0.544 GiB at 1 MP and 0.560 at 1600x1600 against the VAE's 0.236 (K 0.31 and 0.32, middle 0.32).
    # The decode peak was the same with 1.5 GiB held (1 MP) and flat with the size, so unlike Klein's it has no floor below the free-card peak to probe. Neither term binds: the denoise phase is the peak
    # at every placement, so the estimates are the same as with both at 0 (they were not measured then).
    "krea2": ImageTerms(0.26, 0.0, 0.76, 0.32, 2.20, 0.50, 0.0, 0.50, 0.0),
}
# A model with no row (FLUX.1, FLUX.2 user and custom models, an undetected family, a Klein of unknown size) reads the generic estimate: its weights in the
# placement plus the generic working memory in every phase, no size or reference terms. not measured: the CUDA stand-in, which predates the bench.
_GENERIC_WORKING_GIB = 1.5
_DEFAULT_TERMS = ImageTerms(_GENERIC_WORKING_GIB, 0.0, _GENERIC_WORKING_GIB, _GENERIC_WORKING_GIB, _GENERIC_WORKING_GIB, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class _MlxImageTerms:
    """What an MLX image job adds, in GiB, to its model's estimate at 1 MP with no reference (the weights and the working memory).

    The peak is the larger of the size phase and the reference phase. The size phase adds ``per_megapixel`` per megapixel of canvas above
    ``flat_to`` (the peak is flat below it, set by the load phase). The reference phase adds ``reference`` once the job holds any reference
    picture, and ``per_reference`` per megapixel of the references. ``picture`` is added per megapixel of canvas when the job starts from a
    picture (see :attr:`MemoryJob.from_picture`). ``working`` is added to the generic working memory of the 1 MP estimate, for a model
    whose measured working memory differs from it.
    """

    per_megapixel: float
    reference: float = 0.0
    per_reference: float = 0.0
    picture: float = 0.0
    working: float = 0.0
    flat_to: float = _ANCHOR_MP


# MLX terms per model and level, from bench run ``mac-r11`` (32 GB M5, MLX buffer cache off); each entry is measured unless marked. All are fitted
# to the process's peak footprint, through the middle of the model's rows. Where a model's rows are flat and then rising (Klein 9B), the size
# phase is flat up to ``flat_to`` and rises from there. A term the model's own rows do not isolate is either the same model's measured term at
# another level (a reference or picture term, which belongs to the model's pipeline, not to the level) or 0; both are marked not measured.
_MLX_IMAGE_TERMS: dict[tuple[str, int | None], _MlxImageTerms] = {
    # Klein 4B q4 is fitted to the footprints of bench runs ``mac-r11`` (1024x1024 5.71 GiB, 1600x1600 9.18) and ``mac-r12`` (a 32 GB M5-class Mac, the 2048x2048 rows and the references). The
    # peak rises from 1 MP (no flat part), so the size line is a least-squares fit through 1024x1024, 1600x1600 and 2048x2048 (5.71, 9.18, 13.57): 2.50 GiB per megapixel above 1 MP, working -0.04
    # (the rows sit within 0.21 GiB of it). A job with references peaks in the reference phase, fitted through the four reference rows of ``mac-r12`` above the size line's 1 MP figure
    # (4.06, 4.96 and 5.16 GiB for one, three and four references at 1024x1024, 5.85 for three at 1600x1600): 3.83 GiB plus 0.41 per reference megapixel. An Extend adds the picture
    # term: the app encodes the source at the canvas size itself before the loop (the model's own reference is a second encode), and the peak sits at the 2048x2048 level at 1.48 MP
    # (13.77 GiB for ``klein4b-q4-extend-right512``, 4.06 GiB above the one-reference peak, which the model reads as 2.65 GiB per canvas megapixel). A whole-picture edit is not from a picture: the 2048x2048
    # edit (13.59) peaks where the plain 2048x2048 draw does (13.57). The q8 row takes the same reference and picture terms (the model's pipeline, not its level; not measured at q8).
    ("flux2_klein:4b", 4): _MlxImageTerms(per_megapixel=2.50, reference=3.83, per_reference=0.41, picture=2.65, working=-0.04),
    ("flux2_klein:4b", 8): _MlxImageTerms(per_megapixel=1.15, reference=3.83, per_reference=0.41, picture=2.65, working=-0.07),
    # Klein 9B q4: the peak is flat to 1.8 MP and rises from there (1408x1408 sits on the flat part, 1600x1600 and 2048x2048 on the rise).
    ("flux2_klein:9b", 4): _MlxImageTerms(per_megapixel=2.55, reference=2.16, per_reference=0.98, picture=1.33, working=-0.07, flat_to=1.8),
    # Klein 9B q8: the peak is flat from 1024x1024 to 1600x1600 (the load phase holds the most) and the 2048x2048 edit rises above it; the rise
    # is that one row's, measured above 1600x1600 only there.
    ("flux2_klein:9b", 8): _MlxImageTerms(per_megapixel=1.63, per_reference=0.14, picture=1.08, working=-0.10, flat_to=2.56),
    # Unquantized: flat from 1024x1024 to 1600x1600 (weights fill the memory the working set would take). The rise above 1600x1600, the reference
    # and the picture terms are q8's (the same model's), not measured.
    ("flux2_klein:9b", None): _MlxImageTerms(per_megapixel=1.63, per_reference=0.14, picture=1.08, working=-0.83, flat_to=2.56),
    ("zimage", 4): _MlxImageTerms(per_megapixel=2.16, picture=4.05, working=0.42),  # picture: q8's, not measured
    ("zimage", 8): _MlxImageTerms(per_megapixel=0.99, picture=4.05, working=0.29),  # picture: an upscaled 1600x1600 peaked 10.4 GiB above the plain one
    ("krea2", 4): _MlxImageTerms(per_megapixel=7.85, working=0.30),  # picture: not measured (0)
    ("krea2", 8): _MlxImageTerms(per_megapixel=7.45, working=-0.14),
    # Ideogram 4 takes neither references nor a start picture. Each level's line is fitted to its rows at 1024x1024, 1408x1408 and 1600x1600;
    # ``working`` is the level's distance from the constant of ``model_memory`` (its q8 figure).
    ("ideogram4", 4): _MlxImageTerms(per_megapixel=1.75, working=0.26),
    ("ideogram4", 8): _MlxImageTerms(per_megapixel=1.75),
    ("ideogram4", None): _MlxImageTerms(per_megapixel=2.11, working=-0.23),
}
# A model with no row at all reads its weights and the generic working memory, with no terms. not measured.
_NO_MLX_TERMS = _MlxImageTerms(0.0)


@dataclass(frozen=True)
class MemoryJob:
    """What a model is asked to do: the setup the verdict judges.

    Image jobs use *family*, *quantize*, the size and the references; video jobs use *family* (``"ltx"`` or ``"h3"``),
    *low_memory*, the size, *frames* and the video fields; audio jobs need only *media*.
    """

    media: str  # "image", "video" or "audio"
    family: str | None = None  # the model's family: an image family, or "ltx" / "h3" for video
    model_size: str | None = None  # Klein's "4b" or "9b"
    quantize: int | None = None  # the image level: None (unquantized), 4 or 8
    low_memory: bool = False  # LTX: whether the transformer streams to the GPU instead of staying there
    width: int = 1024  # the size the model draws
    height: int = 1024
    reference_megapixels: tuple[float, ...] = ()  # image: one entry per reference picture, an edit's picture included
    lora_bytes: int = 0  # the total size of the user LoRA files
    frames: int = 0  # video: the frame count the model generates
    mode: str = "t2v"  # H3: "t2v", "i2v" or "extend"
    references: int = 0  # H3: how many reference images the clip holds
    from_picture: bool = False  # image on a Mac: whether the model starts from a picture at the canvas size (img2img, an upscale's refine, a painted edit or an Extend)
    upscale: bool = False  # video: whether the clip is upscaled 2x (width and height are the enlarged size)
    review: bool = False  # H3 on CUDA: whether the upscale pauses to review the half-size render
    forwards: int = 8  # H3: the transformer forwards of the run, which set the expected time

    @property
    def megapixels(self) -> float:
        """The drawn size in decimal megapixels."""
        return self.width * self.height / 1e6


@dataclass(frozen=True)
class MemoryWeights:
    """The model's weights as the estimators read them, which the caller resolves from local headers or a shipped table."""

    components: Mapping[str, WeightTotals] = field(default_factory=dict)  # image: each component's totals
    stored_bits: int | None = None  # image: the level the weights are stored at, which wins over the requested one
    turbo_lora_bytes: int = 0  # MOSS-Image's built-in Turbo LoRA
    ltx_cuda: LtxCudaMemoryEstimate | None = None  # LTX on CUDA: the system memory of its weights
    ltx_mlx: LtxMlxWeights | None = None  # LTX on macOS: the bytes of its components
    audio_peak_bytes: int | None = None  # audio: the measured GPU peak on this platform


@dataclass(frozen=True)
class MemoryVerdict:
    """Whether a job fits, and what it needs.

    ``gpu_bytes`` is the need compared with the card: on CUDA the allocated peak of the job plus the CUDA offset; on a Mac the whole
    unified-memory need (fitted to the process's peak footprint), and ``ram_bytes`` is ``None``, as it is for audio on CUDA and for an
    image level whose RAM was not measured.
    """

    gpu_bytes: int
    ram_bytes: int | None
    placement: str
    status: str  # "fits", "tight" or "too_large": the worse of the GPU and the RAM status
    resource: str  # "gpu", "ram" or "unified": what set the status; on a tie, the GPU
    expected_seconds: int | None = None  # H3 only


def judge(job: MemoryJob, machine: MemoryBudget, weights: MemoryWeights) -> MemoryVerdict:
    """Return whether *job* fits *machine*, given the model's *weights*.

    Args:
        job: The job as set up.
        machine: The memory this machine offers: the card and the RAM on CUDA, the total unified memory on a Mac.
        weights: What the model's files hold; only the fields the job's media and platform read are needed.

    Raises:
        ValueError: If the media is unknown, the weights lack what the job needs, a level is not 4, 8 or ``None``, or an H3 canvas or mode is invalid.
    """
    if job.media == "audio":
        return _judge_audio(machine, weights)
    if job.media == "video":
        return _judge_video(job, machine, weights)
    if job.media == "image":
        return _judge_image(job, machine, weights)
    raise ValueError(f"Unknown media {job.media!r}; expected image, video or audio.")


def image_placement_candidates(quantize: int | None) -> tuple[str, ...]:
    """Return the CUDA placements an image level tries, in order: NF4 weights stay on the GPU or take turns; others stream.

    Only q4 has a measured resident run. The q8 and unquantized models keep streaming from system memory until a run on a 16 to
    24 GB card backs a resident placement for them, and the loader forces only q4 placements. Turns join the second list only
    where a bench run shows them faster than streaming; none does yet.
    """
    _check_level(quantize)
    return (RESIDENT, TURNS) if quantize == _NF4_LEVEL else (STREAMED,)


def cuda_image_gpu_bytes(
    placement: str,
    components: Mapping[str, WeightTotals],
    quantize: int | None,
    *,
    family: str | None,
    model_size: str | None = None,
    lora_bytes: int = 0,
    job_megapixels: float,
    reference_megapixels: float = 0.0,
) -> int:
    """Return the GPU bytes of a CUDA image job in one *placement*, before the CUDA offset: the largest of its phases.

    Phases: *encode* and *reference encode* (turns), *denoise* (all) and *decode* (all). Weights count in their placement's
    format and the LoRA bytes sit with the transformer; the activation terms apply to the denoise phase only.

    Args:
        placement: ``"resident"``, ``"turns"`` or ``"streamed"``.
        components: Each component's totals.
        quantize: The level the model loads at.
        family: The image family, which picks the terms; a model with no row reads the generic estimate.
        model_size: Klein's ``"4b"`` or ``"9b"``.
        lora_bytes: The total size of the user LoRA files.
        job_megapixels: The size the model draws, in decimal megapixels.
        reference_megapixels: The references' total after each is capped at 1024x1024.

    Raises:
        ValueError: If *placement* or *quantize* is unknown.
    """
    if placement not in (RESIDENT, TURNS, STREAMED):
        raise ValueError(f"Unknown image placement {placement!r}; expected resident, turns or streamed.")
    terms = _terms_for(family, model_size)
    delta = max(0.0, job_megapixels - _ANCHOR_MP)
    vae = _vae_gib(components)
    text_encoder, transformer = _level_gib(components, quantize)
    lora = lora_bytes / GIB
    if placement == STREAMED:
        denoise = terms.streamed + terms.streamed_per_megapixel * delta + terms.streamed_per_reference * reference_megapixels
        return _bytes(vae + max(terms.decode, terms.streamed_decode, denoise))
    denoise = terms.denoise + terms.per_megapixel * delta + terms.per_reference * reference_megapixels
    if placement == RESIDENT:
        # Every weight is on the card in each phase, so the reference encode (working memory above the weights present) joins the max.
        reference_encode = terms.reference_encode if reference_megapixels > 0 else 0.0
        return _bytes(text_encoder + transformer + lora + vae + max(terms.decode, denoise, reference_encode))
    phases = [text_encoder + vae + terms.encode, transformer + lora + vae + denoise, vae + terms.decode]
    if reference_megapixels > 0 and terms.reference_encode:
        # The VAE sits on the GPU outside the offload chain, so it encodes the references while the text encoder is still there.
        phases.append(text_encoder + vae + terms.reference_encode)
    return _bytes(max(phases))


def cuda_image_placement(
    components: Mapping[str, WeightTotals],
    quantize: int | None,
    gpu_budget_bytes: int,
    *,
    family: str | None,
    model_size: str | None = None,
    lora_bytes: int = 0,
    job_megapixels: float,
    reference_megapixels: float = 0.0,
) -> tuple[str, int]:
    """Return the CUDA placement of an image job and its GPU bytes before the CUDA offset.

    Each level tries its candidates in order (:func:`image_placement_candidates`) and takes the first whose estimate plus the
    CUDA offset fits the card (the fits line of :func:`~zvisiongenerator.utils.memory_grade.grade`). When none does, it takes
    the one that needs the least GPU memory. The tag and the loader share this choice.

    Args:
        components: Each component's totals.
        quantize: The level the model loads at.
        gpu_budget_bytes: The card's budget: its total less the driver's reserve.
        family: The image family.
        model_size: Klein's ``"4b"`` or ``"9b"``.
        lora_bytes: The total size of the user LoRA files.
        job_megapixels: The size the model draws, in decimal megapixels.
        reference_megapixels: The references' total after each is capped at 1024x1024.
    """
    candidates = image_placement_candidates(quantize)
    sized = [
        (
            placement,
            cuda_image_gpu_bytes(
                placement, components, quantize, family=family, model_size=model_size, lora_bytes=lora_bytes, job_megapixels=job_megapixels, reference_megapixels=reference_megapixels
            ),
        )
        for placement in candidates
    ]
    for placement, need in sized:
        if grade(need + CUDA_OFFSET_BYTES, gpu_budget_bytes, CUDA_GPU_RATIOS) == FITS:
            return placement, need
    return min(sized, key=lambda candidate: candidate[1])


def _terms_for(family: str | None, model_size: str | None) -> ImageTerms:
    """Return the terms of an image model: Klein keys on its size, and a model with no row gets the generic estimate."""
    key = f"flux2_klein:{model_size}" if family == "flux2_klein" else family or ""
    return _IMAGE_TERMS.get(key, _DEFAULT_TERMS)


def _mlx_terms_for(family: str | None, model_size: str | None, level: int | None) -> _MlxImageTerms:
    """Return the MLX terms of an image model at a level: Klein keys on its size.

    A level the model has no row for reads the size, reference and picture terms of the model's nearest level (unquantized counts as 16
    bits; the smaller level on a tie), which are the model's own and grow with the job; its ``working`` adjustment of the 1 MP base
    belongs to that level's load phase and does not carry. A model with no row at all has no terms.
    """
    key = f"flux2_klein:{model_size}" if family == "flux2_klein" else family or ""
    levels = [listed for model, listed in _MLX_IMAGE_TERMS if model == key]
    if not levels:
        return _NO_MLX_TERMS
    if level in levels:
        return _MLX_IMAGE_TERMS[(key, level)]
    bits = level or _UNQUANTIZED_BITS
    nearest = min(levels, key=lambda listed: (abs((listed or _UNQUANTIZED_BITS) - bits), listed or _UNQUANTIZED_BITS))
    return replace(_MLX_IMAGE_TERMS[(key, nearest)], working=0.0)


def _check_level(quantize: int | None) -> None:
    if quantize not in (None, _NF4_LEVEL, _FP8_LEVEL):
        raise ValueError(f"Unsupported quantize level {quantize!r}; expected None, 4 or 8.")


def _bytes(gib: float) -> int:
    return round(gib * GIB)


def _vae_gib(components: Mapping[str, WeightTotals]) -> float:
    vae = components.get("vae")
    return vae.bfloat16_bytes / GIB if vae is not None else 0.0


def _level_gib(components: Mapping[str, WeightTotals], quantize: int | None) -> tuple[float, float]:
    """Return the text encoder's and the transformer's GiB in the format their level holds them: NF4, FP8 or bfloat16."""
    _check_level(quantize)
    text_encoder, transformer = components.get("text_encoder"), components.get("transformer")
    return (
        level_component_bytes("text_encoder", text_encoder, quantize) / GIB if text_encoder else 0.0,
        level_component_bytes("transformer", transformer, quantize) / GIB if transformer else 0.0,
    )


def capped_reference_megapixels(megapixels: Iterable[float]) -> float:
    """Return the references' total megapixels, each capped at 1024x1024 where the Klein pipeline caps its condition images."""
    return sum(min(size, _ANCHOR_MP) for size in megapixels)


def _discrete_verdict(
    gpu_bytes: int,
    ram_bytes: int | None,
    machine: MemoryBudget,
    *,
    placement: str,
    ram_mapped: bool = False,
    expected_seconds: int | None = None,
) -> MemoryVerdict:
    """Grade a CUDA job's GPU and RAM needs; the status is the worse, and on a tie the GPU is named.

    The CUDA offset joins the GPU need here, once, whatever the job.

    Args:
        gpu_bytes: The need before the CUDA offset, compared with the card.
        ram_bytes: The need compared with system memory, or ``None`` for none.
        machine: The card and the system memory.
        placement: Where the weights sit.
        ram_mapped: Whether the RAM need is weights mapped from disk, which run past RAM, only slower: never too large.
        expected_seconds: The time estimate to carry.
    """
    gpu_bytes += CUDA_OFFSET_BYTES
    gpu = grade(gpu_bytes, machine.gpu_bytes, CUDA_GPU_RATIOS)
    ram = FITS
    if ram_bytes is not None and machine.system_bytes:
        ram = grade(ram_bytes, machine.system_bytes, CUDA_RAM_RATIOS)
        if ram_mapped and ram == TOO_LARGE:
            ram = TIGHT
    if _RANK[ram] > _RANK[gpu]:
        return MemoryVerdict(gpu_bytes, ram_bytes, placement, ram, "ram", expected_seconds)
    return MemoryVerdict(gpu_bytes, ram_bytes, placement, gpu, "gpu", expected_seconds)


def _unified_verdict(need_bytes: int, machine: MemoryBudget, *, placement: str = _UNIFIED, expected_seconds: int | None = None) -> MemoryVerdict:
    """Grade a Mac job's need, the process's peak footprint the Mac estimators are fitted to (no offset is added), against the machine's total unified memory.

    Raises:
        ValueError: If *machine* does not know its total unified memory (MLX always reports it).
    """
    if not machine.unified_bytes:
        raise ValueError("A Mac verdict needs the machine's total unified memory (unified_bytes).")
    return MemoryVerdict(need_bytes, None, placement, grade(need_bytes, machine.unified_bytes, MAC_RATIOS), "unified", expected_seconds)


def _is_mac(machine: MemoryBudget) -> bool:
    return machine.system_bytes is None


def _judge_audio(machine: MemoryBudget, weights: MemoryWeights) -> MemoryVerdict:
    if weights.audio_peak_bytes is None:
        raise ValueError("An audio verdict needs the model's measured GPU peak (audio_peak_bytes).")
    if _is_mac(machine):
        return _unified_verdict(weights.audio_peak_bytes, machine, placement=RESIDENT)
    return _discrete_verdict(weights.audio_peak_bytes, None, machine, placement=RESIDENT)


def _judge_video(job: MemoryJob, machine: MemoryBudget, weights: MemoryWeights) -> MemoryVerdict:
    if job.family == "h3":
        return _judge_h3(job, machine)
    if job.family != "ltx":
        raise ValueError(f"Unknown video family {job.family!r}; expected ltx or h3.")
    if _is_mac(machine):
        if weights.ltx_mlx is None:
            raise ValueError("An LTX verdict on macOS needs the pack's component bytes (ltx_mlx).")
        mlx = weights.ltx_mlx if job.upscale else replace(weights.ltx_mlx, upsampler=None)
        need = ltx_mlx_clip_bytes(mlx, width=job.width, height=job.height, frames=job.frames, low_memory=job.low_memory, unified_bytes=machine.unified_bytes, lora_bytes=job.lora_bytes, mode=job.mode)
        return _unified_verdict(need, machine)
    if weights.ltx_cuda is None:
        raise ValueError("An LTX verdict on CUDA needs the weights' system memory (ltx_cuda).")
    estimate = estimate_ltx_cuda_clip_memory(weights.ltx_cuda, width=job.width, height=job.height, frames=job.frames, low_memory=job.low_memory, upscale=job.upscale, lora_bytes=job.lora_bytes)
    return _discrete_verdict(estimate.gpu_bytes, estimate.system_bytes, machine, placement=STREAMED if job.low_memory else RESIDENT)


def _judge_h3(job: MemoryJob, machine: MemoryBudget) -> MemoryVerdict:
    sizes = h3_nominal_references(job.references, job.width, job.height)
    if _is_mac(machine):
        # User LoRAs add nothing here: the backend combines them into one adapter file that the fork's provider streams from disk a block at a
        # time (``load_block`` in ``packages/minimax_h3_mlx/native_turbo_lora.py``), so only one block's share is ever in memory.
        need = estimate_h3_mlx_clip_memory(job.width, job.height, job.frames, mode=job.mode, references=sizes)
        seconds = estimate_h3_mlx_seconds(job.width, job.height, job.frames, forwards=job.forwards, mode=job.mode, references=sizes)
        return _unified_verdict(need, machine, expected_seconds=seconds)
    # An upscale without review keeps the transformer loaded between its passes where the card holds that.
    keep = job.upscale and not job.review and h3_upscale_keeps_transformer(job.width, job.height, job.frames, machine.gpu_bytes, mode=job.mode, loras_bytes=job.lora_bytes, references=sizes)
    estimate = estimate_h3_cuda_clip_memory(job.width, job.height, job.frames, mode=job.mode, loras_bytes=job.lora_bytes, upscale=job.upscale, keep_transformer=keep, references=sizes)
    seconds = h3_expected_seconds(job.width, job.height, job.frames, forwards=job.forwards, mode=job.mode, loras=job.lora_bytes > 0, upscale=job.upscale, keep_transformer=keep, references=sizes)
    return _discrete_verdict(estimate.gpu_bytes, estimate.system_bytes, machine, placement=_KEPT if keep else STREAMED, expected_seconds=seconds)


def _judge_image(job: MemoryJob, machine: MemoryBudget, weights: MemoryWeights) -> MemoryVerdict:
    _check_level(job.quantize)
    if not weights.components:
        raise ValueError("An image verdict needs the model's component totals (components).")
    # A stored quant loads at its own level whatever level is asked for.
    level = weights.stored_bits or job.quantize
    _check_level(level)
    if _is_mac(machine):
        return _unified_verdict(_mac_image_bytes(job, weights, level), machine)
    if job.family == "ideogram4":
        return _judge_ideogram4(job, machine, level)
    if job.family == "moss":
        return _judge_moss(job, machine, weights, level)
    megapixels, references = job.megapixels, capped_reference_megapixels(job.reference_megapixels)
    placement, need = cuda_image_placement(
        weights.components, level, machine.gpu_bytes, family=job.family, model_size=job.model_size, lora_bytes=job.lora_bytes, job_megapixels=megapixels, reference_megapixels=references
    )
    ram_need, mapped = _generic_ram_bytes(weights.components, level, placement, job.lora_bytes, job.family)
    return _discrete_verdict(need, ram_need, machine, placement=placement, ram_mapped=mapped)


def _generic_ram_bytes(components: Mapping[str, WeightTotals], level: int | None, placement: str, lora_bytes: int, family: str = "") -> tuple[int, bool]:
    """Return the RAM need of a CUDA image placement and whether its weights are mapped from disk.

    The need is the weights held in RAM plus the placement's working term and the user LoRAs' share (:data:`_RAM_LORA_SHARE`): nothing
    is held for a resident model, the NF4 pair for one that takes turns, and the text encoder and transformer in their level's format
    when streamed. Unquantized weights that stream are memory-mapped from disk, and only :data:`_MAPPED_RESIDENT_SHARE` of them counts.
    """
    working = _RAM_WORKING_GIB[placement]
    if placement == STREAMED and level == _FP8_LEVEL:
        working = _STREAMED_Q8_RAM_WORKING_GIB.get(family, working)
    lora = lora_bytes / GIB * _RAM_LORA_SHARE[placement]
    if placement == RESIDENT:
        return _bytes(working + lora), False
    text_encoder, transformer = _level_gib(components, level)
    mapped = placement == STREAMED and level is None
    return _bytes((text_encoder + transformer) * (_MAPPED_RESIDENT_SHARE if mapped else 1.0) + lora + working), mapped


def _judge_ideogram4(job: MemoryJob, machine: MemoryBudget, level: int | None) -> MemoryVerdict:
    delta = max(0.0, job.megapixels - _ANCHOR_MP)
    gpu = _bytes(_IDEOGRAM_BASE_GIB[level] + _IDEOGRAM_SLOPE_GIB[level] * delta)
    ram = _IDEOGRAM_RAM_GIB[level] + job.lora_bytes / GIB * _IDEOGRAM_RAM_LORA_SHARE
    return _discrete_verdict(gpu, _bytes(ram), machine, placement=STREAMED, ram_mapped=level is None)


def _judge_moss(job: MemoryJob, machine: MemoryBudget, weights: MemoryWeights, level: int | None) -> MemoryVerdict:
    """Grade a MOSS-Image job: the Turbo and user LoRAs count on the GPU and not in system memory, where the level's measured figure holds (:data:`_MOSS_RAM_GIB`)."""
    vae = weights.components.get("vae")
    vae_bytes = image_component_bytes("vae", vae, None) if vae is not None else 0
    gpu = round(vae_bytes + weights.turbo_lora_bytes + job.lora_bytes + _MOSS_WORKING_GIB * GIB)
    return _discrete_verdict(gpu, _bytes(_MOSS_RAM_GIB[level]), machine, placement=TURNS if level == _NF4_LEVEL else STREAMED, ram_mapped=level is None)


def _mac_image_bytes(job: MemoryJob, weights: MemoryWeights, level: int | None) -> int:
    """Return an image job's unified-memory need on a Mac at *level*: the weights and working memory at 1 MP, plus the larger of the size and reference phases and the picture term.

    LoRAs count nothing except MOSS-Image's, which mflux keeps as separate layers; for every other family it merges them into the
    weights at load. MOSS-Image's peak does not grow with the size, so it has no terms.
    """
    moss = job.family == "moss"
    base = mlx_image_bytes(dict(weights.components), level, family=job.family, lora_bytes=weights.turbo_lora_bytes + job.lora_bytes if moss else 0)
    if moss:
        return base
    terms = _mlx_terms_for(job.family, job.model_size, level)
    references = capped_reference_megapixels(job.reference_megapixels)
    extra = terms.per_megapixel * max(0.0, job.megapixels - terms.flat_to)
    if references > 0:
        extra = max(extra, terms.reference + terms.per_reference * references)
    if job.from_picture:
        extra += terms.picture * job.megapixels
    return base + _bytes(terms.working + extra)
