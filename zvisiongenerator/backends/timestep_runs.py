"""Run an LTX-2 transformer on per-token timesteps without a per-token copy of its modulation.

Image-to-video gives every video token its own timestep: the conditioned first frame is clean (0), the rest is
noisy. diffusers then embeds the timestep per token and every block expands its nine modulation vectors per token,
which is several GB on a long clip, where text-to-video holds one row. The timesteps only take a few values, in runs
of consecutive tokens, so :func:`modulate_by_runs` hands the transformer one timestep per sample and gives each
block its modulation as :class:`TokenRuns`: one row per run, expanded only inside the products and sums that use it.
The results are the same values as diffusers' per-token path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["TokenRuns", "modulate_by_runs", "token_runs"]

_MAX_RUNS = 16  # more distinct runs than this and the per-token path is used unchanged


class TokenRuns:
    """A ``[batch, tokens, dim]`` tensor that is constant over runs of tokens, stored as one ``[batch, 1, dim]`` row per run.

    Adding or multiplying it with a full ``[batch, tokens, …]`` tensor gives a plain tensor, computed run by run into
    one output; with a scalar or a row-shaped tensor it gives another :class:`TokenRuns`.
    """

    def __init__(self, rows: list[Any], bounds: list[tuple[int, int]]):
        self.rows = rows
        self.bounds = bounds
        self.tokens = bounds[-1][1]

    @classmethod
    def __torch_function__(cls, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        import torch

        if kwargs or len(args) != 2:
            return NotImplemented
        left, right = args
        if func in (torch.Tensor.__add__, torch.Tensor.__radd__, torch.add):
            return _combine(left, right, torch.add)
        if func in (torch.Tensor.__mul__, torch.Tensor.__rmul__, torch.mul):
            return _combine(left, right, torch.mul)
        return NotImplemented

    def __add__(self, other: Any) -> Any:
        import torch

        return _combine(self, other, torch.add)

    def __mul__(self, other: Any) -> Any:
        import torch

        return _combine(self, other, torch.mul)

    def __radd__(self, other: Any) -> Any:
        import torch

        return _combine(other, self, torch.add)

    def __rmul__(self, other: Any) -> Any:
        import torch

        return _combine(other, self, torch.mul)


def _combine(left: Any, right: Any, op: Any) -> Any:
    """Apply *op* to *left* and *right*, at least one of them :class:`TokenRuns`, keeping the operand order."""
    import torch

    runs = left if isinstance(left, TokenRuns) else right
    other = right if runs is left else left
    if isinstance(other, TokenRuns):
        if other.bounds != runs.bounds:
            return NotImplemented
        return TokenRuns([op(a, b) for a, b in zip(left.rows, right.rows, strict=True)], runs.bounds)
    if not (isinstance(other, torch.Tensor) and other.dim() >= 2 and other.shape[1] == runs.tokens and runs.tokens > 1):
        rows = [op(row, other) if runs is left else op(other, row) for row in runs.rows]
        return TokenRuns(rows, runs.bounds)
    shape = torch.broadcast_shapes(other.shape, runs.rows[0].shape)
    out = torch.empty(shape, dtype=torch.result_type(other, runs.rows[0]), device=other.device)
    for (start, end), row in zip(runs.bounds, runs.rows, strict=True):
        part = other[:, start:end]
        op(part, row, out=out[:, start:end]) if runs is right else op(row, part, out=out[:, start:end])
    return out


def token_runs(timestep: Any) -> list[tuple[int, int]] | None:
    """Return the ``(start, end)`` runs of equal values along the tokens of a ``[batch, tokens]`` *timestep*.

    Returns:
        The runs, the same for every sample, or None when the samples differ or there are more than ``_MAX_RUNS``.
    """
    import torch

    if timestep.dim() != 2 or timestep.shape[1] < 1:
        return None
    if timestep.shape[0] > 1 and not bool((timestep == timestep[:1]).all()):
        return None
    row = timestep[0]
    changes = (torch.nonzero(row[1:] != row[:-1]).flatten() + 1).tolist()
    if len(changes) + 1 > _MAX_RUNS:
        return None
    edges = [0, *changes, row.shape[0]]
    return list(zip(edges[:-1], edges[1:], strict=True))


@dataclass
class _Context:
    """The runs of the current forward pass, and each run's timestep embeddings except the last's."""

    bounds: list[tuple[int, int]] = field(default_factory=list)
    starts: Any = None  # [batch, runs - 1] timesteps, embedded on first use (inside the forward pass, where the
    # embedding layer is on the GPU even when the model is offloaded)
    hidden_dtype: Any = None
    temb: list[Any] = field(default_factory=list)  # [batch, 1, params * dim] per run
    embedded: list[Any] = field(default_factory=list)  # [batch, 1, dim] per run
    normed: Any = None  # the output norm's result, for the output modulation


def modulate_by_runs(transformer: Any) -> list[Any]:
    """Make *transformer* (diffusers' ``LTX2VideoTransformer3DModel``) handle per-token timesteps by runs.

    A forward pass whose ``timestep`` is ``[batch, tokens]`` with the cross-attention and audio timesteps given per
    sample (``use_cross_timestep`` and ``audio_timestep``, as diffusers' LTX-2 pipelines pass them) runs with the last
    run's timestep for the whole clip; each block's video modulation and the output modulation are then put right
    for the other runs. Any other forward pass is left as it is.

    Returns:
        The hook handles.
    """
    context = _Context()
    handles = [
        transformer.register_forward_pre_hook(_runs_pre_hook(transformer, context), with_kwargs=True),
        transformer.register_forward_hook(lambda *_args: _clear(context)),
        transformer.norm_out.register_forward_hook(lambda _module, _args, output: _keep_normed(context, output)),
        transformer.proj_out.register_forward_pre_hook(_output_pre_hook(transformer, context)),
    ]
    for block in transformer.transformer_blocks:
        block.get_mod_params = _block_mod_params(transformer, block, context)
    return handles


def _runs_pre_hook(transformer: Any, context: _Context) -> Any:
    """Return the transformer pre-hook that finds the timestep runs and swaps in one timestep per sample."""

    def _hook(_module: Any, args: Any, kwargs: dict[str, Any]) -> Any:
        _clear(context)
        timestep = kwargs.get("timestep")
        hidden_states = kwargs.get("hidden_states")
        if timestep is None or hidden_states is None or kwargs.get("audio_timestep") is None or not kwargs.get("use_cross_timestep"):
            return None
        bounds = token_runs(timestep)
        if bounds is None:
            return None
        if len(bounds) > 1:
            context.starts = timestep[:, [start for start, _end in bounds[:-1]]]
            context.hidden_dtype = hidden_states.dtype
        context.bounds = bounds
        return args, {**kwargs, "timestep": timestep[:, -1:]}

    return _hook


def _embed_runs(transformer: Any, context: _Context) -> None:
    """Embed the timesteps of every run but the last, once per forward pass."""
    if context.starts is None:
        return
    batch_size, runs = context.starts.shape
    temb, embedded = transformer.time_embed(context.starts.flatten(), batch_size=batch_size, hidden_dtype=context.hidden_dtype)
    context.temb = list(temb.view(batch_size, runs, -1).split(1, dim=1))
    context.embedded = list(embedded.view(batch_size, runs, -1).split(1, dim=1))
    context.starts = None


def _block_mod_params(transformer: Any, block: Any, context: _Context) -> Any:
    """Return *block*'s ``get_mod_params``, giving its video modulation as :class:`TokenRuns` when there are runs."""
    own = type(block).get_mod_params

    def _get_mod_params(scale_shift_table: Any, temb: Any, batch_size: int) -> tuple[Any, ...]:
        if len(context.bounds) < 2 or scale_shift_table is not block.scale_shift_table:
            return own(scale_shift_table, temb, batch_size)
        _embed_runs(transformer, context)
        per_run = [own(scale_shift_table, run_temb, batch_size) for run_temb in [*context.temb, temb]]
        return tuple(TokenRuns(list(rows), context.bounds) for rows in zip(*per_run, strict=True))

    return _get_mod_params


def _keep_normed(context: _Context, output: Any) -> None:
    """Keep the output norm's result while there are runs, so the output modulation can be redone for them."""
    if len(context.bounds) > 1:
        context.normed = output


def _output_pre_hook(transformer: Any, context: _Context) -> Any:
    """Return the ``proj_out`` pre-hook that redoes the output modulation of every run but the last with its own timestep."""

    def _hook(_module: Any, args: Any) -> Any:
        if context.normed is None:
            return None
        (hidden_states,) = args
        _embed_runs(transformer, context)
        for (start, end), embedded in zip(context.bounds, context.embedded, strict=False):
            values = transformer.scale_shift_table[None, None] + embedded[:, :, None]
            shift, scale = values[:, :, 0], values[:, :, 1]
            hidden_states[:, start:end] = context.normed[:, start:end] * (1 + scale) + shift
        context.normed = None
        return None

    return _hook


def _clear(context: _Context) -> None:
    """Forget the runs of the last forward pass."""
    context.bounds, context.temb, context.embedded, context.normed, context.starts = [], [], [], None, None
