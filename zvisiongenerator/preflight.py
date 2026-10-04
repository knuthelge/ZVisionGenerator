"""Preflight phase — plan every iteration of a job before the generation model loads."""

from __future__ import annotations

import random

from zvisiongenerator.core.job_plan import IterationPlan
from zvisiongenerator.utils.prompt_compose import expand_random_choices
from zvisiongenerator.utils.prompt_enhance import EnhanceSettings, entry_enhance, resolve_item_enhance

__all__ = ["plan_iterations"]


def plan_iterations(
    prompts_data: dict[str, list[tuple[str, str | None]]],
    *,
    runs: int,
    seed: int | None,
    seed_min: int,
    seed_max: int,
    json_prompt: bool,
    disabled: bool,
    override: EnhanceSettings | None,
    enhance_by_set: dict[str, list[EnhanceSettings | None]] | None,
) -> tuple[IterationPlan, ...]:
    """Enumerate the job's iterations with seeds, resolved prompts and enhancement settings (no model I/O).

    Iterations are ordered runs → sets (in *prompts_data* order) → prompts, the order the runners loop in.

    Args:
        prompts_data: Set name → list of ``(prompt, negative_prompt)`` tuples.
        runs: Number of passes over every prompt.
        seed: Fixed seed for every iteration, or ``None`` to draw one per iteration.
        seed_min: Lower bound for drawn seeds.
        seed_max: Upper bound for drawn seeds.
        json_prompt: Prompts are structured JSON captions: kept verbatim and never enhanced.
        disabled: ``--no-enhance`` was given.
        override: Job-wide enhancement settings (``--enhance``).
        enhance_by_set: Per-entry prompt-file ``enhance:`` settings aligned with *prompts_data*.
    """
    iterations: list[IterationPlan] = []
    for run_index in range(runs):
        for set_name, prompts in prompts_data.items():
            for prompt_index, (prompt, _negative) in enumerate(prompts):
                enhance = None if json_prompt else resolve_item_enhance(disabled=disabled, override=override, entry=entry_enhance(enhance_by_set, set_name, prompt_index))
                iterations.append(
                    IterationPlan(
                        run_index=run_index,
                        set_name=set_name,
                        prompt_index=prompt_index,
                        prompt=prompt,
                        seed=seed if seed is not None else random.randint(seed_min, seed_max),
                        resolved_prompt=prompt if json_prompt else expand_random_choices(prompt),
                        enhance=enhance,
                    )
                )
    return tuple(iterations)
