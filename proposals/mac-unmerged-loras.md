# Unmerged LoRAs on macOS

**Status:** Proposed (2026-10-31)

## Problem

On macOS the mflux backend loads Z-Image, FLUX.2 Klein, Krea 2 and Ideogram 4 LoRAs through each model's constructor and never passes `bake_lora`. mflux defaults it to `True` for those families (`LoRALoader.load_and_apply_lora`, and each constructor's own default), so a LoRA is added into the transformer's weights once at load. MOSS-Image is the exception: its backend passes `bake_lora=False`.

mflux's MOSS-Image notes measure what a merge costs there: a merge into bfloat16 weights keeps 44 to 49% of a step-distilled LoRA's update, because most of the update is smaller than half a bfloat16 step of the weight it lands on; on a quantized model a merge keeps about 20%. The same arithmetic applies to any family: a LoRA whose update is small next to the weights loses part of its effect when merged. Style LoRAs are usually larger and may lose little, but nothing measures it today, and Windows and Linux keep LoRAs unmerged (diffusers/PEFT), so a LoRA can look weaker on a Mac.

## Evidence to gather

1. For each family, one curated LoRA at its usual strength: the same seed merged and unmerged (`bake_lora=False`), at q8 and q4, side by side.
2. The cost of unmerged: seconds per step and peak memory, with one and with two LoRAs.
3. mflux's own report: it warns when a bake keeps less than its threshold of the update (`LoRASaver`); log what it says for each file.

## Proposed change

Pass `bake_lora=False` for the families where the comparison shows a visible loss and the cost is small, as MOSS-Image does. Stored copies are unaffected: they are saved without LoRAs.

## Alternatives

- Keep merging, and document that LoRAs can apply more weakly on macOS.
- Let the user choose per job. Rejected unless the comparison shows the trade-off differs a lot between LoRAs.
