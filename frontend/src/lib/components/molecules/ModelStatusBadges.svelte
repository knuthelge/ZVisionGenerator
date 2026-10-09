<script lang="ts" module>
  import type { MemoryFit, MemoryFitEstimate, MemoryFitStatus, MemoryKind } from '$lib/types';

  export const MEMORY_FIT_LABELS: Record<MemoryFitStatus, string> = {
    fits: 'Fits',
    tight: 'Tight',
    too_large: 'Too large'
  };

  export const DOWNLOADED_TOOLTIP = 'Downloaded: the model files are on this machine.';
  export const NOT_DOWNLOADED_TOOLTIP = 'Not downloaded: it downloads the first time you generate with it.';

  /** What each fit level means in practice on each kind of memory; fits needs no explanation. */
  const MEMORY_FIT_CONSEQUENCES: Record<MemoryKind, Partial<Record<MemoryFitStatus, string>>> = {
    unified: {
      tight: 'It runs, but macOS compresses or swaps other memory to make room, so the Mac can slow down while generating.',
      too_large: 'More than MLX will try to fit (1.5× the recommendation): expect heavy swapping or an out-of-memory error.'
    },
    discrete: {
      tight: 'It runs, but slowly: the weights do not fit in system memory and partly stream from disk, or the GPU is nearly full.',
      too_large: "More than this GPU's memory: expect an out-of-memory error. Try a lower quant."
    }
  };

  function needLine(fit: MemoryFit, estimate: MemoryFitEstimate): string {
    if (fit.kind !== 'discrete') return `Needs ~${estimate.required_gb} GB; this Mac recommends up to ~${fit.budget_gb} GB for the GPU.`;
    return `Needs ~${estimate.required_gb} GB of GPU memory (this GPU has ${fit.budget_gb} GB) and ~${estimate.system_gb} GB of system memory (this machine has ${fit.system_budget_gb} GB).`;
  }

  function levelSummary(fit: MemoryFit, value: MemoryFitEstimate): string {
    const need = fit.kind === 'discrete' ? `~${value.required_gb} GB GPU, ~${value.system_gb} GB system` : `~${value.required_gb} GB`;
    return `${need} (${MEMORY_FIT_LABELS[value.status]})`;
  }

  /**
   * Pick the estimate for the selected settings. Returns null rather than a different setting's estimate when
   * the selected quantize level was not estimated, so the badge never describes something else.
   */
  export function memoryFitFor(fit: MemoryFit | null | undefined, quantize: number | null, lowMemory = true): MemoryFitEstimate | null {
    if (!fit) return null;
    if (!lowMemory && fit.without_low_memory) return fit.without_low_memory;
    return fit.by_quantize[quantize === null ? 'none' : String(quantize)] ?? null;
  }

  /** Describe the estimate, plus the quantized alternatives when they exist. */
  export function memoryFitTitle(fit: MemoryFit, estimate: MemoryFitEstimate): string {
    const lines = [needLine(fit, estimate)];
    const consequence = MEMORY_FIT_CONSEQUENCES[fit.kind ?? 'unified'][estimate.status];
    if (consequence) lines.push(consequence);
    const quantized = Object.entries(fit.by_quantize).filter(([key]) => key !== 'none');
    if (quantized.length > 0) {
      lines.push(quantized.map(([key, value]) => `q${key}: ${levelSummary(fit, value)}`).join(' · '));
    }
    if (fit.without_low_memory) {
      const staged = fit.by_quantize.none;
      const resident = fit.without_low_memory;
      lines.push(`Low memory on: ~${staged.required_gb} GB (${MEMORY_FIT_LABELS[staged.status]}) · off: ~${resident.required_gb} GB (${MEMORY_FIT_LABELS[resident.status]})`);
    }
    if (fit.note) lines.push(fit.note);
    return lines.join('\n');
  }
</script>

<script lang="ts">
  import { Badge, Tooltip } from '$lib/components/atoms';

  interface Props {
    downloaded?: boolean | null;
    memoryFit?: MemoryFit | null;
    quantize?: number | null;
    /** Whether low-memory mode is on (video); the estimate switches to all-components-resident when off. */
    lowMemory?: boolean;
    tooltipPlacement?: 'top' | 'bottom';
    tooltipAlign?: 'start' | 'end';
    /** `control` matches the height of the controls beside it, as in the Workspace toolbar. */
    size?: 'badge' | 'control';
    class?: string;
  }

  let { downloaded = null, memoryFit = null, quantize = null, lowMemory = true, tooltipPlacement = 'top', tooltipAlign = 'start', size = 'badge', class: extraClass = '' }: Props = $props();

  const badgeClass = $derived(size === 'control' ? 'ui-badge-control whitespace-nowrap' : 'whitespace-nowrap');

  const variants: Record<MemoryFitStatus, 'success' | 'warning' | 'error'> = {
    fits: 'success',
    tight: 'warning',
    too_large: 'error'
  };

  const estimate = $derived(memoryFitFor(memoryFit, quantize, lowMemory));
</script>

{#if downloaded === false || (memoryFit && estimate)}
  <span class="inline-flex flex-wrap items-center gap-1 {extraClass}">
    {#if downloaded === false}
      <Tooltip text={NOT_DOWNLOADED_TOOLTIP} placement={tooltipPlacement} align={tooltipAlign} testId="model-download-status">
        <Badge variant="neutral" class={badgeClass}>Not downloaded</Badge>
      </Tooltip>
    {/if}
    {#if memoryFit && estimate}
      <Tooltip text={memoryFitTitle(memoryFit, estimate)} placement={tooltipPlacement} align={tooltipAlign} testId="model-memory-fit">
        <span data-status={estimate.status}>
          <Badge variant={variants[estimate.status]} class={badgeClass}>{MEMORY_FIT_LABELS[estimate.status]}</Badge>
        </span>
      </Tooltip>
    {/if}
  </span>
{/if}
