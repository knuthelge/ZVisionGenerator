<script lang="ts" module>
  import type { MemoryFit, MemoryFitEstimate, MemoryFitStatus } from '$lib/types';

  export const MEMORY_FIT_LABELS: Record<MemoryFitStatus, string> = {
    fits: 'Fits',
    tight: 'Tight',
    too_large: 'Too large'
  };

  export const DOWNLOADED_TOOLTIP = 'Downloaded: the model files are on this machine.';
  export const NOT_DOWNLOADED_TOOLTIP = 'Not downloaded: it downloads the first time you generate with it.';

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
    const lines = [`Needs ~${estimate.required_gb} GB; this Mac recommends up to ~${fit.budget_gb} GB for the GPU.`];
    const quantized = Object.entries(fit.by_quantize).filter(([key]) => key !== 'none');
    if (quantized.length > 0) {
      lines.push(quantized.map(([key, value]) => `q${key}: ~${value.required_gb} GB (${MEMORY_FIT_LABELS[value.status]})`).join(' · '));
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
    class?: string;
  }

  let { downloaded = null, memoryFit = null, quantize = null, lowMemory = true, tooltipPlacement = 'top', tooltipAlign = 'start', class: extraClass = '' }: Props = $props();

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
        <Badge variant="neutral" class="whitespace-nowrap">Not downloaded</Badge>
      </Tooltip>
    {/if}
    {#if memoryFit && estimate}
      <Tooltip text={memoryFitTitle(memoryFit, estimate)} placement={tooltipPlacement} align={tooltipAlign} testId="model-memory-fit">
        <span data-status={estimate.status}>
          <Badge variant={variants[estimate.status]} class="whitespace-nowrap">{MEMORY_FIT_LABELS[estimate.status]}</Badge>
        </span>
      </Tooltip>
    {/if}
  </span>
{/if}
