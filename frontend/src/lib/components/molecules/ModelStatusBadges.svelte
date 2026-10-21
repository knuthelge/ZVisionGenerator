<script lang="ts" module>
  import { MEMORY_FIT_CONSEQUENCES } from '$lib/state/memoryFit';
  import type { MemoryFit, MemoryFitEstimate, MemoryFitStatus } from '$lib/types';

  export const MEMORY_FIT_LABELS: Record<MemoryFitStatus, string> = {
    fits: 'Fits',
    tight: 'Tight',
    too_large: 'Too large'
  };

  export const DOWNLOADED_TOOLTIP = 'Downloaded: the model files are on this machine.';
  /** Shown in the Workspace, where a model can be downloaded ahead from the Models page. */
  export const NOT_DOWNLOADED_TOOLTIP = 'Not downloaded: it downloads the first time you generate with it, or ahead from the Models page.';
  /** Shown instead of the not-downloaded text while a download of the model is queued or running. */
  export const DOWNLOADING_TOOLTIP = 'Download in progress: the model is ready to use when it finishes.';
  /** Shown on the Models page for a row with a Download button. */
  export const DOWNLOADABLE_TOOLTIP = 'Not downloaded: Download fetches it now, or it downloads the first time you generate with it.';
  /** Shown on the Models page for a row without a Download button. */
  export const NOT_DOWNLOADED_FIRST_USE_TOOLTIP = 'Not downloaded: it downloads the first time you generate with it.';

  /** Shown on the Models page for a row with nothing left to fetch that the first use still has to make. */
  export const PREPARED_SEEDVR2_TOOLTIP = 'Downloaded. The first upscale makes its 8-bit copy.';
  export const PREPARED_INSTANT_TOOLTIP = 'Downloaded. The first image builds it.';

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
    /** Text for a queued or running download of the model, such as "Downloading 42%"; replaces "Not downloaded" while it is set. */
    downloadStatus?: string | null;
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

  let { downloaded = null, downloadStatus = null, memoryFit = null, quantize = null, lowMemory = true, tooltipPlacement = 'top', tooltipAlign = 'start', size = 'badge', class: extraClass = '' }: Props = $props();

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
      <Tooltip text={downloadStatus ? DOWNLOADING_TOOLTIP : NOT_DOWNLOADED_TOOLTIP} placement={tooltipPlacement} align={tooltipAlign} testId="model-download-status">
        <Badge variant="neutral" class={badgeClass}>{downloadStatus ?? 'Not downloaded'}</Badge>
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
