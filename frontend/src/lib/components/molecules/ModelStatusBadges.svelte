<script lang="ts" module>
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
</script>

<script lang="ts">
  import { Badge, Tooltip } from '$lib/components/atoms';
  import { MEMORY_VARIANTS, currentRow } from '$lib/state/memoryEstimate';
  import type { JobMemory } from '$lib/types';

  /**
   * A model's download status and, in a list, the memory tag of its reference job. The tag here is the plain word with no popover, because
   * these lists sit inside popovers and tables that cannot hold one; the Workspace toolbar and the Models page use `MemoryTag`.
   */
  interface Props {
    downloaded?: boolean | null;
    /** Text for a queued or running download of the model, such as "Downloading 42%"; replaces "Not downloaded" while it is set. */
    downloadStatus?: string | null;
    /** The memory answer of the model's reference job. */
    memory?: JobMemory | null;
    /** The level the tag judges (`none`, `8`, `4`, `on`, `off`); the answer's own default when absent. */
    level?: string | null;
    tooltipPlacement?: 'top' | 'bottom';
    /** `control` matches the height of the controls beside it, as in the Workspace toolbar. */
    size?: 'badge' | 'control';
  }

  let { downloaded = null, downloadStatus = null, memory = null, level = null, tooltipPlacement = 'top', size = 'badge' }: Props = $props();

  const badgeClass = $derived(size === 'control' ? 'ui-badge-control whitespace-nowrap' : 'whitespace-nowrap');

  const current = $derived(memory ? currentRow(memory, level) : null);
  const status = $derived(current?.status ?? null);
</script>

{#if downloaded === false || status !== null}
  <span class="inline-flex flex-wrap items-center gap-1">
    {#if downloaded === false}
      <Tooltip text={downloadStatus ? DOWNLOADING_TOOLTIP : NOT_DOWNLOADED_TOOLTIP} placement={tooltipPlacement} testId="model-download-status">
        <Badge variant="neutral" class={badgeClass}>{downloadStatus ?? 'Not downloaded'}</Badge>
      </Tooltip>
    {/if}
    {#if status !== null}
      <span data-status={status} data-testid="model-memory">
        <span class="sr-only">Memory:</span>
        <Badge variant={MEMORY_VARIANTS[status]} class={badgeClass}>{current?.word}</Badge>
      </span>
    {/if}
  </span>
{/if}
