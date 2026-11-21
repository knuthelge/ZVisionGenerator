<script lang="ts">
  import { Badge, Tooltip } from '$lib/components/atoms';
  import { MEMORY_VARIANTS, currentRow } from '$lib/state/memoryEstimate';
  import type { JobMemory } from '$lib/types';
  import MemoryEstimate from './MemoryEstimate.svelte';

  /** The memory tag: Fits, Tight or Too large for the job as set up (or a list's reference job), with the estimate on hover, focus or tap. */
  interface Props {
    memory: JobMemory;
    /** The level the job uses; the answer's own current one when absent. */
    level?: string | null;
    /** `control` matches the height of the controls beside it, as in the Workspace toolbar. */
    size?: 'badge' | 'control';
    align?: 'start' | 'end';
    /** Makes the table's rows choosable, as in the Workspace toolbar: called with the key of the level the user picked. */
    onselect?: (key: string) => void;
    /** The row keys `onselect` may take; the other rows stay plain. */
    selectable?: readonly string[];
    /** The answer is of the previous model or job while the new one is asked: the badge is dimmed and its rows cannot be chosen. */
    pending?: boolean;
    testId?: string;
  }

  let { memory, level = null, size = 'badge', align = 'start', onselect, selectable, pending = false, testId = 'memory-tag' }: Props = $props();

  const current = $derived(currentRow(memory, level));
  const status = $derived(current.status);
  // A level chosen from the previous model's rows would be set for the wrong model.
  const choose = $derived(pending ? undefined : onselect);
</script>

<Tooltip placement="bottom" {align} {testId} interactive={choose !== undefined}>
  <span data-status={status} aria-busy={pending ? 'true' : undefined} class="transition-opacity duration-150 {pending ? 'opacity-50' : 'opacity-100'}">
    <span class="sr-only">Memory:</span>
    <Badge variant={MEMORY_VARIANTS[status]} class={size === 'control' ? 'ui-badge-control whitespace-nowrap' : 'whitespace-nowrap'}>{current.word}</Badge>
  </span>
  {#snippet content()}
    <MemoryEstimate {memory} {level} onselect={choose} selectable={pending ? undefined : selectable} />
  {/snippet}
</Tooltip>
