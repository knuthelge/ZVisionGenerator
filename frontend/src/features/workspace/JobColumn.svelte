<script lang="ts">
  // The running job's card and the queue under it: the stage's column on a wide screen, and the Job sheet's body on a phone.
  // The card never shrinks. When the card and the expanded queue are taller than the column by more than one queue row,
  // the queue folds to its one-line head; a smaller overflow scrolls the column, with the last row's edge peeking out.
  import { tick, untrack } from 'svelte';
  import { JobCard } from '$lib/components/molecules';
  import type { RingMascot } from '$lib/components/molecules/RingHero.svelte';
  import type { ActiveJobState, JobSnapshot } from '$lib/types';
  import QueuePanel from './QueuePanel.svelte';
  import { shouldFoldQueue } from './queueView';

  type ControlCallback = (jobId: string) => void | Promise<unknown>;

  interface Props {
    /** The job the card shows; the queue stands alone without one. */
    job: ActiveJobState | null;
    queued: JobSnapshot[];
    oncancel?: ControlCallback;
    onpause?: ControlCallback;
    onresume?: ControlCallback;
    onnext?: ControlCallback;
    onrepeat?: ControlCallback;
    onremove: (job: JobSnapshot) => void;
    onload: (job: JobSnapshot) => void;
    onclear: () => void;
    mascot?: RingMascot;
    /** `panel` is two panels in a column that folds the queue to fit; `sheet` is bands on the Job sheet's surface. */
    variant?: 'panel' | 'sheet';
  }

  let { job, queued, oncancel, onpause, onresume, onnext, onrepeat, onremove, onload, onclear, mascot, variant = 'panel' }: Props = $props();

  let column = $state<HTMLElement | null>(null);
  let folded = $state(false);
  // The job the user last folded or unfolded the queue for: their choice wins for the rest of that job.
  let choiceFor: string | null = null;

  // The queue is re-measured when the job's state changes (not on notices, which are transient) and when the column resizes.
  const stateKey = $derived(`${job?.job_id}:${job?.status}:${job?.phase}:${job?.connection}:${queued.length}`);

  async function decide(): Promise<void> {
    if (variant !== 'panel' || !column) return;
    if (queued.length === 0) {
      folded = false;
      return;
    }
    if (choiceFor !== null && choiceFor === job?.job_id) return;
    // Measure the expanded queue: unfold, let the layout settle (still before paint), then keep it only if it fits.
    folded = false;
    await tick();
    if (column) folded = shouldFoldQueue(column.scrollHeight - column.clientHeight);
  }

  $effect(() => {
    void stateKey;
    untrack(() => void decide());
  });
  $effect(() => {
    if (!column) return;
    const watcher = new ResizeObserver(() => untrack(() => void decide()));
    watcher.observe(column);
    return () => watcher.disconnect();
  });
</script>

<div class="job-column {variant}" bind:this={column} data-testid="job-column">
  {#if job}
    <JobCard {job} {oncancel} {onpause} {onresume} {onnext} {onrepeat} {mascot} {variant} />
  {/if}
  <QueuePanel jobs={queued} {onremove} {onload} {onclear} bind:folded ontoggle={() => { choiceFor = job?.job_id ?? null; }} {variant} />
</div>

<style>
  .job-column.panel { display: flex; flex: none; width: 400px; max-width: 46%; flex-direction: column; gap: 12px; overflow-y: auto; }
  .job-column.sheet { display: flex; flex-direction: column; padding-bottom: 12px; }
</style>
