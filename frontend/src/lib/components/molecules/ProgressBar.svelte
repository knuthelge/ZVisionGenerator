<script lang="ts">
  interface Props {
    /** 0–1 fraction for stage-level progress (teal bar) */
    stagePct?: number;
    /** Stage label text shown on left */
    stageLabel?: string;
    /** Stage percentage text shown on right */
    stageValueLabel?: string;
    /** 0–1 fraction for step-level progress (cyan bar) */
    stepPct?: number;
    /** Phase text shown on left under step bar */
    stepPhase?: string;
    /** Step count label shown on right e.g. "12 / 20" */
    stepLabel?: string;
    /** Batch label shown above stage bar */
    batchLabel?: string;
    /** Batch meta (e.g. run count) shown on right of batch label */
    batchMeta?: string;
    /** Elapsed time string e.g. "0:42" */
    elapsed?: string;
    /** Remaining time string e.g. "1:15" */
    remaining?: string;
    /** Terminal state changes bar colors */
    state?: 'running' | 'completed' | 'failed';
  }

  let {
    stagePct = 0,
    stageLabel = 'Workflow',
    stageValueLabel,
    stepPct = 0,
    stepPhase = 'Waiting for first denoiser step...',
    stepLabel = '0 / 0',
    batchLabel = 'Waiting for batch context...',
    batchMeta = '0 / 0',
    elapsed = '00:00',
    remaining = '--:--',
    state = 'running'
  }: Props = $props();

  const stageFill = $derived(
    state === 'completed' ? 'bg-emerald-400' :
    state === 'failed'    ? 'bg-red-400'     :
                            'bg-teal-400'
  );

  const stageWidth = $derived(`${Math.min(100, Math.max(0, stagePct * 100))}%`);
  const stepWidth  = $derived(`${Math.min(100, Math.max(0, stepPct  * 100))}%`);

  const displayStageValue = $derived(stageValueLabel ?? `${Math.round(stagePct * 100)}%`);
</script>

<div>
  <!-- Batch row -->
  <div class="flex items-center justify-between gap-3 text-[11px] uppercase tracking-[0.18em] text-zinc-500">
    <span>{batchLabel}</span>
    <span class="font-mono">{batchMeta}</span>
  </div>

  <!-- Stage progress bar -->
  <div class="mt-3 h-2 overflow-hidden rounded-full bg-zinc-800">
    <div
      class="h-full rounded-full {stageFill} transition-all duration-300"
      style="width: {stageWidth}"
    ></div>
  </div>
  <div class="mt-2 flex items-center justify-between gap-4 text-[11px] uppercase tracking-[0.18em] text-zinc-500">
    <span>{stageLabel}</span>
    <span>{displayStageValue}</span>
  </div>

  <!-- Step progress bar -->
  <div class="mt-3 h-1.5 overflow-hidden rounded-full bg-zinc-900">
    <div
      class="h-full rounded-full bg-cyan-400 transition-all duration-300"
      style="width: {stepWidth}"
    ></div>
  </div>
  <div class="mt-2 flex items-center justify-between gap-4 text-[11px] text-zinc-500">
    <span>{stepPhase}</span>
    <span class="font-mono">{stepLabel}</span>
  </div>

  <!-- Elapsed / Remaining -->
  <div class="mt-3 grid grid-cols-2 gap-3 text-[11px] text-zinc-500">
    <div class="rounded-md border border-zinc-800 bg-zinc-900/70 px-3 py-2">
      <span class="block uppercase tracking-[0.18em] text-zinc-600">Elapsed</span>
      <span class="mt-1 block font-mono text-zinc-300">{elapsed}</span>
    </div>
    <div class="rounded-md border border-zinc-800 bg-zinc-900/70 px-3 py-2">
      <span class="block uppercase tracking-[0.18em] text-zinc-600">Remaining</span>
      <span class="mt-1 block font-mono text-zinc-300">{remaining}</span>
    </div>
  </div>
</div>
