<script lang="ts">
  import { onMount, tick } from 'svelte';
  import { Icon, Mascot, Spinner } from '$lib/components/atoms';
  import type { MascotMood } from '$lib/components/atoms/Mascot.svelte';
  import { Modal } from '$lib/components/molecules';
  import type { DownloadRequestResult } from '$lib/state/downloads.svelte';
  import { layout } from '$lib/state/layoutMode.svelte';
  import type { ModelOption } from '$lib/types';

  interface Props {
    /** The image models to choose from, in the order the menu lists them. */
    models: ModelOption[];
    /** The model preselected and shown as the recommended card, when it is one of `models`. */
    recommended: string | null;
    /** Start the download of the chosen model; the dialog waits for the answer. */
    ondownload: (model: ModelOption) => Promise<DownloadRequestResult>;
    onskip: () => void;
    /** Escape, the backdrop or Close: put the dialog away for this page load. */
    onclose: () => void;
  }

  let { models, recommended, ondownload, onskip, onclose }: Props = $props();

  // The hero picture is decorative. Generated with the app (Krea 2 q8, seed 2154127710, ukiyo-e enhanced prompt).
  const STRIP_SRC = '/app-static/welcome/strip.webp';
  // How long Blob waves before settling into idle.
  const WAVE_MS = 2400;

  const uid = $props.id();
  const featured = $derived(models.find((model) => model.id === recommended) ?? null);
  const others = $derived(models.filter((model) => model.id !== featured?.id));

  let open = $state(true);
  let selected = $state('');
  // Starts from the preselected model; later models changes keep the user's choice while it is still listed.
  $effect(() => {
    if (!models.some((model) => model.id === selected)) selected = featured?.id ?? models[0]?.id ?? '';
  });
  const chosen = $derived(models.find((model) => model.id === selected) ?? null);

  let pending = $state(false);
  let failed = $state(false);
  let waving = $state(true);
  let stripLoaded = $state(false);
  let root = $state<HTMLDivElement | null>(null);
  let stripImage = $state<HTMLImageElement | null>(null);

  const mood = $derived<MascotMood>(pending ? 'thinking' : failed ? 'sad' : waving ? 'waving' : 'idle');

  onMount(() => {
    const timer = setTimeout(() => { waving = false; }, WAVE_MS);
    // The checked model takes focus without a ring; the ring appears once the keyboard is used.
    void tick().then(() => root?.querySelector<HTMLInputElement>('input[type="radio"]:checked')?.focus({ focusVisible: false } as FocusOptions));
    return () => clearTimeout(timer);
  });

  // A picture that was cached never fires its load event after the handler is attached.
  $effect(() => {
    if (stripImage?.complete && stripImage.naturalWidth > 0) stripLoaded = true;
  });

  function select(): void {
    failed = false;
  }

  /** Enter picks nothing and downloads nothing: only the Download button starts a download. */
  function keepEnterFromActing(event: KeyboardEvent): void {
    if (event.key === 'Enter') event.preventDefault();
  }

  async function download(): Promise<void> {
    if (pending || !chosen) return;
    failed = false;
    pending = true;
    try {
      const result = await ondownload(chosen);
      failed = result === 'failed';
    } finally {
      pending = false;
    }
  }

  function skip(): void {
    if (!pending) onskip();
  }
</script>

{#snippet tickMark()}
  <span class="fr-tick" aria-hidden="true">
    <svg viewBox="0 0 24 24" width="100%" height="100%" fill="none" stroke="currentColor" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12l5 5L20 7" /></svg>
  </span>
{/snippet}

{#snippet actions(main: boolean)}
  <div class="ui-action-bar fr-actions" class:fr-actions-main={main}>
    <button type="button" class="ui-btn ui-btn-quiet" data-action="first-run-skip" aria-disabled={pending ? 'true' : undefined} onclick={skip}>Skip</button>
    <button
      type="button"
      class="ui-btn ui-btn-primary fr-download"
      class:ui-btn-main={main}
      data-action="first-run-download"
      aria-disabled={pending || !chosen ? 'true' : undefined}
      onclick={() => void download()}
    >
      {#if pending}
        <Spinner size="sm" />
        <span class="fr-download-text">Preparing…</span>
      {:else}
        <Icon name="download" size={15} />
        <span class="fr-download-text">Download {chosen?.label ?? ''}</span>
      {/if}
    </button>
  </div>
{/snippet}

{#snippet phoneFoot()}
  {@render actions(true)}
{/snippet}

<Modal bind:open size="xl" fill labelledby="{uid}-title" describedby="{uid}-lede" {onclose} footer={layout.compact ? phoneFoot : undefined}>
  <div bind:this={root} class="fr" class:compact={layout.compact} data-testid="first-run-dialog">
    <div class="fr-strip" class:loaded={stripLoaded} aria-hidden="true">
      <img bind:this={stripImage} src={STRIP_SRC} alt="" decoding="async" onload={() => { stripLoaded = true; }} />
    </div>
    <div class="fr-pool" aria-hidden="true"></div>

    <Mascot size={layout.compact ? 96 : 150} {mood} class="fr-blob" />

    <div class="fr-head">
      <span class="ui-area-label fr-kicker">Welcome to Z-Vision Generator</span>
      <h2 id="{uid}-title" class="text-display fr-title">{layout.compact ? 'Your studio is ready.' : 'Your studio is ready. It needs a model.'}</h2>
      <p id="{uid}-lede" class="fr-lede">It downloads once, then everything runs on this machine.</p>
    </div>

    <fieldset class="fr-pick" disabled={pending} data-testid="first-run-models">
      <legend class="sr-only">Image model</legend>

      {#if featured}
        <label class="fr-feat" class:selected={selected === featured.id} data-testid="first-run-recommended">
          <input
            type="radio"
            class="sr-only"
            name="first-run-model"
            value={featured.id}
            bind:group={selected}
            onchange={select}
            onkeydown={keepEnterFromActing}
            aria-describedby="{uid}-badge{featured.download_size_label ? ` ${uid}-size-${featured.id}` : ''}"
            data-testid="first-run-model-{featured.id}"
          />
          <span id="{uid}-badge" class="ui-area-label fr-badge"><Icon name="sparkle" size={12} />Recommended for this machine</span>
          <span class="text-display fr-feat-name">{featured.label}</span>
          {#if featured.download_size_label}
            <span id="{uid}-size-{featured.id}" class="fr-feat-size">≈{featured.download_size_label} · one-time download</span>
          {/if}
          {@render tickMark()}
        </label>
      {/if}

      {#if others.length > 0}
        <span class="ui-area-label fr-menu-title">{featured ? 'Or choose another' : 'Choose a model'}</span>
        <div class="fr-menu">
          {#each others as model (model.id)}
            <label class="fr-row" class:selected={selected === model.id}>
              <input
                type="radio"
                class="sr-only"
                name="first-run-model"
                value={model.id}
                bind:group={selected}
                onchange={select}
                onkeydown={keepEnterFromActing}
                aria-describedby={model.download_size_label ? `${uid}-size-${model.id}` : undefined}
                data-testid="first-run-model-{model.id}"
              />
              {@render tickMark()}
              <span class="fr-row-name">{model.label}</span>
              {#if model.download_size_label}
                <span class="fr-leader" aria-hidden="true"></span>
                <span id="{uid}-size-{model.id}" class="fr-row-size" data-testid="first-run-size">≈{model.download_size_label}</span>
              {/if}
            </label>
          {/each}
        </div>
      {/if}
    </fieldset>

    {#if !layout.compact}
      <div class="fr-cta">{@render actions(false)}</div>
    {/if}

    <span class="sr-only" role="status">{pending && chosen ? `Preparing ${chosen.label}…` : ''}</span>
  </div>
</Modal>

<style>
  .fr {
    position: relative;
    isolation: isolate;
    display: flex;
    flex: 1 1 auto;
    flex-direction: column;
    align-items: center;
    min-height: 0;
    overflow-y: auto;
    overscroll-behavior: contain;
    padding: 44px 32px 40px;
    background: var(--color-bg-overlay);
  }
  .fr.compact { padding: 0 0 16px; background: var(--color-bg-surface); }

  /* The hero strip: a gradient until the picture loads, then the picture fading into the surface. */
  .fr-strip {
    position: absolute;
    inset: 0 0 auto;
    z-index: -1;
    height: 140px;
    overflow: hidden;
    border-radius: var(--radius-md) var(--radius-md) 0 0;
    background: linear-gradient(to bottom, var(--color-primary-subtle), var(--color-bg-overlay));
  }
  .fr-strip img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: 50% 60%;
    opacity: 0;
    scale: 1.03;
    transition: opacity 240ms cubic-bezier(0.32, 0.72, 0, 1), scale 240ms cubic-bezier(0.32, 0.72, 0, 1);
  }
  .fr-strip.loaded img { opacity: 1; scale: 1; }
  .fr-strip::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(to top, var(--color-bg-overlay) 0%, color-mix(in srgb, var(--color-bg-overlay) 70%, transparent) 25%, transparent 70%);
  }
  .fr.compact .fr-strip { height: 76px; border-radius: 0; }
  .fr.compact .fr-strip::after {
    background: linear-gradient(to top, var(--color-bg-surface) 0%, color-mix(in srgb, var(--color-bg-surface) 70%, transparent) 25%, transparent 70%);
  }

  /* The teal light Blob stands in; it spills out of the picture's foot. */
  .fr-pool {
    position: absolute;
    top: 70px;
    left: 50%;
    z-index: -1;
    width: 560px;
    height: 200px;
    translate: -50% 0;
    background: radial-gradient(closest-side, color-mix(in srgb, var(--color-primary-main) 34%, transparent), color-mix(in srgb, var(--color-primary-main) 10%, transparent) 55%, transparent);
    pointer-events: none;
  }
  .fr.compact .fr-pool { top: 32px; width: 320px; height: 140px; }

  .fr :global(.fr-blob) { flex: none; }
  .fr.compact :global(.fr-blob) { margin-top: 32px; }

  .fr-head { display: flex; flex-direction: column; align-items: center; gap: 10px; max-width: 640px; margin-top: 16px; text-align: center; }
  .fr.compact .fr-head { gap: 6px; padding: 0 16px; }
  .fr-kicker { color: var(--color-primary-main); }
  .fr-title { margin: 0; font-family: var(--font-heading); font-weight: 800; line-height: 1.08; letter-spacing: -0.015em; text-wrap: balance; color: var(--color-text-primary); }
  .fr-lede { margin: 0; font-size: var(--text-content); line-height: 1.55; color: var(--color-text-secondary); }

  .fr-pick { display: flex; flex-direction: column; gap: 14px; width: min(640px, 100%); min-width: 0; margin: 28px 0 0; border: 0; padding: 0; }
  .fr.compact .fr-pick { gap: 12px; padding: 0 16px; margin-top: 20px; }

  /* The tick shared by the card and the rows: an empty disc until chosen, then teal with a check that settles in. */
  .fr-tick { display: grid; flex: none; place-items: center; width: 16px; height: 16px; padding: 3px; border-radius: 50%; background: var(--color-zinc-700); color: var(--color-primary-ink); }
  .fr-tick svg { scale: 0.4; opacity: 0; transition: scale 180ms cubic-bezier(0.32, 0.72, 0, 1), opacity 180ms cubic-bezier(0.32, 0.72, 0, 1); }
  .selected > .fr-tick { background: var(--color-primary-main); }
  .selected > .fr-tick svg { scale: 1; opacity: 1; }

  /* The recommended model: its name in display type over a teal glow. */
  .fr-feat {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 6px;
    padding: 20px 24px 22px;
    border-radius: var(--radius-md);
    background-color: var(--color-bg-surface);
    background-image: radial-gradient(120% 160% at 0 0, color-mix(in srgb, var(--color-primary-main) 16%, transparent), transparent 55%);
    cursor: pointer;
    transition: background-color 0.12s ease, box-shadow 0.12s ease;
  }
  .fr.compact .fr-feat { padding: 14px 16px 16px; }
  .fr-feat.selected { box-shadow: inset 0 0 0 2px var(--color-primary-main); }
  .fr-feat > .fr-tick { position: absolute; top: 18px; right: 18px; width: 20px; height: 20px; padding: 4px; }
  .fr.compact .fr-feat > .fr-tick { top: 14px; right: 14px; }
  .fr-badge { display: inline-flex; align-items: center; gap: 5px; color: var(--color-primary-main); }
  .fr-feat-name { font-family: var(--font-heading); font-weight: 800; line-height: 1.05; color: var(--color-text-primary); }
  .fr-feat-size { font-family: var(--font-mono); font-size: var(--text-content); color: var(--color-text-secondary); }

  .fr-menu-title { margin-bottom: 2px; }
  .fr-menu { display: grid; grid-template-columns: 1fr 1fr; column-gap: 28px; row-gap: 2px; }
  .fr.compact .fr-menu { grid-template-columns: 1fr; }
  .fr-row {
    display: flex;
    align-items: baseline;
    gap: 10px;
    min-width: 0;
    padding: 12px 6px;
    border-radius: var(--radius-sm);
    font-size: var(--text-content);
    cursor: pointer;
    transition: background-color 0.12s ease;
  }
  .fr.compact .fr-row { padding: 11px 6px; min-height: var(--spacing-row); font-size: 15px; }
  .fr-row > .fr-tick { align-self: center; }
  .fr-row-name { min-width: 0; overflow: hidden; font-weight: 600; white-space: nowrap; text-overflow: ellipsis; color: var(--color-text-primary); }
  .fr-row.selected .fr-row-name { color: var(--color-primary-main); }
  .fr-leader { flex: 1; min-width: 16px; border-bottom: 1px dotted var(--color-border-strong); translate: 0 -3px; }
  .fr-row-size { flex: none; font-family: var(--font-mono); font-size: var(--text-ui); font-variant-numeric: tabular-nums; color: var(--color-text-secondary); }

  @media (hover: hover) {
    .fr-row:hover { background: var(--color-bg-surface); }
    .fr.compact .fr-row:hover { background: var(--color-bg-surface-hover); }
    .fr-feat:hover { background-color: var(--color-bg-surface-hover); }
    .fr-actions .fr-download[aria-disabled='true']:hover { background: var(--color-bg-surface-hover); color: var(--color-text-muted); }
  }
  @media (hover: none) {
    .fr-row:active, .fr-feat:active { background-color: var(--color-bg-control-hover); }
  }
  /* The global ring on the card or row that holds the focused radio; on the selected card it sits outside the inset ring. */
  .fr-feat:has(input:focus-visible), .fr-row:has(input:focus-visible) { outline: 2px solid var(--color-primary-main); outline-offset: 2px; }

  /* A download is being prepared: the choice is locked and what was not chosen steps back. */
  .fr-pick:disabled .fr-feat:not(.selected), .fr-pick:disabled .fr-row:not(.selected) { opacity: 0.5; }
  .fr-pick:disabled .fr-feat, .fr-pick:disabled .fr-row { cursor: progress; }

  .fr-cta { display: flex; justify-content: center; margin-top: 20px; }
  .fr-download { min-width: 300px; }
  .fr-actions-main { width: 100%; }
  .fr-actions-main .fr-download { min-width: 0; }
  .fr-download-text { min-width: 0; overflow: hidden; text-overflow: ellipsis; }
  .fr-actions :global(.ui-btn[aria-disabled='true']) { cursor: not-allowed; }
  .fr-actions .fr-download[aria-disabled='true'] { background: var(--color-bg-surface-hover); color: var(--color-text-muted); cursor: progress; }
  .fr-actions .ui-btn-quiet[aria-disabled='true'] { opacity: 0.4; }
</style>
