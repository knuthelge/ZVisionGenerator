<script lang="ts" module>
  const DETAILS_KEY = 'ziv-viewer-details-v1';

  function loadDetailsOpen(): boolean {
    try {
      return localStorage.getItem(DETAILS_KEY) === 'true';
    } catch {
      return false;
    }
  }

  function saveDetailsOpen(open: boolean): void {
    try {
      localStorage.setItem(DETAILS_KEY, String(open));
    } catch {
      // storage unavailable — keep the in-memory choice only
    }
  }
</script>

<script lang="ts">
  import { Icon, ShortcutList } from '$lib/components/atoms';
  import { isTyping } from '$lib/keyboard';
  import { canUseAsReference, describeFallbackReason, upscaleFactor, upscaleFactors, type AssetActionHandlers, type ReferenceTarget } from '$lib/state/assetActions';
  import { addToast } from '$lib/state/toasts.svelte';
  import type { GalleryAsset, UpscaleFactor } from '$lib/types';
  import ActionMenu, { type ActionMenuEntry } from './ActionMenu.svelte';
  import { assetDetailSections, fileName, type DetailFact } from './assetDetails';
  import { referenceEntries, upscaleEntries } from './assetMenu';
  import { VIEWER_SHORTCUTS, stepUpscaleChord, viewerActionFor, type ViewerAction } from './viewerShortcuts';

  interface Props extends Omit<AssetActionHandlers, 'onpreview'> {
    assets: GalleryAsset[];
    currentIndex: number;
    open: boolean;
    /** Where these assets come from, e.g. "History"; shown under the filename. */
    setLabel?: string;
    deletingIds?: ReadonlySet<string>;
    /** Reference targets the current model can't use, with the reason. */
    referenceUnavailable?: Partial<Record<ReferenceTarget, string>>;
    /** Called when the viewer nears the end of the list, so a paged source can load more. */
    onnearend?: () => void;
    onclose: () => void;
    onnavigate: (index: number) => void;
  }

  let {
    assets,
    currentIndex,
    open,
    setLabel = '',
    deletingIds = new Set<string>(),
    referenceUnavailable = {},
    onnearend,
    onclose,
    onnavigate,
    onreuse,
    onreference,
    onupscale,
    ondelete,
  }: Props = $props();

  // A delete can shrink the list under the viewer; stay on the nearest remaining asset.
  const index = $derived(Math.min(currentIndex, assets.length - 1));
  const asset = $derived(index >= 0 ? assets[index] : null);
  const hasPrev = $derived(index > 0);
  const hasNext = $derived(index < assets.length - 1);
  const canReuse = $derived(asset?.has_reusable_config === true);
  const deleting = $derived(asset ? deletingIds.has(asset.id) : false);
  const fallbackReasons = $derived(asset?.reuse_state?.fallback_reasons ?? []);
  let filmEl = $state<HTMLDivElement | null>(null);
  const NEAR_END = 3;
  // Videos have no lazy loading; only those near the current asset load their first frame.
  const FILM_VIDEO_RADIUS = 4;

  $effect(() => {
    if (open && assets.length > 0 && index >= assets.length - NEAR_END) onnearend?.();
  });

  // Keep the current thumbnail centred in the strip as the viewer moves.
  $effect(() => {
    if (!open || !filmEl) return;
    const current = filmEl.querySelector<HTMLElement>(`[data-film-index="${index}"]`);
    current?.scrollIntoView?.({ block: 'nearest', inline: 'center', behavior: 'smooth' });
  });

  let detailsOpen = $state(loadDetailsOpen());
  let referenceOpen = $state(false);
  let referenceButton = $state<HTMLButtonElement | null>(null);
  let upscaleOpen = $state(false);
  let upscaleButton = $state<HTMLButtonElement | null>(null);
  let closeButton = $state<HTMLButtonElement | null>(null);
  let downloadLink = $state<HTMLAnchorElement | null>(null);
  let helpOpen = $state(false);

  const referenceItems = $derived<ActionMenuEntry[]>(
    asset && onreference ? referenceEntries(asset, onreference, referenceUnavailable) : []
  );

  const upscaleItems = $derived<ActionMenuEntry[]>(
    asset && onupscale ? upscaleEntries(asset, onupscale) : []
  );

  const sections = $derived(asset ? assetDetailSections(asset) : null);
  const source = $derived(asset?.details?.source ?? null);
  const sourceIndex = $derived(source?.id ? assets.findIndex((item) => item.id === source.id) : -1);

  /** Upscale the shown image by *factor* from the keyboard; a disallowed factor explains why instead. */
  function runUpscale(factor: UpscaleFactor): boolean {
    if (!asset || !onupscale) return false;
    const option = upscaleFactor(asset, factor);
    if (!option) return false;
    if (!option.allowed) {
      addToast(option.reason ?? `Upscaling ${factor}× is not available for this image.`, 'warning');
      return true;
    }
    onupscale(asset, factor);
    return true;
  }

  function toggleDetails(): void {
    detailsOpen = !detailsOpen;
    saveDetailsOpen(detailsOpen);
  }

  async function copyPrompt(prompt: string): Promise<void> {
    try {
      await navigator.clipboard.writeText(prompt);
      addToast('Prompt copied', 'success');
    } catch {
      addToast('Could not copy the prompt', 'error');
    }
  }

  $effect(() => {
    if (open && assets.length === 0) onclose();
  });

  /** Run a keyboard action on the shown asset; return whether it applied. */
  function runAction(action: ViewerAction): boolean {
    if (!asset) return false;
    switch (action) {
      case 'close':
        if (helpOpen) helpOpen = false;
        else onclose();
        return true;
      case 'prev':
        if (hasPrev) onnavigate(index - 1);
        return hasPrev;
      case 'next':
        if (hasNext) onnavigate(index + 1);
        return hasNext;
      case 'first':
        if (hasPrev) onnavigate(0);
        return hasPrev;
      case 'last':
        if (hasNext) onnavigate(assets.length - 1);
        return hasNext;
      case 'reference':
        if (referenceItems.length === 0) return false;
        referenceOpen = true;
        return true;
      case 'copy':
        if (!asset.prompt) return false;
        void copyPrompt(asset.prompt);
        return true;
      case 'details':
        toggleDetails();
        return true;
      case 'help':
        helpOpen = !helpOpen;
        return true;
      case 'reuse':
        if (!onreuse || !canReuse) return false;
        onreuse(asset);
        return true;
      case 'download':
        downloadLink?.click();
        return downloadLink !== null;
      case 'delete':
      case 'delete-now':
        // Ignore key repeat while the first delete is still in flight.
        if (!ondelete || deleting) return false;
        ondelete(asset, { confirm: action === 'delete' });
        return true;
    }
  }

  $effect(() => {
    if (!open) return;
    queueMicrotask(() => closeButton?.focus());
    let upscaleWaitingSince: number | null = null;
    function handleKeydown(e: KeyboardEvent): void {
      // While a menu is open, keys belong to it.
      if (e.defaultPrevented || referenceOpen || upscaleOpen || isTyping(e.target)) {
        upscaleWaitingSince = null;
        return;
      }
      if (onupscale && asset && upscaleFactors(asset).length > 0) {
        const chord = stepUpscaleChord(upscaleWaitingSince, e, Date.now());
        upscaleWaitingSince = chord.waitingSince;
        if (chord.value) {
          if (runUpscale(chord.value)) e.preventDefault();
          return;
        }
        if (chord.consumed) {
          e.preventDefault();
          return;
        }
      }
      const action = viewerActionFor(e);
      // A held Del must not open a second confirmation.
      if ((action === 'delete' || action === 'delete-now') && e.repeat) {
        e.preventDefault();
        return;
      }
      if (action && runAction(action)) e.preventDefault();
    }
    document.addEventListener('keydown', handleKeydown);
    return () => document.removeEventListener('keydown', handleKeydown);
  });
</script>

{#if open && asset}
  <div
    class="asset-viewer"
    role="dialog"
    aria-modal="true"
    aria-labelledby="asset-viewer-title"
    data-testid="asset-viewer"
    data-details={detailsOpen}
  >
    <div class="viewer-bar">
      <div class="viewer-title">
        <b id="asset-viewer-title" class="truncate">{asset.filename}</b>
        <small>{index + 1} of {assets.length}{setLabel ? ` · ${setLabel}` : ''}</small>
      </div>
      {#if onreuse}
        <button
          type="button"
          class="viewer-btn ui-btn ui-btn-primary"
          data-action="reuse"
          disabled={!canReuse}
          title={canReuse ? 'Load these settings into the workspace (R)' : 'Reusable settings unavailable'}
          onclick={() => onreuse(asset)}
        ><Icon name="reuse" size={14} />Reuse settings</button>
      {/if}
      {#if onreference && canUseAsReference(asset)}
        <button
          type="button"
          bind:this={referenceButton}
          class="viewer-btn ui-btn"
          data-action="reference"
          aria-haspopup="menu"
          aria-expanded={referenceOpen}
          title="Use as reference (E)"
          onclick={() => { referenceOpen = !referenceOpen; }}
        ><Icon name="reference" size={14} />Use as reference<Icon name="chevdown" size={12} class="opacity-70" /></button>
      {/if}
      {#if upscaleItems.length > 0}
        <button
          type="button"
          bind:this={upscaleButton}
          class="viewer-btn ui-btn"
          data-action="upscale"
          aria-haspopup="menu"
          aria-expanded={upscaleOpen}
          title="Upscale (X then 2 or 4)"
          onclick={() => { upscaleOpen = !upscaleOpen; }}
        ><Icon name="expand" size={14} />Upscale<Icon name="chevdown" size={12} class="opacity-70" /></button>
      {/if}
      <a
        bind:this={downloadLink}
        class="viewer-btn ui-btn"
        data-action="download"
        href={asset.url}
        download={asset.filename}
        title="Download (D)"
      >
        <Icon name="download" size={14} />Download
      </a>
      {#if ondelete}
        <button
          type="button"
          class="viewer-btn ui-btn ui-btn-danger"
          data-action="delete"
          disabled={deleting}
          title="Delete (Del)"
          onclick={() => ondelete(asset)}
        ><Icon name="trash" size={14} />{deleting ? 'Deleting…' : 'Delete'}</button>
      {/if}
      <span class="viewer-sep" aria-hidden="true"></span>
      <button
        type="button"
        class="viewer-btn ui-btn"
        data-action="details"
        aria-expanded={detailsOpen}
        aria-controls="asset-viewer-details"
        title="Details (I)"
        onclick={toggleDetails}
      ><Icon name="info" size={14} />Details <kbd class="viewer-kbd">I</kbd></button>
      <button
        type="button"
        class="viewer-btn ui-btn ui-btn-icon"
        data-action="shortcuts"
        aria-label="Keyboard shortcuts"
        aria-expanded={helpOpen}
        aria-controls="asset-viewer-shortcuts"
        title="Keyboard shortcuts (?)"
        onclick={() => { helpOpen = !helpOpen; }}
      ><kbd class="viewer-kbd">?</kbd></button>
      <button
        type="button"
        bind:this={closeButton}
        class="viewer-btn ui-btn ui-btn-icon"
        aria-label="Close viewer"
        title="Close (Esc)"
        onclick={onclose}
      ><Icon name="close" size={16} /></button>
    </div>

    <div class="viewer-main">
      <div class="viewer-stage">
        <button
          type="button"
          class="viewer-nav prev ui-btn ui-btn-icon"
          aria-label="Previous asset"
          disabled={!hasPrev}
          onclick={() => hasPrev && onnavigate(index - 1)}
        ><Icon name="chevleft" size={20} /></button>
        {#if asset.media_type === 'video'}
          <!-- svelte-ignore a11y_media_has_caption because generated videos have no caption tracks -->
          <video src={asset.url} controls class="viewer-media"></video>
        {:else}
          <img src={asset.url} alt={asset.prompt || asset.filename} class="viewer-media">
        {/if}
        <button
          type="button"
          class="viewer-nav next ui-btn ui-btn-icon"
          aria-label="Next asset"
          disabled={!hasNext}
          onclick={() => hasNext && onnavigate(index + 1)}
        ><Icon name="chevright" size={20} /></button>
      </div>

      {#if detailsOpen}
        <aside id="asset-viewer-details" class="viewer-details custom-scrollbar" aria-label="Asset details">
          <section>
            <h4 class="viewer-h ui-area-label">Prompt</h4>
            <p class="viewer-prompt">{asset.prompt || 'No prompt recorded.'}</p>
          </section>
          {#if sections?.negativePrompt}
            <section>
              <h4 class="viewer-h ui-area-label">Negative prompt</h4>
              <p class="viewer-prompt">{sections.negativePrompt}</p>
            </section>
          {/if}
          {#if source}
            <section data-testid="viewer-source">
              <h4 class="viewer-h ui-area-label">Upscaled from</h4>
              {#snippet sourceLabel()}
                <span class="truncate">{fileName(source.path)}</span>
                {#if source.width && source.height}<small>{source.width}×{source.height}</small>{/if}
              {/snippet}
              {#if sourceIndex >= 0}
                <button type="button" class="viewer-source" onclick={() => onnavigate(sourceIndex)}>{@render sourceLabel()}</button>
              {:else}
                <p class="viewer-source" title={source.path}>{@render sourceLabel()}</p>
              {/if}
            </section>
          {/if}
          {#if sections}
            {@render factSection('Generation', sections.generation)}
            {@render factSection('Post-processing', sections.postProcessing)}
            {@render factSection('File', sections.file)}
          {/if}
          {#if fallbackReasons.length > 0}
            <section class="ui-alert ui-alert-warning flex-col gap-1" role="note">
              <h4 class="viewer-h ui-area-label">Reuse notice</h4>
              <ul class="list-inside list-disc space-y-0.5 text-ui text-text-secondary">
                {#each fallbackReasons as reason (reason)}
                  <li>{describeFallbackReason(reason)}</li>
                {/each}
              </ul>
            </section>
          {/if}
        </aside>
      {/if}
    </div>

    {#if assets.length > 1}
      <div class="viewer-film custom-scrollbar" aria-label="Assets in this set" bind:this={filmEl}>
        {#each assets as item, itemIndex (item.id)}
          <button
            type="button"
            data-film-index={itemIndex}
            aria-label="Show {item.filename}"
            aria-current={itemIndex === index}
            onclick={() => onnavigate(itemIndex)}
          >
            {#if item.media_type === 'video' && Math.abs(itemIndex - index) > FILM_VIDEO_RADIUS}
              <span class="film-video-placeholder" aria-hidden="true">▶</span>
            {:else if item.media_type === 'video'}
              <video src={item.thumbnail_url || item.url} muted preload="metadata"></video>
            {:else}
              <img src={item.thumbnail_url || item.url} alt="" loading="lazy">
            {/if}
          </button>
        {/each}
      </div>
    {/if}

    {#if helpOpen}
      <div id="asset-viewer-shortcuts" class="viewer-shortcuts ui-overlay" role="note" aria-label="Keyboard shortcuts">
        <h4 class="viewer-h ui-area-label">Keyboard shortcuts</h4>
        <ShortcutList entries={VIEWER_SHORTCUTS} />
      </div>
    {/if}

    <ActionMenu
      open={referenceOpen}
      anchor={referenceButton}
      items={referenceItems}
      label="Use as reference"
      onclose={() => { referenceOpen = false; }}
    />
    <ActionMenu
      open={upscaleOpen}
      anchor={upscaleButton}
      items={upscaleItems}
      label="Upscale"
      onclose={() => { upscaleOpen = false; }}
    />
  </div>
{/if}

{#snippet factSection(title: string, facts: DetailFact[])}
  {#if facts.length > 0}
    <section>
      <h4 class="viewer-h ui-area-label">{title}</h4>
      <dl class="viewer-facts">
        {#each facts as fact (fact.label)}
          <div class:wide={fact.wide}>
            <dt>{fact.label}</dt>
            <dd title={fact.title ?? fact.value}>{fact.value}</dd>
          </div>
        {/each}
      </dl>
    </section>
  {/if}
{/snippet}

<style>
  .asset-viewer { position: fixed; inset: 0; z-index: 100; display: flex; flex-direction: column; background: color-mix(in srgb, var(--color-bg-base) 70%, black); }
  .viewer-bar { display: flex; flex: none; align-items: center; gap: 6px; min-height: 44px; padding: 6px 12px 6px 16px; border-bottom: 1px solid var(--color-border-strong); background: var(--color-bg-surface); }
  .viewer-title { display: flex; min-width: 0; flex-direction: column; margin-right: auto; }
  .viewer-title b { font-size: var(--text-content); font-weight: 600; }
  .viewer-title small { font-size: var(--text-meta); color: var(--color-text-muted); }
  .viewer-sep { width: 1px; height: 20px; margin: 0 2px; background: var(--color-border-subtle); }
  .viewer-kbd { padding: 0 4px; border: 1px solid var(--color-border-strong); border-radius: var(--radius-xs); font-family: var(--font-mono); font-size: var(--text-meta); line-height: 15px; color: var(--color-text-muted); }
  .viewer-shortcuts { position: absolute; top: 52px; right: 12px; z-index: 3; width: 280px; padding: 12px; }
  .viewer-main { display: flex; flex: 1; min-height: 0; }
  .viewer-stage { position: relative; display: flex; flex: 1; min-width: 0; align-items: center; justify-content: center; padding: 20px 64px; }
  .viewer-media { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: var(--radius-xs); }
  .viewer-nav { position: absolute; top: 50%; z-index: 2; width: var(--spacing-bar); height: var(--spacing-bar); transform: translateY(-50%); }
  .viewer-nav.prev { left: 14px; }
  .viewer-nav.next { right: 14px; }
  .viewer-details { display: flex; flex: none; flex-direction: column; gap: 16px; width: 340px; overflow-y: auto; padding: 16px; border-left: 1px solid var(--color-border-strong); background: var(--color-bg-surface); }
  .viewer-h { margin: 0 0 6px; }
  .viewer-prompt { padding: 8px 10px; border: 1px solid var(--color-border-strong); border-radius: var(--radius-md); background: var(--color-bg-base); font-size: var(--text-content); line-height: 1.55; color: var(--color-text-secondary); white-space: pre-wrap; user-select: text; }
  .viewer-source { display: flex; width: 100%; align-items: baseline; justify-content: space-between; gap: 8px; padding: 8px 10px; border: 1px solid var(--color-border-strong); border-radius: var(--radius-md); background: var(--color-bg-base); font-size: var(--text-ui); color: var(--color-text-primary); text-align: left; }
  .viewer-source small { flex: none; font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); }
  button.viewer-source:hover { border-color: var(--color-primary-border); }
  button.viewer-source:focus-visible, .viewer-film button:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: 2px; }
  .viewer-facts { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .viewer-facts div { min-width: 0; padding: 6px 8px; border: 1px solid var(--color-border-subtle); border-radius: var(--radius-sm); background: var(--color-bg-base); }
  .viewer-facts .wide { grid-column: span 2; }
  .viewer-facts dt { font-size: var(--text-meta); color: var(--color-text-muted); }
  .viewer-facts dd { margin: 2px 0 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: var(--font-mono); font-size: var(--text-ui); color: var(--color-text-primary); }
  .viewer-film { display: flex; flex: none; gap: 6px; overflow-x: auto; padding: 8px 12px 12px; }
  /* Auto margins centre a short strip but keep a long one scrollable from its first item. */
  .viewer-film > :first-child { margin-left: auto; }
  .viewer-film > :last-child { margin-right: auto; }
  .viewer-film button { flex: none; width: 44px; height: 44px; overflow: hidden; padding: 0; border: 2px solid transparent; border-radius: var(--radius-sm); opacity: 0.55; transition: opacity 0.12s ease; }
  .viewer-film button:hover { opacity: 0.9; }
  .viewer-film button[aria-current='true'] { border-color: var(--color-primary-main); opacity: 1; }
  .film-video-placeholder { display: grid; width: 100%; height: 100%; place-items: center; background: var(--color-bg-surface); font-size: var(--text-ui); color: var(--color-text-muted); }
  .viewer-film img, .viewer-film video { display: block; width: 100%; height: 100%; object-fit: cover; }
  @media (max-width: 767px) {
    .viewer-bar { flex-wrap: wrap; padding: 8px; }
    .viewer-stage { padding: 12px 52px; }
    .viewer-details { position: absolute; inset: auto 0 0 0; width: auto; max-height: 50%; border-top: 1px solid var(--color-border-strong); border-left: 0; }
  }
</style>
