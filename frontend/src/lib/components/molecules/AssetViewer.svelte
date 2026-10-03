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
  import { Icon } from '$lib/components/atoms';
  import { canUseAsReference, describeFallbackReason, type AssetActionHandlers, type ReferenceTarget } from '$lib/state/assetActions';
  import type { GalleryAsset } from '$lib/types';
  import ActionMenu, { type ActionMenuEntry } from './ActionMenu.svelte';
  import { referenceEntries } from './assetMenu';

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
  let closeButton = $state<HTMLButtonElement | null>(null);

  const referenceItems = $derived<ActionMenuEntry[]>(
    asset && onreference ? referenceEntries(asset, onreference, referenceUnavailable) : []
  );

  const facts = $derived<{ label: string; value: string; wide?: boolean }[]>(
    asset
      ? [
          { label: 'Model', value: asset.model || '—' },
          { label: 'Workflow', value: asset.workflow },
          { label: 'Dimensions', value: asset.width && asset.height ? `${asset.width}×${asset.height}` : '—' },
          { label: 'Seed', value: asset.seed != null ? String(asset.seed) : '—' },
          { label: 'Steps', value: asset.steps != null ? String(asset.steps) : '—' },
          { label: 'Guidance', value: asset.guidance != null ? String(asset.guidance) : '—' },
          ...(asset.frame_count ? [{ label: 'Frames', value: String(asset.frame_count) }] : []),
          ...(asset.lora ? [{ label: 'LoRAs', value: asset.lora, wide: true }] : []),
          { label: 'Created', value: new Date(asset.created_at).toLocaleString(), wide: true },
        ]
      : []
  );

  function toggleDetails(): void {
    detailsOpen = !detailsOpen;
    saveDetailsOpen(detailsOpen);
  }

  function isTyping(target: EventTarget | null): boolean {
    return target instanceof HTMLElement && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName));
  }

  $effect(() => {
    if (open && assets.length === 0) onclose();
  });

  $effect(() => {
    if (!open) return;
    queueMicrotask(() => closeButton?.focus());
    function handleKeydown(e: KeyboardEvent): void {
      if (e.defaultPrevented || isTyping(e.target) || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === 'Escape') { e.preventDefault(); onclose(); }
      else if (e.key === 'ArrowLeft' && hasPrev) { e.preventDefault(); onnavigate(index - 1); }
      else if (e.key === 'ArrowRight' && hasNext) { e.preventDefault(); onnavigate(index + 1); }
      else if (e.key === 'i' || e.key === 'I') { e.preventDefault(); toggleDetails(); }
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
          class="viewer-btn surface-button-primary"
          data-action="reuse"
          disabled={!canReuse}
          title={canReuse ? 'Load these settings into the workspace' : 'Reusable settings unavailable'}
          onclick={() => onreuse(asset)}
        ><Icon name="reuse" size={14} />Reuse settings</button>
      {/if}
      {#if onreference && canUseAsReference(asset)}
        <button
          type="button"
          bind:this={referenceButton}
          class="viewer-btn surface-overlay-action"
          data-action="reference"
          aria-haspopup="menu"
          aria-expanded={referenceOpen}
          onclick={() => { referenceOpen = !referenceOpen; }}
        ><Icon name="reference" size={14} />Use as reference<Icon name="chevdown" size={12} class="opacity-70" /></button>
      {/if}
      <a class="viewer-btn surface-overlay-action" data-action="download" href={asset.url} download={asset.filename}>
        <Icon name="download" size={14} />Download
      </a>
      {#if ondelete}
        <button
          type="button"
          class="viewer-btn surface-overlay-action-danger"
          data-action="delete"
          disabled={deleting}
          onclick={() => ondelete(asset)}
        ><Icon name="trash" size={14} />{deleting ? 'Deleting…' : 'Delete'}</button>
      {/if}
      <span class="viewer-sep" aria-hidden="true"></span>
      <button
        type="button"
        class="viewer-btn surface-overlay-action"
        data-action="details"
        aria-expanded={detailsOpen}
        aria-controls="asset-viewer-details"
        title="Details (I)"
        onclick={toggleDetails}
      ><Icon name="info" size={14} />Details <kbd class="viewer-kbd">I</kbd></button>
      <button
        type="button"
        bind:this={closeButton}
        class="viewer-btn viewer-icon surface-overlay-action"
        aria-label="Close viewer"
        title="Close (Esc)"
        onclick={onclose}
      ><Icon name="close" size={16} /></button>
    </div>

    <div class="viewer-main">
      <div class="viewer-stage">
        <button
          type="button"
          class="viewer-nav prev surface-overlay-action"
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
          class="viewer-nav next surface-overlay-action"
          aria-label="Next asset"
          disabled={!hasNext}
          onclick={() => hasNext && onnavigate(index + 1)}
        ><Icon name="chevright" size={20} /></button>
      </div>

      {#if detailsOpen}
        <aside id="asset-viewer-details" class="viewer-details custom-scrollbar" aria-label="Asset details">
          <section>
            <h4 class="viewer-h">Prompt</h4>
            <p class="viewer-prompt surface-card">{asset.prompt || 'No prompt recorded.'}</p>
          </section>
          <section>
            <h4 class="viewer-h">Generation</h4>
            <dl class="viewer-facts">
              {#each facts as fact (fact.label)}
                <div class:wide={fact.wide}>
                  <dt>{fact.label}</dt>
                  <dd title={fact.value}>{fact.value}</dd>
                </div>
              {/each}
            </dl>
          </section>
          {#if fallbackReasons.length > 0}
            <section class="surface-warning rounded-md border px-3 py-2" role="note">
              <h4 class="viewer-h">Reuse notice</h4>
              <ul class="list-inside list-disc space-y-0.5 text-xs text-zinc-300">
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

    <ActionMenu
      open={referenceOpen}
      anchor={referenceButton}
      items={referenceItems}
      label="Use as reference"
      onclose={() => { referenceOpen = false; }}
    />
  </div>
{/if}

<style>
  .asset-viewer { position: fixed; inset: 0; z-index: 100; display: flex; flex-direction: column; background: rgb(7 10 11 / 0.985); }
  .viewer-bar { display: flex; flex: none; align-items: center; gap: 8px; height: 52px; padding: 0 12px 0 16px; border-bottom: 1px solid var(--color-border-subtle); background: rgb(16 22 23 / 0.8); }
  .viewer-title { display: flex; min-width: 0; flex-direction: column; margin-right: auto; }
  .viewer-title b { font-size: 13px; font-weight: 600; }
  .viewer-title small { font-size: 11px; color: var(--color-text-muted); }
  .viewer-btn { display: inline-flex; flex: none; align-items: center; gap: 6px; height: 32px; padding: 0 11px; border-radius: var(--radius-sm); font-family: var(--font-display); font-size: 12.5px; font-weight: 600; }
  .viewer-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .viewer-btn:focus-visible, .viewer-nav:focus-visible, .viewer-film button:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: 2px; }
  .viewer-icon { width: 32px; justify-content: center; padding: 0; }
  .viewer-sep { width: 1px; height: 22px; margin: 0 2px; background: var(--color-border-subtle); }
  .viewer-kbd { padding: 0 5px; border: 1px solid var(--color-border-strong); border-radius: 4px; font-family: var(--font-mono); font-size: 10.5px; line-height: 16px; color: var(--color-text-muted); }
  .viewer-main { display: flex; flex: 1; min-height: 0; }
  .viewer-stage { position: relative; display: flex; flex: 1; min-width: 0; align-items: center; justify-content: center; padding: 20px 72px; }
  .viewer-media { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 4px; box-shadow: 0 20px 60px rgb(0 0 0 / 0.5); }
  .viewer-nav { position: absolute; top: 50%; z-index: 2; display: grid; place-items: center; width: 44px; height: 44px; border-radius: var(--radius-md); transform: translateY(-50%); }
  .viewer-nav:disabled { opacity: 0.35; cursor: not-allowed; }
  .viewer-nav.prev { left: 16px; }
  .viewer-nav.next { right: 16px; }
  .viewer-details { display: flex; flex: none; flex-direction: column; gap: 18px; width: 340px; overflow-y: auto; padding: 16px; border-left: 1px solid var(--color-border-subtle); background: var(--color-bg-base); }
  .viewer-h { margin: 0 0 6px; font-family: var(--font-display); font-size: 11px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--color-text-muted); }
  .viewer-prompt { padding: 10px 12px; font-size: 13px; line-height: 1.55; color: var(--color-zinc-300); white-space: pre-wrap; user-select: text; }
  .viewer-facts { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .viewer-facts div { min-width: 0; padding: 6px 9px; border: 1px solid var(--color-border-subtle); border-radius: var(--radius-sm); background: var(--color-bg-surface); }
  .viewer-facts .wide { grid-column: span 2; }
  .viewer-facts dt { font-size: 10px; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; color: var(--color-text-muted); }
  .viewer-facts dd { margin: 2px 0 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: var(--font-mono); font-size: 12.5px; color: var(--color-zinc-200); }
  .viewer-film { display: flex; flex: none; gap: 6px; overflow-x: auto; padding: 8px 12px 12px; }
  /* Auto margins centre a short strip but keep a long one scrollable from its first item. */
  .viewer-film > :first-child { margin-left: auto; }
  .viewer-film > :last-child { margin-right: auto; }
  .viewer-film button { flex: none; width: 44px; height: 44px; overflow: hidden; padding: 0; border: 2px solid transparent; border-radius: 6px; opacity: 0.55; }
  .viewer-film button:hover { opacity: 0.9; }
  .viewer-film button[aria-current='true'] { border-color: var(--color-primary-main); opacity: 1; }
  .film-video-placeholder { display: grid; width: 100%; height: 100%; place-items: center; background: var(--color-bg-surface); font-size: 12px; color: var(--color-text-muted); }
  .viewer-film img, .viewer-film video { display: block; width: 100%; height: 100%; object-fit: cover; }
  @media (max-width: 767px) {
    .viewer-bar { flex-wrap: wrap; height: auto; padding: 8px; }
    .viewer-stage { padding: 12px 56px; }
    .viewer-details { position: absolute; inset: auto 0 0 0; width: auto; max-height: 50%; border-top: 1px solid var(--color-border-subtle); border-left: 0; }
  }
</style>
