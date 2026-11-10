<script lang="ts">
  import { browseInfo, browseMediaUrl, browsePreviewUrl } from '$lib/api/browse';
  import { Icon } from '$lib/components/atoms';
  import { formatBytes, formatModifiedFull, previewFacts, type PreviewFact } from '$lib/state/pathBrowser';
  import type { BrowseEntry, BrowseInfo, BrowsePreviewKind, BrowsePurposeName } from '$lib/types';
  import MediaPlayer from './MediaPlayer.svelte';

  interface Props {
    purpose: BrowsePurposeName;
    /** What the purpose previews. */
    kind: BrowsePreviewKind;
    /** The selected row; `null` when nothing is selected. */
    entry: BrowseEntry | null;
    /** `pane`: a column beside the list. `strip`: the picture beside the name and facts, under the list. `sheet`: the phone's preview sheet. */
    arrangement?: 'pane' | 'strip' | 'sheet';
  }

  let { purpose, kind, entry, arrangement = 'pane' }: Props = $props();

  // How long the selection must rest before anything is requested.
  const SETTLE_MS = 150;

  // Only a file the purpose accepts has a preview; a folder shows its icon, any other file its icon and facts.
  const previewable = $derived(entry !== null && entry.kind === 'file' && entry.selectable);
  const path = $derived(previewable && entry ? entry.path : null);

  /** A file the purpose does not take has no preview, so only what the listing knows. */
  function plainFacts(file: BrowseEntry): PreviewFact[] {
    return [
      { label: 'File', value: file.size === null ? '—' : formatBytes(file.size) },
      { label: 'Modified', value: file.modified === null ? '—' : formatModifiedFull(file.modified) },
    ];
  }

  // Each piece of state names the file it belongs to, so a file the selection has moved off never shows another file's state.
  // The selection has rested for `SETTLE_MS`: arrowing down a folder requests nothing until it stops.
  let settledPath = $state<string | null>(null);
  let failedPicturePath = $state<string | null>(null);
  let failedVideoPath = $state<string | null>(null);
  let loaded = $state<{ path: string; info: BrowseInfo } | null>(null);

  const settled = $derived(path !== null && settledPath === path);
  const pictureFailed = $derived(path !== null && failedPicturePath === path);
  const videoFailed = $derived(path !== null && failedVideoPath === path);
  const info = $derived(loaded?.path === path ? loaded.info : null);

  const facts = $derived(entry === null || entry.kind !== 'file' ? [] : previewable ? previewFacts(kind, entry, info) : plainFacts(entry));
  // The pane is the one place a file shows large, so it takes the large picture.
  const stillUrl = $derived(path ? browsePreviewUrl(purpose, path, { size: 'large' }) : '');
  const mediaUrl = $derived(path ? browseMediaUrl(purpose, path) : '');
  // The small picture the row or tile already loaded stands in while the large one arrives.
  const standIn = $derived(path && kind !== 'audio' ? `url("${browsePreviewUrl(purpose, path)}")` : undefined);
  const showPicture = $derived(settled && !pictureFailed);
  const wide = $derived(arrangement === 'strip');

  $effect(() => {
    const target = path;
    if (target === null) return;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      settledPath = target;
      browseInfo(purpose, target, controller.signal)
        .then((result) => { if (!controller.signal.aborted) loaded = { path: target, info: result }; })
        .catch(() => { /* The facts read a dash; the picture and player stand on their own. */ });
    }, SETTLE_MS);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  });
</script>

<div class="preview" class:strip={wide} class:sheet={arrangement === 'sheet'} data-testid="file-preview" data-arrangement={arrangement} data-kind={entry === null ? 'none' : entry.kind === 'directory' ? 'folder' : kind}>
  {#if entry === null}
    <p class="hint">Select a file to preview it.</p>
  {:else}
    <!-- A media error does not bubble, so the stage listens for it on the way down to the player's video. -->
    <div
      class="stage"
      data-testid="preview-stage"
      class:stage-audio={previewable && kind === 'audio'}
      style:background-image={standIn}
      onerrorcapture={(event) => { if (event.target instanceof HTMLVideoElement) failedVideoPath = path; }}
    >
      {#if entry.kind === 'directory'}
        <span class="icon-stage" data-testid="preview-icon"><Icon name="folder" size={48} /></span>
      {:else if !previewable}
        <span class="icon-stage" data-testid="preview-icon"><Icon name="file" size={48} /></span>
      {:else if kind === 'audio'}
        {#if settled}
          {#key mediaUrl}
            <MediaPlayer kind="audio" size="pane" src={mediaUrl} label={entry.name} peaks={info?.peaks ?? []} duration={info?.seconds ?? null} />
          {/key}
        {/if}
      {:else if kind === 'video'}
        {#if settled && !videoFailed}
          {#key mediaUrl}
            <MediaPlayer kind="video" size="pane" src={mediaUrl} label={entry.name} poster={stillUrl} filePath={entry.path} />
          {/key}
        {:else if settled && !pictureFailed}
          <img class="still" src={stillUrl} alt="" onerror={() => { failedPicturePath = path; }}>
        {/if}
      {:else if showPicture}
        <img class="still ui-checkerboard" src={stillUrl} alt="Preview of {entry.name}" onerror={() => { failedPicturePath = path; }}>
      {/if}
      {#if previewable && pictureFailed}
        <span class="icon-stage" data-testid="preview-icon"><Icon name="file" size={48} /></span>
      {/if}
    </div>
    <div class="details">
      <p class="name" data-testid="preview-name">{entry.name}</p>
      {#if videoFailed}
        <p class="note" role="status">This file can’t be played.</p>
      {/if}
      {#if entry.kind === 'directory'}
        <p class="kind-note">Folder</p>
      {:else}
        <dl class="facts">
          {#each facts as fact (fact.label)}
            <dt>{fact.label}</dt><dd>{fact.value}</dd>
          {/each}
        </dl>
      {/if}
    </div>
  {/if}
</div>

<style>
  .preview { display: flex; flex-direction: column; gap: 10px; height: 100%; min-height: 0; padding: 12px; overflow-y: auto; color: var(--color-text-secondary); font-size: var(--text-ui); }
  .strip { flex-direction: row; gap: 14px; }
  .hint { margin: auto; color: var(--color-text-muted); text-align: center; }
  .stage { display: flex; flex: 1 0 220px; align-items: center; justify-content: center; height: auto; min-width: 0; border-radius: var(--radius-sm); background: var(--color-bg-base) center / contain no-repeat; overflow: hidden; }
  .strip .stage { flex: none; width: 45%; min-height: 0; }
  .sheet .stage { flex: none; height: clamp(220px, 45dvh, 420px); }
  /* A phone held sideways has no height to spare. */
  @media (max-height: 499px) { .sheet .stage { height: 220px; } }
  .stage-audio { flex: none; height: auto; min-height: 110px; background: transparent; overflow: visible; }
  .strip .stage-audio { align-self: flex-start; }
  .still { display: block; max-width: 100%; max-height: 100%; object-fit: contain; }
  .icon-stage { display: flex; align-items: center; justify-content: center; color: var(--color-text-muted); }
  .details { display: flex; flex: none; flex-direction: column; gap: 8px; min-width: 0; min-height: 0; }
  .strip .details { flex: 1; }
  .name { color: var(--color-text-primary); font-size: var(--text-content); font-weight: 600; overflow-wrap: anywhere; }
  .kind-note { color: var(--color-text-muted); }
  .note { color: var(--color-text-muted); }
  .facts { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 4px 12px; }
  dt { color: var(--color-text-muted); }
  dd { color: var(--color-text-primary); font-family: var(--font-mono); font-size: var(--text-meta); overflow-wrap: anywhere; }
</style>
