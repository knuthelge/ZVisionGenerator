<script lang="ts">
  import { draft } from '$lib/state/draft.svelte';
  import { Icon, Spinner } from '$lib/components/atoms';
  import { InspectorRow, InspectorSection, PathBrowser, PathInput } from '$lib/components/molecules';
  import { fileName } from '$lib/components/molecules/assetDetails';
  import { INPUT_DEBOUNCE_MS } from '$lib/state/debounce';
  import { extendSourceFacts, type ExtendGuard } from '$lib/state/extend';
  import { extendInspection } from '$lib/state/extendInspection.svelte';
  import type { WorkspaceContext } from '$lib/types';
  import SourceVideoFeedback from './SourceVideoFeedback.svelte';

  interface Props {
    context: WorkspaceContext;
    /** A submit is in flight. */
    busy: boolean;
    /** Whether the chosen source lets an extension start. */
    guard: ExtendGuard;
  }

  let { context, busy, guard }: Props = $props();

  const info = $derived(draft.extendSource);
  const sourcePath = $derived(draft.state.extendSourcePath);
  const sourceStatus = $derived(extendInspection.statusFor(sourcePath));
  const checking = $derived(sourcePath !== null && guard.kind === 'checking');
  const facts = $derived(info ? extendSourceFacts(info) : []);
  const name = $derived(info?.filename ?? (sourcePath ? fileName(sourcePath) : ''));
  const found = $derived(guard.kind === 'ok');
  // The server reports a source it could not find, or Reuse recorded one that has since moved; both ask for another choice.
  const missing = $derived(draft.state.missingSourceVideo || sourceStatus === 'not_found');
  // The last frame of a file inside the output folder; elsewhere the browser cannot load it, so an icon stands in.
  const thumbUrl = $derived(info?.media_url ? `${info.media_url}#t=${Math.max(0, (info.duration_seconds ?? 0) - 0.05).toFixed(2)}` : null);

  // The path box keeps what is typed; the draft takes it 300 ms after the last keystroke, on Enter, or on leaving the box.
  let pathText = $state(draft.state.extendSourcePath ?? '');
  let pathTimer: ReturnType<typeof setTimeout> | undefined;
  // Typed text the draft has not taken yet; plain, since state reads are not reliable once the section unmounts.
  let pendingPath: string | undefined;
  let browsing = $state(false);

  $effect(() => {
    pathText = sourcePath ?? '';
  });
  // Leaving the extend workflow mid-typing still takes the path; it is checked when the workflow comes back.
  $effect(() => () => {
    clearTimeout(pathTimer);
    if (pendingPath !== undefined) draft.setExtendSourcePath(pendingPath.trim() || null);
  });

  /** Take a newly chosen source (or none): its text replaces both boxes once the server has described it. */
  function chooseSource(path: string | null): void {
    draft.setExtendSourcePath(path);
    void extendInspection.inspect(context, { replaceText: true });
  }

  function commitPath(force = false): void {
    clearTimeout(pathTimer);
    pendingPath = undefined;
    const path = pathText.trim() || null;
    if (path !== sourcePath) chooseSource(path);
    else if (force && path !== null) void extendInspection.inspect(context);
  }

  function onPathInput(value: string): void {
    pathText = value;
    pendingPath = value;
    clearTimeout(pathTimer);
    pathTimer = setTimeout(() => commitPath(), INPUT_DEBOUNCE_MS);
  }

  function onPathKeydown(event: KeyboardEvent): void {
    // Enter checks the path at once; it must not submit the form, whose Generate waits for the check.
    if (event.key !== 'Enter') return;
    event.preventDefault();
    commitPath(true);
  }

</script>

<InspectorSection id="source-video" title="Source video" summary={name || 'None'} active={found}>
  <InspectorRow label="Video">
    <div class="reference-thumb">
      {#if thumbUrl}
        <video src={thumbUrl} muted preload="metadata" aria-hidden="true"></video>
      {:else}
        <Icon name="film" size={16} />
      {/if}
    </div>
    <span class="reference-name" class:reference-empty={!name} title={sourcePath ?? ''}>{name || 'Browse or enter a path'}</span>
    {#if checking}<Spinner size="sm" />{/if}
    <button
      id="ws-source-browse"
      type="button"
      class="ui-btn ui-btn-row"
      aria-label="Browse for a source video"
      title={busy ? 'Available once the job is submitted.' : 'Browse for a video'}
      disabled={busy}
      onclick={() => { browsing = true; }}
    ><Icon name="folder" size={14} /></button>
    <button
      id="ws-source-clear"
      type="button"
      class="ui-btn ui-btn-row"
      aria-label="Clear source video"
      title={busy ? 'Available once the job is submitted.' : 'Clear'}
      disabled={busy || (sourcePath === null && pathText === '')}
      onclick={() => { pathText = ''; chooseSource(null); }}
    ><Icon name="close" size={14} /></button>
  </InspectorRow>
  {#if info && facts.length > 0}
    <p class="row-hint source-facts" data-testid="extend-source-facts">
      {#each facts as fact, index (fact.key)}{#if index > 0}{' · '}{/if}<span class="fact" title={fact.title}>{fact.text}</span>{/each}
    </p>
  {/if}
  <InspectorRow label="Path" forId="ws-source-path">
    <PathInput
      id="ws-source-path"
      name="source_video"
      purpose="source_video"
      class="inspector-text"
      plain
      placeholder="/path/to/clip.mp4"
      ariaInvalid={missing}
      ariaDescribedby={missing ? 'ws-source-help' : undefined}
      value={pathText}
      oninput={onPathInput}
      onpick={(path) => { pathText = path; commitPath(true); }}
      onchange={() => commitPath()}
      onkeydown={onPathKeydown}
    />
  </InspectorRow>
  <SourceVideoFeedback id="ws-source-help" />
</InspectorSection>

<PathBrowser bind:open={browsing} purpose="source_video" initialPath={pathText.trim() || sourcePath} onchoose={(path) => chooseSource(path)} />

<style>
  /* Facts wrap between facts only, so a fact such as "145 frames" never splits. */
  .source-facts { font-family: var(--font-mono); }
  .fact { white-space: nowrap; }
  :global(#ws-source-help > *) { margin: 0 12px 6px; }

  .reference-thumb { display: grid; flex: none; place-items: center; width: 40px; height: 40px; margin: 4px 0; overflow: hidden; border-radius: var(--radius-sm); background: var(--color-bg-surface-hover); color: var(--color-text-muted); }
  .reference-thumb video { width: 100%; height: 100%; object-fit: cover; }
  .reference-empty { color: var(--color-text-muted); }
  .reference-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-ui); }
</style>
