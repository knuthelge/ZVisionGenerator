<script lang="ts">
  import { draft } from '$lib/state/draft.svelte';
  import { errorText } from '$lib/api/client';
  import { openPathPicker } from '$lib/api/promptFiles';
  import { Icon, Spinner } from '$lib/components/atoms';
  import { Alert, InspectorRow, InspectorSection } from '$lib/components/molecules';
  import { fileName } from '$lib/components/molecules/assetDetails';
  import { INPUT_DEBOUNCE_MS } from '$lib/state/debounce';
  import { extendSourceFacts, type ExtendGuard } from '$lib/state/extend';
  import { extendInspection } from '$lib/state/extendInspection.svelte';
  import type { WorkspaceContext } from '$lib/types';

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
  const sourceFailure = $derived(extendInspection.failureFor(sourcePath));
  const checking = $derived(sourcePath !== null && guard.kind === 'checking');
  const facts = $derived(info ? extendSourceFacts(info) : []);
  const name = $derived(info?.filename ?? (sourcePath ? fileName(sourcePath) : ''));
  const found = $derived(guard.kind === 'ok');
  // The server reports a source it could not find, or Reuse recorded one that has since moved; both ask for another choice.
  const missing = $derived(draft.state.missingSourceVideo || sourceStatus === 'not_found');
  const problems = $derived(info?.problems ?? []);
  // The last frame of a file inside the output folder; elsewhere the browser cannot load it, so an icon stands in.
  const thumbUrl = $derived(info?.media_url ? `${info.media_url}#t=${Math.max(0, (info.duration_seconds ?? 0) - 0.05).toFixed(2)}` : null);

  // The path box keeps what is typed; the draft takes it 300 ms after the last keystroke, on Enter, or on leaving the box.
  let pathText = $state(draft.state.extendSourcePath ?? '');
  let pathTimer: ReturnType<typeof setTimeout> | undefined;
  // Typed text the draft has not taken yet; plain, since state reads are not reliable once the section unmounts.
  let pendingPath: string | undefined;
  let pickerMessage = $state<string | null>(null);
  let picking = $state(false);

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
    pickerMessage = null;
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

  async function browseSource(): Promise<void> {
    picking = true;
    pickerMessage = null;
    try {
      const result = await openPathPicker({ kind: 'existing_file', purpose: 'source_video', initial_path: sourcePath });
      if (result.status === 'selected' && result.path) chooseSource(result.path);
      else if (result.status === 'unsupported') pickerMessage = result.message ?? 'Path picking is not supported on this host.';
      else if (result.status === 'error') pickerMessage = result.message ?? 'Path picker failed.';
    } catch (error) {
      pickerMessage = errorText(error, 'Failed to open the path picker.');
    } finally {
      picking = false;
    }
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
      disabled={busy || picking}
      onclick={browseSource}
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
    <input
      id="ws-source-path"
      name="source_video"
      type="text"
      class="inspector-text"
      placeholder="/path/to/clip.mp4"
      spellcheck="false"
      autocomplete="off"
      aria-invalid={missing ? 'true' : undefined}
      aria-describedby={missing ? 'ws-source-help' : undefined}
      value={pathText}
      oninput={(e) => onPathInput(e.currentTarget.value)}
      onchange={() => commitPath()}
      onkeydown={onPathKeydown}
    >
  </InspectorRow>
  <div id="ws-source-help" class="source-feedback" aria-live="polite">
    {#if pickerMessage}
      <p class="ui-help ui-help-warning">{pickerMessage}</p>
    {:else if draft.state.missingSourceVideo}
      <Alert tone="warning" testId="extend-source-moved">The source video has moved or been deleted; choose it again to extend.</Alert>
    {:else if sourceStatus === 'not_found'}
      <p class="ui-help ui-help-error">No video at this path.</p>
    {:else if sourceFailure}
      <p class="ui-help ui-help-error">{sourceFailure}</p>
    {:else if problems.length > 0}
      <Alert tone="error" testId="extend-source-problems">
        <p>This video can't be extended:</p>
        <ul class="problem-list">
          {#each problems as problem (problem.code)}<li>{problem.message}</li>{/each}
        </ul>
      </Alert>
    {/if}
  </div>
</InspectorSection>

<style>
  .inspector-text { width: 100%; min-width: 0; height: 24px; padding: 0 6px; border: 1px solid transparent; border-radius: 4px; background: transparent; font-family: var(--font-mono); font-size: var(--text-ui); color: var(--color-text-primary); }
  .inspector-text:hover { background: var(--color-bg-surface-hover); }
  .inspector-text:focus { outline: 2px solid var(--color-primary-main); outline-offset: 2px; background: var(--color-bg-surface); }
  .inspector-text[aria-invalid='true'] { box-shadow: inset 0 0 0 1px var(--color-error); }
  .row-hint { padding: 0 12px 6px 116px; font-size: var(--text-meta); color: var(--color-text-muted); }

  /* Facts wrap between facts only, so a fact such as "145 frames" never splits. */
  .source-facts { font-family: var(--font-mono); }
  .fact { white-space: nowrap; }
  .source-feedback > :global(*) { margin: 0 12px 6px; }
  .problem-list { margin-top: 2px; padding-left: 16px; list-style: disc; }

  .reference-thumb { display: grid; flex: none; place-items: center; width: 40px; height: 40px; margin: 4px 0; overflow: hidden; border-radius: var(--radius-sm); background: var(--color-bg-surface-hover); color: var(--color-text-muted); }
  .reference-thumb video { width: 100%; height: 100%; object-fit: cover; }
  .reference-empty { color: var(--color-text-muted); }
  .reference-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-ui); }
</style>
