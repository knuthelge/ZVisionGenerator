<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import { draft } from '$lib/state/draft.svelte';
  import { jobStore } from '$lib/state/job.svelte';
  import { jobStatusLine } from '$lib/state/jobStatusLine';
  import { router } from '$lib/state/router.svelte';
  import type { PageId, Workflow } from '$lib/types';

  interface Props {
    currentPage: PageId;
    /** The workflows the server offers (the contract's values); null until the Workspace has loaded them. */
    workflows?: readonly Workflow[] | null;
    /** Open the navigation drawer. */
    onmenu: () => void;
    /** The job state was tapped (the queue panel opens when jobs wait). */
    onstatus?: () => void;
    /** Open the job log drawer. */
    onjoblog?: () => void;
  }

  let { currentPage, workflows = null, onmenu, onstatus, onjoblog }: Props = $props();

  const PAGE_TITLES: Record<PageId, string> = { workspace: 'Workspace', prompts: 'Prompts', gallery: 'Gallery', models: 'Models', config: 'Config' };
  const WORKFLOW_LABELS: { id: Workflow; label: string }[] = [
    { id: 'txt2img', label: 'Text to Image' },
    { id: 'img2img', label: 'Image to Image' },
    { id: 'img2vid', label: 'Image to Video' },
    { id: 'txt2vid', label: 'Text to Video' },
    { id: 'extend', label: 'Extend Video' }
  ];
  // Until the contract is known the four always-available workflows show; Extend Video appears only when the server lists it.
  const visibleWorkflows = $derived(WORKFLOW_LABELS.filter((item) => (workflows === null ? item.id !== 'extend' : workflows.includes(item.id))));
  const workflowLabel = $derived(WORKFLOW_LABELS.find((item) => item.id === draft.state.workflow)?.label ?? '');
  const onWorkspace = $derived(currentPage === 'workspace');
  const status = $derived(jobStatusLine(jobStore.current, jobStore.queuedJobs.length));
  const logoSrc = '/app-static/ziv-icon.svg';
</script>

<!-- The phone's top strip: the menu on every page; on the Workspace the workflow, the job state and the Gallery;
     elsewhere an arrow back to the Workspace and the page's name; the job log button ends the strip. It pads for the status bar in a standalone app. -->
<header class="top-strip" data-testid="top-strip">
  <button type="button" class="strip-btn" aria-label="Menu" onclick={onmenu}><Icon name="menu" size={20} /></button>
  {#if onWorkspace}
    <label class="strip-select ui-select-shell">
      <img src={logoSrc} alt="" class="h-5 w-5 shrink-0">
      <span class="truncate">{workflowLabel}</span>
      <Icon name="chevdown" size={12} class="opacity-70" />
      <select class="strip-native" aria-label="Workflow" value={draft.state.workflow} onchange={(event) => draft.update('workflow', (event.currentTarget as HTMLSelectElement).value as Workflow)}>
        {#each visibleWorkflows as item (item.id)}
          <option value={item.id}>{item.label}</option>
        {/each}
      </select>
    </label>
    <button type="button" class="strip-status" data-tone={status.tone} data-testid="strip-status" onclick={onstatus}>
      <i class="strip-dot" aria-hidden="true"></i>{status.text}
    </button>
    <button type="button" class="strip-btn" aria-label="Gallery" onclick={() => router.navigate('gallery')}><Icon name="grid" size={20} /></button>
  {:else}
    <button type="button" class="strip-btn" aria-label="Back to the Workspace" onclick={() => router.navigate('workspace')}><Icon name="arrowleft" size={20} /></button>
    <h1 class="strip-title">{PAGE_TITLES[currentPage]}</h1>
    {#if status.tone !== 'idle'}
      <button type="button" class="strip-status" data-tone={status.tone} onclick={() => router.navigate('workspace')}>
        <i class="strip-dot" aria-hidden="true"></i>{status.text}
      </button>
    {/if}
  {/if}
  <button type="button" class="strip-btn" aria-label="Job log" onclick={onjoblog}><Icon name="clock" size={20} /></button>
</header>

<style>
  .top-strip { display: flex; flex-shrink: 0; align-items: center; gap: 6px; height: calc(44px + env(safe-area-inset-top, 0px)); padding: env(safe-area-inset-top, 0px) max(8px, env(safe-area-inset-right, 0px)) 0 max(8px, env(safe-area-inset-left, 0px)); background: color-mix(in srgb, var(--color-bg-overlay) 70%, black); }
  .strip-btn { display: grid; flex: none; place-items: center; width: 40px; height: 40px; border-radius: var(--radius-sm); color: var(--color-text-primary); }
  .strip-btn:active { background: var(--color-bg-surface-hover); }
  .strip-select { position: relative; display: inline-flex; min-width: 0; max-width: 48vw; align-items: center; gap: 6px; height: 32px; padding: 0 8px 0 6px; border-radius: 16px; font-size: var(--text-content); font-weight: 700; color: var(--color-text-primary); }
  .strip-native { position: absolute; inset: 0; width: 100%; opacity: 0; }
  .strip-status { display: inline-flex; flex: 1; min-width: 0; align-items: center; justify-content: flex-end; gap: 6px; margin-left: auto; padding: 0 4px; overflow: hidden; white-space: nowrap; font-size: var(--text-meta); font-weight: 600; color: var(--color-text-muted); text-overflow: ellipsis; }
  .strip-dot { flex: none; width: 6px; height: 6px; border-radius: 50%; background: var(--color-text-muted); }
  .strip-status[data-tone='running'] { color: var(--color-primary-main); }
  .strip-status[data-tone='running'] .strip-dot { background: var(--color-primary-main); animation: strip-pulse 1.2s infinite; }
  .strip-status[data-tone='review'] { color: var(--color-warning); }
  .strip-status[data-tone='review'] .strip-dot { background: var(--color-warning); }
  .strip-title { flex: 1; min-width: 0; overflow: hidden; font-family: var(--font-heading); font-size: 15px; font-weight: 800; white-space: nowrap; text-overflow: ellipsis; }
  @keyframes strip-pulse { 50% { opacity: 0.3; } }
</style>
