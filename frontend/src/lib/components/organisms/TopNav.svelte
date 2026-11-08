<script lang="ts">
  import { router } from '$lib/state/router.svelte';
  import { draft } from '$lib/state/draft.svelte';
  import { jobStore } from '$lib/state/job.svelte';
  import { isReviewing } from '$lib/state/jobView';
  import { mascotHome } from '$lib/state/mascotHome.svelte';
  import { Badge, Icon, MascotHome, Wordmark } from '$lib/components/atoms';
  import ProjectPicker from '$lib/components/molecules/ProjectPicker.svelte';
  import { navTab, visibleTabs } from '$lib/state/workspaceTab';
  import type { PageId, WorkspaceWorkflow } from '$lib/types';

  interface Props {
    currentPage: PageId;
    /** The workflows the server offers (the contract's values); null until the Workspace has loaded them. */
    workflows?: readonly WorkspaceWorkflow[] | null;
    /** Open the job log drawer. */
    onjoblog?: () => void;
  }
  let { currentPage, workflows = null, onjoblog }: Props = $props();

  const navItems: { id: PageId; label: string }[] = [
    { id: 'workspace', label: 'Workspace' },
    { id: 'prompts', label: 'Prompts' },
    { id: 'gallery', label: 'Gallery' },
    { id: 'models', label: 'Models' },
    { id: 'config', label: 'Config' }
  ];

  const reviewWaiting = $derived(isReviewing(jobStore.current));
  // Until the contract is known Image and Video show; Audio appears only when the server lists an audio workflow.
  const tabs = $derived(visibleTabs(workflows));
  const activeTab = $derived(navTab(draft.state.workflow));
</script>

<header class="app-nav">
  <div class="nav-workflows">
    <h1 class="brand flex items-center gap-2 shrink-0">
      <MascotHome placement={mascotHome.placement} size={24} />
      <Wordmark height={13.4} />
      <span class="sr-only">Z-Vision</span>
    </h1>

    <ProjectPicker variant="nav" tight={currentPage === 'workspace'} />

    {#if currentPage === 'workspace'}
      <nav class="workflow-tabs" aria-label="Workflow">
        {#each tabs as item (item.id)}
          <button
            type="button"
            class="nav-tab"
            class:active={activeTab === item.id}
            aria-pressed={activeTab === item.id}
            onclick={() => draft.setTab(item.id)}
          >
            {item.label}
          </button>
        {/each}
      </nav>
    {/if}
  </div>

  <div class="nav-end">
  <nav class="main-tabs" aria-label="Main navigation">
    {#each navItems as item}
      <button
        type="button"
        class="nav-tab"
        class:active={currentPage === item.id}
        aria-current={currentPage === item.id ? 'page' : undefined}
        onclick={() => router.navigate(item.id)}
      >
        {item.label}
        {#if item.id === 'workspace' && reviewWaiting}<Badge variant="warning">Review</Badge>{/if}
      </button>
    {/each}
  </nav>
  <button type="button" class="nav-tab nav-icon" aria-label="Job log" title="Job log" onclick={onjoblog}><Icon name="clock" size={16} /></button>
  </div>
</header>

<style>
  .app-nav { display: flex; align-items: stretch; justify-content: space-between; gap: 16px; min-height: 48px; padding: 0 16px; flex-shrink: 0; background: color-mix(in srgb, var(--color-bg-overlay) 70%, black); }
  .nav-workflows { display: flex; align-items: stretch; gap: 16px; min-width: 0; }
  .brand { align-self: center; }
  /* Underline tabs that run the full bar height, so the active line sits on the bar's bottom edge. */
  .workflow-tabs, .main-tabs { display: flex; align-items: stretch; gap: 2px; }
  .workflow-tabs { overflow-x: auto; }
  .main-tabs { flex-shrink: 0; }
  .nav-end { display: flex; align-items: stretch; flex-shrink: 0; gap: 2px; }
  .nav-icon { padding: 0 12px; }
  .brand { color: var(--color-text-primary); font-family: var(--font-heading); font-size: var(--text-title); font-weight: 800; letter-spacing: -0.01em; }
  .nav-tab { display: flex; align-items: center; gap: 6px; padding: 0 10px; border-bottom: 2px solid transparent; white-space: nowrap; font-size: var(--text-ui); font-weight: 600; color: var(--color-text-muted); transition: color 0.12s ease, border-color 0.12s ease; }
  @media (hover: hover) { .nav-tab:hover { color: var(--color-text-primary); } }
  .nav-tab:focus-visible { outline-offset: -2px; }
  .nav-tab.active { border-color: var(--color-primary-main); color: var(--color-text-primary); }
  /* Under 1000 px the bar tightens to stay on one row with three workflow tabs (a 744 px portrait tablet fits, with the
     Review badge): the wordmark goes (the mascot and the screen-reader name stay) and the tabs sit closer. */
  @media (max-width: 1000px) {
    .app-nav { gap: 8px; padding: 0 12px; }
    .nav-workflows { gap: 8px; }
    .brand :global(.wordmark) { display: none; }
    .nav-tab { padding: 0 6px; }
    .nav-icon { padding: 0 10px; }
  }
  /* Too narrow for one row: the workflow tabs take a second row. */
  @media (max-width: 739px) {
    .app-nav { flex-wrap: wrap; gap: 0; }
    .nav-workflows { display: contents; }
    .brand { min-height: 44px; }
    .workflow-tabs { order: 3; width: 100%; min-height: 40px; border-top: 1px solid var(--color-border-subtle); }
  }
</style>
