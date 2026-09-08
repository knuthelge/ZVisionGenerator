<script lang="ts">
  import { router } from '$lib/state/router.svelte';
  import { draft } from '$lib/state/draft.svelte';
  import type { PageId, Workflow } from '$lib/types';

  interface Props {
    currentPage: PageId;
  }
  let { currentPage }: Props = $props();

  const navItems: { id: PageId; label: string }[] = [
    { id: 'workspace', label: 'Workspace' },
    { id: 'gallery', label: 'Gallery' },
    { id: 'models', label: 'Models' },
    { id: 'config', label: 'Config' }
  ];

  const workflowItems: { id: Workflow; label: string }[] = [
    { id: 'txt2img', label: 'Text to Image' },
    { id: 'img2img', label: 'Image to Image' },
    { id: 'img2vid', label: 'Image to Video' },
    { id: 'txt2vid', label: 'Text to Video' }
  ];

  const logoSrc = '/app-static/zvision-white.png';
</script>

<header class="app-nav">
  <div class="nav-workflows">
    <h1 class="text-sm font-semibold tracking-tight text-white flex items-center gap-2 shrink-0">
      <img src={logoSrc} alt="ziv" class="w-4 h-4 object-contain">
      ziv
    </h1>

    {#if currentPage === 'workspace'}
      <nav class="workflow-tabs" aria-label="Workflow">
        {#each workflowItems as item}
          <button
            type="button"
            class="nav-tab"
            class:active={draft.state.workflow === item.id}
            aria-pressed={draft.state.workflow === item.id}
            onclick={() => draft.update('workflow', item.id)}
          >
            {item.label}
          </button>
        {/each}
      </nav>
    {/if}
  </div>

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
      </button>
    {/each}
  </nav>
</header>

<style>
  .app-nav { display: flex; align-items: center; justify-content: space-between; gap: 16px; min-height: 48px; padding: 0 16px; flex-shrink: 0; background: var(--color-bg-base); border-bottom: 1px solid var(--color-border-subtle); }
  .nav-workflows { display: flex; align-items: center; gap: 24px; min-width: 0; }
  .workflow-tabs, .main-tabs { display: flex; align-items: center; gap: 2px; }
  .workflow-tabs { overflow-x: auto; }
  .main-tabs { flex-shrink: 0; }
  .nav-tab { white-space: nowrap; padding: 7px 10px; border-radius: 5px; font-size: 12px; font-weight: 500; color: var(--color-text-muted); transition: background-color 120ms, color 120ms; }
  .nav-tab:hover { background: var(--color-bg-surface); color: var(--color-text-primary); }
  .main-tabs .active { background: var(--color-bg-surface-hover); color: var(--color-text-primary); }
  .workflow-tabs .active { background: var(--color-primary-subtle); color: var(--color-primary-main); }
  @media (max-width: 900px) {
    .app-nav { flex-wrap: wrap; gap: 0; padding: 8px 12px; }
    .nav-workflows { display: contents; }
    .workflow-tabs { order: 3; width: 100%; margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--color-border-subtle); }
  }
</style>
