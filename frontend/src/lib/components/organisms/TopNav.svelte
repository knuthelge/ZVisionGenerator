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
    { id: 'prompts', label: 'Prompts' },
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

  const logoSrc = '/app-static/ziv-icon.svg';
</script>

<header class="app-nav">
  <div class="nav-workflows">
    <h1 class="brand flex items-center gap-2 shrink-0">
      <img src={logoSrc} alt="ziv" class="w-6 h-6 object-contain">
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
  .app-nav { display: flex; align-items: stretch; justify-content: space-between; gap: 16px; min-height: 48px; padding: 0 16px; flex-shrink: 0; background: var(--color-bg-base); border-bottom: 1px solid var(--color-border-subtle); }
  .nav-workflows { display: flex; align-items: stretch; gap: 24px; min-width: 0; }
  .brand { align-self: center; }
  /* Underline tabs that run the full bar height, so the active line sits on the bar's bottom edge. */
  .workflow-tabs, .main-tabs { display: flex; align-items: stretch; gap: 2px; }
  .workflow-tabs { overflow-x: auto; }
  .main-tabs { flex-shrink: 0; }
  .brand { color: var(--color-text-primary); font-family: var(--font-heading); font-size: var(--text-title); font-weight: 800; letter-spacing: -0.01em; }
  .nav-tab { display: flex; align-items: center; margin-bottom: -1px; padding: 0 10px; border-bottom: 2px solid transparent; white-space: nowrap; font-size: var(--text-ui); font-weight: 600; color: var(--color-text-muted); transition: color 0.12s ease, border-color 0.12s ease; }
  .nav-tab:hover { color: var(--color-text-primary); }
  .nav-tab:focus-visible { outline-offset: -2px; }
  .nav-tab.active { border-color: var(--color-primary-main); color: var(--color-text-primary); }
  @media (max-width: 900px) {
    .app-nav { flex-wrap: wrap; gap: 0; padding: 0 12px; }
    .nav-workflows { display: contents; }
    .brand { min-height: 44px; }
    .workflow-tabs { order: 3; width: 100%; min-height: 40px; border-top: 1px solid var(--color-border-subtle); }
  }
</style>
