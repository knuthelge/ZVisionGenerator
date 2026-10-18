<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import type { IconName } from '$lib/components/atoms/Icon.svelte';
  import type { PageId } from '$lib/types';

  interface Props {
    open: boolean;
    currentPage: PageId;
    onnavigate: (page: PageId) => void;
    onclose: () => void;
  }

  let { open, currentPage, onnavigate, onclose }: Props = $props();

  const PAGES: { id: PageId; label: string; icon: IconName }[] = [
    { id: 'workspace', label: 'Workspace', icon: 'bolt' },
    { id: 'prompts', label: 'Prompts', icon: 'edit' },
    { id: 'gallery', label: 'Gallery', icon: 'grid' },
    { id: 'models', label: 'Models', icon: 'cube' },
    { id: 'config', label: 'Config', icon: 'cog' }
  ];
  const logoSrc = '/app-static/ziv-icon.svg';

  let panel = $state<HTMLElement | null>(null);

  function go(page: PageId): void {
    onclose();
    if (page !== currentPage) onnavigate(page);
  }

  // Escape closes; focus starts on the current page's entry.
  $effect(() => {
    if (!open) return;
    const previous = document.activeElement as HTMLElement | null;
    panel?.querySelector<HTMLElement>('[aria-current="page"]')?.focus();
    function handleKeydown(event: KeyboardEvent): void {
      if (event.key === 'Escape') {
        event.preventDefault();
        onclose();
      }
    }
    document.addEventListener('keydown', handleKeydown);
    return () => {
      document.removeEventListener('keydown', handleKeydown);
      previous?.focus?.();
    };
  });
</script>

{#if open}
  <div class="drawer-root" data-testid="nav-drawer">
    <button type="button" class="drawer-scrim" aria-label="Close menu" tabindex="-1" onclick={onclose}></button>
    <div bind:this={panel} class="drawer ui-pane" role="dialog" aria-modal="true" aria-label="Menu" tabindex="-1">
      <h1 class="drawer-brand"><img src={logoSrc} alt="" class="h-6 w-6">ziv</h1>
      <nav class="drawer-pages" aria-label="Pages">
        {#each PAGES as page (page.id)}
          <button type="button" class="drawer-item" class:active={page.id === currentPage} aria-current={page.id === currentPage ? 'page' : undefined} onclick={() => go(page.id)}>
            <Icon name={page.icon} size={18} />{page.label}
          </button>
        {/each}
      </nav>
    </div>
  </div>
{/if}

<style>
  .drawer-root { position: fixed; inset: 0; z-index: 140; }
  .drawer-scrim { position: absolute; inset: 0; background: var(--color-scrim); }
  .drawer { position: absolute; top: 0; bottom: 0; left: 0; display: flex; width: min(300px, 82vw); flex-direction: column; gap: 2px; padding: calc(12px + env(safe-area-inset-top, 0px)) 12px calc(12px + env(safe-area-inset-bottom, 0px)) max(12px, env(safe-area-inset-left, 0px)); background: var(--color-bg-surface); outline: none; }
  .drawer-brand { display: flex; align-items: center; gap: 8px; margin: 4px 8px 16px; font-family: var(--font-heading); font-size: var(--text-title); font-weight: 800; }
  .drawer-pages { display: flex; flex-direction: column; gap: 2px; }
  .drawer-item { display: flex; align-items: center; gap: 12px; height: 48px; padding: 0 12px; border-radius: var(--radius-sm); font-size: 15px; font-weight: 600; text-align: left; color: var(--color-text-secondary); }
  .drawer-item:active { background: var(--color-bg-surface-hover); }
  .drawer-item.active { background: color-mix(in srgb, var(--color-primary-main) 12%, transparent); color: var(--color-primary-main); }
</style>
