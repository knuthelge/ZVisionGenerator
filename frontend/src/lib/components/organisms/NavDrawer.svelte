<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import { portal } from '$lib/actions/portal';
  import { pushEscapeLayer } from '$lib/actions/escapeStack';
  import { pan, rubberBand, type PanEnd } from '$lib/actions/pan';
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

  /** How long the drawer takes to slide in or out, or to settle after a swipe. */
  const SLIDE_MS = 240;
  /** A swipe this far toward the edge closes the drawer, as does a flick. */
  const CLOSE_FRACTION = 0.3;
  const FLICK_PX_PER_MS = 0.4;

  // In the page (`present`) and slid in (`shown`) are apart for the length of the slide, so it animates both ways.
  let present = $state(false);
  let shown = $state(false);
  let dragX = $state<number | null>(null);

  $effect(() => {
    if (open) {
      present = true;
      let second = 0;
      const first = requestAnimationFrame(() => { second = requestAnimationFrame(() => { shown = true; }); });
      return () => { cancelAnimationFrame(first); cancelAnimationFrame(second); };
    }
    shown = false;
    dragX = null;
    const timer = setTimeout(() => { present = false; }, SLIDE_MS);
    return () => clearTimeout(timer);
  });

  // A swipe toward the edge takes the drawer with it; let go, it closes or springs back.
  function onSwipeEnd({ offset, velocity }: PanEnd): void {
    const width = panel?.offsetWidth ?? 300;
    dragX = null;
    if (offset < -width * CLOSE_FRACTION || velocity < -FLICK_PX_PER_MS) onclose();
  }

  function go(page: PageId): void {
    onclose();
    if (page !== currentPage) onnavigate(page);
  }

  // Escape closes; focus starts on the current page's entry.
  $effect(() => {
    if (!open) return;
    const previous = document.activeElement as HTMLElement | null;
    panel?.querySelector<HTMLElement>('[aria-current="page"]')?.focus();
    const release = pushEscapeLayer(() => onclose());
    return () => {
      release();
      previous?.focus?.();
    };
  });
</script>

{#if present}
  <div use:portal class="drawer-root ui-viewport" data-shown={shown} inert={!open} data-testid="nav-drawer">
    <button type="button" class="drawer-scrim" aria-label="Close menu" tabindex="-1" onclick={onclose}></button>
    <div
      bind:this={panel}
      use:pan={{ axes: ['x'], onmove: ({ offset }) => { dragX = offset <= 0 ? offset : rubberBand(offset, 16); }, onend: onSwipeEnd, oncancel: () => { dragX = null; } }}
      class="drawer ui-pane"
      class:dragging={dragX !== null}
      style:translate={dragX !== null ? `${dragX}px 0` : undefined}
      role="dialog"
      aria-modal={open ? 'true' : undefined}
      aria-label="Menu"
      tabindex="-1"
    >
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
  .drawer-root { z-index: var(--z-drawer); }
  .drawer-scrim { position: absolute; inset: 0; background: var(--color-scrim); opacity: 0; transition: opacity 240ms ease; }
  .drawer-root[data-shown='true'] .drawer-scrim { opacity: 1; }
  .drawer { position: absolute; top: 0; bottom: 0; left: 0; display: flex; width: min(300px, 82vw); flex-direction: column; gap: 2px; padding: calc(12px + env(safe-area-inset-top, 0px)) 12px calc(12px + env(safe-area-inset-bottom, 0px)) max(12px, env(safe-area-inset-left, 0px)); background: var(--color-bg-surface); outline: none; translate: -100% 0; transition: translate 240ms cubic-bezier(0.32, 0.72, 0, 1); }
  .drawer-root[data-shown='true'] .drawer { translate: 0 0; }
  .drawer.dragging { transition: none; }
  .drawer-brand { display: flex; align-items: center; gap: 8px; margin: 4px 8px 16px; font-family: var(--font-heading); font-size: var(--text-title); font-weight: 800; }
  .drawer-pages { display: flex; flex-direction: column; gap: 2px; }
  .drawer-item { display: flex; align-items: center; gap: 12px; height: 48px; padding: 0 12px; border-radius: var(--radius-sm); font-size: 15px; font-weight: 600; text-align: left; color: var(--color-text-secondary); }
  .drawer-item:active { background: var(--color-bg-surface-hover); }
  .drawer-item.active { background: color-mix(in srgb, var(--color-primary-main) 12%, transparent); color: var(--color-primary-main); }
</style>
