<script lang="ts">
  import type { ActionMenuEntry } from '$lib/components/molecules/ActionMenu.svelte';
  import ActionMenu from '$lib/components/molecules/ActionMenu.svelte';
  import Button from '$lib/components/atoms/Button.svelte';
  import Icon from '$lib/components/atoms/Icon.svelte';
  import Toggle from '$lib/components/atoms/Toggle.svelte';
  import Tooltip from '$lib/components/atoms/Tooltip.svelte';

  interface Props {
    /** State of Start at login; null hides the toggle. */
    autostart: boolean | null;
    /** Toggle and More are shown but disabled. */
    busy: boolean;
    /** The More menu's entries; null hides More. */
    menu: ActionMenuEntry[] | null;
    /** Resolves to the new state of start at login. */
    onautostart: (enabled: boolean) => Promise<boolean>;
    onquit: () => void;
  }

  let { autostart, busy, menu, onautostart, onquit }: Props = $props();

  const HINT = 'Starts in the background at login.';
  const hintId = $props.id();

  let checked = $state(false);
  $effect(() => {
    checked = autostart ?? false;
  });

  let moreOpen = $state(false);
  let moreAnchor = $state<HTMLElement | null>(null);
  let loginEl = $state<HTMLElement>();

  async function change(): Promise<void> {
    const wanted = checked;
    try {
      checked = await onautostart(wanted);
    } catch {
      checked = autostart ?? false;
    }
  }

  // The switch carries the hint for screen readers; Toggle has no prop for aria-describedby.
  $effect(() => {
    loginEl?.querySelector('input[role="switch"]')?.setAttribute('aria-describedby', hintId);
  });

  // Tooltip shows on its trigger's own focus; the focus here lands on the switch inside, so pass it on.
  function relay(type: 'focus' | 'blur'): void {
    loginEl?.querySelector('span[aria-describedby]')?.dispatchEvent(new FocusEvent(type));
  }
</script>

<footer class="bar" data-testid="quiet-bar">
  {#if autostart !== null}
    <span class="login" bind:this={loginEl} onfocusin={() => relay('focus')} onfocusout={() => relay('blur')}>
      <Tooltip text={HINT} focusable={false} placement="top">
        <Toggle bind:checked label="Start at login" disabled={busy} onchange={change} />
      </Tooltip>
      <span id={hintId} class="sr-only">{HINT}</span>
    </span>
  {/if}
  <span class="spacer"></span>
  {#if menu}
    <span bind:this={moreAnchor} class="more">
      <Button
        variant="quiet"
        icon
        disabled={busy}
        aria-label="More actions"
        data-testid="more-button"
        aria-haspopup="menu"
        aria-expanded={moreOpen}
        onclick={() => (moreOpen = !moreOpen)}
      >
        <Icon name="more" size={16} />
      </Button>
    </span>
    <ActionMenu open={moreOpen} anchor={moreAnchor} items={menu} label="More" onclose={() => (moreOpen = false)} />
  {/if}
  <Button variant="quiet" data-quit onclick={onquit}>Quit</Button>
</footer>

<style>
  .bar {
    flex: none;
    display: flex;
    align-items: center;
    gap: 2px;
    height: 48px;
    padding: 0 10px 0 20px;
    border-top: 1px solid var(--color-border-subtle);
  }
  .spacer { flex: 1; }
  .more { display: inline-flex; }
  .bar :global(.ui-btn-quiet) { color: var(--color-text-secondary); }
  .bar :global(.toggle-label) { color: var(--color-text-secondary); }
</style>
