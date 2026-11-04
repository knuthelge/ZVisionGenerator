<script lang="ts">
  import type { ModelLicence } from '$lib/types';
  import { popover } from '$lib/actions/popover';
  import { licenceLinks } from '$lib/state/licenceLinks';
  import Icon from '../atoms/Icon.svelte';

  interface Props {
    /** The records of the model the mark describes; nothing is shown without any. */
    licences: readonly ModelLicence[];
    /** A visible word beside the icon (e.g. "Non-commercial"), which turns the mark into a neutral badge button. */
    label?: string;
    testId?: string;
    class?: string;
  }

  /** How long the pointer rests on the mark before the card opens, so passing over it opens nothing. */
  const OPEN_DELAY_MS = 120;
  /** How long the card waits after the pointer leaves the mark or the card, so the pointer can cross the gap between them. */
  const CLOSE_GRACE_MS = 250;

  let { licences, label, testId, class: extraClass = '' }: Props = $props();

  let open = $state(false);
  // A click, tap, Enter or Space pins the card open: it then stays until it is toggled, dismissed or Escape is pressed. A hover-opened card closes when the pointer leaves.
  let pinned = $state(false);
  let button = $state<HTMLButtonElement>();
  let pop = $state<HTMLDivElement>();
  let timer: ReturnType<typeof setTimeout> | undefined;

  // "Licence: <name>", or "Licences: <a> and <b>" when a model has several records.
  const names = $derived(licences.map((licence) => licence.name));
  const name = $derived(names.length <= 1 ? `Licence: ${names[0] ?? ''}` : `Licences: ${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`);
  const accessibleName = $derived(label ? `${label}. ${name}` : name);

  function clearTimer(): void {
    clearTimeout(timer);
    timer = undefined;
  }

  function close(): void {
    clearTimer();
    open = false;
    pinned = false;
  }

  function toggle(): void {
    clearTimer();
    if (pinned) {
      close();
      return;
    }
    // A card the hover opened is pinned by the click; a closed one opens pinned.
    open = true;
    pinned = true;
  }

  // Touch has no hover: a tap reaches toggle() through its click alone.
  function hoverEnter(event: PointerEvent): void {
    if (event.pointerType === 'touch') return;
    clearTimer();
    if (open) return;
    timer = setTimeout(() => { open = true; }, OPEN_DELAY_MS);
  }

  function hoverLeave(event: PointerEvent): void {
    if (event.pointerType === 'touch') return;
    clearTimer();
    if (!open || pinned) return;
    timer = setTimeout(close, CLOSE_GRACE_MS);
  }

  $effect(() => clearTimer);

  // A pinned card is a dialog: focus moves into it, so Tab walks its links and Escape (handled by the popover action) returns to the mark.
  // A hover-opened one leaves focus where it is.
  $effect(() => {
    if (pinned && pop) pop.focus({ preventScroll: true });
  });

  // Tab past the last link (or Shift+Tab back out of the first stop) closes the popover and hands focus back to the mark.
  function onKeydown(event: KeyboardEvent): void {
    if (event.key !== 'Tab' || !pop) return;
    const stops = Array.from(pop.querySelectorAll<HTMLElement>('a[href]'));
    const active = document.activeElement;
    const leaving = event.shiftKey ? active === pop || active === stops[0] : stops.length === 0 || active === stops[stops.length - 1];
    if (!leaving) return;
    event.preventDefault();
    close();
    button?.focus();
  }
</script>

{#if licences.length > 0}
  <!-- Hovering the mark with a mouse opens the same card a click opens; the card keeps open while the pointer is on it. -->
  <span class="licence-mark-wrap {extraClass}" data-testid={testId ? `${testId}-wrap` : undefined}>
    <button
      bind:this={button}
      type="button"
      class={label ? 'ui-badge licence-badge' : 'licence-mark'}
      aria-label={accessibleName}
      aria-haspopup="dialog"
      aria-expanded={open}
      data-testid={testId}
      onpointerenter={hoverEnter}
      onpointerleave={hoverLeave}
      onclick={toggle}
    >
      <Icon name="info" size={label ? 11 : 14} />{#if label}{label}{/if}
    </button>
  </span>
  {#if open}
    <div
      bind:this={pop}
      use:popover={{ anchor: button ?? null, align: 'start', onclose: close }}
      class="licence-pop ui-overlay"
      role="dialog"
      aria-label={name}
      tabindex="-1"
      data-popover-nested
      data-testid="licence-popover"
      onkeydown={onKeydown}
      onpointerenter={hoverEnter}
      onpointerleave={hoverLeave}
    >
      {#each licences as licence, index (index)}
        <section class="licence-record">
          <p class="licence-name">{licence.name}</p>
          <p class="licence-supplier">{licence.supplier}</p>
          <p class="licence-summary">{licence.summary}</p>
          {#if licence.attribution}<p class="licence-attribution" data-testid="licence-attribution">{licence.attribution}</p>{/if}
          <p class="licence-links">
            {#each licenceLinks(licence) as link (link.href)}
              <a class="ui-btn ui-btn-sm" href={link.href} target="_blank" rel="noopener" aria-label={link.ariaLabel}>{link.label}</a>
            {/each}
          </p>
        </section>
      {/each}
      <p class="licence-note">These are short summaries; the linked licence is what applies.</p>
    </div>
  {/if}
{/if}

<style>
  .licence-mark-wrap { position: relative; display: inline-flex; flex: none; align-items: center; min-width: 0; }
  .licence-mark {
    display: inline-grid;
    place-items: center;
    flex: none;
    width: 20px;
    height: 20px;
    border-radius: var(--radius-xs);
    color: var(--color-text-muted);
  }
  .licence-mark[aria-expanded='true'] { background: var(--color-bg-surface-hover); color: var(--color-text-primary); }
  @media (hover: hover) { .licence-mark:hover { background: var(--color-bg-surface-hover); color: var(--color-text-primary); } }
  .licence-mark:focus-visible, .licence-badge:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: 1px; }
  .licence-badge { cursor: pointer; }
  .licence-badge[aria-expanded='true'] { background: var(--color-bg-surface-hover); color: var(--color-text-primary); }
  @media (hover: hover) { .licence-badge:hover { background: var(--color-bg-surface-hover); color: var(--color-text-primary); } }
  /* A finger gets a 44px target around the 20px mark (and the 18px badge) without the row growing or the hover box swelling. */
  .licence-mark, .licence-badge { position: relative; }
  @media (pointer: coarse) {
    .licence-mark::after { content: ''; position: absolute; inset: -12px; }
    .licence-badge::after { content: ''; position: absolute; inset: -13px -4px; }
  }

  .licence-pop { z-index: var(--z-menu); box-sizing: border-box; width: min(300px, calc(100vw - 16px)); max-height: calc(100vh - 16px); overflow-y: auto; padding: 12px; font-size: var(--text-ui); outline: none; }
  .licence-record + .licence-record { margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--color-border-subtle); }
  .licence-name { font-weight: 600; color: var(--color-text-primary); overflow-wrap: anywhere; }
  .licence-supplier { font-size: var(--text-meta); color: var(--color-text-muted); }
  .licence-summary { margin-top: 4px; color: var(--color-text-secondary); overflow-wrap: anywhere; }
  .licence-note { margin-top: 8px; font-size: var(--text-meta); color: var(--color-text-muted); }
  .licence-attribution { margin-top: 4px; font-size: var(--text-meta); color: var(--color-text-muted); overflow-wrap: anywhere; }
  .licence-links { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
  @media (pointer: coarse) { .licence-links :global(.ui-btn) { min-height: 44px; } }
</style>
