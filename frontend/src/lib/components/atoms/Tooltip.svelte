<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    /** Plain text; ignored when `content` is given. */
    text?: string;
    /** Laid-out content in place of the text, such as the memory estimate. It takes pointer events while open, so a tap on it (touch has no hover to leave) closes it instead of reaching the control beneath. */
    content?: Snippet;
    /** Preferred side; the tooltip flips when that side lacks room in the viewport. */
    placement?: 'top' | 'bottom';
    /** Which edge of the trigger the tooltip lines up with; it is always kept inside the viewport. */
    align?: 'start' | 'end';
    /** Make the trigger a tab stop. Turn off when the tooltip only repeats nearby text (e.g. per-row names). */
    focusable?: boolean;
    /** The content has controls to use: it stays open while the pointer is over it and while focus is inside it, and a tap on it does not close it. Escape inside it returns focus to the trigger. */
    interactive?: boolean;
    testId?: string;
    class?: string;
    children?: Snippet;
  }

  let { text = '', content, placement = 'top', align = 'start', focusable = true, interactive = false, testId, class: extraClass = '', children }: Props = $props();

  const GAP_PX = 6;
  const EDGE_PX = 8;
  // Interactive content sits a gap away from its trigger: the pointer gets this long to cross it.
  const LEAVE_DELAY_MS = 150;
  const id = $props.id();
  let trigger = $state<HTMLSpanElement>();
  let bubble = $state<HTMLSpanElement>();
  let open = $state(false);
  let position = $state('');
  let leaveTimer: ReturnType<typeof setTimeout> | undefined;

  // Fixed to the viewport, so no `overflow: hidden` container (cards, tables) can clip it.
  function place(): void {
    if (!trigger) return;
    const rect = trigger.getBoundingClientRect();
    const height = bubble?.offsetHeight ?? 0;
    const width = bubble?.offsetWidth ?? 0;
    const roomAbove = rect.top - GAP_PX;
    const roomBelow = window.innerHeight - rect.bottom - GAP_PX;
    const fitsPreferred = placement === 'top' ? height <= roomAbove : height <= roomBelow;
    const above = fitsPreferred ? placement === 'top' : roomAbove > roomBelow;
    const top = above ? rect.top - GAP_PX - height : rect.bottom + GAP_PX;
    // Line up with the requested edge of the trigger, then keep the whole bubble inside the viewport.
    const preferredLeft = align === 'end' ? rect.right - width : rect.left;
    const left = Math.max(EDGE_PX, Math.min(preferredLeft, window.innerWidth - width - EDGE_PX));
    position = `top: ${Math.max(0, top)}px; left: ${left}px;`;
  }

  function show(): void {
    clearTimeout(leaveTimer);
    if (!open) place();
    open = true;
  }

  function hide(): void {
    clearTimeout(leaveTimer);
    open = false;
  }

  // Touch has no hover: a tap toggles the tooltip (on pointerdown, before the focus it causes), and the
  // pointerleave that follows a tap is ignored.
  function handlePointerEnter(event: PointerEvent): void {
    if (event.pointerType !== 'touch') show();
  }

  function handlePointerLeave(event: PointerEvent): void {
    if (event.pointerType === 'touch') return;
    if (interactive) leaveTimer = setTimeout(hide, LEAVE_DELAY_MS);
    else hide();
  }

  // Focus moving from the trigger into the content, or between its controls, keeps it open.
  function handleFocusOut(event: FocusEvent): void {
    if (event.relatedTarget instanceof Node && trigger?.contains(event.relatedTarget)) return;
    hide();
  }

  function handlePointerDown(event: PointerEvent): void {
    if (event.pointerType !== 'touch') return;
    if (interactive && event.target instanceof Node && bubble?.contains(event.target)) return;
    if (open) hide();
    else show();
  }

  // Escape closes the tooltip; with focus inside its content, focus goes back to the trigger first, so it is not lost to the page.
  // Focusing the trigger shows the tooltip again, so it closes after.
  function handleKeydown(event: KeyboardEvent): void {
    if (event.key !== 'Escape') return;
    if (bubble?.contains(document.activeElement)) trigger?.focus();
    hide();
  }

  // A fixed tooltip would drift from its trigger when any ancestor scrolls, so close it instead.
  $effect(() => () => clearTimeout(leaveTimer));

  $effect(() => {
    if (!open) return;
    window.addEventListener('scroll', hide, true);
    window.addEventListener('resize', hide);
    return () => {
      window.removeEventListener('scroll', hide, true);
      window.removeEventListener('resize', hide);
    };
  });
</script>

<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_static_element_interactions -->
<span
  bind:this={trigger}
  class="relative inline-flex min-w-0 {extraClass}"
  tabindex={focusable ? 0 : undefined}
  aria-describedby={id}
  data-testid={testId}
  onpointerenter={handlePointerEnter}
  onpointerleave={handlePointerLeave}
  onpointerdown={handlePointerDown}
  onfocus={show}
  onblur={interactive ? undefined : hide}
  onfocusout={interactive ? handleFocusOut : undefined}
  onkeydown={handleKeydown}
>
  {@render children?.()}
  <span
    bind:this={bubble}
    role={interactive ? 'group' : 'tooltip'}
    {id}
    style="{position}{content ? ` pointer-events: ${open ? 'auto' : 'none'};` : ''}"
    data-open={open ? 'true' : 'false'}
    class="ui-overlay {content ? 'tip-content' : 'pointer-events-none'} fixed z-50 w-max text-meta font-normal normal-case tracking-normal transition-opacity duration-100 {content ? 'tip-rich' : 'max-w-64 whitespace-pre-line rounded-sm px-2 py-1'} {open ? 'opacity-100' : 'opacity-0'}"
  >{#if content}{@render content()}{:else}{text}{/if}</span>
</span>

<style>
  /* Rich content brings its own padding; it may be as wide as a phone allows. */
  .tip-rich { max-width: min(352px, calc(100vw - 16px)); padding: 0; border-radius: var(--radius-md); box-shadow: 0 0 0 1px color-mix(in srgb, var(--color-text-primary) 10%, transparent); }
  /* Laid-out content takes the pointer (set inline) and the tab key only while it is open. */
  .tip-content[data-open='false'] { visibility: hidden; transition: opacity 100ms, visibility 0s linear 100ms; }
</style>
