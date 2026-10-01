<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    text: string;
    /** Preferred side; the tooltip flips when that side lacks room in the viewport. */
    placement?: 'top' | 'bottom';
    /** Which edge of the trigger the tooltip lines up with; it is always kept inside the viewport. */
    align?: 'start' | 'end';
    /** Make the trigger a tab stop. Turn off when the tooltip only repeats nearby text (e.g. per-row names). */
    focusable?: boolean;
    testId?: string;
    class?: string;
    children?: Snippet;
  }

  let { text, placement = 'top', align = 'start', focusable = true, testId, class: extraClass = '', children }: Props = $props();

  const GAP_PX = 6;
  const EDGE_PX = 8;
  const id = $props.id();
  let trigger = $state<HTMLSpanElement>();
  let bubble = $state<HTMLSpanElement>();
  let open = $state(false);
  let position = $state('');

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
    place();
    open = true;
  }

  function hide(): void {
    open = false;
  }

  // Touch has no hover: a tap toggles the tooltip (on pointerdown, before the focus it causes), and the
  // pointerleave that follows a tap is ignored.
  function handlePointerEnter(event: PointerEvent): void {
    if (event.pointerType !== 'touch') show();
  }

  function handlePointerLeave(event: PointerEvent): void {
    if (event.pointerType !== 'touch') hide();
  }

  function handlePointerDown(event: PointerEvent): void {
    if (event.pointerType !== 'touch') return;
    if (open) hide();
    else show();
  }

  function handleKeydown(event: KeyboardEvent): void {
    if (event.key === 'Escape') hide();
  }

  // A fixed tooltip would drift from its trigger when any ancestor scrolls, so close it instead.
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
  onblur={hide}
  onkeydown={handleKeydown}
>
  {@render children?.()}
  <span
    bind:this={bubble}
    role="tooltip"
    {id}
    style={position}
    data-open={open ? 'true' : 'false'}
    class="surface-tooltip pointer-events-none fixed z-50 w-max max-w-64 whitespace-pre-line px-2.5 py-1.5 text-xs font-normal normal-case tracking-normal transition-opacity duration-100 {open ? 'opacity-100' : 'opacity-0'}"
  >{text}</span>
</span>
