<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import type { ViewerAct } from './viewerActions';

  interface Props {
    act: ViewerAct;
    /** `bar`: a toolbar button, with a mouse. `cell`: a dock cell, an icon over its label, for a finger. */
    variant: 'bar' | 'cell';
    /** Whether the popup this action opens is open; left out for an action that opens none. */
    expanded?: boolean;
    /** The click, whose pointer type says whether a mouse, a finger or a key made it, and whose target is the popup's anchor. */
    onact: (act: ViewerAct, event: MouseEvent & { currentTarget: HTMLElement }) => void;
  }

  let { act, variant, expanded, onact }: Props = $props();

  const item = $derived(act.item);
  const cell = $derived(variant === 'cell');
  // Off for a reason a tap or a key explains: the button stays focusable. Only Move and Delete are disabled outright, while a delete runs.
  const greyed = $derived(item.disabled === true && act.disabled !== true);
  // The label without the end that gives way first; a cell never shows it, and its chevron hangs from this label.
  const label = $derived(act.tail ? item.label.slice(0, item.label.length - act.tail.length) : item.label);
</script>

{#snippet face()}
  {#if cell}
    <Icon name={item.icon ?? 'more'} size={20} solid={act.solid} />
    <span class="ui-tool-label">{label}{#if act.menu}<Icon name="chevdown" size={10} class="chev" />{/if}</span>
    {#if item.warn}<em class="ui-choice-warn">· {item.warn}</em>{/if}
  {:else}
    <Icon name={item.icon ?? 'more'} size={14} solid={act.solid} />
    <span class="vb-label">{label}{#if act.tail}<span class="vb-tail">{act.tail}</span>{/if}{#if item.warn}{' '}<em class="ui-choice-warn">· {item.warn}</em>{/if}</span>
    {#if act.menu}<Icon name="chevdown" size={12} class="vb-chev" />{/if}
  {/if}
{/snippet}

{#if item.href}
  <a
    class={cell ? 'ui-tool' : 'viewer-btn ui-btn'}
    data-action={act.id}
    data-group={act.group}
    href={item.href}
    download={item.download}
    title={act.title}
  >{@render face()}</a>
{:else}
  <button
    type="button"
    class={cell ? 'ui-tool' : 'viewer-btn ui-btn'}
    class:ui-tool-primary={cell && item.primary}
    class:ui-tool-danger={cell && item.danger}
    class:ui-btn-primary={!cell && item.primary}
    class:ui-btn-danger={!cell && item.danger}
    class:ui-btn-reason={!cell}
    data-action={act.id}
    data-group={act.group}
    data-menu={act.menu}
    aria-haspopup={expanded === undefined ? undefined : 'menu'}
    aria-expanded={expanded}
    aria-disabled={greyed ? 'true' : undefined}
    disabled={act.disabled}
    title={act.title}
    onclick={(event) => onact(act, event)}
  >{@render face()}</button>
{/if}
