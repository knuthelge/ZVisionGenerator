<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import { ActionMenu, type ActionMenuEntry } from '$lib/components/molecules';
  import type { TriggerMenu } from './triggerMenu';

  interface Props {
    menu: TriggerMenu;
    /** A word was chosen that the text does not hold yet. */
    onpick: (word: string) => void;
  }

  let { menu, onpick }: Props = $props();

  let buttonEl = $state<HTMLButtonElement | null>(null);
  let open = $state(false);

  // One heading per LoRA, one item per word; a word the text holds has a check, and choosing it does nothing.
  const items = $derived<ActionMenuEntry[]>(
    menu.groups.flatMap((group) => [
      { kind: 'heading', label: group.title } as const,
      ...group.words.map((entry): ActionMenuEntry => ({ kind: 'item', id: `trigger-${group.name}-${entry.word}`, label: entry.word, checked: entry.inPrompt, onselect: () => { if (!entry.inPrompt) onpick(entry.word); } })),
    ]),
  );
</script>

<!-- A tag icon, and the count of words the prompt lacks when there are any (the one thing to act on); no word, no chevron. -->
<button
  bind:this={buttonEl}
  type="button"
  id="ws-triggers"
  class="ui-btn ui-btn-sm triggers-button"
  class:ui-btn-icon={menu.missing === 0}
  aria-haspopup="menu"
  aria-expanded={open}
  aria-label={menu.missing > 0 ? `Trigger words, ${menu.missing} not in the prompt` : 'Trigger words'}
  title="Trigger words of the LoRAs in use"
  onclick={() => { open = !open; }}
>
  <Icon name="tag" size={14} />{#if menu.missing > 0}<span class="triggers-count" data-testid="triggers-count">{menu.missing}</span>{/if}
</button>
<ActionMenu {open} anchor={buttonEl} {items} label="Triggers" align="end" onclose={() => { open = false; }} />

<style>
  .triggers-button { gap: 4px; color: var(--color-primary-main); }
  /* The count sits inline and centred on the icon, as the Reset count in the Settings footer does. */
  .triggers-count { display: block; line-height: 1; font-family: var(--font-mono); font-size: var(--text-meta); font-weight: 600; font-variant-numeric: tabular-nums; text-box: trim-both cap alphabetic; }
</style>
