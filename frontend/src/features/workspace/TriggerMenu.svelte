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

<button
  bind:this={buttonEl}
  type="button"
  id="ws-triggers"
  class="ui-btn ui-btn-sm triggers-button"
  aria-haspopup="menu"
  aria-expanded={open}
  title="Trigger words of the LoRAs in use"
  onclick={() => { open = !open; }}
>
  Triggers{#if menu.missing > 0}<span class="triggers-count" data-testid="triggers-count">{menu.missing}</span>{/if}<Icon name="chevdown" size={12} />
</button>
<ActionMenu {open} anchor={buttonEl} {items} label="Triggers" align="end" onclose={() => { open = false; }} />

<style>
  .triggers-button { gap: 6px; color: var(--color-primary-main); }
  .triggers-count { font-family: var(--font-mono); font-size: var(--text-meta); font-weight: 600; }
</style>
