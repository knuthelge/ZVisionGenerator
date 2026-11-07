<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import { ActionMenu, type ActionMenuEntry, type ActionMenuItem } from '$lib/components/molecules';
  import { wordsAround } from '$lib/state/caretInsert';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { TEXT_SNIPPETS, insertSnippet, markGroups, setDeliveryMark, type TextSnippet } from '$lib/state/speechMarks';
  import { editTextarea } from '$lib/textareaEdit';
  import type { AudioSettingsVocabulary } from '$lib/types';

  interface Props {
    /** The Speech text, which the menu edits at its caret; the menu is off until it is there. */
    textarea: HTMLTextAreaElement | null;
    vocabulary?: AudioSettingsVocabulary;
    /** The voice reads the delivery: a voice that follows a recording is read as recorded, so the delivery marks are not offered. */
    delivery: boolean;
  }

  let { textarea, vocabulary, delivery }: Props = $props();

  let buttonEl = $state<HTMLButtonElement | null>(null);
  let open = $state(false);
  // Where in the text the menu was opened, kept because the text is not focused while the menu is.
  let range = $state({ start: 0, end: 0 });
  let quote = $state('');
  // The menu was opened from the text with Alt+Enter, so closing it returns to the text and not to the button.
  let fromText = false;

  const markIds = $derived(vocabulary?.marks.map((mark) => mark.id) ?? []);

  function openAtCaret(): void {
    if (!textarea) return;
    range = { start: textarea.selectionStart, end: textarea.selectionEnd };
    quote = wordsAround(textarea.value, range.start);
    open = true;
  }

  function close(): void {
    open = false;
    // Escape and a click outside also end here; only a menu opened from the text gives the text its focus back.
    if (fromText && textarea && !layout.coarse) queueMicrotask(() => textarea?.focus({ preventScroll: true }));
    fromText = false;
  }

  function insert(compute: (text: string) => { text: string; caret: number }): void {
    if (!textarea) return;
    const next = compute(textarea.value);
    // A finger would raise the keyboard over the result.
    editTextarea(textarea, next.text, next.caret, { focus: !layout.coarse });
  }

  // Alt+Enter in the text opens the menu at the caret.
  $effect(() => {
    const node = textarea;
    if (!node) return;
    function onKeydown(event: KeyboardEvent): void {
      if (event.key !== 'Enter' || !event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      event.preventDefault();
      fromText = true;
      openAtCaret();
    }
    node.addEventListener('keydown', onKeydown);
    return () => node.removeEventListener('keydown', onKeydown);
  });

  const snippetItem = (snippet: TextSnippet): ActionMenuItem => ({
    kind: 'item',
    id: `insert-${snippet.id}`,
    label: snippet.label,
    hint: snippet.hint,
    onselect: () => insert((text) => insertSnippet(text, range.start, range.end, snippet)),
  });

  const markItems = $derived<(ActionMenuItem | { kind: 'heading'; label: string })[]>(
    vocabulary
      ? markGroups(vocabulary).flatMap((group) => [
          { kind: 'heading', label: group.title } as const,
          ...group.marks.map((mark): ActionMenuItem => ({
            kind: 'item',
            id: `insert-mark-${mark.id.replace(/\s+/g, '-')}`,
            label: mark.label,
            hint: mark.hint,
            onselect: () => insert((text) => setDeliveryMark(text, range.start, mark.id, markIds)),
          })),
        ])
      : [],
  );

  const items = $derived<ActionMenuEntry[]>([
    // On a phone the sheet covers the text, so it says where the insertion goes.
    ...(layout.compact && quote ? [{ kind: 'note', text: `Inserts at the caret: ${quote}` } as const] : []),
    ...TEXT_SNIPPETS.map(snippetItem),
    ...(delivery && markItems.length > 0 ? [{ kind: 'separator' } as const, { kind: 'group', id: 'insert-delivery', label: 'Delivery from here', items: markItems } as const] : []),
  ]);
</script>

<button
  bind:this={buttonEl}
  type="button"
  id="ws-insert"
  class="ui-btn ui-btn-sm"
  aria-haspopup="menu"
  aria-expanded={open}
  disabled={textarea === null}
  title="Insert a pause, a laugh or a delivery mark at the caret (Alt+Enter)"
  onclick={() => { if (open) { close(); return; } fromText = false; openAtCaret(); }}
><Icon name="plus" size={13} />Insert<Icon name="chevdown" size={12} /></button>
<ActionMenu {open} anchor={buttonEl} {items} label="Insert" align="start" onclose={close} />
