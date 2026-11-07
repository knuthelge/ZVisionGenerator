import type { AudioSettingsVocabulary } from '$lib/types';
import { insertAt, sentenceStart, type InsertSpacing, type Insertion } from './caretInsert';
import { BRACKETED, deliveryLabel, markName } from './speechDelivery';

/** Something the Insert menu puts into the text at the caret, as the text reads: a pause is `…`, a laugh `haha`. */
export interface TextSnippet {
  id: string;
  label: string;
  text: string;
  /** Muted words after the label: what is inserted. */
  hint: string;
  spacing: InsertSpacing;
}

/** The snippets of the Insert menu, in menu order. */
export const TEXT_SNIPPETS: readonly TextSnippet[] = [
  { id: 'pause', label: 'Pause', text: '…', hint: '…', spacing: { after: true } },
  { id: 'laugh', label: 'Laugh', text: 'haha', hint: 'haha', spacing: { before: true, after: true } },
];

/** A mark the Delivery group offers: the id that goes between the brackets, its menu label and (for `as set`) what it does. */
export interface MarkChoice {
  id: string;
  label: string;
  hint?: string;
}

/** A heading of the Delivery group and the marks under it. */
export interface MarkGroup {
  title: string;
  marks: MarkChoice[];
}

/**
 * The marks the vocabulary lists as the Insert menu groups them: the emotions by their group, then the manners, then `as set`.
 * A mark whose emotion has no group (none today) is left out.
 */
export function markGroups(vocabulary: AudioSettingsVocabulary): MarkGroup[] {
  const groups = new Map<string, MarkChoice[]>();
  const manners: MarkChoice[] = [];
  const reset: MarkChoice[] = [];
  const emotionGroup = new Map(vocabulary.emotions.map((emotion) => [emotion.id, emotion.group]));
  for (const mark of vocabulary.marks) {
    if (mark.field === 'emotion') {
      const group = emotionGroup.get(mark.id);
      if (group) groups.set(group, [...(groups.get(group) ?? []), { id: mark.id, label: deliveryLabel(vocabulary, 'emotion', mark.id) }]);
    } else if (mark.field === 'manner') {
      manners.push({ id: mark.id, label: deliveryLabel(vocabulary, 'manner', mark.id) });
    } else if (mark.field === null) {
      reset.push({ id: mark.id, label: 'As set', hint: 'your settings' });
    }
  }
  return [
    ...Array.from(groups, ([title, marks]) => ({ title, marks })),
    ...(manners.length > 0 ? [{ title: 'Manner', marks: manners }] : []),
    ...(reset.length > 0 ? [{ title: 'Reset', marks: reset }] : []),
  ];
}

/** Insert *snippet* in place of `start..end` of *text*. */
export function insertSnippet(text: string, start: number, end: number, snippet: TextSnippet): Insertion {
  return insertAt(text, start, end, snippet.text, snippet.spacing);
}

/** A bracketed run at the start of a text, with the spaces after it: the shared mark pattern, anchored. */
const LEADING_BRACKETED = new RegExp(`^${BRACKETED.source}[ \\t]*`);

/** The end of the marks that open the sentence at *start*, with the spaces after them; *start* itself when none opens it. */
function leadingMarksEnd(text: string, start: number, marks: ReadonlySet<string>): number {
  let end = start;
  for (;;) {
    const found = LEADING_BRACKETED.exec(text.slice(end));
    if (found === null || !marks.has(markName(found[1]))) return end;
    end += found[0].length;
  }
}

/**
 * Set the Delivery from the start of the sentence holding *caret* to the mark *id*: `[sad] ` goes in front of it, in place of any marks
 * that already open it (the last of several would win anyway). The caret keeps its place in the text.
 */
export function setDeliveryMark(text: string, caret: number, id: string, marks: readonly string[]): Insertion {
  const start = sentenceStart(text, caret);
  const end = leadingMarksEnd(text, start, new Set(marks));
  const mark = `[${id}]${/^\s/.test(text.slice(end)) ? '' : ' '}`;
  const next = text.slice(0, start) + mark + text.slice(end);
  const moved = caret >= end ? caret + mark.length - (end - start) : caret > start ? start + mark.length : caret;
  return { text: next, caret: moved };
}
