/** Text after an insertion and where the caret goes. */
export interface Insertion {
  text: string;
  caret: number;
}

/** Which sides of an inserted snippet get a separating space when they would touch a word. */
export interface InsertSpacing {
  /** A space before the snippet when the text before it does not end in whitespace. */
  before?: boolean;
  /** A space after the snippet when the text after it starts with a letter or a digit. */
  after?: boolean;
}

/**
 * Replace `start..end` of *text* with *snippet* and put the caret after it (and after the space that follows it, if one is added).
 * With no *spacing* the snippet goes in as it is.
 */
export function insertAt(text: string, start: number, end: number, snippet: string, spacing: InsertSpacing = {}): Insertion {
  const before = text.slice(0, start);
  const after = text.slice(end);
  const lead = spacing.before && before !== '' && !/\s$/.test(before) ? ' ' : '';
  const trail = spacing.after && /^[\p{L}\p{N}]/u.test(after) ? ' ' : '';
  const inserted = lead + snippet + trail;
  return { text: before + inserted + after, caret: start + inserted.length };
}

const WORDS_AROUND = 5;

// A sentence ends at one of these with an optional closing quote or bracket and white space, or at a line break.
const SENTENCE_END = /[.!?…。！？]["”’)\]]?\s+|\n+/g;

/** Where the sentence holding *caret* starts: after the white space that follows the sentence before it, or the first word of the text. */
export function sentenceStart(text: string, caret: number): number {
  let start = 0;
  for (const found of text.matchAll(SENTENCE_END)) {
    const end = found.index + found[0].length;
    if (end > caret) break;
    start = end;
  }
  while (start < text.length && /\s/.test(text[start])) start++;
  return start;
}

/** The few words each side of *caret*, with `|` at the caret: `…and said no | to him all…`. Empty for a text without words. */
export function wordsAround(text: string, caret: number): string {
  const before = text.slice(0, caret).split(/\s+/).filter(Boolean);
  const after = text.slice(caret).split(/\s+/).filter(Boolean);
  if (before.length === 0 && after.length === 0) return '';
  const words = (list: string[], mark: string) => (list.length > WORDS_AROUND ? mark : '');
  return `${words(before, '…')}${before.slice(-WORDS_AROUND).join(' ')} | ${after.slice(0, WORDS_AROUND).join(' ')}${words(after, '…')}`.trim();
}
