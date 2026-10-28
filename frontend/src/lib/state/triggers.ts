/** Trigger text matching: whole word, case-insensitive, and never twice. */

/** The characters that continue a word; a trigger only matches where none sits beside it. */
const WORD = '[\\p{L}\\p{N}_]';

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** Collapse a trigger's inner whitespace, as the server stores it. */
export function cleanTrigger(trigger: string): string {
  return trigger.trim().replace(/\s+/g, ' ');
}

function matcher(trigger: string, flags: string): RegExp {
  const words = cleanTrigger(trigger).split(' ').map(escapeRegExp).join('\\s+');
  return new RegExp(`(?<!${WORD})${words}(?!${WORD})`, flags);
}

/** Whether *text* holds *trigger* as a whole word or phrase, in any case. */
export function containsTrigger(text: string, trigger: string): boolean {
  const cleaned = cleanTrigger(trigger);
  return cleaned !== '' && matcher(cleaned, 'iu').test(text);
}

/** Put *trigger* at the start of *text*, followed by ", "; an empty text gets the trigger and the separator alone. */
export function addTrigger(text: string, trigger: string): string {
  const cleaned = cleanTrigger(trigger);
  return `${cleaned}, ${text.replace(/^\s+/, '')}`;
}

/** The words of a comma-separated trigger, trimmed and without blanks or repeats (case-insensitive); empty for none. */
export function splitTriggers(trigger: string | null | undefined): string[] {
  return uniqueTriggers((trigger ?? '').split(','));
}

/** The triggers without repeats (case-insensitive), blanks dropped; the first spelling of each stays. */
function uniqueTriggers(triggers: readonly (string | null | undefined)[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const raw of triggers) {
    const cleaned = raw ? cleanTrigger(raw) : '';
    const key = cleaned.toLowerCase();
    if (cleaned === '' || seen.has(key)) continue;
    seen.add(key);
    result.push(cleaned);
  }
  return result;
}
