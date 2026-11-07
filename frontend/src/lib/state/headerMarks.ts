/** What the Enhanced tab of a text box carries: at most one mark inside it, and at most one control after the tablist. Your own tab never carries one. */
export interface HeaderMarks {
  /** Inside the Enhanced tab while it is closed: a quiet spinner (Enhance runs), ⚠ (sent, out of date) or a dot (sent). */
  mark: 'spinner' | 'warn' | 'dot' | null;
  /** After the tablist while the Enhanced tab is open: the stop ring (Enhance runs) or ✕ Scratch (the Enhanced text is in use). */
  after: 'ring' | 'scratch' | null;
  /** The Enhanced tab's description for assistive technology (the marks themselves are decorative); empty for none. */
  description: string;
}

/** What the box's Enhanced text is doing: *open* is the Enhanced tab shown, *used* is Generate sending its text, *running* is Enhance running. */
export interface HeaderState {
  open: boolean;
  used: boolean;
  stale: boolean;
  running: boolean;
}

/** The marks for a box in *state*. */
export function headerMarks({ open, used, stale, running }: HeaderState): HeaderMarks {
  const mark = open ? null : running ? 'spinner' : used && stale ? 'warn' : used ? 'dot' : null;
  const after = !open ? null : running ? 'ring' : used ? 'scratch' : null;
  const description = used ? `sent by Generate${stale ? ', out of date' : ''}${running ? ', enhancing' : ''}` : running ? 'enhancing' : '';
  return { mark, after, description };
}
