/**
 * The runs that are writing text Generate would send: Enhance, the Script and Lyrics helpers and Fill in. While one runs, Generate waits for it
 * and says so, so a take is never made from the version before the one being written. Held in memory only.
 */
let runs = $state<{ id: number; label: string }[]>([]);
let nextId = 0;

export const textWriting = {
  /** Why Generate waits ("Enhancing… Generate when it's done."), or null when nothing is being written. */
  get reason(): string | null {
    const run = runs[0];
    return run === undefined ? null : `${run.label}… Generate when it's done.`;
  },
  /** Record a run named *label* ("Enhancing", "Writing"); the returned function ends it and may be called any number of times. */
  begin(label: string): () => void {
    const id = nextId++;
    runs = [...runs, { id, label }];
    return () => {
      runs = runs.filter((run) => run.id !== id);
    };
  },
};
