import type { JobInput, JobInputRole } from '$lib/types';

/** One line of the job's inputs on the card: a role's files by name. */
export interface InputGroup {
  key: string;
  /** "Source", "Start image", "End image", "Reference" or "2 references". */
  label: string;
  names: string[];
  kind: JobInput['kind'];
}

const ROLE_LABELS: Record<Exclude<JobInputRole, 'reference'>, (kind: JobInput['kind']) => string> = {
  source: (kind) => (kind === 'video' ? 'Source clip' : 'Source'),
  start: () => 'Start image',
  end: () => 'End image',
};

/** The card's input lines: one per role, in the job's order, with the references of a job together on one line. */
export function inputGroups(inputs: readonly JobInput[]): InputGroup[] {
  const groups: InputGroup[] = [];
  for (const input of inputs) {
    const existing = input.role === 'reference' ? groups.find((group) => group.key === 'reference') : undefined;
    if (existing) {
      existing.names.push(input.name);
      existing.label = `${existing.names.length} references`;
      continue;
    }
    const label = input.role === 'reference' ? 'Reference' : ROLE_LABELS[input.role](input.kind);
    groups.push({ key: input.role, label, names: [input.name], kind: input.kind });
  }
  return groups;
}

/** A queue row's one-line summary of a job's inputs: the first file, and how many more there are. */
export interface InputSummary {
  name: string;
  more: number;
  /** Every input as "Role: name", for the tooltip. */
  title: string;
}

export function inputSummary(inputs: readonly JobInput[]): InputSummary | null {
  if (!inputs.length) return null;
  const [first] = inputs;
  const title = inputGroups(inputs).map((group) => `${group.label}: ${group.names.join(', ')}`).join('\n');
  return { name: first.name, more: inputs.length - 1, title };
}
