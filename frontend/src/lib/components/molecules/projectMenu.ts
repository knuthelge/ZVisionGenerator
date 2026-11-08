import type { Project } from '$lib/types';
import type { ActionMenuEntry } from './ActionMenu.svelte';

export interface ProjectMenuOptions {
  /** The first row's label: `All files` in the picker, `No project` as a Move target. */
  noneLabel: string;
  /** A row was chosen: a project's name, or null for the none row. */
  onchoose: (name: string | null) => void;
  /** `New project…` was chosen. */
  onnew: () => void;
  /** Disable the checked row (the file's current location in a Move list); it stays reachable by keys. */
  disableCurrent?: boolean;
  /** A heading above the rows, such as `Move to`. */
  heading?: string;
}

/**
 * Build the project list's menu rows: an optional heading, the none row, the projects as given, then `New project…`.
 *
 * `current` is a project's name, null for the none row, or undefined when nothing is checked (a mixed selection).
 */
export function projectMenuEntries(projects: readonly Project[], current: string | null | undefined, options: ProjectMenuOptions): ActionMenuEntry[] {
  const { noneLabel, onchoose, onnew, disableCurrent = false, heading } = options;
  const entries: ActionMenuEntry[] = [];
  if (heading) entries.push({ kind: 'heading', label: heading });
  entries.push({ kind: 'item', id: 'project-none', label: noneLabel, checked: current === null, disabled: disableCurrent && current === null, onselect: () => onchoose(null) });
  entries.push({ kind: 'separator' });
  for (const project of projects) {
    const checked = current === project.name;
    entries.push({ kind: 'item', id: `project:${project.name}`, label: project.name, checked, disabled: disableCurrent && checked, onselect: () => onchoose(project.name) });
  }
  if (projects.length > 0) entries.push({ kind: 'separator' });
  entries.push({ kind: 'item', id: 'project-new', label: 'New project…', icon: 'plus', onselect: onnew });
  return entries;
}
