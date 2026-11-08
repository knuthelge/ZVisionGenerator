import { describe, expect, it, vi } from 'vitest';

import type { Project } from '$lib/types';
import type { ActionMenuItem } from './ActionMenu.svelte';
import { projectMenuEntries } from './projectMenu';

const PROJECTS: Project[] = [
  { name: 'Night Bus', path: '/out/Night Bus' },
  { name: 'Zoo', path: '/out/Zoo' },
];

function build(current: string | null | undefined, extra: Partial<Parameters<typeof projectMenuEntries>[2]> = {}) {
  const onchoose = vi.fn();
  const onnew = vi.fn();
  const entries = projectMenuEntries(PROJECTS, current, { noneLabel: 'All files', onchoose, onnew, ...extra });
  return { entries, onchoose, onnew };
}

const items = (entries: ReturnType<typeof build>['entries']) => entries.filter((entry): entry is ActionMenuItem => entry.kind === 'item');

describe('projectMenuEntries', () => {
  it('lists the none row, the projects in the order given, then New project', () => {
    const { entries } = build('Zoo');

    expect(entries.map((entry) => entry.kind)).toEqual(['item', 'separator', 'item', 'item', 'separator', 'item']);
    expect(items(entries).map((item) => item.label)).toEqual(['All files', 'Night Bus', 'Zoo', 'New project…']);
    expect(items(entries).at(-1)?.icon).toBe('plus');
  });

  it('checks the picked project, or the none row when current is null, and nothing when undefined', () => {
    const checks = (current: string | null | undefined) => items(build(current).entries).map((item) => item.checked);

    expect(checks('Zoo')).toEqual([false, false, true, undefined]);
    expect(checks(null)).toEqual([true, false, false, undefined]);
    expect(checks(undefined)).toEqual([false, false, false, undefined]);
  });

  it('uses the label it is given for the none row', () => {
    expect(items(build(null, { noneLabel: 'No project' }).entries)[0].label).toBe('No project');
  });

  it('adds a heading first when given one', () => {
    const { entries } = build(null, { heading: 'Move to' });

    expect(entries[0]).toEqual({ kind: 'heading', label: 'Move to' });
  });

  it('disables only the checked row, and only when asked', () => {
    expect(items(build('Zoo', { disableCurrent: true }).entries).map((item) => Boolean(item.disabled))).toEqual([false, false, true, false]);
    expect(items(build(null, { disableCurrent: true }).entries).map((item) => Boolean(item.disabled))).toEqual([true, false, false, false]);
    expect(items(build(undefined, { disableCurrent: true }).entries).every((item) => !item.disabled)).toBe(true);
    expect(items(build('Zoo').entries).every((item) => !item.disabled)).toBe(true);
  });

  it('reports a chosen project by name, the none row as null, and New project through onnew', () => {
    const { entries, onchoose, onnew } = build(null);
    const [none, night, , add] = items(entries);

    night.onselect?.();
    none.onselect?.();
    add.onselect?.();

    expect(onchoose.mock.calls).toEqual([['Night Bus'], [null]]);
    expect(onnew).toHaveBeenCalledOnce();
  });

  it('shows just the none row and New project with no projects', () => {
    const entries = projectMenuEntries([], null, { noneLabel: 'All files', onchoose: vi.fn(), onnew: vi.fn() });

    expect(entries.map((entry) => entry.kind)).toEqual(['item', 'separator', 'item']);
  });
});
