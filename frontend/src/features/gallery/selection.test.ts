import { describe, expect, it } from 'vitest';

import { modeLine, rangeIds, selectionProject, withAdded, withToggled } from './selection';

describe('rangeIds', () => {
  const tiles = ['t0', 't1', 't2', 't3', 't4', 't5'];

  it('runs from the anchor to the target, inclusive', () => {
    expect(rangeIds(tiles, 't1', 't4')).toEqual(['t1', 't2', 't3', 't4']);
  });

  it('runs in grid order when the target is before the anchor', () => {
    expect(rangeIds(tiles, 't4', 't0')).toEqual(['t0', 't1', 't2', 't3', 't4']);
  });

  it('is the target alone with no anchor, or an anchor that is not loaded', () => {
    expect(rangeIds(tiles, null, 't3')).toEqual(['t3']);
    expect(rangeIds(tiles, 'gone', 't3')).toEqual(['t3']);
  });

  it('is the target alone when the target is the anchor, and when it is not loaded', () => {
    expect(rangeIds(tiles, 't2', 't2')).toEqual(['t2']);
    expect(rangeIds(tiles, 't2', 'far')).toEqual(['far']);
  });
});

describe('selection sets', () => {
  it('toggles one ID in a copy', () => {
    const before = new Set(['a']);
    expect([...withToggled(before, 'b')]).toEqual(['a', 'b']);
    expect([...withToggled(before, 'a')]).toEqual([]);
    expect([...before]).toEqual(['a']);
  });

  it('adds IDs and never removes one', () => {
    const before = new Set(['a', 'b']);
    expect([...withAdded(before, ['b', 'c'])]).toEqual(['a', 'b', 'c']);
    expect([...before]).toEqual(['a', 'b']);
  });
});

describe('selectionProject', () => {
  const loaded = [
    { id: 'a', project: 'A' },
    { id: 'b', project: 'A' },
    { id: 'c', project: null },
  ];

  it('is the shared project of loaded files, null for no project', () => {
    expect(selectionProject(new Set(['a', 'b']), loaded, null, 'all')).toBe('A');
    expect(selectionProject(new Set(['c']), loaded, null, 'all')).toBeNull();
  });

  it('is undefined for a mix and for an empty selection', () => {
    expect(selectionProject(new Set(['a', 'c']), loaded, null, 'all')).toBeUndefined();
    expect(selectionProject(new Set(), loaded, null, 'all')).toBeUndefined();
  });

  it('places an unloaded file in the picked project', () => {
    expect(selectionProject(new Set(['a', 'unloaded']), loaded, 'A', 'all')).toBe('A');
    expect(selectionProject(new Set(['c', 'unloaded']), loaded, 'A', 'all')).toBeUndefined();
  });

  it('places an unloaded file in no project with the Not in a project filter', () => {
    expect(selectionProject(new Set(['c', 'unloaded']), loaded, null, 'no_project')).toBeNull();
  });

  it('cannot place an unloaded file with All files and another filter', () => {
    expect(selectionProject(new Set(['a', 'unloaded']), loaded, null, 'image')).toBeUndefined();
    expect(selectionProject(new Set(['unloaded']), loaded, null, 'all')).toBeUndefined();
  });
});

describe('modeLine', () => {
  const base = { project: null, starred: false, filter: 'all', coarse: false } as const;

  it('names the project when one is picked, and only then', () => {
    expect(modeLine({ ...base, project: 'Night Bus' })).toContain('Night Bus');
    expect(modeLine(base)).not.toContain('Night Bus');
  });
});
