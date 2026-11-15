import { describe, expect, it } from 'vitest';

import { inputGroups, inputSummary } from './jobInputs';

describe('inputGroups', () => {
  const reference = (name: string) => ({ role: 'reference' as const, name, kind: 'image' as const });

  it('gives each role its own line, in the job\'s order', () => {
    const start = { role: 'start' as const, name: 'a.png', kind: 'image' as const };
    const end = { role: 'end' as const, name: 'b.png', kind: 'image' as const };

    expect(inputGroups([start, end]).map(({ label, names }) => ({ label, names }))).toEqual([
      { label: 'Start image', names: ['a.png'] },
      { label: 'End image', names: ['b.png'] },
    ]);
  });

  it('names a source by what it is', () => {
    expect(inputGroups([{ role: 'source', name: 'a.png', kind: 'image' }])[0].label).toBe('Source');
    expect(inputGroups([{ role: 'source', name: 'a.mp4', kind: 'video' }])[0].label).toBe('Source clip');
  });

  it('joins the references of a job on one line and counts them', () => {
    const groups = inputGroups([reference('fox.png'), reference('owl.png')]);

    expect(groups).toHaveLength(1);
    expect(groups[0]).toMatchObject({ label: '2 references', names: ['fox.png', 'owl.png'] });
    expect(inputGroups([reference('fox.png')])[0].label).toBe('Reference');
  });

  it('is empty for a job with no inputs', () => {
    expect(inputGroups([])).toEqual([]);
  });
});

describe('inputSummary', () => {
  it('names the first file and how many more follow, with every input in the tooltip', () => {
    const summary = inputSummary([
      { role: 'start', name: 'a.png', kind: 'image' },
      { role: 'reference', name: 'b.png', kind: 'image' },
    ]);

    expect(summary).toEqual({ name: 'a.png', more: 1, title: 'Start image: a.png\nReference: b.png' });
  });

  it('is null for a job with no inputs', () => {
    expect(inputSummary([])).toBeNull();
  });
});
