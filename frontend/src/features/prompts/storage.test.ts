import { beforeEach, describe, expect, it } from 'vitest';
import { clearUnsaved, forgetRecent, loadUnsaved, recentFiles, rememberRecent, saveUnsaved } from './storage';

describe('prompt builder storage', () => {
  beforeEach(() => localStorage.clear());

  it('keeps unsaved documents per file', () => {
    const document = { snippets: [], sets: [] };
    saveUnsaved('/a.yaml', { revision: 'r1', document });
    saveUnsaved('/b.yaml', { revision: 'r2', document });

    expect(loadUnsaved('/a.yaml')).toEqual({ revision: 'r1', document });
    clearUnsaved('/a.yaml');
    expect(loadUnsaved('/a.yaml')).toBeNull();
    expect(loadUnsaved('/b.yaml')?.revision).toBe('r2');
  });

  it('lists recent files newest first without repeats, up to eight', () => {
    for (let index = 0; index < 10; index += 1) rememberRecent(`/${index}.yaml`);
    rememberRecent('/5.yaml');

    expect(recentFiles()).toEqual(['/5.yaml', '/9.yaml', '/8.yaml', '/7.yaml', '/6.yaml', '/4.yaml', '/3.yaml', '/2.yaml']);
    forgetRecent('/9.yaml');
    expect(recentFiles()[1]).toBe('/8.yaml');
  });

  it('survives unreadable storage', () => {
    localStorage.setItem('ziv.promptBuilder.recent', '{not json');
    expect(recentFiles()).toEqual([]);
  });
});
