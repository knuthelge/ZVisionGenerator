import { beforeEach, describe, expect, it } from 'vitest';
import type { BrowseEntry } from '$lib/types';
import {
  breadcrumbs,
  completedPath,
  completionTarget,
  formatBytes,
  jumpToLetter,
  matchCompletions,
  readRecentPaths,
  readShowHidden,
  rememberRecentPath,
  writeShowHidden,
} from './pathBrowser';

function entry(name: string, kind: BrowseEntry['kind'] = 'file'): BrowseEntry {
  return { name, path: `/x/${name}`, kind, size: null, modified: null, selectable: true };
}

describe('completionTarget', () => {
  it.each([
    ['~/models/lo', { dir: '~/models/', prefix: 'lo' }],
    ['/', { dir: '/', prefix: '' }],
    ['~', { dir: '~/', prefix: '' }],
    ['C:\\Users\\user', { dir: 'C:\\Users\\', prefix: 'user' }],
  ])('splits %s', (text, expected) => {
    expect(completionTarget(text)).toEqual(expected);
  });

  it.each(['', 'models', 'models/lo'])('ignores %s, which is not absolute', (text) => {
    expect(completionTarget(text)).toBeNull();
  });
});

describe('completions', () => {
  const entries = [entry('loras', 'directory'), entry('Local', 'directory'), entry('models', 'directory'), entry('lo.safetensors')];

  it('matches a prefix ignoring case', () => {
    expect(matchCompletions(entries, 'lo').map((item) => item.name)).toEqual(['loras', 'Local', 'lo.safetensors']);
    expect(matchCompletions(entries, '', 2)).toHaveLength(2);
  });

  it('ends a folder with the separator that was typed', () => {
    expect(completedPath('~/models/', entries[0])).toBe('~/models/loras/');
    expect(completedPath('C:\\m\\', entries[0])).toBe('C:\\m\\loras\\');
    expect(completedPath('~/models/', entries[3])).toBe('~/models/lo.safetensors');
  });
});

describe('breadcrumbs', () => {
  it('splits a POSIX path from the root', () => {
    expect(breadcrumbs('/home/user/models', '/')).toEqual([
      { label: '/', path: '/' },
      { label: 'home', path: '/home' },
      { label: 'user', path: '/home/user' },
      { label: 'models', path: '/home/user/models' },
    ]);
    expect(breadcrumbs('/', '/')).toEqual([{ label: '/', path: '/' }]);
  });

  it('splits a Windows path from the drive', () => {
    expect(breadcrumbs('C:\\Users\\user', '\\')).toEqual([
      { label: 'C:', path: 'C:\\' },
      { label: 'Users', path: 'C:\\Users' },
      { label: 'user', path: 'C:\\Users\\user' },
    ]);
  });
});

describe('helpers', () => {
  it('jumps to the next name starting with a letter, wrapping', () => {
    const entries = [entry('alpha'), entry('beta'), entry('apple')];
    expect(jumpToLetter(entries, 'a', 0)).toBe(2);
    expect(jumpToLetter(entries, 'a', 2)).toBe(0);
    expect(jumpToLetter(entries, 'z', 0)).toBe(-1);
  });

  it('formats sizes', () => {
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(144 * 1024 * 1024)).toBe('144 MB');
    expect(formatBytes(6.4 * 1024 ** 3)).toBe('6.4 GB');
  });
});

describe('remembered choices', () => {
  beforeEach(() => localStorage.clear());

  it('keeps the last five paths per purpose, newest first and without repeats', () => {
    for (const name of ['a', 'b', 'c', 'd', 'e', 'f', 'c']) rememberRecentPath('lora_file', `/l/${name}`);
    rememberRecentPath('checkpoint_file', '/m/x');

    expect(readRecentPaths('lora_file')).toEqual(['/l/c', '/l/f', '/l/e', '/l/d', '/l/b']);
    expect(readRecentPaths('checkpoint_file')).toEqual(['/m/x']);
  });

  it('survives broken storage', () => {
    localStorage.setItem('ziv-recent-paths-v1', '{not json');
    expect(readRecentPaths('lora_file')).toEqual([]);
  });

  it('remembers the hidden-files switch', () => {
    expect(readShowHidden()).toBe(false);
    writeShowHidden(true);
    expect(readShowHidden()).toBe(true);
  });
});
