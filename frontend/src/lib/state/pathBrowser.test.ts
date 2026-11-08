import { beforeEach, describe, expect, it } from 'vitest';
import type { BrowseEntry } from '$lib/types';
import {
  breadcrumbs,
  completedPath,
  completionTarget,
  DEFAULT_BROWSE_VIEW,
  formatBytes,
  formatModified,
  formatModifiedShort,
  formatModifiedFull,
  gridMove,
  itemCountLabel,
  jumpToLetter,
  matchCompletions,
  middleTruncate,
  orderAfterClick,
  pickerStart,
  previewFacts,
  readBrowseView,
  readRecentPaths,
  readShowHidden,
  rememberRecentPath,
  SORT_CHOICES,
  sortChoiceLabel,
  sortEntries,
  writeBrowseView,
  writeShowHidden,
  type BrowseOrder,
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

function file(name: string, size: number | null, modified: number | null, kind: BrowseEntry['kind'] = 'file'): BrowseEntry {
  return { name, path: `/x/${name}`, kind, size, modified, selectable: true };
}

describe('sortEntries', () => {
  const entries = [
    file('b.png', 300, 20),
    file('zeta', null, 5, 'directory'),
    file('a.png', 100, 30),
    file('alpha', null, 50, 'directory'),
    file('c.png', 200, 10),
  ];
  const names = (order: BrowseOrder) => sortEntries(entries, order).map((item) => item.name);

  it.each<[string, BrowseOrder, string[]]>([
    ['Name A-Z', { sort: 'name', dir: 'asc' }, ['alpha', 'zeta', 'a.png', 'b.png', 'c.png']],
    ['Name Z-A', { sort: 'name', dir: 'desc' }, ['zeta', 'alpha', 'c.png', 'b.png', 'a.png']],
    ['Newest first', { sort: 'modified', dir: 'desc' }, ['alpha', 'zeta', 'a.png', 'b.png', 'c.png']],
    ['Oldest first', { sort: 'modified', dir: 'asc' }, ['zeta', 'alpha', 'c.png', 'b.png', 'a.png']],
    ['Largest first', { sort: 'size', dir: 'desc' }, ['alpha', 'zeta', 'b.png', 'c.png', 'a.png']],
    ['Smallest first', { sort: 'size', dir: 'asc' }, ['alpha', 'zeta', 'a.png', 'c.png', 'b.png']],
  ])('puts folders on top for %s', (_label, order, expected) => {
    expect(names(order)).toEqual(expected);
  });

  it('orders folders by name, in either direction, when sorting by size', () => {
    const folders = [file('b', null, 1, 'directory'), file('a', null, 2, 'directory')];

    expect(sortEntries(folders, { sort: 'size', dir: 'desc' }).map((item) => item.name)).toEqual(['a', 'b']);
    expect(sortEntries(folders, { sort: 'size', dir: 'asc' }).map((item) => item.name)).toEqual(['a', 'b']);
  });

  it('compares names naturally, ignoring case', () => {
    const takes = ['take-10.wav', 'Take-2.wav', 'take-1.wav'].map((name) => file(name, 1, 1));

    expect(sortEntries(takes, { sort: 'name', dir: 'asc' }).map((item) => item.name)).toEqual(['take-1.wav', 'Take-2.wav', 'take-10.wav']);
  });

  it('breaks ties by natural name and treats missing sizes and dates as zero', () => {
    const tied = [file('take-10', 5, 5), file('take-2', 5, 5), file('none', null, null)];

    expect(sortEntries(tied, { sort: 'modified', dir: 'desc' }).map((item) => item.name)).toEqual(['take-2', 'take-10', 'none']);
    expect(sortEntries(tied, { sort: 'size', dir: 'asc' }).map((item) => item.name)).toEqual(['none', 'take-2', 'take-10']);
  });

  it('leaves the input list untouched', () => {
    const before = entries.map((item) => item.name);

    sortEntries(entries, { sort: 'name', dir: 'desc' });

    expect(entries.map((item) => item.name)).toEqual(before);
  });
});

describe('orderAfterClick', () => {
  it('reverses the active column', () => {
    expect(orderAfterClick({ sort: 'modified', dir: 'desc' }, 'modified')).toEqual({ sort: 'modified', dir: 'asc' });
    expect(orderAfterClick({ sort: 'name', dir: 'asc' }, 'name')).toEqual({ sort: 'name', dir: 'desc' });
  });

  it('starts another column at names A-Z, largest first or newest first', () => {
    const current: BrowseOrder = { sort: 'modified', dir: 'asc' };

    expect(orderAfterClick(current, 'name')).toEqual({ sort: 'name', dir: 'asc' });
    expect(orderAfterClick(current, 'size')).toEqual({ sort: 'size', dir: 'desc' });
    expect(orderAfterClick({ sort: 'name', dir: 'desc' }, 'modified')).toEqual({ sort: 'modified', dir: 'desc' });
  });
});

describe('formatModifiedShort', () => {
  const now = new Date(2026, 10, 6, 10, 0, 0);
  const at = (...parts: [number, number, number, number, number]) => new Date(...parts).getTime() / 1000;

  it('writes this year\'s older dates as day and month, and keeps Today, Yesterday and other years', () => {
    expect(formatModifiedShort(at(2026, 10, 6, 9, 12), now)).toBe(formatModified(at(2026, 10, 6, 9, 12), now));
    expect(formatModifiedShort(at(2026, 10, 5, 21, 40), now)).toBe(formatModified(at(2026, 10, 5, 21, 40), now));
    expect(formatModifiedShort(at(2026, 10, 3, 14, 5), now)).toBe('3 Nov 14:05');
    expect(formatModifiedShort(at(2025, 0, 2, 8, 0), now)).toBe('2025-01-02 08:00');
  });
});

describe('formatModified', () => {
  const now = new Date(2026, 10, 6, 10, 0, 0);
  const at = (...parts: [number, number, number, number, number]) => new Date(...parts).getTime() / 1000;

  it('names today and yesterday, with the time', () => {
    expect(formatModified(at(2026, 10, 6, 9, 12), now)).toBe('Today 09:12');
    expect(formatModified(at(2026, 10, 5, 21, 40), now)).toBe('Yesterday 21:40');
  });

  it('gives the date and time for anything older', () => {
    expect(formatModified(at(2026, 10, 3, 14, 5), now)).toBe('2026-11-03 14:05');
    expect(formatModified(at(2025, 11, 31, 23, 59), now)).toBe('2025-12-31 23:59');
  });

  it('finds yesterday across a month boundary', () => {
    const first = new Date(2026, 11, 1, 8, 0, 0);

    expect(formatModified(at(2026, 10, 30, 18, 30), first)).toBe('Yesterday 18:30');
  });

  it('shows a time later today as today', () => {
    expect(formatModified(at(2026, 10, 6, 23, 59), now)).toBe('Today 23:59');
  });
});

describe('formatModifiedFull', () => {
  it('adds the seconds', () => {
    expect(formatModifiedFull(new Date(2026, 10, 3, 14, 5, 9).getTime() / 1000)).toBe('2026-11-03 14:05:09');
  });
});

describe('middleTruncate', () => {
  it('keeps a short name whole', () => {
    expect(middleTruncate('fox.png', 20)).toBe('fox.png');
    expect(middleTruncate('exactly-ten', 11)).toBe('exactly-ten');
  });

  it('cuts the middle and keeps the start and the end', () => {
    const cut = middleTruncate('fox-portrait_2026-11-05_21-38-51.png', 26);

    expect(cut).toHaveLength(26);
    expect(cut.startsWith('fox-portrait_')).toBe(true);
    expect(cut.endsWith('38-51.png')).toBe(true);
    expect(cut).toContain('…');
  });
});

describe('itemCountLabel', () => {
  it.each([
    [10, 'image', '10 images'],
    [1, 'image', '1 image'],
    [0, 'audio', '0 audio files'],
    [1, 'audio', '1 audio file'],
    [2, 'video', '2 videos'],
    [3, 'folder', '3 folders'],
    [1, 'folder', '1 folder'],
    [4, null, '4 files'],
    [1, null, '1 file'],
  ] as const)('says %i of %s as "%s"', (count, noun, expected) => {
    expect(itemCountLabel(count, noun)).toBe(expected);
  });
});

describe('gridMove', () => {
  // Seven tiles in three columns:  0 1 2 / 3 4 5 / 6
  it.each([
    [4, 'ArrowLeft', 3],
    [0, 'ArrowLeft', 0],
    [4, 'ArrowRight', 5],
    [6, 'ArrowRight', 6],
    [4, 'ArrowUp', 1],
    [1, 'ArrowUp', 1],
    [1, 'ArrowDown', 4],
    [3, 'ArrowDown', 6],
    [5, 'ArrowDown', 6],
    [6, 'ArrowDown', 6],
    [2, 'Enter', 2],
  ])('moves %i with %s to %i', (index, key, expected) => {
    expect(gridMove(index, key, 3, 7)).toBe(expected);
  });

  it('stays put with no tiles', () => {
    expect(gridMove(-1, 'ArrowDown', 3, 0)).toBe(-1);
  });
});

describe('browse view', () => {
  beforeEach(() => localStorage.clear());

  it('starts at newest first, list and preview on', () => {
    expect(readBrowseView()).toEqual({ sort: 'modified', dir: 'desc', view: 'list', preview: true });
    expect(readBrowseView()).toEqual(DEFAULT_BROWSE_VIEW);
  });

  it('round-trips through the one key', () => {
    writeBrowseView({ sort: 'size', dir: 'asc', view: 'grid', preview: false });

    expect(readBrowseView()).toEqual({ sort: 'size', dir: 'asc', view: 'grid', preview: false });
    expect(JSON.parse(localStorage.getItem('ziv-browse-view-v1')!)).toEqual({ sort: 'size', dir: 'asc', view: 'grid', preview: false });
  });

  it('keeps the hidden-files key apart', () => {
    writeShowHidden(true);
    writeBrowseView({ sort: 'name', dir: 'asc', view: 'list', preview: true });

    expect(readShowHidden()).toBe(true);
    expect(localStorage.getItem('ziv-browse-hidden-v1')).toBe('true');
  });

  it.each(['not json', '[]', '"x"', '42', 'null'])('falls back to the defaults for %s', (raw) => {
    localStorage.setItem('ziv-browse-view-v1', raw);

    expect(readBrowseView()).toEqual(DEFAULT_BROWSE_VIEW);
  });

  it('replaces only the invalid fields', () => {
    localStorage.setItem('ziv-browse-view-v1', JSON.stringify({ sort: 'name', dir: 'sideways', view: 'grid', preview: 'yes' }));

    expect(readBrowseView()).toEqual({ sort: 'name', dir: 'desc', view: 'grid', preview: true });
  });
});

describe('sort choices', () => {
  it('offers the six orders, each with its own label', () => {
    expect(SORT_CHOICES.map((choice) => [choice.order.sort, choice.order.dir])).toEqual([
      ['name', 'asc'], ['name', 'desc'], ['modified', 'desc'], ['modified', 'asc'], ['size', 'desc'], ['size', 'asc'],
    ]);
    expect(new Set(SORT_CHOICES.map((choice) => choice.id)).size).toBe(6);
    for (const choice of SORT_CHOICES) expect(sortChoiceLabel(choice.order)).toBe(choice.label);
  });
});

describe('previewFacts', () => {
  const file = { name: 'fox.PNG', size: 2048, modified: new Date(2026, 10, 3, 14, 5, 9).getTime() / 1000 };
  const facts = (kind: Parameters<typeof previewFacts>[0], info: Parameters<typeof previewFacts>[2]) => Object.fromEntries(previewFacts(kind, file, info).map((fact) => [fact.label, fact.value]));

  it('lists pixels, file size and type, and the full date for an image', () => {
    expect(facts('image', { width: 640, height: 480 })).toEqual({ Pixels: '640 × 480', File: '2 KB · PNG', Modified: '2026-11-03 14:05:09' });
  });

  it('lists length, format and file size for audio', () => {
    expect(previewFacts('audio', { ...file, name: 'take.flac' }, { seconds: 72.4, sample_rate: 44100, channels: 2 })).toEqual([
      { label: 'Length', value: '1:12' },
      { label: 'Format', value: 'FLAC · 44.1 kHz · stereo' },
      { label: 'File', value: '2 KB' },
    ]);
  });

  it('lists length, frames, frame rate and pixels for a video', () => {
    expect(facts('video', { seconds: 10, frames: 240, fps: 23.976, width: 1280, height: 720 })).toEqual({ Length: '0:10', Frames: '240', 'Frame rate': '23.98 fps', Pixels: '1280 × 720', File: '2 KB · PNG' });
  });

  it('reads a dash for the facts the server has not sent', () => {
    expect(facts('video', null)).toMatchObject({ Length: '—', Frames: '—', 'Frame rate': '—', Pixels: '—' });
    expect(facts('image', null).Pixels).toBe('—');
  });
});

describe('pickerStart', () => {
  it('opens the media pickers in the project folder', () => {
    for (const purpose of ['reference_image', 'audio_track', 'source_video'] as const) {
      expect(pickerStart(purpose, '/out/Night Bus')).toBe('/out/Night Bus');
    }
  });

  it('leaves every other purpose at its own place', () => {
    for (const purpose of ['prompt_file', 'prompt_file_folder', 'output_directory', 'checkpoint_file', 'lora_file'] as const) {
      expect(pickerStart(purpose, '/out/Night Bus')).toBeNull();
    }
  });

  it('has no start without a project', () => {
    expect(pickerStart('reference_image', null)).toBeNull();
  });
});
