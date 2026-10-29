import { describe, expect, it } from 'vitest';

import { isMissingRow, referenceGuard, splitImageFiles } from './references';

const file = (name: string, type = 'image/png'): File => new File(['x'], name, { type });

describe('splitImageFiles', () => {
  it('keeps PNG, JPEG and WebP files by extension, whatever the case, and counts the rest', () => {
    const { supported, unsupported } = splitImageFiles([file('a.png'), file('notes.txt', 'text/plain'), file('b.webp'), file('c.gif'), file('E.PNG'), file('f.jpeg')]);

    expect(supported.map((f) => f.name)).toEqual(['a.png', 'b.webp', 'E.PNG', 'f.jpeg']);
    expect(unsupported).toBe(2);
  });
});

describe('missing references', () => {
  const rows = [{ path: '/a.png', file: null }, { path: '/gone.png', file: null }, { path: '', file: file('up.png') }];

  it('flags a path found missing, never a row that holds an upload', () => {
    expect(isMissingRow(rows[1]!, ['/gone.png'])).toBe(true);
    expect(isMissingRow(rows[0]!, ['/gone.png'])).toBe(false);
    expect(isMissingRow(rows[2]!, [''])).toBe(false);
  });

  it('counts an upload whose file was lost (an empty path and no file) as missing, in its own place', () => {
    const lost = [{ path: '/a.png', file: null }, { path: '', file: null }, { path: '/c.png', file: null }];

    expect(referenceGuard(lost, [])).toContain('2');
  });

  it('blocks Generate for the first missing row and for none when every row can be read', () => {
    expect(referenceGuard(rows, ['/gone.png'])).toContain('2');
    expect(referenceGuard(rows, [])).toBeNull();
  });
});
