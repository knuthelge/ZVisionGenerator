import { describe, expect, it } from 'vitest';

import { sentReferenceCount, splitImageFiles } from './references';

const file = (name: string, type = 'image/png'): File => new File(['x'], name, { type });

describe('sentReferenceCount', () => {
  it('counts the rows that hold a path or an uploaded file, and not an empty one', () => {
    expect(sentReferenceCount([])).toBe(0);
    expect(sentReferenceCount([{ path: '/a.png', file: null }, { path: '', file: file('b.png') }, { path: '', file: null }])).toBe(2);
  });
});

describe('splitImageFiles', () => {
  it('keeps PNG, JPEG and WebP files by extension, whatever the case, and counts the rest', () => {
    const { supported, unsupported } = splitImageFiles([file('a.png'), file('notes.txt', 'text/plain'), file('b.webp'), file('c.gif'), file('E.PNG'), file('f.jpeg')]);

    expect(supported.map((f) => f.name)).toEqual(['a.png', 'b.webp', 'E.PNG', 'f.jpeg']);
    expect(unsupported).toBe(2);
  });
});

