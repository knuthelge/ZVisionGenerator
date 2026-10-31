import { describe, expect, it } from 'vitest';

import { splitImageFiles } from './references';

const file = (name: string, type = 'image/png'): File => new File(['x'], name, { type });

describe('splitImageFiles', () => {
  it('keeps PNG, JPEG and WebP files by extension, whatever the case, and counts the rest', () => {
    const { supported, unsupported } = splitImageFiles([file('a.png'), file('notes.txt', 'text/plain'), file('b.webp'), file('c.gif'), file('E.PNG'), file('f.jpeg')]);

    expect(supported.map((f) => f.name)).toEqual(['a.png', 'b.webp', 'E.PNG', 'f.jpeg']);
    expect(unsupported).toBe(2);
  });
});

