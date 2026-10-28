import { describe, expect, it } from 'vitest';
import { addTrigger, cleanTrigger, containsTrigger, splitTriggers } from './triggers';

describe('containsTrigger', () => {
  it('matches a whole word in any case', () => {
    expect(containsTrigger('A T3chnic4lly photo', 't3chnic4lly')).toBe(true);
    expect(containsTrigger('a photo', 't3chnic4lly')).toBe(false);
  });

  it('does not match inside a longer word', () => {
    expect(containsTrigger('a crtanimations reel', 'crtanim')).toBe(false);
    expect(containsTrigger('a crtanim_v2 reel', 'crtanim')).toBe(false);
    expect(containsTrigger('crtanim, a reel', 'crtanim')).toBe(true);
  });

  it('matches a phrase across any spacing and keeps regex characters literal', () => {
    expect(containsTrigger('Clay   animation of a fox', 'clay animation')).toBe(true);
    expect(containsTrigger('in the style of (x)', 'style of (x)')).toBe(true);
    expect(containsTrigger('anything', '.*')).toBe(false);
  });

  it('never matches a blank trigger', () => {
    expect(containsTrigger('anything', '  ')).toBe(false);
  });
});

describe('addTrigger', () => {
  it('puts the trigger first, followed by a comma and a space', () => {
    expect(addTrigger('a lighthouse', 'ohwx')).toBe('ohwx, a lighthouse');
  });

  it('leaves the separator for the user to type after in an empty prompt', () => {
    expect(addTrigger('', 'ohwx')).toBe('ohwx, ');
  });

  it('cleans the trigger the way the server stores it', () => {
    expect(addTrigger('x', '  clay   animation ')).toBe('clay animation, x');
    expect(cleanTrigger('  clay   animation ')).toBe('clay animation');
  });
});

describe('splitTriggers', () => {
  it('splits a comma-separated trigger into trimmed words without blanks or repeats', () => {
    expect(splitTriggers('Pixel Art,  PixArFK , ,pixel art')).toEqual(['Pixel Art', 'PixArFK']);
  });

  it('gives nothing for no trigger', () => {
    expect(splitTriggers(null)).toEqual([]);
    expect(splitTriggers(undefined)).toEqual([]);
    expect(splitTriggers('  ')).toEqual([]);
  });
});
