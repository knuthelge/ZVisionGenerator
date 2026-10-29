import { describe, expect, it } from 'vitest';

import { buildQuery, firstValues, listOf, parseQuery } from './prefillParams';

describe('prefill params', () => {
  it('turns a repeated name into a list in order and leaves a single one a string', () => {
    expect(parseQuery('workflow=txt2img&reference_path=%2Fa.png&reference_path=%2Fb.png')).toEqual({
      workflow: 'txt2img',
      reference_path: ['/a.png', '/b.png'],
    });
    expect(parseQuery('')).toEqual({});
  });

  it('repeats a list name once per value and round-trips with the parser', () => {
    const params = { workflow: 'txt2img', reference_path: ['/a b.png', '/c.png'], model: 'klein4b' };

    expect(buildQuery(params)).toBe('workflow=txt2img&reference_path=%2Fa+b.png&reference_path=%2Fc.png&model=klein4b');
    expect(parseQuery(buildQuery(params))).toEqual(params);
  });

  it('reduces each parameter to its first value, an empty list to an empty string', () => {
    expect(firstValues({ a: 'x', b: ['y', 'z'], c: [] })).toEqual({ a: 'x', b: 'y', c: '' });
    expect(listOf('x')).toEqual(['x']);
    expect(listOf(['x', 'y'])).toEqual(['x', 'y']);
  });
});
