import { describe, expect, it } from 'vitest';

import { headerMarks } from './headerMarks';

const state = (overrides: Partial<Parameters<typeof headerMarks>[0]> = {}) => ({ open: false, used: false, stale: false, running: false, ...overrides });

describe('headerMarks, with the Enhanced tab closed', () => {
  it('has no mark without Enhanced text', () => {
    expect(headerMarks(state())).toEqual({ mark: null, after: null, description: '' });
  });

  it('marks Enhanced text that Generate sends with a dot', () => {
    expect(headerMarks(state({ used: true }))).toEqual({ mark: 'dot', after: null, description: 'sent by Generate' });
  });

  it('marks Enhanced text that is out of date with a warning instead of the dot', () => {
    expect(headerMarks(state({ used: true, stale: true }))).toEqual({ mark: 'warn', after: null, description: 'sent by Generate, out of date' });
  });

  it('shows a spinner while Enhance runs, one mark whatever else is true', () => {
    expect(headerMarks(state({ running: true }))).toEqual({ mark: 'spinner', after: null, description: 'enhancing' });
    expect(headerMarks(state({ used: true, stale: true, running: true }))).toEqual({ mark: 'spinner', after: null, description: 'sent by Generate, out of date, enhancing' });
  });
});

describe('headerMarks, with the Enhanced tab open', () => {
  it('has no mark in the tab and nothing after it without Enhanced text', () => {
    expect(headerMarks(state({ open: true }))).toEqual({ mark: null, after: null, description: '' });
  });

  it('puts ✕ Scratch after the tab while the Enhanced text is in use, out of date or not', () => {
    expect(headerMarks(state({ open: true, used: true })).after).toBe('scratch');
    expect(headerMarks(state({ open: true, used: true, stale: true })).after).toBe('scratch');
  });

  it('puts the stop ring in the place of the ✕ while Enhance runs, with or without text', () => {
    expect(headerMarks(state({ open: true, running: true })).after).toBe('ring');
    expect(headerMarks(state({ open: true, used: true, running: true })).after).toBe('ring');
  });

  it('never marks inside the open tab', () => {
    for (const used of [false, true]) for (const stale of [false, true]) for (const running of [false, true]) expect(headerMarks(state({ open: true, used, stale, running })).mark).toBeNull();
  });
});
