import { beforeEach, describe, expect, it, vi } from 'vitest';

const addToast = vi.hoisted(() => vi.fn());
vi.mock('$lib/state/toasts.svelte', () => ({ addToast }));

import { completionToast } from './completionToast';

beforeEach(() => addToast.mockReset());

describe('completionToast', () => {
  it('is a plain success when the result is in the pick', () => {
    completionToast([{ project: 'Night Bus' }], 'Night Bus');
    expect(addToast).toHaveBeenCalledWith(expect.not.stringContaining('Night Bus'), 'success');
  });

  it('is a plain success with All files, whatever project the result is in', () => {
    completionToast([{ project: 'Night Bus' }, { project: null }], null);
    expect(addToast).toHaveBeenCalledWith(expect.not.stringContaining('Night Bus'), 'success');
  });

  it('names the project a result landed in when it is not the pick', () => {
    completionToast([{ project: 'Night Bus' }, { project: 'Zoo' }], 'Night Bus');
    expect(addToast).toHaveBeenCalledWith(expect.stringContaining('Zoo'), 'success');
  });

  it('names No project for a result at the top of the output folder', () => {
    completionToast([{ project: null }], 'Night Bus');
    expect(addToast).toHaveBeenCalledWith(expect.stringContaining('No project'), 'success');
  });

  it('is a plain success for a job with no outputs', () => {
    completionToast([], 'Night Bus');
    expect(addToast).toHaveBeenCalledOnce();
  });
});
