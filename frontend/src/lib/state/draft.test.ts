import { describe, it, expect, beforeEach } from 'vitest';
import { draft } from './draft.svelte';

describe('draft store', () => {
  beforeEach(() => {
    localStorage.clear();
    draft.reset();
  });

  it('loads default state when localStorage is empty', () => {
    expect(draft.state.workflow).toBe('txt2img');
    expect(draft.state.prompt).toBe('');
  });

  it('updates a field and saves to localStorage', () => {
    draft.update('prompt', 'test prompt');
    expect(draft.state.prompt).toBe('test prompt');
    expect(localStorage.getItem('ziv-workspace-draft-v1')).toContain('test prompt');
  });

  it('applies URL prefill correctly', () => {
    draft.loadFromUrl({ workflow: 'img2img', prompt: 'from url', model: 'flux-dev' });
    expect(draft.state.workflow).toBe('img2img');
    expect(draft.state.prompt).toBe('from url');
  });

  it('ignores invalid workflow in URL prefill', () => {
    draft.loadFromUrl({ workflow: 'invalid-workflow' });
    expect(draft.state.workflow).toBe('txt2img'); // unchanged
  });
});
