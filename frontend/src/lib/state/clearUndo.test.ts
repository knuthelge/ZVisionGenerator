import { beforeEach, describe, expect, it } from 'vitest';

import { ClearUndo, stillCleared, takeFields } from './clearUndo.svelte';
import { draft } from './draft.svelte';
import { WriterRun } from './writerRun.svelte';
import type { PromptEnhancerContract } from '$lib/types';

beforeEach(() => {
  localStorage.clear();
  draft.reset();
});

describe('takeFields and stillCleared', () => {
  it('remembers the fields as they were, and knows them cleared only while they hold what Clear left', () => {
    const state = { ...draft.state, audioStyle: 'folk', lyricsVersions: [{ text: 'a', mine: true }] };
    const taken = takeFields(state, { audioStyle: '', lyricsVersions: [] }, 'enhanced');

    expect(taken.before).toEqual({ audioStyle: 'folk', lyricsVersions: [{ text: 'a', mine: true }] });
    expect(taken.view).toBe('enhanced');
    expect(stillCleared({ ...state, audioStyle: '', lyricsVersions: [] }, taken)).toBe(true);
    expect(stillCleared({ ...state, audioStyle: 'j', lyricsVersions: [] }, taken)).toBe(false);
  });
});

describe('ClearUndo', () => {
  it('clears the fields, offers Undo, and Undo restores them with the view', () => {
    draft.patch({ audioStyle: 'folk', soundPrompt: 'rain' });
    const cleared = new ClearUndo<string>();
    expect(cleared.offered).toBe(false);

    cleared.clear({ audioStyle: '' }, 'enhanced');

    expect(draft.state.audioStyle).toBe('');
    expect(cleared.offered).toBe(true);
    expect(cleared.undo()).toEqual({ view: 'enhanced' });
    expect(draft.state).toMatchObject({ audioStyle: 'folk', soundPrompt: 'rain' });
    expect(cleared.offered).toBe(false);
    expect(cleared.undo()).toBeNull();
  });

  it('stops offering Undo once a cleared field changes, and when it is forgotten', () => {
    draft.patch({ audioStyle: 'folk' });
    const cleared = new ClearUndo();
    cleared.clear({ audioStyle: '' }, null);

    draft.update('audioStyle', 'jazz');

    expect(cleared.offered).toBe(false);
    expect(cleared.undo()).toBeNull();
    expect(draft.state.audioStyle).toBe('jazz');

    draft.update('audioStyle', 'blues');
    cleared.clear({ audioStyle: '' }, null);
    cleared.forget();
    expect(cleared.offered).toBe(false);
  });

  it('keeps the newest Clear when there is another', () => {
    draft.patch({ audioStyle: 'folk' });
    const cleared = new ClearUndo();
    cleared.clear({ audioStyle: '' }, null);
    draft.update('audioStyle', 'jazz');
    cleared.clear({ audioStyle: '' }, null);

    cleared.undo();

    expect(draft.state.audioStyle).toBe('jazz');
  });
});

describe('WriterRun tier', () => {
  const contract = { smart: { downloaded: true, download_size_label: null, unavailable: null } } as PromptEnhancerContract;

  it('reads the shared enhancer, or the one Fill in has of its own', () => {
    draft.patch({ enhanceModel: 'smart', describeTier: 'fast' });

    expect(new WriterRun().tier(contract)).toBe('smart');
    expect(new WriterRun('describeTier').tier(contract)).toBe('fast');
  });

  it('falls back to Fast where Smart is unavailable', () => {
    draft.patch({ describeTier: 'smart' });

    expect(new WriterRun('describeTier').tier({ smart: { downloaded: false, download_size_label: null, unavailable: 'No' } } as PromptEnhancerContract)).toBe('fast');
  });
});
