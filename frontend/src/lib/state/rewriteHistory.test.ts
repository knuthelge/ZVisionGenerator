import { describe, expect, it } from 'vitest';

import { clearRewritesPatch, editEnhancedPatch, pushRewrite, rewriteResultPatch, rewriteStepper, scratchRewritePatch, stepRewritePatch } from './rewriteHistory';
import type { DraftState, EnhanceRecord, EnhancedRewrite, EnhancedSource } from '$lib/types';

const RECORD: EnhanceRecord = { settings: { style: 'photo' }, model: 'org/fast', format: 'text' };

function rewrite(prompt: string, from: Partial<EnhancedSource> = {}, scene = ''): EnhancedRewrite {
  return { prompt, scene, from: { prompt: 'a fox', scene: '', mode: 'image', record: RECORD, ...from } };
}

function state(overrides: Partial<DraftState> = {}): DraftState {
  return {
    workflow: 'txt2img',
    prompt: 'a fox',
    scene: '',
    enhanceAuto: false,
    enhancedPrompt: '',
    enhancedScene: '',
    enhancedFrom: null,
    enhancedHistory: [],
    enhancedScratched: null,
    extendSourcePath: null,
    extendFrames: null,
    ...overrides,
  } as DraftState;
}

/** A draft that shows `shown` of three rewrites of "a fox". */
function showing(shown: EnhancedRewrite, history: EnhancedRewrite[], overrides: Partial<DraftState> = {}): DraftState {
  return state({ enhancedPrompt: shown.prompt, enhancedScene: shown.scene, enhancedFrom: shown.from, enhancedHistory: history, ...overrides });
}

describe('pushRewrite', () => {
  it('appends the newest rewrite last', () => {
    const first = rewrite('one');
    const second = rewrite('two');
    expect(pushRewrite(pushRewrite([], first), second)).toEqual([first, second]);
  });

  it('keeps only the last five of one source, dropping the oldest', () => {
    let history: EnhancedRewrite[] = [];
    for (let i = 1; i <= 6; i += 1) history = pushRewrite(history, rewrite(`rewrite ${i}`));

    expect(history.map((entry) => entry.prompt)).toEqual(['rewrite 2', 'rewrite 3', 'rewrite 4', 'rewrite 5', 'rewrite 6']);
  });

  it.each([
    ['prompt', { prompt: 'a cat' }],
    ['extend frames', { extendFrames: 48 }],
  ])('keeps the earlier rewrites made from another %s', (_name, from) => {
    const history = pushRewrite([rewrite('one'), rewrite('two')], rewrite('three', from));

    expect(history.map((entry) => entry.prompt)).toEqual(['one', 'two', 'three']);
  });

  it('moves a Reuse of an earlier version to the newest place, so the stepper goes back to the other version and ends', () => {
    const [a, b] = [rewrite('A'), rewrite('B')];
    const history = pushRewrite([a, b], rewrite('A'));

    expect(history.map((entry) => entry.prompt)).toEqual(['B', 'A']);
    const shown = state({ enhancedPrompt: 'A', enhancedHistory: history });
    expect(rewriteStepper(shown)).toEqual({ index: 2, count: 2 });
    const back = { ...shown, ...stepRewritePatch(shown, 1) } as DraftState;
    expect(back.enhancedPrompt).toBe('B');
    expect(rewriteStepper(back)).toEqual({ index: 1, count: 2 });
  });

  it('adds no duplicate when the newest version is reused', () => {
    const history = pushRewrite([rewrite('A'), rewrite('B')], rewrite('B'));

    expect(history.map((entry) => entry.prompt)).toEqual(['A', 'B']);
  });
});

describe('rewriteResultPatch', () => {
  it('shows the result, adds it as the newest version and forgets a scratch', () => {
    const history = [rewrite('one')];

    const patch = rewriteResultPatch(state({ enhancedHistory: history, enhancedScratched: 0 }), rewrite('two'));

    expect(patch).toMatchObject({ enhancedPrompt: 'two', enhancedScratched: null });
    expect(patch.enhancedHistory?.map((entry) => entry.prompt)).toEqual(['one', 'two']);
  });
});

describe('rewriteStepper', () => {
  const entries = [rewrite('one'), rewrite('two'), rewrite('three')];

  it('numbers the shown rewrite among all of them', () => {
    expect(rewriteStepper(showing(entries[1], entries))).toEqual({ index: 2, count: 3 });
    expect(rewriteStepper(showing(entries[2], entries))).toEqual({ index: 3, count: 3 });
  });

  it.each([
    ['there are fewer than two rewrites', () => showing(entries[0], [entries[0]])],
    ['the draft has no rewrites', () => state()],
    ['the shown pair is not an entry', () => showing(entries[1], entries, { enhancedPrompt: 'two, edited by hand' })],
    ['auto-enhance is on', () => showing(entries[1], entries, { enhanceAuto: true })],
  ])('is null when %s', (_name, draftState) => {
    expect(rewriteStepper(draftState())).toBeNull();
  });

  it('stays visible while the Enhanced text is out of date', () => {
    expect(rewriteStepper(showing(entries[1], entries, { prompt: 'a fox, edited' }))).toEqual({ index: 2, count: 3 });
  });

  it('is null while Enhance runs', () => {
    expect(rewriteStepper(showing(entries[1], entries), true)).toBeNull();
  });

  it('shows "– of N" after Scratch, with ‹ bringing back the version that was scratched', () => {
    const scratched = { ...state({ enhancedHistory: entries }), ...scratchRewritePatch(showing(entries[1], entries)) } as DraftState;

    expect(rewriteStepper(scratched)).toEqual({ index: null, count: 3, back: 2 });
    const back = { ...scratched, ...stepRewritePatch(scratched, 2) } as DraftState;
    expect(back).toMatchObject({ enhancedPrompt: 'two', enhancedScratched: null });
    expect(rewriteStepper(back)).toEqual({ index: 2, count: 3 });
  });

  it('shows "– of 1" after Scratch with one version, and none for an empty draft that was never scratched', () => {
    const only = showing(entries[0], [entries[0]]);

    expect(rewriteStepper({ ...only, ...scratchRewritePatch(only) } as DraftState)).toEqual({ index: null, count: 1, back: 1 });
    expect(rewriteStepper(state({ enhancedHistory: entries }))).toBeNull();
  });

  it('matches the scene as well as the action', () => {
    const video = [rewrite('She waves.', { mode: 'video', scene: 'A pier.' }, 'A pier at dawn.'), rewrite('She waves.', { mode: 'video', scene: 'A pier.' }, 'A pier at dusk.')];
    const shown = showing(video[0], video, { workflow: 'txt2vid', scene: 'A pier.' });

    expect(rewriteStepper(shown)).toEqual({ index: 1, count: 2 });
  });
});

describe('editEnhancedPatch', () => {
  const entries = [rewrite('one'), rewrite('two')];

  it('replaces the shown version, found by its text before the edit, and drops its record', () => {
    const shown = showing(entries[0], entries);

    const patch = editEnhancedPatch(shown, { prompt: 'one, edited', scene: '' });

    expect(patch).toMatchObject({ enhancedPrompt: 'one, edited', enhancedScratched: null });
    expect(patch.enhancedFrom).not.toHaveProperty('record');
    expect(patch.enhancedFrom).toMatchObject({ prompt: 'a fox' });
    expect(patch.enhancedHistory?.map((entry) => entry.prompt)).toEqual(['one, edited', 'two']);
    expect(patch.enhancedHistory?.[0].from).not.toHaveProperty('record');
    expect(patch.enhancedHistory?.[1].from.record).toEqual(RECORD);
    // The next key finds the edited version again.
    const again = editEnhancedPatch({ ...shown, ...patch } as DraftState, { prompt: 'one, edited!', scene: '' });
    expect(again.enhancedHistory?.map((entry) => entry.prompt)).toEqual(['one, edited!', 'two']);
  });

  it('edits the scene of a video pair the same way', () => {
    const video = [rewrite('She waves.', { mode: 'video', scene: 'A pier.' }, 'A pier at dawn.')];
    const shown = showing(video[0], video, { workflow: 'txt2vid', scene: 'A pier.' });

    const patch = editEnhancedPatch(shown, { prompt: 'She waves.', scene: 'A pier at dusk.' });

    expect(patch.enhancedHistory).toMatchObject([{ prompt: 'She waves.', scene: 'A pier at dusk.' }]);
  });

  it('is a Scratch when both parts are emptied', () => {
    expect(editEnhancedPatch(showing(entries[1], entries), { prompt: ' ', scene: '' })).toEqual({ enhancedPrompt: '', enhancedScene: '', enhancedFrom: null, enhancedScratched: 1 });
  });

  it('keeps hand-typed text outside the versions, and leaves them alone', () => {
    const typed = state({ enhancedPrompt: 'by hand', enhancedHistory: entries });

    expect(editEnhancedPatch(typed, { prompt: 'by hand, more', scene: '' }).enhancedHistory).toBe(entries);
  });
});

describe('clearRewritesPatch', () => {
  it('drops the Enhanced text, its source and every version', () => {
    expect(clearRewritesPatch()).toEqual({ enhancedPrompt: '', enhancedScene: '', enhancedFrom: null, enhancedHistory: [], enhancedScratched: null });
  });
});
