import { describe, expect, it } from 'vitest';

import { pushRewrite, rewriteStepper } from './rewriteHistory';
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
  ])('starts a new list when the %s differs', (_name, from) => {
    const history = pushRewrite([rewrite('one'), rewrite('two')], rewrite('three', from));

    expect(history.map((entry) => entry.prompt)).toEqual(['three']);
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
    ['the Enhanced text is out of date', () => showing(entries[1], entries, { prompt: 'a fox, edited' })],
    ['auto-enhance is on', () => showing(entries[1], entries, { enhanceAuto: true })],
  ])('is null when %s', (_name, draftState) => {
    expect(rewriteStepper(draftState())).toBeNull();
  });

  it('matches the scene as well as the action', () => {
    const video = [rewrite('She waves.', { mode: 'video', scene: 'A pier.' }, 'A pier at dawn.'), rewrite('She waves.', { mode: 'video', scene: 'A pier.' }, 'A pier at dusk.')];
    const shown = showing(video[0], video, { workflow: 'txt2vid', scene: 'A pier.' });

    expect(rewriteStepper(shown)).toEqual({ index: 1, count: 2 });
  });
});
