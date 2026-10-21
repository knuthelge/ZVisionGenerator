import { describe, expect, it } from 'vitest';

import { folderOf, promptToSave, saveTargets, type SaveSelection, type SaveView } from './savePrompt';

type State = Parameters<typeof promptToSave>[0];

const state = (overrides: Partial<State> = {}): State => ({
  prompt: '',
  scene: '',
  negativePrompt: '',
  enhancedPrompt: '',
  enhancedScene: '',
  enhanceAuto: false,
  ...overrides,
});

const view = (overrides: Partial<SaveView> = {}): SaveView => ({
  enhancedView: false,
  enhancing: false,
  jsonMode: false,
  pairMode: false,
  showNegative: true,
  ...overrides,
});

const entry = (value: { prompt: string; scene?: string; negative?: string }): SaveSelection => ({ entry: value });
const block = (value: 'json' | 'enhancing' | 'empty'): SaveSelection => ({ block: value });

const cases: [string, State, SaveView, SaveSelection][] = [
  ['image Original view with a negative shown', state({ prompt: 'a fox', negativePrompt: 'blurry' }), view(), entry({ prompt: 'a fox', negative: 'blurry' })],
  ['image Original view with the negative hidden', state({ prompt: 'a fox', negativePrompt: 'blurry' }), view({ showNegative: false }), entry({ prompt: 'a fox' })],
  ['image Original view with a blank negative', state({ prompt: 'a fox', negativePrompt: '  ' }), view(), entry({ prompt: 'a fox' })],
  ['image Original view with a blank prompt', state({ prompt: '  ' }), view(), block('empty')],
  ['image Original view ignores the scene', state({ prompt: 'a fox', scene: 'stale' }), view(), entry({ prompt: 'a fox' })],
  ['image Enhanced view uses the enhanced text', state({ prompt: 'a fox', enhancedPrompt: 'A red fox.' }), view({ enhancedView: true }), entry({ prompt: 'A red fox.' })],
  ['image Enhanced view with no enhanced text', state({ prompt: 'a fox' }), view({ enhancedView: true }), block('empty')],
  ['Enhanced view with auto-enhance', state({ prompt: 'a fox', enhancedPrompt: 'A red fox.', enhanceAuto: true }), view({ enhancedView: true }), block('empty')],
  ['Enhanced view while enhancing', state({ prompt: 'a fox', enhancedPrompt: 'A red fox.' }), view({ enhancedView: true, enhancing: true }), block('enhancing')],
  ['Original view while enhancing', state({ prompt: 'a fox' }), view({ enhancing: true }), entry({ prompt: 'a fox' })],
  ['JSON mode', state({ prompt: 'a fox' }), view({ jsonMode: true }), block('json')],
  ['video Original view with scene and action', state({ scene: 'A street.', prompt: 'She walks.', negativePrompt: 'blurry' }), view({ pairMode: true }), entry({ scene: 'A street.', prompt: 'She walks.', negative: 'blurry' })],
  ['video Original view with a scene only', state({ scene: 'A street.' }), view({ pairMode: true }), block('empty')],
  ['video Original view with an action only', state({ prompt: 'She walks.' }), view({ pairMode: true }), entry({ prompt: 'She walks.' })],
  ['video Enhanced view with only the action enhanced', state({ scene: 'A street.', prompt: 'walks', enhancedPrompt: 'She walks slowly.' }), view({ pairMode: true, enhancedView: true }), entry({ scene: 'A street.', prompt: 'She walks slowly.' })],
  ['video Enhanced view with both parts enhanced', state({ scene: 'street', prompt: 'walks', enhancedScene: 'A wet street.', enhancedPrompt: 'She walks slowly.' }), view({ pairMode: true, enhancedView: true }), entry({ scene: 'A wet street.', prompt: 'She walks slowly.' })],
  ['video Enhanced view with only the scene enhanced', state({ scene: 'street', prompt: 'walks', enhancedScene: 'A wet street.' }), view({ pairMode: true, enhancedView: true }), entry({ scene: 'A wet street.', prompt: 'walks' })],
  ['video Enhanced view with only the scene enhanced and no action', state({ scene: 'street', enhancedScene: 'A wet street.' }), view({ pairMode: true, enhancedView: true }), block('empty')],
  ['video Enhanced view with nothing enhanced', state({ scene: 'street', prompt: 'walks' }), view({ pairMode: true, enhancedView: true }), block('empty')],
  ['whitespace is trimmed', state({ prompt: '  a fox \n', negativePrompt: ' blurry ' }), view(), entry({ prompt: 'a fox', negative: 'blurry' })],
];

describe('promptToSave', () => {
  it.each(cases)('%s', (_name, draftState, saveView, expected) => {
    expect(promptToSave(draftState, saveView)).toEqual(expected);
  });
});

describe('saveTargets', () => {
  it('keeps unique non-blank paths in first-seen order', () => {
    expect(saveTargets('/b.yaml', null, '/a.yaml', undefined, '/b.yaml', '', '  ', '/c.yaml', '/a.yaml')).toEqual(['/b.yaml', '/a.yaml', '/c.yaml']);
  });

  it('returns nothing for no candidates', () => {
    expect(saveTargets()).toEqual([]);
    expect(saveTargets(null, undefined)).toEqual([]);
  });
});

describe('folderOf', () => {
  it('returns the folder with its trailing separator', () => {
    expect(folderOf('/a/b.yaml')).toBe('/a/');
    expect(folderOf('C:\\x\\y.yaml')).toBe('C:\\x\\');
  });

  it('returns null without a path or a folder', () => {
    expect(folderOf(null)).toBeNull();
    expect(folderOf('')).toBeNull();
    expect(folderOf('b.yaml')).toBeNull();
  });
});
