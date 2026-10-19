import { describe, expect, it } from 'vitest';
import type { EnhanceAxis, PromptDocument } from '$lib/types';
import {
  canMakeChoice,
  choiceAt,
  choiceOptions,
  copyEntry,
  dropPosition,
  enhanceSettingsFor,
  enhanceShortLabel,
  enhanceSummary,
  flattenValue,
  moveItem,
  newEntry,
  optionId,
  remapOptionIds,
  renameRefs,
  renameRefsInValue,
  replaceChoice,
  setNameProblem,
  snippetNameProblem,
  snippetQueryAt,
  structuredUses,
  tokenizePrompt,
  uniqueName,
  valueAsText,
} from './document';

const AXES: EnhanceAxis[] = [
  { key: 'style', label: 'Style', multi: false, video_only: false, audio_only: false, options: [{ slug: 'keep', label: 'Keep', description: '', group: null }, { slug: 'cinematic', label: 'Cinematic', description: '', group: null }], default: ['keep'] },
  { key: 'mood', label: 'Mood', multi: false, video_only: false, audio_only: false, options: [{ slug: 'keep', label: 'Keep', description: '', group: null }, { slug: 'eerie', label: 'Eerie', description: '', group: null }], default: ['keep'] },
  { key: 'details', label: 'Details', multi: true, video_only: false, audio_only: false, options: [{ slug: 'lighting', label: 'Lighting', description: '', group: null }, { slug: 'camera', label: 'Camera & lens', description: '', group: null }], default: ['lighting'] },
  { key: 'length', label: 'Length', multi: false, video_only: false, audio_only: false, options: [{ slug: 'same', label: 'Same', description: '', group: null }, { slug: 'longer', label: 'Longer', description: '', group: null }], default: ['same'] },
  { key: 'motion', label: 'Motion', multi: true, video_only: true, audio_only: false, options: [{ slug: 'action', label: 'Action / sequence', description: '', group: null }], default: ['action'] },
];
const DEFAULTS = { style: 'keep', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: ['lighting'], length: 'same', motion: ['action'], sound: [], note: '' };

describe('tokenizePrompt', () => {
  it('splits text, snippet references and choices', () => {
    expect(tokenizePrompt('a {red|blue} car at $diner.')).toEqual([
      { kind: 'text', text: 'a ' },
      { kind: 'choice', start: 2, end: 12, options: [[{ kind: 'text', text: 'red' }], [{ kind: 'text', text: 'blue' }]] },
      { kind: 'text', text: ' car at ' },
      { kind: 'snippet', name: 'diner' },
      { kind: 'text', text: '.' },
    ]);
  });

  it('keeps references inside choices and empty options', () => {
    const [choice] = tokenizePrompt('{$a|}');
    expect(choice).toEqual({ kind: 'choice', start: 0, end: 5, options: [[{ kind: 'snippet', name: 'a' }], []] });
  });

  it('only tokenizes the innermost choice of a nested one', () => {
    const kinds = tokenizePrompt('{Nikon {50mm|35mm}|Canon}').map((token) => token.kind);
    expect(kinds).toEqual(['text', 'choice', 'text']);
  });
});

describe('choices', () => {
  it('reads and replaces a choice, keeping empty options and spacing', () => {
    const text = 'a {|very } big dog';
    expect(choiceOptions(text, 2, 10)).toEqual(['', 'very ']);
    expect(replaceChoice(text, 2, 10, ['', 'very ', 'quite '])).toBe('a {|very |quite } big dog');
  });

  it('turns a single option into plain text', () => {
    expect(replaceChoice('a {red|blue} car', 2, 12, ['red'])).toBe('a red car');
  });
});

describe('choices while typing', () => {
  it('finds the choice around the caret', () => {
    expect(choiceAt('a {red|blue} car', 5)).toEqual({ start: 2, end: 12 });
    expect(choiceAt('a {red|blue} car', 2)).toBeNull();
    expect(choiceAt('a {red|blue} car', 14)).toBeNull();
  });

  it('only lets plain selected text become a choice', () => {
    expect(canMakeChoice('red car')).toBe(true);
    expect(canMakeChoice('  ')).toBe(false);
    expect(canMakeChoice('{a|b}')).toBe(false);
  });
});

describe('snippetQueryAt', () => {
  it('returns the partial name being typed after $', () => {
    expect(snippetQueryAt('a $di', 5)).toBe('di');
    expect(snippetQueryAt('a $', 3)).toBe('');
    expect(snippetQueryAt('a di', 4)).toBeNull();
  });
});

describe('flattenValue', () => {
  it('flattens like the prompt loader', () => {
    expect(flattenValue({ Subjects: ['Lisa', { Nina: { Hair: 'Tidy' } }], Style: ' warm ', Empty: '' })).toBe('Subjects: Lisa. Nina: Hair: Tidy. Style: warm. Empty');
    expect(flattenValue(true)).toBe('true');
    expect(flattenValue(null)).toBe('');
  });

  it('turns fields into the same text', () => {
    expect(valueAsText({ kind: 'fields', fields: [{ key: 'Subject', value: 'a fox' }, { key: 'Style', value: '$light' }] })).toBe('Subject: a fox. Style: $light');
  });
});

describe('names', () => {
  it('makes unique names with a counter', () => {
    expect(uniqueName('set', ['a'])).toBe('set');
    expect(uniqueName('set', ['set', 'set_2'])).toBe('set_3');
  });

  it('checks set names', () => {
    expect(setNameProblem('', [])).not.toBeNull();
    expect(setNameProblem('snippets', [])).not.toBeNull();
    expect(setNameProblem('a', ['a'])).not.toBeNull();
    expect(setNameProblem('b', ['a'])).toBeNull();
  });

  it('checks snippet names', () => {
    expect(snippetNameProblem('2tone', [])).not.toBeNull();
    expect(snippetNameProblem('a b', [])).not.toBeNull();
    expect(snippetNameProblem('a', ['a'])).not.toBeNull();
    expect(snippetNameProblem('_ok2', [])).toBeNull();
  });
});

describe('renaming references', () => {
  it('renames whole references only', () => {
    expect(renameRefs('$a and $ab and $a.', 'a', 'b')).toBe('$b and $ab and $b.');
  });

  it('renames in fields but never in structured values', () => {
    expect(renameRefsInValue({ kind: 'fields', fields: [{ key: 'K', value: '$a' }] }, 'a', 'b')).toEqual({ kind: 'fields', fields: [{ key: 'K', value: '$b' }] });
    const structured = { kind: 'structured' as const, data: ['$a'] };
    expect(renameRefsInValue(structured, 'a', 'b')).toBe(structured);
  });

  it('finds references in structured values', () => {
    const document: PromptDocument = {
      snippets: [],
      sets: [{ id: 's0', name: 's', entries: [{ id: 'e', prompt: { kind: 'structured', data: { A: ['$light'] } }, scene: null, negative: null, active: true, enhance: null }] }],
    };
    expect(structuredUses(document, 'light')).toBe(true);
    expect(structuredUses(document, 'lig')).toBe(false);
  });

  it('finds references in a structured scene', () => {
    const document: PromptDocument = {
      snippets: [],
      sets: [{ id: 's0', name: 's', entries: [{ id: 'e', scene: { kind: 'structured', data: ['$castle'] }, prompt: { kind: 'text', text: 'x' }, negative: null, active: true, enhance: null }] }],
    };
    expect(structuredUses(document, 'castle')).toBe(true);
  });

  it('starts a new entry without a scene, and a copy keeps the scene', () => {
    expect(newEntry().scene).toBeNull();
    const copy = copyEntry({ ...newEntry(), scene: { kind: 'text', text: 'A castle.' } });
    expect(copy.scene).toEqual({ kind: 'text', text: 'A castle.' });
  });
});

describe('moveItem and dropPosition', () => {
  it('moves within and between lists', () => {
    const a = ['x', 'y', 'z'];
    moveItem(a, 'x', a, 'z', 'after');
    expect(a).toEqual(['y', 'z', 'x']);
    const b: string[] = [];
    moveItem(a, 'y', b, null, 'after');
    expect([a, b]).toEqual([['z', 'x'], ['y']]);
  });

  it('drops before or after by the pointer half', () => {
    expect(dropPosition({ top: 100, height: 40 }, 110)).toBe('before');
    expect(dropPosition({ top: 100, height: 40 }, 130)).toBe('after');
  });
});

describe('prompt ids', () => {
  const document: PromptDocument = {
    snippets: [],
    sets: [{ id: 's0', name: 'portrait', entries: [
      { id: 'a', prompt: { kind: 'text', text: 'x' }, scene: null, negative: null, active: false, enhance: null },
      { id: 'b', prompt: { kind: 'text', text: 'y' }, scene: null, negative: null, active: true, enhance: null },
    ] }],
  };

  it('gives active entries their set:index id', () => {
    expect(optionId(document, 'b')).toBe('portrait:1');
    expect(optionId(document, 'a')).toBeNull();
  });

  it('remaps a selection and drops ids that are gone', () => {
    expect(remapOptionIds(['p:0', 'p:1', 'q:0'], { 'p:0': 'p:2', 'q:0': 'q:0' })).toEqual(['p:2', 'q:0']);
  });
});

describe('enhance', () => {
  it('fills the picker from an entry mapping, ignoring unknown values', () => {
    expect(enhanceSettingsFor({ style: 'cinematic', details: ['camera', 'nope'], mood: 'nope' }, AXES, DEFAULTS)).toEqual({
      style: 'cinematic', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: ['camera'], length: 'same', motion: ['action'], sound: [], note: '',
    });
    expect(enhanceSettingsFor(true, AXES, DEFAULTS)).toEqual(DEFAULTS);
  });

  it('carries an entry Rewrite choice through, and never writes the null backend default', () => {
    const backend = { ...DEFAULTS, parts: null } as unknown as typeof DEFAULTS;
    expect(enhanceSettingsFor({ parts: 'action' }, AXES, backend)).toMatchObject({ parts: 'action' });
    expect(enhanceSettingsFor({ parts: 'both' }, AXES, backend)).toMatchObject({ parts: 'both' });
    expect('parts' in enhanceSettingsFor({ style: 'cinematic' }, AXES, backend)).toBe(false);
    expect('parts' in enhanceSettingsFor({ parts: 'scene' }, AXES, backend)).toBe(false);
    expect('parts' in enhanceSettingsFor(true, AXES, backend)).toBe(false);
  });

  it('reads an entry direction', () => {
    expect(enhanceSettingsFor({ note: 'at dusk' }, AXES, DEFAULTS).note).toBe('at dusk');
  });

  it('summarizes sound and a direction', () => {
    expect(enhanceSummary({ sound: ['ambient', 'music'], note: 'at dusk' }, AXES)).toBe('2 sound · Direction');
  });

  it('summarizes an entry for its chip', () => {
    expect(enhanceSummary({ parts: 'action', style: 'cinematic' }, AXES)).toBe('Action only · Cinematic');
    expect(enhanceSummary(true, AXES)).toBe('Default options');
    expect(enhanceSummary({ style: 'cinematic', mood: 'eerie', theme: 'keep', camera: 'keep', pace: 'keep', details: ['camera', 'lighting'], length: 'longer' }, AXES)).toBe('Cinematic · Eerie · Longer · +2 details');
    expect(enhanceSummary({ style: 'keep' }, AXES)).toBe('Custom options');
  });

  it('shortens the chip label to two choices and a count', () => {
    expect(enhanceShortLabel({ style: 'cinematic', mood: 'eerie', theme: 'keep', camera: 'keep', pace: 'keep', details: ['camera', 'lighting'], length: 'longer' }, AXES)).toBe('Cinematic · Eerie +2');
    expect(enhanceShortLabel({ style: 'cinematic', mood: 'eerie' }, AXES)).toBe('Cinematic · Eerie');
    expect(enhanceShortLabel(true, AXES)).toBe('Default options');
  });
});
