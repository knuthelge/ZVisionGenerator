import { describe, expect, it } from 'vitest';
import type { DraftState, LoraInfo } from '$lib/types';
import { draft } from '$lib/state/draft.svelte';
import { insertWordPatch, triggerMenu, triggerModes, type TriggerModes } from './triggerMenu';

const base = (overrides: Partial<DraftState> = {}): DraftState => ({ ...draft.state, workflow: 'txt2img', promptSource: 'inline', prompt: '', scene: '', jsonPromptEnabled: false, jsonPrompt: '', enhancedPrompt: '', enhancedScene: '', enhanceAuto: false, ...overrides });
const modes = (overrides: Partial<TriggerModes> = {}): TriggerModes => ({ pair: false, json: false, file: false, ...overrides });
const lora = (name: string, trigger: string | null, title: string | null = null): LoraInfo => ({ name, path: `/l/${name}.safetensors`, trigger, title });
const chip = (name: string) => ({ name, weight: 1 });
const menu = (state: DraftState, m: TriggerModes, chips: string[], installed: LoraInfo[], pending: Record<string, string> = {}) =>
  triggerMenu({ state, modes: m, chips: chips.map(chip), installed, pending });

describe('triggerMenu', () => {
  const installed = [lora('clay', 'Clay animation, Clay', 'Clay Animation (Redmond)'), lora('tech', 't3chnic4lly'), lora('plain', null)];

  it('groups the words of each LoRA in use under its title, in the order of the chips', () => {
    const result = menu(base({ prompt: 'a lighthouse' }), modes(), ['tech', 'clay'], installed);

    expect(result?.groups.map((group) => [group.title, group.words.map((entry) => entry.word)])).toEqual([
      ['tech', ['t3chnic4lly']],
      ['Clay Animation (Redmond)', ['Clay animation', 'Clay']],
    ]);
  });

  it('marks a word the text already holds, in any case and as a whole word only', () => {
    const result = menu(base({ prompt: 'CLAY animation, a clayey fox' }), modes(), ['clay'], installed);

    expect(result?.groups[0].words).toEqual([{ word: 'Clay animation', inPrompt: true }, { word: 'Clay', inPrompt: true }]);
    expect(menu(base({ prompt: 'a clayey fox' }), modes(), ['clay'], installed)?.groups[0].words.map((entry) => entry.inPrompt)).toEqual([false, false]);
  });

  it('counts the words the text lacks, once however many LoRAs name them', () => {
    const shared = [lora('a', 'ohwx, one'), lora('b', 'OHWX, two')];

    expect(menu(base(), modes(), ['a', 'b'], shared)?.missing).toBe(3);
    expect(menu(base({ prompt: 'ohwx' }), modes(), ['a', 'b'], shared)?.missing).toBe(2);
    expect(menu(base({ prompt: 'ohwx, one, two' }), modes(), ['a', 'b'], shared)?.missing).toBe(0);
  });

  it('is not offered when no LoRA in use has a trigger', () => {
    expect(menu(base(), modes(), ['plain'], installed)).toBeNull();
    expect(menu(base(), modes(), [], installed)).toBeNull();
  });

  it('knows the trigger of a LoRA that is still downloading', () => {
    expect(menu(base(), modes(), ['new'], installed, { new: 'fresh, new' })?.groups[0]).toMatchObject({ name: 'new', words: [{ word: 'fresh' }, { word: 'new' }] });
  });

  it('reads the text Generate sends: the Enhanced text when that is used', () => {
    const state = base({ prompt: 'mine t3chnic4lly', enhancedPrompt: 'a lighthouse' });

    expect(menu(state, modes(), ['tech'], installed)?.missing).toBe(1);
  });

  it('looks in the scene as well as the action for a video', () => {
    const state = base({ workflow: 'txt2vid', scene: 'Clay animation scene', prompt: 'he walks' });

    expect(menu(state, modes({ pair: true }), ['clay'], installed)?.groups[0].words[0].inPrompt).toBe(true);
  });

  it('is hidden for a JSON caption and for a prompt from a file', () => {
    expect(menu(base({ jsonPromptEnabled: true, jsonPrompt: '{}' }), modes({ json: true }), ['tech'], installed)).toBeNull();
    expect(menu(base(), modes({ file: true }), ['tech'], installed)).toBeNull();
  });
});

describe('insertWordPatch', () => {
  it('adds at the start of the prompt, or of the scene for a video', () => {
    expect(insertWordPatch(base({ prompt: 'a fox' }), modes(), 'ohwx')).toEqual({ prompt: 'ohwx, a fox' });
    expect(insertWordPatch(base({ scene: 'a stair', prompt: 'he walks' }), modes({ pair: true }), 'ohwx')).toEqual({ scene: 'ohwx, a stair' });
  });

  it('writes into the Enhanced text when Generate sends that', () => {
    expect(insertWordPatch(base({ prompt: 'mine', enhancedPrompt: 'rewritten' }), modes(), 'ohwx')).toEqual({ enhancedPrompt: 'ohwx, rewritten' });
  });
});

describe('triggerModes', () => {
  it('reads the layout from the capabilities and the prompt source', () => {
    const caps = { showScene: true, showPromptInline: true, supportsJsonPrompt: false };
    expect(triggerModes(caps, { promptSource: 'inline', jsonPromptEnabled: false })).toEqual({ pair: true, json: false, file: false });
    expect(triggerModes(caps, { promptSource: 'file', jsonPromptEnabled: false })).toEqual({ pair: false, json: false, file: true });
    expect(triggerModes({ ...caps, showScene: false, supportsJsonPrompt: true }, { promptSource: 'inline', jsonPromptEnabled: true }).json).toBe(true);
  });
});
