import { describe, it, expect } from 'vitest';
import { effectiveEnhanceSettings, settingsPayload } from './promptEnhance';
import type { DraftState, PromptEnhancerContract } from '$lib/types';

/** The server's matrix contract: `defaults.parts` is null (auto), the per-workflow default is the client's job. */
const CONTRACT = {
  matrix: {
    axes: [],
    defaults: { style: 'keep', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: ['lighting', 'composition'], length: 'same', motion: ['action'], sound: [], note: '', parts: null },
    parts: { key: 'parts', label: 'Rewrite', video_only: true, options: [{ slug: 'both', label: 'Scene and action', description: '', group: null }, { slug: 'action', label: 'Action only', description: '', group: null }], default: null },
  },
} as unknown as PromptEnhancerContract;

function state(overrides: Partial<DraftState> = {}): DraftState {
  return { workflow: 'txt2vid', promptSource: 'inline', prompt: 'a fox', scene: 'A forest.', enhanceSettings: null, ...overrides } as DraftState;
}

describe('the server matrix contract with defaults.parts = null', () => {
  it.each([
    ['txt2vid', 'both'],
    ['img2vid', 'action'],
    ['extend', 'action'],
  ] as const)('does not override the %s default (%s)', (workflow, parts) => {
    expect(effectiveEnhanceSettings(state({ workflow }), CONTRACT).parts).toBe(parts);
  });

  it('leaves parts out of image settings', () => {
    expect('parts' in effectiveEnhanceSettings(state({ workflow: 'txt2img' }), CONTRACT)).toBe(false);
  });

  it('never sends a null parts or the browser-only memory to the server', () => {
    const remembered = effectiveEnhanceSettings(state({ workflow: 'extend', enhanceSettings: { style: 'keep', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: [], length: 'same', motion: [], sound: [], note: '', partsByWorkflow: { extend: 'both' } } }), CONTRACT);
    expect(remembered.parts).toBe('both');
    const payload = settingsPayload(remembered, 'video', true);
    expect(payload.parts).toBe('both');
    expect(Object.keys(payload)).not.toContain('partsByWorkflow');
    expect(JSON.stringify(settingsPayload(effectiveEnhanceSettings(state({ workflow: 'txt2img' }), CONTRACT), 'image'))).not.toMatch(/parts/);
  });
});
