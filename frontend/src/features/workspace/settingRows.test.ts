import { beforeEach, describe, expect, it } from 'vitest';

import { makeContext } from '../../test-utils/workspace';
import { defaultsFor, draft } from '$lib/state/draft.svelte';
import { sourceInfo } from '$lib/state/extend.fixtures';
import { presetDraftKeys } from '$lib/state/presets';
import { keysOf } from '$lib/state/resetScope';
import type { DraftState, WorkspaceContext } from '$lib/types';
import { workspaceCapabilities } from './capabilities';
import { changedRowCount, chipChanged, chipKeys, rowChanged, SETTING_ROWS } from './settingRows';

const base = makeContext();
const ctx: WorkspaceContext = { ...base, workflow_contract: { ...base.workflow_contract, values: [...base.workflow_contract.values, 'extend'] } };

beforeEach(() => {
  localStorage.clear();
  draft.reset();
});

describe('SETTING_ROWS', () => {
  it('names every row once', () => {
    const ids = SETTING_ROWS.map((row) => row.id);

    expect(new Set(ids).size).toBe(ids.length);
  });

  it('covers the draft keys of the setting scopes, so a new setting needs a row', () => {
    // The custom size's width and height are compared through the size row's dimension mode.
    const exempt = new Set<keyof DraftState>(['width', 'height']);
    const scoped = new Set<keyof DraftState>([...keysOf(['settings', 'shared', 'extendSettings', 'audioSettings']), 'referenceImageStrength']);
    const rowKeys = new Set<keyof DraftState>(SETTING_ROWS.flatMap((row) => row.keys));

    expect([...scoped].filter((key) => !rowKeys.has(key) && !exempt.has(key))).toEqual([]);
    expect([...rowKeys].filter((key) => !scoped.has(key))).toEqual([]);
  });
});

describe('presetDraftKeys', () => {
  it('stays within the setting scopes, the model and what is remembered for the model', () => {
    const allowed = new Set<keyof DraftState>([...keysOf(['settings', 'shared', 'model', 'memory']), 'referenceImageStrength']);

    for (const workflow of ['txt2img', 'txt2vid'] as const) {
      expect([...presetDraftKeys({ workflow })].filter((key) => !allowed.has(key))).toEqual([]);
    }
  });
});

describe('the Extend length (C3)', () => {
  function onExtend(extendFrames: number): { state: DraftState; defaults: DraftState } {
    draft.hydrateFromContext(ctx, null);
    draft.setTab('video');
    draft.setVideoKind('extend', ctx);
    draft.onTabChange(ctx);
    draft.applyExtendSource(sourceInfo(), ctx);
    draft.update('extendFrames', extendFrames);
    const state = draft.state;
    return { state, defaults: defaultsFor(ctx, state, draft.extendSource) };
  }

  it('counts as changed on the Settings pane, the chip and the count alike once it leaves the source\'s default', () => {
    const { state, defaults } = onExtend(16);
    const caps = workspaceCapabilities(ctx, state);

    expect(rowChanged('extendLength', state, defaults)).toBe(true);
    expect(chipChanged('extendSeconds', caps, state, defaults)).toBe(true);
    expect(changedRowCount(caps, state, defaults)).toBe(1);
    expect(chipKeys('extendSeconds', caps, state)).toEqual(['extendFrames']);
  });

  it('is unchanged at the default length', () => {
    const { state, defaults } = onExtend(48);
    const caps = workspaceCapabilities(ctx, state);

    expect(chipChanged('extendSeconds', caps, state, defaults)).toBe(false);
    expect(changedRowCount(caps, state, defaults)).toBe(0);
  });
});

describe('the popovers\' Reset', () => {
  it('puts back every setting the chip\'s rows hold, the video upscale factor included', () => {
    draft.hydrateFromContext(ctx, null);
    draft.setTab('video');
    draft.onTabChange(ctx);
    const caps = workspaceCapabilities(ctx, draft.state);

    expect(chipKeys('audio', caps, draft.state)).toEqual(expect.arrayContaining(['audio', 'lowMemory', 'videoUpscaleEnabled', 'videoUpscaleFactor']));
  });
});
