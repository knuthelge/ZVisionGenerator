import { beforeEach, describe, expect, it } from 'vitest';

import { AUDIO_MODELS, SA3_MUSIC, withAudio } from '../../test-utils/audio';
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
    const scoped = new Set<keyof DraftState>([...keysOf(['settings', 'shared', 'extendSettings', 'audioSettings', 'audioDelivery', 'voice']), 'referenceImageStrength']);
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

describe('the Audio voice', () => {
  const audioCtx = withAudio(ctx);

  function onSpeech(): { state: DraftState; defaults: DraftState } {
    draft.hydrateFromContext(audioCtx, null);
    draft.setTab('audio');
    draft.onTabChange(audioCtx);
    draft.setAudioKind('speech', audioCtx);
    return { state: draft.state, defaults: defaultsFor(audioCtx, draft.state, null) };
  }

  it('counts in the Reset badge and the phone count once a recording or a transcript is set, and not before', () => {
    const { state, defaults } = onSpeech();
    const caps = workspaceCapabilities(audioCtx, state);
    expect(changedRowCount(caps, state, defaults)).toBe(0);
    expect(chipChanged('voice', caps, state, defaults)).toBe(false);

    draft.setVoice({ path: '/voices/me.wav' });
    expect(changedRowCount(caps, draft.state, defaults)).toBe(1);
    expect(chipChanged('voice', caps, draft.state, defaults)).toBe(true);
    expect(chipKeys('voice', caps, draft.state)).toEqual(['voiceMode', 'speechSpeaker', 'voicePath', 'voiceUpload', 'voiceText']);
  });

  it('counts a built-in voice other than the default as a change, and the default as none', () => {
    const { state, defaults } = onSpeech();
    const caps = workspaceCapabilities(audioCtx, state);

    draft.update('speechSpeaker', 'vivian');
    expect(chipChanged('voice', caps, draft.state, defaults)).toBe(true);
    expect(changedRowCount(caps, draft.state, defaults)).toBe(1);

    draft.update('speechSpeaker', null);
    expect(chipChanged('voice', caps, draft.state, defaults)).toBe(false);
  });
});

describe('the Music settings', () => {
  const musicCtx = withAudio(ctx, [...AUDIO_MODELS, SA3_MUSIC]);

  function onMusic(model: string): { state: DraftState; defaults: DraftState } {
    draft.hydrateFromContext(musicCtx, model);
    draft.setTab('audio');
    draft.onTabChange(musicCtx);
    draft.setAudioKind('music', musicCtx);
    draft.hydrateFromContext(musicCtx, model);
    return { state: draft.state, defaults: defaultsFor(musicCtx, draft.state, null) };
  }

  it('shows Tempo and Key for YuE2, and Length too for Stable Audio, each on its own phone chip', () => {
    const yue = onMusic('yue2');
    const yueCaps = workspaceCapabilities(musicCtx, yue.state);
    expect(chipKeys('tempo', yueCaps, yue.state)).toEqual(['audioTempo']);
    expect(chipKeys('key', yueCaps, yue.state)).toEqual(['audioKey']);
    expect(chipKeys('length', yueCaps, yue.state)).toEqual([]);

    const sa3 = onMusic('sa3');
    const sa3Caps = workspaceCapabilities(musicCtx, sa3.state);
    expect(chipKeys('length', sa3Caps, sa3.state)).toEqual(['musicSeconds']);
  });

  it('counts a tempo, a key and a song\'s length as changes, and Auto and the default length as none', () => {
    const { state, defaults } = onMusic('sa3');
    const caps = workspaceCapabilities(musicCtx, state);
    expect(changedRowCount(caps, state, defaults)).toBe(0);

    draft.patch({ audioTempo: 92, audioKey: 'Am', musicSeconds: 60 });
    expect(changedRowCount(caps, draft.state, defaults)).toBe(3);
    expect(chipChanged('tempo', caps, draft.state, defaults)).toBe(true);
    expect(chipChanged('key', caps, draft.state, defaults)).toBe(true);
    expect(chipChanged('length', caps, draft.state, defaults)).toBe(true);

    draft.patch({ audioTempo: null, audioKey: null, musicSeconds: 0 });
    expect(changedRowCount(caps, draft.state, defaults)).toBe(0);
  });
});

describe('the Sound settings', () => {
  const soundCtx = withAudio(ctx, AUDIO_MODELS);

  function onSound(): { state: DraftState; defaults: DraftState } {
    draft.hydrateFromContext(soundCtx);
    draft.setTab('audio');
    draft.onTabChange(soundCtx);
    draft.setAudioKind('sound', soundCtx);
    return { state: draft.state, defaults: defaultsFor(soundCtx, draft.state, null) };
  }

  it('gives the Type and the Length a phone chip each', () => {
    const { state } = onSound();
    const caps = workspaceCapabilities(soundCtx, state);

    expect(chipKeys('soundType', caps, state)).toEqual(['soundType']);
    expect(chipKeys('length', caps, state)).toEqual(['audioSeconds']);
  });

  it('counts a type other than Auto and a locked length as changes, and Auto as none', () => {
    const { state, defaults } = onSound();
    const caps = workspaceCapabilities(soundCtx, state);
    expect(changedRowCount(caps, state, defaults)).toBe(0);

    draft.patch({ soundType: 'effect', audioSeconds: 12 });
    expect(changedRowCount(caps, draft.state, defaults)).toBe(2);
    expect(chipChanged('soundType', caps, draft.state, defaults)).toBe(true);
    expect(chipChanged('length', caps, draft.state, defaults)).toBe(true);

    draft.patch({ soundType: 'auto', audioSeconds: 0 });
    expect(changedRowCount(caps, draft.state, defaults)).toBe(0);
  });
});

describe('the Delivery row', () => {
  const speech = { ...withAudio(makeContext(), AUDIO_MODELS) };

  function onSpeech(delivery: Partial<DraftState['speechDelivery']> = {}, voiceMode: DraftState['voiceMode'] = 'builtin'): { state: DraftState; defaults: DraftState } {
    draft.hydrateFromContext(speech, null);
    draft.patch({ workflow: 'speech', voiceMode, speechDelivery: { ...draft.state.speechDelivery, ...delivery } });
    const state = draft.state;
    return { state, defaults: defaultsFor(speech, state, draft.extendSource) };
  }

  it('is changed by what the five values say, so a pick that lands on the defaults is not a change', () => {
    const { state, defaults } = onSpeech();
    expect(rowChanged('delivery', state, defaults)).toBe(false);

    const same = { ...state, speechDelivery: { ...state.speechDelivery } };
    expect(rowChanged('delivery', same, defaults)).toBe(false);
    expect(rowChanged('delivery', { ...state, speechDelivery: { ...state.speechDelivery, pace: 'fast' } }, defaults)).toBe(true);
  });

  it('belongs to the Delivery chip, which a recording hides and All then leaves out', () => {
    const { state, defaults } = onSpeech({ emotion: 'sad' });
    const caps = workspaceCapabilities(speech, state);

    expect(chipKeys('delivery', caps, state)).toEqual(['speechDelivery']);
    expect(chipChanged('delivery', caps, state, defaults)).toBe(true);
    expect(changedRowCount(caps, state, defaults)).toBe(1);

    const recorded = onSpeech({ emotion: 'sad' }, 'recording');
    const recordedCaps = workspaceCapabilities(speech, recorded.state);
    expect(chipKeys('delivery', recordedCaps, recorded.state)).toEqual([]);
    expect(chipChanged('delivery', recordedCaps, recorded.state, recorded.defaults)).toBe(false);
    expect(changedRowCount(recordedCaps, recorded.state, recorded.defaults)).toBe(1); // the voice alone
  });
});
