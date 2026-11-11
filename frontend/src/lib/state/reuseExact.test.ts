import { beforeEach, describe, expect, it } from 'vitest';

import { withAudio } from '../../test-utils/audio';
import { makeContext } from '../../test-utils/workspace';
import { draft } from './draft.svelte';
import { REUSE_SCOPE, RESET_SCOPE, keysOf } from './resetScope';
import { voiceConsent } from './voiceConsent.svelte';
import type { DraftState, EnhanceSettings, WorkspaceContext } from '$lib/types';
import { openKind } from '../../test-utils/draftKind';

const base = makeContext();
const ctx: WorkspaceContext = withAudio({
  ...base,
  loras: [{ name: 'detail', path: '/loras/detail.safetensors' }],
  workflow_contract: { ...base.workflow_contract, values: [...base.workflow_contract.values, 'extend'] },
});

const SETTINGS: EnhanceSettings = { style: 'cinematic', mood: 'keep', theme: 'keep', shot: 'keep', angle: 'keep', palette: 'keep', colors: [], camera: 'keep', pace: 'keep', details: [], length: 'longer', motion: [], sound: [], note: '' };

/** Apply a take's params the way a Reuse and a Load settings do. */
function reuse(params: Record<string, string>): void {
  draft.loadFromUrl(params, ctx, { replaceSlots: true, exact: true });
}

/** A form the person has edited away from every default, on the Image tab. */
function editedImageForm(): void {
  draft.hydrateFromContext(ctx, 'zit');
  draft.patch({
    prompt: 'old prompt', negativePrompt: 'blurry', runs: 3, seed: 9, steps: 5, guidance: 1.5, scheduler: 'custom',
    loraString: 'detail:0.8', quantize: 4, postprocessContrastEnabled: true, postprocessContrastAmount: 1.4,
    upscaleEnabled: true, upscaleFactor: 4, upscaleReview: true, firstSigma: 0.3,
    enhanceSettings: SETTINGS, enhanceSettingsVideo: { ...SETTINGS, style: 'dreamy' }, enhanceAuto: true, enhanceModel: 'smart', enhanceFormat: 'json',
  });
}

beforeEach(() => {
  localStorage.clear();
  draft.reset();
  voiceConsent.reset();
});

describe('REUSE_SCOPE', () => {
  it('holds each tab\'s Reset scopes and adds the text, the model and, on Extend, the source', () => {
    for (const tab of ['image', 'video', 'extend'] as const) {
      expect(keysOf(REUSE_SCOPE[tab])).toEqual(expect.arrayContaining(keysOf(RESET_SCOPE[tab])));
      expect(keysOf(REUSE_SCOPE[tab])).toEqual(expect.arrayContaining(['prompt', 'model', 'loraString', 'quantize']));
    }
    expect(keysOf(REUSE_SCOPE.extend)).toContain('extendSourcePath');
    expect(keysOf(REUSE_SCOPE.image)).not.toContain('extendSourcePath');
  });
});

describe('an exact prefill', () => {
  it('puts back what the take does not name instead of keeping what the form held', () => {
    editedImageForm();

    reuse({ workflow: 'txt2img', model: 'zit', prompt: 'new prompt' });

    expect(draft.state).toMatchObject({
      prompt: 'new prompt', negativePrompt: '', runs: 1, seed: null, steps: 28, guidance: 6.2, scheduler: 'beta', firstSigma: null,
      loraString: '', quantize: null, postprocessContrastEnabled: false, postprocessContrastAmount: 1, upscaleEnabled: false, upscaleReview: false,
      enhanceSettings: null,
    });
  });

  it('sets exactly what the take names', () => {
    editedImageForm();

    reuse({ workflow: 'txt2img', model: 'zit', prompt: 'new prompt', negative_prompt: 'ugly', seed: '7', steps: '12', lora: 'detail:0.5', quantize: '8', contrast_enabled: 'true', sharpen_enabled: 'true' });

    expect(draft.state).toMatchObject({ prompt: 'new prompt', negativePrompt: 'ugly', seed: 7, runs: 1, steps: 12, loraString: 'detail:0.5', quantize: 8 });
  });

  it('keeps the enhancer, its format, the auto choice and the other mode\'s options', () => {
    editedImageForm();

    reuse({ workflow: 'txt2img', model: 'zit', prompt: 'new prompt' });

    expect(draft.state).toMatchObject({ enhanceSettingsVideo: { style: 'dreamy' }, enhanceAuto: true, enhanceModel: 'smart', enhanceFormat: 'json' });
  });

  it('fits a size the model does not offer to the model, and a patch-only prefill keeps the current size instead', () => {
    editedImageForm();
    draft.patch({ ratio: '2:3', size: 'm' });

    reuse({ workflow: 'txt2img', model: 'zit', prompt: 'p', ratio: '16:9', size: 'm' });
    expect(draft.state).toMatchObject({ ratio: '16:9', size: 'l', dimensionMode: 'ratio' });

    draft.patch({ ratio: '2:3', size: 'm' });
    draft.loadFromUrl({ workflow: 'txt2img', model: 'zit', ratio: '16:9', size: 'm' }, ctx, { replaceSlots: true });
    expect(draft.state).toMatchObject({ ratio: '2:3', size: 'm' });
  });

  it('leaves the other tabs and the other mode alone', () => {
    editedImageForm();
    draft.patch({
      audioStyle: 'calm', speechText: 'say it', voiceMode: 'recording', voicePath: '/voices/me.wav', extendSourcePath: '/clips/a.mp4', firstFramePath: '/in/first.png',
      savedText: { video: { prompt: 'an action', scene: 'a scene', negativePrompt: '', jsonPromptEnabled: false, jsonPrompt: '', enhancedPrompt: '', enhancedScene: '', enhancedFrom: null, enhancedHistory: [], enhancedScratched: null } },
    });
    const before = { ...draft.state };

    reuse({ workflow: 'txt2img', model: 'zit', prompt: 'new prompt' });

    for (const key of ['audioStyle', 'speechText', 'voiceMode', 'voicePath', 'extendSourcePath', 'firstFramePath', 'savedText', 'promptFilePath'] as (keyof DraftState)[]) {
      expect(draft.state[key], key).toStrictEqual(before[key]);
    }
  });

  it('is patch-only for a Use as, a link and a prefill that is not marked exact', () => {
    editedImageForm();

    draft.loadFromUrl({ workflow: 'img2img', model: 'zit', image_path: '/in/a.png', use_as: 'start' }, ctx, { replaceSlots: true, exact: true });
    expect(draft.state).toMatchObject({ runs: 3, seed: 9, loraString: 'detail:0.8', quantize: 4, negativePrompt: 'blurry' });

    draft.loadFromUrl({ workflow: 'txt2img', model: 'zit', prompt: 'from a link' }, ctx, { replaceSlots: true });
    expect(draft.state).toMatchObject({ prompt: 'from a link', runs: 3, seed: 9, loraString: 'detail:0.8', quantize: 4, negativePrompt: 'blurry' });
  });

  it('puts an Extend take back to its source, length and track, and keeps New video\'s own text and settings', () => {
    draft.hydrateFromContext(ctx, null);
    draft.setTab('video');
    draft.onTabChange(ctx);
    draft.setAudioTrack('video', { path: '/audio/video.mp3' });
    draft.patch({ prompt: 'new video prompt', runs: 2 });
    openKind('video', 'extend', ctx);
    draft.onTabChange(ctx);
    draft.patch({ extendSourcePath: '/clips/old.mp4', extendAudioTrackPath: '/audio/old.mp3', extendFrames: 16, loraString: 'detail:0.4', runs: 4 });

    reuse({ workflow: 'extend', model: 'ltx-8', source_video: '/clips/new.mp4', extend_frames: '48', scene: 's', prompt: 'p' });

    expect(draft.state).toMatchObject({
      workflow: 'extend', extendSourcePath: '/clips/new.mp4', extendFrames: 48, extendAudioTrackPath: null, loraString: '', runs: 1, audioTrackPath: '/audio/video.mp3',
    });
    openKind('video', 'new', ctx);
    expect(draft.state).toMatchObject({ workflow: 'txt2vid', prompt: 'new video prompt', runs: 2 });
  });
});

describe('an exact prefill on the Audio tab', () => {
  it('puts Batch and Seed back, sets the kind\'s fields and leaves the voice and its tick of another kind', () => {
    draft.hydrateFromContext(ctx, null);
    draft.setTab('audio');
    draft.onTabChange(ctx);
    draft.setAudioKind('speech', ctx);
    draft.setVoice({ path: '/voices/me.wav' });
    voiceConsent.ticked = true;
    draft.patch({ runs: 3, seed: 5, audioStyle: 'old style' });

    reuse({ workflow: 'music', model: 'yue2', prompt: 'calm piano' });

    expect(draft.state).toMatchObject({ workflow: 'music', audioStyle: 'calm piano', runs: 1, seed: null, voicePath: '/voices/me.wav' });
    expect(voiceConsent.ticked).toBe(true);
  });

  it('sets the settings of the take, and puts Auto and the defaults back for any it does not name', () => {
    draft.hydrateFromContext(ctx, null);
    draft.setTab('audio');
    draft.onTabChange(ctx);
    draft.patch({ audioTempo: 150, audioKey: 'Bb', soundType: 'ambience', speechDelivery: { emotion: 'angry', manner: 'shout', pace: 'fast', pitch: 'high', expressiveness: 'low' } });

    reuse({ workflow: 'music', model: 'yue2', prompt: 'calm piano', tempo: '92', key: 'F#m' });
    expect(draft.state).toMatchObject({ audioTempo: 92, audioKey: 'F#m' });

    reuse({ workflow: 'music', model: 'yue2', prompt: 'calm piano' });
    expect(draft.state).toMatchObject({ audioTempo: null, audioKey: null });
  });

  it('leaves the settings of the other kinds as the form has them', () => {
    draft.hydrateFromContext(ctx, null);
    draft.setTab('audio');
    draft.onTabChange(ctx);
    const delivery = { emotion: 'angry', manner: 'shout', pace: 'fast', pitch: 'high', expressiveness: 'low' } as const;
    draft.patch({ audioTempo: 150, audioKey: 'Bb', audioSeconds: 7, soundType: 'effect', speechDelivery: delivery });

    reuse({ workflow: 'speech', model: 'qwen3-tts', prompt: 'hello there' });
    expect(draft.state).toMatchObject({ audioTempo: 150, audioKey: 'Bb', audioSeconds: 7, soundType: 'effect', speechDelivery: { emotion: 'neutral', manner: 'natural' } });

    reuse({ workflow: 'music', model: 'yue2', prompt: 'calm piano' });
    expect(draft.state).toMatchObject({ audioTempo: null, audioKey: null, audioSeconds: 7, soundType: 'effect' });
  });
});

describe('the scopes of the Audio tab', () => {
  it('reset and reuse put back the settings and the delivery', () => {
    for (const scopes of [RESET_SCOPE.audio, REUSE_SCOPE.audio]) {
      expect(keysOf(scopes)).toEqual(expect.arrayContaining(['audioSeconds', 'audioTempo', 'audioKey', 'musicSeconds', 'soundType', 'speechDelivery']));
    }
  });

  it('keep the text and the model of a Reuse out of the reset, as the kinds keep their own', () => {
    expect(keysOf(REUSE_SCOPE.audio)).not.toEqual(expect.arrayContaining(['audioStyle']));
    expect(keysOf(REUSE_SCOPE.audio)).not.toContain('voicePath');
  });
});
