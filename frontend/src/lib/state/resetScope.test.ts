import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { withAudio } from '../../test-utils/audio';
import { makeContext } from '../../test-utils/workspace';
import { defaultsFor, draft } from './draft.svelte';
import { sourceInfo } from './extend.fixtures';
import { downloadStore } from './downloads.svelte';
import { generateBlock } from './generateBlock.svelte';
import { heldUploads } from './heldUploads.svelte';
import { FIELD_SCOPE, keysOf, RESET_SCOPE } from './resetScope';
import { voiceConsent } from './voiceConsent.svelte';
import { isAudioWorkflow, navTab, videoKindOf } from './workspaceTab';
import type { AudioWorkflow, DownloadItem, DraftState, EnhanceSettings, WorkspaceContext, WorkspaceTab, Workflow } from '$lib/types';

const base = makeContext();
const ctx: WorkspaceContext = withAudio({
  ...base,
  loras: [{ name: 'detail', path: '/loras/detail.safetensors' }],
  workflow_contract: { ...base.workflow_contract, values: [...base.workflow_contract.values, 'extend'] },
});

const file = (name: string): File => new File(['x'], name);

/** Open the tab a workflow belongs to the way the top bar and the Workspace do: pick it, then re-hydrate for it. */
function openTab(workflow: Workflow | AudioWorkflow): void {
  const kind = videoKindOf(workflow);
  if (kind === null) draft.setTab(navTab(workflow));
  else {
    draft.setTab('video');
    draft.setVideoKind(kind, ctx);
  }
  draft.onTabChange(ctx);
  if (isAudioWorkflow(workflow)) draft.setAudioKind(workflow, ctx);
}

const SETTINGS: EnhanceSettings = { style: 'cinematic', mood: 'keep', theme: 'keep', shot: 'keep', angle: 'keep', palette: 'keep', colors: [], camera: 'keep', pace: 'keep', details: [], length: 'longer', motion: [], sound: [], note: '' };

/** A distinctive value for every field of the draft; the compiler fails when a field is missing. */
const MARKED: DraftState = {
  workflow: 'img2img', promptSource: 'file', prompt: 'a prompt', scene: 'a scene', jsonPromptEnabled: true, jsonPrompt: '{"a":1}', firstSigma: 0.4, negativePrompt: 'blurry',
  promptFilePath: '/prompts/set.yaml', promptFileOptionIds: ['one', 'two'], promptFileLoaded: { path: '/prompts/set.yaml', count: 2 },
  model: 'zit', ratio: '16:9', dimensionMode: 'custom', size: 'l', steps: 17, guidance: 2.5, width: 1000, height: 600, runs: 4, seed: 42,
  loraString: 'detail:0.8', lorasByFamily: { 'model:ltx-8': 'detail:0.5' }, lastImageModel: 'zit', lastVideoModel: 'ltx-8', lastImageQuantize: 4,
  referenceImagePath: '/refs/ref.png', referenceImageUpload: 'start.png', firstFramePath: '/refs/first.png', firstFrameUpload: 'first.png',
  referenceImagePaths: ['/refs/a.png', '/refs/b.png'], missingReferencePaths: ['/refs/b.png'], endImagePath: '/refs/end.png', endImageUpload: 'end.png',
  audioTrackPath: '/audio/song.mp3', audioTrackUpload: 'memo.m4a', audioTrackStart: 90,
  extendAudioTrackPath: '/audio/other.mp3', extendAudioTrackUpload: 'extend.m4a', extendAudioTrackStart: 12,
  referenceImageStrength: 0.9, frameCount: 33, fps: 30, audio: false, lowMemory: false, upscaleEnabled: true, upscaleFactor: 4, quantize: 4,
  historyCollapsed: true, sidebarCollapsed: true, lastGeneratedAt: '2026-01-01T00:00:00Z', version: 3, scheduler: 'beta',
  postprocessSharpenEnabled: false, postprocessSharpenAmount: 0.6, postprocessContrastEnabled: true, postprocessContrastAmount: 1.3,
  postprocessSaturationEnabled: true, postprocessSaturationAmount: 1.2, postprocessTransparent: true,
  upscaleDenoise: 0.3, upscaleSteps: 12, upscaleGuidance: 1.5, upscaleSharpen: false, upscaleReview: true,
  videoUpscaleEnabled: true, videoUpscaleFactor: 4, videoUpscaleReview: true,
  enhancedPrompt: 'enhanced prompt', enhancedScene: 'enhanced scene', enhancedFrom: { prompt: 'a prompt', scene: 'a scene', mode: 'image' },
  enhancedHistory: [{ prompt: 'plain rewrite', scene: '', from: { prompt: 'a prompt', scene: 'a scene', mode: 'image' } }],
  savedText: { video: { prompt: 'an action', scene: 'a video scene', negativePrompt: '', jsonPromptEnabled: false, jsonPrompt: '', enhancedPrompt: '', enhancedScene: '', enhancedFrom: null, enhancedHistory: [] } },
  enhanceSettings: SETTINGS, enhanceSettingsVideo: { ...SETTINGS, style: 'dreamy', partsByWorkflow: {} }, enhanceAuto: true, enhanceModel: 'smart', enhanceFormat: 'json',
  extendSourcePath: '/clips/source.mp4', extendFrames: 16, missingSourceVideo: true, extendPrefill: { path: '/clips/source.mp4', scene: 's', prompt: 'p' },
  appliedPreset: 'look', videoKind: 'new', audioKind: 'speech', audioModels: { music: 'yue2', speech: 'higgs3', sound: 'sa3' },
  audioStyle: 'a style', lyrics: '[verse]\nla', speechText: 'some text', soundPrompt: 'rain', audioSeconds: 30,
  voiceMode: 'recording', voicePath: '/voices/me.wav', voiceUpload: 'me.wav', voiceText: 'hello',
};

/** What each tab's draft looks like when that tab is open: its workflow and the model of its mode. */
const OPEN: Record<WorkspaceTab, Partial<DraftState>> = {
  image: {},
  video: { workflow: 'img2vid', model: 'ltx-8', quantize: null, videoKind: 'new' },
  extend: { workflow: 'extend', model: 'ltx-8', quantize: null, videoKind: 'extend' },
  audio: { workflow: 'speech', model: 'higgs3' },
};

const TEXT_WORKFLOW: Record<WorkspaceTab, string> = { image: 'txt2img', video: 'txt2vid', extend: 'extend', audio: 'speech' };

const AUDIO_TAB_KEYS = ['audioKind', 'audioModels', 'audioStyle', 'lyrics', 'speechText', 'soundPrompt', 'audioSeconds', 'voiceMode', 'voicePath', 'voiceUpload', 'voiceText'] as const;

beforeEach(() => {
  localStorage.clear();
  draft.reset();
  voiceConsent.reset();
});

afterEach(() => vi.restoreAllMocks());

describe('FIELD_SCOPE', () => {
  it('has a scope for every field the live draft holds, and for nothing else', () => {
    draft.hydrateFromContext(ctx, null);

    expect(Object.keys(FIELD_SCOPE).sort()).toEqual(Object.keys(draft.state).sort());
  });

  it('lists only scopes a tab can reset or a transition keeps', () => {
    for (const tab of Object.keys(RESET_SCOPE) as WorkspaceTab[]) {
      expect(keysOf(RESET_SCOPE[tab]).length).toBeGreaterThan(0);
      // Batch and Seed are shared by every tab, so each Reset puts them back.
      expect(keysOf(RESET_SCOPE[tab])).toEqual(expect.arrayContaining(['runs', 'seed']));
    }
  });
});

describe('Reset on each tab', () => {
  /** Hold a file for each tab and tick the speaker's consent, so a Reset shows what it forgets. */
  function holdUploads(): void {
    heldUploads.setStart('image', file('start.png'));
    heldUploads.setStart('video', file('first.png'));
    heldUploads.setTrack('extend', file('extend.m4a'));
    heldUploads.setVoice(file('me.wav'));
    voiceConsent.ticked = true;
  }

  const held = () => ({ image: heldUploads.startImage, video: heldUploads.firstFrame, extend: heldUploads.extendAudioTrack, audio: heldUploads.voice });

  it.each(['image', 'video', 'extend', 'audio'] as WorkspaceTab[])('on %s changes its own scope to the defaults and leaves every other field as it was', (tab) => {
    draft.hydrateFromContext(ctx, null);
    draft.applyExtendSource(sourceInfo(), ctx);
    draft.patch({ ...MARKED, ...OPEN[tab] });
    holdUploads();
    const before = { ...draft.state };
    const expected = defaultsFor(ctx, before, draft.extendSource);

    draft.resetSelections(ctx);

    const own = new Set(keysOf(RESET_SCOPE[tab]));
    for (const key of Object.keys(FIELD_SCOPE) as (keyof DraftState)[]) {
      if (own.has(key)) expect(draft.state[key], `${key} takes its default`).toStrictEqual(expected[key]);
      else if (FIELD_SCOPE[key] !== 'derived') expect(draft.state[key], `${key} is left alone`).toStrictEqual(before[key]);
    }
    expect(draft.state.workflow).toBe(TEXT_WORKFLOW[tab]);
    // Only the open tab forgets the files it held.
    const owner: Record<WorkspaceTab, string | null> = { image: 'image', video: 'video', extend: 'extend', audio: 'audio' };
    for (const [name, value] of Object.entries(held())) expect(value === null, `${name} upload`).toBe(name === owner[tab]);
    expect(voiceConsent.ticked).toBe(tab !== 'audio');
  });

  it('puts the image model\'s own defaults on the Image tab', () => {
    draft.hydrateFromContext(ctx, null);
    draft.patch({ steps: 3, guidance: 9, referenceImageStrength: 0.9, runs: 5, seed: 7 });

    draft.resetSelections(ctx);

    expect(draft.state).toMatchObject({ steps: 28, guidance: 6.2, referenceImageStrength: 0.5, runs: 1, seed: null });
  });

  it('keeps the model, LoRAs, quantization, prompt, prompt file, extend source and layout', () => {
    draft.hydrateFromContext(ctx, null);
    draft.patch({ ...MARKED, ...OPEN.extend });
    const before = { ...draft.state };

    draft.resetSelections(ctx);

    for (const key of ['model', 'loraString', 'quantize', 'prompt', 'scene', 'enhancedPrompt', 'promptFilePath', 'promptSource', 'extendSourcePath', 'extendPrefill', 'sidebarCollapsed', 'historyCollapsed'] as const) {
      expect(draft.state[key], key).toStrictEqual(before[key]);
    }
  });
});

describe('what a Reset leaves alone (M1, P2, P3, P4)', () => {
  it('M1: an Image or Video Reset leaves the whole Audio tab and the held voice', () => {
    draft.hydrateFromContext(ctx, null);
    draft.patch({ ...MARKED });
    heldUploads.setVoice(file('me.wav'));
    voiceConsent.ticked = true;
    const audio = Object.fromEntries(AUDIO_TAB_KEYS.map((key) => [key, draft.state[key]]));

    draft.resetSelections(ctx);
    openTab('txt2vid');
    draft.resetSelections(ctx);

    expect(draft.state).toMatchObject(audio);
    expect(heldUploads.voice?.name).toBe('me.wav');
    expect(voiceConsent.ticked).toBe(true);
  });

  it('P2: an Image Reset leaves the Video tab on the kind it was on', () => {
    draft.hydrateFromContext(ctx, null);
    openTab('extend');
    openTab('txt2img');

    draft.resetSelections(ctx);
    draft.setTab('video');
    draft.onTabChange(ctx);

    expect(draft.state).toMatchObject({ workflow: 'extend', videoKind: 'extend' });
  });

  it('P3: a Video Reset on Extend stays on Extend', () => {
    draft.hydrateFromContext(ctx, null);
    openTab('extend');

    draft.resetSelections(ctx);

    expect(draft.state).toMatchObject({ workflow: 'extend', videoKind: 'extend' });
  });

  it('P4: a Reset clears the enhancement options of its own mode only, and keeps the enhancer, format and auto choice', () => {
    draft.hydrateFromContext(ctx, null);
    draft.patch({ enhanceSettings: SETTINGS, enhanceSettingsVideo: { ...SETTINGS, partsByWorkflow: {} }, enhanceAuto: true, enhanceModel: 'smart', enhanceFormat: 'json' });

    draft.resetSelections(ctx);
    expect(draft.state).toMatchObject({ enhanceSettings: null, enhanceSettingsVideo: { style: 'cinematic' }, enhanceAuto: true, enhanceModel: 'smart', enhanceFormat: 'json' });

    draft.patch({ enhanceSettings: SETTINGS });
    openTab('txt2vid');
    draft.resetSelections(ctx);
    expect(draft.state).toMatchObject({ enhanceSettings: { style: 'cinematic' }, enhanceSettingsVideo: null, enhanceAuto: true, enhanceModel: 'smart', enhanceFormat: 'json' });
  });

  it('returns an Extend Reset\'s length to the source\'s default and its track to the source\'s own', () => {
    draft.hydrateFromContext(ctx, null);
    openTab('extend');
    draft.setExtendSourcePath('/clips/source.mp4');
    draft.applyExtendSource(sourceInfo({ path: '/clips/source.mp4', audio_track_path: '/audio/source.mp3', audio_track_end: 4 }), ctx);
    draft.patch({ extendFrames: 16, extendAudioTrackPath: '/audio/chosen.mp3', extendAudioTrackStart: 2 });

    draft.resetSelections(ctx);

    expect(draft.state).toMatchObject({ extendFrames: 48, extendAudioTrackPath: '/audio/source.mp3', extendAudioTrackStart: 4 });
  });
});

describe('round trips between the tabs (G3, M2, M3)', () => {
  it('M2: the LoRAs and the quantization survive a visit to the Audio tab', () => {
    draft.hydrateFromContext(ctx, null);
    draft.patch({ loraString: 'detail:0.8', quantize: 4 });

    openTab('music');
    openTab('txt2img');

    expect(draft.state).toMatchObject({ workflow: 'txt2img', model: 'zit', loraString: 'detail:0.8', quantize: 4 });
  });

  it('M2: the Video tab\'s LoRAs survive a visit to the Audio tab too', () => {
    draft.hydrateFromContext(ctx, null);
    openTab('txt2vid');
    draft.patch({ loraString: 'detail:0.6' });

    openTab('sound');
    openTab('txt2vid');

    expect(draft.state).toMatchObject({ workflow: 'txt2vid', loraString: 'detail:0.6' });
  });

  it('M3: the image quantization survives a visit to the Video tab, and a Reset keeps it', () => {
    draft.hydrateFromContext(ctx, null);
    draft.update('quantize', 4);

    openTab('txt2vid');
    expect(draft.state.quantize).toBeNull();
    openTab('txt2img');
    expect(draft.state).toMatchObject({ model: 'zit', quantize: 4 });

    draft.resetSelections(ctx);
    expect(draft.state.quantize).toBe(4);
  });

  it('M3: it survives Image, Audio, Video, Image as well', () => {
    draft.hydrateFromContext(ctx, null);
    draft.update('quantize', 8);

    openTab('speech');
    openTab('txt2vid');
    openTab('txt2img');

    expect(draft.state.quantize).toBe(8);
  });

  it('keeps each tab\'s own state across Image, Video, Audio and back', () => {
    draft.hydrateFromContext(ctx, null);
    draft.patch({ loraString: 'detail:0.8', quantize: 4, prompt: 'an image' });
    openTab('extend');
    draft.setExtendSourcePath('/clips/source.mp4');
    openTab('speech');
    draft.patch({ speechText: 'say this', voiceMode: 'recording', voicePath: '/voices/me.wav', voiceText: 'hi' });
    draft.setAudioKind('music', ctx);
    draft.patch({ audioStyle: 'calm' });

    openTab('txt2img');
    expect(draft.state).toMatchObject({ prompt: 'an image', loraString: 'detail:0.8', quantize: 4 });
    draft.setTab('video');
    draft.onTabChange(ctx);
    expect(draft.state).toMatchObject({ workflow: 'extend', videoKind: 'extend', extendSourcePath: '/clips/source.mp4' });
    openTab('music');
    expect(draft.state).toMatchObject({ workflow: 'music', audioStyle: 'calm', speechText: 'say this', voicePath: '/voices/me.wav', voiceText: 'hi', audioModels: { music: 'yue2', speech: 'higgs3', sound: '' } });
  });

  it('M2: each mode gets its own LoRAs back after a visit to Audio from the other mode', () => {
    draft.hydrateFromContext(ctx, null);
    draft.patch({ loraString: 'detail:0.8' });
    openTab('txt2vid');
    draft.patch({ loraString: 'detail:0.6' });

    openTab('music');
    openTab('txt2img');
    expect(draft.state.loraString).toBe('detail:0.8');
    openTab('txt2vid');
    expect(draft.state.loraString).toBe('detail:0.6');
  });
});

describe('Generate\'s LoRA gate (M4)', () => {
  const downloading = { id: 'd1', kind: 'lora', name: 'ghost', label: 'LoRA ghost', state: 'running', message: null, bytes_done: 40, bytes_total: 100 } as DownloadItem;

  it('holds an image or video Generate back for a LoRA still downloading, and never an audio one', () => {
    vi.spyOn(downloadStore, 'items', 'get').mockReturnValue([downloading]);
    const loraString = 'ghost:0.8';

    expect(generateBlock({ workflow: 'txt2img', loraString }, ctx.loras).blocked).toBe(true);
    expect(generateBlock({ workflow: 'txt2vid', loraString }, ctx.loras).blocked).toBe(true);
    for (const workflow of ['music', 'speech', 'sound'] as const) expect(generateBlock({ workflow, loraString }, ctx.loras).blocked).toBe(false);
  });
});

describe('the speaker\'s tick follows the voice (G4)', () => {
  function speechWithTickedVoice(): void {
    draft.hydrateFromContext(ctx, null);
    openTab('speech');
    draft.setVoice({ path: '/voices/me.wav' });
    draft.update('voiceText', 'hello');
    voiceConsent.ticked = true;
  }

  it('is kept by a music or a sound Reuse, which never touch the voice', () => {
    speechWithTickedVoice();

    draft.loadFromUrl({ workflow: 'music', model: 'yue2', prompt: 'calm piano', lyrics: '' }, ctx);
    draft.loadFromUrl({ workflow: 'sound', model: 'sa3', prompt: 'rain', seconds: '8' }, ctx);

    expect(draft.state).toMatchObject({ voiceMode: 'recording', voicePath: '/voices/me.wav', voiceText: 'hello' });
    expect(voiceConsent.ticked).toBe(true);
  });

  it('is cleared by a prefill that replaces the voice, and by a new recording', () => {
    speechWithTickedVoice();
    draft.loadFromUrl({ workflow: 'speech', model: 'higgs3', prompt: 'Welcome', voice_path: '/voices/other.wav', voice_text: 'Hi' }, ctx);
    expect(voiceConsent.ticked).toBe(false);

    voiceConsent.ticked = true;
    draft.setVoice({ path: '/voices/third.wav' });
    expect(voiceConsent.ticked).toBe(false);
  });
});
