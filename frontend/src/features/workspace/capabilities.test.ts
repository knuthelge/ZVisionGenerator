import { describe, expect, it } from 'vitest';

import { AUDIO_MODELS, withAudio } from '../../test-utils/audio';
import { makeContext, makeImageDefaults } from '../../test-utils/workspace';
import type { AudioModelOption, DraftState, VideoModelDefaults, WorkspaceContext } from '$lib/types';
import { workspaceCapabilities } from './capabilities';
import { stepsSpec, SPEC } from './settingsSpec';

const VIDEO_CONTROLS = ['workflow', 'model', 'runs', 'frame_count', 'steps', 'seed', 'audio', 'low_memory', 'video_upscale_enabled', 'video_upscale_factor', 'video_upscale_review'];

function videoDefaults(overrides: Partial<VideoModelDefaults> = {}): VideoModelDefaults {
  return {
    ratio: '16:9', size: 's', steps: 8, width: 768, height: 448, frame_count: 121, audio: true, low_memory: true, supports_i2v: true,
    supports_quantize: false, quantize: null, max_steps: 8, fps: 24, frame_rule: { alignment: 1, offset: 1, round: 'nearest' }, supports_extend: true, supports_audio_track_upscale: true, extend_frame_step: 8, upscale: { enabled: false, factor: 2, steps: null }, ...overrides,
  };
}

function contextOf(models: Record<string, VideoModelDefaults>): WorkspaceContext {
  return {
    video_model_defaults: models,
    prompt_sources: ['inline'],
    scheduler_options: [],
    workflow_contract: { values: ['txt2vid'], definitions: { txt2vid: { mode: 'video', model_kind: 'video', visible_controls: VIDEO_CONTROLS, requires_reference_image: false, requires_source_video: false, clear_fields: [] } }, field_precedence: { defaults: [], dimensions: '' } },
  } as unknown as WorkspaceContext;
}

const stateFor = (model: string): DraftState => ({ workflow: 'txt2vid', model, referenceImagePaths: [] }) as unknown as DraftState;

describe('workspaceCapabilities for a video model', () => {
  const context = contextOf({
    ltx: videoDefaults({ hidden_controls: [] }),
    h3: videoDefaults({ hidden_controls: ['low_memory'] }),
    'h3-mac': videoDefaults({ hidden_controls: ['low_memory', 'video_upscale_enabled', 'video_upscale_factor', 'video_upscale_review'] }),
  });

  it('shows every control of a model that hides none', () => {
    const caps = workspaceCapabilities(context, stateFor('ltx'));
    expect([caps.showAudio, caps.showLowMemory, caps.showVideoUpscale, caps.showVideoUpscaleFactor, caps.showVideoUpscaleReview]).toEqual([true, true, true, true, true]);
  });

  it('hides only low memory for a model without that mode', () => {
    const caps = workspaceCapabilities(context, stateFor('h3'));
    expect([caps.showAudio, caps.showLowMemory, caps.showVideoUpscale, caps.showVideoUpscaleFactor, caps.showVideoUpscaleReview]).toEqual([true, false, true, true, true]);
  });

  it('hides the upscale controls too when the platform has no upscale', () => {
    const caps = workspaceCapabilities(context, stateFor('h3-mac'));
    expect([caps.showAudio, caps.showLowMemory, caps.showVideoUpscale, caps.showVideoUpscaleFactor, caps.showVideoUpscaleReview]).toEqual([true, false, false, false, false]);
    expect(caps.showSteps).toBe(true);
  });

  it('shows the controls of a model that sends no hidden list', () => {
    const caps = workspaceCapabilities(contextOf({ legacy: videoDefaults() }), stateFor('legacy'));
    expect(caps.showLowMemory).toBe(true);
  });
});

describe('workspaceCapabilities for the last frame', () => {
  const IMAGE_CONTROLS = [...VIDEO_CONTROLS, 'reference_image', 'reference_image_path', 'reference_image_clear', 'end_image', 'end_image_path', 'end_image_clear'];
  const END_CONTROLS = ['end_image', 'end_image_clear', 'end_image_path'];

  /** A video context whose two workflows both list the start and last-frame controls, as the server's contract does. */
  function videoContext(models: Record<string, VideoModelDefaults>): [WorkspaceContext, (model: string, workflow?: 'img2vid' | 'txt2vid') => DraftState] {
    const definition = (controls: string[]) => ({ mode: 'video', model_kind: 'video', visible_controls: controls, requires_reference_image: true, requires_source_video: false, clear_fields: [] });
    const context = {
      video_model_defaults: models,
      prompt_sources: ['inline'],
      scheduler_options: [],
      workflow_contract: { values: ['txt2vid', 'img2vid'], definitions: { txt2vid: definition(IMAGE_CONTROLS), img2vid: definition(IMAGE_CONTROLS) }, field_precedence: { defaults: [], dimensions: '' } },
    } as unknown as WorkspaceContext;
    return [context, (model, workflow = 'img2vid') => ({ workflow, model, referenceImagePaths: [] }) as unknown as DraftState];
  }

  it('takes a last frame on a model that can end a clip on an image, in both video workflows', () => {
    const [context, stateOf] = videoContext({ h3: videoDefaults({ hidden_controls: ['low_memory'] }) });
    expect(workspaceCapabilities(context, stateOf('h3', 'img2vid')).takesLastFrame).toBe(true);
    expect(workspaceCapabilities(context, stateOf('h3', 'txt2vid')).takesLastFrame).toBe(true);
  });

  it('hides the slot for a model whose hidden controls name it, and keeps the first-frame slot', () => {
    const [context, stateOf] = videoContext({ ltx: videoDefaults({ hidden_controls: END_CONTROLS }) });
    const caps = workspaceCapabilities(context, stateOf('ltx'));
    expect([caps.takesLastFrame, caps.showRefImage]).toEqual([false, true]);
  });

  it('hides both slots for a video model that cannot start from an image', () => {
    const [context, stateOf] = videoContext({ t2v: videoDefaults({ supports_i2v: false, hidden_controls: [] }) });
    const caps = workspaceCapabilities(context, stateOf('t2v', 'txt2vid'));
    expect([caps.showRefImage, caps.takesLastFrame]).toEqual([false, false]);
  });

  it('follows the default video model for a model the context lists no defaults for', () => {
    const [context, stateOf] = videoContext({});
    (context as unknown as { video_defaults: VideoModelDefaults }).video_defaults = videoDefaults({ hidden_controls: END_CONTROLS });
    const caps = workspaceCapabilities(context, stateOf('unlisted', 'txt2vid'));
    expect([caps.showRefImage, caps.takesLastFrame]).toEqual([true, false]);
  });
});

describe('workspaceCapabilities for the start image', () => {
  const IMAGE_CONTROLS = ['reference_image', 'reference_image_path', 'reference_image_clear'];

  function imageContext(controls: string[]): WorkspaceContext {
    const definition = (visible: string[]) => ({ mode: 'image', model_kind: 'image', visible_controls: visible, requires_reference_image: false, requires_source_video: false, clear_fields: [] });
    return {
      image_model_defaults: { zit: { supports_img2img: true, max_reference_images: 0 }, ideo: { supports_img2img: false, max_reference_images: 0 }, klein: { supports_img2img: false, max_reference_images: 4 } },
      prompt_sources: ['inline'],
      scheduler_options: [],
      workflow_contract: { values: ['txt2img', 'img2img'], definitions: { txt2img: definition(controls), img2img: definition(controls) }, field_precedence: { defaults: [], dimensions: '' } },
    } as unknown as WorkspaceContext;
  }
  const stateOf = (model: string, workflow: 'txt2img' | 'img2img'): DraftState => ({ workflow, model, referenceImagePaths: [] }) as unknown as DraftState;

  it('the start image slot shows in both image workflows, not without img2img', () => {
    const context = imageContext(IMAGE_CONTROLS);
    expect(workspaceCapabilities(context, stateOf('zit', 'txt2img')).showRefImage).toBe(true);
    expect(workspaceCapabilities(context, stateOf('zit', 'img2img')).showRefImage).toBe(true);
    expect(workspaceCapabilities(context, stateOf('ideo', 'txt2img')).showRefImage).toBe(false);
  });

  it('takes a start image or references independently', () => {
    const context = imageContext([...IMAGE_CONTROLS, 'references']);
    expect(workspaceCapabilities(context, stateOf('ideo', 'txt2img')).showRefImage).toBe(false);
    const caps = workspaceCapabilities(context, stateOf('klein', 'txt2img'));
    expect([caps.showRefImage, caps.maxReferences]).toEqual([false, 4]);
  });
});

describe('workspaceCapabilities for references', () => {
  const imageContext = (cap: number | undefined, controls: string[] = ['references', 'references_clear']): WorkspaceContext => ({
    image_model_defaults: { klein: { max_reference_images: cap }, zit: {} },
    prompt_sources: ['inline'],
    scheduler_options: [],
    workflow_contract: {
      values: ['txt2img'],
      definitions: { txt2img: { mode: 'image', model_kind: 'image', visible_controls: controls, requires_reference_image: false, requires_source_video: false, clear_fields: [] } },
      field_precedence: { defaults: [], dimensions: '' },
    },
  }) as unknown as WorkspaceContext;
  const stateOf = (model: string): DraftState => ({ workflow: 'txt2img', model, referenceImagePaths: [] }) as unknown as DraftState;

  it('shows the section with the model cap on a model that takes references', () => {
    const caps = workspaceCapabilities(imageContext(4), stateOf('klein'));
    expect(caps.maxReferences).toBe(4);
  });

  it('hides it for a model without references, and for a model that sends no cap', () => {
    expect(workspaceCapabilities(imageContext(0), stateOf('klein')).maxReferences).toBe(0);
    expect(workspaceCapabilities(imageContext(4), stateOf('zit')).maxReferences).toBe(0);
  });

  it('hides it where the workflow does not list the controls', () => {
    expect(workspaceCapabilities(imageContext(4, []), stateOf('klein')).maxReferences).toBe(0);
  });
});

describe('workspaceCapabilities for Transparent', () => {
  const contextWith = (workflow: 'txt2img' | 'txt2vid', visible: string[]): WorkspaceContext => ({
    video_model_defaults: {},
    prompt_sources: ['inline'],
    scheduler_options: [],
    workflow_contract: {
      values: [workflow],
      definitions: { [workflow]: { mode: workflow === 'txt2img' ? 'image' : 'video', model_kind: workflow === 'txt2img' ? 'image' : 'video', visible_controls: visible, requires_reference_image: false, requires_source_video: false, clear_fields: [] } },
      field_precedence: { defaults: [], dimensions: '' },
    },
  }) as unknown as WorkspaceContext;
  const stateOf = (workflow: string): DraftState => ({ workflow, model: 'm', referenceImagePaths: [] }) as unknown as DraftState;

  it('shows in an image workflow that lists the control', () => {
    expect(workspaceCapabilities(contextWith('txt2img', ['postprocess_transparent']), stateOf('txt2img')).showPostprocessTransparent).toBe(true);
  });

  it('is hidden where the workflow does not list it, and in a video workflow', () => {
    expect(workspaceCapabilities(contextWith('txt2img', ['postprocess_sharpen']), stateOf('txt2img')).showPostprocessTransparent).toBe(false);
    expect(workspaceCapabilities(contextWith('txt2vid', ['postprocess_transparent']), stateOf('txt2vid')).showPostprocessTransparent).toBe(false);
  });
});

describe('stepsSpec', () => {
  it('steps between the offered counts', () => {
    expect(stepsSpec([4, 8])).toEqual({ step: 4, min: 4, max: 8 });
  });

  it('keeps the usual control for a model that offers any count', () => {
    expect(stepsSpec(null)).toBe(SPEC.steps);
    expect(stepsSpec(undefined)).toBe(SPEC.steps);
    expect(stepsSpec([])).toBe(SPEC.steps);
    expect(stepsSpec(null, null)).toBe(SPEC.steps);
  });

  it('cuts the usual control at the most steps the model runs', () => {
    expect(stepsSpec(null, 8)).toEqual({ ...SPEC.steps, max: 8 });
    expect(stepsSpec(undefined, 500)).toEqual(SPEC.steps);
  });
});

describe('workspaceCapabilities for the audio track', () => {
  const TRACK_CONTROLS = ['audio_track', 'audio_track_path', 'audio_track_start'];
  const definition = (controls: string[]) => ({ mode: 'video', model_kind: 'video', visible_controls: controls, requires_reference_image: false, requires_source_video: false, clear_fields: [] });
  const imageDefinition = { mode: 'image', model_kind: 'image', visible_controls: ['workflow', 'model'], requires_reference_image: false, requires_source_video: false, clear_fields: [] };
  const context = {
    video_model_defaults: {
      ltx: videoDefaults({ hidden_controls: [] }),
      h3: videoDefaults({ hidden_controls: [], supports_audio_track_upscale: false }),
      plain: videoDefaults({ hidden_controls: TRACK_CONTROLS }),
    },
    prompt_sources: ['inline'],
    scheduler_options: [],
    workflow_contract: {
      values: ['txt2img', 'txt2vid', 'extend'],
      definitions: { txt2img: imageDefinition, txt2vid: definition([...VIDEO_CONTROLS, ...TRACK_CONTROLS]), extend: definition([...VIDEO_CONTROLS, ...TRACK_CONTROLS]) },
      field_precedence: { defaults: [], dimensions: '' },
    },
  } as unknown as WorkspaceContext;
  const stateOf = (model: string, overrides: Partial<DraftState> = {}): DraftState => ({ workflow: 'txt2vid', model, referenceImagePaths: [], audioTrackPath: null, audioTrackUpload: null, extendAudioTrackPath: null, extendAudioTrackUpload: null, ...overrides }) as unknown as DraftState;

  it('offers the section on a model that follows a track, with the Audio switch until one is set', () => {
    const caps = workspaceCapabilities(context, stateOf('ltx'));

    expect([caps.showAudioTrack, caps.audioTrackInEffect, caps.showAudio]).toEqual([true, false, true]);
  });

  it('takes the Audio switch away while a track is set, as a path or as an upload', () => {
    for (const set of [{ audioTrackPath: '/a/song.mp3' }, { audioTrackUpload: 'memo.m4a' }]) {
      const caps = workspaceCapabilities(context, stateOf('ltx', set));
      expect([caps.audioTrackInEffect, caps.showAudio]).toEqual([true, false]);
    }
  });

  it('reads the track of the open tab: the Extend track does not count in Video', () => {
    expect(workspaceCapabilities(context, stateOf('ltx', { extendAudioTrackPath: '/a/song.mp3' })).audioTrackInEffect).toBe(false);
    expect(workspaceCapabilities(context, stateOf('ltx', { workflow: 'extend', extendAudioTrackPath: '/a/song.mp3' })).audioTrackInEffect).toBe(true);
    expect(workspaceCapabilities(context, stateOf('ltx', { workflow: 'extend', audioTrackPath: '/a/song.mp3' })).audioTrackInEffect).toBe(false);
  });

  it('hides the section on a model without the capability, keeps the kept track out of effect and brings the Audio switch back', () => {
    const caps = workspaceCapabilities(context, stateOf('plain', { audioTrackPath: '/a/song.mp3' }));

    expect([caps.showAudioTrack, caps.audioTrackInEffect, caps.showAudio]).toEqual([false, false, true]);
  });

  it('offers no section in Image', () => {
    const caps = workspaceCapabilities(context, stateOf('ltx', { workflow: 'txt2img', audioTrackPath: '/a/song.mp3' }));

    expect([caps.showAudioTrack, caps.audioTrackInEffect]).toEqual([false, false]);
  });

  it('keeps the Upscale controls with a track on a model that can upscale one', () => {
    const caps = workspaceCapabilities(context, stateOf('ltx', { audioTrackPath: '/a/song.mp3' }));

    expect([caps.showVideoUpscale, caps.showVideoUpscaleFactor, caps.showVideoUpscaleReview]).toEqual([true, true, true]);
  });

  it('hides the Upscale controls while a track is in effect on a model that cannot upscale one, and shows them without', () => {
    const withTrack = workspaceCapabilities(context, stateOf('h3', { audioTrackPath: '/a/song.mp3' }));
    const without = workspaceCapabilities(context, stateOf('h3'));

    expect([withTrack.showVideoUpscale, withTrack.showVideoUpscaleFactor, withTrack.showVideoUpscaleReview, withTrack.showLowMemory]).toEqual([false, false, false, true]);
    expect([without.showVideoUpscale, without.showVideoUpscaleFactor, without.showVideoUpscaleReview]).toEqual([true, true, true]);
  });
});

describe('workspaceCapabilities for an audio model', () => {
  const sa3Sound = AUDIO_MODELS[2];
  // Stable Audio as music: it has a length but no lyrics, so its row hides the Lyrics box and (here) lists a length control.
  const sa3Music: AudioModelOption = { ...sa3Sound, kind: 'music', hidden_controls: ['lyrics'], limits: { min_seconds: 1, max_seconds: 380, default_seconds: 30 } };
  const base = withAudio(makeContext(), [...AUDIO_MODELS, sa3Music]);
  const context = {
    ...base,
    workflow_contract: {
      ...base.workflow_contract,
      definitions: {
        ...base.workflow_contract.definitions,
        music: { ...base.workflow_contract.definitions.music!, visible_controls: [...base.workflow_contract.definitions.music!.visible_controls, 'length'] },
      },
    },
  } as WorkspaceContext;
  const audioState = (workflow: 'music' | 'speech' | 'sound', model: string): DraftState => ({ workflow, audioKind: workflow, model }) as unknown as DraftState;

  it('hides the controls the row of the model lists for the open kind', () => {
    expect(workspaceCapabilities(context, audioState('music', 'yue2')).showLength).toBe(false); // the lyrics set a song's length
    expect(workspaceCapabilities(context, audioState('music', 'sa3')).showLength).toBe(true);
    expect(workspaceCapabilities(context, audioState('sound', 'sa3')).showLength).toBe(true);
  });

  it('reads the row of the kind, not another row of the same model', () => {
    const noLength = { ...sa3Sound, hidden_controls: ['length'] } as AudioModelOption;
    const mixed = { ...context, audio_models: [noLength, { ...sa3Music, hidden_controls: [] }] } as WorkspaceContext;

    expect(workspaceCapabilities(mixed, audioState('sound', 'sa3')).showLength).toBe(false);
    expect(workspaceCapabilities(mixed, audioState('music', 'sa3')).showLength).toBe(true);
  });

  it('hides nothing for a model the context does not list', () => {
    expect(workspaceCapabilities(context, audioState('music', 'nope')).showLength).toBe(true);
  });

  it('shows the Tempo and Key of music, the Delivery of speech and the Sound Type of sound, each only on its kind', () => {
    const flags = (kind: 'music' | 'speech' | 'sound', model: string, overrides: Partial<DraftState> = {}) => {
      const caps = workspaceCapabilities(context, { ...audioState(kind, model), voiceMode: 'builtin', ...overrides } as DraftState);
      return [caps.showTempo, caps.showKey, caps.showDelivery, caps.showSoundType];
    };

    expect(flags('music', 'yue2')).toEqual([true, true, false, false]);
    expect(flags('speech', 'qwen3_tts')).toEqual([false, false, true, false]);
    expect(flags('sound', 'sa3')).toEqual([false, false, false, true]);
  });

  it('shows no Delivery controls for a voice that follows a recording, but still offers the section, which says why', () => {
    const caps = workspaceCapabilities(context, { ...audioState('speech', 'qwen3_tts'), voiceMode: 'recording' } as DraftState);

    expect(caps.showDelivery).toBe(false);
    expect(caps.offersDelivery).toBe(true);
    expect(workspaceCapabilities(context, audioState('music', 'yue2')).offersDelivery).toBe(false);
  });

  it('shows no Delivery controls for a saved voice either, which is read as recorded, and still offers the section', () => {
    const caps = workspaceCapabilities(context, { ...audioState('speech', 'qwen3_tts'), voiceMode: 'saved', savedVoice: 'narrator' } as DraftState);

    expect(caps.showDelivery).toBe(false);
    expect(caps.offersDelivery).toBe(true);
  });

  it('shows the Lyrics box only where the row of the model does not hide it', () => {
    expect(workspaceCapabilities(context, audioState('music', 'yue2')).showLyrics).toBe(true);
    expect(workspaceCapabilities(context, audioState('music', 'sa3')).showLyrics).toBe(false);
  });
});

describe('workspaceCapabilities for the controls the contract lists per workflow', () => {
  const FLAGS = [false, true];
  const modelId = (negative: boolean, json: boolean, sigma: boolean): string => `m-${negative}-${json}-${sigma}`;
  const context: WorkspaceContext = (() => {
    const base = withAudio(makeContext());
    const models = Object.fromEntries(FLAGS.flatMap((negative) => FLAGS.flatMap((json) => FLAGS.map((sigma) => [
      modelId(negative, json, sigma),
      makeImageDefaults({ supports_negative_prompt: negative, supports_json_prompt: json, supports_first_sigma: sigma }),
    ]))));
    return { ...base, image_model_defaults: { ...base.image_model_defaults, ...models } };
  })();
  const stateOf = (workflow: string, model: string): DraftState => ({ workflow, model, promptSource: 'inline', referenceImagePaths: [] }) as unknown as DraftState;

  it('shows a control where the workflow lists it and the model supports it, and nowhere else', () => {
    const all = modelId(true, true, true);
    const none = modelId(false, false, false);
    const listing = (controls: string[]) => ({
      ...context,
      workflow_contract: { ...context.workflow_contract, definitions: { ...context.workflow_contract.definitions, edit: { ...context.workflow_contract.definitions.edit!, visible_controls: ['workflow', 'model', 'prompt_inline', ...controls] } } },
    });
    const flagsOf = (ctx: WorkspaceContext, model: string) => {
      const caps = workspaceCapabilities(ctx, stateOf('edit', model));
      return [caps.showPreset, caps.showNegativePrompt, caps.supportsJsonPrompt, caps.supportsFirstSigma];
    };

    expect(flagsOf(listing([]), all)).toEqual([false, false, false, false]);
    expect(flagsOf(listing(['preset', 'negative_prompt', 'json_prompt', 'first_sigma']), all)).toEqual([true, true, true, true]);
    expect(flagsOf(listing(['preset', 'negative_prompt', 'json_prompt', 'first_sigma']), none)).toEqual([true, false, false, false]);
  });
});
