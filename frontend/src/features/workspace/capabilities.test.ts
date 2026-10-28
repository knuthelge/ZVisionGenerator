import { describe, expect, it } from 'vitest';

import type { DraftState, VideoModelDefaults, WorkspaceContext } from '$lib/types';
import { workspaceCapabilities } from './capabilities';
import { stepsSpec, SPEC } from './settingsSpec';

const VIDEO_CONTROLS = ['workflow', 'model', 'runs', 'frame_count', 'steps', 'seed', 'audio', 'low_memory', 'video_upscale_enabled', 'video_upscale_factor', 'video_upscale_review'];

function videoDefaults(overrides: Partial<VideoModelDefaults> = {}): VideoModelDefaults {
  return {
    ratio: '16:9', size: 's', steps: 8, width: 768, height: 448, frame_count: 121, audio: true, low_memory: true, supports_i2v: true,
    supports_quantize: false, quantize: null, max_steps: 8, fps: 24, frame_rule: { alignment: 1, offset: 1, round: 'nearest' }, supports_extend: true, extend_frame_step: 8, upscale: { enabled: false, factor: 2, steps: null }, ...overrides,
  };
}

function contextOf(models: Record<string, VideoModelDefaults>): WorkspaceContext {
  return {
    video_model_defaults: models,
    prompt_sources: ['inline'],
    scheduler_options: [],
    workflow_contract: { values: ['txt2vid'], definitions: { txt2vid: { mode: 'video', model_kind: 'video', visible_controls: VIDEO_CONTROLS, supports_reference_image: false, requires_reference_image: false, requires_source_video: false, clear_fields: [] } }, field_precedence: { defaults: [], dimensions: '' } },
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
  const IMG2VID = [...VIDEO_CONTROLS, 'reference_image', 'reference_image_path', 'reference_image_clear', 'end_image', 'end_image_path', 'end_image_clear'];
  const END_CONTROLS = ['end_image', 'end_image_clear', 'end_image_path'];

  function img2vid(models: Record<string, VideoModelDefaults>, workflow: 'img2vid' | 'txt2vid' = 'img2vid'): [WorkspaceContext, (model: string) => DraftState] {
    const definition = (controls: string[]) => ({ mode: 'video', model_kind: 'video', visible_controls: controls, supports_reference_image: true, requires_reference_image: true, requires_source_video: false, clear_fields: [] });
    const context = {
      video_model_defaults: models,
      prompt_sources: ['inline'],
      scheduler_options: [],
      workflow_contract: { values: ['txt2vid', 'img2vid'], definitions: { txt2vid: definition(VIDEO_CONTROLS), img2vid: definition(IMG2VID) }, field_precedence: { defaults: [], dimensions: '' } },
    } as unknown as WorkspaceContext;
    return [context, (model) => ({ workflow, model, referenceImagePaths: [] }) as unknown as DraftState];
  }

  it('offers the slot to a model that can end a clip on an image, in Image to Video only', () => {
    const [context, stateOf] = img2vid({ h3: videoDefaults({ hidden_controls: ['low_memory'] }) });
    expect(workspaceCapabilities(context, stateOf('h3')).showEndImage).toBe(true);
    const [textContext, textState] = img2vid({ h3: videoDefaults({ hidden_controls: ['low_memory'] }) }, 'txt2vid');
    expect(workspaceCapabilities(textContext, textState('h3')).showEndImage).toBe(false);
  });

  it('hides the slot for a model whose hidden controls name it, and keeps the first-frame picker', () => {
    const [context, stateOf] = img2vid({ ltx: videoDefaults({ hidden_controls: END_CONTROLS }) });
    const caps = workspaceCapabilities(context, stateOf('ltx'));
    expect([caps.showEndImage, caps.showRefImage]).toEqual([false, true]);
  });
});

describe('workspaceCapabilities for references', () => {
  const imageContext = (cap: number | undefined, controls: string[] = ['references', 'references_clear']): WorkspaceContext => ({
    image_model_defaults: { klein: { max_reference_images: cap }, zit: {} },
    prompt_sources: ['inline'],
    scheduler_options: [],
    workflow_contract: {
      values: ['txt2img'],
      definitions: { txt2img: { mode: 'image', model_kind: 'image', visible_controls: controls, supports_reference_image: false, requires_reference_image: false, requires_source_video: false, clear_fields: [] } },
      field_precedence: { defaults: [], dimensions: '' },
    },
  }) as unknown as WorkspaceContext;
  const stateOf = (model: string): DraftState => ({ workflow: 'txt2img', model, referenceImagePaths: [] }) as unknown as DraftState;

  it('shows the section with the model cap on a model that takes references', () => {
    const caps = workspaceCapabilities(imageContext(4), stateOf('klein'));
    expect([caps.showReferences, caps.maxReferences]).toEqual([true, 4]);
  });

  it('hides it for a model without references, and for a model that sends no cap', () => {
    expect(workspaceCapabilities(imageContext(0), stateOf('klein')).showReferences).toBe(false);
    expect(workspaceCapabilities(imageContext(4), stateOf('zit')).showReferences).toBe(false);
  });

  it('counts the first rows of the list the model takes, and none where the section is hidden', () => {
    const state = (model: string, count: number): DraftState => ({ ...stateOf(model), referenceImagePaths: Array.from({ length: count }, (_, i) => `/in/${i}.png`) }) as DraftState;

    expect(workspaceCapabilities(imageContext(4), state('klein', 3)).referenceCount).toBe(3);
    expect(workspaceCapabilities(imageContext(2), state('klein', 3)).referenceCount).toBe(2);
    expect(workspaceCapabilities(imageContext(0), state('klein', 3)).referenceCount).toBe(0);
  });

  it('hides it where the workflow does not list the controls', () => {
    expect(workspaceCapabilities(imageContext(4, []), stateOf('klein')).showReferences).toBe(false);
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
  });
});
