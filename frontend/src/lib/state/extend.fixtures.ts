import type { ExtendSourceInfo } from '$lib/types';

/** Test fixture: what `POST /api/extend/source` reports for a readable 24 fps clip with no recorded text. */
export function sourceInfo(overrides: Partial<ExtendSourceInfo> = {}): ExtendSourceInfo {
  return {
    path: '/clips/walk.mp4', filename: 'walk.mp4', media_url: '/media/walk.mp4', width: 512, height: 288, rotation: 0, fps: 24, frame_count: 97,
    duration_seconds: 4.042, has_audio: true, scene: null, prompt: null, legacy: false, model: null, lora: null, problems: [],
    extend_frames: 48, window_frames: 73, memory_fit: null, ...overrides,
  };
}

/** Test fixture: a source that recorded a scene and an action. */
export function sceneSource(overrides: Partial<ExtendSourceInfo> = {}): ExtendSourceInfo {
  return sourceInfo({ scene: 'A woman on a rainy street.', prompt: 'She looks up.', ...overrides });
}

/** Test fixture: a source that recorded one prompt and no scene. */
export function legacySource(overrides: Partial<ExtendSourceInfo> = {}): ExtendSourceInfo {
  return sourceInfo({ scene: null, prompt: 'A woman stands in the rain and says "Hello".', legacy: true, ...overrides });
}
