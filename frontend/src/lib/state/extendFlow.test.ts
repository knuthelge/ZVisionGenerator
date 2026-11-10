import { beforeEach, describe, expect, it } from 'vitest';

import type { GalleryAsset } from '$lib/types';

import { extendParams } from './assetActions';
import { draft } from './draft.svelte';
import { isOriginalLegacyPrompt } from './extend';
import { legacySource, sceneSource, sourceInfo } from './extend.fixtures';
import reuseFixture from '../../../../tests/fixtures/extend_reuse_params.json';
import { jobSettingsPrefill } from './jobSettings';
import { enhancedStaleReason } from './promptEnhance';

// End-to-end flows through the draft: the params a gallery action or Reuse builds, then the server's answer for the source.

const SOURCE = '/clips/walk.mp4';

function extendAsset(overrides: Partial<GalleryAsset> = {}): GalleryAsset {
  return {
    id: 'out/walk.mp4', url: '/media/out/walk.mp4', thumbnail_url: '/media/out/walk.mp4', filename: 'walk.mp4', created_at: '2026-01-01T00:00:00Z',
    workflow: 'txt2vid', prompt: 'She looks up.', model: 'ltx', media_type: 'video', reuse_workspace_url: '#/workspace', file_path: SOURCE,
    extend: { available: true, reason: null }, ...overrides,
  } as GalleryAsset;
}

describe('extend flows', () => {
  beforeEach(() => {
    localStorage.clear();
    draft.reset();
    draft.update('workflow', 'extend');
  });

  it('the Extend action of a scene-bearing video fills Scene from the server and leaves Prompt empty', () => {
    draft.loadFromUrl(extendParams(extendAsset({ scene: 'A woman on a rainy street.' })));
    draft.applyExtendSource(sceneSource({ path: SOURCE, scene: 'A woman on a rainy street.' }));
    expect(draft.state).toMatchObject({ extendSourcePath: SOURCE, scene: 'A woman on a rainy street.', prompt: '' });
  });

  it('the Extend action of a legacy video fills Prompt from the server and flags it as the original', () => {
    const asset = extendAsset({ scene: null, prompt: 'A woman says "Hello".' });
    draft.loadFromUrl(extendParams(asset));
    expect(draft.state).toMatchObject({ scene: '', prompt: '', extendPrefill: null });
    const info = legacySource({ prompt: 'A woman says "Hello".' });
    draft.applyExtendSource(info);
    expect(draft.state).toMatchObject({ scene: '', prompt: 'A woman says "Hello".' });
    expect(isOriginalLegacyPrompt(draft.state, info)).toBe(true);
    draft.update('prompt', 'She leaves.');
    expect(isOriginalLegacyPrompt(draft.state, info)).toBe(false);
  });

  it('the Extend action of a video with no recorded config leaves both boxes empty, not the file name the gallery shows', () => {
    draft.loadFromUrl(extendParams(extendAsset({ scene: null, prompt: 'walk' })));
    draft.applyExtendSource(sourceInfo());
    expect(draft.state).toMatchObject({ scene: '', prompt: '' });
  });

  it('a Reuse of an extension keeps the track it ran with through the inspection that follows', () => {
    draft.loadFromUrl({ workflow: 'extend', source_video: SOURCE, extend_frames: '48', scene: 'A singer.', audio_track_path: '/audio/song.mp3', audio_track_start: '10' }, null, { replaceSlots: true });

    draft.applyExtendSource(sceneSource({ path: SOURCE, audio_track_path: '/audio/song.mp3', audio_track_end: 6.65 }));

    expect(draft.state).toMatchObject({ extendAudioTrackPath: '/audio/song.mp3', extendAudioTrackStart: 10 });
  });

  it('a Reuse of an extension that recorded no text keeps both boxes blank through the inspection', () => {
    draft.loadFromUrl({ workflow: 'extend', source_video: SOURCE, extend_frames: '48', scene: '', prompt: '' });
    draft.applyExtendSource(sceneSource());
    expect(draft.state).toMatchObject({ scene: '', prompt: '', extendFrames: 48 });
  });

  it('a Reuse keeps its Enhanced pair current through the inspection that follows', () => {
    draft.loadFromUrl({
      workflow: 'extend', source_video: SOURCE, extend_frames: '48', scene: 'My street.', prompt: 'She waves.', enhanced_scene: 'A wet street.', enhanced_prompt: 'She waves slowly.',
    });
    draft.applyExtendSource(sceneSource());
    expect(draft.state).toMatchObject({ scene: 'My street.', prompt: 'She waves.', enhancedScene: 'A wet street.', enhancedPrompt: 'She waves slowly.', extendFrames: 48 });
    expect(enhancedStaleReason(draft.state)).toBeNull();
  });

  it('edits survive a second inspection, a new source and are replaced only on request', () => {
    draft.loadFromUrl(extendParams(extendAsset({ scene: 'A woman on a rainy street.' })));
    draft.applyExtendSource(sceneSource());
    draft.patch({ scene: 'My own scene.', prompt: 'My action.' });

    draft.applyExtendSource(sceneSource({ scene: 'Another recorded scene.' }));
    expect(draft.state).toMatchObject({ scene: 'My own scene.', prompt: 'My action.' });

    draft.setExtendSourcePath('/clips/other.mp4');
    draft.applyExtendSource(sceneSource({ path: '/clips/other.mp4', scene: 'Other scene.' }));
    expect(draft.state).toMatchObject({ scene: 'My own scene.', prompt: 'My action.' });

    draft.applyExtendSource(sceneSource({ path: '/clips/other.mp4', scene: 'Other scene.' }), null, { replaceText: true });
    expect(draft.state).toMatchObject({ scene: 'Other scene.', prompt: '', enhancedPrompt: '', enhancedScene: '', enhancedFrom: null });
  });

  it('an untouched prefill is replaced when another source is chosen', () => {
    draft.loadFromUrl(extendParams(extendAsset({ scene: 'A woman on a rainy street.' })));
    draft.applyExtendSource(sceneSource());
    draft.setExtendSourcePath('/clips/other.mp4');
    draft.applyExtendSource(legacySource({ path: '/clips/other.mp4' }));
    expect(draft.state).toMatchObject({ scene: '', prompt: 'A woman stands in the rain and says "Hello".' });
  });

  it('restoring the settings of an enhanced extension job leaves its Enhanced pair current', () => {
    const { params, patch } = jobSettingsPrefill({
      mode: 'video', workflow: 'extend', source_video: SOURCE, extend_frames: '48', scene_input: 'My street.', prompt_input: 'She waves.', scene: 'A wet street.', prompt: 'She waves slowly.',
    });
    draft.loadFromUrl(params);
    draft.patch(patch);
    expect(draft.state).toMatchObject({ enhancedScene: 'A wet street.', enhancedPrompt: 'She waves slowly.', extendSourcePath: SOURCE, extendFrames: 48 });
    expect(enhancedStaleReason(draft.state)).toBeNull();
  });

  // tests/fixtures/extend_reuse_params.json is also checked by pytest against the gallery's `reuse_params` output.
  it("loads the server's Reuse params of an enhanced extension whose source moved", () => {
    draft.loadFromUrl(reuseFixture.enhanced_source_moved);
    expect(draft.state).toMatchObject({
      workflow: 'extend', extendSourcePath: '/clips/walk.mp4', extendFrames: 48, missingSourceVideo: true,
      scene: 'My street.', prompt: 'She waves.', enhancedScene: 'A wet street at night.', enhancedPrompt: 'She waves slowly.',
    });
    expect(enhancedStaleReason(draft.state)).toBeNull();
  });

  it("keeps the blank text of the server's Reuse params through the inspection of the source", () => {
    draft.loadFromUrl(reuseFixture.blank_text);
    draft.applyExtendSource(sceneSource({ path: '/clips/walk.mp4' }));
    expect(draft.state).toMatchObject({ scene: '', prompt: '', enhancedScene: '', enhancedPrompt: '', extendSourcePath: '/clips/walk.mp4' });
  });

  it("loads the server's Reuse params of an extension that continued from a point", () => {
    draft.loadFromUrl(reuseFixture.extend_from);
    expect(draft.state).toMatchObject({ workflow: 'extend', extendSourcePath: '/clips/walk.mp4', extendFrom: 3.208333, extendFrames: 48 });
  });

  describe('the point to extend from', () => {
    // A 144-frame clip at 24 fps with a track that ended at 10 s.
    const long = (overrides: Partial<ReturnType<typeof sceneSource>> = {}) =>
      sceneSource({ path: SOURCE, frame_count: 144, fps: 24, video_start: 0, audio_track_path: '/audio/song.mp3', audio_track_end: 10, ...overrides });

    it('is set by a link that names it, and read as seconds', () => {
      draft.loadFromUrl({ workflow: 'extend', source_video: SOURCE, extend_from: '3.5' });
      expect(draft.state.extendFrom).toBe(3.5);
    });

    it('is cleared by an Extend that names none', () => {
      draft.loadFromUrl({ workflow: 'extend', source_video: SOURCE, extend_from: '3.5' });
      draft.loadFromUrl(extendParams(extendAsset({ scene: 'A street.' })));
      expect(draft.state.extendFrom).toBeNull();
    });

    it('is not read from a value that is not a number of seconds', () => {
      draft.loadFromUrl({ workflow: 'extend', source_video: SOURCE, extend_from: 'soon' });
      expect(draft.state.extendFrom).toBeNull();
      draft.loadFromUrl({ workflow: 'extend', source_video: SOURCE, extend_from: '-1' });
      expect(draft.state.extendFrom).toBeNull();
    });

    it('belongs to the source: choosing another source resets it, choosing the same one keeps it', () => {
      draft.loadFromUrl({ workflow: 'extend', source_video: SOURCE, extend_from: '3.5' });
      draft.setExtendSourcePath(SOURCE);
      expect(draft.state.extendFrom).toBe(3.5);
      draft.setExtendSourcePath('/clips/other.mp4');
      expect(draft.state.extendFrom).toBeNull();
    });

    it('is kept by Retry and Load settings, which carry the recorded value', () => {
      const { params, patch } = jobSettingsPrefill({ mode: 'video', workflow: 'extend', source_video: SOURCE, extend_frames: '48', extend_from: '3.208333', scene: 'A street.', prompt: '' });
      draft.loadFromUrl(params);
      draft.patch(patch);
      expect(draft.state.extendFrom).toBe(3.208333);
    });

    it('is moved to a point the source can use when the source is described', () => {
      draft.loadFromUrl({ workflow: 'extend', source_video: SOURCE, extend_from: '0.5' });
      draft.applyExtendSource(long());
      expect(draft.state.extendFrom).toBeCloseTo(24 / 24, 6);
      draft.setExtendFrom(99);
      draft.applyExtendSource(long());
      expect(draft.state.extendFrom).toBeNull();
    });

    it('starts the first inspection\'s track at the moved Start at', () => {
      draft.loadFromUrl({ workflow: 'extend', source_video: SOURCE, extend_from: String(76 / 24) });
      draft.applyExtendSource(long());
      expect(draft.state).toMatchObject({ extendAudioTrackPath: '/audio/song.mp3' });
      expect(draft.state.extendAudioTrackStart).toBeCloseTo(10 - 67 / 24, 6);
    });

    it('moves Start at with the handle while the track is still the source\'s own', () => {
      draft.loadFromUrl(extendParams(extendAsset({ scene: 'A street.' })));
      draft.applyExtendSource(long());
      expect(draft.state.extendAudioTrackStart).toBe(10);

      draft.setExtendFrom(76 / 24);
      expect(draft.state.extendAudioTrackStart).toBeCloseTo(10 - 67 / 24, 6);
      draft.setExtendFrom(100 / 24);
      expect(draft.state.extendAudioTrackStart).toBeCloseTo(10 - 43 / 24, 6);
      draft.setExtendFrom(null);
      expect(draft.state.extendAudioTrackStart).toBe(10);
    });

    it('floors the moved Start at at 0', () => {
      draft.loadFromUrl(extendParams(extendAsset({ scene: 'A street.' })));
      draft.applyExtendSource(long({ audio_track_end: 1 }));
      draft.setExtendFrom(76 / 24);
      expect(draft.state.extendAudioTrackStart).toBe(0);
    });

    it('leaves a track the person chose where they put it', () => {
      draft.loadFromUrl(extendParams(extendAsset({ scene: 'A street.' })));
      draft.applyExtendSource(long());
      draft.patch({ extendAudioTrackStart: 42 });
      draft.setExtendFrom(76 / 24);
      expect(draft.state.extendAudioTrackStart).toBe(42);
    });

    it('keeps a Reset\'s Start at where the point put it', () => {
      draft.loadFromUrl(extendParams(extendAsset({ scene: 'A street.' })));
      draft.applyExtendSource(long());
      draft.setExtendFrom(76 / 24);
      expect(draft.state.extendPrefill?.audioTrack?.start).toBeCloseTo(10 - 67 / 24, 6);
    });

    it('moves Start at when an Extend from a point arrives for a source that is already described', () => {
      draft.loadFromUrl(extendParams(extendAsset({ scene: 'A street.' })), null, { replaceSlots: true });
      draft.applyExtendSource(long(), null, { replaceText: true });
      expect(draft.state.extendAudioTrackStart).toBe(10);

      draft.loadFromUrl({ ...extendParams(extendAsset({ scene: 'A street.' })), extend_from: String(76 / 24) }, null, { replaceSlots: true });
      draft.applyExtendSource(long(), null, { replaceText: true });

      expect(draft.state.extendAudioTrackStart).toBeCloseTo(10 - 67 / 24, 6);
      expect(draft.state.extendPrefill?.audioTrack?.start).toBeCloseTo(10 - 67 / 24, 6);
    });
  });

  it('persists the chosen path and not the source facts', () => {
    draft.loadFromUrl(extendParams(extendAsset({ scene: 'A street.' })));
    draft.applyExtendSource(sceneSource());
    const saved = JSON.parse(localStorage.getItem('ziv-workspace-draft-v1') ?? '{}') as Record<string, unknown>;
    expect(saved.extendSourcePath).toBe(SOURCE);
    expect(saved.version).toBe(3);
    expect(JSON.stringify(saved)).not.toContain('duration_seconds');
  });
});
