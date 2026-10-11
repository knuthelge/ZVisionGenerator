import { beforeEach, describe, expect, it } from 'vitest';

import type { GalleryAsset } from '$lib/types';

import { extendParams } from './assetActions';
import { draft } from './draft.svelte';
import { isOriginalLegacyPrompt } from './extend';
import { legacySource, sceneSource, sourceInfo } from './extend.fixtures';
import reuseFixture from '../../../../tests/fixtures/extend_reuse_params.json';
import { jobSettingsPrefill } from './jobSettings';
import { isEnhancedStale } from './promptEnhance';

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
    expect(isEnhancedStale(draft.state)).toBe(false);
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
    expect(isEnhancedStale(draft.state)).toBe(false);
  });

  // tests/fixtures/extend_reuse_params.json is also checked by pytest against the gallery's `reuse_params` output.
  it("loads the server's Reuse params of an enhanced extension whose source moved", () => {
    draft.loadFromUrl(reuseFixture.enhanced_source_moved);
    expect(draft.state).toMatchObject({
      workflow: 'extend', extendSourcePath: '/clips/walk.mp4', extendFrames: 48, missingSourceVideo: true,
      scene: 'My street.', prompt: 'She waves.', enhancedScene: 'A wet street at night.', enhancedPrompt: 'She waves slowly.',
    });
    expect(isEnhancedStale(draft.state)).toBe(false);
  });

  it("keeps the blank text of the server's Reuse params through the inspection of the source", () => {
    draft.loadFromUrl(reuseFixture.blank_text);
    draft.applyExtendSource(sceneSource({ path: '/clips/walk.mp4' }));
    expect(draft.state).toMatchObject({ scene: '', prompt: '', enhancedScene: '', enhancedPrompt: '', extendSourcePath: '/clips/walk.mp4' });
  });

  it('persists the chosen path and not the source facts', () => {
    draft.loadFromUrl(extendParams(extendAsset({ scene: 'A street.' })));
    draft.applyExtendSource(sceneSource());
    const saved = JSON.parse(localStorage.getItem('ziv-workspace-draft-v1') ?? '{}') as Record<string, unknown>;
    expect(saved.extendSourcePath).toBe(SOURCE);
    expect(saved.version).toBe(3);
    expect(JSON.stringify(saved)).not.toContain('media_url');
    expect(JSON.stringify(saved)).not.toContain('duration_seconds');
  });
});
