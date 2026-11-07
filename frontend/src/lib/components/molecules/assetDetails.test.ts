import { describe, expect, it } from 'vitest';

import type { GalleryAsset } from '$lib/types';

import { AUDIO_SETTINGS, makeAudioAsset, makeSpeechAsset } from '../../../test-utils/audio';
import { assetDetailSections, describeUpscale } from './assetDetails';

describe('describeUpscale', () => {
  it('leads with the factor and names the method, model and softness of a SeedVR2 upscale', () => {
    const text = describeUpscale({ factor: 2, method: 'seedvr2', model: 'seedvr2-7b', quantize: 8, softness: 0.2 });
    expect(text).toBe('2× · SeedVR2 · seedvr2-7b q8 · softness 0.2');
  });

  it('names an LTX video upscale with its model and steps', () => {
    expect(describeUpscale({ factor: 2, method: 'ltx', model: 'ltx-4', steps: 2, seed: 42 })).toBe('2× · LTX · ltx-4 · 2 steps');
  });

  it('names a creative upscale and keeps its settings', () => {
    expect(describeUpscale({ factor: 4, method: 'creative', denoise: 0.4, steps: 3 })).toBe('4× · creative · denoise 0.4 · 3 steps');
  });

  it('describes files recorded without a method by the factor and settings alone', () => {
    expect(describeUpscale({ factor: 2, denoise: 0.4 })).toBe('2× · denoise 0.4');
    expect(describeUpscale(undefined)).toBe('');
  });
});

describe('assetDetailSections', () => {
  const clip = (overrides: Partial<GalleryAsset> = {}): GalleryAsset => ({
    id: 'out/c.mp4', url: '/media/out/c.mp4', thumbnail_url: '/media/out/c.mp4', filename: 'c.mp4', created_at: '2026-10-03T12:00:00Z',
    workflow: 'img2vid', prompt: 'a fox', model: 'h3', media_type: 'video', reuse_workspace_url: '', ...overrides,
  });
  const labels = (asset: GalleryAsset) => assetDetailSections(asset).generation.map((fact) => fact.label);

  it('lists the last frame after the first frame, by file name', () => {
    const asset = clip({ image_path: '/in/start.png', end_image_path: '/in/end.png' });
    const generation = assetDetailSections(asset).generation;
    expect(labels(asset).indexOf('Last frame')).toBe(labels(asset).indexOf('First frame') + 1);
    expect(generation.find((fact) => fact.label === 'Last frame')?.value).toBe('end.png');
  });

  it('lists the references of an image by file name, in order, apart from its start image', () => {
    const asset = clip({ media_type: 'image', workflow: 'img2img', image_path: '/in/start.png', reference_images: ['/in/woman.png', '/in/jacket.png'] });
    const generation = assetDetailSections(asset).generation;

    expect(generation.find((fact) => fact.label === 'Start image')?.value).toBe('start.png');
    expect(generation.find((fact) => fact.label === 'References')?.value).toBe('woman.png, jacket.png');
    expect(labels(clip({ media_type: 'image' }))).not.toContain('References');
  });

  it('lists a transparent background in the post-processing facts, and only when the image has one', () => {
    const image = (generation: object) => clip({ media_type: 'image', details: { generation } } as Partial<GalleryAsset>);
    const facts = (asset: GalleryAsset) => assetDetailSections(asset).postProcessing;
    expect(facts(image({ transparent: true }))).toContainEqual({ label: 'Transparent background', value: 'On' });
    expect(facts(image({ sharpen: 0.5 })).map((fact) => fact.label)).not.toContain('Transparent background');
    expect(facts(clip()).map((fact) => fact.label)).not.toContain('Transparent background');
  });

  it('leaves the last frame out of a clip without one', () => {
    expect(labels(clip({ image_path: '/in/start.png', end_image_path: null }))).not.toContain('Last frame');
  });

  it('lists the audio track of a clip that followed one, with the span of the track it used', () => {
    const generation = assetDetailSections(clip({ audio_track_path: '/music/song.mp3', audio_track_start: 90, audio_track_end: 94.2 })).generation;
    expect(generation.find((fact) => fact.label === 'Audio track')?.value).toBe('song.mp3 · 1:30–1:34.2');
  });

  it('names only where the span starts when the track end was not recorded, and leaves the row out of other clips', () => {
    const generation = assetDetailSections(clip({ audio_track_path: '/music/song.mp3', audio_track_start: 12, audio_track_end: null })).generation;
    expect(generation.find((fact) => fact.label === 'Audio track')?.value).toBe('song.mp3 · from 0:12');
    expect(labels(clip())).not.toContain('Audio track');
  });
});

describe('assetDetailSections for an audio take', () => {
  const facts = (asset: GalleryAsset, vocabulary = AUDIO_SETTINGS): Record<string, string> =>
    Object.fromEntries(assetDetailSections(asset, undefined, vocabulary).generation.map((fact) => [fact.label, fact.value]));
  const excited = AUDIO_SETTINGS.vibes.find((vibe) => vibe.id === 'excited-host')!;

  it('joins the tempo and the key of a song in one fact, with the key named as the client names it', () => {
    expect(facts(makeAudioAsset({ tempo: 92, key: 'Am' }))['Tempo · Key']).toBe('92 BPM · A minor');
    expect(facts(makeAudioAsset({ tempo: 120, key: 'F#m' }))['Tempo · Key']).toBe('120 BPM · F♯ minor');
  });

  it('shows only the part of Tempo · Key the song set, and nothing for a song that set neither', () => {
    expect(facts(makeAudioAsset({ tempo: 92, key: null }))).toMatchObject({ Tempo: '92 BPM' });
    expect(facts(makeAudioAsset({ tempo: null, key: 'C' }))).toMatchObject({ Key: 'C major' });
    expect(Object.keys(facts(makeAudioAsset({ tempo: null, key: null })))).not.toEqual(expect.arrayContaining(['Tempo', 'Key', 'Tempo · Key']));
  });

  it('names the Vibe the delivery of a speech take matches, and Custom when it matches none', () => {
    expect(facts(makeSpeechAsset({ ...excited.delivery }))['Delivery']).toBe(excited.label);
    expect(facts(makeSpeechAsset({ ...excited.delivery, pitch: 'low' }))['Delivery']).toBe('Custom');
  });

  it('shows no Delivery for a speech take that recorded none, or without the vocabulary to name it', () => {
    expect(facts(makeSpeechAsset())).not.toHaveProperty('Delivery');
    expect(assetDetailSections(makeSpeechAsset({ ...excited.delivery })).generation.map((fact) => fact.label)).not.toContain('Delivery');
  });

  it('names the sound type of a sound by its label, and shows none for a take without one', () => {
    const label = AUDIO_SETTINGS.sound_types.find((type) => type.id === 'ambience')!.label;
    expect(facts(makeAudioAsset({ kind: 'sound', sound_type: 'ambience' }))['Sound type']).toBe(label);
    expect(facts(makeAudioAsset({ kind: 'sound', sound_type: null }))).not.toHaveProperty('Sound type');
  });
});
