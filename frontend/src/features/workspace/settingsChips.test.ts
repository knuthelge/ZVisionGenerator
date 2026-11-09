import { describe, expect, it } from 'vitest';

import { AUDIO_FIELD_DEFAULTS } from '$lib/state/audioDraft';
import type { DraftState } from '$lib/types';
import type { WorkspaceCapabilities } from './capabilities';
import type { ImagesSummary } from './imagesSummary';
import { audioChips, pendingChips, settingsChips, type AudioChipFacts, type ChipFacts } from './settingsChips';

const imageCaps = {
  isImageMode: true, showFrameCount: false, showExtendLength: false, showSteps: true, showGuidance: true, showI2IStrength: false, showSeed: true,
  showRuns: true, showRefImage: false, showSourceVideo: false, showDimensions: true, showAudio: false, showLowMemory: false, showVideoUpscale: false, showImageUpscale: true,
} as WorkspaceCapabilities;

const state = {
  steps: 32, guidance: 4, referenceImageStrength: 0.75, seed: null, runs: 1, audio: true, lowMemory: false, videoUpscaleEnabled: false,
  upscaleEnabled: false, upscaleFactor: 2, missingSourceVideo: false, postprocessTransparent: false,
} as DraftState;

function facts(overrides: Partial<ChipFacts> = {}): ChipFacts {
  return {
    caps: imageCaps, state, changed: () => false, width: 1024, height: 1024, seconds: null, extendSeconds: null,
    loraNames: [], videoKind: null, modelLabel: 'Z-Image Turbo', modelDownloaded: true, sourceName: null, images: null, track: null, memoryWarn: false,
    presetName: null, presetSlug: null, presetChanged: false,
    ...overrides,
  };
}

describe('settingsChips', () => {
  it('offers placeholder chips in the workflow order, with no value, while the context loads', () => {
    const image = pendingChips('txt2img');
    expect(image.map((chip) => chip.key)).toEqual(['preset', 'steps', 'guidance', 'seed', 'runs', 'images', 'size', 'loras', 'upscale', 'model']);
    expect(image.every((chip) => chip.value === '—' && !chip.changed)).toBe(true);
    expect(pendingChips('txt2vid').map((chip) => chip.key)).toEqual(['kind', 'preset', 'seconds', 'steps', 'guidance', 'seed', 'images', 'track', 'size', 'audio', 'loras', 'model']);
    expect(pendingChips('extend').map((chip) => chip.key).slice(0, 3)).toEqual(['kind', 'preset', 'extendSeconds']);
    expect(pendingChips('extend').map((chip) => chip.key)).not.toContain('images');
  });

  it('carries what the LoRA files are doing on the LoRA chip only', () => {
    const chips = settingsChips(facts({ loraNames: ['style'], loraDownload: { kind: 'downloading', percent: 40 } }));

    expect(chips.find((chip) => chip.key === 'loras')?.download).toEqual({ kind: 'downloading', percent: 40 });
    expect(chips.filter((chip) => chip.download).map((chip) => chip.key)).toEqual(['loras']);
  });

  it('orders an image workflow by how often each setting changes, model last', () => {
    expect(settingsChips(facts()).map((chip) => chip.key)).toEqual(['preset', 'steps', 'guidance', 'seed', 'runs', 'size', 'loras', 'upscale', 'model']);
  });

  it('puts the clip length first and audio after size for video, without the image upscale', () => {
    const caps = { ...imageCaps, isImageMode: false, showFrameCount: true, showGuidance: false, showImageUpscale: false, showAudio: true, showLowMemory: true, showVideoUpscale: true } as WorkspaceCapabilities;
    expect(settingsChips(facts({ caps, seconds: 5 })).map((chip) => chip.key)).toEqual(['preset', 'seconds', 'steps', 'seed', 'runs', 'size', 'audio', 'loras', 'model']);
  });

  describe('the Kind chip of Video', () => {
    const videoCaps = { ...imageCaps, isImageMode: false, showFrameCount: true, showImageUpscale: false } as WorkspaceCapabilities;
    const extendCaps = { ...videoCaps, showFrameCount: false, showExtendLength: true, showSourceVideo: true } as WorkspaceCapabilities;

    it('comes first and names the open kind, when the machine offers more than one', () => {
      const chips = settingsChips(facts({ caps: videoCaps, seconds: 5, videoKind: 'new' }));
      expect(chips[0]).toMatchObject({ key: 'kind', value: 'New' });
      expect(settingsChips(facts({ caps: extendCaps, extendSeconds: 2, videoKind: 'extend' }))[0]).toMatchObject({ key: 'kind', value: 'Extend' });
      expect(settingsChips(facts({ caps: videoCaps, seconds: 5 })).map((chip) => chip.key)).not.toContain('kind');
    });

    it('calls the extension length Length, so no two chips are named Extend', () => {
      const chips = settingsChips(facts({ caps: extendCaps, extendSeconds: 2, videoKind: 'extend' }));
      expect(chips.find((chip) => chip.key === 'extendSeconds')).toMatchObject({ label: 'Length', value: '+2 s' });
    });
  });

  describe('the Track chip', () => {
    const track = (f: Partial<ChipFacts>) => settingsChips(facts(f)).find((chip) => chip.key === 'track');
    const videoCaps = { ...imageCaps, isImageMode: false, showFrameCount: true, showImageUpscale: false, showSourceVideo: false } as WorkspaceCapabilities;

    it('comes right after Images, and after Source in Extend', () => {
      const images: ImagesSummary = { text: 'None', tone: 'muted' };
      const keys = (f: Partial<ChipFacts>) => settingsChips(facts({ caps: videoCaps, seconds: 5, ...f })).map((chip) => chip.key);

      expect(keys({ images, track: { set: false, start: 0, gone: false } })).toEqual(['preset', 'seconds', 'steps', 'guidance', 'seed', 'runs', 'images', 'track', 'size', 'loras', 'model']);
      const extend = { ...videoCaps, showSourceVideo: true, showFrameCount: false } as WorkspaceCapabilities;
      expect(settingsChips(facts({ caps: extend, track: { set: false, start: 0, gone: false } })).map((chip) => chip.key).slice(-5)).toEqual(['source', 'track', 'size', 'loras', 'model']);
    });

    it('is left out where the section is not offered', () => {
      expect(track({ track: null })).toBeUndefined();
    });

    it('reads none unlit, on at 0, or where the clip starts, and warns when an uploaded track did not come back', () => {
      expect(track({ track: { set: false, start: 0, gone: false } })).toMatchObject({ label: 'Track', value: 'none', on: false, warn: false });
      expect(track({ track: { set: true, start: 0, gone: false } })).toMatchObject({ value: 'on', on: true, warn: false });
      expect(track({ track: { set: true, start: 90, gone: false } })).toMatchObject({ value: '1:30', on: true });
      expect(track({ track: { set: true, start: 0, gone: true } })).toMatchObject({ warn: true });
    });
  });

  describe('the Images chip', () => {
    const summary = (overrides: Partial<ImagesSummary> = {}): ImagesSummary => ({ text: 'pond.png · 2 references', tone: 'active', ...overrides });
    const images = (f: Partial<ChipFacts>) => settingsChips(facts(f)).find((chip) => chip.key === 'images');

    it('is one chip for every image, between Batch and Size, with the section summary as its value', () => {
      const keys = settingsChips(facts({ images: summary() })).map((chip) => chip.key);

      expect(keys).toEqual(['preset', 'steps', 'guidance', 'seed', 'runs', 'images', 'size', 'loras', 'upscale', 'model']);
      expect(images({ images: summary() })).toMatchObject({ label: 'Images', value: 'pond.png · 2 references', on: true });
    });

    it('is left out where the section is not offered', () => {
      expect(images({ images: null })).toBeUndefined();
    });

    it('reads None unlit, and warns when an image the run uses is gone', () => {
      expect(images({ images: summary({ text: 'None', tone: 'muted' }) })).toMatchObject({ value: 'None', on: false, warn: false });
      expect(images({ images: summary({ tone: 'error' }) })).toMatchObject({ on: false, warn: true });
    });

    it('carries the changed flag of Strength', () => {
      expect(images({ images: summary(), changed: (chip) => chip === 'images' })?.changed).toBe(true);
      expect(images({ images: summary(), changed: () => false })?.changed).toBe(false);
    });

    it('replaces the old per-image chips', () => {
      const everything = { ...imageCaps, showRefImage: true, takesLastFrame: true, maxReferences: 4, showI2IStrength: true } as WorkspaceCapabilities;
      const keys = settingsChips(facts({ caps: everything, images: summary() })).map((chip) => chip.key as string);

      for (const key of ['strength', 'reference', 'references', 'endReference']) expect(keys).not.toContain(key);
    });
  });

  it("a tab's two workflows share placeholders", () => {
    const keys = (workflow: Parameters<typeof pendingChips>[0]) => pendingChips(workflow).map((chip) => chip.key);

    expect(keys('img2img')).toEqual(keys('txt2img'));
    expect(keys('img2vid')).toEqual(keys('txt2vid'));
  });

  it('adds Source and Extend for an extension', () => {
    const extend = { ...imageCaps, isImageMode: false, showExtendLength: true, showSourceVideo: true, showGuidance: false, showImageUpscale: false, showDimensions: false } as WorkspaceCapabilities;
    const keys = settingsChips(facts({ caps: extend, extendSeconds: 3, sourceName: 'harbour.mp4' })).map((chip) => chip.key);
    expect(keys.slice(0, 2)).toEqual(['preset', 'extendSeconds']);
    expect(keys).toContain('source');
  });

  it('shows values, the changed flag and the on state', () => {
    const chips = settingsChips(facts({ state: { ...state, seed: 4242, upscaleEnabled: true, upscaleFactor: 4 }, changed: (chip) => chip === 'seed', loraNames: ['detail-tweaker'] }));
    const byKey = Object.fromEntries(chips.map((chip) => [chip.key, chip]));
    expect(byKey.seed).toMatchObject({ value: '4242', changed: true, on: true });
    expect(byKey.upscale).toMatchObject({ value: '4×', on: true });
    expect(byKey.loras).toMatchObject({ value: 'detail-tweaker', on: true });
    expect(byKey.size.value).toBe('1024×1024');
    expect(byKey.runs.value).toBe('×1');
  });

  describe('the Transparent chip', () => {
    const caps = { ...imageCaps, showPostprocessTransparent: true } as WorkspaceCapabilities;
    const transparent = (f: Partial<ChipFacts>) => settingsChips(facts({ caps, ...f })).find((chip) => chip.key === 'transparent');

    it('shows only while the switch is on, between Upscale and Model', () => {
      expect(transparent({})).toBeUndefined();
      const on = { ...state, postprocessTransparent: true } as DraftState;
      expect(transparent({ state: on })).toMatchObject({ label: 'Transparent', value: 'on', on: true });
      expect(settingsChips(facts({ caps, state: on })).map((chip) => chip.key).slice(-3)).toEqual(['upscale', 'transparent', 'model']);
    });

    it('stays, reading off, while its popover is open', () => {
      expect(transparent({ openKey: 'transparent' })).toMatchObject({ value: 'off', on: false });
      expect(transparent({ openKey: 'steps' })).toBeUndefined();
    });

    it('carries the changed flag', () => {
      const on = { ...state, postprocessTransparent: true } as DraftState;
      expect(transparent({ state: on, changed: (chip) => chip === 'transparent' })?.changed).toBe(true);
    });

    it('is never offered where the workflow has no Transparent control, even when the draft has it on', () => {
      const on = { ...state, postprocessTransparent: true } as DraftState;
      expect(settingsChips(facts({ state: on, openKey: 'transparent' })).map((chip) => chip.key)).not.toContain('transparent');
    });
  });

  it('puts the preset first with its name, or none, and the changed flag', () => {
    expect(settingsChips(facts())[0]).toMatchObject({ key: 'preset', applied: '', changed: false, on: false });
    const applied = settingsChips(facts({ presetName: 'Soft portrait', presetSlug: 'soft-portrait', presetChanged: true }))[0];
    expect(applied).toMatchObject({ key: 'preset', applied: 'soft-portrait', changed: true, on: true });
  });

  it('warns on a model that is not downloaded and a clip that may not fit', () => {
    const chips = settingsChips(facts({ modelDownloaded: false, caps: { ...imageCaps, showFrameCount: true } as WorkspaceCapabilities, seconds: 8, memoryWarn: true }));
    expect(chips.find((chip) => chip.key === 'model')?.warn).toBe(true);
    expect(chips.find((chip) => chip.key === 'seconds')?.warn).toBe(true);
  });
});

describe('audioChips', () => {
  const audioState = { workflow: 'speech', audioSeconds: 12, audioTempo: null, seed: null, runs: 1, audioDescribe: AUDIO_FIELD_DEFAULTS.audioDescribe } as DraftState;
  const voice = { summary: { text: 'Ryan', tone: 'muted' as const }, cloned: false, chip: 'Ryan', pending: 0 };

  function audioFacts(overrides: Partial<AudioChipFacts> = {}): AudioChipFacts {
    return { kind: 'speech', state: audioState, modelLabel: 'Qwen3-TTS 1.7B', modelDownloaded: true, voice, voiceBlocks: false, changed: () => false, delivery: { shown: false, summary: 'Natural' }, showLength: false, seconds: null, lengthAuto: false, soundType: { shown: false, label: 'Auto' }, tempo: { shown: false }, key: { shown: false, name: null }, showSeed: true, showRuns: true, ...overrides };
  }

  it('lists Kind, Model, Voice, Delivery, Seed and Batch for Speech, with no preset, size or LoRA', () => {
    const chips = audioChips(audioFacts({ delivery: { shown: true, summary: 'Natural' } }));

    expect(chips.map((chip) => chip.key)).toEqual(['kind', 'model', 'voice', 'delivery', 'seed', 'runs']);
    expect(chips[0]).toMatchObject({ value: 'Speech', icon: 'speech' });
    expect(chips[2]).toMatchObject({ label: 'Voice', value: 'Ryan', on: false });
  });

  it('writes the Vibe, Natural or Custom on the Delivery chip, marks it changed and leaves it out for a recording', () => {
    const chip = (overrides: Partial<AudioChipFacts> = {}) => audioChips(audioFacts({ delivery: { shown: true, summary: 'Natural' }, ...overrides })).find((item) => item.key === 'delivery');

    expect(chip()).toMatchObject({ label: 'Delivery', value: 'Natural', changed: false, on: false });
    expect(chip({ delivery: { shown: true, summary: 'Excited host' }, changed: (key) => key === 'delivery' })).toMatchObject({ value: 'Excited host', changed: true, on: true });
    expect(chip({ delivery: { shown: false, summary: 'Natural' } })).toBeUndefined();
  });

  it('puts Type and Length in place of Voice for Sound, and Tempo and Key for Music', () => {
    const sound = audioFacts({ kind: 'sound', voice: null, showLength: true, seconds: 12, soundType: { shown: true, label: 'Auto' } });
    expect(audioChips(sound).map((chip) => chip.key)).toEqual(['kind', 'model', 'soundType', 'length', 'seed', 'runs']);
    expect(audioChips(sound).find((chip) => chip.key === 'length')?.value).toBe('12 s');
    const music = audioFacts({ kind: 'music', voice: null, tempo: { shown: true }, key: { shown: true, name: null } });
    expect(audioChips(music).map((chip) => chip.key)).toEqual(['kind', 'model', 'tempo', 'key', 'seed', 'runs']);
  });

  it('writes the Sound Type, and a length on Auto with the number it will use, and marks a locked one changed', () => {
    const chip = (key: string, overrides: Partial<AudioChipFacts> = {}) =>
      audioChips(audioFacts({ kind: 'sound', voice: null, showLength: true, seconds: 5, lengthAuto: true, soundType: { shown: true, label: 'Effect' }, ...overrides })).find((item) => item.key === key);

    expect(chip('soundType')).toMatchObject({ label: 'Type', value: 'Effect', changed: false, on: false });
    expect(chip('length')).toMatchObject({ label: 'Length', value: 'auto ≈ 5 s', changed: false, on: false });
    const locked = { seconds: 8, lengthAuto: false, changed: (key: string) => key === 'length' || key === 'soundType' };
    expect(chip('length', locked)).toMatchObject({ value: '8 s', changed: true, on: true });
    expect(chip('soundType', locked)).toMatchObject({ changed: true, on: true });
  });

  it('lists a song\'s Length after its Key on a model that sets it', () => {
    const facts = audioFacts({ kind: 'music', voice: null, tempo: { shown: true }, key: { shown: true, name: null }, showLength: true, seconds: 30 });

    expect(audioChips(facts).map((chip) => chip.key)).toEqual(['kind', 'model', 'tempo', 'key', 'length', 'seed', 'runs']);
    expect(audioChips(facts).find((chip) => chip.key === 'length')).toMatchObject({ label: 'Length', value: '30 s' });
  });

  it('writes Tempo and Key as Auto until they are set, the key in the width of a chip, and marks them changed', () => {
    const chip = (key: string, overrides: Partial<AudioChipFacts> = {}) => audioChips(audioFacts({ kind: 'music', voice: null, tempo: { shown: true }, key: { shown: true, name: null }, ...overrides })).find((item) => item.key === key);

    expect(chip('tempo')).toMatchObject({ label: 'Tempo', value: 'Auto', changed: false });
    expect(chip('key')).toMatchObject({ label: 'Key', value: 'Auto', changed: false });
    const set = { state: { ...audioState, audioTempo: 92 } as DraftState, key: { shown: true, name: 'A minor' }, changed: (key: string) => key === 'tempo' || key === 'key' };
    expect(chip('tempo', set)).toMatchObject({ value: '92', changed: true });
    expect(chip('key', set)).toMatchObject({ value: 'A min', changed: true });
  });

  it('marks a model that is not downloaded, a followed recording, and a recording that is wrong', () => {
    expect(audioChips(audioFacts({ modelDownloaded: false })).find((chip) => chip.key === 'model')?.warn).toBe(true);
    const cloned = { summary: { text: 'me.wav', tone: 'active' as const }, cloned: true, chip: 'me.wav', pending: 0 };
    expect(audioChips(audioFacts({ voice: cloned })).find((chip) => chip.key === 'voice')).toMatchObject({ value: 'me.wav', on: true, warn: false });
    const wrong = { summary: { text: 'Use a recording of 3 to 30 seconds', tone: 'error' as const }, cloned: true, chip: 'me.wav', pending: 0 };
    expect(audioChips(audioFacts({ voice: wrong })).find((chip) => chip.key === 'voice')?.warn).toBe(true);
  });

  it('counts the designed voices waiting to be picked on the Voice chip, and warns on a deleted voice', () => {
    const waiting = { ...voice, pending: 3 };
    expect(audioChips(audioFacts({ voice: waiting })).find((chip) => chip.key === 'voice')).toMatchObject({ value: 'Ryan', count: 3 });
    expect(audioChips(audioFacts({ voice })).find((chip) => chip.key === 'voice')).not.toHaveProperty('count');
    const deleted = { summary: { text: 'Voice deleted', tone: 'error' as const }, cloned: true, chip: 'deleted', pending: 0 };
    expect(audioChips(audioFacts({ voice: deleted })).find((chip) => chip.key === 'voice')).toMatchObject({ value: 'deleted', warn: true });
  });

  it('marks the Voice chip changed when the voice differs from its default', () => {
    const chip = (changedKeys: string[]) => audioChips(audioFacts({ changed: (key) => changedKeys.includes(key) })).find((item) => item.key === 'voice');

    expect(chip(['voice'])?.changed).toBe(true);
    expect(chip([])?.changed).toBe(false);
  });

  it('warns on the Voice chip while the voice holds Generate back', () => {
    const cloned = { summary: { text: 'me.wav', tone: 'active' as const }, cloned: true, chip: 'me.wav', pending: 0 };

    expect(audioChips(audioFacts({ voice: cloned, voiceBlocks: true })).find((chip) => chip.key === 'voice')?.warn).toBe(true);
  });
});
