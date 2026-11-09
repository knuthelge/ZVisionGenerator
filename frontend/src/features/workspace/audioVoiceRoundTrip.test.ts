import { beforeEach, describe, expect, it, vi } from 'vitest';

import { AUDIO_MODELS, QWEN_VOICES, withAudio } from '../../test-utils/audio';
import { makeContext } from '../../test-utils/workspace';
import { formSettings } from '$lib/state/cliCommand';
import { defaultsFor, draft } from '$lib/state/draft.svelte';
import { jobSettingsPrefill } from '$lib/state/jobSettings';
import { voiceConsent } from '$lib/state/voiceConsent.svelte';
import { voicesStore } from '$lib/state/voices.svelte';
import type { AudioModelOption, DraftState, SavedVoice, WorkspaceContext } from '$lib/types';
import { audioFormData, chooseVoice, currentAudioView } from './audioView';
import { changedRowCount, rowChanged, rowDefaults } from './settingRows';
import { workspaceCapabilities } from './capabilities';

const api = vi.hoisted(() => ({ listVoices: vi.fn<() => Promise<SavedVoice[]>>(), deleteVoice: vi.fn<(slug: string) => Promise<{ status: string }>>() }));

vi.mock('$lib/api/voices', async (importOriginal) => ({ ...(await importOriginal<typeof import('$lib/api/voices')>()), ...api }));

const voice = (slug: string, name: string): SavedVoice => ({ slug, name, text: `${name} says hello.`, origin: 'recording', seconds: 5, description: null, created_at: '2026-11-08T10:00:00Z' });

/** The library holds *voices* from the next read on. */
async function library(...voices: SavedVoice[]): Promise<void> {
  api.listVoices.mockResolvedValue(voices);
  await voicesStore.load();
}

const ctx = withAudio(makeContext());

function openSpeech(context: WorkspaceContext = ctx): void {
  draft.hydrateFromContext(context, null);
  draft.setTab('audio');
  draft.onTabChange(context);
  draft.setAudioKind('speech', context);
}

const sent = (context: WorkspaceContext = ctx): FormData => audioFormData(currentAudioView(context), draft.state);

beforeEach(() => {
  localStorage.clear();
  draft.reset();
});

describe('the built-in voice, from a Reuse to the next Generate', () => {
  it('shows and sends the voice a Reuse of a take names, and the default one as no change', () => {
    openSpeech();

    draft.loadFromUrl({ workflow: 'speech', model: 'qwen3_tts', prompt: 'Welcome', speaker: 'vivian', seed: '7' }, ctx, { exact: true });
    expect(currentAudioView(ctx).voice).toMatchObject({ speaker: 'vivian', speakerLabel: 'Vivian', choice: 'vivian', chip: 'Vivian' });
    expect(sent().get('speaker')).toBe('vivian');
    expect(rowChanged('voice', draft.state, defaultsFor(ctx, draft.state, null))).toBe(true);

    draft.loadFromUrl({ workflow: 'speech', model: 'qwen3_tts', prompt: 'Welcome', speaker: 'ryan' }, ctx, { exact: true });
    expect(sent().get('speaker')).toBe('ryan');
    expect(rowChanged('voice', draft.state, defaultsFor(ctx, draft.state, null))).toBe(false);
  });

  it('follows a recording again when the take had one, with no speaker sent', () => {
    openSpeech();
    draft.loadFromUrl({ workflow: 'speech', model: 'qwen3_tts', prompt: 'Welcome', speaker: 'vivian' }, ctx, { exact: true });

    draft.loadFromUrl({ workflow: 'speech', model: 'qwen3_tts', prompt: 'Welcome', voice_path: '/voices/me.wav', voice_text: 'Hi, it is me.' }, ctx, { exact: true });

    expect(draft.state).toMatchObject({ voiceMode: 'recording', voicePath: '/voices/me.wav', voiceText: 'Hi, it is me.' });
    const form = sent();
    expect(form.has('speaker')).toBe(false);
    expect(form.get('voice_path')).toBe('/voices/me.wav');
  });

  it('loads the voice back from a speech job the same way a Reuse does', () => {
    openSpeech();
    const { params } = jobSettingsPrefill({ mode: 'audio', workflow: 'speech', model: 'qwen3_tts', prompt: 'Welcome', speaker: 'sohee', runs: '1' });

    draft.loadFromUrl(params, ctx, { exact: true });

    expect(draft.state).toMatchObject({ voiceMode: 'builtin', speechSpeaker: 'sohee' });
    expect(sent().get('speaker')).toBe('sohee');
  });

  it('keeps the voice across Music and back, and puts the default back with the row reset', () => {
    openSpeech();
    draft.patch({ voiceMode: 'builtin', speechSpeaker: 'eric' });

    draft.setAudioKind('music', ctx);
    draft.setAudioKind('speech', ctx);
    expect(draft.state.speechSpeaker).toBe('eric');

    draft.patch(rowDefaults('voice', defaultsFor(ctx, draft.state, null)));
    expect(draft.state).toMatchObject({ voiceMode: 'builtin', speechSpeaker: null });
    expect(sent().get('speaker')).toBe('ryan');
    expect(changedRowCount(workspaceCapabilities(ctx, draft.state), draft.state, defaultsFor(ctx, draft.state, null))).toBe(0);
  });
});

describe('a voice the chosen model does not have', () => {
  const OTHER_SPEECH: AudioModelOption = {
    ...AUDIO_MODELS[1], id: 'other_tts', label: 'Other TTS', voices: QWEN_VOICES.filter((voice) => voice.id !== 'vivian'), default_voice: 'aiden',
  } as AudioModelOption;
  const both = withAudio(makeContext(), [...AUDIO_MODELS, OTHER_SPEECH]);

  it('is dropped when the context settles on a model without it, and the new model\'s default is read', () => {
    openSpeech(both);
    draft.patch({ speechSpeaker: 'vivian' });
    expect(draft.state.speechSpeaker).toBe('vivian');

    draft.loadFromUrl({ workflow: 'speech', model: 'other_tts', prompt: 'Hello' }, both, { exact: true });

    expect(draft.state.model).toBe('other_tts');
    expect(draft.state.speechSpeaker).toBeNull();
    expect(sent(both).get('speaker')).toBe('aiden');
  });

  it('is read as the model\'s default when a Reuse names it for a model that lacks it', () => {
    openSpeech(both);

    draft.loadFromUrl({ workflow: 'speech', model: 'other_tts', prompt: 'Hello', speaker: 'vivian' }, both, { exact: true });

    expect(draft.state.speechSpeaker).toBeNull();
    expect(sent(both).get('speaker')).toBe('aiden');
  });
});

describe('a draft saved before the voice select', () => {
  it('opens on the default built-in voice and sends it; its old voice mode reads as a change until Reset', () => {
    localStorage.setItem('ziv-workspace-draft-v1', JSON.stringify({ version: 3, workflow: 'speech', audioKind: 'speech', model: 'qwen3_tts', speechText: 'Hi', voiceMode: 'any' }));
    draft.loadDraft();
    draft.hydrateFromContext(ctx, null);

    const view = currentAudioView(ctx);

    expect(view.voice).toMatchObject({ recording: false, speaker: 'ryan', choice: 'ryan', cloneDownload: false });
    expect(sent().get('speaker')).toBe('ryan');
    expect(rowChanged('voice', draft.state, defaultsFor(ctx, draft.state, null))).toBe(true);
  });
});

describe('a draft with the voice at its defaults', () => {
  it('has the voice mode and speaker keys of the draft in the Voice row, so a Reset puts both back', () => {
    openSpeech();
    const defaults: DraftState = defaultsFor(ctx, draft.state, null);

    expect(defaults).toMatchObject({ voiceMode: 'builtin', speechSpeaker: null });
  });
});

describe('a saved voice, from a Reuse to the next Generate', () => {
  const reuse = { workflow: 'speech', model: 'qwen3_tts', prompt: 'Welcome', saved_voice: 'narrator', seed: '7' };

  beforeEach(async () => {
    api.deleteVoice.mockResolvedValue({ status: 'ok' });
    await library(voice('narrator', 'Narrator'), voice('anna', 'Anna'));
  });

  it('shows and sends the saved voice a Reuse names, with no recording, transcript, consent, speaker or delivery', () => {
    openSpeech();
    draft.patch({ voicePath: '/voices/old.wav', voiceText: 'old words', speechSpeaker: 'vivian' });

    draft.loadFromUrl(reuse, ctx, { exact: true });

    expect(draft.state).toMatchObject({ voiceMode: 'saved', savedVoice: 'narrator', voicePath: null, voiceUpload: null, voiceText: '' });
    expect(currentAudioView(ctx).voice).toMatchObject({ saved: true, choice: 'saved:narrator', chip: 'Narrator', needsConsent: false });
    const form = sent();
    expect(form.get('saved_voice')).toBe('narrator');
    for (const field of ['speaker', 'voice_path', 'voice_file', 'voice_text', 'voice_consent', 'emotion', 'pace']) expect(form.has(field)).toBe(false);
  });

  it('comes back as the same voice through the form, the job\'s settings and the next Load settings', () => {
    openSpeech();
    draft.loadFromUrl(reuse, ctx, { exact: true });
    const settings = formSettings(sent());
    expect(settings).toMatchObject({ workflow: 'speech', saved_voice: 'narrator' });

    draft.reset();
    openSpeech();
    expect(draft.state).toMatchObject({ voiceMode: 'builtin', savedVoice: null });
    const { params } = jobSettingsPrefill(settings);
    expect(params).toMatchObject({ saved_voice: 'narrator' });
    draft.loadFromUrl(params, ctx, { exact: true });

    expect(draft.state).toMatchObject({ voiceMode: 'saved', savedVoice: 'narrator', speechText: 'Welcome' });
    expect(sent().get('saved_voice')).toBe('narrator');
  });

  it('leaves the saved voice again when a later Reuse names a built-in voice or a recording', () => {
    openSpeech();
    draft.loadFromUrl(reuse, ctx, { exact: true });

    draft.loadFromUrl({ workflow: 'speech', model: 'qwen3_tts', prompt: 'Welcome', speaker: 'vivian' }, ctx, { exact: true });
    expect(draft.state).toMatchObject({ voiceMode: 'builtin', savedVoice: null, speechSpeaker: 'vivian' });

    draft.loadFromUrl(reuse, ctx, { exact: true });
    draft.loadFromUrl({ workflow: 'speech', model: 'qwen3_tts', prompt: 'Welcome', voice_path: '/voices/me.wav', voice_text: 'Hi.' }, ctx, { exact: true });
    expect(draft.state).toMatchObject({ voiceMode: 'recording', savedVoice: null, voicePath: '/voices/me.wav' });
    expect(sent().has('saved_voice')).toBe(false);
  });

  it('counts as a change in the Voice row, and the row reset puts the default built-in voice back', () => {
    openSpeech();
    draft.loadFromUrl(reuse, ctx, { exact: true });
    const defaults = defaultsFor(ctx, draft.state, null);
    expect(rowChanged('voice', draft.state, defaults)).toBe(true);

    draft.patch(rowDefaults('voice', defaults));

    expect(draft.state).toMatchObject({ voiceMode: 'builtin', savedVoice: null });
    expect(sent().get('speaker')).toBe('ryan');
    expect(rowChanged('voice', draft.state, defaults)).toBe(false);
  });

  it('is saved with the draft and found again after a reload', () => {
    openSpeech();
    draft.setSavedVoice('narrator');
    draft.saveDraft();

    draft.loadDraft();

    expect(draft.state).toMatchObject({ voiceMode: 'saved', savedVoice: 'narrator' });
  });

  it('is chosen and left for a default voice from the Voice select', () => {
    openSpeech();
    const voiceOf = () => currentAudioView(ctx).voice;

    chooseVoice(voiceOf(), 'saved:anna');
    expect(draft.state).toMatchObject({ voiceMode: 'saved', savedVoice: 'anna' });

    chooseVoice(voiceOf(), 'saved:narrator');
    chooseVoice(voiceOf(), 'sohee');
    expect(draft.state).toMatchObject({ voiceMode: 'builtin', savedVoice: null, speechSpeaker: 'sohee' });

    chooseVoice(voiceOf(), 'ryan');
    expect(draft.state.speechSpeaker).toBeNull();
  });

  it('drops a one-off recording, its words and its tick when a voice is chosen from the Voice select', () => {
    openSpeech();
    draft.setVoice({ path: '/voices/me.wav' });
    draft.update('voiceText', 'A take.');
    voiceConsent.ticked = true;

    chooseVoice(currentAudioView(ctx).voice, 'saved:anna');

    expect(draft.state).toMatchObject({ voiceMode: 'saved', savedVoice: 'anna', voicePath: null, voiceUpload: null, voiceText: '' });
    expect(voiceConsent.ticked).toBe(false);
    expect(currentAudioView(ctx).voice).toMatchObject({ recording: false, set: false });
  });

  describe('a one-off recording', () => {
    it('is the voice, with an empty Voice select, and leaving it returns to the saved voice chosen before it', () => {
      openSpeech();
      draft.setSavedVoice('anna');

      draft.setVoice({ path: '/voices/me.wav' });
      expect(draft.state).toMatchObject({ voiceMode: 'recording', savedVoice: 'anna' });
      expect(currentAudioView(ctx).voice).toMatchObject({ recording: true, cloned: true, choice: '' });

      draft.update('voiceText', 'A take.');
      voiceConsent.ticked = true;
      draft.leaveRecording();

      expect(draft.state).toMatchObject({ voiceMode: 'saved', savedVoice: 'anna', voicePath: null, voiceText: '' });
      expect(voiceConsent.ticked).toBe(false);
      expect(currentAudioView(ctx).voice).toMatchObject({ recording: false, choice: 'saved:anna' });
    });

    it('returns to the built-in voice chosen before it when no saved voice was chosen', () => {
      openSpeech();
      chooseVoice(currentAudioView(ctx).voice, 'sohee');
      draft.setVoice({ path: '/voices/me.wav' });

      draft.leaveRecording();

      expect(draft.state).toMatchObject({ voiceMode: 'builtin', speechSpeaker: 'sohee' });
      expect(currentAudioView(ctx).voice.speaker).toBe('sohee');
    });
  });
});

describe('a saved voice that is deleted', () => {
  const DELETED = 'This saved voice was deleted. Choose another voice.';

  beforeEach(async () => {
    api.deleteVoice.mockResolvedValue({ status: 'ok' });
    await library(voice('narrator', 'Narrator'), voice('anna', 'Anna'));
  });

  it('on a Reuse or Load settings of a take whose voice is gone: the choice is kept and flagged, until another voice is chosen', async () => {
    openSpeech();
    await library(voice('anna', 'Anna'));

    draft.loadFromUrl({ workflow: 'speech', model: 'qwen3_tts', prompt: 'Welcome', saved_voice: 'narrator' }, ctx, { exact: true });

    expect(draft.state).toMatchObject({ voiceMode: 'saved', savedVoice: 'narrator' });
    expect(currentAudioView(ctx).reason).toBe(DELETED);

    chooseVoice(currentAudioView(ctx).voice, 'saved:anna');
    expect(currentAudioView(ctx).reason).toBeNull();
  });
});
