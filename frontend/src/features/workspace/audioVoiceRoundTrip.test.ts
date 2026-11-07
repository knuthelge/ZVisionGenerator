import { beforeEach, describe, expect, it } from 'vitest';

import { AUDIO_MODELS, QWEN_VOICES, withAudio } from '../../test-utils/audio';
import { makeContext } from '../../test-utils/workspace';
import { defaultsFor, draft } from '$lib/state/draft.svelte';
import { jobSettingsPrefill } from '$lib/state/jobSettings';
import type { AudioModelOption, DraftState, WorkspaceContext } from '$lib/types';
import { audioFormData, currentAudioView } from './audioView';
import { changedRowCount, rowChanged, rowDefaults } from './settingRows';
import { workspaceCapabilities } from './capabilities';

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
