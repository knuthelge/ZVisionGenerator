import { beforeEach, describe, expect, it, vi } from 'vitest';

import { audioView } from '../../features/workspace/audioView';
import { AUDIO_MODELS } from '../../test-utils/audio';
import type { DraftState } from '$lib/types';

const NO_SETTINGS = { showLyrics: false, showLength: false, showTempo: false, showKey: false, showDelivery: false, showSoundType: false };

const SPEECH_MODEL = AUDIO_MODELS[1];

/** What a reload does to the page's state: every module starts again, and only localStorage is left. */
async function reload(): Promise<{ draft: typeof import('./draft.svelte').draft; voiceConsent: typeof import('./voiceConsent.svelte').voiceConsent }> {
  vi.resetModules();
  const [{ draft }, { voiceConsent }] = await Promise.all([import('./draft.svelte'), import('./voiceConsent.svelte')]);
  return { draft, voiceConsent };
}

describe('the speaker\'s consent across a reload', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('forgets the tick and the Any-take exemption, so the tick is asked for again', async () => {
    const first = await reload();
    // A voice that is an Any take (exempt), already ticked as well, and saved as the draft is.
    first.draft.patch({ workflow: 'speech', speechText: 'Hello there.', voiceMode: 'recording', voicePath: '/outputs/any.flac', voiceText: 'A take.' });
    first.voiceConsent.reset(true);
    first.voiceConsent.ticked = true;
    first.draft.saveDraft();

    const second = await reload();

    expect(second.draft.state).toMatchObject({ voiceMode: 'recording', voicePath: '/outputs/any.flac' });
    expect(second.voiceConsent.ticked).toBe(false);
    expect(second.voiceConsent.exempt).toBe(false);
    const view = audioView({
      state: second.draft.state as DraftState,
      caps: { isAudio: true, audioKind: 'speech', showVoice: true, ...NO_SETTINGS },
      model: SPEECH_MODEL,
      file: null,
      restored: true,
      lengthOf: () => null,
      consent: second.voiceConsent,
    });
    expect(view.voice.needsConsent).toBe(true);
    expect(view.reason).toBe('Tick the box to confirm you may clone this voice.');
  });

  it('asks for no tick for a saved voice, whose record holds the consent, though the one-off recording behind it needs one', async () => {
    const first = await reload();
    first.draft.patch({ workflow: 'speech', speechText: 'Hello there.', voiceMode: 'recording', voicePath: '/voices/me.wav', voiceText: 'A take.' });
    first.voiceConsent.reset(false);
    first.draft.setSavedVoice('narrator');

    const view = audioView({
      state: first.draft.state as DraftState,
      caps: { isAudio: true, audioKind: 'speech', showVoice: true, ...NO_SETTINGS },
      model: SPEECH_MODEL,
      file: null,
      restored: true,
      lengthOf: () => null,
      consent: first.voiceConsent,
      voices: { loaded: true, list: [{ slug: 'narrator', name: 'Narrator', text: 'Hi.', origin: 'recording', seconds: 5, description: null, created_at: '2026-11-08T10:00:00Z' }] },
    });

    expect(view.voice).toMatchObject({ saved: true, needsConsent: false });
    expect(view.reason).toBeNull();
  });

  it('still asks for the tick for a new one-off recording after a saved voice replaced the ticked one', async () => {
    const { draft, voiceConsent } = await reload();
    // The Voice select's choice goes through the page's own modules, which a reload replaced.
    const { chooseVoice } = await import('../../features/workspace/audioView');
    draft.patch({ workflow: 'speech', speechText: 'Hello there.', voiceText: 'A take.' });
    draft.setVoice({ path: '/voices/me.wav' });
    voiceConsent.ticked = true;
    const view = () => audioView({
      state: draft.state as DraftState,
      caps: { isAudio: true, audioKind: 'speech', showVoice: true, ...NO_SETTINGS },
      model: SPEECH_MODEL,
      file: null,
      restored: true,
      lengthOf: () => null,
      consent: voiceConsent,
      voices: { loaded: true, list: [{ slug: 'narrator', name: 'Narrator', text: 'Hi.', origin: 'recording', seconds: 5, description: null, created_at: '2026-11-08T10:00:00Z' }] },
    });
    expect(view().reason).toBeNull();

    chooseVoice({ defaultSpeaker: 'ryan' }, 'saved:narrator');
    expect(view().voice).toMatchObject({ saved: true, recording: false, needsConsent: false });
    expect(voiceConsent.ticked).toBe(false);
    draft.setVoice({ path: '/voices/other.wav' });
    draft.update('voiceText', 'Another take.');

    expect(view().voice).toMatchObject({ recording: true, needsConsent: true, ticked: false });
    expect(view().reason).toBe('Tick the box to confirm you may clone this voice.');
  });

  it('leaves nothing of the consent in the saved draft', async () => {
    const first = await reload();
    first.draft.patch({ workflow: 'speech', voiceMode: 'recording', voicePath: '/voices/me.wav' });
    first.voiceConsent.ticked = true;
    first.draft.saveDraft();

    const saved = JSON.parse(localStorage.getItem('ziv-workspace-draft-v1') ?? '{}') as Record<string, unknown>;

    expect(Object.keys(saved).some((key) => key.toLowerCase().includes('consent'))).toBe(false);
  });
});
