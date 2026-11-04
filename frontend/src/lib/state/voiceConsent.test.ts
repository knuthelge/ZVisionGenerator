import { beforeEach, describe, expect, it, vi } from 'vitest';

import { audioView } from '../../features/workspace/audioView';
import { AUDIO_MODELS } from '../../test-utils/audio';
import type { DraftState } from '$lib/types';

const HIGGS = AUDIO_MODELS[1];

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
      caps: { isAudio: true, audioKind: 'speech', showVoice: true },
      model: HIGGS,
      file: null,
      restored: true,
      lengthOf: () => null,
      consent: second.voiceConsent,
    });
    expect(view.voice.needsConsent).toBe(true);
    expect(view.reason).toBe('Tick the box to confirm you may clone this voice.');
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
