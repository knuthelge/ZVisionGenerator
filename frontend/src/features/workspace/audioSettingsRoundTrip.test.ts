import { beforeEach, describe, expect, it } from 'vitest';

import cliCases_ from '../../../../tests/fixtures/audio_cli_cases.json';
import settingsCases from '../../../../tests/fixtures/audio_settings_cases.json';
import { AUDIO_MODELS, SA3_MUSIC, withAudio } from '../../test-utils/audio';
import { makeContext } from '../../test-utils/workspace';
import { DELIVERY_FIELDS } from '$lib/state/audioDraft';
import { draft } from '$lib/state/draft.svelte';
import { cliCommand, formSettings } from '$lib/state/cliCommand';
import { jobSettingsPrefill } from '$lib/state/jobSettings';
import { voiceConsent } from '$lib/state/voiceConsent.svelte';
import type { AudioWorkflow, JobSettings, PrefillParams, WorkspaceContext } from '$lib/types';
import { audioFormData, currentAudioView } from './audioView';

const audioCliCases = cliCases_ as { name: string; settings: Record<string, unknown>; command: string }[];

const ctx: WorkspaceContext = withAudio(makeContext(), [...AUDIO_MODELS, SA3_MUSIC]);

type Case = (typeof settingsCases.cases)[number];

/** Open the Audio tab on a kind, as a person does before a Reuse or a Retry lands. */
function openAudio(kind: AudioWorkflow): void {
  draft.hydrateFromContext(ctx, null);
  draft.setTab('audio');
  draft.onTabChange(ctx);
  draft.setAudioKind(kind, ctx);
}

/** The fields the Audio tab posts now, less the tick (a Reuse never restores it). */
function posted(): Record<string, FormDataEntryValue> {
  const form = audioFormData(currentAudioView(ctx), draft.state);
  form.delete('voice_consent');
  return Object.fromEntries(form.entries());
}

/** The form of a case less the tick, as strings. */
function expectedForm(form: Record<string, unknown>): Record<string, unknown> {
  const { voice_consent: _consent, ...rest } = form;
  return rest;
}

beforeEach(() => {
  localStorage.clear();
  draft.reset();
  voiceConsent.reset();
});

describe('a take that sets every new value, through the Audio tab', () => {
  it.each(settingsCases.cases.map((item) => [item.name, item] as const))('comes back from a Reuse as the form that made it: %s', (_name, item: Case) => {
    // The Reuse starts from a form on another kind, so nothing of the take is already there.
    openAudio(item.form.workflow === 'music' ? 'sound' : 'music');

    draft.loadFromUrl(item.reuse as unknown as PrefillParams, ctx, { replaceSlots: true, exact: true });

    expect(posted()).toEqual(expectedForm(('reuse_form' in item ? item.reuse_form : item.form) as Record<string, unknown>));
  });

  it.each(settingsCases.cases.map((item) => [item.name, item] as const))('comes back from Retry and Load settings as the form that made it: %s', (_name, item: Case) => {
    openAudio(item.form.workflow === 'music' ? 'sound' : 'music');
    const { params } = jobSettingsPrefill(item.form as unknown as JobSettings);

    draft.loadFromUrl(params, ctx, { replaceSlots: true, exact: true });

    const expected = expectedForm(item.form as Record<string, unknown>);
    // A recording has no delivery: the job recorded the values that were posted, the tab sends none.
    if ('voice_path' in item.form) for (const field of DELIVERY_FIELDS) delete expected[field];
    expect(posted()).toEqual(expected);
  });
});

describe('a Reuse of a take whose length was left on Auto', () => {
  const reuseOf = (name: string): Case => settingsCases.cases.find((item) => item.name === name)!;

  it('leaves a sound\'s Length on Auto, so a later Type brings the Auto length of that type', () => {
    openAudio('music');

    draft.loadFromUrl(reuseOf('a sound on Auto').reuse as unknown as PrefillParams, ctx, { replaceSlots: true, exact: true });
    draft.patch({ soundType: 'effect' });

    expect(draft.state.audioSeconds).toBe(0);
    expect(posted().seconds).toBe('');
  });

  it('locks a sound\'s Length that was typed, whatever Type follows', () => {
    openAudio('music');

    draft.loadFromUrl(reuseOf('an ambience with a chosen length').reuse as unknown as PrefillParams, ctx, { replaceSlots: true, exact: true });
    draft.patch({ soundType: 'effect' });

    expect(posted().seconds).toBe('20');
  });

  it('leaves a Stable Audio song\'s Length on Auto, and locks one that was typed', () => {
    openAudio('sound');

    draft.loadFromUrl(reuseOf('Stable Audio under Music with the length on Auto').reuse as unknown as PrefillParams, ctx, { replaceSlots: true, exact: true });
    expect(draft.state.musicSeconds).toBe(0);

    draft.loadFromUrl(reuseOf('Stable Audio under Music with a chosen length').reuse as unknown as PrefillParams, ctx, { replaceSlots: true, exact: true });
    expect(draft.state.musicSeconds).toBe(45);
  });
});

describe('a Reuse of a take without a tempo, key, delivery or sound type', () => {
  it('puts the settings of the kind it opens back to Auto or their defaults instead of keeping the form\'s, and leaves the other kinds\' alone', () => {
    openAudio('music');
    const delivery = { emotion: 'angry', manner: 'shout', pace: 'fast', pitch: 'high', expressiveness: 'low' } as const;
    draft.patch({ audioTempo: 150, audioKey: 'Bb', musicSeconds: 90, audioSeconds: 7, soundType: 'one-shot', speechDelivery: delivery });

    draft.loadFromUrl({ workflow: 'music', model: 'yue2', prompt: 'calm piano' }, ctx, { exact: true });

    expect(draft.state).toMatchObject({ audioTempo: null, audioKey: null, musicSeconds: 0, audioSeconds: 7, soundType: 'one-shot', speechDelivery: delivery });
  });

  it('leaves them as they are when the params are a link or a preset, which name only what they set', () => {
    openAudio('music');
    draft.patch({ audioTempo: 150, audioKey: 'Bb' });

    draft.loadFromUrl({ workflow: 'music', model: 'yue2', prompt: 'calm piano' }, ctx);

    expect(draft.state).toMatchObject({ audioTempo: 150, audioKey: 'Bb' });
  });
});

describe('Copy CLI for the same takes', () => {
  // SC-3: the form's fields (the Audio tab's submit) make the command the Python CLI test turns back into the form's request.
  const cliCases = audioCliCases.filter((item) => !('voice_path' in item.settings));

  it.each(cliCases.map((item) => [item.name, item] as const))('copies the command of the form that made it: %s', (_name, item) => {
    openAudio(item.settings.workflow as AudioWorkflow);
    const { params } = jobSettingsPrefill(item.settings as unknown as JobSettings);
    draft.loadFromUrl(params, ctx, { replaceSlots: true, exact: true });

    const settings = formSettings(audioFormData(currentAudioView(ctx), draft.state));

    expect(cliCommand(settings).command).toEqual(item.command);
  });
});
