import { flushSync } from 'svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const layoutMock = vi.hoisted(() => ({ compact: false, mode: 'wide', coarse: false }));
vi.mock('$lib/state/layoutMode.svelte', () => ({ layout: layoutMock }));

const api = vi.hoisted(() => ({ saveRecordingVoice: vi.fn() }));
vi.mock('$lib/api/voices', async (importOriginal) => ({ ...(await importOriginal<typeof import('$lib/api/voices')>()), saveRecordingVoice: api.saveRecordingVoice }));

import { ApiError } from '$lib/api/client';
import { toasts } from '$lib/state/toasts.svelte';
import { voicesStore } from '$lib/state/voices.svelte';
import type { GalleryAsset, SavedVoice } from '$lib/types';
import { AUDIO_SETTINGS, makeSpeechAsset } from '../../../test-utils/audio';
import { mountComponent, settle } from '../../../test-utils/mount';
import { reactiveProps } from '../../../test-utils/reactiveProps.svelte';
import SaveVoiceDialog from './SaveVoiceDialog.svelte';

const MORGAN: SavedVoice = { slug: 'harbour-take', name: 'Harbour take', text: 'Welcome back to the show.', origin: 'recording', seconds: 11, description: null, created_at: '2026-11-08T11:00:00Z' };

const field = <T extends HTMLElement>(selector: string) => document.querySelector<T>(selector)!;
const nameInput = () => field<HTMLInputElement>('#sv-name');
const textBox = () => field<HTMLTextAreaElement>('#sv-text');
const tick = () => document.querySelector<HTMLInputElement>('input[type="checkbox"]');
const saveButton = () => field<HTMLButtonElement>('[data-action="save-voice"]');

function type(input: HTMLInputElement | HTMLTextAreaElement, value: string): void {
  input.value = value;
  flushSync(() => input.dispatchEvent(new Event('input', { bubbles: true })));
}

function mountDialog(asset: GalleryAsset | null, overrides: Record<string, unknown> = {}) {
  const props = reactiveProps({ asset, model: 'qwen3_tts', vocabulary: AUDIO_SETTINGS, onclose: vi.fn(), ...overrides });
  mountComponent(SaveVoiceDialog, props);
  return props;
}

/** The form data of the nth save call, as plain fields (the hidden "false" and the tick are both kept). */
function sent(call = 0): Record<string, string[]> {
  const form = api.saveRecordingVoice.mock.calls[call][0] as FormData;
  return Object.fromEntries([...new Set(form.keys())].map((key) => [key, form.getAll(key) as string[]]));
}

describe.each([['wide', false], ['compact', true]])('SaveVoiceDialog on %s', (_name, compact) => {
  beforeEach(() => {
    Object.assign(layoutMock, { compact, mode: compact ? 'compact' : 'wide', coarse: compact });
    api.saveRecordingVoice.mockReset();
    voicesStore.reset();
    toasts.splice(0, toasts.length);
  });

  async function open(asset: GalleryAsset = makeSpeechAsset({ prompt: 'Welcome back [sad] to the show.' }), overrides: Record<string, unknown> = {}) {
    const props = mountDialog(asset, overrides);
    await settle();
    return props;
  }

  it('is closed without a take, and opens on one with the clip, an empty name and its words', async () => {
    const props = mountDialog(null);
    await settle();
    expect(document.querySelector('[role="dialog"]')).toBeNull();

    props.asset = makeSpeechAsset({ prompt: 'Hello there.', duration: 11 });
    await settle();
    expect(document.querySelector('[role="dialog"]')?.getAttribute('aria-label')).toBe('Save as voice');
    expect(document.querySelector('.sv-name')?.textContent).toBe('speech.flac');
    expect(field('[data-testid="save-voice-length"]').textContent).toBe('0:11');
    expect(document.querySelector('audio, button[aria-label^="Play"]')).not.toBeNull();
    expect(nameInput().value).toBe('');
    expect(textBox().value).toBe('Hello there.');
  });

  it('leaves the Delivery marks out of the words it offers', async () => {
    await open();

    expect(textBox().value).not.toContain('[sad]');
    expect(textBox().value).toBe('Welcome back to the show.');
  });

  it('shows the tick only for a take that itself followed a recording', async () => {
    await open(makeSpeechAsset({ has_reference_voice: false }));
    expect(tick()).toBeNull();
  });

  it('asks for the tick of a take that followed a recording, and keeps Save voice off until it is given', async () => {
    await open(makeSpeechAsset({ has_reference_voice: true }));
    type(nameInput(), 'Harbour take');
    expect(tick()).not.toBeNull();
    expect(saveButton().disabled).toBe(true);

    flushSync(() => tick()!.click());
    expect(saveButton().disabled).toBe(false);
  });

  it('keeps Save voice off until the voice has a name and words', async () => {
    await open();
    expect(saveButton().disabled).toBe(true);

    type(nameInput(), '  ');
    expect(saveButton().disabled).toBe(true);
    type(nameInput(), 'Harbour take');
    expect(saveButton().disabled).toBe(false);
    type(textBox(), ' ');
    expect(saveButton().disabled).toBe(true);
  });

  it('keeps Save voice off when no speech model can clone', async () => {
    await open(makeSpeechAsset(), { model: null });
    type(nameInput(), 'Harbour take');

    expect(saveButton().disabled).toBe(true);
  });

  it('saves the take by its path with the edited words, says so in a toast and closes without leaving the page', async () => {
    api.saveRecordingVoice.mockResolvedValue(MORGAN);
    const props = await open(makeSpeechAsset({ prompt: 'Welcome back [sad] to the show.', file_path: '/outputs/out/take.flac' }));
    type(nameInput(), ' Harbour take ');
    type(textBox(), 'Welcome back to the old harbour.');
    flushSync(() => saveButton().click());
    await settle();

    expect(sent()).toEqual({
      model: ['qwen3_tts'],
      name: ['Harbour take'],
      voice_path: ['/outputs/out/take.flac'],
      voice_text: ['Welcome back to the old harbour.'],
    });
    expect(toasts.map((toast) => [toast.type, toast.message])).toEqual([['success', 'Saved voice "Harbour take"']]);
    expect(props.onclose).toHaveBeenCalledTimes(1);
    expect(voicesStore.list).toContainEqual(MORGAN);
  });

  it('sends the tick with a take that followed a recording', async () => {
    api.saveRecordingVoice.mockResolvedValue(MORGAN);
    await open(makeSpeechAsset({ has_reference_voice: true }));
    type(nameInput(), 'Harbour take');
    flushSync(() => tick()!.click());
    flushSync(() => saveButton().click());
    await settle();

    expect(sent().voice_consent).toEqual(['on']);
  });

  it('saves with Enter in the name field', async () => {
    api.saveRecordingVoice.mockResolvedValue(MORGAN);
    await open();
    type(nameInput(), 'Harbour take');
    flushSync(() => field<HTMLFormElement>('form').dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })));
    await settle();

    expect(api.saveRecordingVoice).toHaveBeenCalledTimes(1);
  });

  it('shows a name that is taken under Name, keeps the dialog open and the fields as they were', async () => {
    api.saveRecordingVoice.mockRejectedValue(new ApiError('Conflict', 409, 'A voice named "Harbour take" already exists.'));
    const props = await open();
    type(nameInput(), 'Harbour take');
    flushSync(() => saveButton().click());
    await settle();

    const error = field('#sv-name-error');
    expect(error.textContent).toBe('A voice named "Harbour take" already exists.');
    expect(nameInput().getAttribute('aria-describedby')).toBe('sv-name-error');
    expect(nameInput().value).toBe('Harbour take');
    expect(props.onclose).not.toHaveBeenCalled();
    expect(toasts).toHaveLength(0);

    type(nameInput(), 'Harbour take 2');
    expect(document.querySelector('#sv-name-error')).toBeNull();
  });

  it('shows any other refusal in the dialog, not under Name', async () => {
    api.saveRecordingVoice.mockRejectedValue(new ApiError('Unprocessable', 422, 'Use a recording of 3 to 30 seconds.'));
    const props = await open();
    type(nameInput(), 'Harbour take');
    flushSync(() => saveButton().click());
    await settle();

    expect(document.querySelector('[role="dialog"] [role="alert"]')?.textContent).toBe('Use a recording of 3 to 30 seconds.');
    expect(document.querySelector('#sv-name-error')).toBeNull();
    expect(props.onclose).not.toHaveBeenCalled();
  });

  it('starts each take from its own words with no name, no tick and no old error', async () => {
    api.saveRecordingVoice.mockRejectedValue(new ApiError('Conflict', 409, 'Taken.'));
    const props = await open(makeSpeechAsset({ prompt: 'First take.', has_reference_voice: true }));
    type(nameInput(), 'Harbour take');
    flushSync(() => tick()!.click());
    flushSync(() => saveButton().click());
    await settle();

    props.asset = makeSpeechAsset({ id: 'out/other.flac', prompt: 'Second take.', has_reference_voice: true });
    await settle();

    expect(textBox().value).toBe('Second take.');
    expect(nameInput().value).toBe('');
    expect(tick()!.checked).toBe(false);
    expect(document.querySelector('#sv-name-error')).toBeNull();
  });

  it('keeps what was typed when the vocabulary is replaced while it is open', async () => {
    const props = await open(makeSpeechAsset({ prompt: 'Welcome back [sad] to the show.', has_reference_voice: true }));
    type(nameInput(), 'Harbour take');
    type(textBox(), 'My own words.');
    flushSync(() => tick()!.click());

    props.vocabulary = structuredClone(AUDIO_SETTINGS);
    await settle();

    expect(nameInput().value).toBe('Harbour take');
    expect(textBox().value).toBe('My own words.');
    expect(tick()!.checked).toBe(true);
  });

  it(compact ? 'has Save voice alone in the foot' : 'has Cancel and Save voice in the foot, and Cancel closes', async () => {
    const props = await open();
    const footer = Array.from(document.querySelectorAll('button')).filter((button) => button.textContent?.trim() === 'Cancel');

    if (compact) {
      expect(footer).toHaveLength(0);
    } else {
      expect(footer).toHaveLength(1);
      flushSync(() => footer[0].click());
      expect(props.onclose).toHaveBeenCalledTimes(1);
    }
  });
});
