import { beforeEach, describe, expect, it } from 'vitest';

import { recordingForm } from './recordingForm.svelte';

const file = (name: string): File => new File(['x'], name, { type: 'audio/wav' });

beforeEach(() => {
  recordingForm.clear();
});

describe('recordingForm', () => {
  it('asks for the tick once a recording is chosen, and clears the tick when the recording changes', () => {
    recordingForm.setRecording(null, file('a.wav'));
    recordingForm.ticked = true;
    expect(recordingForm).toMatchObject({ recorded: true, needsTick: true, ticked: true });

    recordingForm.setRecording('/voices/b.wav', null);

    expect(recordingForm).toMatchObject({ path: '/voices/b.wav', file: null, ticked: false, needsTick: true });
  });

  it('does not count an empty path as a recording', () => {
    recordingForm.setRecording('', null);

    expect(recordingForm.recorded).toBe(false);
  });

  it('fills from the one-off voice, keeps its words, and names nothing and ticks nothing', () => {
    recordingForm.name = 'left over';
    recordingForm.ticked = true;

    recordingForm.fillFrom({ path: '/voices/me.wav', file: null, text: 'Hi.', needsConsent: true });

    expect(recordingForm).toMatchObject({ path: '/voices/me.wav', text: 'Hi.', ticked: false, name: '', needsTick: true });
  });

  it('counts a form that was never filled as not filled from any recording, even a held upload with no path', () => {
    expect(recordingForm.filledFrom({ path: null, file: null })).toBe(false);

    recordingForm.fillFrom({ path: null, file: null, text: '', needsConsent: true });
    expect(recordingForm.filledFrom({ path: null, file: null })).toBe(true);

    recordingForm.clear();
    expect(recordingForm.filledFrom({ path: null, file: null })).toBe(false);
  });

  it('needs no tick for a filled recording that is exempt, until a different recording is chosen', () => {
    recordingForm.fillFrom({ path: '/outputs/take.flac', file: null, text: 'A take.', needsConsent: false });
    expect(recordingForm.needsTick).toBe(false);

    recordingForm.setRecording('/voices/other.wav', null);

    expect(recordingForm.needsTick).toBe(true);
  });

  it('prefers the held file over its path when it fills from the one-off voice', () => {
    const held = file('held.wav');

    recordingForm.fillFrom({ path: '/voices/old.wav', file: held, text: '', needsConsent: true });

    expect(recordingForm).toMatchObject({ path: null, file: held });
  });

  it('builds the fields of POST /api/voices: the file wins over the path, "on" only when ticked and asked for', () => {
    recordingForm.setRecording('/voices/me.wav', null);
    recordingForm.text = 'Hi.';
    recordingForm.name = '  Morgan ';
    let form = recordingForm.formData('qwen3_tts');
    expect(form.get('model')).toBe('qwen3_tts');
    expect(form.get('name')).toBe('Morgan');
    expect(form.get('voice_path')).toBe('/voices/me.wav');
    expect(form.has('voice_file')).toBe(false);
    expect(form.get('voice_text')).toBe('Hi.');
    expect(form.has('voice_consent')).toBe(false);

    recordingForm.setRecording(null, file('me.wav'));
    recordingForm.ticked = true;
    form = recordingForm.formData('qwen3_tts');

    expect((form.get('voice_file') as File).name).toBe('me.wav');
    expect(form.has('voice_path')).toBe(false);
    expect(form.getAll('voice_consent')).toEqual(['on']);
  });

  it('sends no tick for an exempt recording even if the box was ticked', () => {
    recordingForm.fillFrom({ path: '/outputs/take.flac', file: null, text: 'A take.', needsConsent: false });
    recordingForm.ticked = true;

    expect(recordingForm.formData('qwen3_tts').has('voice_consent')).toBe(false);
  });

  it('empties everything with clear', () => {
    recordingForm.fillFrom({ path: '/voices/me.wav', file: null, text: 'Hi.', needsConsent: true });
    recordingForm.ticked = true;
    recordingForm.name = 'Morgan';

    recordingForm.clear();

    expect(recordingForm).toMatchObject({ path: null, file: null, text: '', ticked: false, name: '', recorded: false });
  });
});
