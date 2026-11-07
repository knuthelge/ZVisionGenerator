import { describe, expect, it } from 'vitest';

import { AUDIO_MODELS, AUDIO_SETTINGS } from '../../test-utils/audio';
import { emptyAudioRewrites } from '$lib/state/audioRewrites';
import { DEFAULT_SPEECH_DELIVERY } from '$lib/state/audioDraft';
import type { AudioModelOption, DraftState } from '$lib/types';
import { audioFormData, audioView, type AudioViewInput } from './audioView';

/** The controls each kind's first model shows: a song with lyrics sets no length. */
const SHOWN = {
  music: { showLyrics: true, showTempo: true, showKey: true, showLength: false, showDelivery: false, showSoundType: false },
  speech: { showLyrics: false, showTempo: false, showKey: false, showLength: false, showDelivery: true, showSoundType: false },
  sound: { showLyrics: false, showTempo: false, showKey: false, showLength: true, showDelivery: false, showSoundType: true },
};

const YUE2 = AUDIO_MODELS[0];
const SPEECH_MODEL = AUDIO_MODELS[1];
const SA3 = AUDIO_MODELS[2];

function state(overrides: Partial<DraftState> = {}): DraftState {
  return {
    workflow: 'music', model: 'yue2', audioKind: 'music', audioStyle: '', lyrics: '', speechText: '', soundPrompt: '', audioSeconds: 10,
    audioTempo: null, audioKey: null, musicSeconds: 0, soundType: 'auto', speechDelivery: DEFAULT_SPEECH_DELIVERY,
    voiceMode: 'builtin', voicePath: null, voiceUpload: null, voiceText: '', runs: 1, seed: null, audioRewrites: emptyAudioRewrites(),
    ...overrides,
  } as DraftState;
}

function view(overrides: Partial<DraftState> = {}, input: Partial<AudioViewInput> = {}) {
  const full = state(overrides);
  const kind = full.workflow as 'music' | 'speech' | 'sound';
  const model = { music: YUE2, speech: SPEECH_MODEL, sound: SA3 }[kind];
  return audioView({
    state: full,
    caps: { isAudio: true, audioKind: kind, showVoice: kind === 'speech', ...SHOWN[kind] },
    model,
    file: null,
    restored: true,
    lengthOf: () => null,
    consent: { ticked: false, exempt: false },
    ...input,
  });
}

const RECORDING = { workflow: 'speech' as const, speechText: 'Hello there.', voiceMode: 'recording' as const, voicePath: '/voices/me.wav', voiceText: 'Hi, it is me.' };

describe('the voice of the Speech tab', () => {
  const speech = (overrides: Partial<DraftState> = {}, input: Partial<AudioViewInput> = {}) => view({ workflow: 'speech', speechText: 'Hello there.', ...overrides }, input).voice;

  it('reads in the model\'s default built-in voice until another is chosen, and shows its name', () => {
    expect(speech()).toMatchObject({ recording: false, speaker: 'ryan', speakerLabel: 'Ryan', defaultSpeaker: 'ryan', choice: 'ryan', summary: { text: 'Ryan', tone: 'muted' }, chip: 'Ryan' });
    expect(speech({ speechSpeaker: 'vivian' })).toMatchObject({ speaker: 'vivian', choice: 'vivian', summary: { text: 'Vivian', tone: 'muted' }, chip: 'Vivian' });
  });

  it('lists the nine built-in voices with their languages and then From a recording', () => {
    const options = speech().options;

    expect(options).toHaveLength(10);
    expect(options[0]).toEqual({ value: 'ryan', label: 'Ryan · English' });
    expect(options[5]).toEqual({ value: 'dylan', label: 'Dylan · Beijing Chinese' });
    expect(options[9]).toEqual({ value: 'recording', label: 'From a recording' });
  });

  it('falls back to the default voice when the draft names one the model does not have', () => {
    expect(speech({ speechSpeaker: 'nobody' })).toMatchObject({ speaker: 'ryan', choice: 'ryan' });
  });

  it('sends no speaker with a recording, and offers the recording as the choice', () => {
    expect(speech(RECORDING)).toMatchObject({ recording: true, speaker: '', choice: 'recording', summary: { text: 'me.wav', tone: 'active' }, chip: 'me.wav' });
  });

  it('says the cloning files are still to download only for a recording, and only while they are not downloaded', () => {
    const downloaded = { ...SPEECH_MODEL, voice_download: { downloaded: true } } as typeof SPEECH_MODEL;
    const unknown = { ...SPEECH_MODEL, voice_download: null } as typeof SPEECH_MODEL;

    expect(speech(RECORDING).cloneDownload).toBe(true);
    expect(speech().cloneDownload).toBe(false);
    expect(view({ ...RECORDING }, { model: downloaded }).voice.cloneDownload).toBe(false);
    expect(view({ ...RECORDING }, { model: unknown }).voice.cloneDownload).toBe(false);
  });

  it('has no voice on a kind that has none', () => {
    expect(view({ workflow: 'music' }).voice).toMatchObject({ recording: false, speaker: '', options: [], chip: '' });
  });
});

describe('what a submit sends for the voice', () => {
  const sent = (overrides: Partial<DraftState>) => {
    const fields = { workflow: 'speech' as const, model: 'qwen3_tts', speechText: 'Hello there.', ...overrides };
    return audioFormData(view(fields), state(fields));
  };

  it('sends the built-in voice as the speaker, the default included', () => {
    expect(sent({}).get('speaker')).toBe('ryan');
    expect(sent({ speechSpeaker: 'sohee' }).get('speaker')).toBe('sohee');
  });

  it('sends the recording and no speaker when it follows one', () => {
    const form = sent({ ...RECORDING, speechSpeaker: 'sohee' });

    expect(form.has('speaker')).toBe(false);
    expect(form.get('voice_path')).toBe('/voices/me.wav');
  });

  it('sends no speaker for music or sound', () => {
    expect(audioFormData(view({ workflow: 'music', audioStyle: 'calm' }), state({ workflow: 'music', audioStyle: 'calm' })).has('speaker')).toBe(false);
  });
});

describe('what a submit sends for the settings of each kind', () => {
  const sentFrom = (overrides: Partial<DraftState>, input: Partial<AudioViewInput> = {}) => {
    const fields = { audioStyle: 'calm', soundPrompt: 'rain', speechText: 'Hello there.', ...overrides };
    return Object.fromEntries(audioFormData(view(fields, input), state(fields)).entries());
  };
  const shown = (kind: keyof typeof SHOWN, overrides: Partial<(typeof SHOWN)['music']> = {}) => ({ caps: { isAudio: true as const, audioKind: kind, showVoice: kind === 'speech', ...SHOWN[kind], ...overrides } });

  it('sends the tempo and key of a song, blank for Auto', () => {
    expect(sentFrom({ audioTempo: 92, audioKey: 'F#m' })).toMatchObject({ tempo: '92', key: 'F#m' });
    expect(sentFrom({})).toMatchObject({ tempo: '', key: '' });
  });

  it('sends the length of a song only for a model that sets it, blank for the model\'s default', () => {
    const model: AudioModelOption = { id: 'sa3', label: 'Stable Audio 3 Medium', kind: 'music', type: 'audio', hidden_controls: ['lyrics'], limits: { max_seconds: 380, min_seconds: 1, default_seconds: 30 }, licences: [] };
    const sets = { ...shown('music', { showLength: true }), model };

    expect(sentFrom({ musicSeconds: 45 }, sets)).toMatchObject({ seconds: '45' });
    expect(sentFrom({ musicSeconds: 0 }, sets)).toMatchObject({ seconds: '' });
    expect(sentFrom({ musicSeconds: 45 })).not.toHaveProperty('seconds');
  });

  it('sends nothing for a control the model hides, though the draft keeps its value', () => {
    const form = sentFrom({ audioTempo: 92, audioKey: 'C', lyrics: '[verse]\nla' }, shown('music', { showLyrics: false, showTempo: false, showKey: false }));

    expect(form).not.toHaveProperty('lyrics');
    expect(form).not.toHaveProperty('tempo');
    expect(form).not.toHaveProperty('key');
  });

  it('sends the sound type and the length, the length blank when none is chosen yet', () => {
    expect(sentFrom({ workflow: 'sound', soundType: 'effect', audioSeconds: 8 })).toMatchObject({ sound_type: 'effect', seconds: '8' });
    expect(sentFrom({ workflow: 'sound', audioSeconds: 0 })).toMatchObject({ sound_type: 'auto', seconds: '' });
    expect(sentFrom({ soundType: 'effect' })).not.toHaveProperty('sound_type');
  });

  it('sends the five delivery values with a built-in voice, and none with a recording or for another kind', () => {
    const delivery = { emotion: 'sad', manner: 'whisper', pace: 'slow', pitch: 'low', expressiveness: 'high' };

    expect(sentFrom({ workflow: 'speech', speechDelivery: delivery })).toMatchObject(delivery);
    expect(sentFrom({ speechDelivery: delivery, ...RECORDING })).not.toHaveProperty('emotion');
    expect(sentFrom({ workflow: 'speech', speechDelivery: delivery }, shown('speech', { showDelivery: false }))).not.toHaveProperty('pace');
    expect(sentFrom({ workflow: 'sound', speechDelivery: delivery })).not.toHaveProperty('manner');
  });
});

describe('audioView reasons', () => {

  it('puts a run that is writing the text before every other reason, and then it is no reason of the voice', () => {
    const writing = "Enhancing… Generate when it's done.";

    expect(view({ workflow: 'sound', soundPrompt: 'rain' }, { writing }).reason).toBe(writing);
    expect(view({ workflow: 'sound', soundPrompt: '' }, { writing }).reason).toBe(writing);
    expect(view({ ...RECORDING, voiceText: '' }, { writing }).reasonFromVoice).toBe(false);
    expect(view({ workflow: 'sound', soundPrompt: 'rain' }, { writing: null }).reason).toBeNull();
    expect(view({ workflow: 'sound', soundPrompt: 'rain' }).reason).toBeNull();
  });

  it('tells a reason of the voice from one of the text', () => {
    expect(view({ ...RECORDING, voiceText: '' }).reasonFromVoice).toBe(true);
    expect(view(RECORDING).reasonFromVoice).toBe(true);
    expect(view({ ...RECORDING, speechText: '' }).reasonFromVoice).toBe(false);
    expect(view({ ...RECORDING, speechText: 'a'.repeat(10_001) }, { consent: { ticked: true, exempt: false } }).reasonFromVoice).toBe(false);
    expect(view({ workflow: 'speech', speechText: 'Hello there.' }).reasonFromVoice).toBe(false);
  });

  it('refuses a bracketed word that is not a Delivery mark with the server\'s words, whichever voice reads it', () => {
    const words = "[sadd] isn't a Delivery mark. Use one of the Delivery marks, or remove the brackets.";
    const text = { speechText: 'Ha [sadd] ha.' };

    expect(view({ workflow: 'speech', ...text }, { vocabulary: AUDIO_SETTINGS }).reason).toBe(words);
    expect(view({ ...RECORDING, ...text }, { vocabulary: AUDIO_SETTINGS }).reason).toBe(words);
    expect(view({ ...RECORDING, ...text }, { vocabulary: AUDIO_SETTINGS }).reasonFromVoice).toBe(false);
    expect(view({ workflow: 'speech', speechText: 'Ha [sad] ha.' }, { vocabulary: AUDIO_SETTINGS }).reason).toBeNull();
    expect(view({ workflow: 'sound', soundPrompt: 'a [laughs] crowd' }, { vocabulary: AUDIO_SETTINGS }).reason).toBeNull();
  });

  it('names what to write instead of a [pause] or a [laugh], with the same words as the server', () => {
    const reason = (speechText: string) => view({ workflow: 'speech', speechText }, { vocabulary: AUDIO_SETTINGS }).reason;

    expect(reason('Wait [pause] now.')).toBe("[pause] isn't a Delivery mark. For a pause, type … (the Insert menu adds one).");
    expect(reason('Wait [Pauses] now.')).toBe("[Pauses] isn't a Delivery mark. For a pause, type … (the Insert menu adds one).");
    for (const mark of ['[laugh]', '[laughs]', '[laughing]']) {
      expect(reason(`Ha ${mark} ha.`)).toBe(`${mark} isn't a Delivery mark. For a laugh, write haha (the Insert menu adds one).`);
    }
  });

  it('asks for text when the speech text holds only marks, as the server does', () => {
    const reason = (speechText: string, vocabulary = AUDIO_SETTINGS) => view({ workflow: 'speech', speechText }, { vocabulary }).reason;

    expect(reason('[sad] [whisper]')).toBe('Write the text to speak.');
    expect(reason('[sad] [whisper] Hello.')).toBeNull();
  });

  describe('with an idea in Describe it', () => {
    const idea = (workflow: 'music' | 'speech' | 'sound', text: string) => ({ workflow, audioDescribe: { music: { text: workflow === 'music' ? text : '' }, speech: { text: workflow === 'speech' ? text : '' }, sound: { text: '' } } } as unknown as Partial<DraftState>);
    const reason = (overrides: Partial<DraftState>, describes = true) => view(overrides, { describes }).reason;

    it('points an empty Style or Text to Fill in', () => {
      expect(reason(idea('music', 'a sad folk song'))).toBe('Press Fill in, or write the style.');
      expect(reason(idea('speech', 'a welcome message'))).toBe('Press Fill in, or write the text to speak.');
    });

    it('keeps the usual words without an idea, without a served Describe it, and for a Sound, which has none', () => {
      expect(reason(idea('music', '  '))).toBe('Write the style.');
      expect(reason(idea('music', 'a sad folk song'), false)).toBe('Write the style.');
      expect(reason(idea('sound', 'rain'))).toBe('Describe the sound.');
    });

    it('says nothing about Fill in once the Style is written', () => {
      expect(reason({ ...idea('music', 'a sad folk song'), audioStyle: 'folk' })).toBeNull();
    });
  });

  it('checks the Written text, which is what is sent', () => {
    const rewrites = { ...emptyAudioRewrites(), text: { text: 'Hi [laughs].', from: null, history: [] } };

    expect(view({ workflow: 'speech', speechText: 'Hi.', audioRewrites: rewrites }, { vocabulary: AUDIO_SETTINGS }).reason).toContain('[laughs] isn\'t a Delivery mark');
  });

  it('puts the over-long text last, with its own words', () => {
    const long = 'a'.repeat(10_001);

    expect(view({ ...RECORDING, speechText: long }, { consent: { ticked: true, exempt: false } }).reason).toBe("Shorten the text: it's over 10,000 characters, about 10 minutes of speech.");
    expect(view({ ...RECORDING, speechText: long }).reason).toBe('Tick the box to confirm you may clone this voice.');
    expect(view({ workflow: 'speech', speechText: 'a'.repeat(10_000) }).reason).toBeNull();
  });

  it('counts a character outside the BMP once, as Python does', () => {
    expect(view({ workflow: 'speech', speechText: 'a😀b' }).characters).toBe(3);
    expect(view({ workflow: 'speech', speechText: '😀'.repeat(10_000) }).reason).toBeNull();
  });

  it('leaves the marks out of the count, so a text over the cap only by its marks is not over', () => {
    const marked = { workflow: 'speech' as const, speechText: `[sad]${'a'.repeat(10_000)}[as set]` };

    expect(view(marked, { vocabulary: AUDIO_SETTINGS }).characters).toBe(10_000);
    expect(view(marked, { vocabulary: AUDIO_SETTINGS }).reason).toBeNull();
    expect(view(marked).reason).toContain('Shorten the text');
  });

  it('tells the reading time at the open Pace, at Normal for a recording, and none off Speech or without the vocabulary', () => {
    const text = { workflow: 'speech' as const, speechText: 'a'.repeat(1_400) };
    const seconds = (overrides: Partial<DraftState> = {}) => view({ ...text, ...overrides }, { vocabulary: AUDIO_SETTINGS }).readingSeconds;

    expect(seconds()).toBeCloseTo(100);
    expect(seconds({ speechDelivery: { ...DEFAULT_SPEECH_DELIVERY, pace: 'slow' } })).toBeCloseTo(100 / AUDIO_SETTINGS.reading.pace_factor.slow);
    expect(seconds({ speechText: '' })).toBe(0);
    expect(view({ ...RECORDING, speechText: text.speechText, speechDelivery: { ...DEFAULT_SPEECH_DELIVERY, pace: 'slow' } }, { vocabulary: AUDIO_SETTINGS }).readingSeconds).toBeCloseTo(100);
    expect(view(text).readingSeconds).toBeNull();
    expect(view({ workflow: 'music' }, { vocabulary: AUDIO_SETTINGS }).readingSeconds).toBeNull();
  });

  it('says an uploaded recording that did not come back is gone, once the held uploads have been read', () => {
    const gone = { ...RECORDING, voicePath: null, voiceUpload: 'narrator.wav' };

    expect(view(gone, { restored: true }).reason).toBe('Choose the recording again.');
    // Before the read it is not known to be lost: the upload still counts as chosen.
    expect(view(gone, { restored: false }).voice.gone).toBe(false);
    expect(view(gone, { restored: false }).reason).toBe('Tick the box to confirm you may clone this voice.');
  });
});

describe('the Enhanced text of a box', () => {
  const RECORD = { settings: { space: 'forest' }, model: 'owner/llm', format: 'text', task: 'sound' } as const;
  const rewrites = (text: string, input: string, model = 'sa3') => ({
    ...emptyAudioRewrites(),
    sound: { text, from: { input, model, task: 'sound' as const, record: RECORD }, history: [] },
  });

  it('sends the own text alone when the box has no Enhanced text', () => {
    const sound = view({ workflow: 'sound', model: 'sa3', soundPrompt: 'rain' });

    expect(sound).toMatchObject({ box: 'sound', enhanced: false, stale: false, sent: { text: 'rain', input: null, record: null } });
    expect(audioFormData(sound, state({ workflow: 'sound', soundPrompt: 'rain' })).has('enhance_record')).toBe(false);
  });

  it('sends the Enhanced text as the prompt, with the own text it was made from and the record, and it is not out of date', () => {
    const own = state({ workflow: 'sound', model: 'sa3', soundPrompt: 'rain', audioRewrites: rewrites('Heavy rain.', 'rain') });
    const sound = view(own);

    expect(sound).toMatchObject({ enhanced: true, stale: false, sent: { text: 'Heavy rain.', input: 'rain', record: RECORD } });
    const form = audioFormData(sound, own);
    expect(form.get('prompt')).toBe('Heavy rain.');
    expect(form.get('prompt_input')).toBe('rain');
    expect(JSON.parse(form.get('enhance_record') as string)).toEqual(RECORD);
  });

  it('is out of date once the own text or the model is not the one it was made from, and still runs', () => {
    expect(view({ workflow: 'sound', model: 'sa3', soundPrompt: 'rain, closer', audioRewrites: rewrites('Heavy rain.', 'rain') })).toMatchObject({ stale: true, enhanced: true });
    expect(view({ workflow: 'sound', model: 'sa4', soundPrompt: 'rain', audioRewrites: rewrites('Heavy rain.', 'rain') }).stale).toBe(true);
  });

  it('says why it is out of date in a sentence that names the box', () => {
    const own = (text: string, model: string) => view({ workflow: 'sound', model, soundPrompt: text, audioRewrites: rewrites('Heavy rain.', 'rain') });

    expect(own('rain', 'sa3').staleReason).toBeNull();
    expect(own('rain, closer', 'sa3').staleReason).toBe('Your Prompt changed since this was written.');
    expect(own('rain', 'sa4').staleReason).toBe('The model changed since this was written.');
    expect(view({ workflow: 'speech', speechText: 'Hi.' }).staleReason).toBeNull();
  });

  it('lets Generate go with an empty own text while the Enhanced text stands, and asks for text without it', () => {
    expect(view({ workflow: 'sound', model: 'sa3', soundPrompt: '', audioRewrites: rewrites('Heavy rain.', '') }).reason).toBeNull();
    expect(view({ workflow: 'sound', model: 'sa3', soundPrompt: '' }).reason).toBe('Describe the sound.');
  });

  it('sends an empty own text beside the rewrite, as the take records it', () => {
    const own = state({ workflow: 'sound', model: 'sa3', soundPrompt: '', audioRewrites: rewrites('Heavy rain.', '') });

    expect(audioFormData(view(own), own).get('prompt_input')).toBe('');
  });
});
