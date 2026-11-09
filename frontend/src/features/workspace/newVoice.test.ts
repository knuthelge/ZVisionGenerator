import { describe, expect, it } from 'vitest';

import type { ActiveJobState } from '$lib/types';

import { AUDIO_SETTINGS } from '../../test-utils/audio';
import { DESCRIPTION_MAX, defaultPreviewLine, EXAMPLE_DESCRIPTIONS, lineHint, PREVIEW_LINES, PREVIEW_MAX, previewLanguage, runLine, type PreviewLanguage } from './newVoice';

const READING = AUDIO_SETTINGS.reading;
const RANGE = { min: 3, max: 30 };

describe('previewLanguage', () => {
  it.each<[string, string, PreviewLanguage]>([
    ['Hangul', '안녕하세요, 오늘은 날씨가 좋네요.', 'Korean'],
    ['kana', 'こんにちは、今日はいい天気ですね。', 'Japanese'],
    ['kana without any kanji', 'おはようございます', 'Japanese'],
    ['Han alone', '今天天气很好，我们去公园散步吧。', 'Chinese'],
    ['Cyrillic', 'Здравствуйте, сегодня хорошая погода.', 'Russian'],
  ])('reads %s as its language', (_name, text, language) => {
    expect(previewLanguage(text)).toBe(language);
  });

  it.each<[string, PreviewLanguage]>([
    ['Welcome back to the show. Tonight we are walking through the harbour.', 'English'],
    ['Willkommen zurück in der Show. Heute Abend gehen wir durch den Hafen.', 'German'],
    ['Bienvenue dans l’émission. Ce soir, nous nous promenons dans le port.', 'French'],
    ['Bienvenidos de nuevo al programa. Esta noche caminamos por el puerto con los amigos.', 'Spanish'],
    ['Bentornati allo show. Questa sera passeggiamo per il porto con una guida.', 'Italian'],
    ['Bem-vindos de volta ao programa. Esta noite caminhamos pelo porto com uma guia, não é?', 'Portuguese'],
  ])('votes among the Latin languages: %s', (text, language) => {
    expect(previewLanguage(text)).toBe(language);
  });

  it('reads English for empty text, text without letters, and a tie', () => {
    expect(previewLanguage('')).toBe('English');
    expect(previewLanguage('  123 … !? ')).toBe('English');
    expect(previewLanguage('Hello')).toBe('English');
    expect(previewLanguage('xyzzy plugh')).toBe('English');
  });

  it('goes by the script with the most letters, and ignores [marks]', () => {
    expect(previewLanguage('Hello 你好')).toBe('English');
    expect(previewLanguage('你好，今天天气很好 ok')).toBe('Chinese');
    expect(previewLanguage('[pause] 今天天气很好 [laugh] haha')).toBe('Chinese');
  });

  it('recognises each default line as its own language', () => {
    for (const [language, line] of Object.entries(PREVIEW_LINES)) expect(previewLanguage(line)).toBe(language);
  });
});

describe('defaultPreviewLine', () => {
  it('gives the line of the take text language, English for no text', () => {
    expect(defaultPreviewLine('')).toBe(PREVIEW_LINES.English);
    expect(defaultPreviewLine('Guten Tag, meine Damen und Herren, ich bin hier.')).toBe(PREVIEW_LINES.German);
    expect(defaultPreviewLine('今天天气很好')).toBe(PREVIEW_LINES.Chinese);
  });

  it('has a line for each language, each under what the server takes in one part', () => {
    expect(Object.keys(PREVIEW_LINES).length).toBeGreaterThan(1);
    for (const line of Object.values(PREVIEW_LINES)) {
      expect(line.trim()).not.toBe('');
      expect(line.length).toBeLessThan(PREVIEW_MAX);
    }
  });

  it('has default lines that read for 3 to 30 seconds, so the default never warns', () => {
    for (const line of Object.values(PREVIEW_LINES)) expect(lineHint(line, RANGE, READING)?.warning).toBeNull();
  });
});

describe('lineHint', () => {
  it('says the read time in whole seconds', () => {
    expect(lineHint(PREVIEW_LINES.English, RANGE, READING)).toEqual({ read: 'About 5 seconds to read', warning: null });
  });

  it('warns under the shortest voice, in the words of the range', () => {
    expect(lineHint('Hello there.', RANGE, READING)).toEqual({ read: 'About 1 second to read', warning: 'A voice needs 3 to 30 seconds, so this one may not be saveable.' });
  });

  it('warns over the longest voice and reads minutes as a clock', () => {
    const hint = lineHint('word '.repeat(200), RANGE, READING);
    expect(hint?.warning).not.toBeNull();
    expect(hint?.read).toMatch(/^About \d+:\d\d to read$/);
    expect(lineHint('word '.repeat(80), RANGE, READING)?.read).toMatch(/^About \d+ seconds to read$/);
  });

  it('counts a Chinese, Japanese or Korean character for more than a letter, and leaves out [marks]', () => {
    const wide = lineHint('你'.repeat(40), RANGE, READING);
    const narrow = lineHint('a'.repeat(40), RANGE, READING);
    expect(wide?.read).not.toBe(narrow?.read);
    expect(lineHint('[pause]hello there you', RANGE, READING)?.read).toBe(lineHint('hello there you', RANGE, READING)?.read);
  });

  it('says nothing for an empty line or before the vocabulary has loaded', () => {
    expect(lineHint('   ', RANGE, READING)).toBeNull();
    expect(lineHint('Hello there, my friend.', RANGE, undefined)).toBeNull();
  });
});

describe('EXAMPLE_DESCRIPTIONS', () => {
  it('lists distinct examples, each a label and a description the server takes', () => {
    expect(EXAMPLE_DESCRIPTIONS.length).toBeGreaterThan(1);
    expect(new Set(EXAMPLE_DESCRIPTIONS.map((example) => example.label)).size).toBe(EXAMPLE_DESCRIPTIONS.length);
    for (const example of EXAMPLE_DESCRIPTIONS) {
      expect(example.label.trim()).not.toBe('');
      expect(example.text.length).toBeGreaterThan(10);
      expect(example.text.length).toBeLessThanOrEqual(DESCRIPTION_MAX);
    }
  });
});

describe('runLine', () => {
  const design = (overrides: Partial<ActiveJobState> = {}): ActiveJobState => ({
    job_id: 'job-design', workflow: 'voice_design', media_type: 'audio', prompt: 'A warm woman', model: 'qwen3_tts', runs: 3, created_at: '', status: 'running', phase: 'generate',
    audio: { phase: 'Design', step: 2, total: 3, part: 1, parts: 1 }, ...overrides,
  }) as ActiveJobState;

  it('reads as the job line for a first run', () => {
    expect(runLine(design(), false)).toBe('Voice 2 of 3');
  });

  it('says more once candidates are listed', () => {
    expect(runLine(design(), true)).toBe('Making 3 more: 2 of 3');
    expect(runLine(design({ audio: null }), true)).toBe('Making 3 more');
  });

  it('keeps the job line while it waits or loads', () => {
    expect(runLine(design({ phase: 'load' }), true)).toBe('Loading VoiceDesign…');
    expect(runLine(design({ phase: 'queue', status: 'queued' }), true)).toBe('Waiting in the queue');
  });
});
