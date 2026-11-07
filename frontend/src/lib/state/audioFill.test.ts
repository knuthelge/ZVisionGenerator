import { describe, expect, it } from 'vitest';

import { AUDIO_SETTINGS } from '../../test-utils/audio';
import { AUDIO_FIELD_DEFAULTS, DEFAULT_SPEECH_DELIVERY, audioPrefill } from './audioDraft';
import {
  applyFill,
  canUndoFill,
  describeFolded,
  fieldList,
  fillContext,
  fillHint,
  fillRequest,
  fillSummary,
  fillTargets,
  fillable,
  isEmpty,
  keptFields,
  markedFields,
  undoFill,
  unmarkOnEdit,
} from './audioFill';
import { EMPTY_REWRITES, pushRewrite, rewriteStaleReason, rewriteUsed, sentText } from './audioRewrites';
import { jobSettingsPrefill } from './jobSettings';
import type { DraftState, FillField } from '$lib/types';

const SONG = '[Verse]\nI pack the boat at dawn\n\n[Chorus]\nGoodbye to the coast';
const FIELDS = { style: 'folk, acoustic guitar, wistful vocal', tempo: '84', key: 'Am', lyrics: SONG };
const SHOWN = { showLyrics: true, showTempo: true, showKey: true, showDelivery: true };

function state(overrides: Partial<DraftState> = {}): DraftState {
  return { ...AUDIO_FIELD_DEFAULTS, model: 'yue2', ...overrides } as DraftState;
}

const MUSIC: readonly FillField[] = ['style', 'lyrics', 'tempo', 'key'];
const fill = (current: DraftState, fields: Record<string, string> = FIELDS, targets: readonly FillField[] = MUSIC) =>
  applyFill(current, 'music', targets, { fields, vocabulary: AUDIO_SETTINGS });
const after = (current: DraftState, patch: Partial<DraftState>): DraftState => ({ ...current, ...patch });

describe('isEmpty', () => {
  it('counts a blank own Style as empty, whatever the Enhanced tab holds', () => {
    expect(isEmpty(state(), 'style')).toBe(true);
    expect(isEmpty(state({ audioStyle: '   \n' }), 'style')).toBe(true);
    expect(isEmpty(state({ audioStyle: 'folk' }), 'style')).toBe(false);
    const enhanced = { ...state().audioRewrites.style, text: 'Folk, guitar' };
    expect(isEmpty(state({ audioRewrites: { ...state().audioRewrites, style: enhanced } }), 'style')).toBe(true);
  });

  it('counts Lyrics with no words as empty, whether blank, whitespace or only section tags, and any words as the user’s', () => {
    expect(isEmpty(state({ lyrics: '' }), 'lyrics')).toBe(true);
    expect(isEmpty(state({ lyrics: ' \n\t ' }), 'lyrics')).toBe(true);
    expect(isEmpty(state({ lyrics: '[Verse]\n\n[Chorus]\n' }), 'lyrics')).toBe(true);
    expect(isEmpty(state({ lyrics: '[Verse]\nla la' }), 'lyrics')).toBe(false);
    expect(isEmpty(state({ lyrics: 'A title\n[Verse]\n' }), 'lyrics')).toBe(false);
  });

  it('writes Lyrics into a song of only tags, and sends those tags as the context', () => {
    const tags = state({ lyrics: '[Verse]\n\n[Chorus]\n', audioStyle: 'folk' });
    const shown = { showLyrics: true, showTempo: true, showKey: true, showDelivery: false };

    expect(fillTargets(tags, 'music', ['style', 'lyrics'])).toEqual(['lyrics']);
    expect(fillContext(tags, 'music', shown)).toMatchObject({ lyrics: '[Verse]\n\n[Chorus]\n' });
  });

  it('counts Auto as empty for Tempo and Key, and any set value as the user’s', () => {
    expect(isEmpty(state(), 'tempo')).toBe(true);
    expect(isEmpty(state({ audioTempo: 40 }), 'tempo')).toBe(false);
    expect(isEmpty(state(), 'key')).toBe(true);
    expect(isEmpty(state({ audioKey: 'C' }), 'key')).toBe(false);
  });

  it('counts a Vibe as empty only while all five delivery values are at their defaults', () => {
    expect(isEmpty(state(), 'vibe')).toBe(true);
    for (const field of ['emotion', 'manner', 'pace', 'pitch', 'expressiveness'] as const) {
      const value = field === 'emotion' ? 'sad' : field === 'manner' ? 'whisper' : field === 'pace' ? 'slow' : 'high';
      expect(isEmpty(state({ speechDelivery: { ...DEFAULT_SPEECH_DELIVERY, [field]: value } }), 'vibe'), field).toBe(false);
    }
    const quiet = { ...DEFAULT_SPEECH_DELIVERY, pace: 'slow' };
    expect(isEmpty(state({ speechDelivery: quiet }), 'vibe', quiet)).toBe(true);
  });

  it('counts a blank Text as empty', () => {
    expect(isEmpty(state(), 'text')).toBe(true);
    expect(isEmpty(state({ speechText: '  ' }), 'text')).toBe(true);
    expect(isEmpty(state({ speechText: 'Hello.' }), 'text')).toBe(false);
  });
});

describe('fillable', () => {
  it('leaves out what the open row does not show', () => {
    expect(fillable('music', SHOWN)).toEqual(['style', 'lyrics', 'tempo', 'key']);
    expect(fillable('music', { ...SHOWN, showLyrics: false })).toEqual(['style', 'tempo', 'key']);
    expect(fillable('music', { showLyrics: false, showTempo: false, showKey: false, showDelivery: false })).toEqual(['style']);
    expect(fillable('speech', SHOWN)).toEqual(['text', 'vibe']);
    expect(fillable('speech', SHOWN, true)).toEqual(['text']);
  });
});

describe('fillTargets', () => {
  it('asks for the requested fields that are empty, and keeps the user’s own', () => {
    const current = state({ audioKey: 'Em', lyrics: SONG });

    expect(fillTargets(current, 'music', MUSIC)).toEqual(['style', 'tempo']);
    expect(keptFields(current, 'music', MUSIC)).toEqual(['lyrics', 'key']);
  });

  it('asks only for what was requested', () => {
    expect(fillTargets(state(), 'music', ['style', 'tempo'])).toEqual(['style', 'tempo']);
  });

  it('asks again for a field that is still marked, but not for one the user edited', () => {
    const filled = state({ ...fill(state()).patch });
    expect(fillTargets(filled, 'music', MUSIC)).toEqual(['style', 'lyrics', 'tempo', 'key']);

    const edited = after(filled, { audioTempo: 90, lyrics: `${SONG}\nmore` });
    expect(fillTargets(edited, 'music', MUSIC)).toEqual(['style', 'key']);
  });

  it('has nothing to ask for when every field is the user’s', () => {
    const mine = state({ audioStyle: 'folk', lyrics: SONG, audioTempo: 80, audioKey: 'C' });

    expect(fillTargets(mine, 'music', MUSIC)).toEqual([]);
    expect(fillHint([], keptFields(mine, 'music', MUSIC))).toContain('nothing to write');
  });
});

describe('applyFill', () => {
  it('writes the Style into the own Style box, with no Enhanced version, so it is what Generate sends', () => {
    const current = state();
    const filled = after(current, fill(current).patch);

    expect(filled.audioStyle).toBe(FIELDS.style);
    expect(filled.audioRewrites).toEqual(current.audioRewrites);
  });

  it('makes an Enhanced text written before the fill out of date in the normal way', () => {
    const written = pushRewrite(state().audioRewrites.style, { text: 'Folk, piano', from: { input: '', model: 'yue2', task: 'style' } });
    const current = state({ audioRewrites: { ...state().audioRewrites, style: written } });
    const filled = after(current, fill(current).patch);

    expect(rewriteStaleReason(filled.audioRewrites.style, filled.audioStyle, filled.model, 'Style')).toBe('Your Style changed since this was written.');
  });

  it('sets the lyrics, tempo and key, and records the lyrics as a filled version', () => {
    const filled = after(state(), fill(state()).patch);

    expect(filled).toMatchObject({ lyrics: SONG, audioTempo: 84, audioKey: 'Am', lyricsCurrent: 0 });
    expect(filled.lyricsVersions).toEqual([{ text: SONG, mine: false }]);
  });

  it('adds a version to filled lyrics when it fills again, and replaces the filled Style', () => {
    let current = after(state(), fill(state()).patch);
    for (let round = 1; round <= 6; round += 1) current = after(current, fill(current, { ...FIELDS, style: `folk ${round}`, lyrics: `${SONG} ${round}` }).patch);

    expect(current.audioStyle).toBe('folk 6');
    expect(current.lyricsVersions.map((version) => version.text)).toEqual([2, 3, 4, 5, 6].map((round) => `${SONG} ${round}`));
  });

  it('writes only the targets, and ignores a field the answer lacks or cannot be read', () => {
    const current = state({ audioKey: 'Em' });

    const { patch, wrote } = fill(current, { style: FIELDS.style, tempo: 'fast', key: 'Am' }, ['style', 'tempo']);

    expect(wrote).toEqual(['style']);
    expect(patch.audioKey).toBeUndefined();
    expect(patch.audioTempo).toBeUndefined();
  });

  it('sets the five delivery values of a Vibe and the Text of a speech', () => {
    const current = state();

    const { patch, wrote } = applyFill(current, 'speech', ['text', 'vibe'], { fields: { vibe: 'whispered-secret', text: 'Come closer.' }, vocabulary: AUDIO_SETTINGS });

    expect(patch.speechText).toBe('Come closer.');
    expect(patch.speechDelivery).toEqual({ emotion: 'neutral', manner: 'whisper', pace: 'slow', pitch: 'low', expressiveness: 'low' });
    expect(wrote).toEqual(['text', 'vibe']);
  });

  it('ignores a Vibe the vocabulary does not serve', () => {
    const { wrote } = applyFill(state(), 'speech', ['vibe'], { fields: { vibe: 'villain' }, vocabulary: AUDIO_SETTINGS });

    expect(wrote).toEqual([]);
  });

  it('marks what it wrote, folds the box and keeps the idea', () => {
    const current = state({ audioDescribe: { ...state().audioDescribe, music: { ...state().audioDescribe.music, text: 'a sad folk song' } } });
    const filled = after(current, fill(current).patch);

    expect(filled.audioDescribe.music).toMatchObject({ text: 'a sad folk song', marked: ['style', 'lyrics', 'tempo', 'key'], folded: true });
    expect(markedFields(filled, 'music')).toEqual(['style', 'lyrics', 'tempo', 'key']);
    expect(filled.audioDescribe.speech).toEqual(current.audioDescribe.speech);
  });
});

describe('unmarkOnEdit', () => {
  const filled = after(state(), fill(state()).patch);

  it('keeps a mark while the field is exactly as Fill in wrote it', () => {
    expect(unmarkOnEdit(filled.audioDescribe.music, filled)).toBe(filled.audioDescribe.music);
  });

  it('takes the mark off a field the user edits, and only that one', () => {
    expect(markedFields(after(filled, { lyrics: `${SONG}!` }), 'music')).toEqual(['style', 'tempo', 'key']);
    expect(markedFields(after(filled, { audioTempo: 85 }), 'music')).toEqual(['style', 'lyrics', 'key']);
    expect(markedFields(after(filled, { audioKey: null }), 'music')).toEqual(['style', 'lyrics', 'tempo']);
  });

  it('takes the Style mark off when the user edits or empties the Style', () => {
    expect(markedFields(after(filled, { audioStyle: `${FIELDS.style}!` }), 'music')).not.toContain('style');
    expect(markedFields(after(filled, { audioStyle: '' }), 'music')).not.toContain('style');
  });

  it('takes the Vibe mark off when any delivery value changes', () => {
    const current = state();
    const spoken = after(current, applyFill(current, 'speech', ['vibe'], { fields: { vibe: 'storyteller' }, vocabulary: AUDIO_SETTINGS }).patch);
    expect(markedFields(spoken, 'speech')).toEqual(['vibe']);

    expect(markedFields(after(spoken, { speechDelivery: { ...spoken.speechDelivery, pace: 'fast' } }), 'speech')).toEqual([]);
  });
});

describe('Undo fill', () => {
  it('puts back exactly what was there', () => {
    const before = state({ audioKey: null, audioTempo: null, lyrics: '', audioStyle: '' });
    const filled = after(before, fill(before).patch);
    expect(canUndoFill(filled, 'music')).toBe(true);

    const undone = after(filled, undoFill(filled, 'music')!);

    expect(undone.audioStyle).toBe('');
    expect({ lyrics: undone.lyrics, lyricsVersions: undone.lyricsVersions, lyricsCurrent: undone.lyricsCurrent }).toEqual({ lyrics: '', lyricsVersions: [], lyricsCurrent: 0 });
    expect({ tempo: undone.audioTempo, key: undone.audioKey }).toEqual({ tempo: null, key: null });
    expect(undone.audioDescribe.music).toMatchObject({ marked: [], undo: null });
    expect(canUndoFill(undone, 'music')).toBe(false);
  });

  it('puts back the Style that was there, and leaves the Enhanced text alone', () => {
    const earlier = { text: 'Folk, piano', from: { input: '', model: 'yue2', task: 'style' as const }, history: [{ text: 'Folk, piano', from: { input: '', model: 'yue2', task: 'style' as const } }] };
    const before = state({ audioStyle: '', audioRewrites: { ...state().audioRewrites, style: earlier } });
    const filled = after(before, fill(before).patch);

    const undone = after(filled, undoFill(filled, 'music')!);

    expect(undone.audioStyle).toBe('');
    expect(undone.audioRewrites).toEqual(before.audioRewrites);
  });

  it('puts back only the fields the user has not edited since', () => {
    const before = state({ audioTempo: null, audioKey: null });
    const filled = after(before, fill(before).patch);
    const edited = after(filled, { audioTempo: 100 });

    const undone = after(edited, undoFill(edited, 'music')!);

    expect(undone.audioTempo).toBe(100);
    expect(undone.audioKey).toBeNull();
    expect(undone.lyrics).toBe('');
  });

  it('has nothing to undo once every field it wrote is the user’s, or when nothing was filled', () => {
    const filled = after(state(), fill(state()).patch);
    const mine = after(filled, { lyrics: 'x', audioTempo: 90, audioKey: 'C', audioStyle: 'mine' });

    expect(canUndoFill(mine, 'music')).toBe(false);
    expect(undoFill(mine, 'music')).toBeNull();
    expect(undoFill(state(), 'music')).toBeNull();
  });

  it('goes back to the marks of the fill before when a second fill is undone', () => {
    const first = after(state(), fill(state()).patch);
    const second = after(first, fill(first, { ...FIELDS, tempo: '120', lyrics: `${SONG}\n\n[Outro]\nSail on` }).patch);
    expect(second.audioTempo).toBe(120);

    const undone = after(second, undoFill(second, 'music')!);

    expect(undone.audioTempo).toBe(84);
    expect(undone.lyrics).toBe(SONG);
    expect(markedFields(undone, 'music')).toEqual(['style', 'lyrics', 'tempo', 'key']);
  });
});

describe('text of the box', () => {
  it('names the fields in the order of the summary', () => {
    expect(fieldList(['style', 'lyrics', 'tempo'])).toBe('Style, Lyrics and Tempo');
    expect(fieldList(['key'])).toBe('Key');
    expect(fieldList([])).toBe('');
  });

  it('says before the fill what will be filled and what is kept, and after it only the fields still marked', () => {
    expect(fillHint(['style', 'lyrics', 'tempo'], ['key'])).toBe('Will fill Style, Lyrics and Tempo · keeps your Key.');
    expect(fillSummary(['style', 'lyrics', 'tempo'])).toBe('Filled Style, Lyrics and Tempo');
    expect(fillSummary([])).toBe('Nothing was filled');
  });

  it('keeps the summary equal to what Undo fill restores after the user edits a filled field', () => {
    const filled = after(state(), fill(state()).patch);
    const edited = after(filled, { audioTempo: 90, audioStyle: 'my folk' });

    expect(fillSummary(markedFields(edited, 'music'))).toBe('Filled Lyrics and Key');
  });

  it('folds from the start while the Lyrics or Text have words, unless the user chose', () => {
    const music = state().audioDescribe.music;
    expect(describeFolded(music, state(), 'music')).toBe(false);
    expect(describeFolded(music, state({ lyrics: SONG }), 'music')).toBe(true);
    expect(describeFolded(music, state({ lyrics: '[Verse]\n\n[Chorus]' }), 'music')).toBe(false);
    expect(describeFolded({ ...music, folded: false }, state({ lyrics: SONG }), 'music')).toBe(false);
    expect(describeFolded({ ...music, folded: true }, state(), 'music')).toBe(true);
    expect(describeFolded(state().audioDescribe.speech, state({ speechText: 'Hello.' }), 'speech')).toBe(true);
  });
});

describe('the request', () => {
  it('sends the idea and the fields to fill, with what the user already set as context', () => {
    const current = state({ audioStyle: 'dark synth', audioKey: 'Em', audioTempo: 100, lyrics: SONG });

    const request = fillRequest('yue2', 'music', 'a night drive', ['tempo'], fillContext(current, 'music', SHOWN));

    expect(request).toEqual({
      mode: 'audio',
      task: 'fill',
      model: 'yue2',
      kind: 'music',
      prompt: 'a night drive',
      settings: { fields: ['tempo'] },
      context: { style: 'dark synth', lyrics: SONG, tempo: 100, key: 'Em' },
    });
  });

  it('sends nothing the open row hides, and no context when nothing is set', () => {
    const current = state({ audioTempo: 100, audioKey: 'Em', lyrics: SONG });

    expect(fillContext(current, 'music', { ...SHOWN, showLyrics: false, showTempo: false, showKey: false })).toEqual({});
    expect(fillRequest('sa3', 'music', 'calm', ['style'], {})).not.toHaveProperty('context');
  });

  it('tells a Speech fill the text so far', () => {
    expect(fillContext(state({ speechText: 'Come closer.' }), 'speech', SHOWN)).toEqual({ text: 'Come closer.' });
    expect(fillContext(state(), 'speech', SHOWN)).toEqual({});
  });

  it('cuts the context to the lengths the server keeps', () => {
    expect(fillContext(state({ lyrics: 'x'.repeat(900) }), 'music', SHOWN).lyrics).toHaveLength(600);
    expect(fillContext(state({ speechText: 'x'.repeat(900) }), 'speech', SHOWN).text).toHaveLength(600);
  });
});

describe('a take Fill in made, read back', () => {
  // The form a Generate after a fill posts: the filled Style is the take's own style, with no rewrite behind it.
  const musicForm = { mode: 'audio', workflow: 'music', model: 'yue2', prompt: FIELDS.style, lyrics: SONG, tempo: FIELDS.tempo, key: FIELDS.key };

  /** What Retry makes of the form a job was posted with, as the strings Reuse reads. */
  const retryParams = (form: Record<string, string>): Record<string, string | undefined> => jobSettingsPrefill(form).params as Record<string, string | undefined>;

  it('Retry puts back the filled Style as the own Style, the lyrics, the Tempo and the Key, with no Enhanced text', () => {
    const { patch } = audioPrefill(retryParams(musicForm), null, 'music');

    expect(patch).toMatchObject({ audioStyle: FIELDS.style, lyrics: SONG, audioTempo: 84, audioKey: 'Am' });
    expect(rewriteUsed(patch.audioRewrites?.style ?? EMPTY_REWRITES)).toBe(false);
  });

  it('sends the filled Style as the own text', () => {
    const filled = after(state(), fill(state()).patch);

    expect(sentText(filled.audioRewrites.style, filled.audioStyle)).toEqual({ text: FIELDS.style, input: null, record: null });
    expect(sentText(filled.audioRewrites.style, 'my own folk')).toEqual({ text: 'my own folk', input: null, record: null });
  });

  it('puts back a speech the Text and the five Vibe values it was made with', () => {
    const form = { mode: 'audio', workflow: 'speech', model: 'qwen3_tts', prompt: 'Come closer, Marta.', speaker: 'ryan', emotion: 'neutral', manner: 'whisper', pace: 'slow', expressiveness: 'low', pitch: 'low' };
    const { patch } = audioPrefill(retryParams(form), null, 'speech');

    expect(patch.speechText).toBe('Come closer, Marta.');
    expect(patch.speechDelivery).toEqual({ emotion: 'neutral', manner: 'whisper', pace: 'slow', expressiveness: 'low', pitch: 'low' });
  });
});
