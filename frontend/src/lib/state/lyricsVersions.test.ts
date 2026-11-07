import { describe, expect, it } from 'vitest';
import { LYRICS_VERSION_LIMIT, editLyrics, inStep, pushVersion, showVersion, versionStepper, writtenByHelper } from './lyricsVersions';
import type { LyricsVersion } from '$lib/types';

type State = { lyrics: string; lyricsVersions: LyricsVersion[]; lyricsCurrent: number };

const fresh = (lyrics: string): State => ({ lyrics, lyricsVersions: [], lyricsCurrent: 0 });
/** Apply a patch the way the draft does. */
const apply = (state: State, patch: Partial<State>): State => ({ ...state, ...patch });

/** Run the helper *n* times from *lyrics*; each result is `v<k>`. */
function improved(lyrics: string, n: number): State {
  let state = fresh(lyrics);
  for (let k = 1; k <= n; k += 1) state = apply(state, { ...pushVersion(state, `v${k}`), lyrics: `v${k}` });
  return state;
}

describe('pushVersion', () => {
  it('saves the user\'s lyrics as mine with the first result, and shows the result', () => {
    const state = fresh('my words');

    const patch = pushVersion(state, 'better words');

    expect(patch.lyricsVersions).toEqual([{ text: 'my words', mine: true }, { text: 'better words', mine: false }]);
    expect(patch.lyricsCurrent).toBe(1);
  });

  it('adds each later result as the newest version, from the shown lyrics', () => {
    const second = improved('my words', 2);

    expect(second.lyricsVersions.map((version) => version.text)).toEqual(['my words', 'v1', 'v2']);
    expect(second.lyricsCurrent).toBe(2);
    expect(second.lyricsVersions.map((version) => version.mine)).toEqual([true, false, false]);
  });

  it('keeps at most five versions: the user\'s own and the newest results', () => {
    const state = improved('my words', 7);

    expect(LYRICS_VERSION_LIMIT).toBe(5);
    expect(state.lyricsVersions.map((version) => version.text)).toEqual(['my words', 'v4', 'v5', 'v6', 'v7']);
    expect(state.lyricsCurrent).toBe(4);
  });

  it('adds a result after the newest even when an earlier version is shown', () => {
    const state = improved('mine', 3);
    const back = apply(state, showVersion(state, 2)!);

    const next = apply(back, { ...pushVersion(back, 'v-new'), lyrics: 'v-new' });

    expect(next.lyricsVersions.map((version) => version.text)).toEqual(['mine', 'v1', 'v2', 'v3', 'v-new']);
    expect(next.lyricsCurrent).toBe(4);
  });

  it('starts a new set from the shown lyrics when something else changed them', () => {
    const state = apply(improved('mine', 2), { lyrics: 'a recent prompt\'s lyrics' });

    const patch = pushVersion(state, 'improved recent');

    expect(patch.lyricsVersions).toEqual([{ text: 'a recent prompt\'s lyrics', mine: true }, { text: 'improved recent', mine: false }]);
    expect(patch.lyricsCurrent).toBe(1);
  });

  it('saves nothing as mine when the lyrics were empty', () => {
    expect(pushVersion(fresh('  '), 'new')).toEqual({ lyricsVersions: [{ text: 'new', mine: false }], lyricsCurrent: 0 });
  });
});

describe('versionStepper', () => {
  it('counts from 1 over every version, mine included', () => {
    const state = improved('mine', 2);

    expect(versionStepper(state)).toEqual({ index: 3, count: 3 });
    expect(versionStepper(apply(state, showVersion(state, 1)!))).toEqual({ index: 1, count: 3 });
  });

  it('has none before there are two versions or when the versions are out of step', () => {
    expect(versionStepper(fresh('mine'))).toBeNull();
    expect(versionStepper({ lyrics: 'x', lyricsVersions: [{ text: 'x', mine: true }], lyricsCurrent: 0 })).toBeNull();
    expect(versionStepper(apply(improved('mine', 2), { lyrics: 'changed elsewhere' }))).toBeNull();
  });
});

describe('showVersion', () => {
  it('shows the lyrics of a version and says which one is current', () => {
    const state = improved('mine', 2);

    expect(showVersion(state, 2)).toMatchObject({ lyrics: 'v1', lyricsCurrent: 1 });
    expect(showVersion(state, 2)!.lyricsVersions).toBe(state.lyricsVersions);
  });

  it('ignores an index outside the versions', () => {
    const state = improved('mine', 1);

    expect(showVersion(state, 0)).toBeNull();
    expect(showVersion(state, 3)).toBeNull();
  });
});

describe('writtenByHelper', () => {
  it('is true while the version shown is a helper\'s, and false for the user\'s own', () => {
    const state = improved('my words', 2);

    expect(writtenByHelper(state)).toBe(true);
    expect(writtenByHelper(apply(state, showVersion(state, 1)!))).toBe(false);
  });

  it('is false without versions or when they are out of step', () => {
    expect(writtenByHelper(fresh('mine'))).toBe(false);
    expect(writtenByHelper(apply(improved('mine', 1), { lyrics: 'elsewhere' }))).toBe(false);
  });

  it('goes once the shown version is typed in, as it is the user\'s then', () => {
    const state = improved('mine', 2);

    const typed = apply(state, editLyrics(state, 'v2, with my words'));

    expect(writtenByHelper(typed)).toBe(false);
    expect(typed.lyricsVersions.map((version) => version.mine)).toEqual([true, false, true]);
    expect(writtenByHelper(apply(typed, showVersion(typed, 2)!))).toBe(true);
  });
});

describe('editLyrics', () => {
  it('changes the shown version, so it is found again after stepping away', () => {
    const state = improved('mine', 2);

    const edited = apply(state, editLyrics(state, 'v2 with my fix'));

    expect(edited.lyrics).toBe('v2 with my fix');
    expect(edited.lyricsVersions.map((version) => version.text)).toEqual(['mine', 'v1', 'v2 with my fix']);
    expect(inStep(edited)).toBe(true);
    expect(apply(edited, showVersion(edited, 2)!).lyrics).toBe('v1');
    expect(apply(edited, showVersion(edited, 3)!).lyrics).toBe('v2 with my fix');
  });

  it('edits mine when mine is shown', () => {
    const state = improved('mine', 1);
    const onMine = apply(state, showVersion(state, 1)!);

    expect(editLyrics(onMine, 'mine, edited').lyricsVersions![0]).toEqual({ text: 'mine, edited', mine: true });
  });

  it('only sets the lyrics when there are no versions or they are out of step', () => {
    expect(editLyrics(fresh('a'), 'ab')).toEqual({ lyrics: 'ab' });
    expect(editLyrics(apply(improved('mine', 1), { lyrics: 'elsewhere' }), 'elsewhere!')).toEqual({ lyrics: 'elsewhere!' });
  });

  it('does not change the other versions', () => {
    const state = improved('mine', 2);

    const { lyricsVersions } = editLyrics(state, 'x');

    expect(lyricsVersions![0]).toBe(state.lyricsVersions[0]);
    expect(lyricsVersions![1]).toBe(state.lyricsVersions[1]);
  });
});
