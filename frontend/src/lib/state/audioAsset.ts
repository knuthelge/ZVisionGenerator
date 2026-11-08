import type { AudioModelOption, GalleryAsset } from '$lib/types';
import { formatClock } from './clock';
import { isSectionLine } from './musicMode';

/** The lyric lines a tile shows: every non-blank line that is not a `[section]` marker. */
export function lyricLines(lyrics: string | null | undefined): string[] {
  return (lyrics ?? '').split(/\r?\n/).map((line) => line.trim()).filter((line) => line !== '' && !isSectionLine(line));
}

/**
 * The words a take's tile shows, one per line: a song's style then its lyric lines without the `[section]` lines, the text of
 * speech, the prompt of a sound. Empty for a file without this app's settings (its prompt is only its file name).
 */
export function tileWords(asset: GalleryAsset): string {
  if (asset.kind == null) return '';
  const prompt = (asset.prompt ?? '').trim();
  return [prompt, ...(asset.kind === 'music' ? lyricLines(asset.lyrics) : [])].filter((line) => line !== '').join('\n');
}

/** The text of the viewer's words and the stage caption: a song's style (its lyrics follow in the viewer), the speech text, the sound prompt. */
export function takeWords(asset: GalleryAsset): string {
  return asset.kind == null ? '' : (asset.prompt ?? '').trim();
}

/**
 * The foot's second line: `music · 01/11/2026`, or `audio · no settings` for a file this app did not make. With
 * `showProject`, a file in a project leads with the project's name in place of the kind word.
 */
export function audioFoot(asset: GalleryAsset, date: string, showProject = false): string {
  const lead = showProject && asset.project !== null ? asset.project : (asset.kind ?? 'audio');
  return asset.kind == null ? `${lead} · no settings` : `${lead} · ${date}`;
}

/** Whether a tile has nothing to read: a file without settings, or one whose words are empty. It shows only its icon and waveform. */
export function hasNoWords(asset: GalleryAsset): boolean {
  return tileWords(asset) === '';
}

/** The row of the model that made a take, as the kind it made: a model of several kinds has one row each, with their own limits. A take without a kind (a file the app did not make) matches by id alone. */
export function audioRowOf(asset: Pick<GalleryAsset, 'model'> & Partial<Pick<GalleryAsset, 'kind'>>, models: readonly AudioModelOption[] | undefined): AudioModelOption | undefined {
  return models?.find((item) => item.id === asset.model && (asset.kind == null || item.kind === asset.kind));
}

/** The name of the model that made a take: the label the server gives it, else the label this machine's model list has, else the id the file records. */
export function audioModelLabel(asset: Pick<GalleryAsset, 'model' | 'model_label'> & Partial<Pick<GalleryAsset, 'kind'>>, models: readonly AudioModelOption[] | undefined): string {
  return asset.model_label || audioRowOf(asset, models)?.label || asset.model;
}

/** What a cut take's badge and tooltip say. */
interface CutNote {
  text: string;
  tip: string;
}

/**
 * Describe where a take that reached its model's cap was cut, or null for a take that was not. The cap is the model's own
 * ceiling when the Workspace knows it, else the take's length.
 */
export function cutNote(asset: GalleryAsset, models: readonly AudioModelOption[] | undefined): CutNote | null {
  if (asset.cut !== true) return null;
  const model = audioRowOf(asset, models);
  const cap = (model && model.kind !== 'speech' ? model.limits.max_seconds : null) ?? asset.duration ?? null;
  if (cap === null) return null;
  const at = formatClock(cap);
  return { text: `⚠ Cut at ${at}`, tip: `${audioModelLabel(asset, models)} stops at ${at}. Remove sections to hear its ending.` };
}

/** The viewer's length fact: `2:00`, with "(cut at the cap)" for a take that was cut. */
export function lengthFact(asset: GalleryAsset): string {
  if (asset.duration == null) return '—';
  return `${formatClock(asset.duration)}${asset.cut === true ? ' (cut at the cap)' : ''}`;
}

/** The viewer's File fact: `FLAC · 24 kHz · mono`; a part that is not known is left out. */
export function formatFact(asset: GalleryAsset): string {
  const extension = asset.filename.includes('.') ? asset.filename.split('.').pop()!.toUpperCase() : 'Audio';
  const rate = asset.sample_rate ? `${Number((asset.sample_rate / 1000).toFixed(1))} kHz` : null;
  const channels = asset.channels ? (asset.channels === 1 ? 'mono' : asset.channels === 2 ? 'stereo' : `${asset.channels} channels`) : null;
  return [extension, rate, channels].filter((part) => part !== null).join(' · ');
}

const KIND_NAMES = { music: 'Music', speech: 'Speech', sound: 'Sound' } as const;

/** The kind's name for the viewer's Kind fact: "Music", "Speech", "Sound"; "Audio" without settings. */
export function kindFact(asset: GalleryAsset): string {
  return asset.kind == null ? 'Audio' : KIND_NAMES[asset.kind];
}

/** The fraction (0 to 1) of a waveform's width a pointer at *clientX* is over, for seeking. */
export function seekFraction(clientX: number, box: { left: number; width: number }): number {
  if (!(box.width > 0)) return 0;
  return Math.min(1, Math.max(0, (clientX - box.left) / box.width));
}

/** One labelled block of a take's words under the viewer's player. */
interface WordBlock {
  label: string;
  text: string;
}

/**
 * The words under the viewer's player: a song's style and lyrics (with their `[section]` lines), the text of speech, the prompt
 * of a sound. Empty for a file without this app's settings.
 */
export function wordBlocks(asset: GalleryAsset): WordBlock[] {
  if (asset.kind == null) return [];
  const prompt = (asset.prompt ?? '').trim();
  const lyrics = (asset.lyrics ?? '').trim();
  const label = asset.kind === 'speech' ? 'Text' : asset.kind === 'sound' ? 'Prompt' : 'Style';
  return [
    ...(prompt !== '' ? [{ label, text: prompt }] : []),
    ...(asset.kind === 'music' && lyrics !== '' ? [{ label: 'Lyrics', text: lyrics }] : []),
  ];
}
