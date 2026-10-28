import { shellQuote } from '$lib/state/cliCommand';
import { chipDownload, type UnlistedItems } from '$lib/state/loraDownloads';
import { familyLabel, loraMatches, modelLoraFamily, type LoraChip } from '$lib/state/loras';
import type { DownloadItem, HfCheck, HfFile, HfItem, HfRepo, LoraInfo, CuratedEntry, WorkspaceContext, Workflow } from '$lib/types';

/** The model a drawer is locked to: its LoRA family, the size or version that narrows it, and how the header names it. */
export interface DiscoverTarget {
  family: string;
  size: string | null;
  version: string | null;
  /** "Z-Image Turbo", "LTX-2.3": what "for …" and "Made for" say. */
  name: string;
  /** The family as a person names it. */
  familyName: string;
}

/** The LTX version of the video model *modelId*, as the server read it from the model; LTX-2.3 when the context does not say. */
function ltxVersion(context: WorkspaceContext, modelId: string): '2.3' | '2.5' {
  return context.video_models.find((model) => model.id === modelId)?.version === '2.5' ? '2.5' : '2.3';
}

/** Describe what Discover is locked to for *modelId*, or null when the model's LoRA family is not known. */
export function discoverTarget(context: WorkspaceContext, modelId: string, modelLabel: string): DiscoverTarget | null {
  const found = modelLoraFamily(context, modelId);
  if (!found?.family) return null;
  const version = found.family === 'ltx' ? ltxVersion(context, modelId) : null;
  return {
    family: found.family,
    size: found.sizeVariant,
    version,
    name: version ? `LTX-${version}` : modelLabel || familyLabel(found.family, found.sizeVariant),
    familyName: familyLabel(found.family, found.sizeVariant),
  };
}

/** What the file of a LoRA is on this computer: there, on its way, failed, not here, or not available under its name because a different file holds it. */
export type FileState = 'available' | 'installed' | 'downloading' | 'failed' | 'taken';

export interface UseFacts {
  chips: readonly LoraChip[];
  installed: readonly LoraInfo[];
  items: readonly DownloadItem[];
  /** Finished downloads whose file never showed in the LoRA list; see `chipDownload`. */
  unlisted?: UnlistedItems;
}

/** A LoRA as the drawer sees it: where its file is, how far a download is, and whether a chip uses it. */
export interface LoraStatus {
  file: FileState;
  /** How far a download is, 0 to 100, or null while its size is not known. */
  percent: number | null;
  inUse: boolean;
}

/**
 * The status of the LoRA saved as *name*. A name the server says a different file holds (*served* is `name_taken`) is
 * taken: that file is not this LoRA, so it is not installed, not in use and not this entry's to delete. Otherwise the
 * file on disk wins, then the lane (downloading, failed), then what the server said about the curated entry; a chip
 * makes it in use whatever the file is doing.
 */
export function loraStatus(name: string, served: 'available' | 'installed' | 'downloading' | 'name_taken' | null, facts: UseFacts): LoraStatus {
  if (served === 'name_taken') return { file: 'taken', percent: null, inUse: false };
  const inUse = facts.chips.some((chip) => chip.name === name);
  const download = chipDownload(name, facts.installed, facts.items, facts.unlisted);
  if (facts.installed.some((lora) => lora.name === name)) return { file: 'installed', percent: null, inUse };
  if (download.state === 'downloading') return { file: 'downloading', percent: download.percent, inUse };
  if (download.state === 'failed') return { file: 'failed', percent: null, inUse };
  if (served === 'installed') return { file: 'installed', percent: null, inUse };
  if (served === 'downloading') return { file: 'downloading', percent: null, inUse };
  return { file: 'available', percent: null, inUse };
}

/** Which kinds of LoRA a list shows. */
export type SectionFilter = 'all' | 'style' | 'tool';

/** Keep the items whose text holds every word of *query* (any order, any case). */
export function searchItems<T>(items: readonly T[], query: string, text: (item: T) => string): T[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (words.length === 0) return [...items];
  return items.filter((item) => {
    const haystack = text(item).toLowerCase();
    return words.every((word) => haystack.includes(word));
  });
}

/** The words a curated entry is found by: name, description, what it does, author and trigger. */
export function curatedText(entry: CuratedEntry): string {
  return [entry.display_name, entry.name, entry.description, entry.what_it_does, entry.author, entry.trigger].filter(Boolean).join(' ');
}

/** Keep the entries of the chosen kind that hold every word of *query*, in curated order. */
export function filterCurated(entries: readonly CuratedEntry[], query: string, section: SectionFilter): CuratedEntry[] {
  return searchItems(
    entries.filter((entry) => section === 'all' || entry.section === section),
    query,
    curatedText,
  );
}

const WORKFLOW_LABELS: Record<Workflow, string> = {
  txt2img: 'Text to Image',
  img2img: 'Image to Image',
  txt2vid: 'Text to Video',
  img2vid: 'Image to Video',
  extend: 'Extend Video',
};

export function workflowLabel(workflow: Workflow): string {
  return WORKFLOW_LABELS[workflow];
}

/** "1.2k", "34k", "980": counts as the lists show them. */
export function compactCount(value: number): string {
  if (value < 1000) return String(value);
  const thousands = value / 1000;
  return `${thousands.toFixed(thousands >= 10 ? 0 : 1).replace(/\.0$/, '')}k`;
}

/** The `ziv-model lora` line that downloads the same file, with the name it saves as. */
export function cliLine(repo: string, file: string, name: string): string {
  return ['ziv-model', 'lora', '--hf', repo, '--file', file, '--name', name].map((part) => shellQuote(part)).join(' ');
}

/** One tile of the Curated or Installed grid. */
export interface TileView {
  key: string;
  image: string | null;
  title: string;
  tool: boolean;
  /** One line: what a tool does, a style's description; null for a file of your own. */
  line: string | null;
  author: string | null;
  downloads: number | null;
  nonCommercial: boolean;
  status: LoraStatus;
}

/** The tile of a curated entry. */
export function curatedTile(entry: CuratedEntry, status: LoraStatus): TileView {
  const tool = entry.section === 'tool';
  return {
    key: entry.id,
    image: entry.previews[0] ?? null,
    title: entry.display_name,
    tool,
    line: tool ? entry.what_it_does || entry.description : entry.description,
    author: entry.author,
    downloads: entry.downloads,
    nonCommercial: entry.non_commercial,
    status,
  };
}

/** The curated entry an installed file is, when the list for the model holds it. */
export function curatedOf(lora: LoraInfo, entries: readonly CuratedEntry[]): CuratedEntry | null {
  return lora.curated_id ? (entries.find((entry) => entry.id === lora.curated_id) ?? null) : null;
}

/** The tile of an installed file: its curated entry's, what a Hugging Face download kept, or just its name. */
export function installedTile(lora: LoraInfo, entries: readonly CuratedEntry[], status: LoraStatus): TileView {
  const entry = curatedOf(lora, entries);
  // The tile opens the file, so it is keyed by the file's name whichever list entry describes it.
  if (entry) return { ...curatedTile(entry, status), key: lora.name };
  const hub = lora.hub ?? null;
  return { key: lora.name, image: hub?.thumbnail_url ?? null, title: hub?.name ?? lora.name, tool: false, line: null, author: hub?.author ?? null, downloads: hub?.downloads ?? null, nonCommercial: false, status };
}

/** The words an installed file is found by. */
export function installedText(lora: LoraInfo, entries: readonly CuratedEntry[]): string {
  const entry = curatedOf(lora, entries);
  return entry ? `${curatedText(entry)} ${lora.name}` : [lora.name, lora.hub?.name, lora.hub?.author, lora.trigger].filter(Boolean).join(' ');
}

/** The kind an installed file is listed under: its curated entry's, else a style (which only the All filter shows). */
export function installedKind(lora: LoraInfo, entries: readonly CuratedEntry[]): 'style' | 'tool' | 'other' {
  const entry = curatedOf(lora, entries);
  return entry ? entry.section : 'other';
}

/** Keep the installed files of the chosen kind that hold every word of *query*; a file of your own and a Hugging Face download show under All only. */
export function filterInstalled(loras: readonly LoraInfo[], entries: readonly CuratedEntry[], query: string, section: SectionFilter): LoraInfo[] {
  return searchItems(
    loras.filter((lora) => section === 'all' || installedKind(lora, entries) === section),
    query,
    (lora) => installedText(lora, entries),
  );
}

/**
 * The installed files of the model: `main` fit it (their family and size match), `other` are of a family nobody could
 * tell. A file made for another known model is in neither: it is not this model's.
 */
export function installedForModel(loras: readonly LoraInfo[], target: DiscoverTarget): { main: LoraInfo[]; other: LoraInfo[] } {
  const model = { family: target.family, sizeVariant: target.size };
  const byName = (a: LoraInfo, b: LoraInfo): number => a.name.localeCompare(b.name);
  return {
    main: loras.filter((lora) => loraMatches(lora, model)).sort(byName),
    other: loras.filter((lora) => !lora.family).sort(byName),
  };
}

/** What the detail shows, whether it came from the curated list, from Hugging Face or from your own files. */
export interface DetailView {
  title: string;
  tool: boolean;
  /** Under the name: a curated entry's description (what a tool does), or a Hugging Face repository's name. */
  lead: string | null;
  /** The lead is a repository id, set in the monospace face. */
  leadIsRepo: boolean;
  /** "Works with": the workflow a tool needs, or every workflow of the model. */
  worksWith: string | null;
  trigger: string | null;
  weight: number | null;
  /** The name the file saves as; once it is installed the row reads "File name". */
  saveAs: string;
  size: string | null;
  downloads: number | null;
  /** The licence as shown; null reads "None given"; undefined leaves the row out (a file of your own has none to give). */
  licence: string | null | undefined;
  nonCommercial: boolean;
  modelCardUrl: string | null;
  previews: string[];
  /** What the empty picture box says when there are no previews; null for a file of your own, which has none to give. */
  emptyNote: string | null;
  cli: string | null;
}

/** What the empty picture box says for a repository whose author added no images: that, or that only videos were added. */
function authorNote(videos: readonly string[]): string {
  return videos.length > 0 ? 'The author added only preview videos' : 'The author added no preview images';
}

/** The detail of a curated entry. */
export function curatedDetail(entry: CuratedEntry, modelName: string): DetailView {
  const tool = entry.section === 'tool';
  return {
    title: entry.display_name,
    tool,
    lead: tool ? entry.what_it_does || entry.description : entry.description,
    leadIsRepo: false,
    worksWith: tool ? (entry.workflow ? workflowLabel(entry.workflow) : `Every workflow of ${modelName}`) : null,
    trigger: entry.trigger,
    weight: entry.weight,
    saveAs: entry.name,
    size: entry.size_label,
    downloads: entry.downloads,
    licence: entry.licence_label,
    nonCommercial: entry.non_commercial,
    modelCardUrl: entry.model_card_url,
    previews: entry.previews,
    emptyNote: authorNote(entry.preview_videos),
    cli: cliLine(entry.repo, entry.file, entry.name),
  };
}

/** The detail of a file in a Hugging Face repository. */
export function hfDetail(repo: HfRepo, file: HfFile): DetailView {
  return {
    title: repo.name,
    tool: false,
    lead: repo.repo,
    leadIsRepo: true,
    worksWith: null,
    trigger: repo.instance_prompt,
    weight: 1,
    saveAs: file.save_as,
    size: file.size_label,
    downloads: repo.downloads,
    licence: repo.licence_label ?? null,
    nonCommercial: false,
    modelCardUrl: repo.model_card_url,
    previews: repo.previews,
    emptyNote: authorNote(repo.preview_videos),
    cli: cliLine(repo.repo, file.file, file.save_as),
  };
}

/** The detail of an installed file that is not a curated entry: what a Hugging Face download kept, or just the file. */
export function installedDetail(lora: LoraInfo): DetailView {
  const hub = lora.hub ?? null;
  const source = lora.source ?? null;
  return {
    title: hub?.name ?? lora.name,
    tool: false,
    lead: source?.repo ?? null,
    leadIsRepo: true,
    worksWith: null,
    trigger: lora.trigger ?? null,
    weight: hub ? 1 : null,
    saveAs: lora.name,
    size: lora.size_label ?? null,
    downloads: hub?.downloads ?? null,
    licence: hub ? (hub.licence_label ?? null) : undefined,
    nonCommercial: false,
    modelCardUrl: hub?.model_card_url ?? null,
    previews: hub?.thumbnail_url ? [hub.thumbnail_url] : [],
    emptyNote: hub ? 'The author added no preview images' : null,
    cli: source?.repo && source.file ? cliLine(source.repo, source.file, lora.name) : null,
  };
}

/** What the "Made for" row says: a match (green check), or another model (amber warning). */
export type MadeFor = { kind: 'match'; text: string } | { kind: 'mismatch'; text: string };

/** The row for a Hugging Face file, from the check of its header; null while that is unknown, so the row stays out. */
export function madeFor(check: HfCheck | null, target: Pick<DiscoverTarget, 'name'>): MadeFor | null {
  if (!check) return null;
  if (check.status === 'match') return { kind: 'match', text: target.name };
  if (check.status === 'mismatch') return { kind: 'mismatch', text: check.family ? familyLabel(check.family, check.size_variant) : 'Another model' };
  return null;
}

/** The row for an installed file, from the family the app detected in it; null when it could not tell. */
export function installedMadeFor(lora: LoraInfo, target: DiscoverTarget): MadeFor | null {
  if (!lora.family) return null;
  return loraMatches(lora, { family: target.family, sizeVariant: target.size }) ? { kind: 'match', text: target.name } : { kind: 'mismatch', text: familyLabel(lora.family, lora.size_variant) };
}

/** The sentence a row of a Hugging Face listing's meta line leaves out: one file, a licence. */
export function hfRowFacts(item: Pick<HfItem, 'files' | 'licence'>): { files: boolean; noLicence: boolean } {
  return { files: item.files > 1, noLicence: !item.licence };
}

/** The text of a detail's footer button while the file downloads: "Downloading 40 %". */
export function downloadingLabel(percent: number | null): string {
  return percent === null ? 'Downloading…' : `Downloading ${percent} %`;
}

export const DELETE_BLOCKED_TEXT = 'You can delete this when the current job is done.';
export const OFFLINE_TEXT = "You're offline.";

/** What the footer says when a different file already has the name a curated entry saves as. */
export function takenText(name: string): string {
  return `You already have a different file called ${name}.`;
}

/** What the detail's footer offers. */
export interface FooterView {
  /** The button on the left: Download, the progress of one, Retry download, or Delete file. */
  action: { kind: 'download' | 'downloading' | 'retry' | 'delete'; label: string; disabled: boolean };
  /** The "Use in LoRAs" switch. */
  use: { on: boolean; disabled: boolean };
  /** Why something is off: a job is running, or the computer is offline. */
  note: string | null;
}

export interface FooterFacts {
  status: LoraStatus;
  /** The name the file saves as. */
  name: string;
  offline: boolean;
  /** Delete is off while a job runs. */
  deleteBlocked: boolean;
  /** A request of the footer is on its way. */
  busy: boolean;
}

/** Decide what the footer offers from the file's status and the page's facts. */
export function footerView({ status, name, offline, deleteBlocked, busy }: FooterFacts): FooterView {
  const use = (disabled: boolean): FooterView['use'] => ({ on: status.inUse, disabled: disabled || busy });
  switch (status.file) {
    case 'installed':
      return { action: { kind: 'delete', label: 'Delete file', disabled: deleteBlocked || busy }, use: use(false), note: deleteBlocked ? DELETE_BLOCKED_TEXT : null };
    case 'downloading':
      return { action: { kind: 'downloading', label: downloadingLabel(status.percent), disabled: true }, use: use(false), note: null };
    case 'failed':
      return { action: { kind: 'retry', label: 'Retry download', disabled: offline || busy }, use: use(offline), note: offline ? OFFLINE_TEXT : null };
    case 'taken':
      return { action: { kind: 'download', label: 'Download', disabled: true }, use: { on: false, disabled: true }, note: takenText(name) };
    case 'available':
      return { action: { kind: 'download', label: 'Download', disabled: offline || busy }, use: use(offline), note: offline ? OFFLINE_TEXT : null };
  }
}
