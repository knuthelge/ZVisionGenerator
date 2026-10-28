import { api } from './client';
import type { DiscoverDownload, HfCheck, HfPage, HfRepo, HfSort, LoraTriggerResult, CuratedList } from '$lib/types';

/** Whether Hugging Face LoRAs are shown; the server keeps the answer, so it also applies when the app is opened from another device. */
export function setDiscoverSettings(hfEnabled: boolean): Promise<{ hf_enabled: boolean }> {
  return api.post<{ hf_enabled: boolean }>('/api/discover/settings', { hf_enabled: hfEnabled });
}

/** The model a Discover request is about: its family, and the size or version where it has one. */
export interface DiscoverModel {
  family: string;
  size?: string | null;
  version?: string | null;
}

function query(params: Record<string, string | null | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) if (value) search.set(key, value);
  return search.toString();
}

export function getCurated(model: DiscoverModel): Promise<CuratedList> {
  return api.get<CuratedList>(`/api/discover/curated?${query({ family: model.family, size: model.size, version: model.version })}`);
}

export function getHfList(model: DiscoverModel, options: { q?: string; sort?: HfSort; cursor?: string | null } = {}): Promise<HfPage> {
  return api.get<HfPage>(`/api/discover/hf?${query({ family: model.family, size: model.size, version: model.version, q: options.q, sort: options.sort, cursor: options.cursor })}`);
}

export function getHfRepo(repo: string): Promise<HfRepo> {
  return api.get<HfRepo>(`/api/discover/hf/repo?${query({ repo })}`);
}

/** Which model a repository file was made for, read from its header before anything downloads. */
export function getHfCheck(repo: string, file: string, model: DiscoverModel): Promise<HfCheck> {
  return api.get<HfCheck>(`/api/discover/hf/check?${query({ repo, file, family: model.family, size: model.size })}`);
}

/** What to download: a curated entry, or one file of a repository under an optional local name. */
export type DiscoverSource = { curated_id: string } | { repo: string; file: string; name?: string };

export function startDiscoverDownload(source: DiscoverSource): Promise<DiscoverDownload> {
  return api.post<DiscoverDownload>('/api/discover/download', source);
}

/** Set the trigger the user typed for an installed LoRA; null or blank clears it. */
export function setLoraTrigger(name: string, trigger: string | null): Promise<LoraTriggerResult> {
  return api.put<LoraTriggerResult>(`/api/loras/${encodeURIComponent(name)}/trigger`, { trigger });
}
