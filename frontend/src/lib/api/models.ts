import { api } from './client';
import type { LoraImportResult, ConversionStatus, DownloadItem, DownloadList, DownloadPlan, CheckpointInspection, ModelInventory, ModelOperationResult } from '$lib/types';

export function getModelInventory(): Promise<ModelInventory> {
  return api.get<ModelInventory>('/api/models');
}

export function convertCheckpoint(data: {
  input_path: string;
  name?: string;
  model_type: string;
  base_model?: string;
  copy?: boolean;
  /** '4' or '8' also saves a quantized copy; empty for none. For ltx-2.3 and ltx-2.5 it is required and is the only level stored. */
  quantize?: string;
  /** ltx-2.3 and ltx-2.5 only: merge Lightricks' distilled LoRA into a plain-dev checkpoint. */
  distill?: boolean;
  /** h3 only: the checkpoint is already step-distilled, so the turbo LoRA is not merged. */
  turbo?: boolean;
}): Promise<ModelOperationResult> {
  return api.post<ModelOperationResult>('/api/models/convert', data);
}

/** Inspect a video checkpoint as *modelType* before converting it (nothing is downloaded). */
export function inspectCheckpoint(inputPath: string, modelType: string): Promise<CheckpointInspection> {
  return api.post<CheckpointInspection>('/api/models/convert/inspect', { input_path: inputPath, model_type: modelType });
}

/** The state and progress of the current or last video conversion. */
export function getConversionStatus(): Promise<ConversionStatus> {
  return api.get<ConversionStatus>('/api/models/convert/status');
}

export function importLoraLocal(data: {
  source_path: string;
  name?: string;
}): Promise<ModelOperationResult> {
  return api.post<ModelOperationResult>('/api/models/import-lora/local', data);
}

export function importLoraHF(data: {
  repo_id: string;
  filename?: string;
  name?: string;
}): Promise<LoraImportResult> {
  return api.post<LoraImportResult>('/api/models/import-lora/hf', data);
}

export function deleteModel(name: string): Promise<ModelOperationResult> {
  return api.delete<ModelOperationResult>(`/api/models/${encodeURIComponent(name)}`);
}

export function deleteEnhancerModel(tier: string): Promise<ModelOperationResult> {
  return api.delete<ModelOperationResult>(`/api/enhancer-models/${encodeURIComponent(tier)}`);
}

export function deleteOtherDownload(repoId: string): Promise<ModelOperationResult> {
  return api.delete<ModelOperationResult>(`/api/hf-cache/${repoId.split('/').map(encodeURIComponent).join('/')}`);
}

export function freeStrandedCacheFiles(): Promise<ModelOperationResult> {
  return api.delete<ModelOperationResult>('/api/hf-cache/stranded');
}

export function deleteLora(name: string): Promise<ModelOperationResult> {
  return api.delete<ModelOperationResult>(`/api/loras/${encodeURIComponent(name)}`);
}

/** What downloading a model or enhancer would fetch: parts, sizes and free space. */
export function getDownloadPlan(kind: 'model' | 'enhancer', name: string): Promise<DownloadPlan> {
  return api.get<DownloadPlan>(`/api/models/download/plan?kind=${encodeURIComponent(kind)}&name=${encodeURIComponent(name)}`);
}

/** Queue a Models-page download; an already queued or running one for the same row is returned. */
export function startDownload(kind: 'model' | 'enhancer', name: string): Promise<DownloadItem> {
  return api.post<DownloadItem>('/api/models/download', { kind, name });
}

/** The downloads lane: queued, running and recently finished items. */
export function getDownloads(): Promise<DownloadList> {
  return api.get<DownloadList>('/api/models/downloads');
}

export function cancelDownload(id: string): Promise<DownloadItem> {
  return api.post<DownloadItem>(`/api/models/downloads/${encodeURIComponent(id)}/cancel`, {});
}
