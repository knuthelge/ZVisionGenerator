import { api } from './client';
import type { ModelInventory, ModelOperationResult } from '$lib/types';

export function getModelInventory(): Promise<ModelInventory> {
  return api.get<ModelInventory>('/api/models');
}

export function convertCheckpoint(data: {
  input_path: string;
  name?: string;
  model_type: string;
  base_model?: string;
  copy?: boolean;
}): Promise<ModelOperationResult> {
  return api.post<ModelOperationResult>('/api/models/convert', data);
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
}): Promise<ModelOperationResult> {
  return api.post<ModelOperationResult>('/api/models/import-lora/hf', data);
}
