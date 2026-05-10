import { api } from './client';
import type { PathPickerResult, PromptFileDocument, PromptFileInspection } from '$lib/types';

export function openPathPicker(data: {
  kind: 'existing_file' | 'directory';
  purpose: string;
  initial_path?: string | null;
}): Promise<PathPickerResult> {
  return api.post<PathPickerResult>('/api/picker', data);
}

export function inspectPromptFile(path: string): Promise<PromptFileInspection> {
  return api.post<PromptFileInspection>('/api/prompt-files/inspect', { path });
}

export function readPromptFile(path: string): Promise<PromptFileDocument> {
  return api.post<PromptFileDocument>('/api/prompt-files/read', { path });
}

export function writePromptFile(path: string, rawText: string): Promise<PromptFileInspection> {
  return api.put<PromptFileInspection>('/api/prompt-files/write', { path, raw_text: rawText });
}