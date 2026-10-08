import { api } from './client';
import type {
  PathPickerResult,
  PromptDocument,
  PromptDocumentLoad,
  PromptDocumentPreview,
  PromptDocumentSaved,
  PromptFileInspection,
} from '$lib/types';

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

export function writePromptFile(path: string, rawText: string): Promise<PromptFileInspection> {
  return api.put<PromptFileInspection>('/api/prompt-files/write', { path, raw_text: rawText });
}

export function loadPromptDocument(path: string): Promise<PromptDocumentLoad> {
  return api.post<PromptDocumentLoad>('/api/prompt-files/document', { path });
}

export function savePromptDocument(data: {
  path: string;
  revision: string;
  document: PromptDocument;
  /** Overwrite a file that changed on disk, applying the edits onto `base_text` (the text that was loaded). */
  force?: boolean;
  base_text?: string;
}): Promise<PromptDocumentSaved> {
  return api.put<PromptDocumentSaved>('/api/prompt-files/document', data);
}

export function previewPromptDocument(document: PromptDocument, rollEntryId: string | null = null): Promise<PromptDocumentPreview> {
  return api.post<PromptDocumentPreview>('/api/prompt-files/preview', { document, roll_entry_id: rollEntryId });
}

export function createPromptFile(directory: string, name: string): Promise<{ path: string }> {
  return api.post<{ path: string }>('/api/prompt-files/create', { directory, name });
}
