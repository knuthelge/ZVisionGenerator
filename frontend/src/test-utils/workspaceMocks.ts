import { vi } from 'vitest';
import type { DownloadItem, GalleryPage, JobContext, JobSnapshot, RecentPrompt, SettingsPreset, WorkflowMode, WorkspaceContext } from '$lib/types';

// The API mocks of the Workspace page tests. A test file wires them in with `vi.mock(path, async () => ...)`
// factories that import this module, because `vi.mock` only hoists inside a test file.

export const workspaceApiMocks = {
  getWorkspaceContext: vi.fn<() => Promise<WorkspaceContext>>(),
  getWorkspaceCoreContext: vi.fn<() => Promise<WorkspaceContext>>(),
  submitGenerate: vi.fn<(formData: FormData) => Promise<JobContext>>(),
  getJobSnapshot: vi.fn<(jobId: string) => Promise<JobSnapshot>>(),
  parseUrlPrefill: vi.fn<() => Record<string, string>>(),
  checkReferencePaths: vi.fn<(paths: string[]) => Promise<string[]>>(),
  listJobs: vi.fn(async () => ({ active_job: null as JobSnapshot | null, queued_jobs: [] as JobSnapshot[] })),
  clearQueue: vi.fn(async () => ({ cancelled: [] as string[] })),
  cancelJob: vi.fn(async () => undefined),
  skipFirstRun: vi.fn(async () => ({ skipped: true })),
};

/** Lets a test pick the phone layout, or a touch screen at a wide width (`coarse`); the page reads them when it mounts. */
export const layoutProbe = { compact: false, coarse: false };

export const configApiMocks = {
  updateConfig: vi.fn(async () => ({})),
};

export const browseApiMocks = {
  listDirectory: vi.fn(),
  browsePlaces: vi.fn(),
  createFolder: vi.fn(),
};

export const promptFileApiMocks = {
  inspectPromptFile: vi.fn(),
  writePromptFile: vi.fn(),
  appendPromptEntry: vi.fn(),
  createPromptFile: vi.fn(),
};

export const extendApiMocks = {
  inspectExtendSource: vi.fn(),
};

export const videoMemoryApiMocks = {
  checkVideoMemoryFit: vi.fn<(body: Record<string, unknown>) => Promise<{ memory_fit: { status: string; required_gb: number; budget_gb: number } | null; width: number; height: number; frames: number }>>(),
};

export const galleryApiMocks = {
  getGallery: vi.fn<(page?: number, filter?: string, sort?: string) => Promise<GalleryPage>>(),
  deleteAsset: vi.fn<(assetId: string) => Promise<void>>(),
};

export const enhanceApiMocks = {
  enhancePrompt: vi.fn<(body: unknown) => Promise<void>>(),
};

export const presetApiMocks = {
  listPresets: vi.fn<() => Promise<SettingsPreset[]>>(),
  createPreset: vi.fn<(body: unknown) => Promise<SettingsPreset>>(),
  updatePreset: vi.fn<(mode: SettingsPreset['mode'], slug: string, body: unknown) => Promise<SettingsPreset>>(),
  renamePreset: vi.fn<(mode: SettingsPreset['mode'], slug: string, name: string) => Promise<SettingsPreset>>(),
  deletePreset: vi.fn<(mode: SettingsPreset['mode'], slug: string) => Promise<{ status: string }>>(),
};

export const recentPromptsApiMocks = {
  getRecentPrompts: vi.fn<(query: string, mode: WorkflowMode) => Promise<{ entries: RecentPrompt[] }>>(),
  clearRecentPrompts: vi.fn<(mode: WorkflowMode) => Promise<{ deleted_count: number }>>(),
};

export const modelApiMocks = {
  getDownloads: vi.fn<() => Promise<{ items: DownloadItem[] }>>(),
};
