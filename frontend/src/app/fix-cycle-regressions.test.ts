import { readFileSync } from 'node:fs';

import { beforeEach, describe, expect, it } from 'vitest';

import { jobStore } from '$lib/state/job.svelte';

function readSource(relativePath: string): string {
  return readFileSync(new URL(relativePath, import.meta.url), 'utf-8');
}

describe('Svelte 5 fix-cycle regressions', () => {
  beforeEach(() => {
    jobStore.clearJob();
  });

  it('keeps reference image drag-and-drop handlers on the workspace drop zone', () => {
    const source = readSource('../features/workspace/ControlsSidebar.svelte');

    expect(source).toContain('ondragover={handleDragOver}');
    expect(source).toContain('ondragleave={handleDragLeave}');
    expect(source).toContain('ondrop={handleDrop}');
    expect(source).toContain('const file = e.dataTransfer?.files[0] ?? null;');
  });

  it('uses shared token-backed workspace shell classes instead of repeating raw sidebar and toolbar chrome', () => {
    const controlsSource = readSource('../features/workspace/ControlsSidebar.svelte');
    const workspaceSource = readSource('../features/workspace/WorkspacePage.svelte');
    const historySource = readSource('../features/workspace/HistoryPane.svelte');

    expect(controlsSource).toContain('panel-shell panel-shell-left');
    expect(controlsSource).toContain('surface-dropzone');
    expect(controlsSource).toContain('surface-button-primary');
    expect(workspaceSource).toContain('panel-toolbar');
    expect(workspaceSource).toContain('surface-panel-frame');
    expect(historySource).toContain('panel-shell panel-shell-right');
    expect(historySource).toContain('panel-handle-button');
  });

  it('subscribes to pause and resume SSE events and flips paused state', () => {
    const sseSource = readSource('../lib/api/sse.ts');

    expect(sseSource).toContain("'job_paused'");
    expect(sseSource).toContain("'job_resumed'");

    jobStore.startJob({
      job_id: 'job-1',
      workflow: 'txt2img',
      prompt: 'Test prompt',
      model: 'zit',
      runs: 1,
      created_at: '2026-04-22T00:00:00Z'
    });

    const mockEventSource = (globalThis.EventSource as unknown as {
      lastInstance: { emit: (type: string, data: unknown) => void; close: () => void };
    }).lastInstance;

    mockEventSource.emit('job_paused', { type: 'job_paused', job_id: 'job-1' });
    expect(jobStore.current?.paused).toBe(true);

    mockEventSource.emit('job_resumed', { type: 'job_resumed', job_id: 'job-1' });
    expect(jobStore.current?.paused).toBe(false);

    mockEventSource.close();
  });

  it('uses per-model defaults across frontend types, workspace logic, and API contract tests', () => {
    const typesSource = readSource('../lib/types/index.ts');
    const workspaceSource = readSource('../features/workspace/WorkspacePage.svelte');
    const serverTestSource = readSource('../../../tests/test_web_server.py');

    expect(typesSource).toContain('image_model_defaults: Record<string, ImageModelDefaults>;');
    expect(typesSource).toContain('video_model_defaults: Record<string, VideoModelDefaults>;');
    expect(typesSource).toContain('workflow_contract: WorkflowContract;');
    expect(workspaceSource).toContain('draft.hydrateFromContext(context, newModel)');
    expect(workspaceSource).toContain('image_model_defaults');
    expect(serverTestSource).toContain('"image_model_defaults",');
  });

  it('does not append a stale reference image file after switching to a non-reference workflow', () => {
    const workspaceSource = readSource('../features/workspace/WorkspacePage.svelte');

    expect(workspaceSource).not.toContain("if (imageFile) {");
    expect(workspaceSource).toMatch(/imageFile\s*&&\s*\(draft\.state\.workflow === 'img2img'\s*\|\|\s*draft\.state\.workflow === 'img2vid'\)/);
  });
});