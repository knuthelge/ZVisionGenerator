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

    expect(typesSource).toContain('image_model_defaults: Record<string, ModelDefaults>;');
    expect(typesSource).toContain('video_model_defaults: Record<string, ModelDefaults>;');
    expect(workspaceSource).toContain('const defaultsMap = isImageMode ? context.image_model_defaults : context.video_model_defaults;');
    expect(workspaceSource).toContain('const modelDefaults = defaultsMap?.[newModel];');
    expect(serverTestSource).toContain('"image_model_defaults",');
  });
});