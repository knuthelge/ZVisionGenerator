// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { JobSnapshot } from '$lib/types';

import QueuePanel from './QueuePanel.svelte';

function makeJob(overrides: Partial<JobSnapshot> = {}): JobSnapshot {
  return {
    id: 'job-1',
    job_id: 'job-1',
    job_type: 'Text to Image',
    workflow: 'txt2img',
    prompt: 'a fox',
    model: 'zit',
    runs: 1,
    created_at: '2026-10-04T12:00:00Z',
    status: 'queued',
    event_count: 1,
    paused: false,
    settings: { prompt: 'a fox' },
    ...overrides,
  };
}

describe('QueuePanel', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(async () => {
    if (app) { await unmount(app); app = null; }
    target.remove();
  });

  function mountPanel(jobs: JobSnapshot[], extra: Record<string, unknown> = {}): void {
    app = flushSync(() => mount(QueuePanel, { target, props: { jobs, onremove: vi.fn(), onload: vi.fn(), onclear: vi.fn(), ...extra } }));
  }

  function loadButtons(): HTMLButtonElement[] {
    return Array.from(target.querySelectorAll<HTMLButtonElement>('button[aria-label="Load settings"]'));
  }

  const metaParts = (row = 0): string[] => Array.from(target.querySelectorAll('[data-job-id]')[row].querySelectorAll('[data-testid="queue-meta"] span')).map((part) => part.textContent ?? '');

  it('offers Load settings for jobs submitted from the form', () => {
    mountPanel([makeJob()]);
    expect(loadButtons()).toHaveLength(1);
  });

  it('shows a queued design job by its description, its model, the New voice label and its voices, with no Load settings', () => {
    mountPanel([makeJob({
      workflow: 'voice_design', job_type: 'Voice design', prompt: 'A warm woman in her thirties', model: 'qwen3_tts', model_label: 'Qwen3-TTS 1.7B VoiceDesign', runs: 3, meta: '3 voices', settings: {},
    })]);

    expect(target.querySelector('[data-testid="queue-prompt"]')?.textContent?.trim()).toBe('A warm woman in her thirties');
    expect(metaParts()).toEqual(['New voice', 'Qwen3-TTS 1.7B VoiceDesign', '3 voices']);
    expect(loadButtons()).toHaveLength(0);
  });

  it('names the model by its label in the queue row, falling back to its id', () => {
    mountPanel([makeJob({ model: 'klein9b', model_label: 'FLUX.2 Klein 9B' }), makeJob({ id: 'job-2', job_id: 'job-2', model: 'zit' })]);

    expect(metaParts(0)).toContain('FLUX.2 Klein 9B');
    expect(metaParts(0)).not.toContain('klein9b');
    expect(metaParts(1)).toContain('zit');
  });

  it('names the project a queued job saves into, only when it has one', () => {
    mountPanel([makeJob({ save_project: 'Night Bus' }), makeJob({ id: 'job-2', job_id: 'job-2', save_project: null }), makeJob({ id: 'job-3', job_id: 'job-3' })]);

    const rows = Array.from(target.querySelectorAll('[data-job-id]'));
    expect(rows[0].querySelector('[data-testid="queue-project"]')?.getAttribute('title')).toBe('Saves to Night Bus');
    expect(rows[1].querySelector('[data-testid="queue-project"]')).toBeNull();
    expect(rows[2].querySelector('[data-testid="queue-project"]')).toBeNull();
  });

  it('labels a queued upscale and offers no settings to load', () => {
    mountPanel([makeJob({ workflow: 'upscale', job_type: 'Upscale', prompt: '', settings: {}, meta: '2× → 1664×2432' })]);

    expect(loadButtons()).toHaveLength(0);
    expect(target.querySelector('[data-testid="queue-prompt"]')?.textContent?.trim()).toBe('No prompt');
    expect(metaParts()).toContain('2× → 1664×2432');
  });

  it('marks the next job\'s position and numbers the others', () => {
    mountPanel([makeJob(), makeJob({ id: 'job-2', job_id: 'job-2', queue_position: 2 })]);

    const positions = Array.from(target.querySelectorAll('[role="img"][aria-label^="Position"]')).map((mark) => mark.getAttribute('aria-label'));
    expect(positions).toEqual(['Position 1, starts next', 'Position 2']);
  });

  it('shows the name of the first file a job starts from, and how many more, with no picture', () => {
    const inputs = [
      { role: 'start' as const, name: 'door.png', kind: 'image' as const },
      { role: 'reference' as const, name: 'fox.png', kind: 'image' as const },
    ];
    mountPanel([makeJob({ inputs })]);

    const file = target.querySelector('[data-testid="queue-input"]')!;
    expect(file.textContent).toContain('door.png');
    expect(file.textContent).toContain('+1');
    expect(target.querySelector('img')).toBeNull();
  });

  it('leads a queued upscale with its source file, once, in place of an empty prompt', () => {
    const inputs = [{ role: 'source' as const, name: 'portrait.png', kind: 'image' as const }];
    mountPanel([makeJob({ workflow: 'upscale', job_type: 'Upscale', prompt: '', settings: {}, inputs })]);

    expect(target.querySelector('[data-testid="queue-prompt"]')?.textContent).toContain('portrait.png');
    expect(target.querySelector('[data-testid="queue-input"]')).toBeNull();
    expect(target.textContent).not.toContain('No prompt recorded');
  });

  it('shows no files line for a job with neither inputs nor a project', () => {
    mountPanel([makeJob()]);
    expect(target.querySelector('[data-testid="queue-files"]')).toBeNull();
  });

  it('folds to its head on a tap, keeping the count and the first prompt, and says so to assistive technology', () => {
    const ontoggle = vi.fn();
    mountPanel([makeJob({ prompt: 'first prompt' }), makeJob({ id: 'job-2', job_id: 'job-2', prompt: 'second' })], { ontoggle });
    const toggle = target.querySelector<HTMLButtonElement>('button[aria-expanded]')!;
    expect(toggle.getAttribute('aria-expanded')).toBe('true');

    toggle.click();
    flushSync();

    expect(ontoggle).toHaveBeenCalledOnce();
    expect(toggle.hasAttribute('aria-controls')).toBe(false);
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    expect(target.querySelector('[data-job-id]')).toBeNull();
    expect(toggle.textContent).toContain('Up next · 2');
    expect(toggle.textContent).toContain('first prompt');
    expect(Array.from(target.querySelectorAll('button')).some((button) => button.textContent?.includes('Clear queue'))).toBe(false);
  });

  it('removes a job and clears the queue from its buttons', () => {
    const onremove = vi.fn();
    const onclear = vi.fn();
    mountPanel([makeJob()], { onremove, onclear });

    target.querySelector<HTMLButtonElement>('button[aria-label="Remove from queue"]')!.click();
    expect(onremove).toHaveBeenCalledOnce();
    Array.from(target.querySelectorAll('button')).find((button) => button.textContent?.includes('Clear queue'))!.click();
    expect(onclear).toHaveBeenCalledOnce();
  });
});
