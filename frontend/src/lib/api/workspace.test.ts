import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./client', () => ({ api: { get: vi.fn(), post: vi.fn(), delete: vi.fn() } }));

import { api } from './client';
import { planForm, planGenerate, submitGenerate } from './workspace';

beforeEach(() => {
  vi.clearAllMocks();
});

function formWithFiles(): FormData {
  const form = new FormData();
  form.append('mode', 'video');
  form.append('prompt', 'a lake');
  form.append('image_file', new File(['png bytes'], 'start.png', { type: 'image/png' }));
  form.append('audio_track_file', new File(['audio bytes'], 'song.m4a', { type: 'audio/mp4' }));
  form.append('reference', new File(['ref one'], 'r1.png', { type: 'image/png' }));
  form.append('reference', '/typed/ref.png');
  return form;
}

describe('planForm', () => {
  it('keeps every field and file name but sends no file body', () => {
    const plan = planForm(formWithFiles());

    expect(plan.get('mode')).toBe('video');
    expect(plan.get('prompt')).toBe('a lake');
    const image = plan.get('image_file') as File;
    expect([image.name, image.type, image.size]).toEqual(['start.png', 'image/png', 0]);
    const track = plan.get('audio_track_file') as File;
    expect([track.name, track.size]).toEqual(['song.m4a', 0]);
  });

  it('keeps the order of the repeated reference field, a file or a path per row', () => {
    const rows = planForm(formWithFiles()).getAll('reference');

    expect(rows).toHaveLength(2);
    expect((rows[0] as File).name).toBe('r1.png');
    expect((rows[0] as File).size).toBe(0);
    expect(rows[1]).toBe('/typed/ref.png');
  });
});

describe('generate requests', () => {
  it('plans with the fields only', async () => {
    vi.mocked(api.post).mockResolvedValue({ parts: [] });
    const form = formWithFiles();

    await planGenerate(form);

    const [path, sent] = vi.mocked(api.post).mock.calls[0];
    expect(path).toBe('/api/generate/plan');
    const files = [...(sent as FormData).values()].filter((value): value is File => typeof value !== 'string');
    expect(files.length).toBe(3);
    expect(files.every((file) => file.size === 0)).toBe(true);
  });

  it('submits the files themselves, once', async () => {
    vi.mocked(api.post).mockResolvedValue({ job_id: 'j1' });
    const form = formWithFiles();

    await planGenerate(form);
    await submitGenerate(form);

    const submits = vi.mocked(api.post).mock.calls.filter(([path]) => path === '/api/generate');
    expect(submits).toHaveLength(1);
    const sent = submits[0][1] as FormData;
    expect(sent).toBe(form);
    expect((sent.get('image_file') as File).size).toBe('png bytes'.length);
    expect((sent.get('audio_track_file') as File).size).toBe('audio bytes'.length);
  });
});
