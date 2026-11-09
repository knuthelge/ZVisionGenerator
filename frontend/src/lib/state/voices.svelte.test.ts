import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '$lib/api/client';
import type { SavedVoice, VoiceCandidate, VoiceDesignState } from '$lib/types';

const api = vi.hoisted(() => ({
  listVoices: vi.fn<() => Promise<SavedVoice[]>>(),
  saveRecordingVoice: vi.fn<(form: FormData) => Promise<SavedVoice>>(),
  renameVoice: vi.fn<(slug: string, name: string) => Promise<SavedVoice>>(),
  deleteVoice: vi.fn<(slug: string) => Promise<{ status: string }>>(),
  saveCandidate: vi.fn<(candidateId: string, name: string) => Promise<SavedVoice>>(),
  getDesignState: vi.fn<() => Promise<VoiceDesignState>>(),
  clearCandidates: vi.fn<() => Promise<{ status: string }>>(),
}));

vi.mock('$lib/api/voices', () => api);

const voice = (slug: string, name: string, origin: SavedVoice['origin'] = 'recording'): SavedVoice => ({ slug, name, text: `${name} says hello.`, origin, seconds: 5, description: null, created_at: '2026-11-08T10:00:00Z' });

// A fresh module per test gives each an empty list, an unloaded store and a default draft.
let voicesStore: typeof import('./voices.svelte').voicesStore;
let draft: typeof import('./draft.svelte').draft;

beforeEach(async () => {
  localStorage.clear();
  vi.resetModules();
  for (const mock of Object.values(api)) mock.mockReset();
  ({ voicesStore } = await import('./voices.svelte'));
  ({ draft } = await import('./draft.svelte'));
});

describe('voicesStore', () => {
  it('starts empty and unloaded, and turns loaded with the first read', async () => {
    expect(voicesStore.list).toEqual([]);
    expect(voicesStore.loaded).toBe(false);
    api.listVoices.mockResolvedValue([voice('zed', 'zed'), voice('anna', 'Anna'), voice('bob', 'bob')]);

    await voicesStore.load();

    expect(voicesStore.loaded).toBe(true);
    expect(voicesStore.list.map((item) => item.slug)).toEqual(['anna', 'bob', 'zed']);
  });

  it('is not loaded after a failed first read, and keeps the list after a failed later one', async () => {
    api.listVoices.mockRejectedValueOnce(new Error('offline'));
    await voicesStore.load();
    expect(voicesStore.loaded).toBe(false);

    api.listVoices.mockResolvedValueOnce([voice('anna', 'Anna')]);
    await voicesStore.load();
    api.listVoices.mockRejectedValueOnce(new Error('offline'));
    await voicesStore.load();

    expect(voicesStore.loaded).toBe(true);
    expect(voicesStore.list.map((item) => item.slug)).toEqual(['anna']);
  });

  it('adds a voice saved from a recording or a candidate in its place by name', async () => {
    api.listVoices.mockResolvedValue([voice('bob', 'Bob')]);
    await voicesStore.load();
    api.saveRecordingVoice.mockResolvedValueOnce(voice('anna', 'Anna'));
    api.saveCandidate.mockResolvedValueOnce(voice('cy', 'Cy', 'design'));
    const form = new FormData();

    await expect(voicesStore.saveRecording(form)).resolves.toMatchObject({ slug: 'anna' });
    await expect(voicesStore.saveCandidate('c1', 'Cy')).resolves.toMatchObject({ slug: 'cy', origin: 'design' });

    expect(api.saveRecordingVoice).toHaveBeenCalledWith(form);
    expect(api.saveCandidate).toHaveBeenCalledWith('c1', 'Cy');
    expect(voicesStore.list.map((item) => item.slug)).toEqual(['anna', 'bob', 'cy']);
  });

  it('throws the server\'s refusal of a save and leaves the list as it was', async () => {
    api.saveRecordingVoice.mockRejectedValueOnce(new ApiError('POST /api/voices → 409', 409, 'A voice named "Anna" exists.'));

    await expect(voicesStore.saveRecording(new FormData())).rejects.toMatchObject({ status: 409, detail: 'A voice named "Anna" exists.' });
    expect(voicesStore.list).toEqual([]);
  });

  it('renames a voice under the same slug and sorts it by its new name', async () => {
    api.listVoices.mockResolvedValue([voice('anna', 'Anna'), voice('bob', 'Bob')]);
    await voicesStore.load();
    api.renameVoice.mockResolvedValueOnce(voice('anna', 'Zoe'));

    await voicesStore.rename('anna', 'Zoe');

    expect(api.renameVoice).toHaveBeenCalledWith('anna', 'Zoe');
    expect(voicesStore.list.map((item) => [item.slug, item.name])).toEqual([['bob', 'Bob'], ['anna', 'Zoe']]);
  });

  describe('the kept candidates', () => {
    const candidate = (id: string, over: Partial<VoiceCandidate> = {}): VoiceCandidate => ({ id, number: Number(id.slice(1)), seconds: 6, saveable: true, reason: null, saved_as: null, ...over });
    const state = (): VoiceDesignState => ({
      batches: [
        { batch_id: 'b2', description: 'a warm woman', text: 'Hello.', created_at: '2026-11-08T11:00:00Z', candidates: [candidate('c4'), candidate('c5', { saveable: false, reason: 'Too short.' })] },
        { batch_id: 'b1', description: 'an old man', text: 'Hello.', created_at: '2026-11-08T10:00:00Z', candidates: [candidate('c1')] },
      ],
      unpicked: 2,
    });

    it('starts with none, and reads the server\'s batches and unpicked count', async () => {
      expect(voicesStore.design).toEqual({ batches: [], unpicked: 0 });
      api.getDesignState.mockResolvedValue(state());

      await voicesStore.loadDesign();

      expect(voicesStore.design.batches.map((batch) => batch.batch_id)).toEqual(['b2', 'b1']);
      expect(voicesStore.unpicked).toBe(2);
    });

    it('keeps what it has after a failed read', async () => {
      api.getDesignState.mockResolvedValueOnce(state());
      await voicesStore.loadDesign();
      api.getDesignState.mockRejectedValueOnce(new Error('offline'));

      await voicesStore.loadDesign();

      expect(voicesStore.unpicked).toBe(2);
    });

    it('marks a saved candidate with the voice\'s name, clears the count to pick, and keeps the others listed', async () => {
      api.getDesignState.mockResolvedValue(state());
      await voicesStore.loadDesign();
      api.saveCandidate.mockResolvedValue(voice('cy', 'Cy', 'design'));

      await voicesStore.saveCandidate('c4', 'Cy');

      const saved = voicesStore.design.batches[0].candidates.map((item) => [item.id, item.saved_as]);
      expect(saved).toEqual([['c4', 'Cy'], ['c5', null]]);
      expect(voicesStore.design.batches[1].candidates[0].saved_as).toBeNull();
      expect(voicesStore.unpicked).toBe(0);
      expect(voicesStore.list.map((item) => item.slug)).toEqual(['cy']);
    });

    it('leaves the marks as they were when the save is refused', async () => {
      api.getDesignState.mockResolvedValue(state());
      await voicesStore.loadDesign();
      api.saveCandidate.mockRejectedValue(new ApiError('POST → 409', 409, 'A voice named "Cy" exists.'));

      await expect(voicesStore.saveCandidate('c4', 'Cy')).rejects.toMatchObject({ status: 409 });

      expect(voicesStore.unpicked).toBe(2);
      expect(voicesStore.design.batches[0].candidates[0].saved_as).toBeNull();
    });

    it('clears every batch and the count, and keeps them when the server refuses', async () => {
      api.getDesignState.mockResolvedValue(state());
      await voicesStore.loadDesign();
      api.clearCandidates.mockRejectedValueOnce(new ApiError('DELETE → 409', 409, 'A design job is running.'));

      await expect(voicesStore.clearCandidates()).rejects.toMatchObject({ status: 409 });
      expect(voicesStore.unpicked).toBe(2);

      api.clearCandidates.mockResolvedValueOnce({ status: 'ok' });
      await voicesStore.clearCandidates();

      expect(voicesStore.design).toEqual({ batches: [], unpicked: 0 });
    });
  });

  describe('remove', () => {
    beforeEach(async () => {
      api.listVoices.mockResolvedValue([voice('anna', 'Anna'), voice('bob', 'Bob')]);
      await voicesStore.load();
      api.deleteVoice.mockResolvedValue({ status: 'ok' });
    });

    it('puts the draft back on the default built-in voice when the chosen voice is deleted here', async () => {
      draft.patch({ speechSpeaker: 'vivian' });
      draft.setSavedVoice('anna');
      expect(draft.state).toMatchObject({ voiceMode: 'saved', savedVoice: 'anna' });

      await voicesStore.remove('anna');

      expect(voicesStore.list.map((item) => item.slug)).toEqual(['bob']);
      expect(draft.state).toMatchObject({ voiceMode: 'builtin', savedVoice: null, speechSpeaker: null });
    });

    it('leaves the draft alone when another voice is deleted', async () => {
      draft.setSavedVoice('anna');

      await voicesStore.remove('bob');

      expect(draft.state).toMatchObject({ voiceMode: 'saved', savedVoice: 'anna' });
    });

    it('leaves the draft alone when the voice is refused, and keeps it in the list', async () => {
      draft.setSavedVoice('anna');
      api.deleteVoice.mockRejectedValueOnce(new ApiError('DELETE → 409', 409, 'Delete is off while a job runs.'));

      await expect(voicesStore.remove('anna')).rejects.toMatchObject({ status: 409 });

      expect(draft.state).toMatchObject({ voiceMode: 'saved', savedVoice: 'anna' });
      expect(voicesStore.list.map((item) => item.slug)).toEqual(['anna', 'bob']);
    });

    it('does not touch a draft on a recording or a built-in voice', async () => {
      draft.patch({ voiceMode: 'recording', voicePath: '/voices/me.wav' });

      await voicesStore.remove('anna');

      expect(draft.state).toMatchObject({ voiceMode: 'recording', voicePath: '/voices/me.wav' });
    });

    it('keeps a one-off recording but forgets the voice it would have returned to', async () => {
      draft.setSavedVoice('anna');
      draft.setVoice({ path: '/voices/me.wav' });

      await voicesStore.remove('anna');

      expect(draft.state).toMatchObject({ voiceMode: 'recording', voicePath: '/voices/me.wav', savedVoice: null });
    });
  });

  describe('a load that does not find the chosen voice', () => {
    it('changes nothing in the draft: the choice stays for the view to flag', async () => {
      api.listVoices.mockResolvedValueOnce([voice('anna', 'Anna')]);
      await voicesStore.load();
      draft.setSavedVoice('anna');
      api.listVoices.mockResolvedValueOnce([voice('bob', 'Bob')]);

      await voicesStore.load();

      expect(draft.state).toMatchObject({ voiceMode: 'saved', savedVoice: 'anna' });
      expect(voicesStore.list.map((item) => item.slug)).toEqual(['bob']);
    });
  });
});
