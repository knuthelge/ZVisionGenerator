import { afterEach, describe, expect, it, vi } from 'vitest';

import { candidateAudioUrl, clearCandidates, deleteVoice, getDesignState, listVoices, planVoiceDesign, renameVoice, saveCandidate, saveRecordingVoice, submitVoiceDesign, voiceAudioUrl } from './voices';

function stubFetch(body: unknown) {
  const fetchMock = vi.fn(async () => new Response(JSON.stringify(body), { status: 200 }));
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

function lastCall(fetchMock: ReturnType<typeof stubFetch>): { path: string; init: RequestInit } {
  const [url, init] = fetchMock.mock.calls.at(-1) as unknown as [string, RequestInit];
  return { path: new URL(url, 'http://localhost').pathname, init };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('voice API', () => {
  it('lists the voices array of the answer', async () => {
    const voice = { slug: 'anna', name: 'Anna', text: 'Hi.', origin: 'recording', seconds: 4, description: null, created_at: '2026-11-08T10:00:00Z' };
    const fetchMock = stubFetch({ voices: [voice] });

    await expect(listVoices()).resolves.toEqual([voice]);
    expect(lastCall(fetchMock)).toMatchObject({ path: '/api/voices', init: { method: 'GET' } });
  });

  it('saves a recording by posting the form as it is', async () => {
    const fetchMock = stubFetch({ slug: 'anna' });
    const form = new FormData();
    form.set('name', 'Anna');

    await saveRecordingVoice(form);

    const { path, init } = lastCall(fetchMock);
    expect(path).toBe('/api/voices');
    expect(init.method).toBe('POST');
    expect(init.body).toBe(form);
  });

  it('renames with the new name in a JSON body, and names the voice by its slug', async () => {
    const fetchMock = stubFetch({ slug: 'a b' });

    await renameVoice('a b', 'Zoe');

    const { path, init } = lastCall(fetchMock);
    expect(path).toBe('/api/voices/a%20b/rename');
    expect(JSON.parse(init.body as string)).toEqual({ name: 'Zoe' });
  });

  it('deletes with a DELETE of the voice', async () => {
    const fetchMock = stubFetch({ status: 'ok' });

    await deleteVoice('anna');

    expect(lastCall(fetchMock)).toMatchObject({ path: '/api/voices/anna', init: { method: 'DELETE' } });
  });

  it('plans and submits a design with the model, the description and the line', async () => {
    const body = { model: 'qwen3_tts', description: 'a calm old man', text: 'Good evening.' };
    const fetchMock = stubFetch({});

    await planVoiceDesign(body);
    expect(lastCall(fetchMock).path).toBe('/api/voices/design/plan');
    expect(JSON.parse(lastCall(fetchMock).init.body as string)).toEqual(body);

    await submitVoiceDesign(body);
    expect(lastCall(fetchMock)).toMatchObject({ path: '/api/voices/design', init: { method: 'POST' } });
    expect(JSON.parse(lastCall(fetchMock).init.body as string)).toEqual(body);
  });

  it('reads the kept batches and the unpicked count of the answer as they are', async () => {
    const state = {
      batches: [{ batch_id: 'b2', description: 'd', text: 't', created_at: '2026-11-08T10:00:00Z', candidates: [{ id: 'c4', number: 4, seconds: 7, saveable: true, reason: null, saved_as: null }] }],
      unpicked: 1,
    };
    const fetchMock = stubFetch(state);

    await expect(getDesignState()).resolves.toEqual(state);
    expect(lastCall(fetchMock)).toMatchObject({ path: '/api/voices/design', init: { method: 'GET' } });
  });

  it('saves a candidate by its id with the name alone', async () => {
    const fetchMock = stubFetch({ slug: 'cy' });

    await saveCandidate('c 2', 'Cy');

    const { path, init } = lastCall(fetchMock);
    expect(path).toBe('/api/voices/design/c%202/save');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual({ name: 'Cy' });
  });

  it('clears the candidates with a DELETE of the collection', async () => {
    const fetchMock = stubFetch({ status: 'ok' });

    await clearCandidates();

    expect(lastCall(fetchMock)).toMatchObject({ path: '/api/voices/design/candidates', init: { method: 'DELETE' } });
  });

  it('addresses a recording by slug and a candidate by id', () => {
    expect(voiceAudioUrl('anna')).toBe('/api/voices/anna/audio');
    expect(voiceAudioUrl('a b')).toBe('/api/voices/a%20b/audio');
    expect(candidateAudioUrl('c1')).toBe('/api/voices/design/c1/audio');
  });
});
