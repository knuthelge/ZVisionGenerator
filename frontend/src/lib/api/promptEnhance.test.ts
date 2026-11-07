import { afterEach, describe, expect, it, vi } from 'vitest';

import { createNdjsonParser, enhancePrompt, type AudioEnhanceRequest, type MediaEnhanceRequest } from './promptEnhance';
import type { AudioEnhanceFrame, EnhanceFrame } from '$lib/types';

afterEach(() => {
  vi.unstubAllGlobals();
});

function stubStream(lines: string[]) {
  const fetchMock = vi.fn(async () => new Response(lines.map((line) => `${line}\n`).join(''), { status: 200 }));
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

function postedBody(fetchMock: ReturnType<typeof stubStream>): unknown {
  const call = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
  return JSON.parse(call[1].body as string);
}

describe('enhancePrompt with an audio request', () => {
  const request: AudioEnhanceRequest = {
    mode: 'audio',
    task: 'style',
    model: 'yue2',
    kind: 'music',
    prompt: 'sad folk',
    settings: { genre: 'folk', without: ['drums'] },
    context: { instrumental: false, tempo: 90 },
    enhance_model: 'smart',
  };

  it('posts the request as it is and hands the frames with their fields to the caller', async () => {
    const fetchMock = stubStream([
      '{"type":"status","phase":"generating"}',
      '{"type":"text","task":"style","text":"Folk, pi","fields":{"style":"Folk, pi"}}',
      '{"type":"done","task":"style","fields":{"style":"Folk, piano"},"clamped":false,"model":"org/fast"}',
    ]);
    const frames: AudioEnhanceFrame[] = [];

    await enhancePrompt(request, (frame) => frames.push(frame));

    expect(fetchMock).toHaveBeenCalledWith('/api/prompt/enhance', expect.objectContaining({ method: 'POST' }));
    expect(postedBody(fetchMock)).toEqual(request);
    expect(frames.map((frame) => frame.type)).toEqual(['status', 'text', 'done']);
    const done = frames[2];
    expect(done.type === 'done' && done.fields.style).toBe('Folk, piano');
  });

  it('rejects with the server message when the request is refused', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('{"detail":"Enter a sound to enhance."}', { status: 422 })));

    await expect(enhancePrompt({ ...request, task: 'sound', kind: 'sound', prompt: '' }, () => {})).rejects.toThrow('Enter a sound to enhance.');
  });
});

describe('enhancePrompt with an image or video request', () => {
  it('keeps the request and the frames it always had', async () => {
    const request: MediaEnhanceRequest = { prompt: 'a fox', mode: 'image', settings: { style: 'photo' }, max_words: 120 };
    const fetchMock = stubStream(['{"type":"done","prompt":"A fox.","clamped":false,"model":"org/fast"}']);
    const frames: EnhanceFrame[] = [];

    await enhancePrompt(request, (frame) => frames.push(frame));

    expect(postedBody(fetchMock)).toEqual(request);
    expect(frames).toEqual([{ type: 'done', prompt: 'A fox.', clamped: false, model: 'org/fast' }]);
  });
});

describe('createNdjsonParser with audio frames', () => {
  it('reads a frame of any type the caller names', () => {
    const frames: AudioEnhanceFrame[] = [];
    const parser = createNdjsonParser<AudioEnhanceFrame>((frame) => frames.push(frame));

    parser.push('{"type":"error","detail":"x"}\n');

    expect(frames).toEqual([{ type: 'error', detail: 'x' }]);
  });
});
