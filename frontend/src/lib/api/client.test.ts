import { afterEach, describe, expect, it, vi } from 'vitest';

import { api, ApiError, detailFromText, errorText, isRetryable } from './client';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('detailFromText', () => {
  it('returns a FastAPI detail string', () => {
    expect(detailFromText('{"detail":"Field \'width\' must be an integer."}')).toBe("Field 'width' must be an integer.");
  });

  it('joins the messages of a validation-error list', () => {
    const body = JSON.stringify({ detail: [{ loc: ['query', 'page'], msg: 'Input should be greater than or equal to 1' }, { msg: 'Field required' }] });
    expect(detailFromText(body)).toBe('Input should be greater than or equal to 1 Field required');
  });

  it('returns non-JSON text unchanged', () => {
    expect(detailFromText('Bad Gateway')).toBe('Bad Gateway');
  });
});

describe('errorText', () => {
  it('prefers the server detail over the request summary', () => {
    expect(errorText(new ApiError('POST /api/generate → 422: {...}', 422, 'A video model is required.'), 'fallback')).toBe('A video model is required.');
  });

  it('falls back when the server gave no detail', () => {
    expect(errorText(new ApiError('POST /x → 500: ', 500, ''), 'Saving failed.')).toBe('Saving failed.');
  });

  it('uses the message of other errors, or the fallback', () => {
    expect(errorText(new TypeError('Failed to fetch'), 'fallback')).toBe('Failed to fetch');
    expect(errorText('boom', 'fallback')).toBe('fallback');
  });
});

describe('isRetryable', () => {
  it('is false when the server rejected the input and true for server and network faults', () => {
    expect(isRetryable(new ApiError('', 422))).toBe(false);
    expect(isRetryable(new ApiError('', 409))).toBe(false);
    expect(isRetryable(new ApiError('', 500))).toBe(true);
    expect(isRetryable(new TypeError('Failed to fetch'))).toBe(true);
  });
});

describe('api.post', () => {
  it('sends FormData as multipart without a JSON content type', async () => {
    const fetchMock = vi.fn(async () => new Response('{"job_id":"j1"}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const form = new FormData();
    form.set('prompt', 'A lighthouse');

    await expect(api.post('/api/generate', form)).resolves.toEqual({ job_id: 'j1' });

    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(init.body).toBe(form);
    expect(init.headers).toEqual({});
  });

  it('throws ApiError carrying the status and detail', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('{"detail":"A video model is required."}', { status: 422 })));

    await expect(api.post('/api/generate', new FormData())).rejects.toMatchObject({ status: 422, detail: 'A video model is required.' });
  });
});
