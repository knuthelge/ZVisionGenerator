const BASE = '';

/** Non-2xx API response; `status` lets callers tell "gone" (404) from transient failures. */
export class ApiError extends Error {
  readonly status: number;
  /** The server's explanation (a FastAPI `detail` string), or the raw response text. */
  readonly detail: string;

  constructor(message: string, status: number, detail = '') {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

/** Return the server's explanation for a failed request, or *fallback* when it gave none. */
export function errorText(error: unknown, fallback: string): string {
  if (error instanceof ApiError) return error.detail || fallback;
  return error instanceof Error && error.message ? error.message : fallback;
}

/** Whether repeating the request could succeed: true for network and server faults, false when the server rejected the input. */
export function isRetryable(error: unknown): boolean {
  return !(error instanceof ApiError) || error.status >= 500;
}

/** Return the text of a FastAPI error body: its `detail` string, the messages of a validation-error list, or the raw text. */
export function detailFromText(text: string): string {
  try {
    const parsed = JSON.parse(text) as { detail?: unknown };
    if (typeof parsed.detail === 'string') return parsed.detail;
    if (Array.isArray(parsed.detail)) {
      const messages = parsed.detail.map((item: { msg?: unknown } | null) => item?.msg).filter((msg): msg is string => typeof msg === 'string');
      if (messages.length > 0) return messages.join(' ');
    }
  } catch {
    // not JSON
  }
  return text;
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  // FormData sets its own multipart Content-Type; any other body goes as JSON.
  const isForm = body instanceof FormData;
  const response = await fetch(`${BASE}${path}`, {
    method,
    headers: body !== undefined && !isForm ? { 'Content-Type': 'application/json' } : {},
    body: isForm ? body : body !== undefined ? JSON.stringify(body) : undefined
  });

  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText);
    throw new ApiError(`${method} ${path} → ${response.status}: ${text}`, response.status, detailFromText(text));
  }

  return response.json() as Promise<T>;
}

export const api = {
  get: <T>(path: string) => request<T>('GET', path),
  post: <T>(path: string, body?: unknown) => request<T>('POST', path, body),
  put: <T>(path: string, body?: unknown) => request<T>('PUT', path, body),
  delete: <T>(path: string) => request<T>('DELETE', path)
};
