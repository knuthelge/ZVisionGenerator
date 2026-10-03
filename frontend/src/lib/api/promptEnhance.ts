import type { EnhanceFrame, EnhanceSettings, WorkflowMode } from '$lib/types';

export interface EnhanceRequest {
  prompt: string;
  mode: WorkflowMode;
  settings: Partial<EnhanceSettings>;
  max_words?: number;
}

/** Incrementally split an NDJSON byte stream (decoded to text) into frames, tolerating chunk boundaries. */
export function createNdjsonParser(onFrame: (frame: EnhanceFrame) => void): { push: (chunk: string) => void; flush: () => void } {
  let buffer = '';
  function emit(line: string): void {
    const trimmed = line.trim();
    if (!trimmed) return;
    try {
      onFrame(JSON.parse(trimmed) as EnhanceFrame);
    } catch {
      // ignore malformed lines
    }
  }
  return {
    push(chunk: string): void {
      buffer += chunk;
      let newline = buffer.indexOf('\n');
      while (newline >= 0) {
        emit(buffer.slice(0, newline));
        buffer = buffer.slice(newline + 1);
        newline = buffer.indexOf('\n');
      }
    },
    flush(): void {
      emit(buffer);
      buffer = '';
    },
  };
}

async function errorDetail(response: Response): Promise<string> {
  const text = await response.text().catch(() => response.statusText);
  try {
    const parsed = JSON.parse(text) as { detail?: unknown };
    if (typeof parsed.detail === 'string') return parsed.detail;
  } catch {
    // not JSON
  }
  return text || `Enhance failed (${response.status})`;
}

/**
 * Stream a prompt enhancement. Resolves after the final frame; rejects with the server's
 * message for HTTP errors (e.g. 409 while a job runs). Abort with *signal* to cancel.
 */
export async function enhancePrompt(body: EnhanceRequest, onFrame: (frame: EnhanceFrame) => void, signal?: AbortSignal): Promise<void> {
  const response = await fetch('/api/prompt/enhance', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/x-ndjson' },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) throw new Error(await errorDetail(response));
  const parser = createNdjsonParser(onFrame);
  const reader = response.body?.getReader();
  if (!reader) {
    parser.push(await response.text());
    parser.flush();
    return;
  }
  const decoder = new TextDecoder();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parser.push(decoder.decode(value, { stream: true }));
  }
  parser.push(decoder.decode());
  parser.flush();
}
