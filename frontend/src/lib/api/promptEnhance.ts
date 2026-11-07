import { detailFromText } from './client';
import type { AudioEnhanceContext, AudioEnhanceFrame, AudioEnhanceSettings, AudioEnhanceTask, DescribeKind, EnhanceContext, FillField, EnhanceFrame, EnhanceSettings, LyricsSettings, ScriptSettings, WorkflowMode } from '$lib/types';

/** An image or video enhancement; its stream is `EnhanceFrame`s. */
export interface MediaEnhanceRequest {
  prompt: string;
  /** A video's scene; the server then answers with a scene and a prompt. */
  scene?: string;
  mode: Exclude<WorkflowMode, 'audio'>;
  settings: Partial<EnhanceSettings>;
  max_words?: number;
  context?: EnhanceContext;
  /** `smart` runs the larger enhancer; absent is Fast. */
  enhance_model?: 'smart';
  /** `json` writes a structured JSON caption (images on models that take one); absent is text. */
  format?: 'json';
}

/** An audio enhancement of a Sound prompt or a Music style; its stream is `AudioEnhanceFrame`s. */
export interface AudioEnhanceRequest {
  mode: 'audio';
  task: AudioEnhanceTask;
  /** The audio model's id; its family decides how the text is written. */
  model: string;
  kind: 'music' | 'sound';
  /** The text the task works on. */
  prompt: string;
  settings: Partial<AudioEnhanceSettings>;
  context?: AudioEnhanceContext;
  /** `smart` runs the larger enhancer; absent is Fast. */
  enhance_model?: 'smart';
}

/** The lyrics helper: improve a song's lyrics (the whole song, or one section), or write one empty section. The `done` frame's `fields.lyrics` is the whole song. */
export interface LyricsEnhanceRequest {
  mode: 'audio';
  task: 'lyrics';
  model: string;
  kind: 'music';
  /** The song as the Lyrics box holds it, section tags on their own lines. */
  prompt: string;
  settings: LyricsSettings;
  /** The style that will be sent with the take, so the words suit it. */
  context?: Pick<AudioEnhanceContext, 'style'>;
  /** `smart` runs the larger enhancer; absent is Fast. */
  enhance_model?: 'smart';
}

/** The script helper: mark a speech text up with delivery, or fit it to a reading time. The `done` frame's `fields.script` is the whole script. */
export interface ScriptEnhanceRequest {
  mode: 'audio';
  task: 'script';
  model: string;
  kind: 'speech';
  /** The script as the Text box holds it, marks included. */
  prompt: string;
  settings: ScriptSettings;
  /** `smart` runs the larger enhancer; absent is Fast. */
  enhance_model?: 'smart';
}

/**
 * Describe it's Fill in: write the fields still empty from an idea in one answer. The `done` frame's `fields` holds each one it could write
 * (`style`, `tempo` as a number in text, `key` as an ABC id, `lyrics`, `vibe` as a Vibe id, `text`).
 */
export interface FillEnhanceRequest {
  mode: 'audio';
  task: 'fill';
  model: string;
  kind: DescribeKind;
  /** The Describe it text. */
  prompt: string;
  settings: { fields: FillField[] };
  /** What the user already set, so the new fields suit it. */
  context?: Pick<AudioEnhanceContext, 'style' | 'lyrics' | 'text' | 'tempo' | 'key'>;
  /** `smart` runs the larger enhancer; absent is Fast. */
  enhance_model?: 'smart';
}

export type AudioRequest = AudioEnhanceRequest | LyricsEnhanceRequest | ScriptEnhanceRequest | FillEnhanceRequest;

export type EnhanceRequest = MediaEnhanceRequest | AudioRequest;

/** Incrementally split an NDJSON byte stream (decoded to text) into frames, tolerating chunk boundaries. */
export function createNdjsonParser<Frame = EnhanceFrame>(onFrame: (frame: Frame) => void): { push: (chunk: string) => void; flush: () => void } {
  let buffer = '';
  function emit(line: string): void {
    const trimmed = line.trim();
    if (!trimmed) return;
    try {
      onFrame(JSON.parse(trimmed) as Frame);
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
  return detailFromText(text) || `Enhance failed (${response.status})`;
}

/**
 * Stream a prompt enhancement. Resolves after the final frame; rejects with the server's
 * message for HTTP errors (e.g. 409 while a job runs). Abort with *signal* to cancel.
 */
export async function enhancePrompt(body: MediaEnhanceRequest, onFrame: (frame: EnhanceFrame) => void, signal?: AbortSignal): Promise<void>;
export async function enhancePrompt(body: AudioRequest, onFrame: (frame: AudioEnhanceFrame) => void, signal?: AbortSignal): Promise<void>;
export async function enhancePrompt(body: EnhanceRequest, onFrame: (frame: never) => void, signal?: AbortSignal): Promise<void> {
  const response = await fetch('/api/prompt/enhance', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/x-ndjson' },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) throw new Error(await errorDetail(response));
  const parser = createNdjsonParser<never>(onFrame);
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
