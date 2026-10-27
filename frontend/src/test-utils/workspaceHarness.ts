// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount } from '../../node_modules/svelte/src/index-client.js';
import { expect, vi } from 'vitest';

import WorkspacePage from '../features/workspace/WorkspacePage.svelte';
import { firstRun } from '../features/workspace/firstRun.svelte';
import { clipMemory } from '$lib/state/clipMemory.svelte';
import { INPUT_DEBOUNCE_MS } from '$lib/state/debounce';
import { downloadStore } from '$lib/state/downloads.svelte';
import { draft } from '$lib/state/draft.svelte';
import { extendInspection } from '$lib/state/extendInspection.svelte';
import type { sceneSource } from '$lib/state/extend.fixtures';
import { galleryStore } from '$lib/state/gallery.svelte';
import { jobStore } from '$lib/state/job.svelte';
import { workspaceMemory } from '$lib/state/pageMemory';
import { toasts } from '$lib/state/toasts.svelte';
import type { WorkspaceContext } from '$lib/types';
import { makeGalleryPage } from './workspace';
import {
  browseApiMocks, configApiMocks, enhanceApiMocks, extendApiMocks, galleryApiMocks, layoutProbe, modelApiMocks,
  presetApiMocks, promptFileApiMocks, videoMemoryApiMocks, workspaceApiMocks,
} from './workspaceMocks';

// The timer that is real while a test fakes the debounce timers.
const realSetTimeout = globalThis.setTimeout;

/** Return every store and API mock to its starting state: empty stores, no server-side queue, one gallery page. */
export function resetWorkspaceTest(): void {
  layoutProbe.compact = false;
  localStorage.clear();
  sessionStorage.clear();
  draft.reset();
  workspaceMemory.forget();
  extendInspection.reset();
  clipMemory.reset();
  galleryStore.reset();
  jobStore.clearJob();
  downloadStore.reset();
  for (const mock of Object.values(presetApiMocks)) mock.mockReset();
  presetApiMocks.listPresets.mockResolvedValue([]);
  firstRun.reset();
  configApiMocks.updateConfig.mockClear();
  workspaceApiMocks.skipFirstRun.mockClear();
  modelApiMocks.getDownloads.mockReset();
  modelApiMocks.getDownloads.mockResolvedValue({ items: [] });
  toasts.splice(0, toasts.length);
  for (const mock of [
    workspaceApiMocks.getWorkspaceContext, workspaceApiMocks.getWorkspaceCoreContext, workspaceApiMocks.submitGenerate, workspaceApiMocks.getJobSnapshot,
    galleryApiMocks.getGallery, workspaceApiMocks.parseUrlPrefill, workspaceApiMocks.listJobs, browseApiMocks.listDirectory, browseApiMocks.browsePlaces,
    promptFileApiMocks.inspectPromptFile, promptFileApiMocks.writePromptFile, promptFileApiMocks.appendPromptEntry, promptFileApiMocks.createPromptFile, extendApiMocks.inspectExtendSource, enhanceApiMocks.enhancePrompt,
    videoMemoryApiMocks.checkVideoMemoryFit, galleryApiMocks.deleteAsset,
  ]) mock.mockReset();
  workspaceApiMocks.clearQueue.mockClear();
  workspaceApiMocks.cancelJob.mockClear();
  workspaceApiMocks.listJobs.mockResolvedValue({ active_job: null, queued_jobs: [] });
  workspaceApiMocks.parseUrlPrefill.mockReturnValue({});
  galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage());
}

/** Append a fresh host element for a page to the document. */
export function createHost(): HTMLDivElement {
  const host = document.createElement('div');
  document.body.appendChild(host);
  return host;
}

/** Mount the Workspace page on *target* once *context* is the server's answer; the caller keeps the handle to unmount it. */
export async function mountWorkspacePage(context: WorkspaceContext, target: HTMLElement): Promise<Record<string, unknown>> {
  workspaceApiMocks.getWorkspaceCoreContext.mockResolvedValue(context);
  const app = flushSync(() => mount(WorkspacePage, { target }));
  await settle();
  return app;
}

/** Let pending promises, timers and effects run. */
export async function settle(): Promise<void> {
  await new Promise((resolve) => realSetTimeout(resolve, 0));
  await new Promise((resolve) => realSetTimeout(resolve, 0));
  await Promise.resolve();
  await Promise.resolve();
  flushSync();
}

/** Fake the timers the input debounce waits on, so a test steps past it without sleeping; `vi.useRealTimers()` ends it. */
export function fakeDebounceTimers(): void {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
}

/** Step past the input debounce of the memory check and the source check; needs `fakeDebounceTimers`. */
export async function waitDebounce(): Promise<void> {
  await vi.advanceTimersByTimeAsync(INPUT_DEBOUNCE_MS);
  await settle();
}

/** Answer the open confirmation dialog. */
export async function answerConfirm(confirmed: boolean): Promise<void> {
  const action = confirmed ? 'confirm' : 'cancel';
  (document.querySelector(`[role="alertdialog"] [data-action="${action}"]`) as HTMLButtonElement).click();
  await settle();
}

/** Pick prompt-file prompts through the Choose prompts dialog and confirm. */
export async function choosePrompts(container: ParentNode, ids: string[]): Promise<void> {
  (container.querySelector('[data-action="choose-prompts"]') as HTMLButtonElement).click();
  await settle();
  const chooser = document.querySelector('[data-testid="prompt-chooser"]') as HTMLElement;
  for (const box of Array.from(chooser.querySelectorAll<HTMLInputElement>('input[type="checkbox"]'))) {
    if (box.checked !== ids.includes(box.value)) box.click();
  }
  await settle();
  (document.querySelector('[data-action="confirm-prompts"]') as HTMLButtonElement).click();
  await settle();
}

/** Expect every rendered label to point at a control. */
export function expectRenderedLabelsToResolveControls(container: ParentNode): void {
  const explicitLabels = Array.from(container.querySelectorAll('label[for]')) as HTMLLabelElement[];
  expect(explicitLabels.length).toBeGreaterThan(0);
  for (const label of explicitLabels) {
    const control = label.control ?? container.querySelector(`[id="${label.htmlFor}"]`);
    expect(
      control,
      `Expected label "${label.textContent?.trim() ?? label.htmlFor}" to resolve control "${label.htmlFor}"`
    ).not.toBeNull();
  }

  const wrappedLabels = Array.from(container.querySelectorAll('label:not([for])')) as HTMLLabelElement[];
  for (const label of wrappedLabels) {
    expect(
      label.querySelector('input, select, textarea'),
      `Expected wrapped label "${label.textContent?.trim() ?? '(unnamed label)'}" to contain a control`
    ).not.toBeNull();
  }
}

export const field = (id: string) => document.querySelector(`#${id}`) as HTMLTextAreaElement;

/** Type *value* into the field with that id. */
export function type(id: string, value: string): void {
  const el = field(id);
  el.value = value;
  el.dispatchEvent(new Event('input', { bubbles: true }));
}

/** Open the Enhance popover and return its Enhance button. */
export async function openEnhanceRun(): Promise<HTMLButtonElement> {
  (document.querySelector('#ws-enhance-toggle') as HTMLButtonElement).click();
  await settle();
  return document.querySelector('#ws-enhance-run') as HTMLButtonElement;
}

/** Enhance with the current settings through the popover. */
export async function runEnhance(): Promise<void> {
  (await openEnhanceRun()).click();
  await settle();
}

/** Make the next source check wait until the returned function is called with the server's answer. */
export function holdInspection(): (info: ReturnType<typeof sceneSource>) => void {
  let release: (info: ReturnType<typeof sceneSource>) => void = () => undefined;
  extendApiMocks.inspectExtendSource.mockReturnValue(new Promise((resolve) => { release = resolve; }));
  return (info) => release(info);
}

/** Replace the clipboard with a spy and return it. */
export function mockClipboard(): ReturnType<typeof vi.fn<(text: string) => Promise<void>>> {
  const writeText = vi.fn<(text: string) => Promise<void>>(async () => undefined);
  Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
  return writeText;
}

/** Dispatch a keydown from *from* and return it. */
export function pressKey(key: string, init: KeyboardEventInit = {}, from: EventTarget = document): KeyboardEvent {
  const event = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...init });
  from.dispatchEvent(event);
  return event;
}

/** Submit the page's form and return what Generate sent. */
export async function submitForm(target: ParentNode): Promise<FormData> {
  target.querySelector('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
  await settle();
  const [submitted] = workspaceApiMocks.submitGenerate.mock.calls.at(-1) ?? [];
  return submitted as FormData;
}
