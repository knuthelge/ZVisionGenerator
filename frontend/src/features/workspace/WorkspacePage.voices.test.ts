// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '$lib/api/client';
import { trackLengths } from '$lib/state/audioTrackLength.svelte';
import { draft } from '$lib/state/draft.svelte';
import { heldUploads } from '$lib/state/heldUploads.svelte';
import { toasts } from '$lib/state/toasts.svelte';
import { voiceConsent } from '$lib/state/voiceConsent.svelte';
import { voiceWindow } from '$lib/state/voiceWindow.svelte';
import { recordingForm } from '$lib/state/recordingForm.svelte';
import type { AudioModelOption, JobContext, SavedVoice, VoiceDesignState, WorkspaceContext } from '$lib/types';

import { AUDIO_MODELS, withAudio } from '../../test-utils/audio';
import { VOICE_WORDS } from './audioView';
import { makeContext, makeGalleryPage } from '../../test-utils/workspace';
import { openOptions, pickOption } from '../../test-utils/select';
import { answerConfirm, createHost, mountWorkspacePage, resetWorkspaceTest, settle } from '../../test-utils/workspaceHarness';
import { galleryApiMocks, layoutProbe, voiceApiMocks, workspaceApiMocks } from '../../test-utils/workspaceMocks';

vi.mock('$lib/state/layoutMode.svelte', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/state/layoutMode.svelte')>();
  const { layoutProbe } = await import('../../test-utils/workspaceMocks');
  return {
    ...actual,
    layout: {
      get mode() { return layoutProbe.compact ? 'compact' : layoutProbe.tall ? 'tall' : 'wide'; },
      get compact() { return layoutProbe.compact; },
      get coarse() { return layoutProbe.compact || layoutProbe.coarse; },
    },
  };
});

vi.mock('$lib/api/config', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/config')>();
  const { configApiMocks } = await import('../../test-utils/workspaceMocks');
  return { ...actual, updateConfig: configApiMocks.updateConfig };
});

vi.mock('$lib/api/extend', async () => {
  const { extendApiMocks } = await import('../../test-utils/workspaceMocks');
  return { inspectExtendSource: extendApiMocks.inspectExtendSource };
});

vi.mock('$lib/api/videoMemory', async () => {
  const { videoMemoryApiMocks } = await import('../../test-utils/workspaceMocks');
  return { checkVideoMemoryFit: videoMemoryApiMocks.checkVideoMemoryFit };
});

vi.mock('$lib/api/workspace', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/workspace')>();
  const { workspaceApiMocks: mocks } = await import('../../test-utils/workspaceMocks');
  return {
    ...actual,
    getWorkspaceContext: mocks.getWorkspaceContext,
    getWorkspaceCoreContext: mocks.getWorkspaceCoreContext,
    submitGenerate: mocks.submitGenerate,
    planGenerate: mocks.planGenerate,
    getJobSnapshot: mocks.getJobSnapshot,
    parseUrlPrefill: mocks.parseUrlPrefill,
    checkReferencePaths: mocks.checkReferencePaths,
    listJobs: mocks.listJobs,
    clearQueue: mocks.clearQueue,
    cancelJob: mocks.cancelJob,
    skipFirstRun: mocks.skipFirstRun,
  };
});

vi.mock('$lib/api/gallery', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/gallery')>();
  const { galleryApiMocks } = await import('../../test-utils/workspaceMocks');
  return { ...actual, getGallery: galleryApiMocks.getGallery, deleteAssets: galleryApiMocks.deleteAssets };
});

vi.mock('$lib/api/promptEnhance', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/promptEnhance')>();
  const { enhanceApiMocks } = await import('../../test-utils/workspaceMocks');
  return { ...actual, enhancePrompt: enhanceApiMocks.enhancePrompt };
});

const trackApi = vi.hoisted(() => ({ inspectAudioTrack: vi.fn(), readFileLength: vi.fn() }));
vi.mock('$lib/api/audioTrack', () => ({ inspectAudioTrack: trackApi.inspectAudioTrack }));
vi.mock('$lib/state/audioLength', () => ({ readFileLength: trackApi.readFileLength }));

vi.mock('$lib/api/recentPrompts', async () => (await import('../../test-utils/workspaceMocks')).recentPromptsApiMocks);

vi.mock('$lib/api/browse', async () => (await import('../../test-utils/workspaceMocks')).browseApiMocks);

vi.mock('$lib/api/presets', async () => (await import('../../test-utils/workspaceMocks')).presetApiMocks);

vi.mock('$lib/api/voices', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/voices')>();
  const { voiceApiMocks: mocks } = await import('../../test-utils/workspaceMocks');
  return { ...actual, listVoices: mocks.listVoices, saveRecordingVoice: mocks.saveRecordingVoice, renameVoice: mocks.renameVoice, deleteVoice: mocks.deleteVoice, getDesignState: mocks.getDesignState };
});

vi.mock('$lib/api/models', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/models')>();
  const { modelApiMocks } = await import('../../test-utils/workspaceMocks');
  return { ...actual, getDownloads: modelApiMocks.getDownloads };
});

vi.mock('$lib/api/promptFiles', async () => {
  const { promptFileApiMocks } = await import('../../test-utils/workspaceMocks');
  return {
    inspectPromptFile: promptFileApiMocks.inspectPromptFile,
    writePromptFile: promptFileApiMocks.writePromptFile,
    appendPromptEntry: promptFileApiMocks.appendPromptEntry,
    createPromptFile: promptFileApiMocks.createPromptFile,
  };
});



let target: HTMLDivElement;
let app: Record<string, unknown> | null = null;

const NARRATOR: SavedVoice = { slug: 'narrator', name: 'Narrator', text: 'Welcome back to the show.', origin: 'recording', seconds: 8.4, description: null, created_at: '2026-11-08T10:00:00Z' };
const MORGAN: SavedVoice = { slug: 'morgan', name: 'Morgan', text: 'Tonight we walk through the old harbour.', origin: 'design', seconds: 11, description: 'A warm woman', created_at: '2026-11-08T11:00:00Z' };

beforeEach(() => {
  resetWorkspaceTest();
  target = createHost();
  trackLengths.reset();
  voiceConsent.reset();
  recordingForm.clear();
  voiceWindow.open = false;
  trackApi.inspectAudioTrack.mockReset().mockRejectedValue(new Error('unreadable'));
  trackApi.readFileLength.mockReset().mockResolvedValue(12);
  workspaceApiMocks.submitGenerate.mockResolvedValue({ job_id: 'job-1' } as JobContext);
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => undefined);
  // The phone's sheets slide in with the Web Animations API, which jsdom lacks.
  Element.prototype.animate = vi.fn(() => ({ cancel: vi.fn(), finished: Promise.resolve(), onfinish: null })) as unknown as Element['animate'];
  window.location.hash = '#/workspace';
});

afterEach(async () => {
  if (app) { await unmount(app); app = null; }
  delete (Element.prototype as Partial<Element>).animate;
  target.remove();
  document.body.innerHTML = '';
  vi.restoreAllMocks();
});

type Layout = 'wide' | 'compact';

interface Options {
  voices?: SavedVoice[];
  /** The designer's kept candidates the page reads; none by default. */
  design?: VoiceDesignState;
  models?: AudioModelOption[];
  /** Draft fields set before the page mounts, as a restored draft would have them. */
  draftState?: Partial<Parameters<typeof draft.patch>[0]>;
}

/** Mount the Workspace on the Speech tab; on a phone, open the Voice chip's sheet too. */
async function mountSpeech(layoutName: Layout, options: Options = {}): Promise<void> {
  voiceApiMocks.listVoices.mockResolvedValue(options.voices ?? [NARRATOR, MORGAN]);
  voiceApiMocks.getDesignState.mockResolvedValue(options.design ?? { batches: [], unpicked: 0 });
  const context: WorkspaceContext = withAudio(makeContext({ history_assets: [] }), options.models ?? AUDIO_MODELS);
  layoutProbe.compact = layoutName === 'compact';
  galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage([]));
  draft.reset();
  draft.patch({ workflow: 'speech', speechText: 'Hello there.', ...options.draftState });
  app = await mountWorkspacePage(context, target);
  if (layoutName === 'compact') {
    (target.querySelector('[data-chip="voice"]') as HTMLButtonElement).click();
    await settle();
  }
}


/** Where the Voice picker lives: the Settings column's Voice section, or the phone's open Voice sheet. */
const scope = (): HTMLElement => (target.querySelector('[data-section="voice"]') ?? document.querySelector('[data-testid="setting-popover"][data-key="voice"]')) as HTMLElement;
const picker = (): HTMLElement => scope().querySelector('[data-testid="voice-select"] button, [data-testid="pop-voice-select"] button') as HTMLElement;
const playButton = (): HTMLButtonElement => scope().querySelector('[data-testid="audio-row-play"]') as HTMLButtonElement;
const oneOff = (): HTMLElement | null => scope().querySelector('[data-testid="voice-one-off"]');
/** The one-off block of the phone's open Voice sheet, which is not the Settings column's section. */
const sheetOneOff = (): HTMLElement | null => document.querySelector('[data-testid="setting-popover"][data-key="voice"] [data-testid="voice-one-off"]');
const reason = (): string | null => target.querySelector('#ws-generate-reason, #ws-generate-reason-compact')?.textContent ?? null;
const generate = (): HTMLButtonElement => target.querySelector('#ws-submit, #ws-submit-compact') as HTMLButtonElement;
/** The open dialog of that name; a phone's sheet stays in the page, inert, while it slides away, so that is not open. */
const dialog = (label: string): HTMLElement | null => {
  const found = document.querySelector<HTMLElement>(`[role="dialog"][aria-label="${label}"]`);
  for (let node: HTMLElement | null = found; node !== null; node = node.parentElement) if (node.inert) return null;
  return found;
};
const win = (): HTMLElement | null => dialog('Voices');
const selectedTab = (): string | undefined => win()?.querySelector<HTMLElement>('[role="tab"][aria-selected="true"]')?.id.replace('voices-tab-', '');
const tabs = (): string[] => Array.from(win()?.querySelectorAll('[role="tab"]') ?? []).map((tab) => tab.textContent ?? '');
const yourVoice = (slug: string): HTMLElement => win()!.querySelector(`[data-testid="your-voice"][data-slug="${slug}"]`) as HTMLElement;
const groups = (): (string | null)[] => Array.from(document.querySelectorAll('[role="listbox"] [role="group"]')).map((group) => group.getAttribute('aria-label'));

async function choose(value: string): Promise<void> {
  pickOption(picker(), value);
  await settle();
}

async function click(element: Element | null): Promise<void> {
  (element as HTMLElement).click();
  await settle();
}

async function typeInto(input: HTMLInputElement | HTMLTextAreaElement, value: string): Promise<void> {
  input.value = value;
  input.dispatchEvent(new Event('input', { bubbles: true }));
  await settle();
}

async function attach(container: HTMLElement, file: File): Promise<void> {
  const input = container.querySelector('input[type="file"]') as HTMLInputElement;
  Object.defineProperty(input, 'files', { configurable: true, value: [file] });
  input.dispatchEvent(new Event('change', { bubbles: true }));
  await settle();
}

const recording = (name = 'harbour.wav'): File => new File(['voice'], name, { type: 'audio/wav' });
const yoursButton = (): Element | null => scope().querySelector('[data-testid="your-voices-button"]');
const newButton = (): Element | null => scope().querySelector('[data-testid="new-voice-button"]');
const recTab = (): HTMLElement => win()!.querySelector('[data-testid="recording-tab"]') as HTMLElement;
const words = (): HTMLTextAreaElement => win()!.querySelector('#rec-text') as HTMLTextAreaElement;
const nameField = (): HTMLInputElement => win()!.querySelector('#rec-name') as HTMLInputElement;
const useOnce = (): HTMLButtonElement => win()!.querySelector('[data-action="use-once"]') as HTMLButtonElement;
const saveVoice = (): HTMLButtonElement => win()!.querySelector('[data-action="save-voice"]') as HTMLButtonElement;
/** Take the recording out of the tab, to choose another. */
const clearRecording = (): Promise<void> => click(win()!.querySelector('[aria-label="Clear the voice recording"]'));
const tickBox = (): HTMLInputElement | null => win()!.querySelector('#rec-consent');

/** Fill the From recording tab with a recording of a good length, its words and the speaker's tick. */
async function fillRecording(file = recording()): Promise<void> {
  await attach(recTab(), file);
  await typeInto(words(), 'Tonight we walk through the old harbour.');
  if (tickBox() !== null) await click(tickBox());
}

describe.each<Layout>(['wide', 'compact'])('the Voice picker and the Voices window (%s)', (layoutName) => {
  describe('the row', () => {
    it('plays the chosen saved voice from the same button, and shows the playing state on that button only', async () => {
      const play = vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined);
      await mountSpeech(layoutName, { draftState: { voiceMode: 'saved', savedVoice: 'narrator' } });

      expect(playButton().disabled).toBe(false);
      expect(playButton().getAttribute('aria-label')).toBe('Play Narrator');
      expect(scope().querySelector('audio')?.getAttribute('src')).toBe('/api/voices/narrator/audio');
      await click(playButton());
      expect(play).toHaveBeenCalledTimes(1);
      scope().querySelector('audio')!.dispatchEvent(new Event('play'));
      await settle();

      expect(playButton().getAttribute('aria-pressed')).toBe('true');
      expect(picker().textContent).toContain('Narrator');
      expect(picker().getAttribute('aria-pressed')).toBeNull();
    });

    it('plays the preview clip of the chosen default voice, and each other voice from its own source', async () => {
      const play = vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined);
      await mountSpeech(layoutName);

      await choose('vivian');
      expect(playButton().disabled).toBe(false);
      expect(playButton().getAttribute('aria-label')).toBe('Play a preview of Vivian');
      expect(scope().querySelector('audio')?.getAttribute('src')).toBe('/app-static/voices/vivian.mp3');
      await click(playButton());
      expect(play).toHaveBeenCalledTimes(1);

      await choose('saved:narrator');
      expect(playButton().getAttribute('aria-label')).toBe('Play Narrator');
      expect(scope().querySelector('audio')?.getAttribute('src')).toBe('/api/voices/narrator/audio');

      await choose('ryan');
      expect(scope().querySelector('audio')?.getAttribute('src')).toBe('/app-static/voices/ryan.mp3');
    });

    it('plays the one-off recording from the same button', async () => {
      const play = vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined);
      await mountSpeech(layoutName, { draftState: { voiceMode: 'recording', voicePath: '/voices/me.wav', voiceText: 'Hi.' } });

      expect(playButton().getAttribute('aria-label')).toBe('Play the recording');
      expect(scope().querySelector('audio')?.getAttribute('src')).toContain('/voices/me.wav');
      await click(playButton());

      expect(play).toHaveBeenCalledTimes(1);
    });
  });

  describe('the dropdown', () => {
    it('lists Your voices on top, by name, then Default voices, with no entry for a recording', async () => {
      await mountSpeech(layoutName);

      const values = openOptions(picker()).map((option) => option.dataset.value);

      expect(groups()).toEqual(['Your voices', 'Default voices']);
      expect(values.slice(0, 3)).toEqual(['saved:morgan', 'saved:narrator', 'ryan']);
      expect(values).not.toContain('recording');
    });

    it('leaves Your voices out while the library is empty', async () => {
      await mountSpeech(layoutName, { voices: [] });

      openOptions(picker());

      expect(groups()).toEqual(['Default voices']);
    });

    it('chooses a saved voice and then a default voice, and puts the Delivery controls back with the default voice', async () => {
      await mountSpeech(layoutName);

      await choose('saved:narrator');
      expect(draft.state).toMatchObject({ voiceMode: 'saved', savedVoice: 'narrator' });
      expect(picker().textContent).toContain('Narrator');
      expect(reason()).toBeNull();
      expect(generate().disabled).toBe(false);

      await choose('vivian');
      expect(draft.state).toMatchObject({ voiceMode: 'builtin', savedVoice: null, speechSpeaker: 'vivian' });
    });

    it('reads From recording while a one-off recording is the voice, and drops the recording when another voice is chosen', async () => {
      await mountSpeech(layoutName, { draftState: { voiceMode: 'recording', voicePath: '/voices/me.wav', voiceText: 'Hi.' } });

      expect(picker().textContent).toContain('From recording');
      expect(oneOff()).not.toBeNull();

      await choose('saved:morgan');

      expect(oneOff()).toBeNull();
      expect(draft.state).toMatchObject({ voiceMode: 'saved', savedVoice: 'morgan', voicePath: null, voiceText: '' });
    });
  });

  describe('the two buttons', () => {
    it('open the Voices window on the tab of the button, with the three tabs', async () => {
      await mountSpeech(layoutName);
      expect(win()).toBeNull();

      await click(yoursButton());
      expect(tabs()).toEqual(['Your voices', 'New voice', 'From recording']);
      expect(selectedTab()).toBe('yours');
      await click(win()!.querySelector('[aria-label="Close dialog"], [aria-label="Close"]'));
      expect(win()).toBeNull();

      await click(newButton());
      expect(selectedTab()).toBe('new');
    });

    it('open New voice on From recording on a model that cannot design, which has no New voice tab', async () => {
      const models = AUDIO_MODELS.map((model) => (model.kind === 'speech' ? ({ ...model, can_design: false } as AudioModelOption) : model));
      await mountSpeech(layoutName, { models });

      await click(newButton());

      expect(tabs()).toEqual(['Your voices', 'From recording']);
      expect(selectedTab()).toBe('recording');
    });

    it('open New voice on From recording, prefilled, while a one-off recording is in use', async () => {
      await mountSpeech(layoutName, { draftState: { voiceMode: 'recording', voicePath: '/voices/me.wav', voiceText: 'Hello from me.' } });

      await click(newButton());

      expect(selectedTab()).toBe('recording');
      expect(recTab().textContent).toContain('me.wav');
      expect(words().value).toBe('Hello from me.');
    });

    it('open New voice on its own tab while a one-off recording is in use and designed voices wait, and show them as a count on the button', async () => {
      const design: VoiceDesignState = { batches: [{ batch_id: 'b1', description: 'd', text: 't', created_at: '2026-11-08T10:00:00Z', candidates: [{ id: 'c1', number: 1, seconds: 6, saveable: true, reason: null, saved_as: null }] }], unpicked: 3 };
      await mountSpeech(layoutName, { design, draftState: { voiceMode: 'recording', voicePath: '/voices/me.wav', voiceText: 'Hi.' } });

      expect(scope().querySelector('[data-testid="voice-pending"]')?.textContent).toBe('3');
      expect(newButton()?.getAttribute('aria-label')).toBe('New voice, 3 to pick from');
      await click(newButton());

      expect(selectedTab()).toBe('new');
    });

    it('show no count while nothing waits', async () => {
      await mountSpeech(layoutName);

      expect(scope().querySelector('[data-testid="voice-pending"]')).toBeNull();
    });
  });

  describe('the Your voices tab', () => {
    it('lists every voice with Play, Rename, Delete and Use, and names the one in use', async () => {
      await mountSpeech(layoutName, { draftState: { voiceMode: 'saved', savedVoice: 'narrator' } });
      await click(yoursButton());

      expect(Array.from(win()!.querySelectorAll('[data-testid="your-voice"]')).map((item) => item.getAttribute('data-slug'))).toEqual(['morgan', 'narrator']);
      expect(yourVoice('morgan').textContent).toContain('Morgan');
      expect(yourVoice('morgan').textContent).toContain('0:11');
      expect(yourVoice('morgan').querySelector('audio')?.getAttribute('src')).toBe('/api/voices/morgan/audio');
      expect(yourVoice('morgan').querySelector('[aria-label="Play Morgan"]')).not.toBeNull();
      expect(yourVoice('morgan').querySelector('[aria-label="Rename Morgan"]')).not.toBeNull();
      expect(yourVoice('morgan').querySelector('[aria-label="Delete Morgan"]')).not.toBeNull();
      expect(yourVoice('morgan').querySelector('[aria-label="Use Morgan"]')).not.toBeNull();
      expect(yourVoice('narrator').querySelector('[data-testid="voice-in-use"]')?.textContent).toBe('In use');
      expect(yourVoice('narrator').querySelector('[aria-label="Use Narrator"]')).toBeNull();
    });

    it('plays a voice without choosing it', async () => {
      const play = vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined);
      await mountSpeech(layoutName);
      await click(yoursButton());

      await click(yourVoice('morgan').querySelector('[aria-label="Play Morgan"]'));

      expect(play).toHaveBeenCalledTimes(1);
      expect(draft.state.voiceMode).toBe('builtin');
      expect(win()).not.toBeNull();
    });

    it('chooses a voice with Use and closes the window; the name alone does not choose', async () => {
      await mountSpeech(layoutName);
      await click(yoursButton());

      await click(yourVoice('morgan').querySelector('.yv-name'));
      expect(draft.state.voiceMode).toBe('builtin');
      await click(yourVoice('morgan').querySelector('[aria-label="Use Morgan"]'));

      expect(draft.state).toMatchObject({ voiceMode: 'saved', savedVoice: 'morgan' });
      expect(win()).toBeNull();
      expect(picker().textContent).toContain('Morgan');
    });

    it('renames through the API under the same slug and shows the new name in the list and the dropdown', async () => {
      voiceApiMocks.renameVoice.mockResolvedValue({ ...NARRATOR, name: 'Narrator two' });
      await mountSpeech(layoutName, { draftState: { voiceMode: 'saved', savedVoice: 'narrator' } });
      await click(yoursButton());

      await click(yourVoice('narrator').querySelector('[data-action="voice-rename"]'));
      expect(dialog('Rename voice')).not.toBeNull();
      const input = document.querySelector('#preset-name') as HTMLInputElement;
      expect(input.value).toBe('Narrator');
      await typeInto(input, 'Narrator two');
      await click(document.querySelector('[data-action="confirm-preset-name"]'));

      expect(voiceApiMocks.renameVoice).toHaveBeenCalledExactlyOnceWith('narrator', 'Narrator two');
      expect(dialog('Rename voice')).toBeNull();
      expect(yourVoice('narrator').textContent).toContain('Narrator two');
      expect(draft.state.savedVoice).toBe('narrator');
    });

    it('keeps the Rename dialog open and says why when the name is taken', async () => {
      voiceApiMocks.renameVoice.mockRejectedValue(new ApiError('409', 409, 'name taken'));
      await mountSpeech(layoutName);
      await click(yoursButton());

      await click(yourVoice('narrator').querySelector('[data-action="voice-rename"]'));
      await typeInto(document.querySelector('#preset-name') as HTMLInputElement, 'Morgan');
      await click(document.querySelector('[data-action="confirm-preset-name"]'));

      expect(dialog('Rename voice')?.querySelector('[role="alert"]')?.textContent).toBe('name taken');
      expect(yourVoice('narrator').textContent).toContain('Narrator');
    });

    it('asks before it deletes, deletes through the API, and puts the picker back on the default voice when the voice was in use', async () => {
      voiceApiMocks.deleteVoice.mockResolvedValue({ status: 'deleted' });
      await mountSpeech(layoutName, { draftState: { voiceMode: 'saved', savedVoice: 'narrator' } });
      await click(yoursButton());

      await click(yourVoice('narrator').querySelector('[data-action="voice-delete"]'));
      expect(document.querySelector('[role="alertdialog"]')?.textContent).toContain('Delete "Narrator"?');
      expect(voiceApiMocks.deleteVoice).not.toHaveBeenCalled();
      await answerConfirm(true);

      expect(voiceApiMocks.deleteVoice).toHaveBeenCalledExactlyOnceWith('narrator');
      expect(draft.state).toMatchObject({ voiceMode: 'builtin', savedVoice: null });
      expect(win()!.querySelector('[data-testid="your-voice"][data-slug="narrator"]')).toBeNull();
      expect(picker().getAttribute('data-value')).toBe('ryan');
    });

    it('deletes nothing when the question is cancelled', async () => {
      await mountSpeech(layoutName);
      await click(yoursButton());

      await click(yourVoice('narrator').querySelector('[data-action="voice-delete"]'));
      await answerConfirm(false);

      expect(voiceApiMocks.deleteVoice).not.toHaveBeenCalled();
      expect(yourVoice('narrator')).not.toBeNull();
    });

    it('shows the server\'s refusal as a toast, deletes nothing and keeps the voice chosen', async () => {
      voiceApiMocks.deleteVoice.mockRejectedValue(new ApiError('409', 409, 'A job is running; try again when it is done.'));
      await mountSpeech(layoutName, { draftState: { voiceMode: 'saved', savedVoice: 'narrator' } });
      await click(yoursButton());

      await click(yourVoice('narrator').querySelector('[data-action="voice-delete"]'));
      await answerConfirm(true);

      expect(toasts.map((toast) => [toast.type, toast.message])).toEqual([['error', 'A job is running; try again when it is done.']]);
      expect(draft.state).toMatchObject({ voiceMode: 'saved', savedVoice: 'narrator' });
      expect(yourVoice('narrator')).not.toBeNull();
    });

    it('says there are no voices yet, and its New voice button moves to the New voice tab', async () => {
      await mountSpeech(layoutName, { voices: [] });
      await click(yoursButton());

      expect(win()!.querySelector('[data-testid="your-voices-empty"]')?.textContent).toContain('No voices yet');
      await click(win()!.querySelector('[data-testid="your-voices-new"]'));

      expect(selectedTab()).toBe('new');
    });

    it('moves between the tabs with the arrow keys', async () => {
      await mountSpeech(layoutName);
      await click(yoursButton());

      const tab = win()!.querySelector('#voices-tab-yours') as HTMLElement;
      tab.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
      await settle();
      expect(selectedTab()).toBe('new');
      win()!.querySelector('#voices-tab-new')!.dispatchEvent(new KeyboardEvent('keydown', { key: 'End', bubbles: true }));
      await settle();
      expect(selectedTab()).toBe('recording');
      win()!.querySelector('#voices-tab-recording')!.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
      await settle();
      expect(selectedTab()).toBe('yours');
    });
  });

  describe('the From recording tab', () => {
    async function openRecording(options: Options = {}): Promise<void> {
      await mountSpeech(layoutName, { voices: [], ...options });
      await click(newButton());
      if (selectedTab() !== 'recording') await click(win()!.querySelector('#voices-tab-recording'));
    }

    it('needs a recording of the model\'s length, its words, the tick and, to save it, a name', async () => {
      await openRecording();
      expect(useOnce().disabled).toBe(true);
      expect(saveVoice().disabled).toBe(true);

      trackApi.readFileLength.mockResolvedValue(2);
      await attach(recTab(), recording());
      await typeInto(words(), 'Tonight.');
      await click(tickBox());
      await typeInto(nameField(), 'Morgan');
      expect(recTab().querySelector('[role="alert"]')?.textContent).toContain('0:02');
      expect(useOnce().disabled).toBe(true);
      expect(saveVoice().disabled).toBe(true);

      trackApi.readFileLength.mockResolvedValue(12);
      await clearRecording();
      await attach(recTab(), recording('harbour2.wav'));
      expect(tickBox()!.checked).toBe(false);
      expect(useOnce().disabled).toBe(true);
      await click(tickBox());
      expect(useOnce().disabled).toBe(false);
      expect(saveVoice().disabled).toBe(false);
      await typeInto(words(), '   ');
      expect(useOnce().disabled).toBe(true);
      await typeInto(words(), 'Tonight.');
      await typeInto(nameField(), '   ');
      expect(useOnce().disabled).toBe(false);
      expect(saveVoice().disabled).toBe(true);
    });

    it('reads a length that misses the range in whole seconds on the side it misses', async () => {
      await openRecording();

      trackApi.readFileLength.mockResolvedValue(2.6);
      await attach(recTab(), recording());
      expect(recTab().querySelector('[role="alert"]')?.textContent).toContain('0:02');

      trackApi.readFileLength.mockResolvedValue(30.4);
      await clearRecording();
      await attach(recTab(), recording('long.wav'));
      expect(recTab().querySelector('[role="alert"]')?.textContent).toContain('0:31');
    });

    it('Use once makes the recording the voice, closes the window and leaves the one-off block under the dropdown with the tick given', async () => {
      await openRecording();
      await fillRecording();

      await click(useOnce());

      expect(win()).toBeNull();
      expect(draft.state).toMatchObject({ voiceMode: 'recording', voiceText: 'Tonight we walk through the old harbour.', voiceUpload: 'harbour.wav' });
      expect(voiceConsent.ticked).toBe(true);
      expect(oneOff()).not.toBeNull();
      expect(picker().textContent).toContain('From recording');
      expect(oneOff()!.textContent).toContain('harbour.wav');
      expect(reason()).toBeNull();
      expect(voiceApiMocks.saveRecordingVoice).not.toHaveBeenCalled();
      expect(recordingForm.recorded).toBe(false);
    });

    it('Enter in Name does the primary button\'s action: Use once while it is empty', async () => {
      await openRecording();
      await fillRecording();

      nameField().dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
      await settle();

      expect(win()).toBeNull();
      expect(draft.state.voiceMode).toBe('recording');
      expect(voiceApiMocks.saveRecordingVoice).not.toHaveBeenCalled();
    });

    it('Enter in Name saves the voice once it is named', async () => {
      voiceApiMocks.saveRecordingVoice.mockResolvedValue(MORGAN);
      await openRecording();
      await fillRecording();
      await typeInto(nameField(), 'Morgan');

      nameField().dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
      await settle();

      expect(voiceApiMocks.saveRecordingVoice).toHaveBeenCalledTimes(1);
      expect(draft.state).toMatchObject({ voiceMode: 'saved', savedVoice: 'morgan' });
    });

    it('does nothing on Enter in Name while what the primary button needs is missing', async () => {
      await openRecording();

      nameField().dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
      await settle();

      expect(win()).not.toBeNull();
      expect(draft.state.voiceMode).toBe('builtin');
    });

    it('Save voice sends the fields the server reads, chooses the voice, closes the window and says so', async () => {
      voiceApiMocks.saveRecordingVoice.mockResolvedValue(MORGAN);
      await openRecording();
      await fillRecording();
      await typeInto(nameField(), 'Morgan');

      await click(saveVoice());

      expect(voiceApiMocks.saveRecordingVoice).toHaveBeenCalledTimes(1);
      const sent = voiceApiMocks.saveRecordingVoice.mock.calls[0][0];
      expect(sent.get('model')).toBe('qwen3_tts');
      expect(sent.get('name')).toBe('Morgan');
      expect((sent.get('voice_file') as File).name).toBe('harbour.wav');
      expect(sent.get('voice_text')).toBe('Tonight we walk through the old harbour.');
      expect(sent.getAll('voice_consent')).toEqual(['on']);
      expect(draft.state).toMatchObject({ voiceMode: 'saved', savedVoice: 'morgan' });
      expect(win()).toBeNull();
      expect(toasts.map((toast) => toast.message)).toEqual(['Saved voice "Morgan"']);
      expect(picker().getAttribute('data-value')).toBe('saved:morgan');
      expect(oneOff()).toBeNull();
    });

    it('shows the server\'s error under Name and keeps the window and the form', async () => {
      voiceApiMocks.saveRecordingVoice.mockRejectedValue(new ApiError('409', 409, 'name taken'));
      await openRecording({ voices: [MORGAN] });
      await fillRecording();
      await typeInto(nameField(), 'Morgan');

      await click(saveVoice());

      expect(win()!.querySelector('[role="alert"]')?.textContent).toBe('name taken');
      expect(nameField().getAttribute('aria-invalid')).toBe('true');
      expect(win()).not.toBeNull();
      expect(words().value).toBe('Tonight we walk through the old harbour.');
      expect(draft.state.voiceMode).toBe('builtin');
    });

    it('asks for no tick for a one-off recording known to be in a default voice, until the recording is replaced', async () => {
      await mountSpeech(layoutName, { voices: [] });
      draft.setVoice({ path: '/outputs/take.flac' });
      draft.update('voiceText', 'A take.');
      voiceConsent.reset(true);
      await settle();
      expect(oneOff()!.querySelector('input[type="checkbox"]')).toBeNull();

      await click(newButton());
      expect(tickBox()).toBeNull();
      expect(useOnce().disabled).toBe(false);

      await clearRecording();
      await attach(recTab(), recording());
      expect(tickBox()).not.toBeNull();
      expect(useOnce().disabled).toBe(true);
    });

    it('still asks for the tick for a new recording after a saved voice replaced a ticked one', async () => {
      await mountSpeech(layoutName);
      draft.setVoice({ file: recording('first.wav') });
      draft.update('voiceText', 'A take.');
      voiceConsent.ticked = true;
      await settle();
      expect(reason()).toBeNull();

      await choose('saved:narrator');
      expect(oneOff()).toBeNull();
      expect(voiceConsent.ticked).toBe(false);
      await click(newButton());
      await click(win()!.querySelector('#voices-tab-recording'));
      expect(recTab().textContent).not.toContain('first.wav');
      await attach(recTab(), recording('second.wav'));
      await typeInto(words(), 'Another take.');

      expect(tickBox()).not.toBeNull();
      expect(tickBox()!.checked).toBe(false);
      expect(useOnce().disabled).toBe(true);
      await click(tickBox());
      await click(useOnce());

      expect(draft.state).toMatchObject({ voiceMode: 'recording', voiceUpload: 'second.wav' });
      expect(reason()).toBeNull();
    });

    it('keeps a half-filled form through closing the window and opening it again, with the same one-off recording in use', async () => {
      await mountSpeech(layoutName, { draftState: { voiceMode: 'recording', voicePath: '/voices/me.wav', voiceText: 'Hello from me.' } });
      await click(newButton());
      await typeInto(words(), 'Half of it.');
      await typeInto(nameField(), 'Mor');

      await click(win()!.querySelector('[aria-label="Close dialog"], [aria-label="Close"]'));
      expect(win()).toBeNull();
      await click(newButton());

      expect(recTab().textContent).toContain('me.wav');
      expect(words().value).toBe('Half of it.');
      expect(nameField().value).toBe('Mor');
    });

    it('does not carry the Voice section\'s tick into the form, and keeps a tick given in the form through closing and opening the window', async () => {
      await mountSpeech(layoutName, { draftState: { voiceMode: 'recording', voicePath: '/voices/me.wav', voiceText: 'Hello from me.' } });
      voiceConsent.ticked = true;
      await settle();
      await click(newButton());
      expect(tickBox()!.checked).toBe(false);
      expect(words().value).toBe('Hello from me.');

      await click(tickBox());
      await click(win()!.querySelector('[aria-label="Close dialog"], [aria-label="Close"]'));
      voiceConsent.ticked = false;
      await settle();
      await click(newButton());

      expect(tickBox()!.checked).toBe(true);
      expect(words().value).toBe('Hello from me.');
    });
  });

  describe('the one-off recording under the dropdown', () => {
    it('shows the recording, what it says and the tick, then the two buttons, and edits the words in the draft', async () => {
      await mountSpeech(layoutName, { draftState: { voiceMode: 'recording', voicePath: '/voices/me.wav', voiceText: 'Hi.' } });

      const block = oneOff()!;
      expect(block.textContent).toContain('me.wav');
      expect(block.querySelector('[data-testid="slot-thumb"]')).toBeNull();
      expect(block.querySelector('textarea')?.value).toBe('Hi.');
      expect(block.querySelector('input[type="checkbox"]')).not.toBeNull();
      expect(block.compareDocumentPosition(newButton()!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      expect(reason()).toContain('Tick the box');

      await typeInto(block.querySelector('textarea')!, 'Hello again.');
      expect(draft.state.voiceText).toBe('Hello again.');
    });

    it('removes the recording with ✕ and returns to the saved voice chosen before it', async () => {
      await mountSpeech(layoutName, { draftState: { voiceMode: 'saved', savedVoice: 'narrator' } });
      draft.setVoice({ path: '/voices/me.wav' });
      draft.update('voiceText', 'Hi.');
      await settle();
      expect(picker().textContent).toContain('From recording');

      await click(oneOff()!.querySelector('[aria-label="Remove the voice recording"], [aria-label="Clear the voice recording"]'));

      expect(oneOff()).toBeNull();
      expect(draft.state).toMatchObject({ voiceMode: 'saved', savedVoice: 'narrator', voicePath: null, voiceText: '' });
      expect(picker().textContent).toContain('Narrator');
    });

    it('returns to the default voice chosen before it when no saved voice was', async () => {
      await mountSpeech(layoutName);
      await choose('vivian');
      draft.setVoice({ path: '/voices/me.wav' });
      await settle();

      await click(oneOff()!.querySelector('[aria-label="Remove the voice recording"], [aria-label="Clear the voice recording"]'));

      expect(draft.state).toMatchObject({ voiceMode: 'builtin', speechSpeaker: 'vivian' });
      expect(picker().getAttribute('data-value')).toBe('vivian');
    });

    it.runIf(layoutName === 'compact')('offers the sources on the phone again once the recording in use was removed, and takes a file from this device', async () => {
      await mountSpeech(layoutName, { draftState: { voiceMode: 'recording', voicePath: '/voices/me.wav', voiceText: 'Hi.' } });
      draft.setVoice(null);
      await settle();

      expect(sheetOneOff()!.querySelector('.img-sources')).not.toBeNull();
      expect(sheetOneOff()!.querySelector('[aria-label="Remove the voice recording"]')).toBeNull();

      await attach(sheetOneOff()!, recording('new.wav'));

      expect(draft.state.voiceUpload).toBe('new.wav');
      expect(sheetOneOff()!.querySelector('.img-sources')).toBeNull();
    });

    it('keeps the block for a recording that cannot be found, turns Generate off with the reason and offers no ✕', async () => {
      await mountSpeech(layoutName, { draftState: { voiceMode: 'recording', voiceUpload: 'interview_clip.wav', voiceText: 'Hi.' } });

      expect(oneOff()).not.toBeNull();
      expect(oneOff()!.textContent).toContain('interview_clip.wav');
      expect(oneOff()!.querySelector('[aria-label="Remove the voice recording"], [aria-label="Clear the voice recording"]')).toBeNull();
      // The way to another file is on the row: the slot's two buttons on a wide window, the sources list on a phone's sheet.
      expect((layoutName === 'wide' ? oneOff()! : sheetOneOff()!).querySelector(layoutName === 'wide' ? '[aria-label="Browse for a voice recording"]' : '.img-sources')).not.toBeNull();
      expect(playButton().disabled).toBe(true);
      expect(reason()).not.toBeNull();
      expect(generate().disabled).toBe(true);
    });
  });

  describe('a voice that is gone', () => {
    it('keeps the choice, reads Deleted voice on the dropdown with no error line under it, and turns Generate off with the reason', async () => {
      await mountSpeech(layoutName, { draftState: { voiceMode: 'saved', savedVoice: 'gone' } });

      expect(draft.state).toMatchObject({ voiceMode: 'saved', savedVoice: 'gone' });
      expect(picker().textContent).toContain('Deleted voice');
      expect(scope().querySelector('.ui-help-error, .ui-help-error *, [data-testid="voice-deleted"], [data-testid="pop-voice-deleted"]')).toBeNull();
      expect(playButton().disabled).toBe(true);
      expect(reason()).toContain(VOICE_WORDS.deleted);
      expect(generate().disabled).toBe(true);
    });

    it('lets another voice be picked, which turns Generate back on', async () => {
      await mountSpeech(layoutName, { draftState: { voiceMode: 'saved', savedVoice: 'gone' } });

      await choose('saved:morgan');

      expect(reason()).toBeNull();
      expect(generate().disabled).toBe(false);
    });

    it('names no voice as deleted before the library has loaded', async () => {
      voiceApiMocks.listVoices.mockRejectedValue(new Error('offline'));
      await mountSpeech(layoutName, { draftState: { voiceMode: 'saved', savedVoice: 'narrator' } });

      expect(picker().classList.contains('voice-deleted')).toBe(false);
      expect(reason()).toBeNull();
    });
  });

  describe('the section head and the Delivery note', () => {
    it('keeps the Delivery section with its heading and one line for a saved voice, and none of its controls', async () => {
      await mountSpeech(layoutName, { draftState: { voiceMode: 'saved', savedVoice: 'narrator' } });

      if (layoutName === 'wide') {
        const delivery = target.querySelector('[data-section="delivery"]')!;
        expect(delivery.querySelector('.inspector-head')?.textContent).toContain('Delivery');
        expect(delivery.querySelector('[data-testid="delivery-recording-note"]')).not.toBeNull();
        expect(delivery.querySelector('[data-testid="delivery-controls"]')).toBeNull();
        expect(delivery.querySelector('.inspector-summary')).toBeNull();
      } else {
        expect(target.querySelector('[data-chip="delivery"]')).toBeNull();
      }
    });
  });
});

describe('Pick one, from the notice that the new voices are ready', () => {
  beforeEach(() => {
    voiceWindow.requested = false;
  });

  it('opens the Speech tab and the Voices window on New voice from another tab', async () => {
    await mountSpeech('wide', { draftState: { workflow: 'txt2img' } });
    expect(draft.state.workflow).toBe('txt2img');
    expect(win()).toBeNull();

    voiceWindow.request();
    await settle();

    expect(draft.state.workflow).toBe('speech');
    expect(selectedTab()).toBe('new');
    expect(voiceWindow.requested).toBe(false);
  });

  it('waits for a Workspace that is not open yet, then opens the window on New voice', async () => {
    voiceWindow.request();

    await mountSpeech('wide', { draftState: { workflow: 'txt2img' } });

    expect(draft.state.workflow).toBe('speech');
    expect(selectedTab()).toBe('new');
  });

  it('moves a window that is open on another tab to New voice', async () => {
    await mountSpeech('wide');
    await click(yoursButton());
    expect(selectedTab()).toBe('yours');

    voiceWindow.request();
    await settle();

    expect(selectedTab()).toBe('new');
  });
});
