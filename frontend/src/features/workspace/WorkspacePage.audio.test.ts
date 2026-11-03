// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { trackLengths } from '$lib/state/audioTrackLength.svelte';
import { draft } from '$lib/state/draft.svelte';
import { heldUploads } from '$lib/state/heldUploads.svelte';
import { requestJobSettings } from '$lib/state/pendingJobSettings.svelte';
import { voiceConsent } from '$lib/state/voiceConsent.svelte';
import type { GalleryAsset, JobContext, WorkspaceContext } from '$lib/types';

import { AUDIO_MODELS, GEMMA_AUDIO_LICENCE, SA3_LICENCE, YUE2_LICENCE, makeAudioAsset, makeSpeechAsset, withAudio } from '../../test-utils/audio';
import { makeAsset, makeContext, makeGalleryPage, makeVideoDefaults } from '../../test-utils/workspace';
import { lastEventSource } from '../../test-utils/eventSource';
import { createHost, field, mockClipboard, mountWorkspacePage, resetWorkspaceTest, settle, submitForm, type } from '../../test-utils/workspaceHarness';
import { galleryApiMocks, layoutProbe, recentPromptsApiMocks, workspaceApiMocks } from '../../test-utils/workspaceMocks';

vi.mock('$lib/state/layoutMode.svelte', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/state/layoutMode.svelte')>();
  const { layoutProbe } = await import('../../test-utils/workspaceMocks');
  return {
    ...actual,
    layout: {
      get mode() { return layoutProbe.compact ? 'compact' : 'wide'; },
      get compact() { return layoutProbe.compact; },
      get coarse() { return layoutProbe.compact; },
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
  return { ...actual, getGallery: galleryApiMocks.getGallery, deleteAsset: galleryApiMocks.deleteAsset };
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

beforeEach(() => {
  resetWorkspaceTest();
  target = createHost();
  trackLengths.reset();
  voiceConsent.reset();
  trackApi.inspectAudioTrack.mockReset().mockRejectedValue(new Error('unreadable'));
  trackApi.readFileLength.mockReset().mockResolvedValue(12);
  workspaceApiMocks.submitGenerate.mockResolvedValue({ job_id: 'job-1' } as JobContext);
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => undefined);
  window.location.hash = '#/workspace';
});

afterEach(async () => {
  if (app) { await unmount(app); app = null; }
  target.remove();
  document.body.innerHTML = '';
  vi.restoreAllMocks();
});

type Kind = 'music' | 'speech' | 'sound';

/** Mount the Workspace on a machine that runs the audio models, open on the Audio tab's *kind*. */
async function mountAudio(kind: Kind = 'music', options: { models?: typeof AUDIO_MODELS; assets?: GalleryAsset[]; compact?: boolean } = {}): Promise<WorkspaceContext> {
  const assets = options.assets ?? [];
  const context = withAudio(makeContext({ history_assets: assets }), options.models ?? AUDIO_MODELS);
  layoutProbe.compact = options.compact ?? false;
  galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage(assets));
  draft.reset();
  draft.patch({ workflow: kind });
  app = await mountWorkspacePage(context, target);
  return context;
}

const reason = () => target.querySelector('#ws-generate-reason')?.textContent ?? null;
const submitButton = () => target.querySelector('#ws-submit') as HTMLButtonElement;
const kindButton = (kind: Kind) => target.querySelector(`[data-testid="kind-control"] [data-value="${kind}"]`) as HTMLButtonElement;
const status = () => target.querySelector('#ws-music-status')?.textContent ?? null;
const consent = () => target.querySelector('#ws-voice-consent') as HTMLInputElement | null;
const section = (id: string) => target.querySelector(`[data-section="${id}"]`);
const sectionIds = () => Array.from(target.querySelectorAll('[data-section]')).map((node) => node.getAttribute('data-section'));

async function pickKind(kind: Kind): Promise<void> {
  kindButton(kind).click();
  await settle();
}

async function typeInto(id: string, value: string): Promise<void> {
  type(id, value);
  await settle();
}

async function pickVoice(file = new File(['voice'], 'narrator.wav', { type: 'audio/wav' })): Promise<void> {
  const input = target.querySelector('#ws-voice-file') as HTMLInputElement;
  Object.defineProperty(input, 'files', { configurable: true, value: [file] });
  input.dispatchEvent(new Event('change', { bubbles: true }));
  await settle();
}

async function chooseRecording(): Promise<void> {
  (target.querySelector('[data-section="voice"] [data-value="recording"]') as HTMLButtonElement).click();
  await settle();
}

async function tick(): Promise<void> {
  consent()!.click();
  await settle();
}

describe('the Audio tab', () => {
  describe('the toolbar', () => {

    it('switches Compose and Settings with the kind', async () => {
      await mountAudio('music');
      expect(target.querySelector('[data-testid="audio-compose"]')?.getAttribute('data-kind')).toBe('music');
      expect(sectionIds()).toEqual(['sampling']);
      expect(field('ws-audio-style')).not.toBeNull();
      expect(field('ws-lyrics')).not.toBeNull();

      await pickKind('speech');
      expect(target.querySelector('[data-testid="audio-compose"]')?.getAttribute('data-kind')).toBe('speech');
      expect(field('ws-audio-style')).toBeNull();
      expect(sectionIds()).toEqual(['voice', 'sampling']);
      expect(target.querySelector('#ws-model-shell, [data-testid="model-shell"]')!.textContent).toContain('Higgs TTS 3');
      expect(target.querySelector('#ws-length')).toBeNull();

      await pickKind('sound');
      expect(target.querySelector('[data-testid="audio-compose"]')?.getAttribute('data-kind')).toBe('sound');
      expect(sectionIds()).toEqual(['sampling']);
      expect(field('ws-length')).not.toBeNull();
      expect(draft.state).toMatchObject({ workflow: 'sound', audioKind: 'sound', model: 'sa3' });
    });

    it('remembers the model of each kind and the kind the tab was left on', async () => {
      const models = [...AUDIO_MODELS, { ...AUDIO_MODELS[0], id: 'yue3', label: 'YuE3' }];
      await mountAudio('music', { models });
      draft.hydrateFromContext(withAudio(makeContext(), models), 'yue3');
      await pickKind('sound');
      await pickKind('music');

      expect(draft.state.model).toBe('yue3');
      expect(draft.state.audioKind).toBe('music');
    });

    it('shows the licence mark of the selected model and follows the model', async () => {
      const models = AUDIO_MODELS.map((model) => (model.id === 'yue2' ? { ...model, licences: [YUE2_LICENCE] } : model.id === 'sa3' ? { ...model, licences: [SA3_LICENCE, GEMMA_AUDIO_LICENCE] } : model));
      await mountAudio('music', { models });
      expect(target.querySelector('.ui-pane-toolbar [data-testid="model-licence"]')?.getAttribute('aria-label')).toBe(`Licence: ${YUE2_LICENCE.name}`);

      await pickKind('speech');
      expect(target.querySelector('.ui-pane-toolbar [data-testid="model-licence"]')).toBeNull();

      await pickKind('sound');
      expect(target.querySelector('.ui-pane-toolbar [data-testid="model-licence"]')?.getAttribute('aria-label')).toBe(`Licences: ${SA3_LICENCE.name} and ${GEMMA_AUDIO_LICENCE.name}`);
    });
  });

  describe('Music', () => {
    it('stacks a required Style over an optional Lyrics, and enables Generate with Style alone', async () => {
      await mountAudio('music');
      expect(reason()).not.toBeNull();
      expect(submitButton().disabled).toBe(true);
      expect(submitButton().getAttribute('aria-describedby')).toBe('ws-generate-reason');

      await typeInto('ws-audio-style', 'Dreamy synth-pop');

      expect(field('ws-lyrics').value).toBe('');
      expect(reason()).toBeNull();
      expect(submitButton().disabled).toBe(false);
    });

    it('says Song or Instrumental as the Lyrics change', async () => {
      await mountAudio('music');
      expect(status()).toBe('Instrumental · 8 sections · add or remove section tags to change its length');

      await typeInto('ws-lyrics', '[verse]\nThe lamp is lit');
      expect(status()).toBe('Song · length follows the lyrics');
    });

    it('sends the style, the lyrics and nothing of the other kinds, as one audio job', async () => {
      await mountAudio('music');
      await typeInto('ws-audio-style', 'Dreamy synth-pop');
      await typeInto('ws-lyrics', '[verse]\nla la');
      draft.update('seed', 7);
      draft.update('runs', 2);
      await settle();

      const sent = await submitForm(target);

      expect(Object.fromEntries(sent.entries())).toEqual({ mode: 'audio', workflow: 'music', model: 'yue2', prompt: 'Dreamy synth-pop', lyrics: '[verse]\nla la', runs: '2', seed: '7' });
      // The plan that asks about downloads reads the same form.
      expect(workspaceApiMocks.planGenerate.mock.calls.at(-1)?.[0].get('mode')).toBe('audio');
    });
  });

  describe('Speech', () => {
    it('counts characters against the cap and says why Generate is off', async () => {
      await mountAudio('speech');
      await typeInto('ws-prompt', 'Hello there.');
      expect(target.querySelector('[data-testid="speech-count"]')?.textContent).toBe('12 / 10,000 characters');

      await typeInto('ws-prompt', 'a'.repeat(10_001));

      expect(reason()).toContain('10,000');
      expect(submitButton().disabled).toBe(true);
    });

    it('lists one reason at a time, in order, and disables Generate while one shows', async () => {
      await mountAudio('speech');
      expect(reason()).toContain('text');

      await typeInto('ws-prompt', 'Hello there.');
      expect(reason()).toBeNull();
      expect(submitButton().disabled).toBe(false);

      await chooseRecording();
      expect(reason()).toContain('Choose');
      expect(submitButton().disabled).toBe(true);

      trackApi.readFileLength.mockResolvedValue(2);
      await pickVoice();
      expect(target.querySelector('#ws-voice-error')?.textContent).toContain('0:02');
      expect(reason()).toContain('30');

      trackApi.readFileLength.mockResolvedValue(12);
      await pickVoice(new File(['voice2'], 'narrator2.wav', { type: 'audio/wav' }));
      expect(reason()).toContain('says');
      expect(field('ws-voice-text').getAttribute('aria-invalid')).toBe('true');

      await typeInto('ws-voice-text', 'Hello, this is my voice.');
      expect(reason()).toContain('Tick');
      expect(submitButton().disabled).toBe(true);

      await tick();
      expect(reason()).toBeNull();
      expect(submitButton().disabled).toBe(false);
    });

    it('asks for no recording in Any voice', async () => {
      await mountAudio('speech');
      await typeInto('ws-prompt', 'Hello there.');
      expect(consent()).toBeNull();
      const sent = await submitForm(target);
      expect(Object.fromEntries(sent.entries())).toEqual({ mode: 'audio', workflow: 'speech', model: 'higgs3', prompt: 'Hello there.', runs: '1' });
    });

    it('sends the recording, its words and the tick as hidden "false" plus "on", the upload winning over a path', async () => {
      await mountAudio('speech');
      await typeInto('ws-prompt', 'Hello there.');
      await chooseRecording();
      await pickVoice();
      await typeInto('ws-voice-text', 'Hello, this is my voice.');
      await tick();

      const sent = await submitForm(target);

      expect((sent.get('voice_file') as File).name).toBe('narrator.wav');
      expect(sent.has('voice_path')).toBe(false);
      expect(sent.get('voice_text')).toBe('Hello, this is my voice.');
      expect(sent.getAll('voice_consent')).toEqual(['false', 'on']);
    });

    it('sends a false tick, nothing true, for an untoggled box', async () => {
      await mountAudio('speech');
      draft.patch({ speechText: 'Hello', voiceMode: 'recording', voicePath: '/voices/me.wav', voiceText: 'hi' });
      voiceConsent.reset(true);
      await settle();

      const sent = await submitForm(target);

      expect(sent.get('voice_path')).toBe('/voices/me.wav');
      expect(sent.getAll('voice_consent')).toEqual(['false']);
    });
  });

  describe('the speaker\'s consent', () => {
    async function ticked(): Promise<void> {
      await mountAudio('speech');
      await typeInto('ws-prompt', 'Hello there.');
      await chooseRecording();
      await pickVoice();
      await typeInto('ws-voice-text', 'Hello, this is my voice.');
      await tick();
      expect(consent()!.checked).toBe(true);
    }

    it('clears with a new file, which also needs its own transcript', async () => {
      await ticked();

      await pickVoice(new File(['other'], 'other.wav', { type: 'audio/wav' }));

      expect(consent()!.checked).toBe(false);
      expect(reason()).toContain('says');
    });

    it('clears on Reuse of a take, which never carries it', async () => {
      await ticked();
      const take = makeSpeechAsset({ has_reference_voice: true, reuse_workspace_url: '#/workspace?workflow=speech&model=higgs3&prompt=Welcome&voice_path=%2Fvoices%2Fme.wav&voice_text=Hi' });
      galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage([take]));
      draft.loadFromUrl({ workflow: 'speech', model: 'higgs3', prompt: 'Welcome', voice_path: '/voices/me.wav', voice_text: 'Hi' }, withAudio(makeContext()));
      await settle();

      expect(draft.state).toMatchObject({ speechText: 'Welcome', voiceMode: 'recording', voicePath: '/voices/me.wav', voiceText: 'Hi' });
      expect(voiceConsent.ticked).toBe(false);
      expect(consent()!.checked).toBe(false);
    });

    it('clears on Load settings of an audio job, which carries none', async () => {
      await ticked();

      requestJobSettings({ mode: 'audio', workflow: 'speech', model: 'higgs3', prompt: 'Welcome', voice_path: '/voices/me.wav', voice_text: 'Hi', voice_consent: 'on' });
      await settle();

      expect(draft.state).toMatchObject({ speechText: 'Welcome', voicePath: '/voices/me.wav', voiceText: 'Hi' });
      expect(consent()!.checked).toBe(false);
    });

    it('clears on Reset, with the held recording', async () => {
      await ticked();

      (target.querySelector('#ws-reset') as HTMLButtonElement).click();
      await settle();

      expect(voiceConsent.ticked).toBe(false);
      expect(heldUploads.voice).toBeNull();
      expect(draft.state).toMatchObject({ voiceMode: 'any', voicePath: null, voiceUpload: null, voiceText: '', audioKind: 'speech' });
    });

    it('shows the tick for a voice used from a cloned take, unticked, and none for a take in the model\'s own voice', async () => {
      const context = await mountAudio('music');
      draft.loadFromUrl({ workflow: 'speech', voice_path: '/outputs/out/speech.flac', voice_text: 'Welcome back.', use_as: 'voice', voice_clone: 'true' }, context);
      await settle();
      expect(draft.state).toMatchObject({ workflow: 'speech', voiceMode: 'recording', voicePath: '/outputs/out/speech.flac', voiceText: 'Welcome back.' });
      expect(consent()!.checked).toBe(false);

      draft.loadFromUrl({ workflow: 'speech', voice_path: '/outputs/out/any.flac', voice_text: 'A take.', use_as: 'voice' }, context);
      await settle();
      expect(draft.state.voicePath).toBe('/outputs/out/any.flac');
      expect(consent()).toBeNull();
    });

    it('needs no tick before Generate for a voice that is an Any take', async () => {
      const context = await mountAudio('speech');
      draft.update('speechText', 'Hello there.');
      draft.loadFromUrl({ workflow: 'speech', voice_path: '/outputs/out/any.flac', voice_text: 'A take.', use_as: 'voice' }, context);
      await settle();

      expect(reason()).toBeNull();
      expect(submitButton().disabled).toBe(false);
    });
  });

  describe('Sound', () => {
    it('takes a Prompt and a length from 1 to the model\'s longest, 10 s to begin with', async () => {
      await mountAudio('sound');
      expect(reason()).not.toBeNull();
      const length = field('ws-length') as unknown as HTMLInputElement;
      expect(length.value).toBe('10');
      expect(target.querySelector('[data-testid="length-max"]')?.textContent).toBe('max 380');

      await typeInto('ws-prompt', 'rain on a window');

      const sent = await submitForm(target);
      expect(Object.fromEntries(sent.entries())).toEqual({ mode: 'audio', workflow: 'sound', model: 'sa3', prompt: 'rain on a window', seconds: '10', runs: '1' });
    });
  });

  describe('the model status', () => {
    it('refreshes an audio model\'s download state after a job ends', async () => {
      await mountAudio('sound', { models: AUDIO_MODELS.map((model) => ({ ...model, downloaded: false })) });
      expect(target.querySelector('[data-testid="model-download-status"]')).not.toBeNull();

      workspaceApiMocks.getWorkspaceCoreContext.mockResolvedValue(withAudio(makeContext(), AUDIO_MODELS.map((model) => ({ ...model, downloaded: true }))));
      await typeInto('ws-prompt', 'rain on a window');
      await submitForm(target);
      lastEventSource().emit('job_failed', { type: 'job_failed', job_id: 'job-1', message: 'boom' });
      await settle();

      expect(target.querySelector('[data-testid="model-download-status"]')).toBeNull();
    });
  });

  describe('Recent prompts and the footer', () => {
    it('offers Recent prompts only, of the open kind, and puts a pick in the open kind\'s box', async () => {
      recentPromptsApiMocks.getRecentPrompts.mockResolvedValue({ entries: [{ format: 'text', prompt: 'a door slam', scene: '', negative_prompt: '', mode: 'sound', used_at: 1 }] });
      await mountAudio('sound');
      const tools = target.querySelector('[data-kind="sound"] .ui-prompt-tools')!;
      expect(Array.from(tools.querySelectorAll('button')).map((button) => button.getAttribute('aria-label'))).toEqual(['Recent prompts']);

      (target.querySelector('#ws-recent-prompts') as HTMLButtonElement).click();
      await settle();
      expect(recentPromptsApiMocks.getRecentPrompts).toHaveBeenCalledWith('', 'sound');
      (document.querySelector('[role="option"]') as HTMLElement).click();
      await settle();

      expect(draft.state.soundPrompt).toBe('a door slam');
      expect(target.querySelector('#ws-enhance-toggle')).toBeNull();
      expect(target.querySelector('#ws-json-prompt-toggle')).toBeNull();
    });

    it('lists the open kind\'s recents and brings a song\'s lyrics back with its style', async () => {
      recentPromptsApiMocks.getRecentPrompts.mockResolvedValue({ entries: [{ format: 'text', prompt: 'calm piano', scene: '[verse]\nla la', negative_prompt: '', mode: 'music', used_at: 1 }] });
      await mountAudio('music');

      (target.querySelector('#ws-recent-prompts') as HTMLButtonElement).click();
      await settle();
      expect(recentPromptsApiMocks.getRecentPrompts).toHaveBeenCalledWith('', 'music');
      (document.querySelector('[role="option"]') as HTMLElement).click();
      await settle();

      expect(draft.state.audioStyle).toBe('calm piano');
      expect(draft.state.lyrics).toBe('[verse]\nla la');
    });

    it('follows the Copy CLI of the form, with the live tick', async () => {
      const clipboard = mockClipboard();
      await mountAudio('speech');
      await typeInto('ws-prompt', 'Hello there.');
      await chooseRecording();
      draft.patch({ voicePath: '/voices/me.wav', voiceText: 'Hello, this is my voice.' });
      await settle();

      (target.querySelector('button[aria-label="Copy CLI command"]') as HTMLButtonElement).click();
      await settle();
      expect(clipboard.mock.calls.at(-1)?.[0]).toBe("ziv audio --model higgs3 --prompt 'Hello there.' --voice /voices/me.wav --voice-text 'Hello, this is my voice.' --output /tmp/output");

      await tick();
      (target.querySelector('button[aria-label="Copy CLI command"]') as HTMLButtonElement).click();
      await settle();
      expect(clipboard.mock.calls.at(-1)?.[0]).toBe("ziv audio --model higgs3 --prompt 'Hello there.' --voice /voices/me.wav --voice-text 'Hello, this is my voice.' --voice-consent --output /tmp/output");
    });
  });

  describe('held uploads', () => {
    it('keeps the voice recording through a visit to the Gallery', async () => {
      await mountAudio('speech');
      await chooseRecording();
      await pickVoice();
      expect(target.querySelector('#ws-voice-label')).not.toBeNull();

      await unmount(app!);
      app = null;
      target.innerHTML = '';
      const context = withAudio(makeContext());
      app = await mountWorkspacePage(context, target);

      expect(heldUploads.voice?.name).toBe('narrator.wav');
      expect(target.querySelector('[data-section="voice"]')?.textContent).toContain('narrator.wav');
    });
  });

  describe('the phone', () => {
    beforeEach(() => {
      Element.prototype.animate = vi.fn(() => ({ cancel: vi.fn(), finished: Promise.resolve(), onfinish: null })) as unknown as Element['animate'];
    });
    afterEach(() => {
      delete (Element.prototype as Partial<Element>).animate;
    });
    const chips = () => Array.from(target.querySelectorAll<HTMLElement>('[data-testid="settings-strip"] [data-chip]')).map((chip) => chip.dataset.chip);

    it('opens Kind as radio rows and switches the kind', async () => {
      await mountAudio('music', { compact: true });
      (target.querySelector('[data-chip="kind"]') as HTMLButtonElement).click();
      await settle();

      const rows = Array.from(target.querySelectorAll('[data-testid="setting-popover"] [role="radio"]'));
      expect(rows.map((row) => row.getAttribute('aria-checked'))).toEqual(['true', 'false', 'false']);
      (rows[1] as HTMLButtonElement).click();
      await settle();

      expect(draft.state.workflow).toBe('speech');
      expect(chips()).toContain('voice');
    });

    it('opens the Voice popover with What it says and the tick', async () => {
      await mountAudio('speech', { compact: true });
      (target.querySelector('[data-chip="voice"]') as HTMLButtonElement).click();
      await settle();
      const pop = () => target.querySelector('[data-testid="setting-popover"]') as HTMLElement;

      (pop().querySelector('[data-value="recording"]') as HTMLButtonElement).click();
      await settle();
      expect(pop().querySelector('#ws-pop-voice-text')).not.toBeNull();

      draft.setVoice({ path: '/voices/me.wav' });
      await settle();
      expect(pop().querySelector('input[type="checkbox"]')).not.toBeNull();
      expect(pop().querySelector('input[type="checkbox"]')).toHaveProperty('checked', false);
    });

    it('shows the reason above the composer\'s row and points Generate at it', async () => {
      await mountAudio('music', { compact: true });

      expect(target.querySelector('[data-testid="audio-reason"]')).not.toBeNull();
      const generate = target.querySelector('#ws-submit-compact') as HTMLButtonElement;
      expect(generate.disabled).toBe(true);
      expect(generate.getAttribute('aria-describedby')).toBe('ws-generate-reason-compact');
    });
  });

  describe('Use as and Reuse', () => {
    it('opens the Speech tab from another one with the take as the voice and its words as the transcript', async () => {
      const take = makeSpeechAsset({ has_reference_voice: true });
      await mountAudio('music', { assets: [take] });
      (target.querySelector('.latest-region button[aria-label="More actions for speech.flac"]') as HTMLButtonElement).click();
      flushSync();
      (document.querySelector('[role="menu"] [data-action="use-voice"]') as HTMLElement).click();
      await settle();

      expect(draft.state).toMatchObject({ workflow: 'speech', voiceMode: 'recording', voicePath: '/outputs/out/speech.flac', voiceText: 'Welcome back to the workshop.' });
      expect(consent()!.checked).toBe(false);
    });

    it('fills the tab from a reused take: kind, model, words, lyrics and seed', async () => {
      const take = makeAudioAsset({ reuse_workspace_url: '#/workspace?workflow=music&model=yue2&prompt=Dreamy+synth-pop&lyrics=%5Bverse%5D%0Ala&seed=4127' });
      await mountAudio('speech', { assets: [take] });
      (target.querySelector('.latest-region button[aria-label="Reuse settings from song.flac"]') as HTMLButtonElement).click();
      await settle();

      expect(draft.state).toMatchObject({ workflow: 'music', model: 'yue2', audioStyle: 'Dreamy synth-pop', lyrics: '[verse]\nla', seed: 4127 });
    });
  });

  describe('Reset', () => {
    it('keeps the text and clears the length, the batch and the seed, keeping the kind and the models', async () => {
      await mountAudio('sound');
      await typeInto('ws-prompt', 'rain');
      draft.patch({ audioStyle: 'a style', lyrics: 'la', audioSeconds: 60, runs: 3, seed: 5 });
      await settle();

      (target.querySelector('#ws-reset') as HTMLButtonElement).click();
      await settle();

      expect(draft.state).toMatchObject({ audioStyle: 'a style', lyrics: 'la', soundPrompt: 'rain', audioSeconds: 10, runs: 1, seed: null, audioKind: 'sound', workflow: 'sound', model: 'sa3' });
      expect(draft.state.audioModels.sound).toBe('sa3');
    });
  });
});

describe('the tab beside the others', () => {
  it('keeps the image prompt through a visit to the Audio tab', async () => {
    draft.reset();
    draft.patch({ workflow: 'txt2img', prompt: 'a lighthouse' });
    const context = withAudio(makeContext());
    app = await mountWorkspacePage(context, target);

    draft.setTab('audio');
    draft.onTabChange(context);
    await settle();
    expect(draft.state).toMatchObject({ workflow: 'music', prompt: '' });
    draft.setTab('image');
    draft.onTabChange(context);
    await settle();

    expect(draft.state).toMatchObject({ workflow: 'txt2img', prompt: 'a lighthouse', model: 'zit' });
  });

  it('falls back to Image when the machine stops running audio models', async () => {
    draft.reset();
    draft.patch({ workflow: 'speech' });
    app = await mountWorkspacePage(makeContext(), target);

    expect(draft.state.workflow).toBe('txt2img');
    expect(target.querySelector('[data-testid="kind-control"]')).toBeNull();
  });
});

/** Dispatch a touch event with one finger at (x, y). */
function touch(node: Element, type: string, x: number, y: number): void {
  const point = { clientX: x, clientY: y, target: node } as unknown as Touch;
  const event = new Event(type, { bubbles: true, cancelable: true }) as TouchEvent;
  Object.defineProperty(event, 'touches', { value: type === 'touchend' ? [] : [point] });
  Object.defineProperty(event, 'changedTouches', { value: [point] });
  node.dispatchEvent(event);
}

function audioContext(assets: GalleryAsset[], overrides: Partial<WorkspaceContext> = {}): WorkspaceContext {
  return makeContext({ history_assets: assets, audio_models: AUDIO_MODELS, ...overrides });
}

async function mountWith(assets: GalleryAsset[], overrides: Partial<WorkspaceContext> = {}): Promise<void> {
  galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage(assets));
  app = await mountWorkspacePage(audioContext(assets, overrides), target);
}

const caption = () => target.querySelector('[data-testid="audio-caption"]');

describe('the stage with an audio take', () => {
  const image = makeAsset({ id: 'out/older.png', filename: 'older.png' });

  it('plays the latest take in a player panel and captions it with its words, model, length and seed', async () => {
    await mountWith([makeAudioAsset(), image]);

    expect(target.querySelector('audio')?.getAttribute('aria-label')).toBe('Play song.flac');
    expect(target.querySelector('.latest-region img')).toBeNull();
    expect(Array.from(caption()!.children).map((part) => part.textContent)).toEqual(['Dreamy synth-pop, warm analog pads', 'YuE2 3B', '2:00', 'seed 4127']);
    expect(caption()!.querySelector('.ui-badge-warning')).toBeNull();
  });

  it('keeps a sideways drag that starts on the waveform from paging the stage, and pages on the same drag elsewhere', async () => {
    await mountWith([makeAudioAsset(), image]);
    const wave = target.querySelector('[data-testid="audio-wave"]')!;

    touch(wave, 'touchstart', 300, 200);
    touch(wave, 'touchmove', 250, 202);
    touch(wave, 'touchmove', 40, 204);
    touch(wave, 'touchend', 40, 204);
    flushSync();
    expect(target.querySelector('audio')).not.toBeNull();

    const stage = target.querySelector('.stage-body')!;
    touch(stage, 'touchstart', 300, 200);
    touch(stage, 'touchmove', 250, 202);
    touch(stage, 'touchmove', 40, 204);
    touch(stage, 'touchend', 40, 204);
    await settle();
    expect(target.querySelector('audio')).toBeNull();
    expect(target.querySelector('.latest-region img')).not.toBeNull();
  });
});

describe('Use as from an audio take on the stage', () => {
  const song = makeAudioAsset();

  function openMenu(): void {
    (target.querySelector('.latest-region button[aria-label="More actions for song.flac"]') as HTMLButtonElement).click();
    flushSync();
  }
  const item = (action: string) => document.querySelector<HTMLElement>(`[role="menu"] [data-action="${action}"]`);

  it('makes it the audio track of the Video tab from 0 s, leaving the other slots alone', async () => {
    await mountWith([song]);
    draft.patch({ workflow: 'txt2vid', model: 'ltx-8', firstFramePath: '/in/first.png', audioTrackPath: '/in/old.wav', audioTrackStart: 12 });
    await settle();

    openMenu();
    expect(item('use-audio-track')?.textContent).toContain('Video tab, from 0 s');
    item('use-audio-track')!.click();
    await settle();

    expect(draft.state).toMatchObject({ audioTrackPath: '/outputs/out/song.flac', audioTrackStart: 0, audioTrackUpload: null, firstFramePath: '/in/first.png' });
    expect(draft.state.workflow).toBe('img2vid');
  });

  it('disables the audio track with the reason when the video model cannot follow one', async () => {
    const context = audioContext([song], { video_model_defaults: { 'ltx-8': makeVideoDefaults({ hidden_controls: ['audio_track'] }) } });
    galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage([song]));
    app = await mountWorkspacePage(context, target);

    openMenu();

    expect(item('use-audio-track')?.getAttribute('aria-disabled')).toBe('true');
    expect(item('use-audio-track')?.textContent).toContain("ltx-8 can't follow an audio track");
    item('use-audio-track')!.click();
    await settle();
    expect(draft.state.audioTrackPath).not.toBe('/outputs/out/song.flac');
  });
});
