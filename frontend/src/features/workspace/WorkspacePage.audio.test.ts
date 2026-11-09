// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { trackLengths } from '$lib/state/audioTrackLength.svelte';
import { draft } from '$lib/state/draft.svelte';
import { downloadedTiers } from '$lib/state/enhancerDownload.svelte';
import { heldUploads } from '$lib/state/heldUploads.svelte';
import { projectPick } from '$lib/state/projectPick.svelte';
import { jobStore } from '$lib/state/job.svelte';
import { requestJobSettings } from '$lib/state/pendingJobSettings.svelte';
import { toasts } from '$lib/state/toasts.svelte';
import { voiceConsent } from '$lib/state/voiceConsent.svelte';
import type { AudioEnhanceFrame, AudioRewrites, DraftState, GalleryAsset, JobContext, WorkspaceContext } from '$lib/types';

import { AUDIO_MODELS, GEMMA_AUDIO_LICENCE, SA3_LICENCE, SA3_MUSIC, YUE2_LICENCE, makeAudioAsset, makeSpeechAsset, withAudio, withAudioEnhancer } from '../../test-utils/audio';
import { jobSnap, makeAsset, makeContext, makeGalleryPage, makeVideoDefaults } from '../../test-utils/workspace';
import { lastEventSource } from '../../test-utils/eventSource';
import { openOptions, optionLabels, pickOption } from '../../test-utils/select';
import { createHost, field, mockClipboard, mountWorkspacePage, pressKey, resetWorkspaceTest, settle, submitForm, type } from '../../test-utils/workspaceHarness';
import { enhanceApiMocks, galleryApiMocks, layoutProbe, recentPromptsApiMocks, workspaceApiMocks } from '../../test-utils/workspaceMocks';

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
/** What the elements an element's aria-describedby names say. */
const descriptionOf = (node: Element): string => (node.getAttribute('aria-describedby') ?? '').split(' ').map((id) => document.getElementById(id)?.textContent ?? '').join(' ').trim();
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

/** Pick an option of the Voice select (a default voice id, or `saved:<slug>`) in the Settings column, or in the open Voice popover. */
async function chooseVoice(choice: string, within: ParentNode = target): Promise<void> {
  pickOption(within.querySelector('[data-testid="voice-select"] button, [data-testid="pop-voice-select"] button') as HTMLElement, choice);
  await settle();
}

/** Make a one-off recording the voice, as Use once does in the Voices window: the block under the Voice select then holds it. */
async function chooseRecording(file = new File(['voice'], 'narrator.wav', { type: 'audio/wav' })): Promise<void> {
  draft.setVoice({ file });
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
      expect(sectionIds()).toEqual(['music', 'sampling']);
      expect(field('ws-audio-style')).not.toBeNull();
      expect(field('ws-lyrics')).not.toBeNull();

      await pickKind('speech');
      expect(target.querySelector('[data-testid="audio-compose"]')?.getAttribute('data-kind')).toBe('speech');
      expect(field('ws-audio-style')).toBeNull();
      expect(sectionIds()).toEqual(['voice', 'delivery', 'sampling']);
      expect(target.querySelector('#ws-model-shell, [data-testid="model-shell"]')!.textContent).toContain('Qwen3-TTS 1.7B');
      expect(target.querySelector('#ws-length')).toBeNull();

      await pickKind('sound');
      expect(target.querySelector('[data-testid="audio-compose"]')?.getAttribute('data-kind')).toBe('sound');
      expect(sectionIds()).toEqual(['sound', 'sampling']);
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

      expect(Object.fromEntries(sent.entries())).toEqual({ mode: 'audio', workflow: 'music', model: 'yue2', prompt: 'Dreamy synth-pop', lyrics: '[verse]\nla la', tempo: '', key: '', runs: '2', seed: '7' });
      // The plan that asks about downloads reads the same form.
      expect(workspaceApiMocks.planGenerate.mock.calls.at(-1)?.[0].get('mode')).toBe('audio');
    });
  });

  describe('the picked project', () => {
    afterEach(() => projectPick.set(null));

    it('is sent with an audio job, and left out with All files', async () => {
      await mountAudio('music');
      await typeInto('ws-audio-style', 'Dreamy synth-pop');
      await settle();

      expect((await submitForm(target)).has('project')).toBe(false);

      projectPick.set('Night Bus');
      await settle();
      expect((await submitForm(target)).get('project')).toBe('Night Bus');
    });
  });

  describe('Music settings', () => {
    const summary = () => section('music')?.querySelector('.inspector-summary')?.textContent ?? null;
    const dot = (label: string) => Array.from(section('music')!.querySelectorAll('.inspector-row')).find((row) => row.querySelector('.inspector-label-text')?.textContent === label)?.getAttribute('data-changed');
    const commit = async (id: string, value: string): Promise<void> => {
      const input = field(id) as unknown as HTMLInputElement;
      input.value = value;
      input.dispatchEvent(new Event('change', { bubbles: true }));
      await settle();
    };
    const keyBox = () => target.querySelector('#ws-key') as HTMLElement;

    it('puts a Music section before Sampling, with Tempo and Key on Auto and no Length for YuE2', async () => {
      await mountAudio('music');

      expect(sectionIds()).toEqual(['music', 'sampling']);
      expect(summary()).toBe('Auto');
      expect((field('ws-tempo') as unknown as HTMLInputElement).placeholder).toBe('Auto');
      expect(field('ws-tempo').value).toBe('');
      expect(keyBox().textContent).toContain('Auto');
      expect(target.querySelector('#ws-music-length')).toBeNull();
    });

    it('takes a tempo from 40 to 220 and puts it back to Auto when emptied', async () => {
      await mountAudio('music');

      await commit('ws-tempo', '500');
      expect(draft.state.audioTempo).toBe(220);
      await commit('ws-tempo', '3');
      expect(draft.state.audioTempo).toBe(40);
      await commit('ws-tempo', '92.4');
      expect(draft.state.audioTempo).toBe(92);
      expect(summary()).toBe('92 BPM');
      expect(dot('Tempo')).toBe('true');

      await commit('ws-tempo', '');
      expect(draft.state.audioTempo).toBeNull();
      expect(summary()).toBe('Auto');
      expect(dot('Tempo')).toBe('false');
    });

    it('lists Auto, then the twelve major and the twelve minor keys in two columns, and shows the choice in the header', async () => {
      await mountAudio('music');
      const options = openOptions(keyBox());

      expect(options).toHaveLength(25);
      expect(options.map((option) => option.dataset.value).slice(0, 3)).toEqual(['', 'C', 'Db']);
      expect(options.find((option) => option.dataset.value === 'F#m')?.textContent).toContain('F♯ minor');
      expect(document.querySelectorAll('.ui-select-column')).toHaveLength(2);
      expect(Array.from(document.querySelectorAll('.ui-select-group')).map((node) => node.textContent)).toEqual(['Major', 'Minor']);

      pickOption(keyBox(), 'Am');
      await settle();
      expect(draft.state.audioKey).toBe('Am');
      expect(keyBox().textContent).toContain('A minor');
      expect(dot('Key')).toBe('true');

      commit('ws-tempo', '92');
      await settle();
      expect(summary()).toBe('92 BPM · A minor');
    });

    it('puts a row back to Auto with its reset', async () => {
      await mountAudio('music');
      draft.patch({ audioTempo: 120, audioKey: 'Am' });
      await settle();

      (section('music')!.querySelector('[aria-label="Reset Tempo to default"]') as HTMLButtonElement).click();
      await settle();

      expect(draft.state).toMatchObject({ audioTempo: null, audioKey: 'Am' });
      (section('music')!.querySelector('[aria-label="Reset Key to default"]') as HTMLButtonElement).click();
      await settle();
      expect(draft.state.audioKey).toBeNull();
    });

    it('sends the tempo and the key, and nothing for Auto', async () => {
      await mountAudio('music');
      await typeInto('ws-audio-style', 'Dreamy synth-pop');
      draft.patch({ audioTempo: 92, audioKey: 'F#m' });
      await settle();

      expect(Object.fromEntries((await submitForm(target)).entries())).toMatchObject({ tempo: '92', key: 'F#m' });

      draft.patch({ audioTempo: null, audioKey: null });
      await settle();
      expect(Object.fromEntries((await submitForm(target)).entries())).toMatchObject({ tempo: '', key: '' });
    });

    it('has no Music section on Speech or Sound', async () => {
      await mountAudio('music');
      await pickKind('speech');
      expect(section('music')).toBeNull();
      await pickKind('sound');
      expect(section('music')).toBeNull();
    });
  });

  describe('Stable Audio under Music', () => {
    const models = [...AUDIO_MODELS, SA3_MUSIC];
    const mountStableAudio = async (options: { compact?: boolean } = {}): Promise<void> => {
      const context = await mountAudio('music', { models, ...options });
      draft.hydrateFromContext(context, 'sa3');
      await settle();
    };

    it('is offered under Music next to YuE2, and under Sound', async () => {
      await mountAudio('music', { models });
      const labels = optionLabels(target.querySelector('#ws-model') as HTMLElement);

      expect(labels.filter((label) => label.includes('YuE2') || label.includes('Stable Audio'))).toHaveLength(2);
      await pickKind('sound');
      expect(optionLabels(target.querySelector('#ws-model') as HTMLElement).some((label) => label.includes('Stable Audio'))).toBe(true);
    });

    it('hides the Lyrics box and its status line, and shows Length from 1 to 380 with 30 s to begin with', async () => {
      await mountStableAudio();

      expect(draft.state.model).toBe('sa3');
      expect(field('ws-audio-style')).not.toBeNull();
      expect(target.querySelector('#ws-lyrics')).toBeNull();
      expect(status()).toBeNull();
      expect(target.querySelector('#ws-recent-prompts')).not.toBeNull();
      expect(field('ws-music-length').value).toBe('30');
      expect(target.querySelector('[data-testid="music-length-max"]')?.textContent).toBe('max 380');
      expect(section('music')?.querySelector('.inspector-summary')?.textContent).toBe('Auto · 30 s');
      expect(target.querySelector('#ws-length')).toBeNull();
    });

    it('keeps a typed length as the song\'s own, and 30 stays the model\'s default, not a change', async () => {
      await mountStableAudio();
      const length = field('ws-music-length') as unknown as HTMLInputElement;
      length.value = '90';
      length.dispatchEvent(new Event('change', { bubbles: true }));
      await settle();

      expect(draft.state.musicSeconds).toBe(90);
      expect(section('music')?.querySelector('.inspector-summary')?.textContent).toBe('Auto · 90 s');
      expect(draft.state.audioSeconds).toBe(0);

      length.value = '30';
      length.dispatchEvent(new Event('change', { bubbles: true }));
      await settle();
      expect(draft.state.musicSeconds).toBe(0);
    });

    it('sends the tempo, the key and the length, and no lyrics', async () => {
      await mountStableAudio();
      await typeInto('ws-audio-style', 'Warm lo-fi beat');
      draft.patch({ audioTempo: 80, audioKey: 'Dm', musicSeconds: 45, lyrics: 'kept but unsent' });
      await settle();

      const sent = Object.fromEntries((await submitForm(target)).entries());

      expect(sent).toEqual({ mode: 'audio', workflow: 'music', model: 'sa3', prompt: 'Warm lo-fi beat', tempo: '80', key: 'Dm', seconds: '45', runs: '1' });
      expect(draft.state.lyrics).toBe('kept but unsent');
    });

    it('keeps the lyrics, the tempo and the key through a visit to Stable Audio and sends the lyrics again on YuE2', async () => {
      const context = await mountAudio('music', { models });
      await typeInto('ws-audio-style', 'Dreamy synth-pop');
      await typeInto('ws-lyrics', '[verse]\nla la');
      draft.patch({ audioTempo: 92, audioKey: 'Am' });
      await settle();

      draft.hydrateFromContext(context, 'sa3');
      await settle();
      expect(target.querySelector('#ws-lyrics')).toBeNull();
      expect(Object.fromEntries((await submitForm(target)).entries())).toMatchObject({ model: 'sa3', tempo: '92', key: 'Am' });
      expect(Object.fromEntries((await submitForm(target)).entries())).not.toHaveProperty('lyrics');

      draft.hydrateFromContext(context, 'yue2');
      await settle();
      expect(draft.state.lyrics).toBe('[verse]\nla la');
      expect(Object.fromEntries((await submitForm(target)).entries())).toMatchObject({ model: 'yue2', lyrics: '[verse]\nla la', tempo: '92', key: 'Am' });
    });

    it('keeps the song\'s length and the sound\'s length apart on the same model, each in its own Length row', async () => {
      await mountStableAudio();
      draft.patch({ musicSeconds: 90 });
      await settle();

      await pickKind('sound');
      expect(draft.state.model).toBe('sa3');
      expect(target.querySelector('#ws-music-length')).toBeNull();
      expect(field('ws-length').value).toBe('');
      draft.patch({ audioSeconds: 20 });
      await settle();
      await typeInto('ws-prompt', 'rain');
      expect(Object.fromEntries((await submitForm(target)).entries())).toMatchObject({ workflow: 'sound', seconds: '20' });

      await pickKind('music');
      expect(target.querySelector('#ws-length')).toBeNull();
      expect(field('ws-music-length').value).toBe('90');
      expect(draft.state.audioSeconds).toBe(20);
    });

    it('does not write a recent song\'s lyrics into the draft when the model has no Lyrics box', async () => {
      recentPromptsApiMocks.getRecentPrompts.mockResolvedValue({ entries: [{ format: 'text', prompt: 'calm piano', scene: '[verse]\nla la', negative_prompt: '', mode: 'music', used_at: 1 }] });
      await mountStableAudio();
      draft.patch({ lyrics: 'my own lyrics' });
      await settle();

      (target.querySelector('#ws-recent-prompts') as HTMLButtonElement).click();
      await settle();
      (document.querySelector('[role="option"]') as HTMLElement).click();
      await settle();

      expect(draft.state.audioStyle).toBe('calm piano');
      expect(draft.state.lyrics).toBe('my own lyrics');
    });

    it('sends no length for YuE2, whose song length follows its lyrics, and keeps the held value', async () => {
      await mountAudio('music', { models });
      await typeInto('ws-audio-style', 'Dreamy synth-pop');
      draft.patch({ musicSeconds: 45 });
      await settle();

      expect(Object.fromEntries((await submitForm(target)).entries())).not.toHaveProperty('seconds');
      expect(draft.state.musicSeconds).toBe(45);
    });
  });

  describe('Speech', () => {
    it('counts characters against the cap and says why Generate is off', async () => {
      await mountAudio('speech');
      await typeInto('ws-prompt', 'Hello there.');
      expect(target.querySelector('[data-testid="text-count"]')?.textContent).toBe('12 / 10,000 characters · about 0:01');

      await typeInto('ws-prompt', 'a'.repeat(10_001));

      expect(reason()).toContain('10,000');
      expect(submitButton().disabled).toBe(true);
    });

    it('turns Generate off for a bracketed word that is not a Delivery mark, in the server\'s words, and on again once it is gone', async () => {
      await mountAudio('speech');

      await typeInto('ws-prompt', '[sadd] Hello there.');
      expect(reason()).toBe("[sadd] isn't a Delivery mark. Use one of the Delivery marks, or remove the brackets.");
      expect(submitButton().disabled).toBe(true);

      await typeInto('ws-prompt', '[sad] Hello there.');
      expect(reason()).toBeNull();
      expect(submitButton().disabled).toBe(false);
    });

    it('has one line under the text: the characters and the reading time, with no marks counted, and no note about reading in parts', async () => {
      await mountAudio('speech');
      await typeInto('ws-prompt', '[sad] Hello there.');

      const compose = target.querySelector('[data-testid="audio-compose"]') as HTMLElement;
      expect(compose.querySelectorAll('[data-testid="prompt-statusline"]')).toHaveLength(0);
      expect(compose.querySelectorAll('[data-testid="text-count"]')).toHaveLength(1);
      expect(compose.querySelector('[data-testid="text-count"]')?.textContent).toBe('13 / 10,000 characters · about 0:01');
      expect(target.querySelectorAll('[role="tab"]')).toHaveLength(0);
    });

    describe('Insert', () => {
      const insertButton = () => target.querySelector('#ws-insert') as HTMLButtonElement | null;
      const menu = () => document.querySelector('[role="menu"][aria-label="Insert"]') as HTMLElement | null;
      const action = (id: string) => document.querySelector(`[data-action="${id}"]`) as HTMLElement | null;
      const textbox = () => field('ws-prompt') as HTMLTextAreaElement;
      const caretAt = (position: number, end = position) => textbox().setSelectionRange(position, end);

      async function openMenu(): Promise<void> {
        insertButton()!.click();
        await settle();
      }

      async function choose(id: string): Promise<void> {
        action(id)!.click();
        await settle();
      }

      it('is in the Speech tools row beside Recent prompts, and not on Music or Sound', async () => {
        await mountAudio('speech');
        expect(insertButton()).not.toBeNull();
        expect(insertButton()!.getAttribute('aria-haspopup')).toBe('menu');
        expect(target.querySelector('#ws-recent-prompts')).not.toBeNull();

        await pickKind('sound');
        expect(insertButton()).toBeNull();
        await pickKind('music');
        expect(insertButton()).toBeNull();
      });

      it('offers Pause, Laugh and Delivery from here, with no Long pause', async () => {
        await mountAudio('speech');
        await openMenu();

        expect(Array.from(menu()!.querySelectorAll('[role="menuitem"]')).map((node) => node.getAttribute('data-action'))).toEqual(['insert-pause', 'insert-laugh', 'insert-delivery']);
        expect(menu()!.textContent).not.toMatch(/long pause/i);
      });

      it('puts a pause where the caret is and draws the draft along', async () => {
        await mountAudio('speech');
        await typeInto('ws-prompt', 'Well I think so');
        caretAt(4);
        await openMenu();
        await choose('insert-pause');

        expect(draft.state.speechText).toBe('Well… I think so');
        expect(textbox().value).toBe('Well… I think so');
        expect(textbox().selectionStart).toBe(5);
        expect(menu()).toBeNull();
      });

      it('replaces the selected words with a laugh, spaced off the words around it', async () => {
        await mountAudio('speech');
        await typeInto('ws-prompt', 'That was so funny');
        caretAt(9, 11);
        await openMenu();
        await choose('insert-laugh');

        expect(draft.state.speechText).toBe('That was haha funny');
      });

      it('sets the delivery from the start of the sentence the caret is in', async () => {
        await mountAudio('speech');
        await typeInto('ws-prompt', 'I was fine. You never even called. We spoke later.');
        caretAt(25);
        await openMenu();
        expect(action('insert-mark-sad')).toBeNull();
        await choose('insert-delivery');
        expect(Array.from(document.querySelectorAll('[role="menu"] .action-menu-heading')).map((node) => node.textContent)).toEqual(['Bright', 'Warm', 'Thoughtful', 'Heavy', 'Manner', 'Reset']);
        await choose('insert-mark-sad');

        expect(draft.state.speechText).toBe('I was fine. [sad] You never even called. We spoke later.');
        expect(textbox().selectionStart).toBe(31);
      });

      it('swaps the mark that already opens the sentence for the new one, and can return to the settings', async () => {
        await mountAudio('speech');
        await typeInto('ws-prompt', 'Hello. [sad] You never called.');
        caretAt(20);
        await openMenu();
        await choose('insert-delivery');
        await choose('insert-mark-whisper');
        expect(draft.state.speechText).toBe('Hello. [whisper] You never called.');

        caretAt(20);
        await openMenu();
        await choose('insert-delivery');
        await choose('insert-mark-as-set');
        expect(draft.state.speechText).toBe('Hello. [as set] You never called.');
      });

      it('leaves out the delivery marks for a voice that follows a recording', async () => {
        await mountAudio('speech');
        await chooseRecording();
        await openMenu();

        expect(action('insert-pause')).not.toBeNull();
        expect(action('insert-laugh')).not.toBeNull();
        expect(action('insert-delivery')).toBeNull();
      });

      it('opens at the caret with Alt+Enter, keeps the text as it is and gives it the focus back on Escape', async () => {
        await mountAudio('speech');
        await typeInto('ws-prompt', 'Well I think so');
        caretAt(4);
        textbox().focus();

        const event = pressKey('Enter', { altKey: true }, textbox());
        await settle();

        expect(event.defaultPrevented).toBe(true);
        expect(menu()).not.toBeNull();
        expect(draft.state.speechText).toBe('Well I think so');

        pressKey('Escape', {}, menu()!);
        await settle();
        expect(menu()).toBeNull();
        expect(document.activeElement).toBe(textbox());
        expect(textbox().selectionStart).toBe(4);
      });

      it('inserts where Alt+Enter was pressed', async () => {
        await mountAudio('speech');
        await typeInto('ws-prompt', 'Well I think so');
        caretAt(15);
        textbox().focus();
        pressKey('Enter', { altKey: true }, textbox());
        await settle();
        await choose('insert-laugh');

        expect(draft.state.speechText).toBe('Well I think so haha');
      });

      it('ignores a plain Enter and Ctrl+Alt+Enter', async () => {
        await mountAudio('speech');
        textbox().focus();

        expect(pressKey('Enter', {}, textbox()).defaultPrevented).toBe(false);
        pressKey('Enter', { altKey: true, ctrlKey: true }, textbox());
        await settle();
        expect(menu()).toBeNull();
      });

      it('opens as a sheet on a phone that says where it inserts, and inserts without raising the keyboard', async () => {
        await mountAudio('speech', { compact: true });
        await typeInto('ws-prompt', 'Well I think so');
        caretAt(4);
        await openMenu();

        const sheet = document.querySelector('[data-testid="action-sheet"]') as HTMLElement;
        expect(sheet.textContent).toContain('Inserts at the caret: Well | I think so');
        await choose('insert-pause');

        expect(draft.state.speechText).toBe('Well… I think so');
        expect(document.activeElement).not.toBe(textbox());
      });
    });

    it('lists one reason at a time, in order, and disables Generate while one shows', async () => {
      await mountAudio('speech');
      expect(reason()).toContain('text');

      await typeInto('ws-prompt', 'Hello there.');
      expect(reason()).toBeNull();
      expect(submitButton().disabled).toBe(false);

      trackApi.readFileLength.mockResolvedValue(2);
      await chooseRecording();
      expect(target.querySelector('#ws-voice-error')?.textContent).toContain('0:02');
      expect(reason()).toContain('30');

      trackApi.readFileLength.mockResolvedValue(12);
      await chooseRecording(new File(['voice2'], 'narrator2.wav', { type: 'audio/wav' }));
      expect(reason()).toContain('says');
      expect(field('ws-voice-text').getAttribute('aria-invalid')).toBe('true');

      await typeInto('ws-voice-text', 'Hello, this is my voice.');
      expect(reason()).toContain('Tick');
      expect(submitButton().disabled).toBe(true);

      await tick();
      expect(reason()).toBeNull();
      expect(submitButton().disabled).toBe(false);
    });

    it('asks for no recording in the default built-in voice, and sends it as the speaker', async () => {
      await mountAudio('speech');
      await typeInto('ws-prompt', 'Hello there.');
      expect(consent()).toBeNull();
      const sent = await submitForm(target);
      expect(Object.fromEntries(sent.entries())).toEqual({ mode: 'audio', workflow: 'speech', model: 'qwen3_tts', prompt: 'Hello there.', speaker: 'ryan', emotion: 'neutral', manner: 'natural', pace: 'normal', pitch: 'normal', expressiveness: 'normal', runs: '1' });
    });

    it('lists the nine default voices with their languages, and shows the voice in the section header', async () => {
      await mountAudio('speech');
      const select = target.querySelector('[data-testid="voice-select"] button') as HTMLElement;

      expect(optionLabels(select)).toEqual([
        'Ryan · English', 'Aiden · English', 'Vivian · Chinese', 'Serena · Chinese', 'Uncle Fu · Chinese', 'Dylan · Beijing Chinese', 'Eric · Sichuan Chinese', 'Ono Anna · Japanese', 'Sohee · Korean',
      ]);
      expect(target.querySelector('[data-section="voice"] .inspector-summary')?.textContent).toBe('Ryan');
    });

    it('sends the chosen built-in voice, marks the voice as changed and puts the default back with its reset', async () => {
      await mountAudio('speech');
      await typeInto('ws-prompt', 'Hello there.');
      const row = () => target.querySelector('[data-section="voice"] .inspector-row') as HTMLElement;
      expect(row().dataset.changed).toBe('false');

      await chooseVoice('vivian');

      expect(draft.state).toMatchObject({ voiceMode: 'builtin', speechSpeaker: 'vivian' });
      expect(row().dataset.changed).toBe('true');
      expect(target.querySelector('[data-section="voice"] .inspector-summary')?.textContent).toBe('Vivian');
      expect((await submitForm(target)).get('speaker')).toBe('vivian');

      (row().querySelector('.inspector-reset') as HTMLButtonElement).click();
      await settle();

      expect(draft.state).toMatchObject({ voiceMode: 'builtin', speechSpeaker: null });
      expect(row().dataset.changed).toBe('false');
    });

    it('keeps the default voice as no choice, so picking Ryan again is not a change', async () => {
      await mountAudio('speech');
      await chooseVoice('aiden');

      await chooseVoice('ryan');

      expect(draft.state.speechSpeaker).toBeNull();
    });

    it('sends no speaker with a recording', async () => {
      await mountAudio('speech');
      await typeInto('ws-prompt', 'Hello there.');
      await chooseVoice('vivian');
      await chooseRecording();
      await typeInto('ws-voice-text', 'Hello, this is my voice.');
      await tick();

      const sent = await submitForm(target);

      expect(sent.has('speaker')).toBe(false);
      expect((sent.get('voice_file') as File).name).toBe('narrator.wav');
    });

    it('says the cloning files download with the first take that follows a recording, while they are not downloaded', async () => {
      await mountAudio('speech');
      expect(target.querySelector('[data-testid="voice-download-hint"]')).toBeNull();

      await chooseRecording();

      expect(target.querySelector('[data-testid="voice-download-hint"]')?.textContent).toBe('Voice cloning downloads its model the first time you generate with a recording or a saved voice.');
    });

    it('shows no download hint once the cloning files are downloaded', async () => {
      const models = AUDIO_MODELS.map((model) => (model.kind === 'speech' ? { ...model, voice_download: { downloaded: true } } : model));
      await mountAudio('speech', { models });

      await chooseRecording();

      expect(target.querySelector('[data-testid="voice-download-hint"]')).toBeNull();
      expect(target.querySelector('#ws-voice-text')).not.toBeNull();
    });

    it('fills the built-in voice of a Reuse of a take, and the default when the take names none', async () => {
      const context = await mountAudio('speech');
      draft.loadFromUrl({ workflow: 'speech', model: 'qwen3_tts', prompt: 'Welcome', speaker: 'sohee' }, context);
      await settle();
      expect(draft.state).toMatchObject({ speechText: 'Welcome', voiceMode: 'builtin', speechSpeaker: 'sohee' });

      draft.loadFromUrl({ workflow: 'speech', model: 'qwen3_tts', prompt: 'Welcome', speaker: 'ryan' }, context);
      await settle();
      expect(draft.state).toMatchObject({ voiceMode: 'builtin', speechSpeaker: null });
    });

    it('fills the built-in voice of the Load settings of a speech job', async () => {
      await mountAudio('speech');

      requestJobSettings({ mode: 'audio', workflow: 'speech', model: 'qwen3_tts', prompt: 'Welcome', speaker: 'eric' });
      await settle();

      expect(draft.state).toMatchObject({ speechText: 'Welcome', voiceMode: 'builtin', speechSpeaker: 'eric' });
    });

    it('sends the recording, its words and the tick as hidden "false" plus "on", the upload winning over a path', async () => {
      await mountAudio('speech');
      await typeInto('ws-prompt', 'Hello there.');
      await chooseRecording();
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

  describe('Delivery', () => {
    const delivery = () => section('delivery') as HTMLElement;
    const summary = () => delivery().querySelector('.inspector-summary')?.textContent ?? null;
    const dot = (label: string) => Array.from(delivery().querySelectorAll('.inspector-row')).find((row) => row.querySelector('.inspector-label-text')?.textContent === label)?.getAttribute('data-changed');
    const tile = (id: string) => delivery().querySelector(`[data-vibe="${id}"]`) as HTMLButtonElement;
    const emotionBox = () => delivery().querySelector('[data-testid="delivery-emotion"] button') as HTMLElement;
    const segment = (row: 'manner' | 'pitch' | 'expressiveness', value: string) => delivery().querySelector(`[data-testid="delivery-${row}"] [data-value="${value}"]`) as HTMLButtonElement;
    const count = () => target.querySelector('[data-testid="text-count"]')?.textContent ?? null;
    const setPace = async (index: number): Promise<void> => {
      const slider = delivery().querySelector('[data-testid="delivery-pace"]') as HTMLInputElement;
      slider.value = String(index);
      slider.dispatchEvent(new Event('input', { bubbles: true }));
      await settle();
    };

    it('puts a Delivery section after Voice with six Vibes and the five rows, on Natural', async () => {
      await mountAudio('speech');

      expect(sectionIds()).toEqual(['voice', 'delivery', 'sampling']);
      expect(summary()).toBe('Natural');
      expect(Array.from(delivery().querySelectorAll('.vibe-name')).map((node) => node.textContent)).toEqual(['Calm narrator', 'Storyteller', 'Excited host', 'Announcer', 'Sad confession', 'Whispered secret']);
      expect(delivery().querySelectorAll('[aria-pressed="true"].vibe')).toHaveLength(0);
      expect(Array.from(delivery().querySelectorAll('.inspector-label-text')).map((node) => node.textContent)).toEqual(['Vibe', 'Emotion', 'Manner', 'Pace', 'Pitch', 'Expressiveness']);
      expect(emotionBox().textContent).toContain('Neutral');
      expect(delivery().querySelector('[data-testid="delivery-pace-word"]')?.textContent).toBe('Normal');
    });

    it('fills the five values from a Vibe, shows it picked and names it in the header', async () => {
      await mountAudio('speech');

      tile('excited-host').click();
      await settle();

      expect(draft.state.speechDelivery).toEqual({ emotion: 'enthusiastic', manner: 'natural', pace: 'fast', pitch: 'high', expressiveness: 'high' });
      expect(summary()).toBe('Excited host');
      expect(tile('excited-host').getAttribute('aria-pressed')).toBe('true');
      expect(emotionBox().textContent).toContain('Enthusiastic');
      expect(delivery().querySelector('[data-testid="delivery-pace-word"]')?.textContent).toBe('Fast');
      expect(segment('pitch', 'high').getAttribute('aria-pressed')).toBe('true');
      expect(dot('Emotion')).toBe('true');
      expect(dot('Manner')).toBe('false');
    });

    it('keeps every value editable after a Vibe: the header says Custom and no tile is picked', async () => {
      await mountAudio('speech');
      tile('calm-narrator').click();
      await settle();

      segment('manner', 'whisper').click();
      await settle();

      expect(draft.state.speechDelivery).toMatchObject({ emotion: 'thoughtful', manner: 'whisper', pace: 'slow' });
      expect(summary()).toBe('Custom');
      expect(delivery().querySelectorAll('.vibe[aria-pressed="true"]')).toHaveLength(0);
      segment('manner', 'natural').click();
      await settle();
      expect(summary()).toBe('Calm narrator');
    });

    it('lists Neutral, then the emotions under their four groups', async () => {
      await mountAudio('speech');
      const options = openOptions(emotionBox());

      expect(options[0].dataset.value).toBe('neutral');
      expect(options).toHaveLength(21);
      expect(Array.from(document.querySelectorAll('.ui-select-group')).map((node) => node.textContent)).toEqual(['Bright', 'Warm', 'Thoughtful', 'Heavy']);
      expect(document.querySelectorAll('.ui-select-column')).toHaveLength(4);

      pickOption(emotionBox(), 'afraid');
      await settle();
      expect(draft.state.speechDelivery.emotion).toBe('afraid');
      expect(summary()).toBe('Custom');
    });

    it('sets Manner, Pace, Pitch and Expressiveness from their rows', async () => {
      await mountAudio('speech');

      segment('manner', 'shout').click();
      await setPace(4);
      segment('pitch', 'low').click();
      segment('expressiveness', 'high').click();
      await settle();

      expect(draft.state.speechDelivery).toEqual({ emotion: 'neutral', manner: 'shout', pace: 'very-fast', pitch: 'low', expressiveness: 'high' });
      expect(delivery().querySelector('[data-testid="delivery-pace-word"]')?.textContent).toBe('Very fast');
      expect(Array.from(delivery().querySelectorAll('.inspector-row')).map((row) => row.getAttribute('data-changed'))).toEqual(['true', 'false', 'true', 'true', 'true', 'true']);
    });

    it('puts one row back with its reset and all five with the section\'s', async () => {
      await mountAudio('speech');
      tile('sad-confession').click();
      await settle();

      (delivery().querySelector('[aria-label="Reset Pace to default"]') as HTMLButtonElement).click();
      await settle();
      expect(draft.state.speechDelivery).toEqual({ emotion: 'sad', manner: 'natural', pace: 'normal', pitch: 'low', expressiveness: 'normal' });

      (delivery().querySelector('[aria-label="Reset Vibe to default"]') as HTMLButtonElement).click();
      await settle();
      expect(draft.state.speechDelivery).toEqual({ emotion: 'neutral', manner: 'natural', pace: 'normal', pitch: 'normal', expressiveness: 'normal' });
      expect(summary()).toBe('Natural');
    });

    it('sends the five values with a built-in voice', async () => {
      await mountAudio('speech');
      await typeInto('ws-prompt', 'Hello there.');
      tile('whispered-secret').click();
      await settle();

      expect(Object.fromEntries((await submitForm(target)).entries())).toMatchObject({ emotion: 'neutral', manner: 'whisper', pace: 'slow', pitch: 'low', expressiveness: 'low' });
    });

    it('holds one line for a voice from a recording and sends no delivery', async () => {
      await mountAudio('speech');
      await typeInto('ws-prompt', 'Hello there.');
      tile('announcer').click();
      await settle();

      await chooseRecording();

      expect(section('delivery')).not.toBeNull();
      expect(delivery().querySelector('[data-testid="delivery-recording-note"]')?.textContent).toBe('Your voices read as they were made. Delivery works with the default voices.');
      expect(delivery().querySelector('[data-testid="delivery-controls"]')).toBeNull();
      expect(summary()).toBeNull();
      draft.setVoice({ path: '/voices/me.wav' });
      draft.update('voiceText', 'Hi, it is me.');
      voiceConsent.ticked = true;
      await settle();
      const sent = Object.fromEntries((await submitForm(target)).entries());
      expect(sent).not.toHaveProperty('emotion');
      expect(sent).not.toHaveProperty('pace');
      // The values are kept for when the built-in voice returns.
      await chooseVoice('ryan');
      expect(delivery().querySelector('[data-testid="delivery-controls"]')).not.toBeNull();
      expect(summary()).toBe('Announcer');
    });

    it('has no Delivery section on Music or Sound', async () => {
      await mountAudio('speech');
      await pickKind('music');
      expect(section('delivery')).toBeNull();
      await pickKind('sound');
      expect(section('delivery')).toBeNull();
    });

    it('tells the reading time under the text at the open Pace, with the marks left out', async () => {
      await mountAudio('speech');
      await typeInto('ws-prompt', 'a'.repeat(1_400));
      expect(count()).toBe('1,400 / 10,000 characters · about 1:40');

      await setPace(1);
      expect(count()).toBe('1,400 / 10,000 characters · about 2:03');

      await typeInto('ws-prompt', `[sad]${'a'.repeat(1_400)}`);
      expect(count()).toBe('1,400 / 10,000 characters · about 2:03');
      await typeInto('ws-prompt', '');
      expect(count()).toBe('0 / 10,000 characters');
    });

    it('reads a recording at the Normal pace', async () => {
      await mountAudio('speech');
      await typeInto('ws-prompt', 'a'.repeat(1_400));
      tile('calm-narrator').click();
      await settle();
      expect(count()).toContain('about 2:03');

      await chooseRecording();
      expect(count()).toContain('about 1:40');
    });
  });

  describe('the speaker\'s consent', () => {
    async function ticked(): Promise<void> {
      await mountAudio('speech');
      await typeInto('ws-prompt', 'Hello there.');
      await chooseRecording();
      await typeInto('ws-voice-text', 'Hello, this is my voice.');
      await tick();
      expect(consent()!.checked).toBe(true);
    }

    it('clears with a new file, which keeps the typed transcript', async () => {
      await ticked();

      await chooseRecording(new File(['other'], 'other.wav', { type: 'audio/wav' }));

      expect(consent()!.checked).toBe(false);
      expect(draft.state.voiceText).toBe('Hello, this is my voice.');
      expect(reason()).toContain('Tick the box');
    });

    it('clears on Reuse of a take, which never carries it', async () => {
      await ticked();
      const take = makeSpeechAsset({ has_reference_voice: true, reuse_workspace_url: '#/workspace?workflow=speech&model=qwen3_tts&prompt=Welcome&voice_path=%2Fvoices%2Fme.wav&voice_text=Hi' });
      galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage([take]));
      draft.loadFromUrl({ workflow: 'speech', model: 'qwen3_tts', prompt: 'Welcome', voice_path: '/voices/me.wav', voice_text: 'Hi' }, withAudio(makeContext()));
      await settle();

      expect(draft.state).toMatchObject({ speechText: 'Welcome', voiceMode: 'recording', voicePath: '/voices/me.wav', voiceText: 'Hi' });
      expect(voiceConsent.ticked).toBe(false);
      expect(consent()!.checked).toBe(false);
    });

    it('clears on Load settings of an audio job, which carries none', async () => {
      await ticked();

      requestJobSettings({ mode: 'audio', workflow: 'speech', model: 'qwen3_tts', prompt: 'Welcome', voice_path: '/voices/me.wav', voice_text: 'Hi', voice_consent: 'on' });
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
      expect(draft.state).toMatchObject({ voiceMode: 'builtin', voicePath: null, voiceUpload: null, voiceText: '', audioKind: 'speech' });
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
    const lengthInput = () => field('ws-length') as unknown as HTMLInputElement;
    const typeButtons = () => Array.from(target.querySelectorAll<HTMLButtonElement>('[data-testid="sound-type"] button'));
    const pickType = async (type: string): Promise<void> => {
      target.querySelector<HTMLButtonElement>(`[data-testid="sound-type"] [data-value="${type}"]`)!.click();
      await settle();
    };
    const summary = () => section('sound')?.querySelector('.inspector-summary')?.textContent;
    const typeRow = () => typeButtons()[0].closest('.inspector-row') as HTMLElement;
    const lengthRow = () => lengthInput().closest('.inspector-row') as HTMLElement;

    it('puts a Sound section before Sampling with the four types, Auto picked, and a blank Length that shows what Auto will use', async () => {
      await mountAudio('sound');

      expect(sectionIds()).toEqual(['sound', 'sampling']);
      expect(typeButtons().map((button) => [button.textContent, button.getAttribute('aria-pressed')])).toEqual([['Auto', 'true'], ['Ambience', 'false'], ['Effect', 'false'], ['One-shot', 'false']]);
      expect(lengthInput().value).toBe('');
      expect(lengthInput().placeholder).toBe('auto · ≈ 12 s');
      expect(target.querySelector('[data-testid="length-max"]')?.textContent).toBe('max 380');
      expect(summary()).toBe('Auto · ≈ 12 s');
      expect(section('sampling')?.querySelector('#ws-length')).toBeNull();
      expect(typeRow().dataset.changed).toBe('false');
      expect(lengthRow().dataset.changed).toBe('false');
    });

    it('shows the length of each type on Auto: 12 s for Ambience, 5 for an Effect, 2 for a One-shot', async () => {
      await mountAudio('sound');
      const shown: string[][] = [];
      for (const type of ['ambience', 'effect', 'one-shot']) {
        await pickType(type);
        shown.push([draft.state.soundType, lengthInput().placeholder, summary() ?? '']);
        expect(lengthInput().value).toBe('');
      }

      expect(shown).toEqual([['ambience', 'auto · ≈ 12 s', 'Ambience · ≈ 12 s'], ['effect', 'auto · ≈ 5 s', 'Effect · ≈ 5 s'], ['one-shot', 'auto · ≈ 2 s', 'One-shot · ≈ 2 s']]);
      expect(draft.state.audioSeconds).toBe(0);
      expect(typeRow().dataset.changed).toBe('true');
    });

    it('sends the type and a blank seconds on Auto, and the batch is the same for every type', async () => {
      await mountAudio('sound');
      await typeInto('ws-prompt', 'a door slam');
      await pickType('one-shot');

      const sent = await submitForm(target);

      expect(Object.fromEntries(sent.entries())).toEqual({ mode: 'audio', workflow: 'sound', model: 'sa3', prompt: 'a door slam', seconds: '', sound_type: 'one-shot', runs: '1' });
      expect(draft.state.runs).toBe(1);
    });

    it('locks the number typed, marks the row changed and sends it, and a blank field goes back to Auto', async () => {
      await mountAudio('sound');
      await pickType('effect');
      await typeInto('ws-prompt', 'a door slam');
      lengthInput().value = '8';
      lengthInput().dispatchEvent(new Event('change', { bubbles: true }));
      await settle();

      expect(draft.state.audioSeconds).toBe(8);
      expect(lengthRow().dataset.changed).toBe('true');
      expect(summary()).toBe('Effect · 8 s');
      expect(Object.fromEntries((await submitForm(target)).entries())).toMatchObject({ seconds: '8', sound_type: 'effect' });

      lengthInput().value = '';
      lengthInput().dispatchEvent(new Event('change', { bubbles: true }));
      await settle();
      expect(draft.state.audioSeconds).toBe(0);
      expect(lengthRow().dataset.changed).toBe('false');
      expect(summary()).toBe('Effect · ≈ 5 s');
    });

    it('takes a length from 1 to the model\'s longest', async () => {
      await mountAudio('sound');
      for (const [typed, kept] of [['900', 380], ['0.2', 1]] as const) {
        lengthInput().value = typed;
        lengthInput().dispatchEvent(new Event('change', { bubbles: true }));
        await settle();
        expect(draft.state.audioSeconds).toBe(kept);
      }
    });

    it('puts the type and the length back to Auto from their row marks', async () => {
      await mountAudio('sound');
      await pickType('ambience');
      draft.patch({ audioSeconds: 30 });
      await settle();

      (typeRow().querySelector('.inspector-reset') as HTMLButtonElement).click();
      await settle();
      expect(draft.state).toMatchObject({ soundType: 'auto', audioSeconds: 30 });
      (lengthRow().querySelector('.inspector-reset') as HTMLButtonElement).click();
      await settle();

      expect(draft.state).toMatchObject({ soundType: 'auto', audioSeconds: 0 });
      expect(lengthInput().placeholder).toBe('auto · ≈ 12 s');
    });

    it('counts a changed type and a locked length in Reset', async () => {
      await mountAudio('sound');
      expect(target.querySelector('[data-testid="changed-count"]')).toBeNull();
      await pickType('effect');
      draft.patch({ audioSeconds: 8 });
      await settle();

      expect(target.querySelector('[data-testid="changed-count"]')?.textContent).toBe('2');
    });

    it('takes a Prompt, and sends a blank length with the type on a fresh form', async () => {
      await mountAudio('sound');
      expect(reason()).not.toBeNull();

      await typeInto('ws-prompt', 'rain on a window');

      const sent = await submitForm(target);
      expect(Object.fromEntries(sent.entries())).toEqual({ mode: 'audio', workflow: 'sound', model: 'sa3', prompt: 'rain on a window', seconds: '', sound_type: 'auto', runs: '1' });
    });

    it('tints the summary once the type or the length is not Auto', async () => {
      await mountAudio('sound');
      const tone = () => section('sound')?.querySelector('.inspector-summary')?.getAttribute('data-tone');
      expect(tone()).toBe('muted');

      draft.patch({ audioSeconds: 8 });
      await settle();
      expect(tone()).toBe('active');
      draft.patch({ audioSeconds: 0, soundType: 'effect' });
      await settle();
      expect(tone()).toBe('active');
    });
  });

  describe('Sound Enhance', () => {
    const RECORD = { settings: { space: 'forest' }, model: 'owner/llm', format: 'text', task: 'sound' };

    /** Mount the Sound tab on a machine with the prompt enhancer. */
    async function mountEnhance(options: { compact?: boolean; tall?: boolean; context?: (context: WorkspaceContext) => WorkspaceContext } = {}): Promise<WorkspaceContext> {
      const base = withAudioEnhancer(withAudio(makeContext({ history_assets: [] })));
      const context = options.context?.(base) ?? base;
      layoutProbe.compact = options.compact ?? false;
      layoutProbe.tall = options.tall ?? false;
      galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage([]));
      draft.reset();
      draft.patch({ workflow: 'sound' });
      app = await mountWorkspacePage(context, target);
      return context;
    }

    /** Make the enhancer answer each run with the next of *texts*, streaming the first half before the final text. */
    function answerWith(...texts: string[]): void {
      const queue = [...texts];
      enhanceApiMocks.enhancePrompt.mockImplementation(async (_body: unknown, ...rest: unknown[]) => {
        const onFrame = rest[0] as (frame: AudioEnhanceFrame) => void;
        const text = queue.shift() ?? 'Another rewrite.';
        onFrame({ type: 'status', phase: 'generating' });
        onFrame({ type: 'text', task: 'sound', text: text.slice(0, 5), fields: { prompt: text.slice(0, 5) } });
        onFrame({ type: 'done', task: 'sound', fields: { prompt: text }, clamped: false, model: 'owner/llm' });
      });
    }

    const lastBody = () => enhanceApiMocks.enhancePrompt.mock.calls.at(-1)?.[0] as Record<string, unknown>;
    const tab = (name: 'prompt' | 'enhanced') => target.querySelector(`#ws-${name === 'prompt' ? 'prompt' : 'enhanced-prompt'}-tab`) as HTMLButtonElement;
    const counter = () => target.querySelector('.rewrite-count')?.textContent ?? null;
    const toggle = () => document.querySelector('#ws-enhance-toggle') as HTMLButtonElement;
    const run = () => document.querySelector('#ws-enhance-run') as HTMLButtonElement;
    const rewrites = (): AudioRewrites => draft.state.audioRewrites.sound;
    const mark = () => tab('enhanced').querySelector('[data-mark]')?.getAttribute('data-mark') ?? null;
    const sentNote = () => descriptionOf(tab('enhanced'));
    const scratchButton = () => target.querySelector('[aria-label="Scratch the enhanced text"]') as HTMLButtonElement | null;
    const stopRing = () => target.querySelector('[aria-label="Stop enhancing"]') as HTMLButtonElement | null;
    const stepButton = (name: string) => target.querySelector(`[aria-label="${name}"]`) as HTMLButtonElement | null;

    /** Open the popover or sheet, run the enhancer with what it holds and wait for the stream; the panel closes as the run starts. */
    async function enhance(): Promise<void> {
      toggle().click();
      await settle();
      run().click();
      await settle();
    }

    it('adds the Prompt and Enhanced tabs and an Enhance tool, on Sound only', async () => {
      await mountEnhance();

      expect(Array.from(target.querySelectorAll('[role="tab"]')).map((node) => node.textContent?.trim())).toEqual(['Prompt', 'Enhanced']);
      expect(toggle()).not.toBeNull();

      // Speech has its own tabs and helper, and no Enhance.
      await pickKind('speech');
      expect(toggle()).toBeNull();
      expect(Array.from(target.querySelectorAll('[role="tab"]')).map((node) => node.textContent?.trim())).toEqual(['Text', 'Written']);
    });

    it('offers Space, Distance, Over time and Movement with a direction, Reset and the Fast and Smart choice, all on Keep to begin with', async () => {
      await mountEnhance({ context: (context) => { context.prompt_enhancer!.smart = { downloaded: true, download_size_label: null, unavailable: null }; return context; } });
      toggle().click();
      await settle();
      const panel = document.querySelector('#ws-enhance-panel')!;

      expect(Array.from(panel.querySelectorAll('.picker')).map((picker) => (picker as HTMLElement).dataset.value)).toEqual(['keep', 'keep', 'keep', 'keep']);
      expect(Array.from(panel.querySelectorAll('.enhance-options h3')).map((node) => node.textContent)).toEqual(['Place', 'Motion']);
      expect((panel.querySelector('#ws-enhance-audio-note') as HTMLInputElement).placeholder).toContain('Direction');
      expect(Array.from(panel.querySelectorAll('[aria-labelledby="ws-enhance-audio-model-label"] [role="radio"]')).map((node) => node.textContent?.trim())).toEqual(['Fast', 'Smart']);
      expect((panel.querySelector('.enhance-reset') as HTMLButtonElement).disabled).toBe(true);
    });

    it('remembers a choice in the popover, resets it, and sends only what is not a default with the type and length', async () => {
      await mountEnhance();
      draft.patch({ soundPrompt: 'rain on a tin roof', soundType: 'effect', audioSeconds: 8 });
      answerWith('Heavy rain on a tin roof.');
      toggle().click();
      await settle();

      flushSync(() => (document.querySelector('#ws-enhance-audio-space') as HTMLButtonElement).click());
      flushSync(() => (document.querySelector('[role="option"][data-slug="forest"]') as HTMLButtonElement).click());
      const note = document.querySelector('#ws-enhance-audio-note') as HTMLInputElement;
      note.value = 'keep the thunder far off';
      note.dispatchEvent(new Event('input', { bubbles: true }));
      await settle();
      expect(draft.state.enhanceSettingsAudio.sound).toMatchObject({ space: 'forest', note: 'keep the thunder far off' });
      expect((document.querySelector('.enhance-reset') as HTMLButtonElement).disabled).toBe(false);

      run().click();
      await settle();

      expect(lastBody()).toEqual({
        mode: 'audio', task: 'sound', model: 'sa3', kind: 'sound', prompt: 'rain on a tin roof',
        settings: { space: 'forest', note: 'keep the thunder far off' }, context: { sound_type: 'effect', seconds: 8 },
      });
      // The panel got out of the way of the rewrite; the choices stay for the next run, until Reset.
      expect(document.querySelector('#ws-enhance-panel')).toBeNull();
      toggle().click();
      await settle();
      (document.querySelector('.enhance-reset') as HTMLButtonElement).click();
      await settle();
      expect(draft.state.enhanceSettingsAudio.sound).toBeNull();
    });

    it('streams the rewrite into the Enhanced tab, which comes forward, and Generate sends it with the text it was made from and its record', async () => {
      await mountEnhance();
      await typeInto('ws-prompt', 'rain on a tin roof');
      answerWith('Heavy rain on a tin roof, steady, close.');

      await enhance();

      expect(tab('enhanced').getAttribute('aria-selected')).toBe('true');
      expect(field('ws-enhanced-prompt').value).toBe('Heavy rain on a tin roof, steady, close.');
      expect(sentNote()).toBe('sent by Generate');
      expect(target.querySelector('[data-testid="prompt-statusline"]')).toBeNull();
      const sent = Object.fromEntries((await submitForm(target)).entries());
      expect(sent).toMatchObject({ prompt: 'Heavy rain on a tin roof, steady, close.', prompt_input: 'rain on a tin roof', model: 'sa3' });
      expect(JSON.parse(sent.enhance_record as string)).toEqual({ settings: {}, model: 'owner/llm', format: 'text', task: 'sound' });
    });

    it('sends only the own text when there is no Enhanced text, and lets an empty own text go when the Enhanced one stands', async () => {
      await mountEnhance();
      await typeInto('ws-prompt', 'rain');
      expect(Object.fromEntries((await submitForm(target)).entries())).not.toHaveProperty('enhance_record');

      draft.patch({ soundPrompt: '', audioRewrites: { ...draft.state.audioRewrites, sound: { text: 'Rain.', from: { input: '', model: 'sa3', task: 'sound' }, history: [] } } });
      await settle();

      expect(reason()).toBeNull();
      expect(Object.fromEntries((await submitForm(target)).entries())).toMatchObject({ prompt: 'Rain.', prompt_input: '' });
    });

    it('keeps five versions, steps through them and sends the one shown', async () => {
      await mountEnhance();
      await typeInto('ws-prompt', 'rain');
      answerWith('One.', 'Two.', 'Three.', 'Four.', 'Five.', 'Six.');
      for (let run = 0; run < 6; run += 1) await enhance();

      expect(rewrites().history.map((entry) => entry.text)).toEqual(['Two.', 'Three.', 'Four.', 'Five.', 'Six.']);
      expect(counter()).toBe('5 of 5');
      expect(target.querySelector('[role="group"][aria-label="Enhanced versions"]')).not.toBeNull();

      stepButton('Previous version')!.click();
      stepButton('Previous version')!.click();
      await settle();

      expect(counter()).toBe('3 of 5');
      expect(field('ws-enhanced-prompt').value).toBe('Four.');
      expect((await submitForm(target)).get('prompt')).toBe('Four.');
    });

    it('marks the Enhanced text out of date on the first keystroke, from either view, and the fix is on the Enhance button', async () => {
      await mountEnhance();
      await typeInto('ws-prompt', 'rain');
      answerWith('Heavy rain.', 'Heavy rain, closer.');
      await enhance();
      tab('prompt').click();
      await settle();
      expect(mark()).toBe('dot');
      expect(sentNote()).toBe('sent by Generate');

      await typeInto('ws-prompt', 'rain, closer');

      expect(mark()).toBe('warn');
      expect(sentNote()).toBe('sent by Generate, out of date');
      expect(toggle().getAttribute('aria-label')).toBe('Enhance (out of date)');
      expect(target.querySelector('[data-testid="prompt-statusline"]')).toBeNull();
      // From the Enhanced view the warning is on the button, and the ✕ and the stepper stay.
      tab('enhanced').click();
      await settle();
      expect(mark()).toBeNull();
      expect(scratchButton()).not.toBeNull();

      toggle().click();
      await settle();
      expect(document.activeElement).toBe(run());
      expect(run().textContent?.trim()).toBe('Enhance again');
      run().click();
      await settle();

      expect(lastBody()).toMatchObject({ prompt: 'rain, closer' });
      expect(field('ws-enhanced-prompt').value).toBe('Heavy rain, closer.');
      expect(toggle().getAttribute('aria-label')).toBeNull();
      // Enhance again kept the version made from the earlier text.
      expect(rewrites().history.map((entry) => entry.text)).toEqual(['Heavy rain.', 'Heavy rain, closer.']);
      expect(counter()).toBe('2 of 2');
    });

    it('still sends an out-of-date rewrite, with the text it was made from', async () => {
      await mountEnhance();
      await typeInto('ws-prompt', 'rain');
      answerWith('Heavy rain.');
      await enhance();
      await typeInto('ws-prompt', 'rain, closer');

      const sent = await submitForm(target);

      expect(sent.get('prompt')).toBe('Heavy rain.');
      expect(sent.get('prompt_input')).toBe('rain');
    });

    it('marks it out of date when the model changes, keeping the stepper', async () => {
      await mountEnhance();
      await typeInto('ws-prompt', 'rain');
      answerWith('Heavy rain.', 'Heavy rain again.');
      await enhance();
      await enhance();
      tab('prompt').click();
      await settle();
      expect(mark()).toBe('dot');

      draft.patch({ audioRewrites: { ...draft.state.audioRewrites, sound: { ...rewrites(), from: { ...rewrites().from!, model: 'other-model' } } } });
      await settle();

      expect(mark()).toBe('warn');
      tab('enhanced').click();
      await settle();
      expect(counter()).toBe('2 of 2');
    });

    it('marks the closed Enhanced tab with a dot while Generate sends it, and greys your text', async () => {
      await mountEnhance();
      await typeInto('ws-prompt', 'rain');
      expect(mark()).toBeNull();
      expect(field('ws-prompt').hasAttribute('data-unsent')).toBe(false);
      answerWith('Heavy rain.');
      await enhance();
      expect(mark()).toBeNull();
      expect(scratchButton()).not.toBeNull();

      tab('prompt').click();
      await settle();

      expect(mark()).toBe('dot');
      expect(scratchButton()).toBeNull();
      expect(field('ws-prompt').hasAttribute('data-unsent')).toBe(true);
      expect(tab('prompt').querySelector('[data-mark]')).toBeNull();
    });

    it('scratches the Enhanced text: your view with the focus on your tab, and Generate sends your text', async () => {
      await mountEnhance();
      await typeInto('ws-prompt', 'rain');
      answerWith('Heavy rain.');
      await enhance();

      scratchButton()!.click();
      await settle();

      expect(rewrites()).toMatchObject({ text: '', from: null });
      expect(tab('prompt').getAttribute('aria-selected')).toBe('true');
      expect(document.activeElement).toBe(tab('prompt'));
      expect(mark()).toBeNull();
      expect(field('ws-prompt').hasAttribute('data-unsent')).toBe(false);
      expect(Object.fromEntries((await submitForm(target)).entries())).toMatchObject({ prompt: 'rain' });
      expect(Object.fromEntries((await submitForm(target)).entries())).not.toHaveProperty('enhance_record');
    });

    it('says "– of N" in the empty Enhanced view after Scratch, and ‹ brings back the version that was scratched', async () => {
      await mountEnhance();
      await typeInto('ws-prompt', 'rain');
      answerWith('One.', 'Two.', 'Three.');
      await enhance();
      await enhance();
      await enhance();
      stepButton('Previous version')!.click();
      await settle();
      expect(field('ws-enhanced-prompt').value).toBe('Two.');

      scratchButton()!.click();
      await settle();
      tab('enhanced').click();
      await settle();

      expect(counter()).toBe('– of 3');
      expect(field('ws-enhanced-prompt').value).toBe('');
      expect(field('ws-enhanced-prompt').placeholder).toBe('No enhanced text. Generate sends your Prompt.');
      expect(stepButton('Bring back version 2')).not.toBeNull();
      expect(stepButton('Next version')!.disabled).toBe(true);

      stepButton('Bring back version 2')!.click();
      await settle();

      expect(field('ws-enhanced-prompt').value).toBe('Two.');
      expect(counter()).toBe('2 of 3');
      expect((await submitForm(target)).get('prompt')).toBe('Two.');
    });

    it('says "– of 1" after Scratch with one version, and the stepper is gone once the box is typed in', async () => {
      await mountEnhance();
      await typeInto('ws-prompt', 'rain');
      answerWith('One.');
      await enhance();
      scratchButton()!.click();
      await settle();
      tab('enhanced').click();
      await settle();

      expect(counter()).toBe('– of 1');
      expect(stepButton('Bring back version 1')).not.toBeNull();

      await typeInto('ws-enhanced-prompt', 'by hand');

      expect(counter()).toBeNull();
      expect(rewrites()).toMatchObject({ text: 'by hand', from: null });
    });

    it('keeps the earlier versions when Enhance again runs on changed text, and a typed version replaces its own entry and drops its record', async () => {
      await mountEnhance();
      await typeInto('ws-prompt', 'rain');
      answerWith('Heavy rain.', 'Heavy rain, closer.');
      await enhance();
      await typeInto('ws-prompt', 'rain, closer');
      await enhance();
      expect(rewrites().history.map((entry) => entry.text)).toEqual(['Heavy rain.', 'Heavy rain, closer.']);
      stepButton('Previous version')!.click();
      await settle();

      await typeInto('ws-enhanced-prompt', 'Heavy rain, my way.');

      expect(rewrites().history.map((entry) => entry.text)).toEqual(['Heavy rain, my way.', 'Heavy rain, closer.']);
      expect(rewrites().history[0].from).not.toHaveProperty('record');
      expect(rewrites().history[1].from).toHaveProperty('record');
      expect(counter()).toBe('1 of 2');
      const sent = await submitForm(target);
      expect(sent.get('prompt')).toBe('Heavy rain, my way.');
      expect(sent.has('enhance_record')).toBe(false);
    });

    it('hides the stepper for Enhanced text typed by hand that is not a version, and while Enhance runs', async () => {
      await mountEnhance();
      await typeInto('ws-prompt', 'rain');
      answerWith('One.', 'Two.');
      await enhance();
      await enhance();
      expect(counter()).toBe('2 of 2');

      await typeInto('ws-enhanced-prompt', 'Two.');
      expect(counter()).toBe('2 of 2');
      draft.patch({ audioRewrites: { ...draft.state.audioRewrites, sound: { ...rewrites(), text: 'Typed over.', from: null } } });
      await settle();
      expect(counter()).toBeNull();

      let release: () => void = () => {};
      enhanceApiMocks.enhancePrompt.mockImplementation(async (_body: unknown, ...rest: unknown[]) => {
        (rest[0] as (frame: AudioEnhanceFrame) => void)({ type: 'text', task: 'sound', text: 'Hea', fields: { prompt: 'Hea' } });
        await new Promise<void>((resolve) => { release = resolve; });
      });
      draft.patch({ audioRewrites: { ...draft.state.audioRewrites, sound: { ...rewrites(), text: 'Two.', from: rewrites().history[1].from } } });
      await settle();
      expect(counter()).toBe('2 of 2');
      toggle().click();
      await settle();
      run().click();
      await settle();
      expect(counter()).toBeNull();
      release();
      await settle();
    });

    it('shows the stop ring after the tablist while Enhance runs on the open Enhanced tab, and stops it', async () => {
      await mountEnhance();
      await typeInto('ws-prompt', 'rain');
      let aborted = false;
      let release: () => void = () => {};
      enhanceApiMocks.enhancePrompt.mockImplementation(async (_body: unknown, ...rest: unknown[]) => {
        const onFrame = rest[0] as (frame: AudioEnhanceFrame) => void;
        const signal = rest[1] as AbortSignal;
        onFrame({ type: 'text', task: 'sound', text: 'Hea', fields: { prompt: 'Hea' } });
        signal.addEventListener('abort', () => { aborted = true; release(); });
        await new Promise<void>((resolve) => { release = resolve; });
      });
      toggle().click();
      await settle();
      run().click();
      await settle();

      const ring = stopRing()!;
      expect(ring).not.toBeNull();
      expect(ring.closest('[role="tablist"]')).toBeNull();
      expect(ring.previousElementSibling?.getAttribute('role') === 'tablist' || ring.parentElement?.contains(tab('enhanced'))).toBe(true);
      expect(scratchButton()).toBeNull();
      expect(counter()).toBeNull();
      expect(sentNote()).toBe('enhancing');
      ring.focus();

      ring.click();
      await settle();

      expect(aborted).toBe(true);
      expect(stopRing()).toBeNull();
      // The ring went while it had the focus: the focus moves to the Enhanced tab.
      expect(document.activeElement).toBe(tab('enhanced'));
    });

    it('shows a quiet spinner in the Enhanced tab, and no ring, while Enhance runs and your own tab is open', async () => {
      await mountEnhance();
      await typeInto('ws-prompt', 'rain');
      let release: () => void = () => {};
      enhanceApiMocks.enhancePrompt.mockImplementation(async () => { await new Promise<void>((resolve) => { release = resolve; }); });
      toggle().click();
      await settle();
      run().click();
      await settle();
      tab('prompt').click();
      await settle();

      expect(mark()).toBe('spinner');
      expect(tab('enhanced').querySelector('button, a')).toBeNull();
      expect(stopRing()).toBeNull();
      expect(sentNote()).toBe('enhancing');

      release();
      await settle();
      expect(mark()).toBeNull();
    });

    it('has no ring for the Script helper: its panel has its own Stop', async () => {
      await mountEnhance();
      await pickKind('speech');
      await typeInto('ws-prompt', 'Hello there.');

      expect(stopRing()).toBeNull();
    });

    it('reads the Enhanced text in the folded dock on a phone, and the own text again after Scratch', async () => {
      await mountEnhance({ compact: true });
      await typeInto('ws-prompt', 'rain');
      answerWith('Heavy rain.');
      await enhance();

      const dock = () => target.querySelector('.composer-summary')?.textContent ?? '';
      expect(draft.state.audioRewrites.sound.text).toBe('Heavy rain.');
      expect(dock()).toContain('Enhanced');
      expect(dock()).toContain('Heavy rain.');

      scratchButton()!.click();
      await settle();

      expect(dock()).toContain('Prompt');
      expect(dock()).toContain('rain');
      expect(dock()).not.toContain('Enhanced');
    });

    it('is unavailable while a take is being made, and says in the panel when it comes back', async () => {
      const running = jobSnap('job-live', 'running', { workflow: 'sound' });
      workspaceApiMocks.listJobs.mockResolvedValue({ active_job: running, queued_jobs: [] });
      await mountEnhance({ context: (context) => ({ ...context, active_job: running, queued_jobs: [] }) });
      await typeInto('ws-prompt', 'rain');

      expect(target.querySelector('.tools-role')).toBeNull();
      toggle().click();
      await settle();
      expect(run().disabled).toBe(true);
      expect(document.querySelector('#ws-enhance-panel')?.textContent).toContain('Available when the take is done.');
      pressKey('e', { ctrlKey: true }, field('ws-prompt'));
      await settle();
      expect(enhanceApiMocks.enhancePrompt).not.toHaveBeenCalled();
    });

    it('runs with Ctrl+E from the box, and needs text first', async () => {
      await mountEnhance();
      answerWith('Heavy rain.');
      pressKey('e', { ctrlKey: true }, field('ws-prompt'));
      await settle();
      expect(enhanceApiMocks.enhancePrompt).not.toHaveBeenCalled();

      await typeInto('ws-prompt', 'rain');
      pressKey('e', { ctrlKey: true }, field('ws-prompt'));
      await settle();

      expect(lastBody()).toMatchObject({ mode: 'audio', task: 'sound', prompt: 'rain' });
      expect(field('ws-enhanced-prompt').value).toBe('Heavy rain.');
    });

    it('shows an enhancer error under the box and keeps the text', async () => {
      await mountEnhance();
      await typeInto('ws-prompt', 'rain');
      enhanceApiMocks.enhancePrompt.mockImplementation(async (_body: unknown, ...rest: unknown[]) => {
        (rest[0] as (frame: AudioEnhanceFrame) => void)({ type: 'error', detail: 'The enhancer returned nothing usable.' });
      });

      await enhance();

      expect(target.querySelector('[role="alert"]')?.textContent).toBe('The enhancer returned nothing usable.');
      expect(rewrites().text).toBe('');
    });

    it.each([['a phone', 'compact'], ['a tablet', 'tall']] as const)('opens as a sheet on %s that shows the box text at the top', async (_name, layoutName) => {
      await mountEnhance({ compact: layoutName === 'compact', tall: layoutName === 'tall' });
      await typeInto('ws-prompt', 'rain on a tin roof');

      toggle().click();
      await settle();

expect(document.querySelector('#ws-enhance-panel')?.closest('[role="dialog"]')?.textContent).toContain('Enhance prompt');
      expect(document.querySelector('[data-testid="enhance-quote"]')?.textContent).toBe('rain on a tin roof');
      expect(run()).not.toBeNull();
    });

    it('brings a recent prompt back without the Enhanced text of the one it replaces', async () => {
      await mountEnhance();
      await typeInto('ws-prompt', 'rain');
      answerWith('Heavy rain.');
      await enhance();
      recentPromptsApiMocks.getRecentPrompts.mockResolvedValue({ entries: [{ format: 'text', prompt: 'a door slam', scene: '', negative_prompt: '', mode: 'sound', used_at: 1 }] });

      (target.querySelector('#ws-recent-prompts') as HTMLButtonElement).click();
      await settle();
      (document.querySelector('[role="option"]') as HTMLElement).click();
      await settle();

      expect(draft.state.soundPrompt).toBe('a door slam');
      expect(rewrites().text).toBe('');
      // The versions belonged to the text that was replaced, so there is nothing to step through or bring back.
      expect(rewrites().history).toEqual([]);
      expect(counter()).toBeNull();
    });

    it('shows an error or the last result\'s note as one help line under the box, and clears it on the next edit or run', async () => {
      await mountEnhance();
      await typeInto('ws-prompt', 'rain');
      enhanceApiMocks.enhancePrompt.mockImplementation(async (_body: unknown, ...rest: unknown[]) => {
        (rest[0] as (frame: AudioEnhanceFrame) => void)({ type: 'error', detail: 'The enhancer returned nothing usable.' });
      });
      await enhance();
      expect(target.querySelectorAll('[data-testid="help-line"]')).toHaveLength(1);
      expect(target.querySelector('[role="alert"]')?.textContent).toBe('The enhancer returned nothing usable.');

      await typeInto('ws-prompt', 'rain, closer');
      expect(target.querySelector('[role="alert"]')).toBeNull();

      enhanceApiMocks.enhancePrompt.mockImplementation(async (_body: unknown, ...rest: unknown[]) => {
        const onFrame = rest[0] as (frame: AudioEnhanceFrame) => void;
        onFrame({ type: 'done', task: 'sound', fields: { prompt: 'Heavy rain.' }, clamped: false, model: 'owner/llm', warning: 'Already at maximum length.' });
      });
      await enhance();
      expect(target.querySelectorAll('[data-testid="help-line"]')).toHaveLength(1);
      expect(target.querySelector('[data-testid="help-line"]')?.textContent).toBe('Already at maximum length.');
      expect(target.querySelector('[data-testid="help-line"]')?.getAttribute('role')).toBe('status');

      scratchButton()!.click();
      await settle();
      expect(target.querySelector('[data-testid="help-line"]')).toBeNull();
    });
  });

  describe('Music Style Enhance', () => {
    const MODELS = [...AUDIO_MODELS, SA3_MUSIC];

    /** Mount the Music tab on a machine with the prompt enhancer, on YuE2 or (with *model*) on Stable Audio. */
    async function mountStyle(options: { model?: string; compact?: boolean; tall?: boolean } = {}): Promise<void> {
      const context = withAudioEnhancer(withAudio(makeContext({ history_assets: [] }), MODELS));
      layoutProbe.compact = options.compact ?? false;
      layoutProbe.tall = options.tall ?? false;
      galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage([]));
      draft.reset();
      draft.patch({ workflow: 'music' });
      app = await mountWorkspacePage(context, target);
      if (options.model) {
        draft.hydrateFromContext(context, options.model);
        await settle();
      }
    }

    /** Make the enhancer answer each run with the next of *texts*, as a style. */
    function answerWith(...texts: string[]): void {
      const queue = [...texts];
      enhanceApiMocks.enhancePrompt.mockImplementation(async (_body: unknown, ...rest: unknown[]) => {
        const onFrame = rest[0] as (frame: AudioEnhanceFrame) => void;
        const text = queue.shift() ?? 'Another style.';
        onFrame({ type: 'text', task: 'style', text: text.slice(0, 5), fields: { style: text.slice(0, 5) } });
        onFrame({ type: 'done', task: 'style', fields: { style: text }, clamped: false, model: 'owner/llm' });
      });
    }

    const lastBody = () => enhanceApiMocks.enhancePrompt.mock.calls.at(-1)?.[0] as Record<string, unknown>;
    const toggle = () => document.querySelector('#ws-enhance-toggle') as HTMLButtonElement;
    const run = () => document.querySelector('#ws-enhance-run') as HTMLButtonElement;
    const tab = (name: 'style' | 'enhanced') => target.querySelector(`#${name === 'style' ? 'ws-audio-style' : 'ws-enhanced-audio-style'}-tab`) as HTMLButtonElement;
    const styleMark = () => tab('enhanced').querySelector('[data-mark]')?.getAttribute('data-mark') ?? null;
    const styleSent = () => descriptionOf(tab('enhanced'));
    const scratchButton = () => target.querySelector('[aria-label="Scratch the enhanced text"]') as HTMLButtonElement | null;
    const axisLabels = () => Array.from(document.querySelectorAll('#ws-enhance-panel .enhance-options h3, #ws-enhance-panel .enhance-subtitle .ui-area-label')).map((node) => node.textContent?.trim());
    const pickerIds = () => Array.from(document.querySelectorAll('#ws-enhance-panel .picker')).map((picker) => picker.id);

    async function enhance(): Promise<void> {
      toggle().click();
      await settle();
      run().click();
      await settle();
    }

    it('adds the Style and Enhanced tabs and an Enhance tool that says what it does, on YuE2 and on Stable Audio', async () => {
      await mountStyle();
      expect(Array.from(target.querySelectorAll('[role="tab"]')).map((node) => node.textContent?.trim())).toEqual(['Style', 'Enhanced']);
      expect(target.querySelector('.tools-role')).toBeNull();
      expect(toggle()).not.toBeNull();
      expect(target.querySelector('#ws-lyrics')).not.toBeNull();

      draft.hydrateFromContext(withAudioEnhancer(withAudio(makeContext({ history_assets: [] }), MODELS)), 'sa3');
      await settle();

      expect(target.querySelector('#ws-lyrics')).toBeNull();
      expect(toggle()).not.toBeNull();
    });

    it('offers Genre in groups, Mood, Instruments, Without, Singer and Production for a song, and no Place or Motion', async () => {
      await mountStyle();
      await typeInto('ws-lyrics', '[verse]\nla la');
      toggle().click();
      await settle();

      expect(pickerIds()).toEqual(['ws-enhance-audio-genre', 'ws-enhance-audio-music_mood', 'ws-enhance-audio-production', 'ws-enhance-audio-singer']);
      expect(axisLabels()).toEqual(['Sound', 'Instruments · pick any', 'Without · pick any', 'Voice']);
      flushSync(() => (document.querySelector('#ws-enhance-audio-genre') as HTMLButtonElement).click());
      expect(Array.from(document.querySelectorAll('[role="listbox"] .picker-group')).map((node) => node.textContent)).toContain('Pop & rock');
    });

    it.each([
      ['YuE2 with empty lyrics', undefined, ''],
      ['YuE2 with only section tags', undefined, '[intro]\n[outro]'],
      ['Stable Audio', 'sa3', ''],
    ])('hides Singer for %s, and still offers the rest', async (_name, model, lyrics) => {
      await mountStyle({ model });
      if (lyrics) await typeInto('ws-lyrics', lyrics);
      toggle().click();
      await settle();

      expect(pickerIds()).toEqual(['ws-enhance-audio-genre', 'ws-enhance-audio-music_mood', 'ws-enhance-audio-production']);
      expect(axisLabels()).not.toContain('Voice');
    });

    it('shows Singer once the lyrics hold words', async () => {
      await mountStyle();
      toggle().click();
      await settle();
      expect(pickerIds()).not.toContain('ws-enhance-audio-singer');
      document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
      await settle();

      await typeInto('ws-lyrics', 'la la');
      toggle().click();
      await settle();

      expect(pickerIds()).toContain('ws-enhance-audio-singer');
    });

    it('keeps an instrument out of Without once it is asked for, and the other way', async () => {
      await mountStyle();
      toggle().click();
      await settle();
      const chip = (group: string, name: string) => Array.from(document.querySelectorAll(`#ws-enhance-audio-${group}-label ~ .chip-list button`)).find((button) => button.textContent?.trim() === name) as HTMLButtonElement;

      chip('without', 'Drums').click();
      await settle();
      expect(draft.state.enhanceSettingsAudio.style).toMatchObject({ without: ['drums'] });
      expect(chip('without', 'Drums').getAttribute('aria-pressed')).toBe('true');

      chip('instruments', 'Drums').click();
      await settle();
      expect(draft.state.enhanceSettingsAudio.style).toMatchObject({ instruments: ['drums'], without: [] });
      expect(chip('without', 'Drums').disabled).toBe(true);
      expect(chip('without', 'Piano').disabled).toBe(false);
    });

    it('sends the style with the model, the options, the lyrics start, the instrumental flag, the tempo and the key, and never the length', async () => {
      await mountStyle();
      draft.patch({ audioStyle: 'warm folk', lyrics: '[verse]\n' + 'la '.repeat(400), audioTempo: 96, audioKey: 'Am', musicSeconds: 120, enhanceSettingsAudio: { sound: null, style: { genre: 'folk', singer: 'female', instruments: ['piano'] } } });
      await settle();
      answerWith('Folk, warm, piano, female vocal');

      await enhance();

      expect(lastBody()).toEqual({
        mode: 'audio', task: 'style', model: 'yue2', kind: 'music', prompt: 'warm folk',
        settings: { genre: 'folk', instruments: ['piano'], singer: 'female' },
        context: { instrumental: false, lyrics: ('[verse]\n' + 'la '.repeat(400)).slice(0, 600), tempo: 96, key: 'Am' },
      });
    });

    it('sends an instrumental style without the Singer it chose and without lyrics', async () => {
      await mountStyle({ model: 'sa3' });
      draft.patch({ audioStyle: 'warm folk', lyrics: 'kept from YuE2', enhanceSettingsAudio: { sound: null, style: { genre: 'folk', singer: 'female' } } });
      await settle();
      answerWith('A warm folk tune on acoustic guitar.');

      await enhance();

      const body = lastBody();
      expect(body).toMatchObject({ task: 'style', model: 'sa3', kind: 'music', settings: { genre: 'folk' }, context: { instrumental: true } });
      expect(body.settings).not.toHaveProperty('singer');
      expect(body.context).not.toHaveProperty('lyrics');
      expect(body.context).not.toHaveProperty('seconds');
    });

    it('writes into the Enhanced tab, never touches Lyrics, and Generate sends it as the prompt with the style it was made from', async () => {
      await mountStyle();
      await typeInto('ws-audio-style', 'warm folk');
      await typeInto('ws-lyrics', '[verse]\nla la');
      answerWith('Folk, warm, acoustic guitar, intimate');

      await enhance();

      expect(tab('enhanced').getAttribute('aria-selected')).toBe('true');
      expect(field('ws-enhanced-audio-style').value).toBe('Folk, warm, acoustic guitar, intimate');
      expect(draft.state.lyrics).toBe('[verse]\nla la');
      expect(styleSent()).toBe('sent by Generate');
      const sent = Object.fromEntries((await submitForm(target)).entries());
      expect(sent).toMatchObject({ prompt: 'Folk, warm, acoustic guitar, intimate', prompt_input: 'warm folk', lyrics: '[verse]\nla la', model: 'yue2' });
      expect(JSON.parse(sent.enhance_record as string)).toEqual({ settings: {}, model: 'owner/llm', format: 'text', task: 'style' });
    });

    it('marks the Style rewrite out of date when the Music model is switched, and Enhance again writes it for the new model and keeps the first', async () => {
      await mountStyle();
      await typeInto('ws-audio-style', 'warm folk');
      answerWith('Folk, warm, acoustic guitar', 'A warm folk tune on acoustic guitar.');
      await enhance();
      tab('style').click();
      await settle();
      expect(styleMark()).toBe('dot');

      pickOption(target.querySelector('#ws-model') as HTMLElement, 'sa3');
      await settle();

      expect(draft.state.model).toBe('sa3');
      expect(styleMark()).toBe('warn');
      expect(styleSent()).toBe('sent by Generate, out of date');
      expect(toggle().getAttribute('aria-label')).toBe('Enhance (out of date)');
      toggle().click();
      await settle();
      run().click();
      await settle();

      expect(lastBody()).toMatchObject({ task: 'style', model: 'sa3', prompt: 'warm folk' });
      expect(styleMark()).toBeNull();
      expect(field('ws-enhanced-audio-style').value).toBe('A warm folk tune on acoustic guitar.');
      expect(draft.state.audioRewrites.style.history.map((entry) => entry.text)).toEqual(['Folk, warm, acoustic guitar', 'A warm folk tune on acoustic guitar.']);
    });

    it("keeps Sound's rewrite apart from Music's", async () => {
      await mountStyle();
      await typeInto('ws-audio-style', 'warm folk');
      answerWith('Folk, warm');
      await enhance();

      await pickKind('sound');

      expect(target.querySelector('[data-kind="sound"]')).not.toBeNull();
      expect(tab('enhanced')).toBeNull();
      expect(Array.from(target.querySelectorAll('[role="tab"]')).map((node) => node.textContent?.trim())).toEqual(['Prompt', 'Enhanced']);
      expect(target.querySelector('[role="tab"][aria-selected="true"]')?.textContent?.trim()).toBe('Prompt');

      await pickKind('music');

      expect(draft.state.audioRewrites.style.text).toBe('Folk, warm');
      expect(target.querySelector('[role="tab"][aria-selected="true"]')?.textContent?.trim()).toBe('Style');
    });

    it('reads the Enhanced style in the folded dock on a phone, and the own style again after Scratch', async () => {
      await mountStyle({ compact: true });
      await typeInto('ws-audio-style', 'warm folk');
      answerWith('Folk, warm, acoustic guitar');
      await enhance();

      const dock = () => target.querySelector('.composer-summary')?.textContent ?? '';
      expect(dock()).toContain('Enhanced');
      expect(dock()).toContain('Folk, warm, acoustic guitar');

      scratchButton()!.click();
      await settle();

      expect(dock()).toContain('Style');
      expect(dock()).toContain('warm folk');
      expect(dock()).not.toContain('Enhanced');
    });

    it.each([['a phone', 'compact'], ['a tablet', 'tall']] as const)('opens as a sheet on %s that shows the style at the top', async (_name, layoutName) => {
      await mountStyle({ compact: layoutName === 'compact', tall: layoutName === 'tall' });
      await typeInto('ws-audio-style', 'warm folk');

      toggle().click();
      await settle();

      expect(document.querySelector('#ws-enhance-panel')?.closest('[role="dialog"]')?.textContent).toContain('Enhance style');
      expect(document.querySelector('[data-testid="enhance-quote"]')?.textContent).toBe('warm folk');
      expect(run().textContent?.trim()).toBe('Enhance style');
    });

    it('is unavailable while a take is being made', async () => {
      const running = jobSnap('job-live', 'running', { workflow: 'music' });
      workspaceApiMocks.listJobs.mockResolvedValue({ active_job: running, queued_jobs: [] });
      const context = withAudioEnhancer(withAudio(makeContext({ history_assets: [] }), MODELS));
      galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage([]));
      draft.reset();
      draft.patch({ workflow: 'music' });
      app = await mountWorkspacePage({ ...context, active_job: running, queued_jobs: [] }, target);
      await typeInto('ws-audio-style', 'warm folk');

      expect(target.querySelector('.tools-role')).toBeNull();
      toggle().click();
      await settle();
      expect(document.querySelector('#ws-enhance-panel')?.textContent).toContain('Available when the take is done.');
      expect(run().disabled).toBe(true);
    });

    it('still sends the Enhanced style written for the other model when Generate runs out of date, with the style it was made from', async () => {
      await mountStyle();
      await typeInto('ws-audio-style', 'warm folk');
      answerWith('Folk, warm, female lead vocal, acoustic guitar');
      await enhance();

      pickOption(target.querySelector('#ws-model') as HTMLElement, 'sa3');
      await settle();
      tab('style').click();
      await settle();
      expect(styleMark()).toBe('warn');

      const sent = Object.fromEntries((await submitForm(target)).entries());
      expect(sent).toMatchObject({ model: 'sa3', prompt: 'Folk, warm, female lead vocal, acoustic guitar', prompt_input: 'warm folk' });
    });

    it('clears the enhancer error of the box it leaves when the kind changes', async () => {
      await mountStyle();
      await typeInto('ws-audio-style', 'warm folk');
      enhanceApiMocks.enhancePrompt.mockImplementation(async (_body: unknown, ...rest: unknown[]) => {
        (rest[0] as (frame: AudioEnhanceFrame) => void)({ type: 'error', detail: 'The enhancer returned nothing usable.' });
      });
      await enhance();
      expect(target.querySelector('[role="alert"]')?.textContent).toBe('The enhancer returned nothing usable.');

      await pickKind('sound');

      expect(target.querySelector('[role="alert"]')).toBeNull();
      await pickKind('music');
      expect(target.querySelector('[role="alert"]')).toBeNull();
    });

    it('brings a recent song back with its lyrics and without the Enhanced style it replaces', async () => {
      await mountStyle();
      await typeInto('ws-audio-style', 'warm folk');
      answerWith('Folk, warm');
      await enhance();
      recentPromptsApiMocks.getRecentPrompts.mockResolvedValue({ entries: [{ format: 'text', prompt: 'dreamy synth-pop', scene: '[verse]\nla', negative_prompt: '', mode: 'music', used_at: 1 }] });

      (target.querySelector('#ws-recent-prompts') as HTMLButtonElement).click();
      await settle();
      (document.querySelector('[role="option"]') as HTMLElement).click();
      await settle();

      expect(draft.state.audioStyle).toBe('dreamy synth-pop');
      expect(draft.state.lyrics).toBe('[verse]\nla');
      expect(draft.state.audioRewrites.style.text).toBe('');
    });
  });

  describe('Reuse, Retry and Load settings of an Enhanced sound', () => {
    const ENHANCE = JSON.stringify({ format: 'text', model: 'owner/llm', settings: { space: 'forest' }, task: 'sound', tier: 'fast' });

    it('puts the Enhanced text, its options and its enhancer back, not out of date, and the own text in the Prompt tab', async () => {
      await mountAudio('sound');
      draft.patch({ audioRewrites: { ...draft.state.audioRewrites, sound: { text: 'older', from: null, history: [] } } });

      draft.loadFromUrl({ workflow: 'sound', model: 'sa3', prompt: 'rain', enhanced_prompt: 'Heavy rain.', enhance: ENHANCE }, withAudioEnhancer(withAudio(makeContext())), { replaceSlots: true, exact: true });
      await settle();

      expect(draft.state).toMatchObject({ soundPrompt: 'rain', enhanceModel: 'fast' });
      expect(draft.state.enhanceSettingsAudio.sound).toEqual({ space: 'forest' });
      expect(draft.state.audioRewrites.sound).toMatchObject({ text: 'Heavy rain.', from: { input: 'rain', model: 'sa3', task: 'sound', record: { model: 'owner/llm', tier: 'fast' } } });
      expect(field('ws-prompt').value).toBe('rain');
      (target.querySelector('#ws-enhanced-prompt-tab') as HTMLButtonElement).click();
      await settle();
      expect(field('ws-enhanced-prompt').value).toBe('Heavy rain.');
      expect(descriptionOf(target.querySelector('#ws-enhanced-prompt-tab')!)).toBe('sent by Generate');
    });

    it('empties the Enhanced text when the take had none', async () => {
      await mountAudio('sound');
      draft.patch({ audioRewrites: { ...draft.state.audioRewrites, sound: { text: 'older', from: null, history: [] } } });

      draft.loadFromUrl({ workflow: 'sound', model: 'sa3', prompt: 'a door slam' }, withAudio(makeContext()), { replaceSlots: true, exact: true });

      expect(draft.state.audioRewrites.sound).toMatchObject({ text: '', from: null });
    });
  });

  describe('Reuse, Retry and Load settings of an Enhanced song', () => {
    const MODELS = [...AUDIO_MODELS, SA3_MUSIC];
    const ENHANCE = JSON.stringify({ format: 'text', model: 'owner/llm', settings: { genre: 'folk', singer: 'female', instruments: ['guitar'], without: ['drums'], note: 'fingerpicked' }, task: 'style', tier: 'smart' });
    const REUSE = { workflow: 'music', model: 'yue2', prompt: 'warm folk', enhanced_prompt: 'Folk, female lead vocal, acoustic guitar', enhance: ENHANCE, lyrics: '[verse]\nla la', tempo: '96', key: 'Am' };

    async function mountReused(params: Record<string, string>): Promise<void> {
      const context = withAudioEnhancer(withAudio(makeContext({ history_assets: [] }), MODELS));
      galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage([]));
      draft.reset();
      draft.patch({ workflow: 'music' });
      app = await mountWorkspacePage(context, target);
      draft.loadFromUrl(params, context, { replaceSlots: true, exact: true });
      await settle();
    }

    it('puts the style in Style, the rewrite in Enhanced and the options back, with the singer shown for the lyrics', async () => {
      await mountReused(REUSE);

      expect(draft.state).toMatchObject({ audioStyle: 'warm folk', lyrics: '[verse]\nla la', audioTempo: 96, audioKey: 'Am', enhanceModel: 'smart' });
      expect(draft.state.enhanceSettingsAudio.style).toEqual({ genre: 'folk', singer: 'female', instruments: ['guitar'], without: ['drums'], note: 'fingerpicked' });
      expect(field('ws-audio-style').value).toBe('warm folk');
      (target.querySelector('#ws-enhanced-audio-style-tab') as HTMLButtonElement).click();
      await settle();
      expect(field('ws-enhanced-audio-style').value).toBe('Folk, female lead vocal, acoustic guitar');
      expect(descriptionOf(target.querySelector('#ws-enhanced-audio-style-tab')!)).toBe('sent by Generate');
      (document.querySelector('#ws-enhance-toggle') as HTMLButtonElement).click();
      await settle();
      expect(Array.from(document.querySelectorAll('#ws-enhance-panel .picker')).map((picker) => picker.id)).toContain('ws-enhance-audio-singer');
    });

    it('sends the same take again: the rewrite as the prompt, the style it came from and how it was written', async () => {
      await mountReused(REUSE);

      const sent = Object.fromEntries((await submitForm(target)).entries());

      expect(sent).toMatchObject({ model: 'yue2', prompt: 'Folk, female lead vocal, acoustic guitar', prompt_input: 'warm folk', lyrics: '[verse]\nla la', tempo: '96', key: 'Am' });
      expect(JSON.parse(sent.enhance_record as string)).toEqual({ settings: JSON.parse(ENHANCE).settings, model: 'owner/llm', tier: 'smart', format: 'text', task: 'style' });
    });

    it('brings back a Stable Audio song, which has no lyrics and no singer', async () => {
      const record = JSON.stringify({ format: 'text', model: 'owner/llm', settings: { genre: 'folk' }, task: 'style', tier: 'fast' });
      await mountReused({ workflow: 'music', model: 'sa3', prompt: 'warm folk', enhanced_prompt: 'A warm folk tune.', enhance: record, seconds: '45' });

      expect(draft.state.audioRewrites.style).toMatchObject({ text: 'A warm folk tune.', from: { input: 'warm folk', model: 'sa3', task: 'style' } });
      expect(target.querySelector('#ws-lyrics')).toBeNull();
      const sent = Object.fromEntries((await submitForm(target)).entries());
      expect(sent).toMatchObject({ model: 'sa3', prompt: 'A warm folk tune.', prompt_input: 'warm folk' });
    });
  });

  describe('a job that ends', () => {
    const DESIGN = { job_id: 'job-design', workflow: 'voice_design', media_type: 'audio', prompt: 'A warm woman', model: 'qwen3_tts', runs: 3, created_at: '' } as const;

    it('refreshes the Gallery and says Generation complete for a take', async () => {
      await mountAudio('speech');
      toasts.splice(0, toasts.length);
      jobStore.startJob({ job_id: 'job-take', workflow: 'speech', media_type: 'audio', prompt: 'Hello', model: 'qwen3_tts', runs: 1, created_at: '' });
      await settle();
      const loads = galleryApiMocks.getGallery.mock.calls.length;

      lastEventSource().emit('job_completed', { type: 'job_completed', outputs: [] });
      await settle();

      expect(galleryApiMocks.getGallery.mock.calls.length).toBeGreaterThan(loads);
      expect(toasts.map((toast) => toast.message)).toContain('Generation complete');
    });

    it('neither refreshes the Gallery nor says Generation complete for a voice design job, which makes nothing for it', async () => {
      await mountAudio('speech');
      toasts.splice(0, toasts.length);
      jobStore.startJob({ ...DESIGN });
      await settle();
      const loads = galleryApiMocks.getGallery.mock.calls.length;
      const statusReads = workspaceApiMocks.getWorkspaceCoreContext.mock.calls.length;

      lastEventSource().emit('job_completed', { type: 'job_completed', outputs: [] });
      await settle();

      expect(galleryApiMocks.getGallery.mock.calls.length).toBe(loads);
      expect(toasts.map((toast) => toast.message)).not.toContain('Generation complete');
      // The model's download state is still read, so VoiceDesign shows as downloaded.
      expect(workspaceApiMocks.getWorkspaceCoreContext.mock.calls.length).toBeGreaterThan(statusReads);
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

  describe('Song blocks', () => {
    const blocks = () => Array.from(target.querySelectorAll('[data-testid="song-block"]')).map((node) => node.getAttribute('data-section'));
    const words = (index: number) => (field(`ws-lyrics-block-${index}`) as HTMLTextAreaElement).value;
    const asText = () => target.querySelector('#ws-lyrics-as-text') as HTMLButtonElement;
    const click = async (node: Element | null) => {
      (node as HTMLElement).click();
      await settle();
    };
    /** Open block *index*'s ⋯ menu and choose *action*, or open its section-name menu. */
    const choose = async (button: string, action: string) => {
      await click(document.getElementById(button));
      await click(document.querySelector(`[data-action="${action}"]`));
    };
    const SONG = 'Words before\n\n[Verse]\nla la\nlo lo\n\n[Pre-Chorus]\nbuilding\n\n[Chorus]\nhey';

    it('starts as one box for the words, with the sections to add under it', async () => {
      await mountAudio('music');
      expect(blocks()).toEqual([]);
      expect(field('ws-lyrics')).not.toBeNull();
      expect(target.querySelector('[data-action="song-add-verse"]')).not.toBeNull();
      expect(asText().getAttribute('aria-pressed')).toBe('false');
    });

    it('makes an instrumental out of sections without words', async () => {
      await mountAudio('music');
      await click(target.querySelector('[data-action="song-add-verse"]'));
      await click(target.querySelector('[data-action="song-add-chorus"]'));

      expect(draft.state.lyrics).toBe('[Verse]\n\n[Chorus]');
      expect(blocks()).toEqual(['Verse', 'Chorus']);
      expect(status()).toBe('Instrumental · 2 sections · add or remove section tags to change its length');

      type('ws-lyrics-block-0', 'la la');
      await settle();
      expect(draft.state.lyrics).toBe('[Verse]\nla la\n\n[Chorus]');
      expect(status()).toBe('Song · length follows the lyrics');
    });

    it('adds any of the sections the Add menu offers', async () => {
      await mountAudio('music');
      await click(document.getElementById('ws-lyrics-add-more'));
      expect(Array.from(document.querySelectorAll('[role="menuitem"]')).map((item) => item.textContent?.trim())).toEqual(['Intro', 'Verse', 'Pre-Chorus', 'Chorus', 'Post-Chorus', 'Bridge', 'Solo', 'Instrumental', 'Outro']);
      await click(document.querySelector('[data-action="song-section-post-chorus"]'));
      expect(draft.state.lyrics).toBe('[Post-Chorus]');
    });

    it('shows lyrics from a recent song as blocks, with the words before the first tag and unknown sections', async () => {
      await mountAudio('music');
      recentPromptsApiMocks.getRecentPrompts.mockResolvedValue({ entries: [{ format: 'text', prompt: 'dreamy synth-pop', scene: SONG, negative_prompt: '', mode: 'music', used_at: 1 }] });
      await click(document.getElementById('ws-recent-prompts'));
      await click(document.querySelector('[role="option"]'));

      expect(draft.state.lyrics).toBe(SONG);
      expect(blocks()).toEqual(['Verse', 'Pre-Chorus', 'Chorus']);
      expect((field('ws-lyrics') as HTMLTextAreaElement).value).toBe('Words before');
      expect(words(1)).toBe('building');
    });

    it('moves a block up and down in the menu, and not past either end', async () => {
      await mountAudio('music');
      draft.update('lyrics', SONG);
      await settle();

      await choose('ws-lyrics-actions-1', 'song-move-up');
      expect(draft.state.lyrics).toBe('Words before\n\n[Pre-Chorus]\nbuilding\n\n[Verse]\nla la\nlo lo\n\n[Chorus]\nhey');
      await choose('ws-lyrics-actions-1', 'song-move-down');
      expect(draft.state.lyrics).toBe('Words before\n\n[Pre-Chorus]\nbuilding\n\n[Chorus]\nhey\n\n[Verse]\nla la\nlo lo');

      await click(document.getElementById('ws-lyrics-actions-0'));
      expect(document.querySelector('[data-action="song-move-up"]')?.getAttribute('aria-disabled')).toBe('true');
      await click(document.querySelector('[data-action="song-move-up"]'));
      await click(document.getElementById('ws-lyrics-actions-2'));
      expect(document.querySelector('[data-action="song-move-down"]')?.getAttribute('aria-disabled')).toBe('true');
    });

    it('duplicates a block, renames a section, and removes one with a way back', async () => {
      await mountAudio('music');
      draft.update('lyrics', '[Verse]\nla la\n\n[Chorus]\nhey');
      await settle();

      await choose('ws-lyrics-actions-0', 'song-duplicate');
      expect(draft.state.lyrics).toBe('[Verse]\nla la\n\n[Verse]\nla la\n\n[Chorus]\nhey');

      await choose('ws-lyrics-type-1', 'song-section-bridge');
      expect(draft.state.lyrics).toBe('[Verse]\nla la\n\n[Bridge]\nla la\n\n[Chorus]\nhey');

      await choose('ws-lyrics-actions-1', 'song-remove');
      expect(draft.state.lyrics).toBe('[Verse]\nla la\n\n[Chorus]\nhey');
      expect(target.querySelector('.blk-undo')?.textContent).toContain('Removed Bridge');
      await click(target.querySelector('.blk-undo-link'));
      expect(draft.state.lyrics).toBe('[Verse]\nla la\n\n[Bridge]\nla la\n\n[Chorus]\nhey');
      expect(target.querySelector('.blk-undo')).toBeNull();
    });

    it('renames a section to a name of its own', async () => {
      await mountAudio('music');
      draft.update('lyrics', '[Verse]\nla la');
      await settle();

      await choose('ws-lyrics-type-0', 'song-section-custom');
      const input = document.getElementById('ws-lyrics-name-0') as HTMLInputElement;
      input.value = 'Hook 2';
      input.dispatchEvent(new Event('blur'));
      await settle();

      expect(draft.state.lyrics).toBe('[Hook 2]\nla la');
      expect(blocks()).toEqual(['Hook 2']);
    });

    it('selects the whole name when "Other name…" opens, so typing replaces it', async () => {
      await mountAudio('music');
      draft.update('lyrics', '[Hook]\nla la');
      await settle();

      await choose('ws-lyrics-type-0', 'song-section-custom');
      await settle();
      const input = document.getElementById('ws-lyrics-name-0') as HTMLInputElement;
      expect(document.activeElement).toBe(input);
      expect([input.selectionStart, input.selectionEnd]).toEqual([0, 4]);
    });

    it('keeps the keys with the list after a section is removed', async () => {
      await mountAudio('music');
      const focused = async () => {
        await new Promise((resolve) => setTimeout(resolve, 5));
        return document.activeElement?.id;
      };
      draft.update('lyrics', '[Verse]\nla la\n\n[Chorus]\nhey\n\n[Outro]\nend');
      await settle();

      await choose('ws-lyrics-actions-1', 'song-remove');
      expect(await focused()).toBe('ws-lyrics-actions-1');
      await choose('ws-lyrics-actions-1', 'song-remove');
      expect(await focused()).toBe('ws-lyrics-actions-0');
      await choose('ws-lyrics-actions-0', 'song-remove');
      expect(await focused()).toBe('ws-lyrics-undo');

      draft.update('lyrics', '[Verse]');
      await settle();
      await choose('ws-lyrics-actions-0', 'song-remove');
      expect(await focused()).toBe('ws-lyrics');
    });

    it('shows a messy paste as the text it was and keeps it through Edit as text and back', async () => {
      const messy = 'before the tags\r\n[Intro]\n\n\n  [verse 2]  \nla [x] la\n[Chorus] oh oh\n[unclosed\n\n\n';
      await mountAudio('music');
      draft.update('lyrics', messy);
      await settle();
      expect(blocks()).toEqual(['Intro', 'verse 2']);

      await click(asText());
      expect(asText().getAttribute('aria-pressed')).toBe('true');
      expect(target.querySelector('[data-testid="song-block"]')).toBeNull();
      expect((field('ws-lyrics') as HTMLTextAreaElement).value).toBe(messy.replace('\r\n', '\n'));

      type('ws-lyrics', '[Verse]\nnew words\n[Bridge]');
      await settle();
      await click(asText());
      expect(blocks()).toEqual(['Verse', 'Bridge']);
      expect(draft.state.lyrics).toBe('[Verse]\nnew words\n[Bridge]');
    });

    it('keeps the gap and the end of the text as typed when Enter is pressed at the end of a block', async () => {
      await mountAudio('music');
      draft.update('lyrics', '[Verse]\nla la\n\n[Chorus]\nhey');
      await settle();

      // The textarea keeps the trailing newline it was given; the text must not count it a second time.
      type('ws-lyrics-block-0', 'la la\n');
      await settle();
      type('ws-lyrics-block-0', 'la la\nlo lo');
      await settle();
      expect(draft.state.lyrics).toBe('[Verse]\nla la\nlo lo\n\n[Chorus]\nhey');

      type('ws-lyrics-block-1', 'hey\n');
      await settle();
      type('ws-lyrics-block-1', 'hey\n\nho');
      await settle();
      expect(draft.state.lyrics).toBe('[Verse]\nla la\nlo lo\n\n[Chorus]\nhey\n\nho');
    });

    it('moves the caret into the new block when a tag typed by hand starts a section', async () => {
      await mountAudio('music');
      const focused = () => document.activeElement?.id;

      // From the empty box: the box has no words once its text is only a tag, and the next keys must go to the new block.
      field('ws-lyrics').focus();
      type('ws-lyrics', '[Verse]');
      await settle();
      expect(blocks()).toEqual(['Verse']);
      expect(focused()).toBe('ws-lyrics-block-0');

      // From the end of a block: the keys after the closing bracket belong under the new tag, not above it.
      field('ws-lyrics-block-0').focus();
      type('ws-lyrics-block-0', 'la la\n[Chorus]');
      await settle();
      expect(draft.state.lyrics).toBe('[Verse]\nla la\n[Chorus]');
      expect(blocks()).toEqual(['Verse', 'Chorus']);
      expect(focused()).toBe('ws-lyrics-block-1');
    });

    it('sends the lyrics the blocks write', async () => {
      await mountAudio('music');
      await typeInto('ws-audio-style', 'Dreamy synth-pop');
      await click(target.querySelector('[data-action="song-add-verse"]'));
      type('ws-lyrics-block-0', 'The lamp is lit');
      await settle();

      expect(Object.fromEntries((await submitForm(target)).entries())).toMatchObject({ prompt: 'Dreamy synth-pop', lyrics: '[Verse]\nThe lamp is lit' });
    });
  });

  describe('Lyrics helper', () => {
    const MODELS = [...AUDIO_MODELS, SA3_MUSIC];
    const SONG = '[Verse]\nThe porch light flickers, the street is wet\nI still keep your coat by the door\n\n[Chorus]\nCome back home\nCome back home to me\n\n[Bridge]\n\n[Outro]\n';
    const SHORTER = SONG.replace('The porch light flickers, the street is wet\nI still keep your coat by the door', 'Porch light flickers, street is wet\nI keep your coat by the door');
    const EVEN_SHORTER = SHORTER.replace('Come back home\nCome back home to me', 'Come home\nCome home to me');

    async function mountHelper(options: { compact?: boolean; tall?: boolean; lyrics?: string; style?: string; model?: string; active?: boolean } = {}): Promise<WorkspaceContext> {
      const running = jobSnap('job-live', 'running', { workflow: 'music' });
      const base = withAudioEnhancer(withAudio(makeContext({ history_assets: [] }), MODELS));
      base.prompt_enhancer!.smart = { downloaded: true, download_size_label: null, unavailable: null };
      const context = options.active ? { ...base, active_job: running, queued_jobs: [] } : base;
      if (options.active) workspaceApiMocks.listJobs.mockResolvedValue({ active_job: running, queued_jobs: [] });
      layoutProbe.compact = options.compact ?? false;
      layoutProbe.tall = options.tall ?? false;
      galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage([]));
      draft.reset();
      draft.patch({ workflow: 'music', lyrics: options.lyrics ?? SONG, audioStyle: options.style ?? 'warm folk' });
      app = await mountWorkspacePage(context, target);
      if (options.model) {
        draft.hydrateFromContext(context, options.model);
        await settle();
      }
      return context;
    }

    /** Make the enhancer answer each run with the next of *texts*, as the whole song; the second argument may change the draft first. */
    function answerWith(...texts: (string | { text: string; warning?: string; during?: () => void })[]): void {
      const queue = [...texts];
      enhanceApiMocks.enhancePrompt.mockImplementation(async (_body: unknown, ...rest: unknown[]) => {
        const onFrame = rest[0] as (frame: AudioEnhanceFrame) => void;
        const next = queue.shift() ?? 'More.';
        const { text, warning, during } = typeof next === 'string' ? { text: next, warning: undefined, during: undefined } : next;
        onFrame({ type: 'status', phase: 'generating' });
        during?.();
        onFrame({ type: 'done', task: 'lyrics', fields: { lyrics: text }, clamped: false, model: 'owner/llm', ...(warning ? { warning } : {}) });
      });
    }

    const lastBody = () => enhanceApiMocks.enhancePrompt.mock.calls.at(-1)?.[0] as Record<string, unknown>;
    const toggle = () => document.querySelector('#ws-lyrics-helper-toggle') as HTMLButtonElement;
    const run = () => document.querySelector('#ws-lyrics-helper-run') as HTMLButtonElement;
    const panel = () => document.querySelector('#ws-lyrics-helper-panel') as HTMLElement | null;
    const stepperText = () => target.querySelector('[role="group"][aria-label="Lyrics versions"] .rewrite-count')?.textContent ?? null;
    const star = () => target.querySelector('[data-testid="song-blocks"] [aria-label="written by a helper"]');
    /** The sheet the helper is open in (the page also keeps the settings sheet). */
    const helperSheet = () => document.querySelector('#ws-lyrics-helper-panel')?.closest('[data-testid="sheet"]') as HTMLElement;
    const click = async (node: Element | null) => {
      (node as HTMLElement).click();
      await settle();
    };
    const chip = (slug: string) => document.querySelector(`#ws-lyrics-helper-panel [data-improve="${slug}"]`) as HTMLButtonElement;

    async function improve(): Promise<void> {
      await click(toggle());
      await click(run());
    }

    it('adds an Improve lyrics tool, on YuE2 and not on Stable Audio', async () => {
      await mountHelper();
      expect(toggle().textContent).toContain('Improve lyrics');
      expect(target.querySelector('.song-role')).toBeNull();

      draft.hydrateFromContext(withAudioEnhancer(withAudio(makeContext({ history_assets: [] }), MODELS)), 'sa3');
      await settle();

      expect(target.querySelector('#ws-lyrics-helper-toggle')).toBeNull();
    });

    it('has no helper on a machine without the prompt enhancer', async () => {
      await mountAudio('music');
      draft.update('lyrics', SONG);
      await settle();

      expect(target.querySelector('#ws-lyrics-helper-toggle')).toBeNull();
      expect(target.querySelector('#ws-lyrics-help-0')).toBeNull();
      await click(document.getElementById('ws-lyrics-actions-0'));
      expect(document.querySelector('[data-action="song-help"]')).toBeNull();
    });

    it('offers Tighten, Fit the sections, Smooth rhymes, Simpler words and a Direction on the whole song', async () => {
      await mountHelper();
      await click(toggle());

      expect(Array.from(document.querySelectorAll('#ws-lyrics-helper-panel [data-improve]')).map((node) => node.textContent)).toEqual(['Tighten', 'Fit the sections', 'Smooth rhymes', 'Simpler words']);
      expect(document.querySelector('#ws-lyrics-helper-note')).not.toBeNull();
      expect(panel()?.textContent).toContain('Written to match your style: warm folk');
      expect(run().textContent?.trim()).toBe('Improve lyrics');
    });

    it('says an empty song has nothing to improve and points to Describe it, with nothing to run', async () => {
      await mountHelper({ lyrics: '' });
      await click(toggle());

      expect(panel()?.textContent).toContain('Nothing to improve yet');
      expect(panel()?.textContent).toContain('Describe it');
      expect(run()).toBeNull();
      // The one way forward is Describe it, offered as a button.
      expect([...(panel()?.querySelectorAll('button') ?? [])].map((button) => button.textContent?.trim())).toEqual(['Fill in from Describe it']);
    });

    it('treats sections without words as an empty song', async () => {
      await mountHelper({ lyrics: '[Verse]\n\n[Chorus]' });
      await click(toggle());

      expect(panel()?.textContent).toContain('Nothing to improve yet');
    });

    it('improves the whole song: the request names the task, the picks, the direction and the style', async () => {
      await mountHelper();
      answerWith(SHORTER);
      await click(toggle());
      await click(chip('rhymes'));
      type('ws-lyrics-helper-note', 'more hopeful');
      await settle();
      await click(run());

      expect(lastBody()).toEqual({
        mode: 'audio',
        task: 'lyrics',
        model: 'yue2',
        kind: 'music',
        prompt: SONG,
        settings: { improve: ['tighten', 'rhymes'], note: 'more hopeful' },
        context: { style: 'warm folk' },
      });
      expect(draft.state.lyrics).toBe(SHORTER);
      expect(panel()).toBeNull();
    });

    it('opens with Tighten picked, and a polish is what runs when every pick is cleared', async () => {
      await mountHelper();
      answerWith(SHORTER);
      await click(toggle());
      expect(chip('tighten').getAttribute('aria-pressed')).toBe('true');
      expect(chip('rhymes').getAttribute('aria-pressed')).toBe('false');
      await click(run());
      expect(lastBody()).toMatchObject({ settings: { improve: ['tighten'] } });

      await click(toggle());
      await click(chip('tighten'));
      await click(run());
      expect(lastBody()).toMatchObject({ settings: { improve: [] } });
    });

    it('sends the Enhanced style, which is what the take will carry, as the style the words suit', async () => {
      await mountHelper();
      draft.patch({ audioRewrites: { ...draft.state.audioRewrites, style: { text: 'Folk, warm, acoustic guitar', from: { input: 'warm folk', model: 'yue2', task: 'style' }, history: [] } } });
      await settle();
      answerWith(SHORTER);
      await improve();

      expect((lastBody().context as { style: string }).style).toBe('Folk, warm, acoustic guitar');
    });

    it('saves the lyrics as mine with the first result, adds each result, and steps between them', async () => {
      await mountHelper();
      answerWith(SHORTER, EVEN_SHORTER);
      expect(stepperText()).toBeNull();

      await improve();
      expect(draft.state.lyricsVersions).toEqual([{ text: SONG, mine: true }, { text: SHORTER, mine: false }]);
      expect(stepperText()).toBe('2 of 2');

      await improve();
      expect(draft.state.lyricsVersions.map((version) => version.text)).toEqual([SONG, SHORTER, EVEN_SHORTER]);
      expect(stepperText()).toBe('3 of 3');

      await click(target.querySelector('[aria-label="Previous version"]'));
      expect(draft.state.lyrics).toBe(SHORTER);
      expect(stepperText()).toBe('2 of 3');
      await click(target.querySelector('[aria-label="Next version"]'));
      expect(draft.state.lyrics).toBe(EVEN_SHORTER);
    });

    it('works from the version that is shown, and ‹ steps back to the user\'s own lyrics, which carry no ✦', async () => {
      await mountHelper();
      answerWith(SHORTER);
      await improve();
      expect(star()).not.toBeNull();
      expect(document.querySelector('#ws-lyrics-use-mine')).toBeNull();

      await click(target.querySelector('[aria-label="Previous version"]'));

      expect(draft.state.lyrics).toBe(SONG);
      expect(stepperText()).toBe('1 of 2');
      expect(star()).toBeNull();
      answerWith(EVEN_SHORTER);
      await improve();
      expect((lastBody() as { prompt: string }).prompt).toBe(SONG);
      expect(draft.state.lyricsVersions.map((version) => version.text)).toEqual([SONG, SHORTER, EVEN_SHORTER]);
    });

    it('marks the version a helper wrote with a ✦ on the name, and typing in it makes it yours so the ✦ goes', async () => {
      await mountHelper();
      expect(star()).toBeNull();
      answerWith(SHORTER);
      await improve();
      expect(star()).not.toBeNull();

      type('ws-lyrics-block-0', 'My own first verse');
      await settle();

      expect(star()).toBeNull();
      expect(draft.state.lyricsVersions.map((version) => version.mine)).toEqual([true, true]);
      // Stepping to the version before and back finds the user's edit, still theirs.
      await click(target.querySelector('[aria-label="Previous version"]'));
      await click(target.querySelector('[aria-label="Next version"]'));
      expect(draft.state.lyrics).toContain('My own first verse');
      expect(star()).toBeNull();
    });

    it('puts Edit as text and then the stepper at the right of the Lyrics head, with Edit as text a pressed toggle', async () => {
      await mountHelper();
      answerWith(SHORTER);
      await improve();

      const head = target.querySelector('[data-testid="song-blocks"] .head')!;
      const right = head.querySelector('.right')!;
      expect(Array.from(right.children).map((node) => node.getAttribute('aria-label') ?? node.textContent?.trim())).toEqual(['Edit as text', 'Lyrics versions']);
      const asText = document.getElementById('ws-lyrics-as-text') as HTMLButtonElement;
      expect(asText.getAttribute('aria-pressed')).toBe('false');
      await click(asText);
      expect(asText.getAttribute('aria-pressed')).toBe('true');
      expect((field('ws-lyrics') as HTMLTextAreaElement).value).toContain('[Verse]');
    });

    it('keeps five versions, the user\'s own and the newest four', async () => {
      await mountHelper();
      answerWith('[Verse]\nv1', '[Verse]\nv2', '[Verse]\nv3', '[Verse]\nv4', '[Verse]\nv5', '[Verse]\nv6');
      for (let n = 0; n < 6; n += 1) await improve();

      expect(draft.state.lyricsVersions.map((version) => version.text)).toEqual([SONG, '[Verse]\nv3', '[Verse]\nv4', '[Verse]\nv5', '[Verse]\nv6']);
      expect(stepperText()).toBe('5 of 5');
    });

    it('changes the shown version when the lyrics are edited, and Generate sends what is shown', async () => {
      await mountHelper();
      answerWith(SHORTER);
      await improve();
      type('ws-lyrics-block-0', 'My own first verse');
      await settle();

      expect(draft.state.lyrics).toBe(SHORTER.replace('Porch light flickers, street is wet\nI keep your coat by the door', 'My own first verse'));
      expect(draft.state.lyricsVersions.map((version) => version.text)).toEqual([SONG, draft.state.lyrics]);
      await click(target.querySelector('[aria-label="Previous version"]'));
      expect(draft.state.lyrics).toBe(SONG);
      await click(target.querySelector('[aria-label="Next version"]'));
      expect(draft.state.lyrics).toContain('My own first verse');

      const sent = Object.fromEntries((await submitForm(target)).entries());
      expect(sent.lyrics).toBe(draft.state.lyrics);
    });

    it('drops the version stepper when something else sets the lyrics, and starts over from them', async () => {
      await mountHelper();
      answerWith(SHORTER, EVEN_SHORTER);
      await improve();
      draft.update('lyrics', '[Verse]\nfrom a recent song');
      await settle();

      expect(stepperText()).toBeNull();
      expect(star()).toBeNull();
      await improve();
      expect(draft.state.lyricsVersions).toEqual([{ text: '[Verse]\nfrom a recent song', mine: true }, { text: EVEN_SHORTER, mine: false }]);
    });

    it('improves one section from its menu with the song as context, and splices the result', async () => {
      await mountHelper();
      answerWith(SONG.replace('Come back home\nCome back home to me', 'Come home\nCome home to me'));
      await click(document.getElementById('ws-lyrics-actions-1'));
      const item = document.querySelector('[data-action="song-help"]') as HTMLElement;
      expect(item.textContent?.trim()).toBe('Improve this section…');
      await click(item);

      expect(panel()?.getAttribute('aria-label')).toBe('Improve Chorus');
      expect(Array.from(document.querySelectorAll('#ws-lyrics-helper-panel [data-improve]')).map((node) => node.textContent)).toEqual(['Tighten', 'Smooth rhymes', 'Simpler words']);
      expect(panel()?.textContent).toContain('Knows the style and the other 3 sections.');
      expect(chip('tighten').getAttribute('aria-pressed')).toBe('true');
      await click(run());

      expect(lastBody()).toMatchObject({ task: 'lyrics', prompt: SONG, settings: { improve: ['tighten'], section: 1 } });
      expect(draft.state.lyrics).toBe(SONG.replace('Come back home\nCome back home to me', 'Come home\nCome home to me'));
      expect(draft.state.lyricsVersions.map((version) => version.mine)).toEqual([true, false]);
    });

    it('keeps the whole song\'s picks and direction between runs, and starts a section with none', async () => {
      await mountHelper();
      answerWith(SHORTER);
      await click(toggle());
      await click(chip('tighten'));
      await click(chip('rhymes'));
      type('ws-lyrics-helper-note', 'more hopeful');
      await settle();
      await click(run());

      await click(document.getElementById('ws-lyrics-help-1'));
      expect(chip('tighten').getAttribute('aria-pressed')).toBe('true');
      expect((document.querySelector('#ws-lyrics-helper-note') as HTMLInputElement).value).toBe('');
      await click(chip('simpler'));
      await click(document.getElementById('ws-lyrics-help-1'));
      await click(document.getElementById('ws-lyrics-help-1'));
      expect(chip('simpler').getAttribute('aria-pressed')).toBe('false');
      await click(document.getElementById('ws-lyrics-help-1'));

      await click(toggle());
      expect(chip('tighten').getAttribute('aria-pressed')).toBe('false');
      expect(chip('rhymes').getAttribute('aria-pressed')).toBe('true');
      expect((document.querySelector('#ws-lyrics-helper-note') as HTMLInputElement).value).toBe('more hopeful');
    });

    it('opens from the block\'s own Improve button, beside it', async () => {
      await mountHelper();
      const button = document.getElementById('ws-lyrics-help-0') as HTMLButtonElement;
      expect(button.textContent?.trim()).toBe('Improve');
      expect(button.getAttribute('aria-label')).toBe('Improve section 1: Verse');

      await click(button);
      expect(panel()?.getAttribute('aria-label')).toBe('Improve Verse');
      expect(button.getAttribute('aria-expanded')).toBe('true');
      await click(button);
      expect(panel()).toBeNull();
    });

    it('offers Write on an empty section, from the header and the menu, and writes only that section', async () => {
      await mountHelper();
      answerWith(SONG.replace('[Bridge]\n', '[Bridge]\nThe night is long\nbut I am still here\n'));
      const button = document.getElementById('ws-lyrics-help-2') as HTMLButtonElement;
      expect(button.textContent?.trim()).toBe('Write');
      await click(document.getElementById('ws-lyrics-actions-2'));
      expect(document.querySelector('[data-action="song-help"]')?.textContent?.trim()).toBe('Write');
      await click(document.querySelector('[data-action="song-help"]'));

      expect(panel()?.getAttribute('aria-label')).toBe('Write Bridge');
      expect(document.querySelector('#ws-lyrics-helper-panel [data-improve]')).toBeNull();
      type('ws-lyrics-helper-note', 'a turn toward hope');
      await settle();
      expect(run().textContent?.trim()).toBe('Write bridge');
      await click(run());

      expect(lastBody()).toMatchObject({ settings: { action: 'write', section: 2, note: 'a turn toward hope' } });
      expect(draft.state.lyrics).toBe(SONG.replace('[Bridge]\n', '[Bridge]\nThe night is long\nbut I am still here\n'));
      expect(draft.state.lyricsVersions.map((version) => version.mine)).toEqual([true, false]);
      expect(stepperText()).toBe('2 of 2');
    });

    it('opens Write on a song of only tags, with nothing to run and Fill in as the way forward', async () => {
      await mountHelper({ lyrics: '[Verse]\n\n[Chorus]' });
      const opener = document.getElementById('ws-lyrics-help-0') as HTMLButtonElement;
      expect(opener.disabled).toBe(false);
      expect(opener.getAttribute('title')).toBeNull();

      await click(opener);

      expect(panel()?.getAttribute('aria-label')).toBe('Write Verse');
      expect(panel()?.querySelector('[data-testid="lyrics-helper-problem"]')).not.toBeNull();
      expect(run()).toBeNull();
      expect([...(panel()?.querySelectorAll('button') ?? [])].map((button) => button.textContent?.trim())).toEqual(['Fill in from Describe it']);
    });

    it('opens Write from the section menu on a song of only tags', async () => {
      await mountHelper({ lyrics: '[Verse]\n\n[Chorus]' });
      await click(document.getElementById('ws-lyrics-actions-0'));
      const item = document.querySelector('[data-action="song-help"]');

      expect(item?.getAttribute('aria-disabled')).not.toBe('true');
      await click(item);

      expect(panel()?.getAttribute('aria-label')).toBe('Write Verse');
      expect(panel()?.querySelector('#ws-lyrics-helper-fill')).not.toBeNull();
    });

    it('shows the enhancer\'s notice about a changed quote', async () => {
      await mountHelper();
      answerWith({ text: SHORTER, warning: 'A quoted text was changed in the rewrite.' });
      await improve();

      expect(target.querySelector('[data-testid="lyrics-note"]')?.textContent).toBe('A quoted text was changed in the rewrite.');
      type('ws-lyrics-block-0', 'typed');
      await settle();
      expect(target.querySelector('[data-testid="lyrics-note"]')).toBeNull();
    });

    it('says a run that changed nothing as a neutral note, not an error, and keeps the lyrics', async () => {
      await mountHelper();
      enhanceApiMocks.enhancePrompt.mockImplementation(async (_body: unknown, ...rest: unknown[]) => {
        (rest[0] as (frame: AudioEnhanceFrame) => void)({ type: 'error', detail: 'No changes: the helper left your lyrics as they are.', no_change: true });
      });
      await improve();

      expect(panel()?.querySelector('[data-testid="lyrics-helper-note"]')?.textContent).toBe('No changes: the helper left your lyrics as they are.');
      expect(panel()?.querySelector('[role="alert"]')).toBeNull();
      expect(draft.state.lyrics).toBe(SONG);
    });

    it('keeps the lyrics and says why when the helper changed the sections', async () => {
      await mountHelper();
      enhanceApiMocks.enhancePrompt.mockImplementation(async (_body: unknown, ...rest: unknown[]) => {
        (rest[0] as (frame: AudioEnhanceFrame) => void)({ type: 'error', detail: 'The helper changed the sections; try again.' });
      });
      await improve();

      expect(panel()?.querySelector('[role="alert"]')?.textContent).toBe('The helper changed the sections; try again.');
      expect(draft.state.lyrics).toBe(SONG);
      expect(draft.state.lyricsVersions).toEqual([]);
    });

    it('does not put an answer back over lyrics that changed while it was working', async () => {
      await mountHelper();
      answerWith({ text: SHORTER, during: () => draft.update('lyrics', `${SONG}\n[Verse]\nadded meanwhile`) });
      await improve();

      expect(draft.state.lyrics).toBe(`${SONG}\n[Verse]\nadded meanwhile`);
      expect(draft.state.lyricsVersions).toEqual([]);
      expect(panel()?.querySelector('[role="alert"]')?.textContent).toContain('Your lyrics changed while the helper was working');
    });

    it('still opens while a take is being made, and says in the panel when it can run', async () => {
      await mountHelper({ active: true });

      expect(target.querySelector('.song-role')).toBeNull();
      expect(toggle().disabled).toBe(false);
      await click(toggle());
      expect(run().disabled).toBe(true);
      expect(panel()?.textContent).toContain('Available when the take is done.');
      await click(toggle());

      const opener = document.getElementById('ws-lyrics-help-0') as HTMLButtonElement;
      expect(opener.disabled).toBe(false);
      await click(opener);
      expect(panel()?.getAttribute('aria-label')).toBe('Improve Verse');
      expect(run().disabled).toBe(true);
      expect(panel()?.textContent).toContain('Available when the take is done.');
      await click(opener);

      await click(document.getElementById('ws-lyrics-actions-0'));
      const item = document.querySelector('[data-action="song-help"]');
      expect(item?.getAttribute('aria-disabled')).not.toBe('true');
    });

    it('asks for the Smart enhancer when it is chosen', async () => {
      await mountHelper();
      draft.update('enhanceModel', 'smart');
      answerWith(SHORTER);
      await improve();

      expect(lastBody()).toMatchObject({ enhance_model: 'smart' });
    });

    it('works on the whole song while the lyrics are shown as text', async () => {
      await mountHelper();
      await click(document.getElementById('ws-lyrics-as-text'));
      answerWith(SHORTER);
      await improve();

      expect(draft.state.lyrics).toBe(SHORTER);
      expect((field('ws-lyrics') as HTMLTextAreaElement).value).toBe(SHORTER);
    });

    it('closes when the Music model is one that takes no lyrics', async () => {
      await mountHelper();
      await click(toggle());
      expect(panel()).not.toBeNull();

      draft.hydrateFromContext(withAudioEnhancer(withAudio(makeContext({ history_assets: [] }), MODELS)), 'sa3');
      await settle();

      expect(panel()).toBeNull();
    });

    describe('on a tablet and a phone', () => {
      beforeEach(() => {
        Element.prototype.animate = vi.fn(() => ({ cancel: vi.fn(), finished: Promise.resolve(), onfinish: null })) as unknown as Element['animate'];
      });
      afterEach(() => {
        delete (Element.prototype as Partial<Element>).animate;
      });

      it.each([
        ['a phone', { compact: true }],
        ['a tablet', { tall: true }],
      ])('opens as a sheet with the lyrics on top and the run button in its foot on %s', async (_name, layoutOptions) => {
        await mountHelper(layoutOptions);
        answerWith(SHORTER);
        await click(toggle());

        const sheet = helperSheet();
        expect(sheet.querySelector('[data-testid="lyrics-helper-quote"]')?.textContent).toContain('The porch light flickers');
        expect(sheet.querySelector('.sheet-foot #ws-lyrics-helper-run')).not.toBeNull();
        expect(sheet.querySelector('.sheet-title')?.textContent).toBe('Improve lyrics');
        expect(chip('tighten').getAttribute('aria-pressed')).toBe('true');
        await click(chip('simpler'));
        await click(run());

        expect(lastBody()).toMatchObject({ settings: { improve: ['tighten', 'simpler'] } });
        expect(draft.state.lyrics).toBe(SHORTER);
      });

      it('opens Write on a song of only tags in a sheet, with Fill in and no run button, on a phone and a tablet', async () => {
        await mountHelper({ compact: true, lyrics: '[Verse]\n\n[Chorus]' });
        await click(document.getElementById('ws-lyrics-actions-1'));
        await click(document.querySelector('[data-action="song-help"]'));

        const sheet = helperSheet();
        expect(sheet.querySelector('.sheet-title')?.textContent).toBe('Write Chorus');
        expect(sheet.querySelector('[data-testid="lyrics-helper-problem"]')).not.toBeNull();
        expect(sheet.querySelector('#ws-lyrics-helper-fill')).not.toBeNull();
        expect(run()).toBeNull();
      });

      it('opens during a take as a sheet whose run button waits for the take', async () => {
        await mountHelper({ tall: true, active: true });
        await click(document.getElementById('ws-lyrics-help-0'));

        const sheet = helperSheet();
        expect(sheet.querySelector('.sheet-title')?.textContent).toBe('Improve Verse');
        expect((sheet.querySelector('#ws-lyrics-helper-run') as HTMLButtonElement).disabled).toBe(true);
        expect(sheet.textContent).toContain('Available when the take is done.');
      });

      it('writes a section from a sheet that names the section', async () => {
        await mountHelper({ compact: true });
        answerWith(SONG.replace('[Bridge]\n', '[Bridge]\nthe night is long\n'));
        await click(document.getElementById('ws-lyrics-actions-2'));
        await click(document.querySelector('[data-action="song-help"]'));

        const sheet = helperSheet();
        expect(sheet.querySelector('.sheet-title')?.textContent).toBe('Write Bridge');
        expect(sheet.querySelector('[data-testid="lyrics-helper-quote"]')?.textContent).toBe('No words yet');
        await click(run());

        expect(draft.state.lyrics).toContain('[Bridge]\nthe night is long');
      });

      it('explains an empty song in the sheet', async () => {
        await mountHelper({ compact: true, lyrics: '' });
        await click(toggle());

        expect(helperSheet().textContent).toContain('Nothing to improve yet');
        expect(run()).toBeNull();
      });
    });
  });

  describe('Script helper', () => {
    const SCRIPT = 'I never told anyone. Not even you. The door was open.';
    const MARKED = '[whisper] I never told anyone. …\nNot even you. [as set] The door was open.';
    const SHORT = 'I never told anyone. The door was open.';

    async function mountScript(options: { compact?: boolean; tall?: boolean; text?: string; enhancer?: boolean; recording?: boolean; pace?: string; active?: boolean } = {}): Promise<WorkspaceContext> {
      const running = jobSnap('job-live', 'running', { workflow: 'speech' });
      const base = withAudio(makeContext({ history_assets: [] }));
      const withEnhancer = options.enhancer === false ? base : withAudioEnhancer(base);
      if (withEnhancer.prompt_enhancer) withEnhancer.prompt_enhancer.smart = { downloaded: true, download_size_label: null, unavailable: null };
      const context = options.active ? { ...withEnhancer, active_job: running, queued_jobs: [] } : withEnhancer;
      if (options.active) workspaceApiMocks.listJobs.mockResolvedValue({ active_job: running, queued_jobs: [] });
      layoutProbe.compact = options.compact ?? false;
      layoutProbe.tall = options.tall ?? false;
      galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage([]));
      draft.reset();
      draft.patch({ workflow: 'speech', speechText: options.text ?? SCRIPT });
      if (options.pace) draft.patch({ speechDelivery: { ...draft.state.speechDelivery, pace: options.pace as never } });
      app = await mountWorkspacePage(context, target);
      if (options.recording) await chooseRecording();
      return context;
    }

    /** Make the enhancer answer each run with the next of *texts* as the whole script; an object may also carry a notice or change the draft first. */
    function answerWith(...texts: (string | { text: string; warning?: string; during?: () => void })[]): void {
      const queue = [...texts];
      enhanceApiMocks.enhancePrompt.mockImplementation(async (_body: unknown, ...rest: unknown[]) => {
        const onFrame = rest[0] as (frame: AudioEnhanceFrame) => void;
        const next = queue.shift() ?? 'More.';
        const { text, warning, during } = typeof next === 'string' ? { text: next, warning: undefined, during: undefined } : next;
        onFrame({ type: 'status', phase: 'generating' });
        during?.();
        onFrame({ type: 'done', task: 'script', fields: { script: text }, clamped: false, model: 'owner/llm', ...(warning ? { warning } : {}) });
      });
    }

    const lastBody = () => enhanceApiMocks.enhancePrompt.mock.calls.at(-1)?.[0] as Record<string, unknown>;
    const toggle = () => document.querySelector('#ws-script-helper-toggle') as HTMLButtonElement | null;
    const run = () => document.querySelector('#ws-script-helper-run') as HTMLButtonElement | null;
    const panel = () => document.querySelector('#ws-script-helper-panel') as HTMLElement | null;
    const tab = (name: 'text' | 'written') => target.querySelector(`#${name === 'text' ? 'ws-prompt' : 'ws-enhanced-prompt'}-tab`) as HTMLButtonElement;
    const counter = () => target.querySelector('.rewrite-count')?.textContent ?? null;
    const scratchButton = () => target.querySelector('[aria-label="Scratch the enhanced text"]') as HTMLButtonElement | null;
    const mark = () => tab('written').querySelector('[data-mark]')?.getAttribute('data-mark') ?? null;
    const written = (): AudioRewrites => draft.state.audioRewrites.text;
    const action = (name: string) => document.querySelector(`#ws-script-helper-panel [data-testid="script-helper-actions"] [data-value="${name}"]`) as HTMLButtonElement | null;
    const picked = () => Array.from(document.querySelectorAll('#ws-script-helper-panel [data-testid="script-helper-actions"] [data-value]')).filter((node) => node.getAttribute('aria-pressed') === 'true').map((node) => node.getAttribute('data-value'));
    const seconds = () => document.querySelector('#ws-script-helper-seconds') as HTMLInputElement;
    const helperSheet = () => document.querySelector('#ws-script-helper-panel')?.closest('[data-testid="sheet"]') as HTMLElement;
    const click = async (node: Element | null) => {
      (node as HTMLElement).click();
      await settle();
    };

    async function addDelivery(): Promise<void> {
      await click(toggle());
      await click(run());
    }

    it('adds a Written tab and a Script helper tool, beside Insert', async () => {
      await mountScript();

      expect(Array.from(target.querySelectorAll('[role="tab"]')).map((node) => node.textContent?.trim())).toEqual(['Text', 'Written']);
      expect(toggle()?.textContent).toContain('Script helper');
      expect(target.querySelector('#ws-insert')).not.toBeNull();
      expect(target.querySelector('.tools-role')).toBeNull();
    });

    it('has no helper and no Written tab on a machine without the prompt enhancer', async () => {
      await mountScript({ enhancer: false });

      expect(toggle()).toBeNull();
      expect(target.querySelectorAll('[role="tab"]')).toHaveLength(0);
      expect(target.querySelector('.tools-role')).toBeNull();
    });

    it('offers Add delivery and Fit to time as pick-any toggles starting from your Text, with Add delivery picked and no Fit field', async () => {
      await mountScript();
      await click(toggle());

      expect(Array.from(document.querySelectorAll('#ws-script-helper-panel [data-testid="script-helper-actions"] [aria-pressed]')).map((node) => node.textContent)).toEqual(['Fit to time', 'Add delivery']);
      expect(picked()).toEqual(['delivery']);
      expect(panel()?.textContent).toContain('Starts from your Text');
      expect(panel()?.textContent).toContain('never changes a word');
      expect(seconds()).toBeNull();
      expect(run()?.textContent?.trim()).toBe('Add delivery');
    });

    it('names the picks on the run button, shows the Fit field only while Fit is picked, and is off with nothing picked', async () => {
      await mountScript();
      await click(toggle());

      await click(action('fit'));
      expect(picked()).toEqual(['fit', 'delivery']);
      expect(run()?.textContent?.trim()).toBe('Fit and add delivery');
      expect(seconds()).not.toBeNull();

      await click(action('delivery'));
      expect(run()?.textContent?.trim()).toBe('Fit to time');
      expect(seconds()).not.toBeNull();

      await click(action('fit'));
      expect(picked()).toEqual([]);
      expect(run()?.disabled).toBe(true);
      expect(seconds()).toBeNull();
    });

    it('adds delivery: the request names the task and the actions, and the result lands in the Written tab', async () => {
      await mountScript();
      answerWith(MARKED);
      await addDelivery();

      expect(lastBody()).toEqual({ mode: 'audio', task: 'script', model: 'qwen3_tts', kind: 'speech', prompt: SCRIPT, settings: { actions: ['delivery'] } });
      expect(panel()).toBeNull();
      expect(tab('written').getAttribute('aria-selected')).toBe('true');
      expect(field('ws-enhanced-prompt').value).toBe(MARKED);
      expect(draft.state.speechText).toBe(SCRIPT);
      expect(descriptionOf(tab('written'))).toBe('sent by Generate');
      expect(target.querySelector('[data-testid="prompt-statusline"]')).toBeNull();
    });

    it('starts Fit at the reading time rounded and sends the seconds with the pace', async () => {
      await mountScript({ text: 'a'.repeat(700), pace: 'slow' });
      answerWith('b'.repeat(500));
      await click(toggle());
      await click(action('delivery'));
      await click(action('fit'));

      // 700 characters at 14 a second and the Slow factor of 0.81 are 61.7 seconds: 60 when rounded to 5.
      expect(seconds().value).toBe('60');
      expect(panel()?.textContent).toContain('Now about 1:02 at the slow pace');
      expect(run()?.textContent?.trim()).toBe('Fit to time');

      type('ws-script-helper-seconds', '45');
      await settle();
      await click(run());

      expect(lastBody()).toMatchObject({ task: 'script', settings: { actions: ['fit'], seconds: 45, pace: 'slow' } });
    });

    it('fits and adds delivery in one request, with one record, and lands one Written version', async () => {
      await mountScript({ pace: 'fast' });
      answerWith(MARKED);
      await click(toggle());
      await click(action('fit'));
      type('ws-script-helper-seconds', '20');
      await settle();
      expect(run()?.textContent?.trim()).toBe('Fit and add delivery');
      await click(run());

      expect(enhanceApiMocks.enhancePrompt).toHaveBeenCalledTimes(1);
      expect(lastBody()).toEqual({ mode: 'audio', task: 'script', model: 'qwen3_tts', kind: 'speech', prompt: SCRIPT, settings: { actions: ['fit', 'delivery'], seconds: 20, pace: 'fast' } });
      expect(written().history).toHaveLength(1);
      expect(written().text).toBe(MARKED);
      const sent = await submitForm(target);
      expect(sent.get('prompt')).toBe(MARKED);
      expect(sent.get('prompt_input')).toBe(SCRIPT);
      expect(JSON.parse(sent.get('enhance_record') as string)).toEqual({ settings: { actions: ['fit', 'delivery'], seconds: 20, pace: 'fast' }, model: 'owner/llm', format: 'text', task: 'script' });
    });

    it('keeps the fitted text when delivery added nothing, and shows the server\'s note under the box', async () => {
      await mountScript();
      answerWith({ text: SHORT, warning: 'Fitted to time. Nothing to add: the helper found no place for a mark or pause.' });
      await click(toggle());
      await click(action('fit'));
      await click(run());

      expect(written().text).toBe(SHORT);
      expect(target.querySelector('[data-testid="script-note"]')?.textContent).toBe('Fitted to time. Nothing to add: the helper found no place for a mark or pause.');
      expect((await submitForm(target)).get('prompt')).toBe(SHORT);
    });

    it('leaves a Normal pace out of the request, and counts a voice from a recording at Normal', async () => {
      await mountScript({ text: 'a'.repeat(280), pace: 'slow' });
      answerWith('b'.repeat(140));
      await chooseRecording();
      await click(toggle());

      expect(seconds().value).toBe('20');
      type('ws-script-helper-seconds', '10');
      await settle();
      await click(run());

      expect(lastBody()).toMatchObject({ settings: { actions: ['fit'], seconds: 10 } });
      expect((lastBody().settings as Record<string, unknown>).pace).toBeUndefined();
    });

    it('refuses a time the server would, with the run button off', async () => {
      await mountScript();
      await click(toggle());
      await click(action('fit'));
      expect(seconds()).not.toBeNull();
      for (const bad of ['', '2', '601', '12.5']) {
        type('ws-script-helper-seconds', bad);
        await settle();
        expect(run()?.disabled).toBe(true);
        expect(panel()?.querySelector('[role="alert"]')?.textContent).toContain('3 to 600 seconds');
      }
      type('ws-script-helper-seconds', '30');
      await settle();
      expect(run()?.disabled).toBe(false);
    });

    it('leaves Add delivery out for a voice from a recording and says why', async () => {
      await mountScript({ recording: true });
      await click(toggle());

      expect(action('delivery')).toBeNull();
      expect(action('fit')).not.toBeNull();
      expect(picked()).toEqual(['fit']);
      expect(seconds()).not.toBeNull();
      expect(panel()?.querySelector('[data-testid="script-helper-recording"]')?.textContent).toContain('reads as recorded');
      expect(run()?.textContent?.trim()).toBe('Fit to time');
    });

    it('says an empty script has nothing to work on and points to Describe it, with nothing to run', async () => {
      await mountScript({ text: '' });
      await click(toggle());

      expect(panel()?.textContent).toContain('Nothing to work on yet');
      expect(panel()?.textContent).toContain('Describe it');
      expect(run()).toBeNull();
      expect([...(panel()?.querySelectorAll('button') ?? [])].map((button) => button.textContent?.trim())).toEqual(['Fill in from Describe it']);
    });

    it('treats a script of only marks and pauses as empty', async () => {
      await mountScript({ text: '[sad] … [as set]' });
      await click(toggle());

      expect(panel()?.textContent).toContain('Nothing to work on yet');
    });

    it('keeps the own text, adds each result as a version and steps between them, sending the one shown', async () => {
      await mountScript();
      answerWith(MARKED, SHORT);
      await addDelivery();
      await click(toggle());
      await click(action('fit'));
      await click(action('delivery'));
      await click(run());

      expect(written().history.map((entry) => entry.text)).toEqual([MARKED, SHORT]);
      expect(counter()).toBe('2 of 2');
      expect(target.querySelector('[aria-label="Previous version"]')).not.toBeNull();

      (target.querySelector('[aria-label="Previous version"]') as HTMLButtonElement).click();
      await settle();

      expect(counter()).toBe('1 of 2');
      expect(field('ws-enhanced-prompt').value).toBe(MARKED);
      expect((await submitForm(target)).get('prompt')).toBe(MARKED);
      expect(draft.state.speechText).toBe(SCRIPT);
    });

    it('always starts from your Text, whichever tab is shown, and adds a version each run', async () => {
      await mountScript();
      answerWith(MARKED, SHORT, 'Third.');
      await addDelivery();
      expect(tab('written').getAttribute('aria-selected')).toBe('true');
      await addDelivery();
      expect((enhanceApiMocks.enhancePrompt.mock.calls.at(-1)?.[0] as { prompt: string }).prompt).toBe(SCRIPT);

      tab('text').click();
      await settle();
      await addDelivery();

      expect((enhanceApiMocks.enhancePrompt.mock.calls.at(-1)?.[0] as { prompt: string }).prompt).toBe(SCRIPT);
      expect(written().history).toHaveLength(3);
      expect(draft.state.speechText).toBe(SCRIPT);
    });

    it('sends the Written text as the prompt with the own text and the record of what the helper did', async () => {
      await mountScript();
      answerWith(MARKED);
      await addDelivery();

      const sent = Object.fromEntries((await submitForm(target)).entries());

      expect(sent).toMatchObject({ prompt: MARKED, prompt_input: SCRIPT, model: 'qwen3_tts' });
      expect(JSON.parse(sent.enhance_record as string)).toEqual({ settings: { actions: ['delivery'] }, model: 'owner/llm', format: 'text', task: 'script' });
    });

    it('records the time and pace of a fit', async () => {
      await mountScript({ pace: 'fast' });
      answerWith(SHORT);
      await click(toggle());
      await click(action('delivery'));
      await click(action('fit'));
      type('ws-script-helper-seconds', '20');
      await settle();
      await click(run());

      const sent = await submitForm(target);

      expect(JSON.parse(sent.get('enhance_record') as string)).toEqual({ settings: { actions: ['fit'], seconds: 20, pace: 'fast' }, model: 'owner/llm', format: 'text', task: 'script' });
    });

    it('counts the characters and the time of the text Generate uses, marks excluded', async () => {
      await mountScript();
      answerWith(`[sad] ${'x'.repeat(279)}`);
      await addDelivery();

      // The mark and the space after it are not read: 279 characters plus the space at the mark's end are 280 at 14 a second.
      expect(target.querySelector('[data-testid="text-count"]')?.textContent).toBe('280 / 10,000 characters · about 0:20');
      expect(descriptionOf(tab('written'))).toBe('sent by Generate');
    });

    it('scratches the Written text back to the own text, keeping the version to bring back', async () => {
      await mountScript();
      answerWith(MARKED);
      await addDelivery();

      scratchButton()!.click();
      await settle();

      expect(tab('text').getAttribute('aria-selected')).toBe('true');
      expect(document.activeElement).toBe(tab('text'));
      expect(written()).toMatchObject({ text: '', from: null });
      expect(written().history).toHaveLength(1);
      expect(mark()).toBeNull();
      expect(Object.fromEntries((await submitForm(target)).entries())).toMatchObject({ prompt: SCRIPT });
      tab('written').click();
      await settle();
      expect(counter()).toBe('– of 1');
    });

    it('marks the Written tab out of date on the first keystroke in your Text, with ⚠ on the Script helper and no re-enhance', async () => {
      await mountScript();
      answerWith(MARKED);
      await addDelivery();
      tab('text').click();
      await settle();
      expect(mark()).toBe('dot');
      expect(toggle()?.getAttribute('aria-label')).toBeNull();

      await typeInto('ws-prompt', `${SCRIPT} Again.`);

      expect(mark()).toBe('warn');
      expect(descriptionOf(tab('written'))).toBe('sent by Generate, out of date');
      expect(toggle()?.getAttribute('aria-label')).toBe('Script helper (out of date)');
      expect(document.querySelector('#ws-enhance-toggle')).toBeNull();
      expect((await submitForm(target)).get('prompt')).toBe(MARKED);
    });

    it('shows the enhancer\'s notice about a result and clears it when another version is shown', async () => {
      await mountScript();
      answerWith(MARKED, { text: SHORT, warning: 'Came out at about 1:40.' });
      await addDelivery();
      expect(target.querySelector('[data-testid="script-note"]')).toBeNull();
      await click(toggle());
      await click(action('fit'));
      await click(action('delivery'));
      await click(run());

      expect(target.querySelector('[data-testid="script-note"]')?.textContent).toBe('Came out at about 1:40.');

      (target.querySelector('[aria-label="Previous version"]') as HTMLButtonElement).click();
      await settle();

      expect(target.querySelector('[data-testid="script-note"]')).toBeNull();
    });

    it('inserts into the Written text while that tab is shown', async () => {
      await mountScript();
      answerWith(MARKED);
      await addDelivery();
      const box = field('ws-enhanced-prompt');
      box.focus();
      box.setSelectionRange(MARKED.length, MARKED.length);
      await click(document.getElementById('ws-insert'));
      await click(document.querySelector('[data-action="insert-laugh"]'));

      expect(written().text).toBe(`${MARKED} haha`);
      expect(draft.state.speechText).toBe(SCRIPT);
      expect(written().from?.record).toBeUndefined();
    });

    it('applies nothing when the script changed while the helper worked', async () => {
      await mountScript();
      answerWith({ text: MARKED, during: () => draft.update('speechText', `${SCRIPT} More.`) });
      await addDelivery();

      expect(written().text).toBe('');
      expect(panel()?.textContent).toContain('Your script changed while the helper was working');
    });

    it('says a run that changed nothing as a neutral note, not an error', async () => {
      await mountScript();
      enhanceApiMocks.enhancePrompt.mockImplementation(async (_body: unknown, ...rest: unknown[]) => {
        (rest[0] as (frame: AudioEnhanceFrame) => void)({ type: 'error', detail: 'No changes: the helper left your script as it is.', no_change: true });
      });
      await addDelivery();

      expect(panel()?.querySelector('[data-testid="script-helper-note"]')?.textContent).toBe('No changes: the helper left your script as it is.');
      expect(panel()?.querySelector('[role="alert"]')).toBeNull();
      expect(written().text).toBe('');
    });

    it('shows the server\'s refusal in the panel and leaves the text alone', async () => {
      await mountScript();
      enhanceApiMocks.enhancePrompt.mockImplementation(async (_body: unknown, ...rest: unknown[]) => {
        (rest[0] as (frame: AudioEnhanceFrame) => void)({ type: 'error', detail: "Couldn't add delivery without changing your words." });
      });
      await addDelivery();

      expect(panel()?.querySelector('[role="alert"]')?.textContent).toBe("Couldn't add delivery without changing your words.");
      expect(written().text).toBe('');
    });

    it('is not available while a take is made', async () => {
      await mountScript({ active: true });
      await click(toggle());

      expect(run()?.disabled).toBe(true);
      expect(panel()?.textContent).toContain('Available when the take is done');
    });

    it('is not offered for Music or Sound', async () => {
      await mountScript();
      await pickKind('sound');
      expect(toggle()).toBeNull();
      await pickKind('music');
      expect(toggle()).toBeNull();
    });

    describe('on a tablet and a phone', () => {
      beforeEach(() => {
        Element.prototype.animate = vi.fn(() => ({ cancel: vi.fn(), finished: Promise.resolve(), onfinish: null })) as unknown as Element['animate'];
      });
      afterEach(() => {
        delete (Element.prototype as Partial<Element>).animate;
      });

      it.each([
        ['a phone', { compact: true }],
        ['a tablet', { tall: true }],
      ])('opens as a sheet with the script on top and the run button in its foot on %s', async (_name, layoutOptions) => {
        await mountScript(layoutOptions);
        answerWith(MARKED);
        await click(toggle());

        const sheet = helperSheet();
        expect(sheet.querySelector('[data-testid="script-helper-quote"]')?.textContent).toBe(SCRIPT);
        expect(sheet.querySelector('.sheet-foot #ws-script-helper-run')).not.toBeNull();
        expect(sheet.querySelector('.sheet-title')?.textContent).toBe('Script helper');
        await click(run());

        expect(lastBody()).toMatchObject({ settings: { actions: ['delivery'] } });
        expect(field('ws-enhanced-prompt').value).toBe(MARKED);
      });

      it('explains an empty script in the sheet', async () => {
        await mountScript({ compact: true, text: '' });
        await click(toggle());

        expect(helperSheet().textContent).toContain('Nothing to work on yet');
        expect(run()).toBeNull();
      });
    });
  });

  describe('the text boxes after a walk', () => {
    const MODELS = [...AUDIO_MODELS, SA3_MUSIC];
    const NO_REWRITE = 'The enhancer returned no usable rewrite. Try again.';
    const SCRIPT = 'I never told anyone. Not even you. The door was open.';
    const SONG = '[Verse]\nThe porch light flickers, the street is wet\n\n[Chorus]\nCome back home\n';

    async function mountBox(kind: Kind, patch: Partial<DraftState> = {}, screen: { compact?: boolean; tall?: boolean } = {}): Promise<void> {
      const context = withAudioEnhancer(withAudio(makeContext({ history_assets: [] }), MODELS));
      layoutProbe.compact = screen.compact ?? false;
      layoutProbe.tall = screen.tall ?? false;
      galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage([]));
      draft.reset();
      draft.patch({ workflow: kind, ...patch });
      app = await mountWorkspacePage(context, target);
    }

    /** Make each run write *text* in the field the task writes, and finish. */
    function writes(task: 'sound' | 'style' | 'script', field: 'prompt' | 'style' | 'script', text: string, warning?: string): void {
      enhanceApiMocks.enhancePrompt.mockImplementation(async (_body: unknown, ...rest: unknown[]) => {
        const onFrame = rest[0] as (frame: AudioEnhanceFrame) => void;
        onFrame({ type: 'done', task, fields: { [field]: text }, clamped: false, model: 'owner/llm', ...(warning ? { warning } : {}) });
      });
    }

    /** Make each run wait until released (or aborted), as a slow enhancer does. */
    function holdRun(): { release: () => void; readonly aborted: boolean } {
      const held = { aborted: false, release: () => {} };
      enhanceApiMocks.enhancePrompt.mockImplementation(async (_body: unknown, ...rest: unknown[]) => {
        (rest[1] as AbortSignal).addEventListener('abort', () => { held.aborted = true; held.release(); });
        await new Promise<void>((resolve) => { held.release = resolve; });
      });
      return held;
    }

    const el = (selector: string) => document.querySelector(selector) as HTMLButtonElement | null;
    const click = async (node: Element | null) => {
      (node as HTMLElement).click();
      await settle();
    };
    const generateButton = () => target.querySelector('#ws-submit') as HTMLButtonElement;
    const composerReason = () => target.querySelector('[data-testid="audio-reason"]')?.textContent?.trim() ?? null;
    const BOXES = {
      sound: { enhance: 'sound', field: 'prompt', patch: { soundPrompt: 'rain' }, own: '#ws-prompt-tab', enhanced: '#ws-enhanced-prompt-tab', box: 'sound' },
      music: { enhance: 'style', field: 'style', patch: { audioStyle: 'warm folk' }, own: '#ws-audio-style-tab', enhanced: '#ws-enhanced-audio-style-tab', box: 'style' },
    } as const;

    describe('Speech', () => {
      it('gives the Script helper note a line of its own, apart from the character count', async () => {
        await mountBox('speech', { speechText: SCRIPT });
        writes('script', 'script', 'I never told anyone. The door was open.', 'Came out at about 1:40.');
        await click(el('#ws-script-helper-toggle'));
        await click(el('#ws-script-helper-run'));

        const note = target.querySelector('[data-testid="script-note"]')!;
        const count = target.querySelector('[data-testid="text-count"]')!;
        expect(note.textContent).toBe('Came out at about 1:40.');
        expect(count.textContent).toContain('characters');
        expect(count.parentElement?.contains(note)).toBe(false);
        expect(note.parentElement).not.toBe(count.parentElement);
      });
    });

    describe.each(Object.entries(BOXES))('%s Enhance', (_name, box) => {
      const tab = (which: 'own' | 'enhanced') => el(box[which]) as HTMLButtonElement;
      const rewrites = () => draft.state.audioRewrites[box.box];

      it('returns to your own text after a failed run, shows the error under the box and leaves the versions as they were', async () => {
        await mountBox(_name as Kind, box.patch);
        writes(box.enhance, box.field, 'Heavy version.');
        await click(el('#ws-enhance-toggle'));
        await click(el('#ws-enhance-run'));
        expect(tab('enhanced').getAttribute('aria-selected')).toBe('true');
        const before = rewrites();
        enhanceApiMocks.enhancePrompt.mockImplementation(async (_body: unknown, ...rest: unknown[]) => {
          (rest[0] as (frame: AudioEnhanceFrame) => void)({ type: 'error', detail: NO_REWRITE });
        });

        await click(el('#ws-enhance-toggle'));
        await click(el('#ws-enhance-run'));

        expect(tab('own').getAttribute('aria-selected')).toBe('true');
        expect(tab('enhanced').getAttribute('aria-selected')).toBe('false');
        const help = target.querySelector('[data-testid="help-line"]')!;
        expect(help.textContent).toBe(NO_REWRITE);
        expect(help.getAttribute('role')).toBe('alert');
        expect(rewrites()).toEqual(before);
      });

      it('returns to your own text, with the error, when the request itself fails and nothing was written before', async () => {
        await mountBox(_name as Kind, box.patch);
        enhanceApiMocks.enhancePrompt.mockRejectedValue(new Error('The enhancer is not reachable.'));

        await click(el('#ws-enhance-toggle'));
        await click(el('#ws-enhance-run'));

        expect(tab('own').getAttribute('aria-selected')).toBe('true');
        expect(target.querySelector('[data-testid="help-line"]')?.textContent).toBe('The enhancer is not reachable.');
        expect(rewrites().history).toEqual([]);
      });

      it('stays on the Enhanced text after a run that wrote it, and after a stop', async () => {
        await mountBox(_name as Kind, box.patch);
        writes(box.enhance, box.field, 'Heavy version.');
        await click(el('#ws-enhance-toggle'));
        await click(el('#ws-enhance-run'));
        expect(tab('enhanced').getAttribute('aria-selected')).toBe('true');

        const held = holdRun();
        await click(el('#ws-enhance-toggle'));
        await click(el('#ws-enhance-run'));
        (target.querySelector('[aria-label="Stop enhancing"]') as HTMLButtonElement).click();
        await settle();

        expect(held.aborted).toBe(true);
        expect(tab('enhanced').getAttribute('aria-selected')).toBe('true');
        expect(target.querySelector('[data-testid="help-line"]')?.getAttribute('role')).not.toBe('alert');
      });

      it('closes the panel as the run starts and puts the focus on the Enhance button', async () => {
        await mountBox(_name as Kind, box.patch);
        holdRun();
        await click(el('#ws-enhance-toggle'));
        el('#ws-enhance-run')!.focus();

        await click(el('#ws-enhance-run'));

        expect(document.querySelector('#ws-enhance-panel')).toBeNull();
        expect(document.activeElement).toBe(el('#ws-enhance-toggle'));
      });

      it.each([['a tablet', { tall: true }]])('closes the sheet as the run starts on %s and keeps the focus on the Enhance button', async (_screen, screen) => {
        await mountBox(_name as Kind, box.patch, screen);
        holdRun();
        await click(el('#ws-enhance-toggle'));
        el('#ws-enhance-run')!.focus();

        await click(el('#ws-enhance-run'));

        expect(document.activeElement).toBe(el('#ws-enhance-toggle'));
      });

      it.each([['a desktop', {}], ['a tablet', { tall: true }]])('puts the focus on the Enhance button after the panel\'s own Stop on %s', async (_screen, screen) => {
        await mountBox(_name as Kind, box.patch, screen);
        const held = holdRun();
        await click(el('#ws-enhance-toggle'));
        await click(el('#ws-enhance-run'));
        await click(el('#ws-enhance-toggle'));
        const stopButton = () => document.querySelector('#ws-enhance-panel .ui-btn-danger, [data-testid="sheet"] .ui-btn-danger') as HTMLButtonElement | null;
        const stop = stopButton()!;
        expect(stop.textContent?.trim()).toBe('Stop');
        stop.focus();

        await click(stop);

        expect(held.aborted).toBe(true);
        expect(stopButton()).toBeNull();
        expect(document.activeElement).toBe(el('#ws-enhance-toggle'));
      });

      it('says in the panel, beside "Enhance again", why the Enhanced text is out of date, and says nothing while it is current', async () => {
        await mountBox(_name as Kind, box.patch);
        writes(box.enhance, box.field, 'Heavy version.');
        await click(el('#ws-enhance-toggle'));
        await click(el('#ws-enhance-run'));
        const stale = () => document.querySelector('#ws-enhance-panel [data-testid="enhance-stale"]');
        await click(el('#ws-enhance-toggle'));
        expect(stale()).toBeNull();
        document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
        await settle();

        await typeInto(box.own === '#ws-prompt-tab' ? 'ws-prompt' : 'ws-audio-style', `${Object.values(box.patch)[0]}, more`);
        await click(el('#ws-enhance-toggle'));

        const label = _name === 'sound' ? 'Prompt' : 'Style';
        expect(stale()?.textContent).toBe(`Your ${label} changed since this was written.`);
        expect(el('#ws-enhance-run')!.textContent?.trim()).toBe('Enhance again');
        expect(el('#ws-enhance-run')!.parentElement?.contains(stale())).toBe(true);
        document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
        await settle();

        draft.patch({ audioRewrites: { ...draft.state.audioRewrites, [box.box]: { ...rewrites(), from: { ...rewrites().from!, input: Object.values(box.patch)[0] as string, model: 'other-model' } } } });
        await settle();
        await click(el('#ws-enhance-toggle'));
        expect(stale()?.textContent).toBe(`Your ${label} and the model changed since this was written.`);
      });

      it('keeps Generate off, with the reason in view, while Enhance runs, and on again once it ends', async () => {
        await mountBox(_name as Kind, box.patch);
        const held = holdRun();
        expect(generateButton().disabled).toBe(false);
        await click(el('#ws-enhance-toggle'));
        await click(el('#ws-enhance-run'));

        expect(generateButton().disabled).toBe(true);
        expect(reason()).toBe("Enhancing… Generate when it's done.");
        expect(generateButton().getAttribute('aria-describedby')).toBe('ws-generate-reason');
        await submitForm(target);
        expect(workspaceApiMocks.submitGenerate).not.toHaveBeenCalled();

        held.release();
        await settle();

        expect(generateButton().disabled).toBe(false);
        expect(reason()).toBeNull();
      });

      it('keeps Generate on the phone off and says why while Enhance runs', async () => {
        await mountBox(_name as Kind, box.patch, { compact: true });
        const held = holdRun();
        await click(el('#ws-enhance-toggle'));
        await click(el('#ws-enhance-run'));

        expect((target.querySelector('#ws-submit-compact') as HTMLButtonElement).disabled).toBe(true);
        expect(composerReason()).toBe("Enhancing… Generate when it's done.");

        await click(target.querySelector('[aria-label="Stop enhancing"]'));
        held.release();
        await settle();

        expect((target.querySelector('#ws-submit-compact') as HTMLButtonElement).disabled).toBe(false);
        expect(composerReason()).toBeNull();
      });
    });

    describe('Script helper', () => {
      const run = () => el('#ws-script-helper-run');
      const stop = () => el('#ws-script-helper-stop');

      it.each([['a tablet', { tall: true }], ['a phone', { compact: true }]])('keeps its sheet open on %s while writing: Escape, the backdrop and the opener do not hide it, only Stop ends it', async (_screen, screen) => {
        await mountBox('speech', { speechText: SCRIPT }, screen);
        const held = holdRun();
        await click(el('#ws-script-helper-toggle'));
        await click(run());
        const sheetRoot = () => document.querySelector('#ws-script-helper-panel')?.closest('[data-testid="sheet"]') as HTMLElement | null;
        expect(stop()).not.toBeNull();

        document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
        await settle();
        await click(sheetRoot()!.querySelector('.sheet-scrim'));
        await click(sheetRoot()!.querySelector('.sheet-head [aria-label="Close"]'));
        await click(el('#ws-script-helper-toggle'));

        expect(held.aborted).toBe(false);
        expect(sheetRoot()!.inert).toBe(false);
        expect(stop()).not.toBeNull();

        await click(stop());
        expect(held.aborted).toBe(true);
      });

      it('keeps writing when the page scrolls, the page is clicked, the Escape key is pressed or the opener is clicked again; only Stop ends it', async () => {
        await mountBox('speech', { speechText: SCRIPT });
        const held = holdRun();
        await click(el('#ws-script-helper-toggle'));
        await click(run());
        expect(stop()).not.toBeNull();

        document.dispatchEvent(new Event('scroll'));
        document.body.dispatchEvent(new Event('pointerdown', { bubbles: true }));
        document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
        await settle();
        await click(el('#ws-script-helper-toggle'));

        expect(held.aborted).toBe(false);
        expect(stop()).not.toBeNull();
        expect(document.activeElement).toBe(stop());
        expect(document.querySelector('#ws-script-helper-panel')).not.toBeNull();

        await click(stop());
        expect(held.aborted).toBe(true);
      });

      it.each([['a desktop', {}], ['a tablet', { tall: true }]])('keeps the focus in the panel from Run to Stop and back on %s', async (_screen, screen) => {
        await mountBox('speech', { speechText: SCRIPT }, screen);
        const held = holdRun();
        await click(el('#ws-script-helper-toggle'));
        run()!.focus();

        await click(run());

        expect(document.activeElement).toBe(stop());
        expect(stop()).not.toBeNull();

        await click(stop());

        expect(held.aborted).toBe(true);
        expect(document.activeElement).toBe(run());
      });

      it('puts the focus back on its button when a result closes the panel', async () => {
        await mountBox('speech', { speechText: SCRIPT });
        writes('script', 'script', 'I never told anyone. The door was open.');
        await click(el('#ws-script-helper-toggle'));
        run()!.focus();

        await click(run());

        expect(document.querySelector('#ws-script-helper-panel')).toBeNull();
        expect(document.activeElement).toBe(el('#ws-script-helper-toggle'));
      });

      it('keeps Generate off, with the reason, while it works', async () => {
        await mountBox('speech', { speechText: SCRIPT });
        const held = holdRun();
        await click(el('#ws-script-helper-toggle'));
        await click(run());

        expect(generateButton().disabled).toBe(true);
        expect(reason()).toBe("Writing… Generate when it's done.");

        await click(stop());
        expect(held.aborted).toBe(true);
        expect(generateButton().disabled).toBe(false);
        expect(reason()).toBeNull();
      });

      it('says in the panel why the Written text is out of date', async () => {
        await mountBox('speech', { speechText: SCRIPT });
        writes('script', 'script', 'I never told anyone. The door was open.');
        await click(el('#ws-script-helper-toggle'));
        await click(run());
        await click(el('#ws-script-helper-toggle'));
        expect(document.querySelector('[data-testid="script-helper-stale"]')).toBeNull();
        document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
        await settle();

        await typeInto('ws-prompt', `${SCRIPT} Again.`);
        await click(el('#ws-script-helper-toggle'));

        expect(document.querySelector('[data-testid="script-helper-stale"]')?.textContent).toBe('Your Text changed since this was written.');
        expect(run()!.parentElement?.contains(document.querySelector('[data-testid="script-helper-stale"]'))).toBe(true);
      });
    });

    describe('Lyrics helper', () => {
      const run = () => el('#ws-lyrics-helper-run');
      const stop = () => el('#ws-lyrics-helper-stop');

      it.each([['a tablet', { tall: true }], ['a phone', { compact: true }]])('keeps its sheet open on %s while writing: Escape, the backdrop and the opener do not hide it, only Stop ends it', async (_screen, screen) => {
        await mountBox('music', { lyrics: SONG, audioStyle: 'warm folk' }, screen);
        const held = holdRun();
        await click(el('#ws-lyrics-helper-toggle'));
        await click(run());
        const sheetRoot = () => document.querySelector('#ws-lyrics-helper-panel')?.closest('[data-testid="sheet"]') as HTMLElement | null;
        expect(stop()).not.toBeNull();

        document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
        await settle();
        await click(sheetRoot()!.querySelector('.sheet-scrim'));
        await click(sheetRoot()!.querySelector('.sheet-head [aria-label="Close"]'));
        await click(el('#ws-lyrics-helper-toggle'));

        expect(held.aborted).toBe(false);
        expect(sheetRoot()!.inert).toBe(false);
        expect(stop()).not.toBeNull();

        await click(stop());
        expect(held.aborted).toBe(true);
      });

      it('keeps writing when the page scrolls, the page is clicked, the Escape key is pressed or the opener is clicked again; only Stop ends it', async () => {
        await mountBox('music', { lyrics: SONG, audioStyle: 'warm folk' });
        const held = holdRun();
        await click(el('#ws-lyrics-helper-toggle'));
        await click(run());
        expect(stop()).not.toBeNull();

        document.dispatchEvent(new Event('scroll'));
        document.body.dispatchEvent(new Event('pointerdown', { bubbles: true }));
        document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
        await settle();
        await click(el('#ws-lyrics-helper-toggle'));

        expect(held.aborted).toBe(false);
        expect(stop()).not.toBeNull();
        expect(document.activeElement).toBe(stop());
        expect(document.querySelector('#ws-lyrics-helper-panel')).not.toBeNull();

        await click(stop());
        expect(held.aborted).toBe(true);
      });

      it.each([['a desktop', {}], ['a tablet', { tall: true }]])('keeps the focus in the panel from Run to Stop and back on %s', async (_screen, screen) => {
        await mountBox('music', { lyrics: SONG, audioStyle: 'warm folk' }, screen);
        const held = holdRun();
        await click(el('#ws-lyrics-helper-toggle'));
        run()!.focus();

        await click(run());

        expect(document.activeElement).toBe(stop());
        expect(stop()).not.toBeNull();

        await click(stop());

        expect(held.aborted).toBe(true);
        expect(document.activeElement).toBe(run());
      });

      it('keeps Generate off, with the reason, while it works', async () => {
        await mountBox('music', { lyrics: SONG, audioStyle: 'warm folk' });
        const held = holdRun();
        await click(el('#ws-lyrics-helper-toggle'));
        await click(run());

        expect(generateButton().disabled).toBe(true);
        expect(reason()).toBe("Writing… Generate when it's done.");

        held.release();
        await settle();
        expect(generateButton().disabled).toBe(false);
      });
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
      draft.setVoice({ path: '/voices/me.wav' });
      draft.update('voiceText', 'Hello, this is my voice.');
      await settle();

      (target.querySelector('button[aria-label="Copy CLI command"]') as HTMLButtonElement).click();
      await settle();
      expect(clipboard.mock.calls.at(-1)?.[0]).toBe("ziv audio --model qwen3_tts --kind speech --prompt 'Hello there.' --voice /voices/me.wav --voice-text 'Hello, this is my voice.' --output /tmp/output");

      await tick();
      (target.querySelector('button[aria-label="Copy CLI command"]') as HTMLButtonElement).click();
      await settle();
      expect(clipboard.mock.calls.at(-1)?.[0]).toBe("ziv audio --model qwen3_tts --kind speech --prompt 'Hello there.' --voice /voices/me.wav --voice-text 'Hello, this is my voice.' --voice-consent --output /tmp/output");
    });
  });

  describe('held uploads', () => {
    it('keeps the voice recording through a visit to the Gallery', async () => {
      await mountAudio('speech');
      await chooseRecording();
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

    it('lists Tempo and Key between Model and Seed for Music, and Length too for Stable Audio', async () => {
      const context = await mountAudio('music', { compact: true, models: [...AUDIO_MODELS, SA3_MUSIC] });
      expect(chips()).toEqual(['kind', 'model', 'tempo', 'key', 'seed', 'runs']);
      expect(target.querySelector('[data-chip="tempo"]')?.textContent).toContain('Auto');
      expect(target.querySelector('[data-chip="key"]')?.textContent).toContain('Auto');

      draft.hydrateFromContext(context, 'sa3');
      await settle();
      expect(chips()).toEqual(['kind', 'model', 'tempo', 'key', 'length', 'seed', 'runs']);
      expect(target.querySelector('[data-chip="length"]')?.textContent).toContain('30 s');
    });

    it('opens Tempo as a sheet with the number, a slider and Reset, and takes Auto back', async () => {
      await mountAudio('music', { compact: true });
      (target.querySelector('[data-chip="tempo"]') as HTMLButtonElement).click();
      await settle();
      const pop = () => target.querySelector('[data-testid="setting-popover"]') as HTMLElement;
      const reset = () => Array.from(pop().querySelectorAll<HTMLButtonElement>('button')).find((button) => button.textContent?.trim() === 'Reset')!;

      expect(pop().getAttribute('data-key')).toBe('tempo');
      expect(reset().disabled).toBe(true);
      const slider = pop().querySelector('input[type="range"]') as HTMLInputElement;
      slider.value = '96';
      slider.dispatchEvent(new Event('input', { bubbles: true }));
      await settle();

      expect(draft.state.audioTempo).toBe(96);
      expect(target.querySelector('[data-chip="tempo"]')?.textContent).toContain('96');
      expect(target.querySelector('[data-chip="tempo"]')?.getAttribute('data-changed')).toBe('true');
      reset().click();
      await settle();
      expect(draft.state.audioTempo).toBeNull();
    });

    it('opens Key as a sheet with the same two-column list and Auto', async () => {
      await mountAudio('music', { compact: true });
      (target.querySelector('[data-chip="key"]') as HTMLButtonElement).click();
      await settle();
      const box = target.querySelector('[data-testid="pop-key-select"] button') as HTMLElement;

      pickOption(box, 'Bbm');
      await settle();

      expect(draft.state.audioKey).toBe('Bbm');
      expect(target.querySelector('[data-chip="key"]')?.textContent).toContain('B♭ min');
    });

    it('counts a changed tempo and key in All', async () => {
      await mountAudio('music', { compact: true });
      draft.patch({ audioTempo: 90, audioKey: 'C' });
      await settle();

      expect(target.querySelector('.setchip-n')?.textContent).toBe('2');
    });

    it('sets a song\'s Length from its sheet on Stable Audio', async () => {
      const context = await mountAudio('music', { compact: true, models: [...AUDIO_MODELS, SA3_MUSIC] });
      draft.hydrateFromContext(context, 'sa3');
      await settle();
      (target.querySelector('[data-chip="length"]') as HTMLButtonElement).click();
      await settle();
      const slider = target.querySelector('[data-testid="setting-popover"] input[type="range"]') as HTMLInputElement;
      slider.value = '75';
      slider.dispatchEvent(new Event('input', { bubbles: true }));
      await settle();

      expect(draft.state).toMatchObject({ musicSeconds: 75, audioSeconds: 0 });
      expect(target.querySelector('[data-chip="length"]')?.textContent).toContain('75 s');
    });

    it('lists Delivery after Voice for Speech, with the Vibe or Natural as its value, and hides it for a recording', async () => {
      await mountAudio('speech', { compact: true });
      expect(chips()).toEqual(['kind', 'model', 'voice', 'delivery', 'seed', 'runs']);
      expect(target.querySelector('[data-chip="delivery"]')?.textContent).toContain('Natural');

      draft.update('speechDelivery', { emotion: 'enthusiastic', manner: 'natural', pace: 'fast', pitch: 'high', expressiveness: 'high' });
      await settle();
      expect(target.querySelector('[data-chip="delivery"]')?.textContent).toContain('Excited host');
      expect(target.querySelector('[data-chip="delivery"]')?.getAttribute('data-changed')).toBe('true');

      draft.update('voiceMode', 'recording');
      await settle();
      expect(chips()).toEqual(['kind', 'model', 'voice', 'seed', 'runs']);
    });

    it('lists Type and Length between Model and Seed for Sound, Length on Auto with the number its type will use', async () => {
      await mountAudio('sound', { compact: true });
      expect(chips()).toEqual(['kind', 'model', 'soundType', 'length', 'seed', 'runs']);
      expect(target.querySelector('[data-chip="soundType"]')?.textContent).toContain('Auto');
      expect(target.querySelector('[data-chip="length"]')?.textContent).toContain('auto ≈ 12 s');

      draft.patch({ soundType: 'one-shot' });
      await settle();
      expect(target.querySelector('[data-chip="soundType"]')?.textContent).toContain('One-shot');
      expect(target.querySelector('[data-chip="length"]')?.textContent).toContain('auto ≈ 2 s');
    });

    it('opens Type as a sheet of the four types, and Reset puts it back to Auto', async () => {
      await mountAudio('sound', { compact: true });
      (target.querySelector('[data-chip="soundType"]') as HTMLButtonElement).click();
      await settle();
      const pop = () => target.querySelector('[data-testid="setting-popover"]') as HTMLElement;
      const reset = () => Array.from(pop().querySelectorAll<HTMLButtonElement>('button')).find((button) => button.textContent?.trim() === 'Reset')!;

      expect(pop().getAttribute('data-key')).toBe('soundType');
      expect(reset().disabled).toBe(true);
      pop().querySelector<HTMLButtonElement>('[data-testid="pop-sound-type"] [data-value="effect"]')!.click();
      await settle();

      expect(draft.state.soundType).toBe('effect');
      expect(target.querySelector('[data-chip="soundType"]')?.getAttribute('data-changed')).toBe('true');
      reset().click();
      await settle();
      expect(draft.state.soundType).toBe('auto');
    });

    it('opens Length with its field blank on Auto, locks a number from the slider and Reset goes back to Auto', async () => {
      await mountAudio('sound', { compact: true });
      draft.patch({ soundType: 'effect' });
      await settle();
      (target.querySelector('[data-chip="length"]') as HTMLButtonElement).click();
      await settle();
      const pop = () => target.querySelector('[data-testid="setting-popover"]') as HTMLElement;
      const input = pop().querySelector('input[type="number"]') as HTMLInputElement;
      const slider = pop().querySelector('input[type="range"]') as HTMLInputElement;

      expect(input.value).toBe('');
      expect(input.placeholder).toBe('auto · ≈ 5 s');
      expect(slider.value).toBe('5');
      slider.value = '20';
      slider.dispatchEvent(new Event('input', { bubbles: true }));
      await settle();

      expect(draft.state.audioSeconds).toBe(20);
      expect(target.querySelector('[data-chip="length"]')?.textContent).toContain('20 s');
      expect(target.querySelector('[data-chip="length"]')?.getAttribute('data-changed')).toBe('true');
      Array.from(pop().querySelectorAll<HTMLButtonElement>('button')).find((button) => button.textContent?.trim() === 'Reset')!.click();
      await settle();
      expect(draft.state.audioSeconds).toBe(0);
      expect(target.querySelector('[data-chip="length"]')?.textContent).toContain('auto ≈ 5 s');
    });

    it('counts a changed type and a locked length in All', async () => {
      await mountAudio('sound', { compact: true });
      draft.patch({ soundType: 'effect', audioSeconds: 8 });
      await settle();

      expect(target.querySelector('.setchip-n')?.textContent).toBe('2');
    });

    it('opens Delivery as a sheet with the Vibes and the five rows, and Reset puts them back', async () => {
      await mountAudio('speech', { compact: true });
      (target.querySelector('[data-chip="delivery"]') as HTMLButtonElement).click();
      await settle();
      const pop = () => target.querySelector('[data-testid="setting-popover"]') as HTMLElement;
      const reset = () => Array.from(pop().querySelectorAll<HTMLButtonElement>('button')).find((button) => button.textContent?.trim() === 'Reset')!;

      expect(pop().getAttribute('data-key')).toBe('delivery');
      expect(pop().querySelectorAll('.vibe')).toHaveLength(6);
      expect(Array.from(pop().querySelectorAll('.inspector-label-text')).map((node) => node.textContent)).toEqual(['Emotion', 'Manner', 'Pace', 'Pitch', 'Expressiveness']);
      expect(reset().disabled).toBe(true);

      (pop().querySelector('[data-vibe="storyteller"]') as HTMLButtonElement).click();
      await settle();
      expect(draft.state.speechDelivery).toMatchObject({ emotion: 'content', expressiveness: 'high' });
      expect(target.querySelector('[data-chip="delivery"]')?.textContent).toContain('Storyteller');

      (pop().querySelector('[data-testid="pop-delivery-pitch"] [data-value="low"]') as HTMLButtonElement).click();
      await settle();
      expect(target.querySelector('[data-chip="delivery"]')?.textContent).toContain('Custom');

      reset().click();
      await settle();
      expect(draft.state.speechDelivery).toEqual({ emotion: 'neutral', manner: 'natural', pace: 'normal', pitch: 'normal', expressiveness: 'normal' });
      expect(target.querySelector('[data-chip="delivery"]')?.textContent).toContain('Natural');
    });

    it('counts a changed Delivery in All', async () => {
      await mountAudio('speech', { compact: true });
      draft.update('speechDelivery', { emotion: 'sad', manner: 'natural', pace: 'normal', pitch: 'normal', expressiveness: 'normal' });
      await settle();

      expect(target.querySelector('.setchip-n')?.textContent).toBe('1');
    });

    it('opens the Voice popover with What it says and the tick', async () => {
      await mountAudio('speech', { compact: true });
      (target.querySelector('[data-chip="voice"]') as HTMLButtonElement).click();
      await settle();
      const pop = () => target.querySelector('[data-testid="setting-popover"]') as HTMLElement;

      expect(pop().querySelector('#ws-pop-voice-text')).toBeNull();
      draft.setVoice({ path: '/voices/me.wav' });
      await settle();
      expect(pop().querySelector('#ws-pop-voice-text')).not.toBeNull();
      expect(pop().querySelector('input[type="checkbox"]')).not.toBeNull();
      expect(pop().querySelector('input[type="checkbox"]')).toHaveProperty('checked', false);
    });

    it('warns on the Voice chip and opens its popover from the reason while the voice holds Generate back', async () => {
      await mountAudio('speech', { compact: true });
      draft.patch({ speechText: 'Hello there.', voiceMode: 'recording', voicePath: '/voices/me.wav' });
      await settle();

      expect(target.querySelector('[data-testid="audio-reason"]')?.textContent).toContain('Type what the recording says.');
      expect(target.querySelector('[data-chip="voice"]')?.classList.contains('warn')).toBe(true);
      (target.querySelector('[data-testid="audio-reason-voice"]') as HTMLButtonElement).click();
      await settle();

      expect(target.querySelector('[data-testid="setting-popover"]')?.getAttribute('data-key')).toBe('voice');
    });

    it('offers no Voice button for a reason of the text', async () => {
      await mountAudio('speech', { compact: true });

      expect(target.querySelector('[data-testid="audio-reason"]')).not.toBeNull();
      expect(target.querySelector('[data-testid="audio-reason-voice"]')).toBeNull();
    });

    it('puts the voice back with the popover\'s Reset, counted in the "All" settings', async () => {
      await mountAudio('speech', { compact: true });
      draft.setVoice({ path: '/voices/me.wav' });
      draft.update('voiceText', 'Hi');
      voiceConsent.ticked = true;
      await settle();
      (target.querySelector('[data-chip="voice"]') as HTMLButtonElement).click();
      await settle();

      const reset = Array.from(target.querySelectorAll<HTMLButtonElement>('[data-testid="setting-popover"] button')).find((button) => button.textContent?.trim() === 'Reset')!;
      expect(reset.disabled).toBe(false);
      reset.click();
      await settle();

      expect(draft.state).toMatchObject({ voiceMode: 'builtin', voicePath: null, voiceUpload: null, voiceText: '' });
      expect(voiceConsent.ticked).toBe(false);
    });

    it('reads the Audio text in the folded composer even when a prompt file was chosen on Image', async () => {
      await mountAudio('sound', { compact: true });
      draft.patch({ promptSource: 'file', promptFilePath: '/prompts/set.yaml', soundPrompt: 'Rain on a tin roof' });
      await settle();

      const line = target.querySelector('.composer-summary')?.textContent ?? '';
      expect(line).toContain('Rain on a tin roof');
      expect(line).not.toContain('set.yaml');
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

    it('opens the dialog that saves a take as a voice, over the page and without changing the draft', async () => {
      const take = makeSpeechAsset({ prompt: 'Welcome back to the workshop.' });
      await mountAudio('music', { assets: [take] });
      (target.querySelector('.latest-region button[aria-label="More actions for speech.flac"]') as HTMLButtonElement).click();
      flushSync();
      (document.querySelector('[role="menu"] [data-action="use-save-voice"]') as HTMLElement).click();
      await settle();

      expect(document.querySelector('[role="dialog"][aria-label="Save as voice"]')).not.toBeNull();
      expect((document.querySelector('#sv-text') as HTMLTextAreaElement).value).toBe('Welcome back to the workshop.');
      expect(draft.state.workflow).toBe('music');
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
      draft.patch({ audioStyle: 'a style', lyrics: 'la', audioSeconds: 60, soundType: 'effect', runs: 3, seed: 5 });
      await settle();

      (target.querySelector('#ws-reset') as HTMLButtonElement).click();
      await settle();

      expect(draft.state).toMatchObject({ audioStyle: 'a style', lyrics: 'la', soundPrompt: 'rain', audioSeconds: 0, soundType: 'auto', runs: 1, seed: null, audioKind: 'sound', workflow: 'sound', model: 'sa3' });
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

describe('Clear and Undo in the Audio boxes', () => {
  const FROM = { input: 'folk', model: 'yue2', task: 'style' as const };
  const REWRITTEN = { text: 'Warm folk ballad, fingerpicked guitar.', from: FROM };
  const SONG = '[Verse]\nI pack the boat at dawn\n\n[Chorus]\nGoodbye to the coast';
  const clearOf = (what: string) => target.querySelector(`[aria-label="Clear the ${what}"]`) as HTMLButtonElement | null;
  const undoOf = () => target.querySelector('[data-testid="clear-undo"] button') as HTMLButtonElement | null;
  const click = async (node: Element | null) => { (node as HTMLElement).click(); await settle(); };
  const rewrites = (box: 'style' | 'sound' | 'text', value: AudioRewrites): Partial<DraftState> => ({ audioRewrites: { ...draft.state.audioRewrites, [box]: value } });

  it('is not there while the Style box is empty, and sits first in its right group beside the recent prompts of a model without lyrics', async () => {
    await mountAudio('music');
    expect(clearOf('style')).toBeNull();

    await typeInto('ws-audio-style', 'folk');

    expect(clearOf('style')).not.toBeNull();
    expect(clearOf('lyrics')).toBeNull();
  });

  it('empties the Style, its Enhanced text and its versions and keeps the Lyrics, and Undo puts back both texts, the versions and the Enhanced view', async () => {
    await mountAudio('music');
    draft.patch({ audioStyle: 'folk', lyrics: SONG, ...rewrites('style', { ...REWRITTEN, history: [REWRITTEN] }) });
    await settle();
    await click(target.querySelector('#ws-enhanced-audio-style-tab'));
    expect(target.querySelector('#ws-enhanced-audio-style-tab')!.getAttribute('aria-selected')).toBe('true');

    await click(clearOf('style'));

    expect(draft.state.audioStyle).toBe('');
    expect(draft.state.audioRewrites.style).toEqual({ text: '', from: null, history: [] });
    expect(draft.state.lyrics).toBe(SONG);
    expect(target.querySelector('#ws-audio-style-tab')!.getAttribute('aria-selected')).toBe('true');
    expect(clearOf('style')).toBeNull();
    expect(target.querySelector('[data-testid="clear-undo"]')!.textContent?.trim()).toBe('Cleared. Undo');
    expect(document.activeElement).toBe(field('ws-audio-style'));

    await click(undoOf());

    expect(draft.state.audioStyle).toBe('folk');
    expect(draft.state.audioRewrites.style).toEqual({ ...REWRITTEN, history: [REWRITTEN] });
    expect(target.querySelector('#ws-enhanced-audio-style-tab')!.getAttribute('aria-selected')).toBe('true');
    expect(undoOf()).toBeNull();
  });

  it('keeps Undo until the next edit in the box, and a second Clear starts over', async () => {
    await mountAudio('music');
    draft.patch({ audioStyle: 'folk' });
    await settle();
    await click(clearOf('style'));
    expect(undoOf()).not.toBeNull();

    await typeInto('ws-audio-style', 'jazz');

    expect(undoOf()).toBeNull();
    await click(clearOf('style'));
    await click(undoOf());
    expect(draft.state.audioStyle).toBe('jazz');
  });

  it('clears a Sound prompt and its Enhanced text', async () => {
    await mountAudio('sound');
    draft.patch({ soundPrompt: 'rain', ...rewrites('sound', { text: 'Steady heavy rain.', from: { input: 'rain', model: 'sa3', task: 'sound' }, history: [] }) });
    await settle();
    expect(clearOf('prompt')).not.toBeNull();

    await click(clearOf('prompt'));

    expect(draft.state.soundPrompt).toBe('');
    expect(draft.state.audioRewrites.sound.text).toBe('');
    await click(undoOf());
    expect(draft.state).toMatchObject({ soundPrompt: 'rain' });
    expect(draft.state.audioRewrites.sound.text).toBe('Steady heavy rain.');
  });

  it('clears the Speech text and its Written text, with Undo on the left of the line the character count keeps the right of', async () => {
    await mountAudio('speech');
    draft.patch({ speechText: 'Hello there.', ...rewrites('text', { text: 'Hello there, friends.', from: { input: 'Hello there.', model: 'qwen3_tts', task: 'script' }, history: [] }) });
    await settle();

    await click(clearOf('text'));

    expect(draft.state.speechText).toBe('');
    expect(draft.state.audioRewrites.text.text).toBe('');
    const row = target.querySelector('.limit-row')!;
    expect(row.contains(undoOf())).toBe(true);
    expect(row.querySelector('[data-testid="text-count"]')).not.toBeNull();

    await click(undoOf());
    expect(draft.state.speechText).toBe('Hello there.');
    expect(draft.state.audioRewrites.text.text).toBe('Hello there, friends.');
  });

  it('clears the whole song and its versions, in blocks and as text, and Undo brings them back with the view left as it was', async () => {
    await mountAudio('music');
    draft.patch({ lyrics: SONG, lyricsVersions: [{ text: 'my words', mine: true }, { text: SONG, mine: false }], lyricsCurrent: 1 });
    await settle();
    expect(clearOf('lyrics')).not.toBeNull();

    await click(clearOf('lyrics'));

    expect(draft.state).toMatchObject({ lyrics: '', lyricsVersions: [], lyricsCurrent: 0 });
    expect(target.querySelectorAll('[data-testid="song-block"]')).toHaveLength(0);
    expect(clearOf('lyrics')).toBeNull();
    expect(document.activeElement).toBe(field('ws-lyrics'));
    await click(undoOf());
    expect(draft.state).toMatchObject({ lyrics: SONG, lyricsCurrent: 1 });
    expect(draft.state.lyricsVersions).toHaveLength(2);

    await click(target.querySelector('#ws-lyrics-as-text'));
    await click(clearOf('lyrics'));
    expect(draft.state.lyrics).toBe('');
    expect(target.querySelector('[data-testid="song-blocks"]')!.getAttribute('data-view')).toBe('text');
    await click(undoOf());
    expect(draft.state.lyrics).toBe(SONG);
  });

  it('leaves the Style Undo alone when the Lyrics are cleared, and the other way round', async () => {
    await mountAudio('music');
    draft.patch({ audioStyle: 'folk', lyrics: SONG });
    await settle();
    await click(clearOf('style'));
    await click(clearOf('lyrics'));

    expect(target.querySelectorAll('[data-testid="clear-undo"]')).toHaveLength(2);
    await click(target.querySelector('[aria-label="Undo clearing the style"]'));
    expect(draft.state).toMatchObject({ audioStyle: 'folk', lyrics: '' });
  });

  it('gives a finger the Undo to focus', async () => {
    layoutProbe.coarse = true;
    await mountAudio('music');
    draft.patch({ audioStyle: 'folk' });
    await settle();

    await click(clearOf('style'));

    expect(document.activeElement).toBe(undoOf());
  });

  it('drops the Undo of a box when another kind opens', async () => {
    await mountAudio('music');
    draft.patch({ audioStyle: 'folk' });
    await settle();
    await click(clearOf('style'));
    expect(undoOf()).not.toBeNull();

    await pickKind('sound');

    expect(undoOf()).toBeNull();
    await pickKind('music');
    expect(undoOf()).toBeNull();
  });
});

describe('Describe it and Fill in', () => {
  const MODELS = [...AUDIO_MODELS, SA3_MUSIC];
  const IDEA = 'a sad folk song about leaving the coast';
  const SONG = '[Verse]\nI pack the boat at dawn\n\n[Chorus]\nGoodbye to the coast';
  const FILLED = { style: 'folk, acoustic guitar, wistful vocal', tempo: '84', key: 'Am', lyrics: SONG };

  async function mountDescribe(kind: 'music' | 'speech' | 'sound' = 'music', options: { compact?: boolean; tall?: boolean; enhancer?: boolean; active?: boolean; patch?: Parameters<typeof draft.patch>[0]; model?: string; smart?: { downloaded: boolean; download_size_label: string | null } } = {}): Promise<WorkspaceContext> {
    const running = jobSnap('job-live', 'running', { workflow: kind });
    const base = withAudio(makeContext({ history_assets: [] }), MODELS);
    const withEnhancer = options.enhancer === false ? base : withAudioEnhancer(base);
    if (withEnhancer.prompt_enhancer) withEnhancer.prompt_enhancer.smart = { ...(options.smart ?? { downloaded: true, download_size_label: null }), unavailable: null };
    const context = options.active ? { ...withEnhancer, active_job: running, queued_jobs: [] } : withEnhancer;
    if (options.active) workspaceApiMocks.listJobs.mockResolvedValue({ active_job: running, queued_jobs: [] });
    layoutProbe.compact = options.compact ?? false;
    layoutProbe.tall = options.tall ?? false;
    galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage([]));
    draft.reset();
    draft.patch({ workflow: kind, ...(options.patch ?? {}) });
    app = await mountWorkspacePage(context, target);
    if (options.model) {
      draft.hydrateFromContext(context, options.model);
      await settle();
    }
    return context;
  }

  /** Make the enhancer answer each run with the next of *replies* as a Fill in's fields; one may carry a notice or change the draft first. */
  function answerWith(...replies: ({ fields: Record<string, string>; warning?: string; during?: () => void } | 'error')[]): void {
    const queue = [...replies];
    enhanceApiMocks.enhancePrompt.mockImplementation(async (_body: unknown, ...rest: unknown[]) => {
      const onFrame = rest[0] as (frame: AudioEnhanceFrame) => void;
      const next = queue.shift() ?? { fields: {} };
      onFrame({ type: 'status', phase: 'generating' });
      if (next === 'error') {
        onFrame({ type: 'error', detail: "The helper couldn't fill anything from that; try describing it a little differently." });
        return;
      }
      next.during?.();
      onFrame({ type: 'done', task: 'fill', fields: next.fields, clamped: false, model: 'owner/llm', ...(next.warning ? { warning: next.warning } : {}) });
    });
  }

  const lastBody = () => enhanceApiMocks.enhancePrompt.mock.calls.at(-1)?.[0] as Record<string, unknown>;
  const box = () => target.querySelector('[data-testid="describe-it"]') as HTMLElement | null;
  const fillButton = () => document.querySelector('#ws-describe-fill') as HTMLButtonElement | null;
  /** The line under the box: what Fill in will write, or the visible reason it cannot run. */
  const hint = () => target.querySelector('[data-testid="describe-hint"]')?.textContent?.replace(/\s+/g, ' ').trim() ?? null;
  const summary = () => target.querySelector('[data-testid="describe-summary"]')?.textContent?.replace(/\s+/g, ' ').trim() ?? null;
  const undoButton = () => document.querySelector('#ws-describe-undo') as HTMLButtonElement | null;
  const marks = () => Array.from(target.querySelectorAll('[data-testid="filled-mark"]'));
  const click = async (node: Element | null) => {
    (node as HTMLElement).click();
    await settle();
  };
  async function describeAndFill(text = IDEA): Promise<void> {
    type('ws-describe', text);
    await settle();
    await click(fillButton());
  }

  it('is the first thing in Compose for Music and for Speech, and Sound has none', async () => {
    await mountDescribe('music');
    expect(box()).not.toBeNull();
    expect(target.querySelector('.compose-pane')?.querySelector('[data-testid="describe-it"]')).not.toBeNull();
    expect(box()!.compareDocumentPosition(target.querySelector('[data-testid="audio-compose"]')!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();

    await unmount(app!);
    app = null;
    document.body.innerHTML = '';
    target = createHost();
    await mountDescribe('speech');
    expect(box()).not.toBeNull();

    await unmount(app!);
    app = null;
    document.body.innerHTML = '';
    target = createHost();
    await mountDescribe('sound');
    expect(box()).toBeNull();
  });

  it('is not offered on a machine without the prompt enhancer', async () => {
    await mountDescribe('music', { enhancer: false });

    expect(box()).toBeNull();
  });

  it('fills the empty fields of a song in one request, and says what it filled', async () => {
    await mountDescribe('music');
    answerWith({ fields: FILLED });

    await describeAndFill();

    expect(lastBody()).toEqual({ mode: 'audio', task: 'fill', model: 'yue2', kind: 'music', prompt: IDEA, settings: { fields: ['style', 'lyrics', 'tempo', 'key'] } });
    expect(enhanceApiMocks.enhancePrompt).toHaveBeenCalledTimes(1);
    expect(draft.state).toMatchObject({ audioStyle: FILLED.style, lyrics: SONG, audioTempo: 84, audioKey: 'Am' });
    expect(draft.state.audioRewrites.style.text).toBe('');
    expect(summary()).toBe('Filled Style, Lyrics, Tempo and Key Undo fill');
  });

  it('fills the Lyrics of a song that has only section tags, sends those tags, and puts them back with Undo fill', async () => {
    const tags = '[Verse]\n\n[Chorus]';
    await mountDescribe('music', { patch: { lyrics: tags } });
    answerWith({ fields: FILLED });

    await describeAndFill();

    expect((lastBody().settings as { fields: string[] }).fields).toContain('lyrics');
    expect(lastBody()).toMatchObject({ context: { lyrics: tags } });
    expect(draft.state.lyrics).toBe(SONG);

    await click(undoButton());
    expect(draft.state.lyrics).toBe(tags);
  });

  it('writes the Style into your own Style tab, and Generate sends it as your text with no enhance record', async () => {
    await mountDescribe('music');
    answerWith({ fields: FILLED });

    await describeAndFill();
    const form = await submitForm(target);

    expect((target.querySelector('#ws-audio-style') as HTMLTextAreaElement).value).toBe(FILLED.style);
    expect(target.querySelector('#ws-audio-style-tab')?.getAttribute('aria-selected')).toBe('true');
    expect(form.get('prompt')).toBe(FILLED.style);
    expect(form.has('prompt_input')).toBe(false);
    expect(form.has('enhance_record')).toBe(false);
    expect(form.get('lyrics')).toBe(SONG);
    expect(form.get('tempo')).toBe('84');
    expect(form.get('key')).toBe('Am');
  });

  it('sends a Style typed after the fill, not the filled one, and Undo fill puts back the empty Style', async () => {
    await mountDescribe('music');
    answerWith({ fields: FILLED });
    await describeAndFill();

    await click(undoButton());
    expect(draft.state.audioStyle).toBe('');

    type('ws-audio-style', 'my own folk');
    await settle();
    const form = await submitForm(target);

    expect(form.get('prompt')).toBe('my own folk');
  });

  it('marks the filled Tempo and Key rows with a ✦, and the text boxes carry no "From your idea" mark', async () => {
    await mountDescribe('music');
    answerWith({ fields: FILLED });

    await describeAndFill();

    expect(marks()).toHaveLength(2);
    expect(target.querySelector('.text-pair')?.textContent).not.toContain('From your idea');
    expect(target.querySelector('.text-pair [data-testid="filled-mark"]')).toBeNull();
    expect(target.querySelector('[for="ws-tempo"]')?.querySelector('[data-testid="filled-mark"]')).not.toBeNull();
    expect(target.querySelector('[for="ws-key"]')?.querySelector('[data-testid="filled-mark"]')).not.toBeNull();
  });

  it('folds to one line after Fill in, keeping the idea, and opens again to edit it', async () => {
    await mountDescribe('music');
    answerWith({ fields: FILLED });

    await describeAndFill();

    expect(box()!.dataset.folded).toBe('true');
    expect(target.querySelector('#ws-describe')).toBeNull();
    expect(target.querySelector('[data-testid="describe-line"]')?.textContent).toBe(IDEA);

    await click(document.querySelector('#ws-describe-open'));
    expect(box()!.dataset.folded).toBe('false');
    expect((target.querySelector('#ws-describe') as HTMLTextAreaElement).value).toBe(IDEA);
  });

  it('folds Describe it with an arrow of its own, not the Collapse sidebar arrow, and unfolds with the opposite one', async () => {
    await mountDescribe('music');
    const iconPath = (button: Element | null) => button?.querySelector('svg path')?.getAttribute('d') ?? null;
    const foldPath = iconPath(document.querySelector('[aria-label="Fold Describe it"]'));
    const sidebarPath = iconPath(document.querySelector('[aria-label="Collapse sidebar"]'));

    expect(foldPath).not.toBeNull();
    expect(sidebarPath).not.toBeNull();
    expect(foldPath).not.toBe(sidebarPath);

    await click(document.querySelector('[aria-label="Fold Describe it"]'));
    const openPath = iconPath(document.querySelector('#ws-describe-open'));
    expect(openPath).not.toBeNull();
    expect(openPath).not.toBe(foldPath);
  });

  it('starts folded when the user begins with their own lyrics, or their own text', async () => {
    await mountDescribe('music', { patch: { lyrics: SONG } });
    expect(box()!.dataset.folded).toBe('true');
    expect(target.querySelector('[data-testid="describe-line"]')?.textContent).toBe('Fill what is empty from an idea');

    await unmount(app!);
    app = null;
    document.body.innerHTML = '';
    target = createHost();
    await mountDescribe('speech', { patch: { speechText: 'Come closer.' } });
    expect(box()!.dataset.folded).toBe('true');
  });

  it('keeps what the user already set: it asks only for the empty fields and says it kept the rest', async () => {
    await mountDescribe('music', { patch: { audioKey: 'Em', audioStyle: 'dark synth' } });
    answerWith({ fields: { lyrics: SONG, tempo: '90' } });
    type('ws-describe', IDEA);
    await settle();
    expect(target.querySelector('[data-testid="describe-hint"]')?.textContent?.replace(/\s+/g, ' ').trim()).toBe('Will fill Lyrics and Tempo · keeps your Style and Key.');

    await click(fillButton());

    expect(lastBody().settings).toEqual({ fields: ['lyrics', 'tempo'] });
    expect(lastBody().context).toEqual({ style: 'dark synth', key: 'Em' });
    expect(draft.state).toMatchObject({ audioStyle: 'dark synth', audioKey: 'Em', audioTempo: 90, lyrics: SONG });
    expect(draft.state.audioRewrites.style.text).toBe('');
    expect(summary()).toBe('Filled Lyrics and Tempo Undo fill');
  });

  it('never writes a field the user typed in while the helper worked', async () => {
    await mountDescribe('music');
    answerWith({ fields: FILLED, during: () => draft.update('audioTempo', 100) });

    await describeAndFill();

    expect(draft.state.audioTempo).toBe(100);
    expect(draft.state).toMatchObject({ audioKey: 'Am', lyrics: SONG });
    expect(summary()).toContain('Filled Style, Lyrics and Key');
  });

  it('fills again: it rewrites only the fields still marked, and leaves one the user edited', async () => {
    await mountDescribe('music');
    answerWith({ fields: FILLED }, { fields: { style: 'folk, banjo', tempo: '110', key: 'C', lyrics: `${SONG}\n\n[Outro]\nSail on` } });
    await describeAndFill();
    draft.update('audioTempo', 100);
    await settle();

    await click(document.querySelector('#ws-describe-open'));
    expect(fillButton()!.textContent).toContain('Fill in again');
    await click(fillButton());

    expect(lastBody().settings).toEqual({ fields: ['style', 'lyrics', 'key'] });
    expect(draft.state).toMatchObject({ audioTempo: 100, audioKey: 'C', lyrics: `${SONG}\n\n[Outro]\nSail on` });
    expect(draft.state.audioStyle).toBe('folk, banjo');
    expect(draft.state.lyricsVersions).toHaveLength(2);
  });

  it('takes a ✦ off a field the moment the user edits it', async () => {
    await mountDescribe('music');
    answerWith({ fields: FILLED });
    await describeAndFill();
    expect(marks()).toHaveLength(2);

    draft.update('audioTempo', 90);
    await settle();
    expect(marks()).toHaveLength(1);
    expect(target.querySelector('[for="ws-tempo"]')?.querySelector('[data-testid="filled-mark"]')).toBeNull();
    // The summary names only what Undo fill would still restore, not what the user has since taken over.
    expect(summary()).toBe('Filled Style, Lyrics and Key Undo fill');
  });

  it('undoes the fill: every field is back as it was, and the box has nothing left to undo', async () => {
    await mountDescribe('music', { patch: { audioTempo: null } });
    answerWith({ fields: FILLED });
    await describeAndFill();

    await click(undoButton());

    expect(draft.state).toMatchObject({ audioStyle: '', lyrics: '', audioTempo: null, audioKey: null, lyricsVersions: [] });
    expect(draft.state.audioRewrites.style.text).toBe('');
    expect(marks()).toHaveLength(0);
    expect(summary()).toBeNull();
    expect(undoButton()).toBeNull();
  });

  it('fills a Speech text and a Vibe, and the Delivery section says so', async () => {
    await mountDescribe('speech');
    answerWith({ fields: { vibe: 'whispered-secret', text: 'Come closer. I have to tell you something.' } });

    await describeAndFill('a whispered secret');

    expect(lastBody()).toEqual({ mode: 'audio', task: 'fill', model: 'qwen3_tts', kind: 'speech', prompt: 'a whispered secret', settings: { fields: ['text', 'vibe'] } });
    expect(draft.state.speechText).toBe('Come closer. I have to tell you something.');
    expect(draft.state.speechDelivery).toEqual({ emotion: 'neutral', manner: 'whisper', pace: 'slow', pitch: 'low', expressiveness: 'low' });
    expect(summary()).toBe('Filled Text and Vibe Undo fill');
    expect(target.querySelector('[data-testid="delivery-vibe-filled"]')?.textContent).toContain('Whispered secret');
    expect(target.querySelector('#ws-prompt')).not.toBeNull();
    expect(marks().length).toBeGreaterThanOrEqual(1);
    expect(target.querySelector('[data-testid="audio-compose"] [data-testid="filled-mark"]')).toBeNull();
  });

  it('sends the text so far when only the Vibe is empty', async () => {
    await mountDescribe('speech', { patch: { speechText: 'Come closer.' } });
    answerWith({ fields: { vibe: 'sad-confession' } });
    await click(document.querySelector('#ws-describe-open'));
    type('ws-describe', 'read it sadly');
    await settle();

    await click(fillButton());

    expect(lastBody()).toMatchObject({ settings: { fields: ['vibe'] }, context: { text: 'Come closer.' } });
    expect(draft.state.speechText).toBe('Come closer.');
    expect(draft.state.speechDelivery.emotion).toBe('sad');
  });

  it('fills Style, Tempo and Key but no Lyrics on Stable Audio', async () => {
    await mountDescribe('music', { model: 'sa3' });
    answerWith({ fields: { style: 'Calm solo piano.', tempo: '70', key: 'C' } });

    await describeAndFill('calm piano for studying');

    expect(lastBody()).toMatchObject({ model: 'sa3', settings: { fields: ['style', 'tempo', 'key'] } });
    expect(draft.state).toMatchObject({ audioTempo: 70, audioKey: 'C', lyrics: '' });
    expect(draft.state.audioStyle).toBe('Calm solo piano.');
  });

  it('shows the enhancer’s notice and what stayed empty', async () => {
    await mountDescribe('music');
    answerWith({ fields: { style: FILLED.style }, warning: 'Left empty: Tempo, Key and Lyrics.' });

    await describeAndFill();

    expect(summary()).toBe('Filled Style Undo fill');
    expect(target.querySelector('[data-testid="describe-note"]')?.textContent).toBe('Left empty: Tempo, Key and Lyrics.');
    expect(draft.state.audioTempo).toBeNull();
  });

  it('says why nothing was filled and changes nothing', async () => {
    await mountDescribe('music');
    answerWith('error');

    await describeAndFill();

    expect(target.querySelector('[data-testid="describe-error"]')?.textContent).toContain("couldn't fill anything");
    expect(draft.state).toMatchObject({ audioStyle: '', lyrics: '', audioTempo: null });
    expect(box()!.dataset.folded).toBe('false');
    expect(summary()).toBeNull();
  });

  it('has nothing to run until the idea is written, and says when every field is the user’s', async () => {
    await mountDescribe('music');
    expect(fillButton()!.disabled).toBe(true);
    expect(hint()).toBe('Describe it first.');
    expect(fillButton()!.hasAttribute('title')).toBe(false);

    type('ws-describe', IDEA);
    await settle();
    expect(fillButton()!.disabled).toBe(false);

    draft.patch({ audioStyle: 'folk', lyrics: SONG, audioTempo: 80, audioKey: 'C' });
    await settle();
    // The Lyrics have words, so the box folds; opening it shows the button off.
    await click(document.querySelector('#ws-describe-open'));
    expect(fillButton()!.disabled).toBe(true);
    expect(hint()).toContain('Every field is yours');
  });

  it('is unavailable while a take is made', async () => {
    await mountDescribe('music', { active: true });
    type('ws-describe', IDEA);
    await settle();

    expect(fillButton()!.disabled).toBe(true);
    expect(hint()).toBe('Available when the take is done.');
  });

  it('runs on Ctrl+Enter in the idea', async () => {
    await mountDescribe('music');
    answerWith({ fields: FILLED });
    type('ws-describe', IDEA);
    await settle();

    pressKey('Enter', { ctrlKey: true }, field('ws-describe'));
    await settle();

    expect(draft.state.lyrics).toBe(SONG);
  });

  it('sends the Smart enhancer when Smart is chosen', async () => {
    await mountDescribe('music', { patch: { describeTier: 'smart' } });
    answerWith({ fields: FILLED });

    await describeAndFill();

    expect(lastBody()).toMatchObject({ enhance_model: 'smart' });
  });

  it('holds Enhance back on an empty Style, whatever Describe it holds, with the usual empty-text reason', async () => {
    await mountDescribe('music', { patch: { audioDescribe: { music: { text: IDEA, marked: [], filled: {}, undo: null, folded: false }, speech: { text: '', marked: [], filled: {}, undo: null, folded: null } } } });

    (document.querySelector('#ws-enhance-toggle') as HTMLButtonElement).click();
    await settle();

    expect((document.querySelector('#ws-enhance-run') as HTMLButtonElement).disabled).toBe(true);
    expect(document.querySelector('#ws-enhance-panel')?.textContent).toContain('Write the style first.');
    expect(enhanceApiMocks.enhancePrompt).not.toHaveBeenCalled();
  });

  it('enhances the filled Style after a Fill, starting from the text Fill wrote', async () => {
    await mountDescribe('music');
    answerWith({ fields: FILLED });
    await describeAndFill();
    enhanceApiMocks.enhancePrompt.mockImplementation(async (_body: unknown, ...rest: unknown[]) => {
      (rest[0] as (frame: AudioEnhanceFrame) => void)({ type: 'done', task: 'style', fields: { style: 'Folk, guitar, wistful' }, clamped: false, model: 'owner/llm' });
    });

    (document.querySelector('#ws-enhance-toggle') as HTMLButtonElement).click();
    await settle();
    await click(document.querySelector('#ws-enhance-run'));

    expect(lastBody()).toMatchObject({ task: 'style', prompt: FILLED.style });
    expect(lastBody().context).not.toHaveProperty('describe');
    expect(draft.state.audioRewrites.style).toMatchObject({ text: 'Folk, guitar, wistful', from: { input: FILLED.style, task: 'style' } });
  });

  describe('Clear', () => {
    const clearIdea = () => document.querySelector('[aria-label="Clear the idea"]') as HTMLButtonElement | null;
    const clearUndoButton = () => document.querySelector('[data-testid="clear-undo"] button') as HTMLButtonElement | null;

    it('empties the idea from the open box and the folded line, keeps what Fill in wrote and its Undo fill, and Undo brings the idea back', async () => {
      await mountDescribe('music');
      answerWith({ fields: FILLED });
      await describeAndFill();
      expect(box()!.dataset.folded).toBe('true');
      expect(target.querySelector('[data-testid="describe-line"]')?.textContent).toBe(IDEA);

      await click(clearIdea());

      expect(draft.state.audioDescribe.music.text).toBe('');
      expect(draft.state).toMatchObject({ audioStyle: FILLED.style, lyrics: SONG });
      expect(summary()).toBe('Filled Style, Lyrics, Tempo and Key Undo fill');
      expect(undoButton()).not.toBeNull();
      expect(clearIdea()).toBeNull();
      expect(clearUndoButton()!.closest('[data-testid="clear-undo"]')!.textContent?.trim()).toBe('Cleared. Undo');

      await click(clearUndoButton());

      expect(draft.state.audioDescribe.music.text).toBe(IDEA);
      expect(clearUndoButton()).toBeNull();
      expect(clearIdea()).not.toBeNull();
    });

    it('is in the open box too, only while there is an idea, and goes at the next edit of the idea', async () => {
      await mountDescribe('music');
      expect(clearIdea()).toBeNull();

      type('ws-describe', IDEA);
      await settle();
      expect(clearIdea()!.closest('.ui-prompt-tools')).not.toBeNull();
      await click(clearIdea());
      expect(field('ws-describe').value).toBe('');
      expect(document.activeElement).toBe(field('ws-describe'));
      expect(clearUndoButton()).not.toBeNull();

      type('ws-describe', 'a faster one');
      await settle();

      expect(clearUndoButton()).toBeNull();
      expect(draft.state.audioDescribe.music.text).toBe('a faster one');
    });

    it('is hidden while Fill in runs', async () => {
      await mountDescribe('music');
      enhanceApiMocks.enhancePrompt.mockImplementation(() => new Promise<void>(() => undefined));
      type('ws-describe', IDEA);
      await settle();
      expect(clearIdea()).not.toBeNull();

      await click(fillButton());

      expect(document.querySelector('#ws-describe-stop')).not.toBeNull();
      expect(clearIdea()).toBeNull();
    });
  });

  describe('Fill in’s own enhancer', () => {
    const tierButton = () => document.querySelector('#ws-describe-tier') as HTMLButtonElement | null;
    const tierItems = () => Array.from(document.querySelectorAll('[role="menuitemradio"]')) as HTMLElement[];
    const hintText = () => target.querySelector('[data-testid="describe-hint"]')?.textContent?.replace(/\s+/g, ' ').trim() ?? null;
    const chooseTier = async (tier: 'fast' | 'smart') => {
      await click(tierButton());
      await click(document.querySelector(`[data-action="describe-tier-${tier}"]`));
    };

    it('sits right after Fill in as "Fast ▾", and its menu offers Fast and Smart with what each does', async () => {
      await mountDescribe('music');
      type('ws-describe', IDEA);
      await settle();

      expect(tierButton()!.textContent?.trim()).toBe('Fast');
      expect(tierButton()!.getAttribute('aria-label')).toBe('Enhancer for Fill in: Fast');
      expect(fillButton()!.nextElementSibling).toBe(tierButton());

      await click(tierButton());

      expect(tierItems().map((item) => item.textContent?.replace(/\s+/g, ' ').trim())).toEqual(['FastThe default model.', 'SmartFollows your idea more closely.']);
      expect(tierItems().map((item) => item.getAttribute('aria-checked'))).toEqual(['true', 'false']);
    });

    it('is remembered apart from the shared enhancer, which Enhance and the helpers keep using', async () => {
      await mountDescribe('music', { patch: { enhanceModel: 'fast' } });
      type('ws-describe', IDEA);
      await settle();

      await chooseTier('smart');

      expect(draft.state).toMatchObject({ describeTier: 'smart', enhanceModel: 'fast' });
      expect(tierButton()!.textContent?.trim()).toBe('Smart');

      draft.patch({ describeTier: 'fast', enhanceModel: 'smart' });
      await settle();
      expect(tierButton()!.textContent?.trim()).toBe('Fast');

      draft.patch({ describeTier: 'smart', enhanceModel: 'fast' });
      answerWith({ fields: FILLED });
      await click(fillButton());
      expect(lastBody()).toMatchObject({ task: 'fill', enhance_model: 'smart' });
    });

    it('ignores the shared enhancer for a Fill in', async () => {
      await mountDescribe('music', { patch: { enhanceModel: 'smart', describeTier: 'fast' } });
      answerWith({ fields: FILLED });

      await describeAndFill();

      expect(lastBody()).not.toHaveProperty('enhance_model');
    });

    it('is what a helper’s Fill in from Describe it uses', async () => {
      await mountDescribe('music', { patch: { enhanceModel: 'fast', describeTier: 'smart' } });
      type('ws-describe', IDEA);
      await settle();
      answerWith({ fields: FILLED });

      await click(document.querySelector('#ws-lyrics-helper-toggle'));
      await click(document.querySelector('#ws-lyrics-helper-fill'));

      expect(lastBody()).toMatchObject({ task: 'fill', enhance_model: 'smart' });
    });

    it('adds the first-use download of an enhancer that is not on the machine to the hint line, and to the Smart item', async () => {
      downloadedTiers.delete('smart');
      await mountDescribe('music', { smart: { downloaded: false, download_size_label: '9.1 GB' }, patch: { describeTier: 'smart' } });
      type('ws-describe', IDEA);
      await settle();

      expect(hintText()).toBe('Will fill Style, Lyrics, Tempo and Key. First use downloads ≈9.1 GB.');
      await click(tierButton());
      expect(tierItems()[1].textContent).toContain('First use downloads ≈9.1 GB.');
    });

    it('stays where it is while Fill in runs, disabled, with Stop in Fill in’s place', async () => {
      await mountDescribe('music');
      enhanceApiMocks.enhancePrompt.mockImplementation(() => new Promise<void>(() => undefined));
      type('ws-describe', IDEA);
      await settle();

      await click(fillButton());

      expect(fillButton()).toBeNull();
      expect(document.querySelector('#ws-describe-stop')!.nextElementSibling).toBe(tierButton());
      expect(tierButton()!.disabled).toBe(true);
    });

    it('is not on the folded line', async () => {
      await mountDescribe('music');
      answerWith({ fields: FILLED });
      await describeAndFill();

      expect(box()!.dataset.folded).toBe('true');
      expect(tierButton()).toBeNull();
    });

    it('is not offered where there is no Smart enhancer to choose', async () => {
      await mountDescribe('music');
      type('ws-describe', IDEA);
      await settle();
      await unmount(app!);
      app = null;
      document.body.innerHTML = '';
      target = createHost();
      const base = withAudio(makeContext({ history_assets: [] }), MODELS);
      app = await mountWorkspacePage(withAudioEnhancer(base), target);
      type('ws-describe', IDEA);
      await settle();

      expect(tierButton()).toBeNull();
    });
  });

  describe('the helpers’ empty states', () => {
    it('offer Fill in from Describe it: it opens the box when the idea is not written, and fills when it is', async () => {
      await mountDescribe('music');
      answerWith({ fields: FILLED });
      await click(document.querySelector('#ws-describe-open') ?? target.querySelector('#ws-describe'));
      await click(document.querySelector('#ws-lyrics-helper-toggle'));
      await click(document.querySelector('#ws-lyrics-helper-fill'));
      expect(enhanceApiMocks.enhancePrompt).not.toHaveBeenCalled();
      expect(document.activeElement?.id).toBe('ws-describe');

      type('ws-describe', IDEA);
      await settle();
      await click(document.querySelector('#ws-lyrics-helper-toggle'));
      await click(document.querySelector('#ws-lyrics-helper-fill'));

      expect(lastBody()).toMatchObject({ task: 'fill', prompt: IDEA });
      expect(draft.state.lyrics).toBe(SONG);
    });

    it('grey the Fill in from Describe it during a take, with the same reason as Run, and leave the panel open', async () => {
      await mountDescribe('music', { active: true });
      type('ws-describe', IDEA);
      await settle();
      await click(document.querySelector('#ws-lyrics-helper-toggle'));

      const fill = document.querySelector('#ws-lyrics-helper-fill') as HTMLButtonElement;
      expect(fill.disabled).toBe(true);
      expect(document.querySelector('[data-testid="lyrics-helper-fill-reason"]')?.textContent).toBe('Available when the take is done.');
      await click(fill);

      expect(document.querySelector('#ws-lyrics-helper-panel')).not.toBeNull();
      expect(enhanceApiMocks.enhancePrompt).not.toHaveBeenCalled();
    });

    it('does the same for the Script helper during a take', async () => {
      await mountDescribe('speech', { active: true });
      type('ws-describe', 'a short thank-you');
      await settle();
      await click(document.querySelector('#ws-script-helper-toggle'));

      const fill = document.querySelector('#ws-script-helper-fill') as HTMLButtonElement;
      expect(fill.disabled).toBe(true);
      expect(document.querySelector('[data-testid="script-helper-fill-reason"]')?.textContent).toBe('Available when the take is done.');
      await click(fill);

      expect(document.querySelector('#ws-script-helper-panel')).not.toBeNull();
      expect(enhanceApiMocks.enhancePrompt).not.toHaveBeenCalled();
    });

    it('does the same for the Script helper on an empty text', async () => {
      await mountDescribe('speech');
      type('ws-describe', 'a short thank-you');
      await settle();
      answerWith({ fields: { vibe: 'storyteller', text: 'Thank you all for coming.' } });

      await click(document.querySelector('#ws-script-helper-toggle'));
      await click(document.querySelector('#ws-script-helper-fill'));

      expect(draft.state.speechText).toBe('Thank you all for coming.');
    });
  });

  describe('on a phone and a tablet', () => {
    it('is full width at the top of Compose on a phone, with the fill’s marks in the Delivery chip and the text', async () => {
      await mountDescribe('speech', { compact: true });
      answerWith({ fields: { vibe: 'storyteller', text: 'Once upon a time.' } });

      await describeAndFill('a bedtime story');

      expect(box()).not.toBeNull();
      expect(summary()).toBe('Filled Text and Vibe Undo fill');
      const chip = target.querySelector('[data-chip="delivery"]') as HTMLElement;
      expect(chip.getAttribute('aria-label')).toContain('filled from your idea');
      expect(chip.querySelector('[data-testid="filled-mark"]')).not.toBeNull();
    });

    it('marks the Tempo and Key chips of a song', async () => {
      await mountDescribe('music', { compact: true });
      answerWith({ fields: FILLED });

      await describeAndFill();

      for (const key of ['tempo', 'key']) expect((target.querySelector(`[data-chip="${key}"]`) as HTMLElement).querySelector('[data-testid="filled-mark"]')).not.toBeNull();
    });

    it('runs the same on a tablet', async () => {
      await mountDescribe('music', { tall: true });
      answerWith({ fields: FILLED });

      await describeAndFill();

      expect(draft.state.lyrics).toBe(SONG);
      expect(box()!.dataset.folded).toBe('true');
    });
  });
});
