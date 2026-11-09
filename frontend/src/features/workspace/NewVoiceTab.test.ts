// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '$lib/api/client';
import { draft } from '$lib/state/draft.svelte';
import { jobStore } from '$lib/state/job.svelte';
import { newVoiceForm } from '$lib/state/newVoiceForm.svelte';
import { toasts } from '$lib/state/toasts.svelte';
import { voiceWindow } from '$lib/state/voiceWindow.svelte';
import { voicesStore } from '$lib/state/voices.svelte';
import type { JobContext, JobSnapshot, SavedVoice, VoiceCandidate, VoiceDesignBatch, VoiceDesignState } from '$lib/types';

import { AUDIO_MODELS, AUDIO_SETTINGS } from '../../test-utils/audio';
import { lastEventSource } from '../../test-utils/eventSource';
import { answerConfirm, createHost, resetWorkspaceTest, settle } from '../../test-utils/workspaceHarness';
import { EMPTY_JOB_PLAN, layoutProbe, voiceApiMocks, workspaceApiMocks } from '../../test-utils/workspaceMocks';
import type { VoiceView } from './audioView';
import { EXAMPLE_DESCRIPTIONS, PREVIEW_LINES } from './newVoice';
import VoicesWindow from './VoicesWindow.svelte';

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

const control = vi.hoisted(() => ({ sendJobControl: vi.fn() }));

vi.mock('$lib/api/workspace', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/workspace')>();
  const { workspaceApiMocks: mocks } = await import('../../test-utils/workspaceMocks');
  return { ...actual, listJobs: mocks.listJobs, cancelJob: mocks.cancelJob, getJobSnapshot: mocks.getJobSnapshot, clearQueue: mocks.clearQueue, sendJobControl: control.sendJobControl };
});

vi.mock('$lib/api/voices', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/voices')>();
  const { voiceApiMocks: mocks } = await import('../../test-utils/workspaceMocks');
  return {
    ...actual,
    listVoices: mocks.listVoices,
    planVoiceDesign: mocks.planVoiceDesign,
    submitVoiceDesign: mocks.submitVoiceDesign,
    getDesignState: mocks.getDesignState,
    saveCandidate: mocks.saveCandidate,
    clearCandidates: mocks.clearCandidates,
  };
});

const SPEECH = AUDIO_MODELS.find((model) => model.id === 'qwen3_tts')!;
const VOICE = { recording: false, saved: false, set: false } as VoiceView;
const DESIGN_JOB: JobContext = { job_id: 'job-design', workflow: 'voice_design', media_type: 'audio', prompt: 'A warm woman', model: 'qwen3_tts', runs: 3, created_at: '' };
const QUEUED_DESIGN: JobSnapshot = { ...DESIGN_JOB, id: 'job-design', job_type: 'Voice design', status: 'queued', event_count: 1, paused: false, queue_position: 1, settings: {} } as JobSnapshot;

function candidate(number: number, overrides: Partial<VoiceCandidate> = {}): VoiceCandidate {
  return { id: `cand-${number}`, number, seconds: 7, saveable: true, reason: null, saved_as: null, ...overrides };
}

function batch(id: string, description: string, candidates: VoiceCandidate[]): VoiceDesignBatch {
  return { batch_id: id, description, text: PREVIEW_LINES.English, created_at: '2026-11-08T10:00:00Z', candidates };
}

const FIRST: VoiceDesignState = { batches: [batch('b1', 'A warm woman in her thirties', [candidate(1), candidate(2), candidate(3)])], unpicked: 3 };
const TWO_BATCHES: VoiceDesignState = {
  batches: [batch('b2', 'A deep, gravelly man', [candidate(4), candidate(5), candidate(6)]), ...FIRST.batches],
  unpicked: 6,
};

const MORGAN: SavedVoice = { slug: 'morgan', name: 'Morgan', text: PREVIEW_LINES.English, origin: 'design', seconds: 7, description: 'A warm woman in her thirties', created_at: '2026-11-08T11:00:00Z' };

let target: HTMLDivElement;
let app: Record<string, unknown> | null = null;

beforeEach(() => {
  resetWorkspaceTest();
  target = createHost();
  newVoiceForm.clear();
  voiceWindow.open = false;
  control.sendJobControl.mockReset().mockResolvedValue({});
  voiceApiMocks.planVoiceDesign.mockResolvedValue(EMPTY_JOB_PLAN);
  voiceApiMocks.submitVoiceDesign.mockResolvedValue(DESIGN_JOB);
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => undefined);
  // The phone's sheets slide in with the Web Animations API, which jsdom lacks.
  Element.prototype.animate = vi.fn(() => ({ cancel: vi.fn(), finished: Promise.resolve(), onfinish: null })) as unknown as Element['animate'];
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
  design?: VoiceDesignState;
  speechText?: string;
  /** The tab is not opened: the window is mounted closed. */
  closed?: boolean;
}

/** Mount the Voices window and open it on New voice, with the server holding *design*. */
async function openNewVoice(layoutName: Layout, options: Options = {}): Promise<void> {
  layoutProbe.compact = layoutName === 'compact';
  voiceApiMocks.getDesignState.mockResolvedValue(options.design ?? { batches: [], unpicked: 0 });
  voiceApiMocks.listVoices.mockResolvedValue([]);
  draft.reset();
  draft.patch({ workflow: 'speech', speechText: options.speechText ?? 'Hello there.' });
  app = flushSync(() => mount(VoicesWindow, { target, props: { voice: VOICE, model: SPEECH, vocabulary: AUDIO_SETTINGS } }));
  if (options.closed) return;
  voiceWindow.show('new');
  await settle();
}

/** The open window; a phone's sheet stays in the page, inert, while it slides away, so that is not open. */
function win(): HTMLElement | null {
  const found = document.querySelector<HTMLElement>('[role="dialog"][aria-label="Voices"]');
  for (let node: HTMLElement | null = found; node !== null; node = node.parentElement) if (node.inert) return null;
  return found;
}

const inWindow = <T extends Element = HTMLElement>(selector: string): T | null => win()?.querySelector<T>(selector) ?? null;
const description = (): HTMLTextAreaElement => inWindow<HTMLTextAreaElement>('#nv-desc')!;
const line = (): HTMLTextAreaElement => inWindow<HTMLTextAreaElement>('#nv-line')!;
const make = (): HTMLButtonElement => inWindow<HTMLButtonElement>('[data-action="make"]')!;
const stopButton = (): HTMLButtonElement | null => inWindow<HTMLButtonElement>('[data-action="stop"]');
const saveButton = (): HTMLButtonElement | null => inWindow<HTMLButtonElement>('[data-action="save-voice"]');
const nameField = (): HTMLInputElement => inWindow<HTMLInputElement>('#nv-name')!;
const status = (): string | null => inWindow('[data-testid="design-status"]')?.querySelector('.nv-status-line')?.textContent?.trim() ?? null;
const rows = (): HTMLElement[] => Array.from(win()?.querySelectorAll<HTMLElement>('[data-testid="candidate"]') ?? []);
const rowName = (row: HTMLElement): string => row.querySelector('.cand-name')?.textContent?.trim() ?? '';
const radio = (number: number): HTMLInputElement => inWindow<HTMLInputElement>(`input[aria-label="Choose Voice ${number}"]`)!;

async function click(element: Element | null): Promise<void> {
  (element as HTMLElement).click();
  await settle();
}

async function typeInto(input: HTMLInputElement | HTMLTextAreaElement, value: string): Promise<void> {
  input.value = value;
  input.dispatchEvent(new Event('input', { bubbles: true }));
  await settle();
}

/** Start the design job the page follows, and let it report that it has made *done* of 3 voices. */
async function runJob(done: number): Promise<void> {
  jobStore.startJob(DESIGN_JOB);
  await settle();
  lastEventSource().emit('job_started', { type: 'job_started', job_id: DESIGN_JOB.job_id });
  lastEventSource().emit('audio_progress', { type: 'audio_progress', phase: 'Design', step: done, total: 3 });
  await settle();
}

describe.each<Layout>(['wide', 'compact'])('the New voice tab (%s)', (layoutName) => {
  describe('the form', () => {
    it('starts with an empty description, the preview line for the take text, and a Make 3 that waits for the description', async () => {
      await openNewVoice(layoutName);

      expect(description().value).toBe('');
      expect(line().value).toBe(PREVIEW_LINES.English);
      expect(make().textContent).toContain('Make 3');
      expect(make().textContent).not.toContain('more');
      expect(make().disabled).toBe(true);
      expect(inWindow('[data-testid="save-bar"]')).toBeNull();
      expect(inWindow('[data-testid="candidates-panel"]')).not.toBeNull();
      expect(rows()).toHaveLength(0);

      await typeInto(description(), 'A calm man');

      expect(make().disabled).toBe(false);
      expect(inWindow('[data-testid="description-count"]')?.textContent).toBe('10 / 500');
    });

    it('prefills the preview line in the language of the take text, and keeps what the user writes over it', async () => {
      await openNewVoice(layoutName, { speechText: 'Guten Tag, meine Damen und Herren, ich bin hier.' });
      expect(line().value).toBe(PREVIEW_LINES.German);

      await typeInto(line(), 'Welcome home.');
      draft.patch({ speechText: '今天天气很好' });
      await settle();

      expect(line().value).toBe('Welcome home.');
    });

    it('says the read time, and warns outside 3 to 30 seconds without blocking Make 3', async () => {
      await openNewVoice(layoutName);
      await typeInto(description(), 'A calm man');
      expect(inWindow('[data-testid="line-read"]')).not.toBeNull();
      expect(inWindow('[data-testid="line-warning"]')).toBeNull();
      expect(inWindow('#nv-line-help')).not.toBeNull();

      await typeInto(line(), 'Hello.');

      expect(inWindow('[data-testid="line-warning"]')).not.toBeNull();
      expect(inWindow('#nv-line-help')).toBe(inWindow('[data-testid="line-warning"]'));
      expect(make().disabled).toBe(false);
    });

    it('offers the examples in a dropdown that fills the description and leaves it editable', async () => {
      await openNewVoice(layoutName);

      await click(inWindow('#nv-examples'));
      const entries = Array.from(document.querySelectorAll<HTMLElement>('[data-action^="example-"]'));
      expect(entries.map((entry) => entry.querySelector('.action-menu-name')?.textContent)).toEqual(EXAMPLE_DESCRIPTIONS.map((example) => example.label));
      await click(entries[1]);

      expect(description().value).toBe(EXAMPLE_DESCRIPTIONS[1].text);
      await typeInto(description(), `${EXAMPLE_DESCRIPTIONS[1].text}, whispering`);
      expect(description().value).toBe(`${EXAMPLE_DESCRIPTIONS[1].text}, whispering`);
      expect(make().disabled).toBe(false);
    });

    it('keeps the description and the preview line when the window is closed and opened again', async () => {
      await openNewVoice(layoutName);
      await typeInto(description(), 'A calm man');
      await typeInto(line(), 'Welcome home, friend of mine.');

      voiceWindow.open = false;
      await settle();
      expect(win()).toBeNull();
      voiceWindow.show('new');
      await settle();

      expect(description().value).toBe('A calm man');
      expect(line().value).toBe('Welcome home, friend of mine.');
    });
  });

  describe('Make 3', () => {
    async function fill(): Promise<void> {
      await typeInto(description(), 'A calm man');
    }

    it('sends nothing when the download is declined', async () => {
      voiceApiMocks.planVoiceDesign.mockResolvedValue({ ...EMPTY_JOB_PLAN, label: 'VoiceDesign', parts: [{ label: 'Qwen3-TTS VoiceDesign', model: 'qwen3_tts_design', bytes: 4_000_000_000 }], total_bytes: 4_000_000_000 });
      await openNewVoice(layoutName);
      await fill();

      await click(make());
      await answerConfirm(false);

      expect(voiceApiMocks.planVoiceDesign).toHaveBeenCalledOnce();
      expect(voiceApiMocks.submitVoiceDesign).not.toHaveBeenCalled();
      expect(jobStore.current).toBeNull();
      expect(make().disabled).toBe(false);
    });

    it('plans, then queues the job with the model, the description and the line, and follows the job while it waits behind another', async () => {
      voiceApiMocks.submitVoiceDesign.mockResolvedValue({ ...DESIGN_JOB, queue_position: 1 });
      workspaceApiMocks.listJobs.mockResolvedValue({ active_job: null, queued_jobs: [QUEUED_DESIGN] });
      await openNewVoice(layoutName);
      await fill();

      await click(make());

      const body = { model: 'qwen3_tts', description: 'A calm man', text: PREVIEW_LINES.English };
      expect(voiceApiMocks.planVoiceDesign).toHaveBeenCalledWith(body);
      expect(voiceApiMocks.submitVoiceDesign).toHaveBeenCalledWith(body);
      expect(jobStore.queuedJobs.map((job) => job.job_id)).toEqual([QUEUED_DESIGN.job_id]);
      expect(status()).toBe('Waiting in the queue');
      expect(make().disabled).toBe(true);
    });

    it('says why nothing started when the server refuses', async () => {
      voiceApiMocks.submitVoiceDesign.mockRejectedValue(new ApiError('bad', 422, 'line too long'));
      await openNewVoice(layoutName);
      await fill();

      await click(make());

      expect(inWindow('[data-testid="start-error"]')?.textContent).toBe('line too long');
      expect(jobStore.current).toBeNull();
      expect(make().disabled).toBe(false);
    });
  });

  describe('a run, found again after a reload', () => {
    it('shows a queued design job as waiting and takes it off the queue with Stop', async () => {
      jobStore.seedQueue([QUEUED_DESIGN]);
      await openNewVoice(layoutName);
      await typeInto(description(), 'A calm man');

      expect(status()).toBe('Waiting in the queue');
      expect(make().disabled).toBe(true);

      workspaceApiMocks.listJobs.mockResolvedValue({ active_job: null, queued_jobs: [] });
      await click(stopButton());

      expect(workspaceApiMocks.cancelJob).toHaveBeenCalledWith('job-design');
      expect(inWindow('[data-testid="design-status"]')).toBeNull();
      expect(make().disabled).toBe(false);
    });

    it('shows a running design job by how many voices it has made, and Stop sends it quit', async () => {
      await openNewVoice(layoutName);
      await runJob(2);

      expect(status()).toBe('Voice 2 of 3');
      expect(make().disabled).toBe(true);

      await click(stopButton());

      expect(control.sendJobControl).toHaveBeenCalledWith('job-design', 'quit');
      expect(stopButton()?.disabled).toBe(true);
      expect(stopButton()?.textContent).toContain('Stopping');
    });

    it('finds a design job that is already running when the window opens', async () => {
      await openNewVoice(layoutName, { closed: true });
      await runJob(1);

      voiceWindow.show('new');
      await settle();

      expect(status()).toBe('Voice 1 of 3');
    });

    it('ignores a job that makes images', async () => {
      jobStore.startJob({ job_id: 'job-image', workflow: 'txt2img', media_type: 'image', prompt: 'a fox', model: 'zit', runs: 1, created_at: '' });
      await openNewVoice(layoutName);
      await typeInto(description(), 'A calm man');

      expect(inWindow('[data-testid="design-status"]')).toBeNull();
      expect(make().disabled).toBe(false);
    });

    it('says what went wrong when the job failed', async () => {
      await openNewVoice(layoutName);
      await runJob(0);

      lastEventSource().emit('job_failed', { type: 'job_failed', job_id: 'job-design', message: 'out of memory' });
      await settle();

      expect(inWindow('[data-testid="design-failed"]')?.textContent).toContain('out of memory');
      expect(inWindow('[data-testid="design-status"]')).toBeNull();
      await typeInto(description(), 'A calm man');
      expect(make().disabled).toBe(false);
    });

    it('says how far a stopped job got', async () => {
      await openNewVoice(layoutName);
      await runJob(2);

      lastEventSource().emit('job_cancelled', { type: 'job_cancelled', job_id: 'job-design' });
      await settle();

      expect(inWindow('[data-testid="design-stopped"]')?.textContent).toContain('2 of 3');
      expect(inWindow('[data-testid="design-status"]')).toBeNull();
    });
  });

  describe('the candidates', () => {
    it('lists the first voices with their length, Play, a radio and no batch heading', async () => {
      await openNewVoice(layoutName, { design: FIRST });

      expect(rows().map(rowName)).toEqual(['Voice 1', 'Voice 2', 'Voice 3']);
      expect(rows()[0].querySelector('.cand-len')?.textContent).toBe('0:07');
      expect(rows()[0].querySelector('[aria-label="Play Voice 1"]')).not.toBeNull();
      expect(inWindow('[data-testid="candidate-count"]')?.textContent).toBe('3');
      expect(inWindow('[data-testid="candidate-batch"]')).toBeNull();
      expect(make().textContent).toContain('Make 3 more');
      expect(inWindow('[data-testid="save-bar"]')).not.toBeNull();
    });

    it('lists the newest batch first under its description, numbers in the order they were made', async () => {
      await openNewVoice(layoutName, { design: TWO_BATCHES });

      expect(rows().map(rowName)).toEqual(['Voice 4', 'Voice 5', 'Voice 6', 'Voice 1', 'Voice 2', 'Voice 3']);
      expect(Array.from(win()!.querySelectorAll('[data-testid="candidate-batch"]')).map((heading) => heading.textContent)).toEqual(['A deep, gravelly man', 'A warm woman in her thirties']);
    });

    it('adds 3 more with the same description and line, and keeps the earlier ones', async () => {
      await openNewVoice(layoutName, { design: FIRST });
      await typeInto(description(), 'A deep, gravelly man');

      await click(make());
      expect(voiceApiMocks.submitVoiceDesign).toHaveBeenCalledWith({ model: 'qwen3_tts', description: 'A deep, gravelly man', text: PREVIEW_LINES.English });
      await runJob(2);
      expect(status()).toBe('Making 3 more: 2 of 3');
      expect(rows()).toHaveLength(3);

      voiceApiMocks.getDesignState.mockResolvedValue(TWO_BATCHES);
      lastEventSource().emit('job_completed', { type: 'job_completed', outputs: [] });
      await voicesStore.loadDesign();
      await settle();

      expect(rows()).toHaveLength(6);
      expect(status()).toBeNull();
    });

    it('greys out a voice that cannot be saved and says why', async () => {
      const design = { batches: [batch('b1', 'A warm woman', [candidate(1), candidate(2, { seconds: 2, saveable: false, reason: 'Too short to save as a voice; make the line longer' }), candidate(3)])], unpicked: 2 };
      await openNewVoice(layoutName, { design });

      expect(rows()[1].getAttribute('aria-disabled')).toBe('true');
      expect(radio(2).disabled).toBe(true);
      expect(rows()[1].querySelector('[data-testid="candidate-reason"]')?.textContent).toContain('Too short');
      expect(rows()[1].querySelector('[aria-label="Play Voice 2"]')).not.toBeNull();
      expect(radio(1).disabled).toBe(false);
    });

    it('shows a saved voice as Saved as its name, and it cannot be chosen again', async () => {
      const design = { batches: [batch('b1', 'A warm woman', [candidate(1), candidate(2, { saved_as: 'Morgan' }), candidate(3)])], unpicked: 2 };
      await openNewVoice(layoutName, { design });

      expect(rows()[1].querySelector('.cand-saved')?.textContent).toContain('Saved as Morgan');
      expect(inWindow('input[aria-label="Choose Voice 2"]')).toBeNull();
      expect(rows()[1].getAttribute('aria-disabled')).toBeNull();
    });

    it('announces voices that arrive while the tab is open', async () => {
      await openNewVoice(layoutName);
      voiceApiMocks.getDesignState.mockResolvedValue(FIRST);

      await voicesStore.loadDesign();
      await settle();

      expect(inWindow('.sr-only[role="status"]')?.textContent).toContain('3 voices');
    });
  });

  describe('Save voice', () => {
    it('needs a chosen voice and a name', async () => {
      await openNewVoice(layoutName, { design: FIRST });
      expect(saveButton()?.disabled).toBe(true);

      await click(radio(2));
      expect(saveButton()?.disabled).toBe(true);
      expect(rows()[1].getAttribute('data-selected')).toBe('true');

      await typeInto(nameField(), '  ');
      expect(saveButton()?.disabled).toBe(true);
      await typeInto(nameField(), 'Morgan');
      expect(saveButton()?.disabled).toBe(false);
    });

    it('saves the chosen voice, chooses it, closes the window and says so', async () => {
      voiceApiMocks.saveCandidate.mockResolvedValue(MORGAN);
      await openNewVoice(layoutName, { design: FIRST });
      await click(radio(2));
      await typeInto(nameField(), 'Morgan');

      await click(saveButton());

      expect(voiceApiMocks.saveCandidate).toHaveBeenCalledWith('cand-2', 'Morgan');
      expect(draft.state.voiceMode).toBe('saved');
      expect(draft.state.savedVoice).toBe('morgan');
      expect(win()).toBeNull();
      expect(toasts.map((toast) => toast.message)).toContain('Saved voice "Morgan"');
    });

    it('saves with Enter in the name field', async () => {
      voiceApiMocks.saveCandidate.mockResolvedValue(MORGAN);
      await openNewVoice(layoutName, { design: FIRST });
      await click(radio(1));
      await typeInto(nameField(), 'Morgan');

      nameField().dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
      await settle();

      expect(voiceApiMocks.saveCandidate).toHaveBeenCalledWith('cand-1', 'Morgan');
    });

    it('keeps the window open and shows a taken name under the field', async () => {
      voiceApiMocks.saveCandidate.mockRejectedValue(new ApiError('conflict', 409, 'name taken'));
      await openNewVoice(layoutName, { design: FIRST });
      await click(radio(2));
      await typeInto(nameField(), 'Morgan');

      await click(saveButton());

      expect(win()).not.toBeNull();
      expect(inWindow('#nv-name-error')?.textContent).toBe('name taken');
      expect(nameField().getAttribute('aria-invalid')).toBe('true');
      expect(draft.state.voiceMode).not.toBe('saved');
      await typeInto(nameField(), 'Morgan 2');
      expect(inWindow('#nv-name-error')).toBeNull();
    });

    it('reads the list again and says so when the voice is gone', async () => {
      voiceApiMocks.saveCandidate.mockRejectedValue(new ApiError('missing', 404, 'No such candidate.'));
      await openNewVoice(layoutName, { design: FIRST });
      await click(radio(2));
      await typeInto(nameField(), 'Morgan');
      voiceApiMocks.getDesignState.mockClear();
      voiceApiMocks.getDesignState.mockResolvedValue({ batches: [batch('b9', 'Another run', [candidate(7)])], unpicked: 1 });

      await click(saveButton());

      expect(voiceApiMocks.getDesignState).toHaveBeenCalledOnce();
      expect(inWindow('#nv-name-error')?.textContent).toContain('no longer kept');
      expect(rows().map(rowName)).toEqual(['Voice 7']);
      expect(saveButton()?.disabled).toBe(true);
    });
  });

  describe('Clear voices', () => {
    it('asks first, then empties the list', async () => {
      voiceApiMocks.clearCandidates.mockResolvedValue({ status: 'cleared' });
      await openNewVoice(layoutName, { design: TWO_BATCHES });

      await click(inWindow('[data-action="clear-voices"]'));
      expect(document.querySelector('[role="alertdialog"]')?.textContent).toContain('6 candidates');
      await answerConfirm(true);

      expect(voiceApiMocks.clearCandidates).toHaveBeenCalledOnce();
      expect(rows()).toHaveLength(0);
      expect(inWindow('[data-testid="save-bar"]')).toBeNull();
      expect(inWindow('[data-action="clear-voices"]')).toBeNull();
    });

    it('keeps everything when the question is declined', async () => {
      await openNewVoice(layoutName, { design: FIRST });

      await click(inWindow('[data-action="clear-voices"]'));
      await answerConfirm(false);

      expect(voiceApiMocks.clearCandidates).not.toHaveBeenCalled();
      expect(rows()).toHaveLength(3);
    });

    it('counts only the candidates that were not saved', async () => {
      const design = { batches: [batch('b1', 'A warm woman', [candidate(1), candidate(2, { saved_as: 'Morgan' }), candidate(3)])], unpicked: 2 };
      await openNewVoice(layoutName, { design });

      await click(inWindow('[data-action="clear-voices"]'));

      expect(document.querySelector('[role="alertdialog"]')?.textContent).toContain('2 candidates');
      await answerConfirm(false);
    });

    it('is off while a job makes voices', async () => {
      await openNewVoice(layoutName, { design: FIRST });
      expect((inWindow('[data-action="clear-voices"]') as HTMLButtonElement).disabled).toBe(false);

      await runJob(1);

      expect((inWindow('[data-action="clear-voices"]') as HTMLButtonElement).disabled).toBe(true);
    });

    it('shows the server refusal while a job makes voices, and keeps the list', async () => {
      voiceApiMocks.clearCandidates.mockRejectedValue(new ApiError('conflict', 409, 'busy'));
      await openNewVoice(layoutName, { design: FIRST });

      await click(inWindow('[data-action="clear-voices"]'));
      await answerConfirm(true);

      expect(toasts.map((toast) => toast.message)).toContain('busy');
      expect(rows()).toHaveLength(3);
    });
  });

  describe('the layout', () => {
    it('folds the description on a phone once there are candidates, and opens it on a tap', async () => {
      await openNewVoice(layoutName, { design: FIRST });
      newVoiceForm.description = 'A warm woman in her thirties';
      await settle();

      const fold = inWindow('[data-testid="description-fold"]');
      if (layoutName === 'compact') {
        expect(fold?.textContent).toContain('A warm woman in her thirties');
        expect(inWindow('#nv-desc')).toBeNull();
        expect(make().textContent).toContain('Make 3 more');
        await click(fold);
        expect(description().value).toBe('A warm woman in her thirties');
      } else {
        expect(fold).toBeNull();
        expect(description()).not.toBeNull();
      }
    });
  });
});

describe('the New voice form', () => {
  it('returns the default line until one is written, and forgets both on clear', () => {
    expect(newVoiceForm.lineOr('default')).toBe('default');
    newVoiceForm.setLine('');
    expect(newVoiceForm.lineOr('default')).toBe('');
    newVoiceForm.description = 'A calm man';

    newVoiceForm.clear();

    expect(newVoiceForm.lineOr('default')).toBe('default');
    expect(newVoiceForm.description).toBe('');
  });
});
