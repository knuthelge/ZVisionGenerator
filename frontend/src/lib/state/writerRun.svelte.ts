import { enhancePrompt, type AudioRequest } from '$lib/api/promptEnhance';
import { draft } from './draft.svelte';
import { confirmEnhancerDownload, downloadedTiers, tierMissing } from './enhancerDownload.svelte';
import { effectiveTier, enhancePhaseMessage } from './promptEnhance';
import { textWriting } from './textWriting.svelte';
import type { EnhancerTier, PromptEnhancerContract } from '$lib/types';

/** The enhancer's first-use download size for *tier*, or null once it is downloaded. */
export function downloadSizeOf(contract: PromptEnhancerContract, tier: EnhancerTier): string | null {
  if (downloadedTiers.has(tier)) return null;
  return (tier === 'smart' ? contract.smart?.download_size_label : contract.download_size_label) ?? null;
}

/** Why the helper cannot run now, or null: no enhancer model, a take being made or submitted. */
export function writerBlock(contract: PromptEnhancerContract, state: { jobsActive: boolean; busy: boolean }): string | null {
  if (contract.model === null) return contract.error ?? 'No prompt enhancer model is configured.';
  if (state.jobsActive) return 'Available when the take is done.';
  if (state.busy) return 'Available once the take is submitted.';
  return null;
}

/**
 * What a finished run hands back: the text the task wrote in its field, every field it wrote (Fill in writes several), the enhancer's notice
 * about it and the enhancer that wrote it (`REPO[@REVISION]`).
 */
export interface WriterResult {
  text: string;
  fields: Record<string, string>;
  warning: string | null;
  model: string | null;
}

/**
 * One run of a writer task (the lyrics and script helpers): asks to download the enhancer when it is missing, streams
 * the request, and keeps the progress, error and notice the helper's panel shows. The panel owns the request; this owns the wait.
 */
export class WriterRun {
  /** The draft field that holds the enhancer this run's helper uses: the shared choice, or Fill in's own. */
  readonly #tierField: 'enhanceModel' | 'describeTier';
  running = $state(false);
  /** The download plan of the tier about to run is being read, or the dialog about it is open. */
  asking = $state(false);
  statusText = $state<string | null>(null);
  errorText = $state<string | null>(null);
  /** What a run that changed nothing says: shown as a note, not as an error. */
  noteText = $state<string | null>(null);
  #abort: AbortController | null = null;
  /** Ends this run's hold on Generate (see `textWriting`). */
  #release: (() => void) | null = null;
  #phase: 'downloading' | 'loading' | 'generating' | 'generating_cpu' | null = null;

  constructor(tierField: 'enhanceModel' | 'describeTier' = 'enhanceModel') {
    this.#tierField = tierField;
  }

  /** The tier a run would use now. */
  tier(contract: PromptEnhancerContract): EnhancerTier {
    return effectiveTier({ enhanceModel: draft.state[this.#tierField] }, contract);
  }

  /**
   * Run *request* (the tier is added) and resolve with what its task wrote in *field*, or null when nothing was written: the user
   * declined the download, stopped it, or the run failed (then `errorText` says why).
   */
  async start(contract: PromptEnhancerContract, request: AudioRequest, field: string): Promise<WriterResult | null> {
    if (this.running || this.asking) return null;
    this.clear();
    const tier = this.tier(contract);
    if (tierMissing(contract, tier)) {
      this.asking = true;
      try {
        const { confirmed, error } = await confirmEnhancerDownload(contract, tier);
        if (error !== null) this.errorText = error;
        if (!confirmed) return null;
      } finally {
        this.asking = false;
      }
    }
    const abort = new AbortController();
    this.#abort = abort;
    this.running = true;
    this.#release = textWriting.begin('Writing');
    this.statusText = enhancePhaseMessage('generating', null);
    let result: WriterResult | null = null;
    try {
      await enhancePrompt(
        { ...request, ...(tier === 'smart' ? { enhance_model: 'smart' as const } : {}) },
        (frame) => {
          if (frame.type === 'status') {
            this.#phase = frame.phase;
            this.statusText = enhancePhaseMessage(frame.phase, downloadSizeOf(contract, tier));
          } else if (frame.type === 'done') {
            result = { text: frame.fields[field] ?? '', fields: frame.fields, warning: frame.warning ?? null, model: frame.model ?? null };
            downloadedTiers.add(tier);
          } else if (frame.type === 'error') {
            if (frame.no_change) this.noteText = frame.detail;
            else this.errorText = frame.detail;
          }
        },
        abort.signal,
      );
    } catch (error) {
      if (!abort.signal.aborted) this.errorText = error instanceof Error ? error.message : 'Prompt enhancement failed.';
    } finally {
      this.running = false;
      this.#release?.();
      this.#release = null;
      this.statusText = null;
      this.#phase = null;
      this.#abort = null;
    }
    return abort.signal.aborted ? null : result;
  }

  /** Forget the last run's error and note. */
  clear(): void {
    this.errorText = null;
    this.noteText = null;
  }

  /** Stop the run. A download or model load cannot be interrupted: it finishes in the background, which the returned note says. */
  stop(): string | null {
    const phase = this.#phase;
    this.#abort?.abort();
    if (phase === 'downloading') return 'Stopped. The enhancer model keeps downloading in the background; the helper is available when it finishes.';
    if (phase === 'loading') return 'Stopped. The enhancer finishes loading in the background; the helper is available in a moment.';
    return null;
  }

  /** Stop the run and hand back idle state (the panel is going away). */
  dispose(): void {
    this.#abort?.abort();
    this.running = false;
    this.#release?.();
    this.#release = null;
    this.statusText = null;
  }
}
