<script lang="ts">
  import { errorText } from '$lib/api/client';
  import { Button, Icon } from '$lib/components/atoms';
  import { acceptsPageShortcut, hasOpenModal, isTyping } from '$lib/keyboard';
  import { itemNoun, reviewDecisions, reviewHasSound, reviewLine, reviewVideoUrl, type ReviewSize } from '$lib/state/jobView';
  import type { ActiveJobState } from '$lib/types';
  import ActionBar from './ActionBar.svelte';
  import Alert from './Alert.svelte';

  type Decision = 'approve' | 'discard';
  type DecisionCallback = (jobId: string) => void | Promise<unknown>;

  interface Props {
    /** The job whose small render waits for a decision. */
    job: ActiveJobState;
    onapprove: DecisionCallback;
    ondiscard: DecisionCallback;
  }

  let { job, onapprove, ondiscard }: Props = $props();

  const KEYS: Record<string, Decision> = { y: 'approve', n: 'discard' };
  const decisions = $derived(reviewDecisions(job));
  const offline = $derived(job.connection === 'reconnecting');

  const videoUrl = $derived(reviewVideoUrl(job));
  const noun = $derived(itemNoun(job));
  // Sound is for jobs submitted with Audio on. Without it the video is muted and has no volume control: the native
  // controls cannot hide the volume, so a plain play/pause button stands in for them.
  const sound = $derived(reviewHasSound(job));

  let naturalWidth = $state(0);
  let naturalHeight = $state(0);
  let videoWidth = $state(0);
  let videoHeight = $state(0);
  let videoPaused = $state(true);
  let videoElement = $state<HTMLVideoElement | null>(null);
  let root = $state<HTMLElement | null>(null);
  let failedUrl = $state<string | null>(null);
  // The decision sent and the reason the server refused one; both belong to the review they were made for. Each
  // review has its own iteration and attempt (a repeat after a discard is another attempt of the same iteration)
  // and, when it has a preview, its own preview version.
  let sending = $state<{ jobId: string; render: string | null; iteration: number; attempt: number } | null>(null);
  let rejected = $state<{ jobId: string; render: string | null; iteration: number; attempt: number; reason: string } | null>(null);

  const mediaUrl = $derived(videoUrl ?? job.previewUrl);
  const width = $derived(videoUrl ? videoWidth : naturalWidth);
  const height = $derived(videoUrl ? videoHeight : naturalHeight);
  const size = $derived<ReviewSize | null>(width > 0 && height > 0 ? { width, height } : null);
  const line = $derived(reviewLine(job, size));
  const iteration = $derived(job.review?.iteration ?? 0);
  const attempt = $derived(job.review?.attempt ?? 1);
  const busy = $derived(sending?.jobId === job.job_id && sending.render === mediaUrl && sending.iteration === iteration && sending.attempt === attempt);
  const refusal = $derived(
    rejected?.jobId === job.job_id && rejected.render === mediaUrl && rejected.iteration === iteration && rejected.attempt === attempt ? rejected : null
  );
  const shownUrl = $derived(mediaUrl && failedUrl !== mediaUrl ? mediaUrl : null);

  async function decide(action: Decision): Promise<void> {
    if (busy || offline || !decisions.includes(action)) return;
    const callback = action === 'approve' ? onapprove : ondiscard;
    const target = { jobId: job.job_id, render: mediaUrl, iteration, attempt };
    rejected = null;
    sending = target;
    try {
      const result = callback(job.job_id);
      if (result) await result;
      // Left set on success: the render stays until the server's job_resumed event clears the review.
    } catch (error) {
      sending = null;
      rejected = { ...target, reason: errorText(error, 'Try again.') };
    }
  }

  function togglePlayback(): void {
    if (!videoElement) return;
    if (videoElement.paused) void videoElement.play()?.catch(() => {});
    else videoElement.pause();
  }

  // Each review that appears puts focus on Approve, so Enter decides. Focus is not trapped, and it moves only when
  // nothing else has it: not a field being typed in, not a menu or other control the user is working in, and not
  // while a dialog (the asset viewer) is open, since an Enter there must not approve a render nobody has seen.
  $effect(() => {
    void iteration;
    void attempt;
    const active = document.activeElement;
    const idle = !active || active === document.body || active === document.documentElement
      || !!root?.contains(active) || (active instanceof HTMLButtonElement && active.disabled);
    if (!idle || isTyping(active) || hasOpenModal()) return;
    root?.querySelector<HTMLElement>('[data-decision="approve"]:not(:disabled)')?.focus();
  });

  // A new video starts paused on its first frame, and its size is not known until it has loaded.
  $effect(() => {
    void videoUrl;
    videoWidth = 0;
    videoHeight = 0;
    videoPaused = true;
  });

  // Leaving the review (or the video element going away) stops the player and drops its source, so the browser
  // stops buffering a render that is about to be deleted. A changed source needs no such step: the browser abandons
  // the old load itself when `src` changes.
  $effect(() => {
    const video = videoElement;
    if (!video) return;
    return () => {
      video.pause();
      video.removeAttribute('src');
      video.load();
    };
  });

  $effect(() => {
    function handleKeydown(event: KeyboardEvent): void {
      if (!acceptsPageShortcut(event)) return;
      const action = KEYS[event.key.toLowerCase()];
      if (!action || !decisions.includes(action)) return;
      event.preventDefault();
      if (!event.repeat) void decide(action);
    }
    document.addEventListener('keydown', handleKeydown);
    return () => document.removeEventListener('keydown', handleKeydown);
  });
</script>

<div bind:this={root} class="review-stage" role="dialog" aria-modal="false" aria-label="Review" data-review={`${iteration}-${attempt}`}>
  <div class="review-media">
    {#if shownUrl && videoUrl}
      <div class="review-player">
        <video
          bind:this={videoElement}
          src={shownUrl}
          aria-label="The small render, waiting for your decision"
          preload="auto"
          playsinline
          controls={sound}
          muted={!sound}
          bind:paused={videoPaused}
          bind:videoWidth
          bind:videoHeight
          onclick={sound ? undefined : togglePlayback}
          onerror={(event) => { failedUrl = event.currentTarget.getAttribute('src'); }}
        ></video>
        {#if !sound}
          <Button data-playback class="review-playback" aria-label={videoPaused ? 'Play' : 'Pause'} onclick={togglePlayback}>
            <Icon name={videoPaused ? 'play' : 'pause'} size={14} />{videoPaused ? 'Play' : 'Pause'}
          </Button>
        {/if}
      </div>
    {:else if shownUrl}
      <img
        src={shownUrl}
        alt="The small render, waiting for your decision"
        bind:naturalWidth
        bind:naturalHeight
        onerror={(event) => { failedUrl = event.currentTarget.getAttribute('src'); }}
      />
    {:else}
      <p class="ui-help">The small render is not available to show. You can still approve or discard it.</p>
    {/if}
  </div>
  <p class="review-line" role="status">{line}</p>
  {#if refusal}<Alert tone="error" live>The decision didn’t go through. {refusal.reason}</Alert>{/if}
  <ActionBar class="review-actions">
    {#if decisions.includes('approve')}
      <Button variant="primary" data-decision="approve" disabled={busy || offline} aria-label="Approve" title={`Upscale this ${noun} (Y)`} onclick={() => void decide('approve')}>
        <Icon name="check" size={14} />Approve<kbd class="ui-kbd" aria-hidden="true">Y</kbd>
      </Button>
    {/if}
    {#if decisions.includes('discard')}
      <Button data-decision="discard" disabled={busy || offline} aria-label="Discard" title={`Skip this ${noun}: no upscale, nothing saved (N)`} onclick={() => void decide('discard')}>
        <Icon name="close" size={14} />Discard<kbd class="ui-kbd" aria-hidden="true">N</kbd>
      </Button>
    {/if}
  </ActionBar>
</div>

<style>
  .review-stage { display: flex; flex-direction: column; align-items: center; gap: 10px; width: 100%; height: 100%; min-height: 0; padding: 16px; }
  .review-media { display: flex; flex: 1; align-items: center; justify-content: center; width: 100%; min-height: 0; }
  .review-media img, .review-media video { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: var(--radius-md); }
  .review-player { display: flex; flex-direction: column; align-items: center; gap: 8px; max-width: 100%; max-height: 100%; min-height: 0; }
  .review-player video { min-height: 0; flex: 0 1 auto; }
  .review-player :global(.review-playback) { flex: none; }
  .review-line { flex: none; font-size: var(--text-ui); color: var(--color-text-secondary); font-variant-numeric: tabular-nums; }
  .review-stage :global(.review-actions) { flex: none; flex-wrap: wrap; justify-content: center; }
</style>
