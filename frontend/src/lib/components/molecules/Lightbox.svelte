<script lang="ts">
  import type { GalleryAsset } from '$lib/types';

  interface Props {
    assets: GalleryAsset[];
    currentIndex: number;
    open: boolean;
    onclose: () => void;
    onnavigate: (index: number) => void;
  }

  let { assets, currentIndex, open, onclose, onnavigate }: Props = $props();

  const asset = $derived(assets[currentIndex] ?? null);
  const hasPrev = $derived(currentIndex > 0);
  const hasNext = $derived(currentIndex < assets.length - 1);

  $effect(() => {
    if (!open) return;
    function handleKeydown(e: KeyboardEvent): void {
      if (e.key === 'Escape') { e.preventDefault(); onclose(); }
      if (e.key === 'ArrowLeft' && hasPrev) { e.preventDefault(); onnavigate(currentIndex - 1); }
      if (e.key === 'ArrowRight' && hasNext) { e.preventDefault(); onnavigate(currentIndex + 1); }
    }
    document.addEventListener('keydown', handleKeydown);
    return () => document.removeEventListener('keydown', handleKeydown);
  });
</script>

{#if open && asset}
  <div class="fixed inset-0 z-50" data-testid="lightbox">
    <!-- Backdrop -->
    <button
      type="button"
      class="absolute inset-0 bg-black/90"
      onclick={onclose}
      aria-label="Close fullscreen viewer"
      tabindex="-1"
    ></button>

    <!-- Dialog container -->
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Fullscreen viewer"
      class="relative flex h-full w-full items-center justify-center p-6 pointer-events-none"
    >
      <!-- Close button -->
      <button
        type="button"
        class="surface-overlay-action pointer-events-auto absolute right-6 top-6 z-10 rounded-md px-3 py-2 text-sm"
        onclick={onclose}
      >Close</button>

      <!-- Previous button (always rendered; disabled at first asset) -->
      <button
        type="button"
        class="surface-overlay-action pointer-events-auto absolute left-6 top-1/2 z-10 -translate-y-1/2 rounded-md p-3 disabled:cursor-not-allowed disabled:opacity-40"
        aria-label="Previous asset"
        disabled={!hasPrev}
        aria-disabled={!hasPrev}
        onclick={() => hasPrev && onnavigate(currentIndex - 1)}
      >
        <svg class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" d="M15 19l-7-7 7-7" />
        </svg>
      </button>

      <!-- Next button (always rendered; disabled at last asset) -->
      <button
        type="button"
        class="surface-overlay-action pointer-events-auto absolute right-6 top-1/2 z-10 -translate-y-1/2 rounded-md p-3 disabled:cursor-not-allowed disabled:opacity-40"
        aria-label="Next asset"
        disabled={!hasNext}
        aria-disabled={!hasNext}
        onclick={() => hasNext && onnavigate(currentIndex + 1)}
      >
        <svg class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7" />
        </svg>
      </button>

      <!-- Media content -->
      {#if asset.media_type === 'video'}
        <!-- svelte-ignore a11y_media_has_caption because gallery outputs do not include caption tracks -->
        <video
          src={asset.url}
          controls
          class="pointer-events-auto relative z-10 max-h-full max-w-full object-contain"
        ></video>
      {:else}
        <img
          src={asset.url}
          alt={asset.filename}
          class="pointer-events-auto relative z-10 max-h-full max-w-full object-contain"
        >
      {/if}
    </div>
  </div>
{/if}
