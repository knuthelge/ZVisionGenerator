<script lang="ts">
  import { browsePreviewUrl } from '$lib/api/browse';
  import { Icon } from '$lib/components/atoms';
  import type { IconName } from '$lib/components/atoms/Icon.svelte';
  import type { BrowsePreviewKind, BrowsePurposeName } from '$lib/types';

  interface Props {
    purpose: BrowsePurposeName;
    /** The file's absolute path on the server's machine. */
    path: string;
    /** What the purpose previews; `audio` draws an icon and requests nothing. */
    kind: BrowsePreviewKind;
    /** `last` shows a video's last frame (the Extend source) instead of the still 10% in. */
    frame?: 'last';
    /** The second, on the video's clock, whose frame is shown (the Extend source's point); wins over `frame`. */
    at?: number | null;
    /** The icon's size when the thumbnail is an icon. */
    iconSize?: number;
    /** The icon a file that cannot be previewed shows; a slot passes the one it showed before there were pictures. */
    fallbackIcon?: IconName;
  }

  let { purpose, path, kind, frame, at = null, iconSize = 14, fallbackIcon = 'file' }: Props = $props();

  // The URL that failed, so another file in the same box starts over.
  let failedUrl = $state<string | null>(null);

  const url = $derived(kind === 'audio' ? '' : browsePreviewUrl(purpose, path, at === null ? { frame } : { at }));
  const failed = $derived(failedUrl === url);
</script>

<span class="file-thumb" data-kind={kind} data-testid="file-thumb">
  {#if failed}
    <Icon name={fallbackIcon} size={iconSize} />
  {:else if kind === 'audio'}
    <Icon name="waveform" size={iconSize} />
  {:else}
    <img src={url} alt="" loading="lazy" decoding="async" draggable="false" onerror={() => { failedUrl = url; }}>
    {#if kind === 'video' && frame !== 'last' && at === null}
      <span class="play-badge" data-testid="play-badge" aria-hidden="true"><Icon name="play" size={8} /></span>
    {/if}
  {/if}
</span>

<style>
  .file-thumb { position: relative; width: 100%; height: 100%; display: flex; flex: none; align-items: center; justify-content: center; overflow: hidden; border-radius: var(--radius-xs); background: var(--color-bg-fill); color: var(--color-text-secondary); }
  img { display: block; width: 100%; height: 100%; object-fit: cover; }
  .play-badge { position: absolute; right: 2px; bottom: 2px; display: flex; align-items: center; justify-content: center; width: 14px; height: 14px; border-radius: 50%; background: color-mix(in srgb, var(--color-bg-base) 78%, transparent); color: var(--color-text-primary); }
</style>
