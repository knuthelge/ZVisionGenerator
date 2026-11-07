<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import type { IconName } from '$lib/components/atoms/Icon.svelte';
  import { FileThumb } from '$lib/components/molecules';

  interface Props {
    /** The uploaded file the row holds, if any; it shows before the path. */
    file: File | null;
    /** The image's absolute path on the server's machine; empty or null when the row holds no path. */
    path: string | null;
    alt: string;
    /** The glyph shown while there is neither a file nor a path. */
    emptyIcon?: IconName;
  }

  let { file, path, alt, emptyIcon = 'reference' }: Props = $props();

  let fileUrl = $state<string | null>(null);

  // Revoke the object URL whenever the file changes or the thumbnail unmounts.
  $effect(() => {
    const made = file ? URL.createObjectURL(file) : null;
    fileUrl = made;
    return () => { if (made) URL.revokeObjectURL(made); };
  });
</script>

{#if fileUrl}
  <img src={fileUrl} {alt}>
{:else if path}
  <span class="path-thumb" role="img" aria-label={alt}><FileThumb purpose="reference_image" {path} kind="image" iconSize={16} fallbackIcon={emptyIcon} /></span>
{:else}
  <Icon name={emptyIcon} size={16} />
{/if}

<style>
  .path-thumb { display: block; width: 100%; height: 100%; }
</style>
