<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import type { IconName } from '$lib/components/atoms/Icon.svelte';

  interface Props {
    /** The uploaded file the row holds, if any; it shows before the asset. */
    file: File | null;
    /** The gallery asset's URL when the row's path points at one. */
    url: string | null;
    alt: string;
    /** The glyph shown while there is neither a file nor a URL. */
    emptyIcon?: IconName;
  }

  let { file, url, alt, emptyIcon = 'reference' }: Props = $props();

  let fileUrl = $state<string | null>(null);

  // Revoke the object URL whenever the file changes or the thumbnail unmounts.
  $effect(() => {
    const made = file ? URL.createObjectURL(file) : null;
    fileUrl = made;
    return () => { if (made) URL.revokeObjectURL(made); };
  });

  const src = $derived(fileUrl ?? url);
</script>

{#if src}<img {src} {alt}>{:else}<Icon name={emptyIcon} size={16} />{/if}
