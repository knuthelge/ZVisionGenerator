<script lang="ts">
  import Icon from '$lib/components/atoms/Icon.svelte';
  import LauncherLink from './LauncherLink.svelte';
  import type { LinkId } from './types';

  interface Props {
    /** What was checked, e.g. "NVIDIA driver". */
    name: string;
    /** What was found on this computer; null when nothing was. */
    found: string | null;
    /** What is needed, as a sentence. */
    needed: string;
    /** A docs link after the needed line. */
    link?: { id: LinkId; label: string; onopen: () => void };
  }

  let { name, found, needed, link }: Props = $props();
</script>

<div class="reason" data-testid="reason-card">
  <Icon name="circle-x" size={16} />
  <div>
    <span class="reason-name">{name}</span>
    {#if found}<span class="reason-found">{found}</span>{/if}
    <span class="reason-need">{needed}{#if link}{' '}<LauncherLink id={link.id} label={link.label} onclick={link.onopen} />{/if}</span>
  </div>
</div>

<style>
  .reason {
    display: inline-flex;
    align-items: flex-start;
    gap: 10px;
    padding: 9px 14px 9px 11px;
    border-radius: var(--radius-md);
    background: var(--color-bg-surface);
    text-align: left;
    font-size: var(--text-content);
    line-height: 1.4;
  }
  .reason > :global(svg) { flex: none; margin-top: 1px; color: var(--color-error); }
  .reason-name { font-weight: 600; color: var(--color-text-primary); }
  .reason-found { font-family: var(--font-mono); font-size: var(--text-ui); color: var(--color-text-secondary); }
  .reason-need { display: block; font-size: var(--text-ui); color: var(--color-error); }
</style>
