<script lang="ts">
  import Icon, { type IconName } from './Icon.svelte';

  interface Props {
    icon: IconName;
    /** What the pull goes to, while it has not gone far enough. */
    label: string;
    /** What it reads once letting go would go there. */
    readyLabel: string;
    /** Letting go now changes page. */
    ready: boolean;
    /** How far the pull is toward the threshold, from 0 to 1: the hint fades in with it. */
    progress: number;
    /** The edge the hint sits at: the foot of a surface pulled up, its top when pulled down. */
    placement: 'top' | 'bottom';
  }

  let { icon, label, readyLabel, ready, progress, placement }: Props = $props();
</script>

<span class="pull-hint {placement}" class:ready style:opacity={progress} aria-hidden="true">
  <Icon name={icon} size={14} />{ready ? readyLabel : label}
</span>

<style>
  .pull-hint { position: absolute; left: 50%; z-index: 4; display: inline-flex; align-items: center; gap: 6px; height: 32px; padding: 0 14px; border-radius: 16px; translate: -50% 0; background: color-mix(in srgb, var(--color-bg-overlay) 85%, transparent); font-size: var(--text-ui); font-weight: 600; white-space: nowrap; color: var(--color-text-secondary); pointer-events: none; }
  .pull-hint.bottom { bottom: 16px; }
  .pull-hint.top { top: 16px; }
  .pull-hint.ready { background: var(--color-primary-main); color: var(--color-primary-ink); }
</style>
