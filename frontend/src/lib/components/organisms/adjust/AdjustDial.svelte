<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import type { IconName } from '$lib/components/atoms/Icon.svelte';
  import BandSwatch from './BandSwatch.svelte';
  import type { MixTool } from './adjustTools';

  interface Props {
    icon?: IconName;
    /** A Colour mixer band: the dial holds the band's swatch instead of an icon. */
    band?: MixTool['band'];
    /** The name under the dial. */
    label: string;
    /** The accessible name when it differs from the label, such as "Exposure +24". */
    name?: string;
    /** The tool's value, -100 to 100: a teal arc around the dial runs clockwise for a plus and anticlockwise for a minus. */
    value?: number;
    /** A toggle button: whether it is the selected one. Absent for a plain action. */
    pressed?: boolean;
    /** The name stays for screen readers and goes from the screen. */
    hideName?: boolean;
    disabled?: boolean;
    onclick: () => void;
  }

  let { icon, band, label, name, value = 0, pressed, hideName = false, disabled = false, onclick }: Props = $props();

  /** The arc's circle: radius and circumference in the ring's 48-unit box. */
  const RADIUS = 22.5;
  const CIRCUMFERENCE = 2 * Math.PI * RADIUS;
  const arc = $derived((Math.min(100, Math.abs(value)) / 100) * CIRCUMFERENCE);
</script>

<button type="button" class="adj-tool" aria-label={name} aria-pressed={pressed} data-set={value !== 0 ? '' : undefined} {disabled} {onclick}>
  <span class="adj-tool-dial">
    {#if value !== 0}
      <svg class="adj-ring" viewBox="0 0 48 48" data-neg={value < 0 ? '' : undefined} aria-hidden="true">
        <circle cx="24" cy="24" r={RADIUS} fill="none" stroke="var(--color-primary-main)" stroke-width="2.5" stroke-linecap="round" stroke-dasharray="{arc} {CIRCUMFERENCE}" />
      </svg>
    {/if}
    {#if band}<BandSwatch {band} size={22} />{:else if icon}<Icon name={icon} size={20} />{/if}
  </span>
  <span class="adj-tool-name" data-hidden={hideName ? '' : undefined}>{label}</span>
</button>

<style>
  .adj-tool { position: relative; display: flex; flex: 1 1 0; min-width: 0; flex-direction: column; align-items: center; gap: 5px; }
  /* The focus ring follows the round dial, not the button's box. */
  .adj-tool:focus-visible { outline: none; }
  .adj-tool:focus-visible .adj-tool-dial { outline: 2px solid var(--color-primary-main); outline-offset: 2px; }
  .adj-tool:disabled { opacity: 0.4; cursor: not-allowed; }
  .adj-tool-dial { position: relative; display: grid; width: var(--dial, 48px); height: var(--dial, 48px); place-items: center; border-radius: 50%; background: var(--color-bg-surface-hover); color: var(--color-text-secondary); transition: background-color 120ms var(--ease-standard), color 120ms var(--ease-standard); }
  .adj-ring { position: absolute; inset: 0; width: 100%; height: 100%; transform: rotate(-90deg); }
  .adj-ring[data-neg] { transform: rotate(-90deg) scaleY(-1); }
  .adj-tool-name { max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-meta); font-weight: 600; color: var(--color-text-muted); }
  .adj-tool-name[data-hidden] { position: absolute; width: 1px; height: 1px; clip-path: inset(50%); }
  .adj-tool[aria-pressed='true'] .adj-tool-dial { background: color-mix(in srgb, var(--color-primary-main) 16%, transparent); color: var(--color-primary-main); }
  .adj-tool[aria-pressed='true'] .adj-tool-name { color: var(--color-text-primary); }
  .adj-tool[data-set]:not([aria-pressed='true']) .adj-tool-dial { color: var(--color-text-primary); }
  @media (hover: none) { .adj-tool:active .adj-tool-dial { background: var(--color-bg-control-hover); } }
</style>
