<script lang="ts">
  import { bandColour, type MixTool } from './adjustTools';

  interface Props {
    band: MixTool['band'];
    /** The diameter in pixels. */
    size?: number;
  }

  let { band, size = 12 }: Props = $props();
</script>

<!-- A band's colour as a dot, with a faint ring so a dark band still has an edge. The colour comes from the band token. -->
<span class="adj-swatch" style:--band={bandColour(band)} style:width="{size}px" style:height="{size}px" aria-hidden="true"></span>

<style>
  .adj-swatch { display: inline-block; flex: none; border-radius: 50%; background: var(--band); box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--color-text-primary) 20%, transparent); }
</style>
