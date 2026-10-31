<script lang="ts" module>
  export { rememberMascotSpots } from './mascotFlight';
</script>

<script lang="ts">
  import Mascot, { type MascotMood, type MascotTool } from './Mascot.svelte';
  import { arrive, depart } from './mascotFlight';

  interface Props {
    mood?: MascotMood;
    tool?: MascotTool;
    size?: number;
    /** Share (0 to 1) of the active job's steps done; the creating mood's canvas follows it. */
    progress?: number;
    /** Spots sharing a key hand the mascot over to each other as a stream of pixels. */
    travelKey?: string;
    /** Hidden from assistive technology, for a copy that only decorates (the logo). */
    decorative?: boolean;
    class?: string;
  }

  let {
    mood = 'idle',
    tool = 'brush',
    size = 120,
    progress,
    travelKey = 'mascot',
    decorative = false,
    class: extraClass = ''
  }: Props = $props();
</script>

<div
  class={extraClass}
  data-mascot-spot={travelKey}
  in:arrive|global={travelKey}
  out:depart|global={travelKey}
>
  <Mascot {mood} {tool} {size} {progress} {decorative} />
</div>
