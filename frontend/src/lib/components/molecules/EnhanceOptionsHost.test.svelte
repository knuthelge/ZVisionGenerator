<script lang="ts">
  import type { EnhanceAxis, EnhanceSettings, WorkflowMode } from '$lib/types';
  import EnhanceOptions from './EnhanceOptions.svelte';

  interface Props {
    axes: EnhanceAxis[];
    initial: EnhanceSettings;
    mode?: WorkflowMode;
    onchange?: (settings: EnhanceSettings) => void;
  }

  let { axes, initial, mode = 'image', onchange }: Props = $props();

  // Holds the settings the way the Workspace does, so a change flows back into the options.
  // svelte-ignore state_referenced_locally
  let settings = $state<EnhanceSettings>({ ...initial });
</script>

<EnhanceOptions
  {axes}
  {settings}
  {mode}
  onchange={(next) => { settings = next; onchange?.($state.snapshot(next)); }}
/>
