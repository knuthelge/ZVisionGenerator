<script lang="ts">
  import { Segmented } from '$lib/components/molecules';
  import type { AudioWorkflow } from '$lib/types';
  import { KIND_INFO } from './audioKinds';

  interface Props {
    /** The kinds this machine runs, in the order shown. */
    kinds: readonly AudioWorkflow[];
    value: AudioWorkflow;
    disabled?: boolean;
    onchange: (kind: AudioWorkflow) => void;
  }

  let { kinds, value, disabled = false, onchange }: Props = $props();
</script>

<!-- Music, Speech or Sound: the kind decides which boxes Compose shows and which Settings follow. -->
<Segmented
  label="Kind"
  testId="kind-control"
  {value}
  {disabled}
  options={kinds.map((kind) => ({ value: kind, label: KIND_INFO[kind].label, icon: KIND_INFO[kind].icon, title: KIND_INFO[kind].title }))}
  {onchange}
/>
