<script lang="ts" generics="K extends string">
  import { Segmented } from '$lib/components/molecules';
  import type { KindInfo } from './kindInfo';

  interface Props {
    /** The kinds this machine runs, in the order shown. */
    kinds: readonly K[];
    /** How each kind reads. */
    info: Record<K, KindInfo>;
    value: K;
    disabled?: boolean;
    onchange: (kind: K) => void;
  }

  let { kinds, info, value, disabled = false, onchange }: Props = $props();
</script>

<!-- Music, Speech or Sound; New or Extend: the kind decides which boxes Compose shows and which Settings follow. -->
<Segmented
  label="Kind"
  testId="kind-control"
  {value}
  {disabled}
  options={kinds.map((kind) => ({ value: kind, label: info[kind].label, icon: info[kind].icon, title: info[kind].title }))}
  {onchange}
/>
