<script lang="ts">
  import type { LightSettings } from '$lib/state/adjustMath';
  import AdjustDial from './AdjustDial.svelte';
  import AdjustSlider from './AdjustSlider.svelte';
  import { signedValueText } from './adjustFormat';
  import { LIGHT_TOOLS } from './adjustTools';
  import ToolRow from './ToolRow.svelte';

  interface Props {
    light: LightSettings;
    /** The selected tool. */
    tool: keyof LightSettings;
    /** The names go from the screen and stay for screen readers (the narrow column of a phone held sideways). */
    hideNames?: boolean;
    onselect: (tool: keyof LightSettings) => void;
    onchange: (key: keyof LightSettings, value: number) => void;
  }

  let { light, tool, hideNames = false, onselect, onchange }: Props = $props();

  const selected = $derived(LIGHT_TOOLS.find((item) => item.key === tool) ?? LIGHT_TOOLS[0]);
</script>

<ToolRow label="Light & colour tools" selected={tool}>
  {#each LIGHT_TOOLS as item (item.key)}
    <AdjustDial
      icon={item.icon}
      label={item.label}
      name="{item.label} {signedValueText(light[item.key])}"
      value={light[item.key]}
      pressed={item.key === tool}
      hideName={hideNames}
      onclick={() => onselect(item.key)}
    />
  {/each}
</ToolRow>
<AdjustSlider
  large
  label={selected.label}
  icon={selected.icon}
  value={light[selected.key]}
  oninput={(value) => onchange(selected.key, value)}
  onreset={() => onchange(selected.key, 0)}
/>
