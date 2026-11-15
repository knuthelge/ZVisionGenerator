<script lang="ts">
  import type { LightSettings } from '$lib/state/adjustMath';
  import AdjustDial from './AdjustDial.svelte';
  import AdjustSlider from './AdjustSlider.svelte';
  import { signedValueText } from './adjustFormat';
  import { MIX_TOOLS } from './adjustTools';
  import ToolRow from './ToolRow.svelte';

  interface Props {
    light: LightSettings;
    /** The selected band. */
    band: keyof LightSettings;
    /** The names go from the screen and stay for screen readers (the narrow column of a phone held sideways). */
    hideNames?: boolean;
    onselect: (band: keyof LightSettings) => void;
    onchange: (key: keyof LightSettings, value: number) => void;
  }

  let { light, band, hideNames = false, onselect, onchange }: Props = $props();

  const selected = $derived(MIX_TOOLS.find((item) => item.key === band) ?? MIX_TOOLS[0]);
</script>

<ToolRow label="Colour mixer" selected={band}>
  {#each MIX_TOOLS as item (item.key)}
    <AdjustDial
      band={item.band}
      label={item.label}
      name="{item.label} {signedValueText(light[item.key])}"
      value={light[item.key]}
      pressed={item.key === band}
      hideName={hideNames}
      onclick={() => onselect(item.key)}
    />
  {/each}
</ToolRow>
<AdjustSlider
  large
  label="{selected.label} saturation"
  band={selected.band}
  value={light[selected.key]}
  oninput={(value) => onchange(selected.key, value)}
  onreset={() => onchange(selected.key, 0)}
/>
