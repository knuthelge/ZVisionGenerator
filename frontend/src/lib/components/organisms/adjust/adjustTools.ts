import type { IconName } from '$lib/components/atoms/Icon.svelte';
import { MIX_NAMES, type LightSettings } from '$lib/state/adjustMath';

/** One light or colour control: its key in the settings, its name and its icon. */
export interface LightTool {
  key: keyof LightSettings;
  label: string;
  icon: IconName;
}

/** The tone tools, then the colour tools, in the order every layout shows them. */
export const LIGHT_TOOLS: readonly LightTool[] = [
  { key: 'exposure', label: 'Exposure', icon: 'exposure' },
  { key: 'contrast', label: 'Contrast', icon: 'contrast' },
  { key: 'highlights', label: 'Highlights', icon: 'highlights' },
  { key: 'shadows', label: 'Shadows', icon: 'shadows' },
  { key: 'saturation', label: 'Saturation', icon: 'saturation' },
  { key: 'vibrance', label: 'Vibrance', icon: 'vibrance' },
  { key: 'warmth', label: 'Warmth', icon: 'warmth' },
];

/** How many of the first tools are tone; the rest are colour. */
export const TONE_COUNT = 4;

/** One Colour mixer band: its key in the settings, its name and the id of its swatch token (`--color-band-<band>`). */
export interface MixTool {
  key: keyof LightSettings;
  label: string;
  band: (typeof MIX_NAMES)[number];
}

const MIX_LABELS = ['Red', 'Orange', 'Yellow', 'Green', 'Aqua', 'Blue', 'Purple', 'Magenta'] as const;

/** The eight bands in the order every layout shows them. */
export const MIX_TOOLS: readonly MixTool[] = MIX_NAMES.map((band, index) => ({ key: `mix_${band}`, label: MIX_LABELS[index], band }));

/** The CSS custom property value that carries a band's colour: `var(--color-band-blue)`. */
export const bandColour = (band: MixTool['band']): string => `var(--color-band-${band})`;
