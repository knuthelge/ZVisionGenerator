import type { IconName } from '$lib/components/atoms/Icon.svelte';
import type { VideoKind } from '$lib/types';

/** How a kind reads in the Kind control and the phone's Kind popover. */
export interface KindInfo {
  label: string;
  icon: IconName;
  description: string;
  title?: string;
}

/** How each kind of the Video tab reads in the Kind control and the phone's Kind popover. */
export const VIDEO_KIND_INFO: Record<VideoKind, KindInfo> = {
  new: { label: 'New', icon: 'film', description: 'From text or a first frame', title: 'A new clip, from text or a first frame' },
  extend: { label: 'Extend', icon: 'extend', description: 'Continue a clip', title: 'Continue a clip' },
};
