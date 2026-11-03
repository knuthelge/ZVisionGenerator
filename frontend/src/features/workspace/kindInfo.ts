import type { IconName } from '$lib/components/atoms/Icon.svelte';
import type { AudioWorkflow, VideoKind } from '$lib/types';

/** How a kind reads in the Kind control and the phone's Kind popover. */
export interface KindInfo {
  label: string;
  icon: IconName;
  description: string;
  title?: string;
}

/** How each kind of the Video tab reads. */
export const VIDEO_KIND_INFO: Record<VideoKind, KindInfo> = {
  new: { label: 'New', icon: 'film', description: 'From text or a first frame', title: 'A new clip, from text or a first frame' },
  extend: { label: 'Extend', icon: 'extend', description: 'Continue a clip', title: 'Continue a clip' },
};

/** How each kind of the Audio tab reads. */
export const AUDIO_KIND_INFO: Record<AudioWorkflow, KindInfo> = {
  music: { label: 'Music', icon: 'music', description: 'Songs with lyrics' },
  speech: { label: 'Speech', icon: 'speech', description: 'Text to speech' },
  sound: { label: 'Sound', icon: 'waveform', description: 'Ambience and effects', title: 'Ambience and sound effects' },
};

/** The kinds the open tab offers, the open one, and how to switch: Audio's kinds, or Video's New and Extend. */
export interface KindChoice {
  kinds: readonly string[];
  info: Record<string, KindInfo>;
  value: string;
  pick: (kind: string) => void;
}
