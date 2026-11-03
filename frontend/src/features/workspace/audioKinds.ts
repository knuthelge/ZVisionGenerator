import type { AudioWorkflow } from '$lib/types';
import type { KindInfo } from './kindInfo';

/** How each kind of the Audio tab reads in the Kind control and the phone's Kind popover. */
export const KIND_INFO: Record<AudioWorkflow, KindInfo> = {
  music: { label: 'Music', icon: 'music', description: 'Songs with lyrics' },
  speech: { label: 'Speech', icon: 'speech', description: 'Text to speech' },
  sound: { label: 'Sound', icon: 'waveform', description: 'Ambience and effects', title: 'Ambience and sound effects' },
};
