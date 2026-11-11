import { startImageSet } from './imageSlots';
import { AUDIO_KINDS, type AudioWorkflow, type DraftState, type NavTab, type TextTab, type VideoKind, type Workflow, type WorkspaceTab, type WorkspaceWorkflow } from '$lib/types';

/** The tabs that hold images and a text prompt, which every workflow of Image and Video belongs to. */
export type MediaTab = Exclude<WorkspaceTab, 'audio'>;

/** The tabs of the top bar, in order. */
export const WORKSPACE_TABS: readonly { id: NavTab; label: string }[] = [
  { id: 'image', label: 'Image' },
  { id: 'video', label: 'Video' },
  { id: 'audio', label: 'Audio' },
];

/** Whether a workflow belongs to the audio mode. */
export function isAudioWorkflow(workflow: string | undefined): workflow is AudioWorkflow {
  return (AUDIO_KINDS as readonly (string | undefined)[]).includes(workflow);
}

/** The tabs to offer: Image and Video always, and Audio only when the server lists a music, speech or sound workflow (a model of that kind runs on this machine). */
export function visibleTabs(workflows: readonly WorkspaceWorkflow[] | null): readonly { id: NavTab; label: string }[] {
  return WORKSPACE_TABS.filter((tab) => tab.id !== 'audio' || (workflows !== null && workflows.some(isAudioWorkflow)));
}

/** The kinds of the Video tab to offer: New always, Extend only when the server lists the `extend` workflow. One kind means the Kind control is not shown. */
export function videoKinds(workflows: readonly WorkspaceWorkflow[] | null): readonly VideoKind[] {
  return workflows !== null && workflows.includes('extend') ? ['new', 'extend'] : ['new'];
}

/** The kinds of the Image tab: New and Edit, which every machine offers (the contract always lists the `edit` workflow). */
export const IMAGE_KINDS: readonly ('new' | 'edit')[] = ['new', 'edit'];

/** The kind of the Video tab a workflow is, or null for a workflow of another tab. */
export function videoKindOf(workflow: WorkspaceWorkflow): VideoKind | null {
  if (workflow === 'extend') return 'extend';
  return workflow === 'txt2vid' || workflow === 'img2vid' ? 'new' : null;
}

/** The tab of the top bar a workflow belongs to: Extend is a kind of Video and Edit a kind of Image. */
export function navTab(workflow: WorkspaceWorkflow): NavTab {
  if (isAudioWorkflow(workflow)) return 'audio';
  return videoKindOf(workflow) !== null ? 'video' : 'image';
}

/**
 * The slot a workflow keeps its images and audio track in: like `navTab`, but the Extend and Edit kinds have a slot of their
 * own (Extend its track, Edit its image to edit with its paint), so they are told from the new clips and images. An audio
 * workflow has its own and never falls to Image.
 */
export function workspaceTab(workflow: Workflow): MediaTab;
export function workspaceTab(workflow: WorkspaceWorkflow): WorkspaceTab;
export function workspaceTab(workflow: WorkspaceWorkflow): WorkspaceTab {
  if (isAudioWorkflow(workflow)) return 'audio';
  if (workflow === 'extend') return 'extend';
  if (workflow === 'edit') return 'edit';
  return workflow === 'txt2vid' || workflow === 'img2vid' ? 'video' : 'image';
}

/** The tab whose text a workflow writes: Extend writes Video's, every other slot its own. */
export function textTab(workflow: WorkspaceWorkflow): TextTab {
  const tab = workspaceTab(workflow);
  return tab === 'extend' ? 'video' : tab;
}

/**
 * The workflow a slot runs: Extend is `extend` and Edit is `edit`; Image and Video start from an image (`img2img`, `img2vid`) when the
 * tab's start image is set and the model takes one, and from text (`txt2img`, `txt2vid`) otherwise.
 */
export function workflowFor(
  tab: MediaTab,
  state: Pick<DraftState, 'referenceImagePath' | 'referenceImageUpload' | 'firstFramePath' | 'firstFrameUpload'>,
  takesStart: boolean,
): Workflow {
  if (tab === 'extend' || tab === 'edit') return tab;
  const fromImage = takesStart && startImageSet(state, tab);
  if (tab === 'image') return fromImage ? 'img2img' : 'txt2img';
  return fromImage ? 'img2vid' : 'txt2vid';
}
