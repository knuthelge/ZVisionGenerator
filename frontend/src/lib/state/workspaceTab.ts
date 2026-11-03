import { startImageSet } from './imageSlots';
import type { AudioWorkflow, DraftState, NavTab, VideoKind, Workflow, WorkspaceTab, WorkspaceWorkflow } from '$lib/types';

/** The tabs that hold images and a text prompt, which every workflow of Image and Video belongs to. */
export type MediaTab = Exclude<WorkspaceTab, 'audio'>;

/** The tabs of the top bar, in order. */
export const WORKSPACE_TABS: readonly { id: NavTab; label: string }[] = [
  { id: 'image', label: 'Image' },
  { id: 'video', label: 'Video' },
  { id: 'audio', label: 'Audio' },
];

/** Whether a workflow belongs to the audio mode. */
export function isAudioWorkflow(workflow: WorkspaceWorkflow): workflow is AudioWorkflow {
  return workflow === 'music' || workflow === 'speech' || workflow === 'sound';
}

/** The tabs to offer: Image and Video always, and Audio only when the server lists a music, speech or sound workflow (a model of that kind runs on this machine). */
export function visibleTabs(workflows: readonly WorkspaceWorkflow[] | null): readonly { id: NavTab; label: string }[] {
  return WORKSPACE_TABS.filter((tab) => tab.id !== 'audio' || (workflows !== null && workflows.some(isAudioWorkflow)));
}

/** The kinds of the Video tab to offer: New always, Extend only when the server lists the `extend` workflow. One kind means the Kind control is not shown. */
export function videoKinds(workflows: readonly WorkspaceWorkflow[] | null): readonly VideoKind[] {
  return workflows !== null && workflows.includes('extend') ? ['new', 'extend'] : ['new'];
}

/** The kind of the Video tab a workflow is, or null for a workflow of another tab. */
export function videoKindOf(workflow: WorkspaceWorkflow): VideoKind | null {
  if (workflow === 'extend') return 'extend';
  return workflow === 'txt2vid' || workflow === 'img2vid' ? 'new' : null;
}

/** The tab of the top bar a workflow belongs to: Extend is a kind of Video. */
export function navTab(workflow: WorkspaceWorkflow): NavTab {
  if (isAudioWorkflow(workflow)) return 'audio';
  return videoKindOf(workflow) !== null ? 'video' : 'image';
}

/**
 * The slot a workflow keeps its images and audio track in: like `navTab`, but the Extend kind has a slot of its own (its
 * track), so it is told from Video's new clips. An audio workflow has its own and never falls to Image.
 */
export function workspaceTab(workflow: Workflow): MediaTab;
export function workspaceTab(workflow: WorkspaceWorkflow): WorkspaceTab;
export function workspaceTab(workflow: WorkspaceWorkflow): WorkspaceTab {
  if (isAudioWorkflow(workflow)) return 'audio';
  if (workflow === 'extend') return 'extend';
  return workflow === 'txt2vid' || workflow === 'img2vid' ? 'video' : 'image';
}

/**
 * The workflow a slot runs: Extend is `extend`; Image and Video start from an image (`img2img`, `img2vid`) when the
 * tab's start image is set and the model takes one, and from text (`txt2img`, `txt2vid`) otherwise.
 */
export function workflowFor(
  tab: MediaTab,
  state: Pick<DraftState, 'referenceImagePath' | 'referenceImageUpload' | 'firstFramePath' | 'firstFrameUpload'>,
  takesStart: boolean,
): Workflow {
  if (tab === 'extend') return 'extend';
  const fromImage = takesStart && startImageSet(state, tab);
  if (tab === 'image') return fromImage ? 'img2img' : 'txt2img';
  return fromImage ? 'img2vid' : 'txt2vid';
}
