import { startImageSet } from './imageSlots';
import type { AudioWorkflow, DraftState, Workflow, WorkspaceTab, WorkspaceWorkflow } from '$lib/types';

/** The tabs that hold images and a text prompt, which every workflow of Image and Video belongs to. */
export type MediaTab = Exclude<WorkspaceTab, 'audio'>;

/** The tabs of the top bar, in order. */
export const WORKSPACE_TABS: readonly { id: WorkspaceTab; label: string }[] = [
  { id: 'image', label: 'Image' },
  { id: 'video', label: 'Video' },
  { id: 'extend', label: 'Extend Video' },
  { id: 'audio', label: 'Audio' },
];

/** Whether a workflow belongs to the audio mode. */
export function isAudioWorkflow(workflow: WorkspaceWorkflow): workflow is AudioWorkflow {
  return workflow === 'music' || workflow === 'speech' || workflow === 'sound';
}

/**
 * The tabs to offer: Image and Video always, Extend Video only when the server lists the `extend` workflow, and Audio only
 * when it lists a music, speech or sound workflow (a model of that kind runs on this machine).
 */
export function visibleTabs(workflows: readonly WorkspaceWorkflow[] | null): readonly { id: WorkspaceTab; label: string }[] {
  return WORKSPACE_TABS.filter((tab) => {
    if (tab.id === 'extend') return workflows !== null && workflows.includes('extend');
    if (tab.id === 'audio') return workflows !== null && workflows.some(isAudioWorkflow);
    return true;
  });
}

/** The tab a workflow belongs to: an audio workflow has its own tab and never falls to Image. */
export function workspaceTab(workflow: Workflow): MediaTab;
export function workspaceTab(workflow: WorkspaceWorkflow): WorkspaceTab;
export function workspaceTab(workflow: WorkspaceWorkflow): WorkspaceTab {
  if (isAudioWorkflow(workflow)) return 'audio';
  if (workflow === 'extend') return 'extend';
  return workflow === 'txt2vid' || workflow === 'img2vid' ? 'video' : 'image';
}

/**
 * The workflow a tab runs: Extend Video is `extend`; Image and Video start from an image (`img2img`, `img2vid`) when the
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
