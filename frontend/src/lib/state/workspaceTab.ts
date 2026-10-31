import { startImageSet } from './imageSlots';
import type { DraftState, Workflow, WorkspaceTab } from '$lib/types';

/** The tabs of the top bar, in order. */
export const WORKSPACE_TABS: readonly { id: WorkspaceTab; label: string }[] = [
  { id: 'image', label: 'Image' },
  { id: 'video', label: 'Video' },
  { id: 'extend', label: 'Extend Video' },
];

/** The tabs to offer: Image and Video always, and Extend Video only when the server lists the `extend` workflow. */
export function visibleTabs(workflows: readonly Workflow[] | null): readonly { id: WorkspaceTab; label: string }[] {
  return WORKSPACE_TABS.filter((tab) => tab.id !== 'extend' || (workflows !== null && workflows.includes('extend')));
}

/** The tab a workflow belongs to. */
export function workspaceTab(workflow: Workflow): WorkspaceTab {
  if (workflow === 'extend') return 'extend';
  return workflow === 'txt2vid' || workflow === 'img2vid' ? 'video' : 'image';
}

/**
 * The workflow a tab runs: Extend Video is `extend`; Image and Video start from an image (`img2img`, `img2vid`) when the
 * tab's start image is set and the model takes one, and from text (`txt2img`, `txt2vid`) otherwise.
 */
export function workflowFor(
  tab: WorkspaceTab,
  state: Pick<DraftState, 'referenceImagePath' | 'referenceImageUpload' | 'firstFramePath' | 'firstFrameUpload'>,
  takesStart: boolean,
): Workflow {
  if (tab === 'extend') return 'extend';
  const fromImage = takesStart && startImageSet(state, tab);
  if (tab === 'image') return fromImage ? 'img2img' : 'txt2img';
  return fromImage ? 'img2vid' : 'txt2vid';
}
