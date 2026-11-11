import { editableModels } from './editModels';
import { workflowMode } from './promptEnhance';
import type { ModelOption, WorkspaceContext, WorkspaceWorkflow } from '$lib/types';

/** The models of a workflow's mode, before the workflow narrows them: the video models for a video, every image model otherwise. */
function modelsOfMode(context: WorkspaceContext, workflow: WorkspaceWorkflow): ModelOption[] {
  return workflowMode(workflow) === 'video' ? context.video_models : context.image_models;
}

/** The models a workflow offers: the video models for a video, the models that can edit for Edit, every image model otherwise. */
export function modelsOf(context: WorkspaceContext, workflow: WorkspaceWorkflow): ModelOption[] {
  const models = modelsOfMode(context, workflow);
  return workflow === 'edit' ? editableModels(context, models) : models;
}

/**
 * Whether a workflow leaves a model out: the context lists it for the workflow's mode and the workflow does not offer it (a model
 * that cannot edit, for Edit). A model the context does not list at all is not judged here, so a prefill naming one is handled as before.
 * The draft takes a recorded model only where this is false; otherwise the workflow keeps its own model and LoRAs.
 */
export function declinesModel(context: WorkspaceContext, workflow: WorkspaceWorkflow, modelId: string): boolean {
  return modelsOfMode(context, workflow).some((model) => model.id === modelId) && !modelsOf(context, workflow).some((model) => model.id === modelId);
}
