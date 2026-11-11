import type { ImageModelDefaults, ModelOption, WorkspaceContext } from '$lib/types';

/** What the context says about an image model; undefined when it does not list the model. */
function defaultsOf(context: WorkspaceContext | null, modelId: string): ImageModelDefaults | undefined {
  return context?.image_model_defaults?.[modelId];
}

/** Whether a model can edit an image: it can redraw a painted area (`supports_mask_edit`, a fact of the model's metadata). */
export function canEditWith(context: WorkspaceContext | null, modelId: string): boolean {
  return defaultsOf(context, modelId)?.supports_mask_edit === true;
}

/** Whether a model can change the whole picture from the prompt alone, with nothing painted: it takes reference images (`max_reference_images`). */
export function editsWholePicture(context: WorkspaceContext | null, modelId: string): boolean {
  return canEditWith(context, modelId) && (defaultsOf(context, modelId)?.max_reference_images ?? 0) > 0;
}

/** The models that can edit, in the order the context lists them. */
export function editableModels(context: WorkspaceContext | null, models: readonly ModelOption[]): ModelOption[] {
  return models.filter((model) => canEditWith(context, model.id));
}
