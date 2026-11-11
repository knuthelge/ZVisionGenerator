import { hasPath } from '$lib/state/imageSlots';
import type { DraftState } from '$lib/types';

/** What an Edit Generate takes besides the draft: the files held for it, and the fields the page derives. */
export interface EditFormInput {
  /** The image uploaded from this device, which wins over a path; null when the image is a path. */
  sourceFile: File | null;
  /** The painted mask's PNG; null when nothing is painted. */
  maskFile: File | null;
  /** The text to run: the prompt, or its Enhanced version when one is in use. */
  prompt: string;
  /** The LoRA list as the page's chips write it. */
  lora: string;
  /** The model takes a quantization level, so the draft's is sent. */
  quantize: boolean;
}

/**
 * The fields an Edit Generate sends, built from the draft: `new FormData(form)` would carry New's fields (size, guidance, scheduler,
 * post-processing, references), none of which an edit uses. Only what the `edit` workflow reads goes: the image (an upload wins
 * over a path), the painted mask with `edit_painted` (Paint) or the extension (Extend), never both, the prompt, the model, LoRAs, quantization, steps, seed and batch.
 */
export function editFormData(state: DraftState, input: EditFormInput): FormData {
  const form = new FormData();
  form.set('mode', 'image');
  form.set('workflow', 'edit');
  if (input.sourceFile !== null) form.set('edit_source_file', input.sourceFile);
  else if (hasPath(state.editSourcePath)) form.set('edit_source_path', state.editSourcePath);
  // The mask is sent only with its marker in the draft, as every held upload is: a file the draft does not mark is not an input.
  if (state.editMode === 'paint' && input.maskFile !== null && state.editMaskUpload !== null) {
    form.set('edit_mask_file', input.maskFile);
    form.set('edit_painted', 'true');
  }
  if (state.editMode === 'extend' && state.editExtendSide !== null) {
    form.set('edit_extend_side', state.editExtendSide);
    form.set('edit_extend_amount', String(state.editExtendAmount));
  }
  form.set('prompt', input.prompt);
  form.set('model', state.model);
  form.set('lora', input.lora);
  if (input.quantize && state.quantize !== null) form.set('quantize', String(state.quantize));
  form.set('steps', String(state.steps));
  if (state.seed !== null) form.set('seed', String(state.seed));
  form.set('runs', String(state.runs));
  return form;
}
