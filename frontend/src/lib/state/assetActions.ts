import { parseQuery } from '$lib/prefillParams';
import { requestConfirm } from '$lib/components/molecules/confirm.svelte';
import type { AssetUpscaleFactor, GalleryAsset, JobWorkflow, PrefillParams, UpscaleFactor, UpscaleMethod, Workflow } from '$lib/types';

/** Which generation a start image feeds: an image, a video's first frame, or a video's last frame. */
export type ReferenceTarget = 'image' | 'video' | 'last';

/** Callbacks for the actions every asset surface offers; an absent callback hides its action. */
export interface AssetActionHandlers {
  onpreview?: (asset: GalleryAsset, trigger: HTMLElement) => void;
  onreuse?: (asset: GalleryAsset) => void;
  onreference?: (asset: GalleryAsset, target: ReferenceTarget) => void;
  onextend?: (asset: GalleryAsset) => void;
  onupscale?: (asset: GalleryAsset, factor: UpscaleFactor, method: UpscaleMethod) => void;
  ondelete?: (asset: GalleryAsset, options?: DeleteOptions) => void;
}

const WORKFLOW_LABELS: Record<JobWorkflow, string> = {
  txt2img: 'Text to image',
  img2img: 'Image to image',
  txt2vid: 'Text to video',
  img2vid: 'Image to video',
  extend: 'Extend video',
  upscale: 'Upscale',
};

/** Describe a workflow value in words; unknown values are shown as they are. */
export function workflowLabel(workflow: string): string {
  return WORKFLOW_LABELS[workflow as JobWorkflow] ?? workflow;
}

/** Return the upscale factors an image offers, or none for videos and images without size data. */
export function upscaleFactors(asset: GalleryAsset): AssetUpscaleFactor[] {
  return asset.media_type === 'image' ? (asset.upscale?.factors ?? []) : [];
}

/** Return the SeedVR2 upscale options an image or video offers, or none where SeedVR2 is unavailable. */
export function seedvr2Options(asset: GalleryAsset): AssetUpscaleFactor[] {
  return asset.upscale?.seedvr2?.options ?? [];
}

/** Return the upscale factors an asset offers by any method, smallest first (the chord's keys and the button's tooltip). */
export function offeredUpscaleFactors(asset: GalleryAsset): UpscaleFactor[] {
  return [...new Set([...upscaleFactors(asset), ...seedvr2Options(asset)].map((option) => option.factor))].sort((a, b) => a - b);
}

/** Return the method the plain upscale shortcut runs: SeedVR2 where it is offered, otherwise creative. */
export function plainUpscaleMethod(asset: GalleryAsset): UpscaleMethod {
  return asset.upscale?.seedvr2 ? 'seedvr2' : 'creative';
}

/** Return the upscale option for one factor and method, or undefined when the asset does not offer it. */
export function upscaleOption(asset: GalleryAsset, factor: UpscaleFactor, method: UpscaleMethod): AssetUpscaleFactor | undefined {
  const options = method === 'seedvr2' ? seedvr2Options(asset) : upscaleFactors(asset);
  return options.find((option) => option.factor === factor);
}

export interface DeleteOptions {
  /** Ask before deleting; false skips the prompt (Shift+Delete in the viewer). */
  confirm?: boolean;
}

const REFERENCE_WORKFLOWS: Record<ReferenceTarget, Workflow> = {
  image: 'img2img',
  video: 'img2vid',
  last: 'img2vid',
};

const FALLBACK_REASON_TEXT: Record<string, string> = {
  workflow_media_mismatch: 'The saved workflow does not match this media type; the default workflow is used.',
  missing_reference_image: 'The original start image is unknown; reuse starts a text-to-media run instead.',
  model_not_configured: 'The original model is not configured here; the default model is used.',
  missing_source_video: 'The video this one continued has moved or been deleted; choose it again to extend.',
  extend_unavailable: 'Extending a video is not available here; reuse starts a text-to-video run instead.',
};

/** Return the workspace prefill params carried by an asset's reuse URL; a name that repeats (`reference_path`) is a list, in order. */
export function reuseParams(asset: GalleryAsset): PrefillParams {
  const query = (asset.reuse_workspace_url ?? '').replace(/^#\/[^?]*\??/, '');
  const params = parseQuery(query);
  // A reused image has exactly its source's references, none included: a blank name clears the ones already chosen.
  if (query) params.reference_path ??= '';
  return params;
}

/** Return whether an asset can be used as a start image or frame. */
export function canUseAsReference(asset: GalleryAsset): boolean {
  return asset.media_type === 'image' && Boolean(asset.file_path);
}

/**
 * Return the workspace prefill params that make an asset the start image (or a frame), carrying its prompt.
 *
 * For a video the prompt describes what the first frame shows, so it fills the Scene and leaves the Action for what happens;
 * for an image it is the prompt. A last frame is not the opening scene, so it carries no text.
 *
 * `use_as` names the slot the asset fills (`start` or `last`). The Workspace reads it to drop only that slot's held upload,
 * which tells a Use as from a Reuse when both arrive as address params; it is not a draft field.
 */
export function referenceParams(asset: GalleryAsset, target: ReferenceTarget): Record<string, string> {
  const path = asset.file_path ?? '';
  if (target === 'last') return { workflow: REFERENCE_WORKFLOWS[target], end_image_path: path, use_as: 'last' };
  const params: Record<string, string> = { workflow: REFERENCE_WORKFLOWS[target], image_path: path, use_as: 'start' };
  if (asset.prompt) params[target === 'video' ? 'scene' : 'prompt'] = asset.prompt;
  return params;
}

/** Return whether a video can be extended right now. */
export function canExtend(asset: GalleryAsset): boolean {
  return asset.extend?.available === true;
}

/** Return why a video cannot be extended, or null when it can or the asset does not offer extending. */
export function extendReason(asset: GalleryAsset): string | null {
  return asset.extend && !asset.extend.available ? asset.extend.reason : null;
}

/**
 * Return the workspace prefill params that make a video the source of an extension.
 *
 * The text is left out on purpose: the server describes the source when the Workspace inspects it, and that answer fills
 * Scene (a recorded scene) or Prompt (a single recorded prompt) or leaves both empty. The gallery's own prompt may be only
 * a display fallback such as the file name, and must never reach the boxes.
 */
export function extendParams(asset: GalleryAsset): Record<string, string> {
  const params: Record<string, string> = { workflow: 'extend', source_video: asset.file_path ?? '' };
  const model = asset.reuse_state?.resolved_model;
  if (model) params.model = model;
  params.lora = asset.lora ?? '';
  return params;
}

/** Ask the user to approve deleting one asset; resolves true when they confirm. */
export function confirmDeleteAsset(asset: GalleryAsset): Promise<boolean> {
  return requestConfirm({ question: `Delete "${asset.filename}"?`, info: 'This cannot be undone.', confirmLabel: 'Delete' });
}

/** Describe a backend reuse fallback reason in words. */
export function describeFallbackReason(reason: string): string {
  return FALLBACK_REASON_TEXT[reason] ?? reason;
}

/** Return the width/height aspect ratio of an asset, clamped to a displayable range. */
export function assetAspect(asset: GalleryAsset, min = 0.5, max = 2): number {
  if (!asset.width || !asset.height) return 1;
  return Math.min(max, Math.max(min, asset.width / asset.height));
}
