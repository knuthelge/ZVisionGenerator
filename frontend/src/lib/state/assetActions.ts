import { requestConfirm } from '$lib/components/molecules/confirm.svelte';
import type { AssetUpscaleFactor, GalleryAsset, JobWorkflow, UpscaleFactor, Workflow } from '$lib/types';

/** Which generation a reference image feeds. */
export type ReferenceTarget = 'image' | 'video';

/** Callbacks for the actions every asset surface offers; an absent callback hides its action. */
export interface AssetActionHandlers {
  onpreview?: (asset: GalleryAsset, trigger: HTMLElement) => void;
  onreuse?: (asset: GalleryAsset) => void;
  onreference?: (asset: GalleryAsset, target: ReferenceTarget) => void;
  onupscale?: (asset: GalleryAsset, factor: UpscaleFactor) => void;
  ondelete?: (asset: GalleryAsset, options?: DeleteOptions) => void;
}

const WORKFLOW_LABELS: Record<JobWorkflow, string> = {
  txt2img: 'Text to image',
  img2img: 'Image to image',
  txt2vid: 'Text to video',
  img2vid: 'Image to video',
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

/** Return the upscale option for one factor, or undefined when the asset does not offer it. */
export function upscaleFactor(asset: GalleryAsset, factor: UpscaleFactor): AssetUpscaleFactor | undefined {
  return upscaleFactors(asset).find((option) => option.factor === factor);
}

export interface DeleteOptions {
  /** Ask before deleting; false skips the prompt (Shift+Delete in the viewer). */
  confirm?: boolean;
}

const REFERENCE_WORKFLOWS: Record<ReferenceTarget, Workflow> = {
  image: 'img2img',
  video: 'img2vid',
};

const FALLBACK_REASON_TEXT: Record<string, string> = {
  workflow_media_mismatch: 'The saved workflow does not match this media type; the default workflow is used.',
  missing_reference_image: 'The original reference image is unknown; reuse starts a text-to-media run instead.',
  model_not_configured: 'The original model is not configured here; the default model is used.',
};

/** Return the workspace prefill params carried by an asset's reuse URL. */
export function reuseParams(asset: GalleryAsset): Record<string, string> {
  const query = (asset.reuse_workspace_url ?? '').replace(/^#\/[^?]*\??/, '');
  const params: Record<string, string> = {};
  if (query) new URLSearchParams(query).forEach((value, key) => { params[key] = value; });
  return params;
}

/** Return whether an asset can be used as a reference image. */
export function canUseAsReference(asset: GalleryAsset): boolean {
  return asset.media_type === 'image' && Boolean(asset.file_path);
}

/** Return the workspace prefill params that make an asset the reference image. */
export function referenceParams(asset: GalleryAsset, target: ReferenceTarget): Record<string, string> {
  return { workflow: REFERENCE_WORKFLOWS[target], image_path: asset.file_path ?? '' };
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
