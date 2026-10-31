import type { ImagesView } from './imagesView';

/** How the Images section header reads: muted when nothing is used, active when images are, error when a used image is gone. */
export type ImagesTone = 'muted' | 'active' | 'error';

export interface ImagesSummary {
  text: string;
  tone: ImagesTone;
}

function _referencesLabel(count: number): string {
  return `${count} ${count === 1 ? 'reference' : 'references'}`;
}

function _imageSummary(view: ImagesView): ImagesSummary {
  const start = view.start?.set ? view.start : null;
  const references = view.references?.rows ?? [];
  const parts = [start?.name ?? null, references.length > 0 ? _referencesLabel(references.length) : null].filter((part): part is string => part !== null && part !== '');
  if (parts.length === 0) return { text: 'None', tone: 'muted' };
  const gone = start?.gone === true || references.some((row) => row.gone);
  return { text: parts.join(' · '), tone: gone ? 'error' : 'active' };
}

function _videoSummary({ start, last }: ImagesView): ImagesSummary {
  // The first frame is what makes the clip start from an image; a last frame kept without one is not used.
  if (start === null || !start.set) return { text: 'None', tone: 'muted' };
  const lastUsed = last?.used === true && last.set;
  return { text: lastUsed ? `${start.name} → ${last.name}` : start.name, tone: start.gone || (lastUsed && last.gone) ? 'error' : 'active' };
}

/**
 * Say what Generate will use from the Images section: the start image and the references in Image ("pond.png",
 * "pond.png · 2 references", "2 references"), the first and last frame in Video ("dawn.png", "dawn.png → dusk.png"), or
 * *None*. A used image that is gone turns the summary to an error, so a folded section still shows why Generate is off.
 * A last frame without a first frame is not used, so it reads as *None*.
 */
export function imagesSummary(view: ImagesView): ImagesSummary {
  return view.tab === 'image' ? _imageSummary(view) : _videoSummary(view);
}
