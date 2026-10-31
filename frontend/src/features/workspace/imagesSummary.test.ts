import { describe, expect, it } from 'vitest';

import type { SlotState } from '$lib/state/imageSlots';
import { workspaceTab } from '$lib/state/workspaceTab';
import type { Workflow } from '$lib/types';
import type { WorkspaceCapabilities } from './capabilities';
import { imagesSummary } from './imagesSummary';
import { imagesView, type ImagesViewInput } from './imagesView';

const file = (name: string): File => new File(['x'], name, { type: 'image/png' });

type Held = ImagesViewInput['files'];
const NO_FILES: Held = { startImage: null, firstFrame: null, lastFrame: null, references: [] };

function state(workflow: Workflow, overrides: Partial<SlotState> & { referenceImagePaths?: string[] } = {}): ImagesViewInput['state'] {
  return {
    workflow,
    referenceImagePath: null, referenceImageUpload: null, firstFramePath: null, firstFrameUpload: null, endImagePath: null, endImageUpload: null,
    referenceImageStrength: 0.65, missingReferencePaths: [], referenceImagePaths: [],
    ...overrides,
  };
}

function summarize(workflow: Workflow, overrides: Partial<SlotState> & { referenceImagePaths?: string[] } = {}, options: { files?: Partial<Held>; takesLast?: boolean; restored?: boolean } = {}) {
  const inVideo = workspaceTab(workflow) === 'video';
  const caps = { showRefImage: true, takesLastFrame: inVideo && (options.takesLast ?? true), maxReferences: inVideo ? 0 : 3, showI2IStrength: !inVideo } as WorkspaceCapabilities;
  return imagesSummary(imagesView({ state: state(workflow, overrides), caps, files: { ...NO_FILES, ...options.files }, restored: options.restored ?? true }));
}

function image(overrides: Partial<SlotState> & { referenceImagePaths?: string[] } = {}, options: { files?: Partial<Held>; restored?: boolean } = {}) {
  const workflow: Workflow = overrides.referenceImagePath || overrides.referenceImageUpload ? 'img2img' : 'txt2img';
  return summarize(workflow, overrides, options);
}

function video(overrides: Partial<SlotState> = {}, options: { files?: Partial<Held>; takesLast?: boolean; restored?: boolean } = {}) {
  const workflow: Workflow = overrides.firstFramePath || overrides.firstFrameUpload ? 'img2vid' : 'txt2vid';
  return summarize(workflow, overrides, options);
}

describe('imagesSummary in Image', () => {
  it('reads None, muted, when nothing is set', () => {
    expect(image()).toEqual({ text: 'None', tone: 'muted' });
  });

  it('gives the start image by its name, active', () => {
    expect(image({ referenceImagePath: '/in/pond.png' })).toEqual({ text: 'pond.png', tone: 'active' });
  });

  it('adds the reference count, singular for one', () => {
    expect(image({ referenceImagePath: '/in/pond.png', referenceImagePaths: ['/in/r1.png', '/in/r2.png'] })).toEqual({ text: 'pond.png · 2 references', tone: 'active' });
    expect(image({ referenceImagePath: '/in/pond.png', referenceImagePaths: ['/in/r1.png'] }).text).toBe('pond.png · 1 reference');
  });

  it('gives the references alone, active', () => {
    expect(image({ referenceImagePaths: ['/in/r1.png', '/in/r2.png'] })).toEqual({ text: '2 references', tone: 'active' });
  });

  it('names an uploaded start image by its file, or by its marker before the file comes back', () => {
    expect(image({ referenceImageUpload: 'a.png' }, { files: { startImage: file('held.png') } }).text).toBe('held.png');
    expect(image({ referenceImageUpload: 'a.png' }, { restored: false }).text).toBe('a.png');
  });

  it('turns to an error when the start image is gone, and keeps its name', () => {
    expect(image({ referenceImagePath: '/in/pond.png', missingReferencePaths: ['/in/pond.png'] })).toEqual({ text: 'pond.png', tone: 'error' });
  });

  it('turns to an error when a reference is gone, and keeps the count', () => {
    const summary = image({ referenceImagePaths: ['/in/r1.png', '/in/gone.png'], missingReferencePaths: ['/in/gone.png'] });
    expect(summary).toEqual({ text: '2 references', tone: 'error' });
  });
});

describe('imagesSummary in Video', () => {
  it('reads None, muted, with no first frame', () => {
    expect(video()).toEqual({ text: 'None', tone: 'muted' });
  });

  it('gives the first frame by its name, active', () => {
    expect(video({ firstFramePath: '/in/dawn.png' })).toEqual({ text: 'dawn.png', tone: 'active' });
  });

  it('gives both names with an arrow while the model takes a last frame', () => {
    expect(video({ firstFramePath: '/in/dawn.png', endImagePath: '/in/dusk.png' })).toEqual({ text: 'dawn.png → dusk.png', tone: 'active' });
    expect(video({ firstFramePath: '/in/dawn.png', endImagePath: '/in/dusk.png' }, { takesLast: false }).text).toBe('dawn.png');
  });

  it('reads None for a last frame kept without a first frame, which is not used', () => {
    expect(video({ endImagePath: '/in/dusk.png' })).toEqual({ text: 'None', tone: 'muted' });
    expect(video({ endImageUpload: 'dusk.png' }).text).toBe('None');
  });

  it('turns to an error when a frame the run uses is gone', () => {
    expect(video({ firstFramePath: '/in/dawn.png', missingReferencePaths: ['/in/dawn.png'] })).toEqual({ text: 'dawn.png', tone: 'error' });
    expect(video({ firstFramePath: '/in/dawn.png', endImagePath: '/in/dusk.png', missingReferencePaths: ['/in/dusk.png'] })).toEqual({ text: 'dawn.png → dusk.png', tone: 'error' });
  });
});
