import { describe, expect, it } from 'vitest';
import { visibleTabs, WORKSPACE_TABS, workflowFor, workspaceTab } from './workspaceTab';
import type { DraftState, Workflow } from '$lib/types';

const NONE = { referenceImagePath: null, referenceImageUpload: null, firstFramePath: null, firstFrameUpload: null } satisfies Partial<DraftState>;

describe('WORKSPACE_TABS', () => {
  it('lists Image, Video and Extend Video in that order', () => {
    expect(WORKSPACE_TABS.map((tab) => [tab.id, tab.label])).toEqual([['image', 'Image'], ['video', 'Video'], ['extend', 'Extend Video']]);
  });
});

describe('visibleTabs', () => {
  const ids = (workflows: readonly Workflow[] | null): string[] => visibleTabs(workflows).map((tab) => tab.id);

  it('offers Image and Video before the contract loads', () => {
    expect(ids(null)).toEqual(['image', 'video']);
  });

  it('offers Extend Video only when the server lists the extend workflow', () => {
    expect(ids(['txt2img', 'img2img', 'txt2vid', 'img2vid'])).toEqual(['image', 'video']);
    expect(ids(['txt2img', 'img2img', 'txt2vid', 'img2vid', 'extend'])).toEqual(['image', 'video', 'extend']);
  });
});

describe('workspaceTab', () => {
  it('puts both workflows of a tab under it', () => {
    expect(['txt2img', 'img2img'].map((w) => workspaceTab(w as Workflow))).toEqual(['image', 'image']);
    expect(['txt2vid', 'img2vid'].map((w) => workspaceTab(w as Workflow))).toEqual(['video', 'video']);
    expect(workspaceTab('extend')).toBe('extend');
  });
});

describe('workflowFor', () => {
  it('starts an Image run from text until a start image is set, as a path or as an upload', () => {
    expect(workflowFor('image', NONE, true)).toBe('txt2img');
    expect(workflowFor('image', { ...NONE, referenceImagePath: '/in/a.png' }, true)).toBe('img2img');
    expect(workflowFor('image', { ...NONE, referenceImageUpload: 'a.png' }, true)).toBe('img2img');
  });

  it('starts a Video run from text until a first frame is set, as a path or as an upload', () => {
    expect(workflowFor('video', NONE, true)).toBe('txt2vid');
    expect(workflowFor('video', { ...NONE, firstFramePath: '/in/a.png' }, true)).toBe('img2vid');
    expect(workflowFor('video', { ...NONE, firstFrameUpload: 'a.png' }, true)).toBe('img2vid');
  });

  it('reads only the images of its own tab', () => {
    expect(workflowFor('image', { ...NONE, firstFramePath: '/in/a.png' }, true)).toBe('txt2img');
    expect(workflowFor('video', { ...NONE, referenceImagePath: '/in/a.png' }, true)).toBe('txt2vid');
  });

  it('runs from text when the model takes no start image', () => {
    expect(workflowFor('image', { ...NONE, referenceImagePath: '/in/a.png' }, false)).toBe('txt2img');
    expect(workflowFor('video', { ...NONE, firstFramePath: '/in/a.png' }, false)).toBe('txt2vid');
  });

  it('is extend on the Extend Video tab, whatever the images', () => {
    expect(workflowFor('extend', { ...NONE, referenceImagePath: '/in/a.png', firstFramePath: '/in/b.png' }, true)).toBe('extend');
  });
});
