import { describe, expect, it } from 'vitest';
import { IMAGE_KINDS, isAudioWorkflow, navTab, textTab, videoKindOf, videoKinds, visibleTabs, WORKSPACE_TABS, workflowFor, workspaceTab } from './workspaceTab';
import type { DraftState, Workflow, WorkspaceWorkflow } from '$lib/types';

const NONE = { referenceImagePath: null, referenceImageUpload: null, firstFramePath: null, firstFrameUpload: null } satisfies Partial<DraftState>;

describe('WORKSPACE_TABS', () => {
  it('lists Image, Video and Audio in that order', () => {
    expect(WORKSPACE_TABS.map((tab) => [tab.id, tab.label])).toEqual([['image', 'Image'], ['video', 'Video'], ['audio', 'Audio']]);
  });
});

describe('visibleTabs', () => {
  const ids = (workflows: readonly WorkspaceWorkflow[] | null): string[] => visibleTabs(workflows).map((tab) => tab.id);

  it('offers Image and Video before the contract loads', () => {
    expect(ids(null)).toEqual(['image', 'video']);
  });

  it('never offers Extend as a tab of its own', () => {
    expect(ids(['txt2img', 'img2img', 'txt2vid', 'img2vid', 'extend'])).toEqual(['image', 'video']);
  });

  it('hides Audio without a music, speech or sound workflow, and shows it, last, with any of them', () => {
    const media: Workflow[] = ['txt2img', 'img2img', 'txt2vid', 'img2vid', 'extend'];
    expect(ids(media)).not.toContain('audio');
    expect(ids(null)).not.toContain('audio');
    expect(ids([...media, 'sound'])).toEqual(['image', 'video', 'audio']);
    expect(ids(['txt2img', 'img2img', 'txt2vid', 'img2vid', 'music', 'speech'])).toEqual(['image', 'video', 'audio']);
  });
});

describe('videoKinds', () => {
  it('offers Extend beside New only when the server lists the extend workflow', () => {
    expect(videoKinds(null)).toEqual(['new']);
    expect(videoKinds(['txt2img', 'img2img', 'txt2vid', 'img2vid'])).toEqual(['new']);
    expect(videoKinds(['txt2img', 'img2img', 'txt2vid', 'img2vid', 'extend'])).toEqual(['new', 'extend']);
  });
});

describe('navTab and videoKindOf', () => {
  it('puts both kinds of Video under the Video tab, and tells them apart', () => {
    expect((['txt2vid', 'img2vid', 'extend'] as const).map(navTab)).toEqual(['video', 'video', 'video']);
    expect((['txt2vid', 'img2vid', 'extend', 'txt2img', 'music'] as const).map(videoKindOf)).toEqual(['new', 'new', 'extend', null, null]);
    expect((['txt2img', 'img2img', 'sound'] as const).map(navTab)).toEqual(['image', 'image', 'audio']);
  });
});

describe('workspaceTab', () => {
  it('puts both workflows of a tab under it', () => {
    expect(['txt2img', 'img2img'].map((w) => workspaceTab(w as Workflow))).toEqual(['image', 'image']);
    expect(['txt2vid', 'img2vid'].map((w) => workspaceTab(w as Workflow))).toEqual(['video', 'video']);
    expect(workspaceTab('extend')).toBe('extend');
  });

  it('puts every audio workflow under the Audio tab and never under Image', () => {
    expect((['music', 'speech', 'sound'] as const).map((w) => workspaceTab(w))).toEqual(['audio', 'audio', 'audio']);
    expect(isAudioWorkflow('music')).toBe(true);
    expect(isAudioWorkflow('txt2img')).toBe(false);
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

  it('is extend in the Extend slot, whatever the images', () => {
    expect(workflowFor('extend', { ...NONE, referenceImagePath: '/in/a.png', firstFramePath: '/in/b.png' }, true)).toBe('extend');
  });
});

describe('IMAGE_KINDS', () => {
  it('offers New and Edit', () => {
    expect(IMAGE_KINDS).toEqual(['new', 'edit']);
  });

  it('never makes Edit a tab', () => {
    expect(visibleTabs(['txt2img', 'img2img', 'edit']).map((tab) => tab.id)).toEqual(['image', 'video']);
  });
});

describe('Edit, a kind of the Image tab', () => {
  it('belongs to the Image tab of the top bar, and is no kind of Video', () => {
    expect(navTab('edit')).toBe('image');
    expect(videoKindOf('edit')).toBeNull();
  });

  it('has a slot of its own, apart from New\'s', () => {
    expect(workspaceTab('edit')).toBe('edit');
    expect(workspaceTab('txt2img')).toBe('image');
  });

  it('is `edit` in the Edit slot, whatever the images', () => {
    expect(workflowFor('edit', { ...NONE, referenceImagePath: '/in/a.png' }, true)).toBe('edit');
  });

  it('writes a text of its own, where Extend writes Video\'s', () => {
    expect((['txt2img', 'img2img', 'edit', 'txt2vid', 'img2vid', 'extend', 'music'] as const).map(textTab)).toEqual(['image', 'image', 'edit', 'video', 'video', 'video', 'audio']);
  });
});
