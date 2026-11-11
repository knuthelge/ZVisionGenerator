import { describe, expect, it } from 'vitest';

import { makeEditContext } from '../../test-utils/workspace';
import { declinesModel, modelsOf } from './workflowModels';

describe('the models a workflow offers', () => {
  const context = makeEditContext();

  it('offers every image model to New and only the models that can edit to Edit, and the video models to a video', () => {
    expect(modelsOf(context, 'txt2img').map((model) => model.id)).toEqual(['flux-lite', 'zit', 'klein4b', 'klein']);
    expect(modelsOf(context, 'edit').map((model) => model.id)).toEqual(['klein4b', 'klein']);
    expect(modelsOf(context, 'txt2vid').map((model) => model.id)).toEqual(['ltx-8']);
  });

  it('leaves out of Edit a model the context has no metadata for, and keeps the order the context lists', () => {
    const ghost = { ...context, image_models: [...context.image_models, { ...context.image_models[0], id: 'ghost' }] };
    expect(modelsOf(ghost, 'edit').map((model) => model.id)).toEqual(['klein4b', 'klein']);
  });

  it('declines a model the context lists for the mode that the workflow does not offer', () => {
    expect(declinesModel(context, 'edit', 'flux-lite')).toBe(true);
    expect(declinesModel(context, 'edit', 'klein')).toBe(false);
    expect(declinesModel(context, 'txt2img', 'flux-lite')).toBe(false);
    expect(declinesModel(context, 'txt2vid', 'ltx-8')).toBe(false);
  });

  it('leaves a model the context does not list for the mode to the workflow\'s own handling', () => {
    expect(declinesModel(context, 'edit', 'ghost')).toBe(false);
    expect(declinesModel(context, 'edit', 'ltx-8')).toBe(false);
    expect(declinesModel(context, 'txt2vid', 'klein')).toBe(false);
  });
});
