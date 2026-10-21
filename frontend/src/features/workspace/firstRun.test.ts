import { beforeEach, describe, expect, it } from 'vitest';

import type { ModelOption, WorkspaceContext } from '$lib/types';
import { firstRun, shouldOfferFirstRun, type FirstRunFacts } from './firstRun.svelte';

const model = (id: string, downloaded?: boolean | null): ModelOption => ({ id, label: id, type: 'image', ...(downloaded === undefined ? {} : { downloaded }) });
const context = (overrides: Partial<WorkspaceContext> = {}): WorkspaceContext => ({
  image_models: [model('klein9b', false), model('zit', false)],
  first_run: { skipped: false, recommended_model: 'klein9b' },
  ...overrides,
}) as WorkspaceContext;
const facts = (overrides: Partial<FirstRunFacts> = {}): FirstRunFacts => ({ downloadsSeeded: true, activeImageDownload: false, dismissed: false, ...overrides });

describe('shouldOfferFirstRun', () => {
  beforeEach(() => firstRun.reset());

  it('offers the dialog when every image model is missing and nothing is pending', () => {
    expect(shouldOfferFirstRun(context(), facts())).toBe(true);
  });

  it.each([
    ['no context yet', null, facts()],
    ['a context from a server without first-run support', context({ first_run: undefined }), facts()],
    ['the dialog was skipped', context({ first_run: { skipped: true, recommended_model: null } }), facts()],
    ['no image models', context({ image_models: [] }), facts()],
    ['one model downloaded', context({ image_models: [model('klein9b', false), model('zit', true)] }), facts()],
    ['a model of unknown status', context({ image_models: [model('klein9b', false), model('zit', null)] }), facts()],
    ['a model without a status field', context({ image_models: [model('klein9b', false), model('zit')] }), facts()],
    ['the download lane is not read yet', context(), facts({ downloadsSeeded: false })],
    ['an image download is active', context(), facts({ activeImageDownload: true })],
    ['the user put the dialog away', context(), facts({ dismissed: true })],
  ] as [string, WorkspaceContext | null, FirstRunFacts][])('does not offer it with %s', (_why, ctx, given) => {
    expect(shouldOfferFirstRun(ctx, given)).toBe(false);
  });
});

describe('firstRun', () => {
  beforeEach(() => firstRun.reset());

  it('stays dismissed until reset', () => {
    expect(firstRun.dismissed).toBe(false);

    firstRun.dismiss();
    expect(firstRun.dismissed).toBe(true);

    firstRun.reset();
    expect(firstRun.dismissed).toBe(false);
  });
});
