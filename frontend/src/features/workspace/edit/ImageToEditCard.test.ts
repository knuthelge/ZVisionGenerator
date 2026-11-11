import { flushSync } from 'svelte';
import { beforeEach, describe, expect, it } from 'vitest';

import { mountComponent } from '../../../test-utils/mount';
import { draft } from '$lib/state/draft.svelte';
import { heldUploads } from '$lib/state/heldUploads.svelte';
import { memoryUploadStore } from '$lib/state/uploadStore';
import { imageEdit } from './editStore.svelte';
import { sourceOfPath } from './editSource';
import ImageToEditCard from './ImageToEditCard.svelte';

const open = () => document.querySelector('[data-testid="edit-open-painter"]') as HTMLButtonElement;

/** The phone card for the draft as it is, on an opaque picture. */
function card(model = 'klein'): string {
  draft.patch({ model });
  mountComponent(ImageToEditCard, {});
  flushSync();
  return open().getAttribute('aria-label') ?? '';
}

beforeEach(() => {
  localStorage.clear();
  draft.reset();
  heldUploads.useStore(memoryUploadStore());
  imageEdit.setSource({ ...sourceOfPath('/out/a.png'), width: 832, height: 1216, cutout: false });
  draft.setEditSource({ path: '/out/a.png' });
});

describe('the phone card', () => {
  it('is one button that opens the painter and says what will be sent', () => {
    draft.commitEdit({ mode: 'paint', mask: new File(['x'], 'mask.png'), share: 0.06, extension: null, amount: 256 });
    expect(card()).toBe('Open the painter. Paint: 6% of the picture');
  });
});
