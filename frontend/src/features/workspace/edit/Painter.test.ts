import { tick } from 'svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { mountComponent } from '../../../test-utils/mount';
import { draft } from '$lib/state/draft.svelte';
import { imageEdit } from './editStore.svelte';
import { sourceOfPath } from './editSource';
import Painter from './Painter.svelte';

const confirm = vi.hoisted(() => ({ requestConfirm: vi.fn() }));
vi.mock('$lib/components/molecules', async (importOriginal) => ({ ...(await importOriginal<typeof import('$lib/components/molecules')>()), requestConfirm: confirm.requestConfirm }));

/** Open the painter on a picture; the layer is portalled to the body, so everything is looked up on the document. */
function open(cutout: boolean | null = false): void {
  imageEdit.setSource({ ...sourceOfPath('/out/a.png'), width: 800, height: 600, cutout });
  mountComponent(Painter, { onclose: vi.fn() });
}

const modeButton = (mode: 'paint' | 'extend') => document.querySelector(`[data-testid="painter-mode"] [data-value="${mode}"]`) as HTMLButtonElement;
const dab = { erase: false, size: 64, points: [100, 100] };

async function choose(mode: 'paint' | 'extend'): Promise<void> {
  modeButton(mode).click();
  await tick();
  await tick();
}

beforeEach(() => {
  draft.reset();
  imageEdit.setSource(null);
  confirm.requestConfirm.mockReset().mockResolvedValue(true);
});

describe('Extend on a cut-out', () => {
  it('turns Extend off, with the reason as its tooltip, and ignores its key', async () => {
    open(true);
    expect(modeButton('extend').disabled).toBe(true);
    expect(modeButton('extend').title).not.toBe('');
    document.dispatchEvent(new KeyboardEvent('keydown', { key: '2', bubbles: true }));
    await tick();
    expect(imageEdit.mode).toBe('paint');
    expect(document.querySelector('[data-testid="painter-live"]')?.textContent).not.toBe('');
  });

  it('keeps Extend available on an opaque picture', () => {
    open(false);
    expect(modeButton('extend').disabled).toBe(false);
  });
});

describe('switching between Paint and Extend', () => {
  it('switches at once, with no question, when the mode holds nothing', async () => {
    open();
    await choose('extend');
    expect(imageEdit.mode).toBe('extend');
    expect(modeButton('extend').getAttribute('aria-pressed')).toBe('true');
    expect(confirm.requestConfirm).not.toHaveBeenCalled();
  });

  it('asks before removing paint, keeps it on Keep painting, and removes it on Remove paint with Undo bringing it back', async () => {
    open();
    imageEdit.addStroke(dab);
    confirm.requestConfirm.mockResolvedValueOnce(false);

    await choose('extend');
    expect(confirm.requestConfirm).toHaveBeenCalledWith(expect.objectContaining({ question: 'Remove the paint and switch to Extend?', confirmLabel: 'Remove paint', cancelLabel: 'Keep painting' }));
    expect([imageEdit.mode, imageEdit.strokes.length]).toEqual(['paint', 1]);

    await choose('extend');
    expect([imageEdit.mode, imageEdit.strokes.length]).toEqual(['extend', 0]);
    imageEdit.undo();
    expect([imageEdit.mode, imageEdit.strokes.length]).toEqual(['paint', 1]);
  });

  it('asks before removing an extension, and key 1 goes through the same question', async () => {
    open();
    await choose('extend');
    imageEdit.toggleSide('right');
    confirm.requestConfirm.mockResolvedValueOnce(false);

    document.dispatchEvent(new KeyboardEvent('keydown', { key: '1', bubbles: true }));
    await tick();
    await tick();

    expect(confirm.requestConfirm).toHaveBeenCalledWith(expect.objectContaining({ question: 'Remove the extension and switch to Paint?', confirmLabel: 'Remove extension', cancelLabel: 'Keep extending' }));
    expect([imageEdit.mode, imageEdit.extension?.side]).toEqual(['extend', 'right']);

    await choose('paint');
    expect([imageEdit.mode, imageEdit.extension]).toEqual(['paint', null]);
  });

  it('opens in the mode the draft holds, and Cancel puts back the mode it opened in', async () => {
    draft.commitEdit({ mode: 'extend', mask: null, share: 0, extension: { side: 'top', amount: 256 }, amount: 256 });
    open();
    expect(imageEdit.mode).toBe('extend');
    await choose('paint');
    expect([imageEdit.mode, imageEdit.extension]).toEqual(['paint', null]);

    (document.querySelector('[data-testid="painter-cancel"]') as HTMLButtonElement).click();
    await tick();
    await tick();

    expect(confirm.requestConfirm).toHaveBeenLastCalledWith(expect.objectContaining({ question: 'Discard changes?' }));
    expect([imageEdit.mode, imageEdit.extension?.side]).toEqual(['extend', 'top']);
  });
});
