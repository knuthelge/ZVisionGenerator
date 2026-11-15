import { flushSync } from 'svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const layoutMock = vi.hoisted(() => ({ compact: false, mode: 'wide', coarse: false }));
vi.mock('$lib/state/layoutMode.svelte', () => ({ layout: layoutMock }));

const api = vi.hoisted(() => ({ saveHfToken: vi.fn() }));
vi.mock('$lib/api/config', async (importOriginal) => ({ ...(await importOriginal<typeof import('$lib/api/config')>()), saveHfToken: api.saveHfToken }));

import { ApiError } from '$lib/api/client';
import { toasts } from '$lib/state/toasts.svelte';
import { mountComponent, settle } from '../../../test-utils/mount';
import { reactiveProps } from '../../../test-utils/reactiveProps.svelte';
import HfTokenDialog, { hfTokenItem } from './HfTokenDialog.svelte';

const tokenInput = () => document.querySelector<HTMLInputElement>('#hft-token')!;
const saveButton = () => document.querySelector<HTMLButtonElement>('[data-action="save-hf-token"]')!;

function type(value: string): void {
  tokenInput().value = value;
  flushSync(() => tokenInput().dispatchEvent(new Event('input', { bubbles: true })));
}

describe('hfTokenItem', () => {
  it('offers a button only for a token the app can change', () => {
    const onAdd = vi.fn();

    expect(hfTokenItem(null, onAdd).action?.label).toBe('Add token');
    expect(hfTokenItem('saved', onAdd).action?.label).toBe('Replace token');
    expect(hfTokenItem('HF_TOKEN', onAdd).action).toBeUndefined();

    hfTokenItem(null, onAdd).action?.run();
    expect(onAdd).toHaveBeenCalledOnce();
  });
});

describe.each([['wide', false], ['compact', true]])('HfTokenDialog on %s', (_name, compact) => {
  beforeEach(() => {
    Object.assign(layoutMock, { compact, mode: compact ? 'compact' : 'wide', coarse: compact });
    api.saveHfToken.mockReset();
    toasts.splice(0, toasts.length);
  });

  async function open() {
    const props = reactiveProps({ open: true, onclose: vi.fn(), onsaved: vi.fn() });
    mountComponent(HfTokenDialog, props);
    await settle();
    return props;
  }

  it('keeps Save off until a token is typed, in a field password managers leave alone', async () => {
    await open();

    expect(tokenInput().type).toBe('text');
    expect(saveButton().disabled).toBe(true);
    type('   ');
    expect(saveButton().disabled).toBe(true);
    type('hf_abc');
    expect(saveButton().disabled).toBe(false);
  });

  it('saves the token, reports the new source and closes', async () => {
    api.saveHfToken.mockResolvedValue({ huggingface_token_source: 'saved' });
    const props = await open();

    type('hf_abc');
    saveButton().click();
    await settle();

    expect(api.saveHfToken).toHaveBeenCalledWith('hf_abc');
    expect(props.onsaved).toHaveBeenCalledWith('saved');
    expect(props.onclose).toHaveBeenCalledOnce();
  });

  it('stays open with the reason when the token is refused', async () => {
    api.saveHfToken.mockRejectedValue(new ApiError("Hugging Face doesn't accept this token.", 400, "Hugging Face doesn't accept this token."));
    const props = await open();

    type('hf_bad');
    saveButton().click();
    await settle();

    expect(document.querySelector('[role="alert"]')?.textContent).toContain("doesn't accept");
    expect(props.onsaved).not.toHaveBeenCalled();
    expect(props.onclose).not.toHaveBeenCalled();
  });

  it('starts empty each time it opens', async () => {
    const props = await open();
    type('hf_abc');

    props.open = false;
    await settle();
    props.open = true;
    await settle();

    expect(tokenInput().value).toBe('');
  });
});
