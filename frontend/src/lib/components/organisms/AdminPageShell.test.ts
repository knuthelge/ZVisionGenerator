// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { createRawSnippet } from 'svelte';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { reactiveProps } from '../../../test-utils/reactiveProps.svelte';
import AdminPageShell from './AdminPageShell.svelte';

const actions = createRawSnippet(() => ({ render: () => '<button type="button">Save</button>' }));

describe('AdminPageShell', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(async () => {
    if (app) {
      await unmount(app);
      app = null;
    }
    target.remove();
  });

  it('keeps the page actions in the bar while loading, disabled, and wakes them once loaded', () => {
    const props = reactiveProps({ title: 'Config', loading: true, error: null as string | null, actions });
    app = flushSync(() => mount(AdminPageShell, { target, props }));

    // The actions sit in a fieldset, which disables every control inside it while the page loads.
    const actionsBox = () => target.querySelector('[data-testid="page-actions"]') as HTMLFieldSetElement;
    expect(actionsBox().querySelector('button')?.textContent).toBe('Save');
    expect(actionsBox().disabled).toBe(true);
    expect(target.querySelector('[role="status"]')).not.toBeNull();

    props.loading = false;
    flushSync();
    expect(actionsBox().disabled).toBe(false);
    expect(target.querySelector('[role="status"]')).toBeNull();

    props.error = 'Could not read the config file.';
    flushSync();
    expect(actionsBox().disabled).toBe(true);
  });
});
