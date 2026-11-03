// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { FLUX_LICENCE, GEMMA_LICENCE, LTX_LICENCE } from '../../../test-utils/workspace';
import LicenceMark from './LicenceMark.svelte';

describe('LicenceMark', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(() => {
    if (app) unmount(app);
    app = null;
    target.remove();
    document.body.innerHTML = '';
  });

  function render(props: Record<string, unknown> = {}): HTMLButtonElement {
    app = mount(LicenceMark, { target, props: { licences: [FLUX_LICENCE], ...props } });
    flushSync();
    return target.querySelector('button') as HTMLButtonElement;
  }

  const bubble = (): HTMLElement => target.querySelector('[role="tooltip"]') as HTMLElement;
  const popover = (): HTMLElement | null => document.querySelector('[data-testid="licence-popover"]');

  /** A key on a focused button, as a browser delivers it: Enter and Space end in the click that opens the popover. */
  it('is a button named by its record, with a popup and nothing expanded', () => {
    const button = render();

    expect(button.getAttribute('aria-label')).toBe('Licence: FLUX Non-Commercial License v2.1');
    expect(button.getAttribute('aria-haspopup')).toBe('dialog');
    expect(button.getAttribute('aria-expanded')).toBe('false');
    expect(popover()).toBeNull();
  });

  it('names two records together', () => {
    const button = render({ licences: [LTX_LICENCE, GEMMA_LICENCE] });

    expect(button.getAttribute('aria-label')).toBe('Licences: LTX-2 Community License Agreement and Gemma Terms of Use');
  });

  it('shows nothing for a model without a record', () => {
    render({ licences: [] });

    expect(target.querySelector('button')).toBeNull();
  });

  it('with a label is a button whose name starts with the visible word', () => {
    const button = render({ label: 'Non-commercial' });

    expect(button.textContent).toContain('Non-commercial');
    expect(button.getAttribute('aria-label')).toBe('Non-commercial. Licence: FLUX Non-Commercial License v2.1');
  });

  it('shows the terms in brief on hover and on keyboard focus, one line per record', () => {
    const button = render({ licences: [LTX_LICENCE, GEMMA_LICENCE] });

    expect(bubble().textContent).toBe("LTX-2 Community License Agreement: Free under $10M revenue\nGemma Terms of Use: Gemma's use policy applies");
    target.firstElementChild!.dispatchEvent(new Event('pointerenter'));
    flushSync();
    expect(bubble().dataset.open).toBe('true');
    target.firstElementChild!.dispatchEvent(new Event('pointerleave'));
    flushSync();
    expect(bubble().dataset.open).toBe('false');

    button.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
    flushSync();
    expect(bubble().dataset.open).toBe('true');
  });

  it('opens a dialog popover on click with the record, the supplier, the summary and the links', () => {
    const button = render();

    button.click();
    flushSync();

    expect(button.getAttribute('aria-expanded')).toBe('true');
    const pop = popover()!;
    expect(pop.getAttribute('role')).toBe('dialog');
    expect(pop.getAttribute('aria-label')).toBe('Licence: FLUX Non-Commercial License v2.1');
    expect(pop.textContent).toContain('FLUX Non-Commercial License v2.1');
    expect(pop.textContent).toContain('Black Forest Labs');
    expect(pop.textContent).toContain('Non-commercial');
    const links = Array.from(pop.querySelectorAll('a'));
    expect(links.map((link) => [link.textContent, link.getAttribute('href'), link.target, link.rel])).toEqual([
      ['Model page ↗', FLUX_LICENCE.url, '_blank', 'noopener'],
    ]);
    expect(links[0].getAttribute('aria-label')).toBe('FLUX Non-Commercial License v2.1 model page (opens in a new tab)');
  });

  it('shows a record\'s credit line in the popover, and nothing for a record without one', () => {
    const button = render({ licences: [{ ...FLUX_LICENCE, attribution: 'Powered by Example' }, LTX_LICENCE] });

    button.click();
    flushSync();

    expect(Array.from(popover()!.querySelectorAll('[data-testid="licence-attribution"]')).map((line) => line.textContent)).toEqual(['Powered by Example']);
  });

  it('keeps the tooltip away while the popover is open', () => {
    const button = render();

    button.click();
    flushSync();
    button.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
    target.firstElementChild!.dispatchEvent(new Event('pointerenter'));
    flushSync();

    expect(bubble().dataset.open).toBe('false');
  });

  it('shows one block per record in the popover', () => {
    const button = render({ licences: [LTX_LICENCE, GEMMA_LICENCE] });

    button.click();
    flushSync();

    expect(Array.from(popover()!.querySelectorAll('section')).map((section) => section.querySelector('p')!.textContent)).toEqual([LTX_LICENCE.name, GEMMA_LICENCE.name]);
  });

  it('closes on Escape and puts focus back on the mark', () => {
    const button = render();
    button.click();
    flushSync();
    expect(document.activeElement).toBe(popover());

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
    flushSync();

    expect(popover()).toBeNull();
    expect(document.activeElement).toBe(button);
    expect(button.getAttribute('aria-expanded')).toBe('false');
  });

  it('closes on a tap outside', () => {
    const button = render();
    button.click();
    flushSync();

    document.body.dispatchEvent(new Event('pointerdown', { bubbles: true }));
    flushSync();

    expect(popover()).toBeNull();
  });
});
