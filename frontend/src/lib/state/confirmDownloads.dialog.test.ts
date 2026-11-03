import { flushSync } from 'svelte';
import { afterEach, describe, expect, it } from 'vitest';

import type { JobDownloadLicence, JobDownloadPlan } from '$lib/types';
import { FLUX_LICENCE, GEMMA_LICENCE, LTX_LICENCE, MATTE_LICENCE } from '../../test-utils/workspace';
import { confirmDownloads } from './confirmDownloads';

const SPACE = { path: '/data', needed_bytes: 44.1e9, free_bytes: 812e9, enough: true };
const part = (label: string, model: string, bytes: number | null = 35.1e9) => ({ label, model, repo: `org/${model}`, files: 3, bytes });
const tagged = (model: string, ...records: (typeof FLUX_LICENCE)[]): JobDownloadLicence[] => records.map((record) => ({ model, ...record }));

function plan(parts: JobDownloadPlan['parts'], licences: JobDownloadLicence[]): JobDownloadPlan {
  return { parts, total_bytes: 40e9, space: SPACE, unchecked: false, licences };
}

const rows = (): HTMLElement[] => Array.from(document.querySelectorAll('[data-testid="confirm-part"]'));
const linesOf = (row: HTMLElement): HTMLElement[] => Array.from(row.querySelectorAll('.licence-line'));
const squash = (el: Element): string => (el.textContent ?? '').replace(/\s+/g, ' ').trim();

describe('the download dialog with licence lines', () => {
  afterEach(() => {
    document.body.innerHTML = '';
  });

  async function open(next: JobDownloadPlan): Promise<{ answer: Promise<boolean> }> {
    const answer = confirmDownloads(next);
    await Promise.resolve();
    flushSync();
    return { answer };
  }

  it('says once that the terms apply and puts the klein9b line, with both links, under its part', async () => {
    await open(plan([part('FLUX.2 Klein 9B', 'klein9b'), part('Prompt enhancer (Smart)', 'org/enhancer')], tagged('klein9b', FLUX_LICENCE)));

    expect(document.body.textContent?.match(/By downloading you agree to the terms shown under each model\./g)).toHaveLength(1);
    const [klein, enhancer] = rows();
    expect(linesOf(klein).map(squash)).toEqual(['FLUX Non-Commercial License v2.1 (Black Forest Labs): Non-commercial · Licence ↗ · NOTICE ↗']);
    expect(linesOf(enhancer)).toEqual([]);
    const links = Array.from(klein.querySelectorAll('a'));
    expect(links.map((link) => [link.getAttribute('href'), link.getAttribute('target'), link.getAttribute('rel'), link.getAttribute('aria-label')])).toEqual([
      [FLUX_LICENCE.url, '_blank', 'noopener', 'FLUX Non-Commercial License v2.1 licence (opens in a new tab)'],
      [FLUX_LICENCE.notice_url, '_blank', 'noopener', 'FLUX Non-Commercial License v2.1 NOTICE (opens in a new tab)'],
    ]);
  });

  it('gives a part with two records two lines', async () => {
    await open(plan([part('LTX distilled model', 'ltx-4')], tagged('ltx-4', LTX_LICENCE, GEMMA_LICENCE)));

    expect(linesOf(rows()[0]).map(squash)).toEqual([
      'LTX-2 Community License Agreement (Lightricks): Free under $10M revenue · Licence ↗',
      "Gemma Terms of Use (Google): Gemma's use policy applies · Licence ↗",
    ]);
  });

  it('calls the matte link a model card', async () => {
    await open(plan([part('Background removal', 'matte')], tagged('matte', MATTE_LICENCE)));

    expect(squash(linesOf(rows()[0])[0])).toContain('· Model card ↗');
    expect(rows()[0].querySelector('a')!.getAttribute('aria-label')).toBe('BiRefNet (MIT) and ViTMatte (Apache-2.0) model card (opens in a new tab)');
  });

  it('has no licence line and no agreement sentence for zit', async () => {
    await open(plan([part('zit', 'zit')], []));

    expect(linesOf(rows()[0])).toEqual([]);
    expect(document.body.textContent).not.toContain('agree to the terms');
  });
});
