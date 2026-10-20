import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

describe('index.html', () => {
  it('loads the Google Fonts stylesheet without blocking render', () => {
    const html = readFileSync('index.html', 'utf8');
    const links = html.match(/<link rel="stylesheet" href="https:\/\/fonts\.googleapis\.com[^>]*>/g) ?? [];
    const blocking = links.filter((link) => !html.includes(`<noscript>${link}</noscript>`));

    expect(blocking).toHaveLength(1);
    expect(blocking[0]).toContain('media="print"');
    expect(blocking[0]).toContain(`onload="this.media='all'"`);
  });

  it('keeps a home-screen app under an opaque status bar, so iOS gives it the whole screen below it', () => {
    const html = readFileSync('index.html', 'utf8');
    expect(html).toMatch(/<meta name="apple-mobile-web-app-status-bar-style" content="black" \/>/);
  });
});
