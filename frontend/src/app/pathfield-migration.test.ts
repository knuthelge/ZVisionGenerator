import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

function readSource(relativePath: string): string {
  return readFileSync(new URL(relativePath, import.meta.url), 'utf-8');
}

describe('PathField migration — REQ-17 / SC-9', () => {
  // ── ModelsPage ────────────────────────────────────────────────────────────

  it('ModelsPage imports PathField from molecules', () => {
    const source = readSource('../features/models/ModelsPage.svelte');
    expect(source).toContain("import { FormField, PathField } from '$lib/components/molecules'");
  });

  it('ModelsPage uses PathField for "Input Path" (Convert Checkpoint)', () => {
    const source = readSource('../features/models/ModelsPage.svelte');
    expect(source).toContain('<PathField');
    expect(source).toContain('label="Input Path"');
  });

  it('ModelsPage Input Path PathField has pickerKind="existing_file"', () => {
    const source = readSource('../features/models/ModelsPage.svelte');
    // Find the block around "Input Path" and verify pickerKind
    const inputPathIdx = source.indexOf('label="Input Path"');
    expect(inputPathIdx).toBeGreaterThan(-1);
    // pickerKind should appear near the PathField for Input Path
    const surrounding = source.slice(Math.max(0, inputPathIdx - 300), inputPathIdx + 300);
    expect(surrounding).toContain('pickerKind="existing_file"');
  });

  it('ModelsPage uses PathField for "Source Path" (Import Local LoRA)', () => {
    const source = readSource('../features/models/ModelsPage.svelte');
    expect(source).toContain('label="Source Path"');
  });

  it('ModelsPage Source Path PathField has pickerKind="existing_file"', () => {
    const source = readSource('../features/models/ModelsPage.svelte');
    const sourcePathIdx = source.indexOf('label="Source Path"');
    expect(sourcePathIdx).toBeGreaterThan(-1);
    const surrounding = source.slice(Math.max(0, sourcePathIdx - 300), sourcePathIdx + 300);
    expect(surrounding).toContain('pickerKind="existing_file"');
  });

  it('ModelsPage has no page-specific picker function (pickDirectory / pickingDir)', () => {
    const source = readSource('../features/models/ModelsPage.svelte');
    expect(source).not.toContain('pickDirectory');
    expect(source).not.toContain('pickingDir');
    expect(source).not.toContain('openFilePicker');
  });

  // ── ConfigPage ────────────────────────────────────────────────────────────

  it('ConfigPage imports PathField from molecules', () => {
    const source = readSource('../features/config/ConfigPage.svelte');
    expect(source).toContain("import { FormField, PathField } from '$lib/components/molecules'");
  });

  it('ConfigPage uses PathField for "Output Directory"', () => {
    const source = readSource('../features/config/ConfigPage.svelte');
    expect(source).toContain('<PathField');
    expect(source).toContain('label="Output Directory"');
  });

  it('ConfigPage Output Directory PathField has pickerKind="directory"', () => {
    const source = readSource('../features/config/ConfigPage.svelte');
    const outputDirIdx = source.indexOf('label="Output Directory"');
    expect(outputDirIdx).toBeGreaterThan(-1);
    const surrounding = source.slice(Math.max(0, outputDirIdx - 300), outputDirIdx + 300);
    expect(surrounding).toContain('pickerKind="directory"');
  });

  it('ConfigPage has no page-specific picker function (pickDirectory / pickingDir)', () => {
    const source = readSource('../features/config/ConfigPage.svelte');
    expect(source).not.toContain('pickDirectory');
    expect(source).not.toContain('pickingDir');
    expect(source).not.toContain('openFilePicker');
  });

  // ── PathField component ───────────────────────────────────────────────────

  it('PathField component renders Browse and Clear buttons with Enter key validation', () => {
    const source = readSource('../lib/components/molecules/PathField.svelte');
    expect(source).toContain('browseLabel');
    expect(source).toContain('clearLabel');
    // Default labels
    expect(source).toContain("browseLabel = 'Browse'");
    expect(source).toContain("clearLabel = 'Clear'");
    // Enter key validation
    expect(source).toContain('onkeydown') || expect(source).toContain('Enter');
  });

  it('PathField component exposes error state via FormField', () => {
    const source = readSource('../lib/components/molecules/PathField.svelte');
    expect(source).toContain('{error}');
    expect(source).toContain("let error = $state<string | null>(null)");
  });

  it('PathField component is exported from molecules index', () => {
    const source = readSource('../lib/components/molecules/index.ts');
    expect(source).toContain('PathField');
  });

  // ── No regressions in other pages ─────────────────────────────────────────

  it('no page outside PathField.svelte implements a custom picker function', () => {
    // These patterns indicate a page is rolling its own browse/pick logic
    const files = [
      '../features/workspace/WorkspacePage.svelte',
      '../features/workspace/ControlsSidebar.svelte',
      '../features/gallery/GalleryPage.svelte',
    ];
    for (const file of files) {
      const source = readSource(file);
      expect(source, `${file} must not contain pickDirectory`).not.toContain('pickDirectory');
      expect(source, `${file} must not contain pickingDir`).not.toContain('pickingDir');
    }
  });
});
