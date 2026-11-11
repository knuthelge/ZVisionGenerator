import type { ChipFacts } from '../features/workspace/settingsChips';

// The settings chips' facts for a test: every fact has a neutral value, and the test names the ones it cares about.

export function makeChipFacts(overrides: Pick<ChipFacts, 'caps' | 'state'> & Partial<ChipFacts>): ChipFacts {
  return {
    changed: () => false, width: 1024, height: 1024, seconds: null, extendSeconds: null,
    loraNames: [], videoKind: null, modelLabel: 'Z-Image Turbo', modelDownloaded: true, sourceName: null, images: null, track: null, memoryWarn: false,
    presetName: null, presetSlug: null, presetChanged: false,
    ...overrides,
  };
}
