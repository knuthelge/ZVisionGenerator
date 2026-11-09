import { describe, expect, it } from 'vitest';

import { newVoiceTab, voiceWindow } from './voiceWindow.svelte';

describe('newVoiceTab', () => {
  it('is New voice for a model that can design, with no one-off recording in use', () => {
    expect(newVoiceTab({ canDesign: true, oneOff: false, pending: 0 })).toBe('new');
  });

  it('is From recording while a one-off recording is in use, so it can be kept', () => {
    expect(newVoiceTab({ canDesign: true, oneOff: true, pending: 0 })).toBe('recording');
  });

  it('is New voice while designed voices wait to be picked, one-off recording or not', () => {
    expect(newVoiceTab({ canDesign: true, oneOff: true, pending: 2 })).toBe('new');
    expect(newVoiceTab({ canDesign: true, oneOff: false, pending: 2 })).toBe('new');
  });

  it('is From recording for a model that cannot design', () => {
    expect(newVoiceTab({ canDesign: false, oneOff: false, pending: 0 })).toBe('recording');
    expect(newVoiceTab({ canDesign: false, oneOff: false, pending: 3 })).toBe('recording');
  });
});

describe('voiceWindow', () => {
  it('is on New voice only while open on that tab', () => {
    voiceWindow.open = false;
    voiceWindow.tab = 'new';
    expect(voiceWindow.onNewVoice).toBe(false);

    voiceWindow.show('new');
    expect(voiceWindow.onNewVoice).toBe(true);

    voiceWindow.show('yours');
    expect(voiceWindow.onNewVoice).toBe(false);
    voiceWindow.open = false;
  });

  it('hands a request to open on New voice out once', () => {
    expect(voiceWindow.takeRequest()).toBe(false);

    voiceWindow.request();

    expect(voiceWindow.takeRequest()).toBe(true);
    expect(voiceWindow.takeRequest()).toBe(false);
  });
});
