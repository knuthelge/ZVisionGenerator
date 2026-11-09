// Module-level state of the Voices window, using Svelte 5 runes: whether it is open and on which tab.
// It lives here, not in a component, because the Voice section (a column), the Voice sheet (a phone) and a notice can all open it.

/** The tabs of the Voices window. */
export type VoicesTab = 'yours' | 'new' | 'recording';

/**
 * The tab the New voice button opens. Without a model that can design, it is From recording. While a one-off recording is the voice it
 * is From recording too, prefilled, so there is a visible way to keep it, unless designed voices wait to be picked: their marker's promise wins.
 */
export function newVoiceTab(facts: { canDesign: boolean; oneOff: boolean; pending: number }): VoicesTab {
  if (!facts.canDesign) return 'recording';
  return facts.oneOff && facts.pending === 0 ? 'recording' : 'new';
}

class VoiceWindow {
  open = $state(false);
  tab = $state<VoicesTab>('yours');
  /** The Speech tab was asked to open the window on New voice (the ready notice's Pick one, from any page); the Workspace takes it once it has loaded. */
  requested = $state(false);

  /** Open the window on *tab*. */
  show(tab: VoicesTab): void {
    this.tab = tab;
    this.open = true;
  }

  /** Whether the window is open on the New voice tab, where the candidates show as they arrive. */
  get onNewVoice(): boolean {
    return this.open && this.tab === 'new';
  }

  /** Ask the Workspace to open the window on New voice. */
  request(): void {
    this.requested = true;
  }

  /** Take the request, if there is one; it is handed out once. */
  takeRequest(): boolean {
    const was = this.requested;
    this.requested = false;
    return was;
  }
}

export const voiceWindow = new VoiceWindow();
