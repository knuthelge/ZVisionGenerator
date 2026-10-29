// Triggers of LoRAs whose file is still downloading: the download's answer names the trigger, and the installed
// LoRA row carries it once the file is listed. Until then the chip's trigger is looked up here.

let _pending = $state<Record<string, string>>({});

export const pendingTriggers = {
  /** The trigger a downloading LoRA was started with, or null. */
  get(name: string): string | null {
    return _pending[name] ?? null;
  },
  /** Every name with a pending trigger, by name. */
  get all(): Readonly<Record<string, string>> {
    return _pending;
  },
  set(name: string, trigger: string | null): void {
    if (trigger) _pending = { ..._pending, [name]: trigger };
    else this.forget(name);
  },
  forget(name: string): void {
    if (!(name in _pending)) return;
    const { [name]: _gone, ...rest } = _pending;
    _pending = rest;
  },
  /** Forget everything, for tests. */
  reset(): void {
    _pending = {};
  },
};
