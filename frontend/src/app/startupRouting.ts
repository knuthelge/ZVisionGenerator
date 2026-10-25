/** Whether the configured startup view applies: only when the address names no page (the hash is the one form the app reads). */
export function shouldApplyStartupView(hash: string): boolean {
  return hash === '';
}
