export function shouldApplyStartupView(hash: string, search: string): boolean {
  return hash === '' && search === '';
}