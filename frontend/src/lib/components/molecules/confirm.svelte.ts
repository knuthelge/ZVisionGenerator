import { mount, tick, unmount } from 'svelte';
import ConfirmDialog from './ConfirmDialog.svelte';

/** What a confirmation dialog asks; see `ConfirmDialog` for each field. */
export interface ConfirmRequest {
  question: string;
  info?: string;
  confirmLabel: string;
  cancelLabel?: string;
  danger?: boolean;
}

/** Ask the user to approve an action in a `ConfirmDialog`; resolves true when they confirm. */
export function requestConfirm(request: ConfirmRequest): Promise<boolean> {
  return new Promise((resolve) => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    let settled = false;
    const props = $state({ ...request, open: true, onconfirm: () => void settle(true), oncancel: () => void settle(false) });
    const app = mount(ConfirmDialog, { target, props });

    async function settle(confirmed: boolean): Promise<void> {
      if (settled) return;
      settled = true;
      // Closing first lets Modal return focus to the opener before the dialog is removed.
      props.open = false;
      resolve(confirmed);
      await tick();
      await unmount(app);
      target.remove();
    }
  });
}
