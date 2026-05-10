<script lang="ts">
  import Toast from './Toast.svelte';
  import { toasts, dismissToast } from '$lib/state/toasts.svelte';
</script>

<div
  class="fixed top-4 right-4 z-50 flex flex-col gap-2 pointer-events-none"
  aria-label="Notifications"
>
  {#each toasts as toast (toast.id)}
    <Toast
      id={toast.id}
      type={toast.type}
      message={toast.message}
      timeout={toast.timeout}
      ondismiss={dismissToast}
    />
  {/each}
</div>
