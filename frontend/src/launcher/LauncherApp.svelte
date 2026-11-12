<script lang="ts">
  import { onMount, tick } from 'svelte';
  import { fade, fly } from 'svelte/transition';
  import Button from '$lib/components/atoms/Button.svelte';
  import Icon from '$lib/components/atoms/Icon.svelte';
  import { prefersReducedMotion } from '$lib/components/atoms/Mascot.svelte';
  import Wordmark from '$lib/components/atoms/Wordmark.svelte';
  import type { ActionMenuEntry } from '$lib/components/molecules/ActionMenu.svelte';
  import ConfirmDialog from '$lib/components/molecules/ConfirmDialog.svelte';
  import Sheet from '$lib/components/molecules/Sheet.svelte';
  import { copyText, selectElementText } from '$lib/clipboard';
  import { bridge as realBridge, type Bridge } from './bridge';
  import LauncherHero from './LauncherHero.svelte';
  import LauncherLink from './LauncherLink.svelte';
  import LauncherNotice from './LauncherNotice.svelte';
  import QuietBar from './QuietBar.svelte';
  import { UNINSTALL_DIALOG, launcherView, type PrimaryView } from './launcherView';
  import type { LinkId, Primary, Snapshot } from './types';

  interface Props {
    /** The Rust side of the window; the real one unless a test or preview brings its own. */
    bridge?: Bridge;
  }

  let { bridge = realBridge }: Props = $props();

  /** The window waits this long before drawing "Checking this computer": a quick check never flashes it. */
  const CHECKING_DELAY_MS = 300;
  /** How long Blob cheers when the app comes up. */
  const CHEER_MS = 2400;
  const COPIED_MS = 1500;

  let snapshot = $state<Snapshot | null>(null);
  let checkingReady = $state(false);
  let cheering = $state(false);
  let logOpen = $state(false);
  let confirmOpen = $state(false);
  let copied = $state(false);
  let logBox = $state<HTMLPreElement>();
  let shell = $state<HTMLDivElement>();

  const view = $derived(snapshot ? launcherView(snapshot) : null);
  const phase = $derived(snapshot?.phase);
  const visible = $derived(snapshot !== null && (phase !== 'checking' || checkingReady));
  const motion = (ms: number): number => (prefersReducedMotion() ? 0 : ms);

  // ——— Snapshots in ———
  onMount(() => {
    let received = false;
    let stop: (() => void) | undefined;
    let gone = false;
    // Listen first: a snapshot that arrives while `get_snapshot` is in flight is newer than its answer.
    void bridge
      .onSnapshot((next) => {
        received = true;
        snapshot = next;
      })
      .then((unlisten) => {
        if (gone) unlisten();
        else stop = unlisten;
      });
    void bridge
      .getSnapshot()
      .then((first) => {
        if (!received) snapshot = first;
      })
      .catch((error: unknown) => console.error('get_snapshot failed', error));
    return () => {
      gone = true;
      stop?.();
    };
  });

  $effect(() => {
    if (phase !== 'checking') {
      checkingReady = false;
      return;
    }
    const timer = setTimeout(() => (checkingReady = true), CHECKING_DELAY_MS);
    return () => clearTimeout(timer);
  });

  // Blob cheers once when the app comes up (a phase change into running), not when the window opens on it.
  let previousPhase: string | undefined;
  $effect(() => {
    const current = phase;
    const arrived = current === 'running' && previousPhase !== undefined && previousPhase !== 'running';
    previousPhase = current;
    if (!arrived) {
      cheering = false;
      return;
    }
    cheering = true;
    const timer = setTimeout(() => (cheering = false), CHEER_MS);
    return () => clearTimeout(timer);
  });

  // ——— Focus: the primary action, or else Quit, whenever nothing holds focus ———
  $effect(() => {
    void view?.primary?.id;
    void visible;
    void tick().then(() => {
      if (document.activeElement && document.activeElement !== document.body) return;
      const target = shell?.querySelector<HTMLElement>('[data-primary]:not(:disabled)') ?? shell?.querySelector<HTMLElement>('[data-quit]');
      target?.focus();
    });
  });

  // ——— Commands out ———
  /** Run a command; "busy" means the state moved on and the button was stale, which needs no report. */
  async function run<T>(command: () => Promise<T>): Promise<T | undefined> {
    try {
      return await command();
    } catch (error) {
      if (error !== 'busy') console.error('launcher command failed', error);
      return undefined;
    }
  }

  const primaryActions: Record<Primary, () => Promise<unknown>> = {
    check_again: () => bridge.checkAgain(),
    see_requirements: () => bridge.openLink('requirements'),
    install: () => bridge.install(),
    retry: () => bridge.retry(),
    starting: async () => {},
    open: () => bridge.openApp(),
    repair: () => bridge.repairApp(),
    quit: () => bridge.quitLauncher()
  };

  const openLink = (id: LinkId): void => void run(() => bridge.openLink(id));

  function requestUninstall(): void {
    // Windows starts its own uninstaller, which asks for itself.
    if (snapshot?.packaging === 'windows_nsis') void run(() => bridge.uninstallApp());
    else confirmOpen = true;
  }

  function confirmUninstall(): void {
    confirmOpen = false;
    void run(() => bridge.uninstallApp());
  }

  const menuEntries = $derived<ActionMenuEntry[] | null>(
    view && view.menu.length > 0
      ? view.menu.map((item) => ({
          kind: 'item' as const,
          id: item.id,
          icon: item.icon,
          label: item.label,
          description: item.description,
          danger: item.danger,
          onselect: () => (item.id === 'repair' ? void run(() => bridge.repairApp()) : requestUninstall())
        }))
      : null
  );

  // ——— The log ———
  const logText = $derived(snapshot && snapshot.logTail.length > 0 ? snapshot.logTail.join('\n') : 'No log lines were captured.');

  async function copyLog(): Promise<void> {
    if (await copyText(logText)) {
      copied = true;
      setTimeout(() => (copied = false), COPIED_MS);
    } else if (logBox) {
      // No clipboard here: select the text so the keyboard can copy it.
      selectElementText(logBox);
    }
  }

  // The last lines are the ones that matter: open scrolled to the end.
  $effect(() => {
    if (!logOpen) return;
    void tick().then(() => {
      if (logBox) logBox.scrollTop = logBox.scrollHeight;
    });
  });

  // ——— Keys ———
  function onKeydown(event: KeyboardEvent): void {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'w') {
      event.preventDefault();
      void run(() => bridge.closeWindow());
    }
  }

  function primaryClick(primary: PrimaryView): void {
    void run(primaryActions[primary.id]);
  }
</script>

<svelte:window onkeydown={onKeydown} />

<div class="shell" bind:this={shell}>
  {#if view && snapshot && visible}
    <header class="masthead">
      <Wordmark height={18} />
      <span class="sr-only">Z-Vision Generator</span>
    </header>

    <LauncherHero
      mood={cheering ? 'cheerful' : view.hero.mood}
      mascotProgress={view.hero.mascotProgress}
      halo={view.hero.halo}
      percent={view.hero.percent}
      fault={view.hero.fault}
      update={view.hero.update}
      pool={view.hero.pool}
    />

    <div class="text" role="status">
      <p class="ui-area-label kicker" class:accent={view.kickerAccent}>{view.kicker}</p>
      {#key view.headline}
        <div class="swap" in:fly={{ y: 6, duration: motion(220) }} out:fade={{ duration: motion(120) }}>
          <h1 class="headline">{view.headline}</h1>
          {#if view.sub}
            <p class="sub">
              {#if view.sub.lead}<strong role={view.sub.leadAlert ? 'alert' : undefined}>{view.sub.lead}</strong>{/if}
              {view.sub.text}
            </p>
          {/if}
        </div>
      {/key}
    </div>

    <div class="action-zone">
      {#if view.notice}
        <LauncherNotice
          notice={view.notice}
          onlink={openLink}
          onupdate={() => void run(() => bridge.updateApp())}
          onlog={() => (logOpen = true)}
        />
      {/if}
      <div class="slot">
        {#key view.primary?.id ?? (view.caption ? 'caption' : 'none')}
          <div class="slot-in" in:fade={{ duration: motion(160) }}>
            {#if view.primary}
              <div class="ui-action-bar">
                <Button
                  main
                  data-primary
                  variant={view.primary.variant}
                  loading={view.primary.loading}
                  onclick={() => primaryClick(view.primary!)}
                >
                  {#if view.primary.icon}<Icon name={view.primary.icon} size={16} />{/if}{view.primary.label}
                </Button>
              </div>
            {:else if view.caption}
              <div class="caption">
                <span class="caption-label">{view.caption.label}</span>
                {#if view.caption.line}<span class="caption-line">{view.caption.line}</span>{/if}
              </div>
            {/if}
          </div>
        {/key}
      </div>
    </div>

    {#if view.controls.bar}
      <QuietBar
        autostart={view.controls.autostart ? snapshot.autostart : null}
        busy={view.controls.locked}
        menu={menuEntries}
        onautostart={(enabled) => bridge.setAutostart(enabled)}
        onquit={() => void run(() => bridge.quitLauncher())}
      />
    {/if}

    {#if view.notice?.kind === 'log'}
      <Sheet bind:open={logOpen} title={view.notice.sheetTitle}>
        {#snippet headActions()}
          <Button variant="quiet" aria-label="Copy the end of the log" onclick={copyLog}><Icon name="copy" size={14} />{copied ? 'Copied' : 'Copy'}</Button>
          <span class="sr-only" role="status">{copied ? 'Copied' : ''}</span>
        {/snippet}
        <div class="log-body">
          <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
          <pre class="log" bind:this={logBox} tabindex="0" aria-label={view.notice.sheetTitle}>{logText}</pre>
        </div>
      </Sheet>
    {/if}

    <ConfirmDialog bind:open={confirmOpen} question={UNINSTALL_DIALOG.question} info={UNINSTALL_DIALOG.info} confirmLabel={UNINSTALL_DIALOG.confirm} onconfirm={confirmUninstall}>
      <p>
        {UNINSTALL_DIALOG.staysBefore} <code class="path">{UNINSTALL_DIALOG.paths[0]}</code> {UNINSTALL_DIALOG.staysAnd}
        <code class="path">{UNINSTALL_DIALOG.paths[1]}</code>.
      </p>
      <p><LauncherLink label={UNINSTALL_DIALOG.link.label} onclick={() => openLink(UNINSTALL_DIALOG.link.id)} /></p>
    </ConfirmDialog>
  {/if}
</div>

<style>
  .shell {
    position: relative;
    isolation: isolate;
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
    background: var(--color-bg-base);
    color: var(--color-text-primary);
    font-family: var(--font-sans);
    font-size: var(--text-content);
  }
  .masthead { flex: none; display: flex; align-items: flex-end; justify-content: center; height: 40px; }

  /* The text zone is top-anchored and scrolls if it must; everything around it stays put. */
  .text { flex: 1 1 0; min-height: 0; display: grid; grid-template-rows: 16px 1fr; padding: 6px 24px 0; text-align: center; overflow-y: auto; }
  .kicker { height: 16px; line-height: 16px; }
  .kicker.accent { color: var(--color-primary-main); }
  .swap { grid-row: 2; display: flex; flex-direction: column; align-items: center; min-width: 0; }
  .headline { margin: 4px 0 0; font-family: var(--font-heading); font-size: var(--text-display); font-weight: 800; line-height: 1.15; letter-spacing: -0.01em; text-wrap: balance; color: var(--color-text-primary); }
  .sub { margin: 6px 0 0; max-width: 340px; font-size: var(--text-content); line-height: 1.5; text-wrap: pretty; color: var(--color-text-secondary); }
  .sub strong { display: block; font-weight: 600; color: var(--color-text-primary); }

  .action-zone { flex: none; display: flex; flex-direction: column; align-items: center; gap: 10px; padding: 0 32px 16px; }
  .slot { display: grid; width: 100%; min-height: var(--spacing-bar); }
  .slot-in { grid-area: 1 / 1; display: flex; flex-direction: column; justify-content: center; min-width: 0; }
  .slot-in .ui-action-bar { width: 100%; }
  .caption { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 3px; min-height: var(--spacing-bar); max-width: 100%; text-align: center; }
  .caption-label { font-size: var(--text-ui); font-weight: 600; color: var(--color-text-primary); }
  .caption-line { max-width: 340px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); }

  .log-body { padding: 12px 16px 16px; }
  .log {
    margin: 0;
    max-height: 216px;
    overflow: auto;
    padding: 10px 12px;
    border-radius: var(--radius-md);
    background: var(--color-bg-overlay);
    font-family: var(--font-mono);
    font-size: var(--text-meta);
    line-height: 15px;
    color: var(--color-text-secondary);
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  .path { font-family: var(--font-mono); color: var(--color-text-primary); }

  @media (max-height: 499px) {
    .masthead { height: 32px; }
  }
</style>
