<script lang="ts">
  import Button from '$lib/components/atoms/Button.svelte';
  import Icon from '$lib/components/atoms/Icon.svelte';
  import LauncherLink from './LauncherLink.svelte';
  import ReasonCard from './ReasonCard.svelte';
  import type { NoticeView } from './launcherView';
  import type { LinkId } from './types';

  interface Props {
    notice: NoticeView;
    onlink: (id: LinkId) => void;
    onupdate: () => void;
    onlog: () => void;
  }

  let { notice, onlink, onupdate, onlog }: Props = $props();
</script>

<!-- The notice slot above the action: a reason, a note, an alert, the log button or what stays. -->
<div class="notice" data-kind={notice.kind}>
  {#if notice.kind === 'reason'}
    <ReasonCard
      name={notice.name}
      found={notice.found}
      needed={notice.needed}
      link={notice.link ? { label: notice.link.label, onopen: () => onlink(notice.link!.id) } : undefined}
    />
  {:else if notice.kind === 'note'}
    <p class="note">
      <Icon name={notice.icon} size={14} />
      <span>{#if notice.lead}<b>{notice.lead}</b>{' '}{/if}{notice.text}</span>
    </p>
  {:else if notice.kind === 'update'}
    <div class="ui-alert ui-alert-info alert">
      <Icon name="up" size={14} />
      <div class="alert-text"><b>{notice.lead}</b> {notice.text} <LauncherLink label={notice.link.label} onclick={() => onlink(notice.link.id)} /></div>
      <Button onclick={onupdate}>{notice.button}</Button>
    </div>
  {:else if notice.kind === 'newer'}
    <div class="ui-alert ui-alert-warning alert">
      <Icon name="alert" size={14} />
      <div class="alert-text"><b>{notice.lead}</b> {notice.text} <LauncherLink label={notice.link.label} onclick={() => onlink(notice.link.id)} /></div>
    </div>
  {:else if notice.kind === 'log'}
    <Button variant="quiet" size="sm" aria-haspopup="dialog" onclick={onlog}><Icon name="file" size={14} />{notice.label}</Button>
  {:else}
    <div class="kept">
      {#each notice.entries as entry (entry.path)}
        <code>{entry.path}</code><span>{entry.what}</span>
      {/each}
      <p>{notice.help} <LauncherLink label={notice.link.label} onclick={() => onlink(notice.link.id)} /></p>
    </div>
  {/if}
</div>

<style>
  .notice { display: flex; justify-content: center; width: 100%; }
  .notice[data-kind='kept'] { width: calc(100% + 16px); }
  .notice :global(.ui-btn-quiet) { color: var(--color-text-secondary); }
  .note { display: flex; gap: 8px; margin: 0; max-width: 340px; font-size: var(--text-ui); line-height: 1.5; color: var(--color-text-secondary); text-align: left; }
  .note > :global(svg) { flex: none; margin-top: 2px; color: var(--color-primary-main); }
  .note b { font-weight: 600; color: var(--color-text-primary); }
  .alert { width: 100%; box-sizing: border-box; align-items: center; text-align: left; }
  .alert > :global(svg) { align-self: flex-start; margin-top: 2px; }
  .alert > :global(.ui-btn) { margin: -2px -2px -2px 4px; }
  .alert-text { flex: 1; min-width: 0; }
  .alert-text b { font-weight: 600; }
  .kept {
    box-sizing: border-box;
    width: 100%;
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 4px 12px;
    padding: 10px 14px;
    border-radius: var(--radius-md);
    background: var(--color-bg-surface);
    text-align: left;
    font-size: var(--text-ui);
    line-height: 1.45;
  }
  .kept code { font-family: var(--font-mono); color: var(--color-text-primary); }
  .kept span { color: var(--color-text-secondary); }
  .kept p { grid-column: 1 / -1; margin: 4px 0 0; color: var(--color-text-muted); }
</style>
