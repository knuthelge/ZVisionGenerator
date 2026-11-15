<script lang="ts">
  import { onMount, tick } from 'svelte';
  import { pushEscapeLayer } from '$lib/actions/escapeStack';
  import { HOLD_MS } from '$lib/actions/hold';
  import { Button, Icon, Spinner } from '$lib/components/atoms';
  import ActionMenu, { type ActionMenuEntry } from '$lib/components/molecules/ActionMenu.svelte';
  import InspectorSection from '$lib/components/molecules/InspectorSection.svelte';
  import { requestConfirm } from '$lib/components/molecules/confirm.svelte';
  import type { AdjustSave } from '$lib/api/adjust';
  import { isTyping } from '$lib/keyboard';
  import { adjustAndReconcile, loadAdjustSection, rememberReplaceAsked, replaceNeedsAsking, saveAdjustSection, type AdjustSection } from '$lib/state/adjust';
  import {
    fitAspect,
    flipGeometry,
    frameSize,
    normalizedCrop,
    presetRatio,
    rotateGeometry,
    swapRatio,
    type AspectPreset,
    type Crop,
    type Size,
  } from '$lib/state/adjustGeometry';
  import type { LightSettings } from '$lib/state/adjustMath';
  import { cropSummary, lightSummary, mixSummary, signedLabel } from './adjustFormat';
  import { LIGHT_TOOLS, MIX_TOOLS } from './adjustTools';
  import { adjustBody, canReplace, draftChanged, neutralDraft, type AdjustDraft } from '$lib/state/adjustState';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { addToast } from '$lib/state/toasts.svelte';
  import type { GalleryAsset } from '$lib/types';
  import { createCompare, RELEASE_KEYS } from './adjustCompare.svelte';
  import AdjustPreview from './AdjustPreview.svelte';
  import CropControls from './CropControls.svelte';
  import CropFrame from './CropFrame.svelte';
  import BandSwatch from './BandSwatch.svelte';
  import LightControls from './LightControls.svelte';
  import MixControls from './MixControls.svelte';
  import PhoneLight from './PhoneLight.svelte';
  import PhoneMix from './PhoneMix.svelte';

  interface Props {
    /** The picture to adjust. */
    asset: GalleryAsset;
    /** Called when the editor should go away: cancelled, discarded, saved, or the picture would not open. */
    onclose: () => void;
    /** Called after a save, before `onclose`, with the saved asset and whether it replaced the original. */
    onsaved: (saved: GalleryAsset, replaced: boolean) => void;
  }

  let { asset, onclose, onsaved }: Props = $props();

  type Tab = 'crop' | 'light' | 'mixer';
  const TABS: { id: Tab; label: string; icon: 'crop' | 'sliders' | 'mixer' }[] = [
    { id: 'crop', label: 'Crop', icon: 'crop' },
    { id: 'light', label: 'Light & colour', icon: 'sliders' },
    { id: 'mixer', label: 'Colour mixer', icon: 'mixer' },
  ];

  const uid = $props.id();
  const cropHelpId = `${uid}-crop-help`;

  let rootEl = $state<HTMLElement | null>(null);
  let tabEls = $state<Partial<Record<Tab, HTMLElement>>>({});
  /** The tab of the layouts that have tabs. It always opens on Light & colour, whichever tab was used last. */
  let tab = $state<Tab>('light');
  /** The open section of the side panel: at most one, or none. It opens as it was left, and on Light & colour the first time. */
  let open = $state<AdjustSection>(loadAdjustSection());
  let draft = $state<AdjustDraft>(neutralDraft());
  /** The picture's upright size, once it has loaded. */
  let size = $state<Size | null>(null);
  let busy = $state<AdjustSave | null>(null);
  let confirming = false;
  let announcement = $state('');

  /** The tool on the phone's one large slider, and the band on the Colour mixer's. */
  let tool = $state<keyof LightSettings>('exposure');
  let band = $state<keyof LightSettings>('mix_red');
  let saveMenuOpen = $state(false);
  let saveButton = $state<HTMLElement | null>(null);

  const compare = createCompare();

  /** How the editor is composed: the structure follows the layout mode and nothing else (a touch screen only changes sizes). */
  const mode = $derived<'side' | 'bottom' | 'phone' | 'phone-side'>(
    layout.mode === 'wide' ? 'side' : layout.mode === 'tall' ? 'bottom' : layout.short ? 'phone-side' : 'phone',
  );
  const selectedTool = $derived(LIGHT_TOOLS.find((item) => item.key === tool) ?? LIGHT_TOOLS[0]);
  const selectedBand = $derived(MIX_TOOLS.find((item) => item.key === band) ?? MIX_TOOLS[0]);
  /** Whether the crop frame is up: the picture then shows the whole frame, and the frame's box and handles sit on it. */
  const cropOpen = $derived(mode === 'side' ? open === 'crop' : tab === 'crop');
  const saveItems = $derived<ActionMenuEntry[]>([
    {
      kind: 'item',
      id: 'save-copy',
      label: 'Save as copy',
      description: 'Adds an adjusted copy; the original stays',
      onselect: () => void save('copy'),
    },
    {
      kind: 'item',
      id: 'save-replace',
      label: 'Replace original',
      description: `Overwrites ${asset.filename}. It can't be restored.`,
      onselect: () => void save('replace'),
    },
  ]);

  const frame = $derived(size ? frameSize(draft.geometry.rotate, size.width, size.height) : null);
  const changed = $derived(draftChanged(draft, frame ?? undefined));
  const output = $derived(frame ? (draft.geometry.crop ?? frame) : null);
  const replaceable = $derived(canReplace(asset));
  const typeNote = $derived(/\.webp$/i.test(asset.filename) ? 'WebP images are saved as a PNG copy' : 'JPEGs are saved as a PNG copy');
  const canSave = $derived(changed && frame !== null);
  const cropped = $derived(frame !== null && normalizedCrop(draft.geometry.crop, frame) !== null);
  const summaries = $derived({
    crop: cropSummary(draft.aspect, draft.geometry, output, cropped),
    light: lightSummary(draft.light),
    mixer: mixSummary(draft.light),
  });

  // --- Edits --------------------------------------------------------------------------------------------------------
  /** Any edit ends a Compare that a tap left on. */
  function edited(): void {
    compare.clearToggle();
  }

  let announceTimer: ReturnType<typeof setTimeout> | undefined;
  /** Say the crop's size once the user has stopped changing it. */
  function announceCropSoon(): void {
    clearTimeout(announceTimer);
    announceTimer = setTimeout(() => {
      if (output) announcement = `Crop ${output.width} × ${output.height}`;
    }, 300);
  }

  function setLight(key: keyof LightSettings, value: number): void {
    draft = { ...draft, light: { ...draft.light, [key]: value } };
    edited();
  }

  function setCrop(crop: Crop): void {
    draft = { ...draft, geometry: { ...draft.geometry, crop } };
    edited();
  }

  function pickPreset(preset: AspectPreset): void {
    if (!frame) return;
    const ratio = presetRatio(preset, frame);
    const crop = ratio === null ? draft.geometry.crop : fitAspect(draft.geometry.crop, ratio, frame);
    draft = { ...draft, aspect: { preset, ratio }, geometry: { ...draft.geometry, crop } };
    edited();
    announceCropSoon();
  }

  function swap(): void {
    if (!frame) return;
    const aspect = swapRatio(draft.aspect);
    if (aspect === draft.aspect || aspect.ratio === null) return;
    draft = { ...draft, aspect, geometry: { ...draft.geometry, crop: fitAspect(draft.geometry.crop, aspect.ratio, frame) } };
    edited();
    announceCropSoon();
  }

  function rotate(direction: 'left' | 'right'): void {
    if (!size) return;
    // A locked ratio turns with the picture.
    draft = { ...draft, aspect: swapRatio(draft.aspect), geometry: rotateGeometry(draft.geometry, direction, size.width, size.height) };
    edited();
    announceCropSoon();
  }

  function flip(): void {
    if (!size) return;
    draft = { ...draft, geometry: flipGeometry(draft.geometry, size.width, size.height) };
    edited();
    announceCropSoon();
  }

  function resetAll(): void {
    if (busy !== null) return;
    draft = neutralDraft();
    edited();
    announceCropSoon();
  }

  /** The phone's Save: a menu of the two ways on a PNG, a straight "Save as copy" on any other file. */
  function phoneSave(): void {
    if (busy !== null) return;
    if (replaceable) saveMenuOpen = true;
    else void save('copy');
  }

  function pictureFailed(): void {
    addToast('Could not open the picture', 'error');
    onclose();
  }

  // --- Compare ------------------------------------------------------------------------------------------------------
  let press: { timer: ReturnType<typeof setTimeout>; holding: boolean } | null = null;
  /** A pointer touched the button since its last click, so the click that follows is the pointer's and not a key's. */
  let pointerTouched = false;

  function toggleCompare(): void {
    compare.toggle();
    announcement = compare.toggled ? 'Showing the original' : 'Showing your changes';
  }

  function comparePointerDown(event: PointerEvent): void {
    if (event.pointerType === 'mouse' && event.button !== 0) return;
    pointerTouched = true;
    if (press) clearTimeout(press.timer);
    const next = { timer: setTimeout(() => { next.holding = true; compare.hold('button'); }, HOLD_MS), holding: false };
    press = next;
  }

  function comparePointerUp(): void {
    if (!press) return;
    clearTimeout(press.timer);
    const { holding } = press;
    press = null;
    if (holding) compare.release('button');
    else toggleCompare();
  }

  /** The pointer left or was taken over: no tap, and a hold ends. */
  function comparePointerAway(): void {
    if (!press) return;
    clearTimeout(press.timer);
    if (press.holding) compare.release('button');
    press = null;
    pointerTouched = false;
  }

  function compareClick(): void {
    if (pointerTouched) {
      pointerTouched = false;
      return;
    }
    // Enter or Space: a key press clicks without any pointer.
    toggleCompare();
  }

  let stopPictureHold: (() => void) | null = null;

  function holdPicture(): void {
    compare.hold('picture');
    stopPictureHold?.();
    const stop = (): void => {
      compare.release('picture');
      window.removeEventListener('pointerup', stop);
      window.removeEventListener('pointercancel', stop);
      stopPictureHold = null;
    };
    window.addEventListener('pointerup', stop);
    window.addEventListener('pointercancel', stop);
    stopPictureHold = stop;
  }

  let heldCode: string | null = null;

  function onWindowKeydown(event: KeyboardEvent): void {
    if (event.repeat || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key !== 'm' && event.key !== 'M') return;
    const target = event.target;
    // A range is no place for typing: M still compares there. Any other field keeps its own M.
    if (isTyping(target) && !(target instanceof HTMLInputElement && target.type === 'range')) return;
    if (target instanceof Element && target.closest('[role="alertdialog"]')) return;
    heldCode = event.code;
    compare.hold('key');
  }

  function onWindowKeyup(event: KeyboardEvent): void {
    if (compare.held !== 'key') return;
    if (event.code === heldCode || RELEASE_KEYS.has(event.key)) {
      heldCode = null;
      compare.release('key');
    }
  }

  // --- Cancel and save ----------------------------------------------------------------------------------------------
  async function cancel(): Promise<void> {
    if (busy !== null || confirming) return;
    if (changed) {
      confirming = true;
      const discard = await requestConfirm({ question: 'Discard your changes?', confirmLabel: 'Discard', cancelLabel: 'Keep editing' });
      confirming = false;
      if (!discard) return;
    }
    onclose();
  }

  async function save(kind: AdjustSave): Promise<void> {
    if (busy !== null || confirming || !canSave || !frame) return;
    if (kind === 'replace' && replaceNeedsAsking()) {
      confirming = true;
      const approved = await requestConfirm({ question: `Replace ${asset.filename}?`, info: "The original can't be restored.", confirmLabel: 'Replace' });
      confirming = false;
      if (!approved) return;
      rememberReplaceAsked();
    }
    busy = kind;
    compare.releaseAll();
    let saved: GalleryAsset | null;
    try {
      saved = await adjustAndReconcile(asset, adjustBody(asset, draft, kind, frame));
    } finally {
      busy = null;
    }
    // A failure has shown its toast and leaves the editor open with the user's values.
    if (!saved) return;
    onsaved(saved, kind === 'replace');
    onclose();
  }

  $effect(() => pushEscapeLayer(() => void cancel()));

  $effect(() => () => {
    clearTimeout(announceTimer);
    if (press) clearTimeout(press.timer);
    stopPictureHold?.();
  });

  // --- Sections, tabs and focus -------------------------------------------------------------------------------------
  onMount(() => {
    if (mode === 'side') {
      const heads = [...(rootEl?.querySelectorAll<HTMLElement>('.adjust-sections h3 > button') ?? [])];
      (heads.find((head) => head.getAttribute('aria-expanded') === 'true') ?? heads[0])?.focus();
    } else tabEls[tab]?.focus();
  });

  /** Open a section and close the one that was open, or close it when it already is. The focus stays on its header. */
  function toggleSection(section: Exclude<AdjustSection, 'none'>): void {
    const wasCropOpen = open === 'crop';
    open = open === section ? 'none' : section;
    saveAdjustSection(open);
    if (wasCropOpen !== (open === 'crop')) announcement = open === 'crop' ? 'Crop frame shown' : 'Crop frame hidden';
  }

  async function selectTab(next: Tab, focus = false): Promise<void> {
    tab = next;
    if (!focus) return;
    await tick();
    tabEls[next]?.focus();
  }

  function onTabKeydown(event: KeyboardEvent): void {
    const index = TABS.findIndex((item) => item.id === tab);
    const step = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0;
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? TABS.length - 1 : step === 0 ? -1 : (index + step + TABS.length) % TABS.length;
    if (next < 0) return;
    event.preventDefault();
    void selectTab(TABS[next].id, true);
  }

  const FOCUSABLE = 'button:not([disabled]), input:not([disabled]), [href], [tabindex]:not([tabindex="-1"])';

  /** Keep Tab inside the editor, wrapping at both ends, as the Edit painter does. */
  function onRootKeydown(event: KeyboardEvent): void {
    const root = rootEl;
    if (!root || event.key !== 'Tab' || event.defaultPrevented) return;
    const focusable = [...root.querySelectorAll<HTMLElement>(FOCUSABLE)].filter((el) => !el.closest('[inert], [hidden]'));
    if (focusable.length === 0) {
      event.preventDefault();
      return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const at = document.activeElement;
    if (event.shiftKey && (at === first || at === root)) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && (at === last || at === root)) {
      event.preventDefault();
      first.focus();
    }
  }
</script>

<svelte:window onkeydown={onWindowKeydown} onkeyup={onWindowKeyup} onblur={() => compare.releaseAll()} />
<svelte:document onvisibilitychange={() => { if (document.visibilityState === 'hidden') compare.releaseAll(); }} />

{#snippet cropOverlay(box: { frame: Size; scale: number })}
  {#if cropOpen && !compare.showing}
    <CropFrame
      frame={box.frame}
      crop={draft.geometry.crop}
      ratio={draft.aspect.ratio}
      scale={box.scale}
      coarse={layout.coarse}
      disabled={busy !== null}
      describedBy={cropHelpId}
      onchange={setCrop}
      oncommit={announceCropSoon}
    />
  {/if}
{/snippet}

{#snippet compareButton(iconOnly: boolean)}
  <Button
    data-testid="adjust-compare"
    icon={iconOnly}
    aria-pressed={compare.showing}
    aria-label="Compare with the original"
    title="Compare — tap, or hold M"
    onpointerdown={comparePointerDown}
    onpointerup={comparePointerUp}
    onpointercancel={comparePointerAway}
    onpointerleave={comparePointerAway}
    onclick={compareClick}
  >{#if iconOnly}<Icon name="compare" size={18} />{:else}<Icon name="compare" size={14} />Compare <kbd class="adjust-kbd">M</kbd>{/if}</Button>
{/snippet}

{#snippet resetButton(small: boolean)}
  <Button
    size={small ? 'sm' : 'md'}
    disabled={!changed || busy !== null}
    title="Set every slider, the crop, rotation and flip back"
    onclick={resetAll}
  ><Icon name="reset" size={14} />Reset all</Button>
{/snippet}

{#snippet tabsBar(extra: string, icons: boolean)}
  <div class="ui-segmented adjust-tabs {extra}" role="tablist" aria-label="Adjust tools" tabindex="-1" onkeydown={onTabKeydown}>
    {#each TABS as item (item.id)}
      <button
        bind:this={tabEls[item.id]}
        type="button"
        role="tab"
        id="{uid}-tab-{item.id}"
        aria-selected={tab === item.id}
        aria-controls={tab === item.id ? `${uid}-panel-${item.id}` : undefined}
        tabindex={tab === item.id ? 0 : -1}
        onclick={() => selectTab(item.id)}
      >{#if icons}<Icon name={item.icon} size={14} />{/if}{item.label}</button>
    {/each}
  </div>
{/snippet}

{#snippet typeNoteLine()}
  <p class="adjust-type-note">{typeNote}</p>
{/snippet}

<!-- Cancel, Replace and Save as copy: the footer of the side panel and the first row of the tablet's dock. -->
{#snippet actionButtons(main: boolean)}
  <Button onclick={() => void cancel()}>Cancel</Button>
  {#if replaceable}
    <button
      type="button"
      class="ui-btn"
      title="Replace the original file"
      disabled={!canSave}
      aria-busy={busy === 'replace' ? 'true' : undefined}
      onclick={() => void save('replace')}
    >{#if busy === 'replace'}<Spinner size="sm" />Saving…{:else}Replace{/if}</button>
  {/if}
  <button
    type="button"
    class="ui-btn ui-btn-primary"
    class:ui-btn-main={main}
    disabled={!canSave}
    aria-busy={busy === 'copy' ? 'true' : undefined}
    onclick={() => void save('copy')}
  >{#if busy === 'copy'}<Spinner size="sm" />Saving…{:else}Save as copy{/if}</button>
{/snippet}

{#snippet phoneCancel()}
  <Button class="phone-cancel" variant="quiet" onclick={() => void cancel()}>Cancel</Button>
{/snippet}

{#snippet phoneSaveButton()}
  <button
    bind:this={saveButton}
    type="button"
    class="ui-btn ui-btn-primary"
    aria-haspopup={replaceable ? 'menu' : undefined}
    aria-expanded={replaceable ? saveMenuOpen : undefined}
    aria-busy={busy !== null ? 'true' : undefined}
    disabled={!canSave}
    onclick={phoneSave}
  >{#if busy !== null}<Spinner size="sm" />Saving…{:else if replaceable}Save<Icon name="chevdown" size={14} />{:else}Save as copy{/if}</button>
{/snippet}

<!-- The phone's header line: the tool and its value (Crop and its size on the Crop tab), with Reset all at the right. -->
{#snippet dockLine()}
  <div class="dock-line">
    {#if tab === 'light'}
      <b>{selectedTool.label}</b>
      <span class="adj-value" data-set={draft.light[tool] !== 0 ? '' : undefined}>{signedLabel(draft.light[tool])}</span>
    {:else if tab === 'mixer'}
      <BandSwatch band={selectedBand.band} />
      <b>{selectedBand.label}</b>
      <span class="adj-value" data-set={draft.light[band] !== 0 ? '' : undefined}>{signedLabel(draft.light[band])}</span>
    {:else}
      <b>Crop</b>
      {#if output}<span class="adj-value" data-set data-testid="adjust-crop-size">{output.width} × {output.height}</span>{/if}
    {/if}
    {@render resetButton(true)}
  </div>
{/snippet}

{#snippet phoneLight(hideNames: boolean)}
  <PhoneLight light={draft.light} {tool} {hideNames} onselect={(next) => { tool = next; }} onchange={setLight} />
{/snippet}

{#snippet phoneMix(hideNames: boolean)}
  <PhoneMix light={draft.light} {band} {hideNames} onselect={(next) => { band = next; }} onchange={setLight} />
{/snippet}

{#snippet cropControls(variant: 'panel' | 'dock' | 'phone' | 'column')}
  <CropControls {variant} aspect={draft.aspect} {output} helpId={cropHelpId} onpreset={pickPreset} onswap={swap} onrotate={rotate} onflip={flip} />
{/snippet}

<!-- The root covers the whole viewport, the viewer's backdrop included: a click in the gutter does nothing and never closes the viewer. -->
<div
  bind:this={rootEl}
  class="adjust ui-viewport"
  role="dialog"
  aria-modal="true"
  aria-label="Adjust {asset.filename}"
  tabindex="-1"
  data-own-drag
  data-layout={mode}
  data-busy={busy !== null ? '' : undefined}
  onkeydown={onRootKeydown}
>
  <div class="adjust-panel">
    {#if mode === 'side' || mode === 'bottom'}
      <div class="adjust-bar">
        <div class="adjust-title">
          <b>{asset.filename}</b>
          <small>Adjust{#if output}{' · '}<span class="mono">{output.width} × {output.height}</span>{/if}</small>
        </div>
        {@render compareButton(false)}
        {@render resetButton(false)}
      </div>
    {:else if mode === 'phone'}
      <div class="phone-top-wrap">
        <div class="phone-top">
          {@render phoneCancel()}
          {@render compareButton(true)}
          {@render phoneSaveButton()}
        </div>
        {#if !replaceable}<div class="phone-note">{@render typeNoteLine()}</div>{/if}
      </div>
    {/if}

    <!-- The picture sits in one place in this tree whatever the layout, so turning the device does not reload it. A phone held sideways
         puts its column first in the source (focus follows the actions) and the row runs in reverse. -->
    <div class="adjust-main">
      {#if mode === 'phone-side'}
        <section class="phone-col" aria-label="Adjust controls">
          <div class="phone-actions">
            {@render phoneCancel()}
            <span class="phone-grow"></span>
            {@render compareButton(true)}
            {@render phoneSaveButton()}
          </div>
          {#if !replaceable}<div class="phone-note">{@render typeNoteLine()}</div>{/if}
          <div class="phone-col-body adjust-controls" inert={busy !== null}>
            <div class="phone-col-controls">
              {@render dockLine()}
              <div class="phone-col-panel" role="tabpanel" id="{uid}-panel-{tab}" aria-labelledby="{uid}-tab-{tab}">
                {#if tab === 'light'}{@render phoneLight(true)}{:else if tab === 'mixer'}{@render phoneMix(true)}{:else}{@render cropControls('column')}{/if}
              </div>
            </div>
            {@render tabsBar('phone-tabs', false)}
          </div>
        </section>
      {/if}

      <AdjustPreview
        src={asset.url}
        transparent={asset.transparent === true}
        light={draft.light}
        geometry={draft.geometry}
        region={cropOpen ? 'frame' : 'crop'}
        original={compare.showing}
        onloaded={(loaded) => { size = { width: loaded.width, height: loaded.height }; }}
        onfail={pictureFailed}
        onpicturehold={holdPicture}
        overlay={cropOverlay}
      />

      {#if mode === 'side'}
        <aside class="adjust-side" aria-label="Adjust controls">
          <div class="adjust-body adjust-sections adjust-controls" inert={busy !== null}>
            <InspectorSection id="adjust-crop" title="Crop" summary={summaries.crop.text} active={summaries.crop.set} open={open === 'crop'} ontoggle={() => toggleSection('crop')}>
              <div class="adj-section-inner">{@render cropControls('panel')}</div>
            </InspectorSection>
            <InspectorSection id="adjust-light" title="Light & colour" summary={summaries.light.text} active={summaries.light.set} open={open === 'light'} ontoggle={() => toggleSection('light')}>
              <div class="adj-section-inner"><LightControls light={draft.light} coarse={layout.coarse} onchange={setLight} /></div>
            </InspectorSection>
            <InspectorSection id="adjust-mixer" title="Colour mixer" summary={summaries.mixer.text} active={summaries.mixer.set} open={open === 'mixer'} ontoggle={() => toggleSection('mixer')}>
              <div class="adj-section-inner"><MixControls light={draft.light} coarse={layout.coarse} onchange={setLight} /></div>
            </InspectorSection>
          </div>
          <div class="adjust-foot">
            {#if !replaceable}{@render typeNoteLine()}{/if}
            <div class="ui-action-bar">{@render actionButtons(true)}</div>
          </div>
        </aside>
      {:else if mode === 'bottom'}
        <section class="adjust-dock" aria-label="Adjust controls">
          <div class="dock-row">
            <div class="adjust-head dock-tabs adjust-controls" inert={busy !== null}>{@render tabsBar('', true)}</div>
            <span class="dock-spacer"></span>
            {#if !replaceable}{@render typeNoteLine()}{/if}
            <div class="ui-action-bar dock-actions">{@render actionButtons(false)}</div>
          </div>
          <div class="dock-body adjust-controls" role="tabpanel" id="{uid}-panel-{tab}" aria-labelledby="{uid}-tab-{tab}" inert={busy !== null}>
            {#if tab === 'light'}
              <LightControls light={draft.light} coarse={layout.coarse} dock onchange={setLight} />
            {:else if tab === 'mixer'}
              <MixControls light={draft.light} coarse={layout.coarse} dock onchange={setLight} />
            {:else}
              {@render cropControls('dock')}
            {/if}
          </div>
        </section>
      {:else if mode === 'phone'}
        <section class="phone-dock adjust-controls" aria-label="Adjust controls" inert={busy !== null}>
          {@render dockLine()}
          <!-- Both panels share one grid cell, so the dock (and the picture above it) keeps its size on either tab. -->
          <div class="dock-panels">
            {#each TABS as item (item.id)}
              <div
                class="dock-panel"
                role="tabpanel"
                id="{uid}-panel-{item.id}"
                aria-labelledby="{uid}-tab-{item.id}"
                data-off={tab === item.id ? undefined : ''}
                inert={tab !== item.id}
                aria-hidden={tab === item.id ? undefined : 'true'}
              >
                {#if item.id === 'light'}{@render phoneLight(false)}{:else if item.id === 'mixer'}{@render phoneMix(false)}{:else}{@render cropControls('phone')}{/if}
              </div>
            {/each}
          </div>
          {@render tabsBar('phone-tabs', false)}
        </section>
      {/if}
    </div>
  </div>
  <div class="sr-only" role="status" aria-live="polite" data-testid="adjust-announce">{announcement}</div>
</div>

{#if (mode === 'phone' || mode === 'phone-side') && replaceable}
  <ActionMenu open={saveMenuOpen} anchor={saveButton} items={saveItems} label="Save" onclose={() => { saveMenuOpen = false; }} />
{/if}

<style>
  .adjust { z-index: calc(var(--z-viewer) + 1); display: flex; flex-direction: column; padding: 24px; user-select: none; outline: none; }
  .adjust-panel { position: relative; display: flex; flex: 1; min-height: 0; flex-direction: column; overflow: hidden; border-radius: var(--radius-lg); background: var(--color-bg-base); box-shadow: 0 24px 64px var(--color-scrim); }

  /* The top bar has the viewer bar's metrics. */
  .adjust-bar { display: flex; flex: none; align-items: center; gap: 6px; min-height: 44px; padding: 6px 12px 6px 16px; background: var(--color-bg-surface); }
  .adjust-title { display: flex; flex: 1 1 auto; min-width: 7rem; flex-direction: column; }
  .adjust-title b { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-content); font-weight: 600; }
  .adjust-title small { font-size: var(--text-meta); color: var(--color-text-muted); }
  .adjust-title .mono { font-family: var(--font-mono); }
  .adjust-kbd { padding: 0 4px; border-radius: var(--radius-xs); background: var(--color-bg-surface); font-family: var(--font-mono); font-size: var(--text-meta); font-weight: 400; line-height: 15px; color: var(--color-text-muted); }
  :global(.ui-btn[aria-pressed='true']) .adjust-kbd { background: transparent; color: inherit; }

  .adjust-main { display: flex; flex: 1; min-height: 0; }
  .adjust[data-layout='bottom'] .adjust-main, .adjust[data-layout='phone'] .adjust-main { flex-direction: column; }
  .adjust[data-layout='phone-side'] .adjust-main { flex-direction: row-reverse; }

  /* A phone is full screen, like the viewer, with the picture 24px from every edge (room for a handle's 44px hit area). */
  .adjust[data-layout^='phone'] { padding: 0; --adjust-stage-pad: 24px; }
  .adjust[data-layout^='phone'] .adjust-panel { border-radius: 0; box-shadow: none; }
  .adjust[data-layout='phone-side'] { --adjust-stage-pad: 24px 24px calc(12px + env(safe-area-inset-bottom, 0px)) max(24px, env(safe-area-inset-left, 0px)); }

  /* The side panel takes the Details panel's slot and width. */
  .adjust-side { display: flex; flex: none; flex-direction: column; width: 340px; min-height: 0; background: var(--color-bg-surface); }
  .adjust-head { flex: none; padding: 12px 16px 4px; }
  .adjust-tabs { display: flex; width: 100%; }
  .adjust-tabs > button { flex: 1; justify-content: center; gap: 5px; padding: 0 6px; }
  /* The side panel's body is the accordion: it scrolls when the open section is taller than the panel, and the headers stay pinned. */
  .adjust-body { display: flex; flex: 1; min-height: 0; flex-direction: column; overflow-y: auto; overscroll-behavior: contain; }
  .adj-section-inner { display: flex; flex-direction: column; gap: 14px; padding: 12px 16px 18px; }
  .adjust-foot { flex: none; padding: 12px 16px; border-top: 1px solid var(--color-border-subtle); }
  .adjust-foot .ui-action-bar .ui-btn-main { min-width: 0; }
  .adjust-type-note { margin: 0; font-size: var(--text-meta); line-height: 1.4; color: var(--color-text-muted); }
  .adjust-foot > .adjust-type-note { margin-bottom: 8px; text-align: right; }

  /* The tablet's dock: as tall as its content, under the picture. */
  .adjust-dock { display: flex; flex: none; flex-direction: column; gap: 14px; padding: 12px 20px 16px; background: var(--color-bg-surface); }
  .dock-row { display: flex; align-items: center; gap: 8px; }
  .dock-row .adjust-head { padding: 0; }
  .dock-tabs { width: 400px; }
  .dock-spacer { flex: 1; }
  .dock-row > .adjust-type-note { margin-right: 4px; }
  .dock-actions .ui-btn-primary { min-width: 150px; }
  .dock-body { display: flex; flex-direction: column; gap: 12px; }

  /* The phone: the top bar, then the dock under the picture. */
  .phone-top-wrap { flex: none; padding-top: env(safe-area-inset-top, 0px); background: var(--color-bg-surface); }
  .phone-top { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 8px; padding: 6px max(8px, env(safe-area-inset-right, 0px)) 6px max(8px, env(safe-area-inset-left, 0px)); }
  .phone-top > :first-child { justify-self: start; }
  .phone-top > :last-child { justify-self: end; }
  .phone-top :global(.ui-btn), .phone-actions :global(.ui-btn) { height: var(--spacing-bar); padding: 0 14px; font-size: var(--text-content); }
  .phone-top :global(.ui-btn-icon), .phone-actions :global(.ui-btn-icon) { width: var(--spacing-bar); padding: 0; }
  .phone-top :global(.phone-cancel), .phone-actions :global(.phone-cancel) { color: var(--color-text-secondary); }
  .phone-note { display: flex; justify-content: flex-end; padding: 0 max(12px, env(safe-area-inset-right, 0px)) 6px max(12px, env(safe-area-inset-left, 0px)); background: var(--color-bg-surface); }

  .phone-dock { --row-bleed: max(12px, env(safe-area-inset-left, 0px)); display: flex; flex: none; flex-direction: column; gap: 10px; padding: 10px max(12px, env(safe-area-inset-right, 0px)) calc(8px + env(safe-area-inset-bottom, 0px)) max(12px, env(safe-area-inset-left, 0px)); background: var(--color-bg-surface); }
  .dock-line { display: flex; align-items: center; gap: 8px; min-height: 32px; }
  .dock-line b { font-size: var(--text-content); font-weight: 600; }
  .dock-line .adj-value { font-family: var(--font-mono); font-size: var(--text-content); color: var(--color-text-muted); }
  .dock-line .adj-value[data-set] { color: var(--color-text-primary); }
  .dock-line :global(.ui-btn) { margin-left: auto; }
  .phone-tabs { display: flex; }
  /* Three tabs without icons: "Light & colour" and "Colour mixer" fit unabbreviated. */
  .phone-tabs > button { flex: 1; height: calc(var(--spacing-bar) - 4px); padding: 0 4px; font-size: var(--text-content); }

  /* Both tab panels in one grid cell: the dock is as tall as the taller one, so the picture never moves between tabs. */
  .dock-panels { display: grid; grid-template-columns: minmax(0, 1fr); }
  .dock-panel { display: flex; min-width: 0; grid-area: 1 / 1; flex-direction: column; justify-content: space-between; gap: 10px; }
  .dock-panel[data-off] { visibility: hidden; }

  /* A phone held sideways: no top bar (the actions head the column), the picture takes the full height. */
  .phone-col { --dial: 44px; --row-bleed: 12px; --tool-w: 52px; display: flex; flex: none; flex-direction: column; gap: 10px; width: calc(320px + env(safe-area-inset-right, 0px)); padding: 10px max(12px, env(safe-area-inset-right, 0px)) calc(8px + env(safe-area-inset-bottom, 0px)) 12px; background: var(--color-bg-surface); }
  .phone-col .phone-tabs > button { font-size: var(--text-ui); }
  .phone-actions { display: flex; align-items: center; gap: 8px; }
  .phone-grow { flex: 1; }
  .phone-col .phone-note { margin-top: -4px; padding: 0; background: transparent; }
  .phone-col-body { display: flex; flex: 1; min-height: 0; flex-direction: column; justify-content: space-between; gap: 10px; }
  .phone-col-controls, .phone-col-panel { display: flex; flex-direction: column; gap: 10px; }

  /* While a save runs the controls are inert and dim; the pressed button stays as it was. */
  .adjust[data-busy] .adjust-controls { opacity: 0.5; }
</style>
