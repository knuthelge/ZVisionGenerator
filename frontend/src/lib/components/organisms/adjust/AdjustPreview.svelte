<script lang="ts">
  import type { Snippet } from 'svelte';
  import { hold } from '$lib/actions/hold';
  import { Icon, Spinner } from '$lib/components/atoms';
  import { fitRect } from '$lib/components/molecules/viewerFlight';
  import { cssFrameTransform, frameSize, quadCorners, type CropRegion, type Geometry, type Size } from '$lib/state/adjustGeometry';
  import { mixSettings, NEUTRAL_LIGHT, previewTables, type LightSettings, type PreviewTables } from '$lib/state/adjustMath';
  import { createRenderer, textureSize, type AdjustRenderer } from './adjustGl';

  /** What the editor draws over the picture, such as the crop frame; it fills the picture's box. */
  interface OverlayBox {
    /** The displayed frame's size, in source pixels. */
    frame: Size;
    /** CSS pixels per source pixel of the region on show. */
    scale: number;
  }

  interface Props {
    /** The picture to adjust. */
    src: string;
    /** A cut-out: the checkerboard shows behind it. */
    transparent?: boolean;
    light: LightSettings;
    geometry: Geometry;
    /** `frame` shows the whole rotated and flipped frame (the Crop tab), `crop` the cropped result. */
    region: CropRegion;
    /** Show the picture as it was: neutral colour at the same region, with the "Original" tag. */
    original: boolean;
    /** Called once the picture has loaded, with its upright size in pixels. */
    onloaded: (size: Size) => void;
    /** Called when the picture could not be opened. */
    onfail: () => void;
    /** Called when a finger or pen holds the picture, apart from what the overlay claims with `data-crop-part`. */
    onpicturehold?: () => void;
    overlay?: Snippet<[OverlayBox]>;
  }

  let { src, transparent = false, light, geometry, region, original, onloaded, onfail, onpicturehold, overlay }: Props = $props();

  let natural = $state<Size | null>(null);
  let image = $state<HTMLImageElement | null>(null);
  let fitEl = $state<HTMLElement | null>(null);
  let fit = $state<Size>({ width: 0, height: 0 });
  let canvas = $state<HTMLCanvasElement | null>(null);
  /** The renderer is live; false before it is built, without WebGL2 and while the context is lost. */
  let glLive = $state(false);
  /** The renderer has been tried, so the fallback is not shown for the frame before it exists. */
  let glTried = $state(false);
  let renderer: AdjustRenderer | null = null;

  // --- The picture --------------------------------------------------------------------------------------------------
  $effect(() => {
    const url = src;
    let current = true;
    const loading = new Image();
    loading.src = url;
    loading.decode().then(
      () => {
        if (!current) return;
        if (loading.naturalWidth < 1 || loading.naturalHeight < 1) {
          onfail();
          return;
        }
        image = loading;
        natural = { width: loading.naturalWidth, height: loading.naturalHeight };
        onloaded(natural);
      },
      () => {
        if (current) onfail();
      },
    );
    return () => {
      current = false;
    };
  });

  const frame = $derived(natural ? frameSize(geometry.rotate, natural.width, natural.height) : null);
  const shown = $derived(frame ? (region === 'crop' && geometry.crop ? geometry.crop : frame) : null);
  const picture = $derived(shown && fit.width > 0 && fit.height > 0 ? fitRect({ left: 0, top: 0, width: fit.width, height: fit.height }, shown.width / shown.height) : { left: 0, top: 0, width: 0, height: 0 });
  const scale = $derived(shown && shown.width > 0 ? picture.width / shown.width : 0);

  // The room for the picture: what is left of the stage once the notice is in.
  $effect(() => {
    const el = fitEl;
    if (!el) return;
    const read = (): void => {
      fit = { width: el.clientWidth, height: el.clientHeight };
    };
    read();
    const observer = new ResizeObserver(read);
    observer.observe(el);
    return () => observer.disconnect();
  });

  // --- The WebGL2 renderer ------------------------------------------------------------------------------------------
  /** The picture drawn smoothly at `size` on a scratch canvas, which the texture is uploaded from. */
  let scratch: HTMLCanvasElement | null = null;
  function scaledPicture(img: HTMLImageElement, size: Size): TexImageSource {
    scratch ??= document.createElement('canvas');
    scratch.width = size.width;
    scratch.height = size.height;
    const context = scratch.getContext('2d');
    if (!context) return img;
    context.imageSmoothingQuality = 'high';
    context.drawImage(img, 0, 0, size.width, size.height);
    return scratch;
  }

  /** The picture and the texture size the renderer holds now, so a frame uploads again only when either changes. */
  let uploaded: { image: HTMLImageElement; width: number; height: number } | null = null;

  $effect(() => {
    const el = canvas;
    if (!el) return;
    function build(): void {
      uploaded = null;
      renderer = createRenderer(el!);
      glLive = renderer !== null;
      glTried = true;
    }
    function onlost(event: Event): void {
      // Without preventDefault the browser never restores the context.
      event.preventDefault();
      renderer = null;
      glLive = false;
    }
    build();
    el.addEventListener('webglcontextlost', onlost);
    el.addEventListener('webglcontextrestored', build);
    return () => {
      el.removeEventListener('webglcontextlost', onlost);
      el.removeEventListener('webglcontextrestored', build);
      renderer?.dispose();
      renderer = null;
      glLive = false;
    };
  });

  // At most one draw per animation frame, with the tables rebuilt only when a slider moved.
  let tablesFor: { key: string; tables: PreviewTables } | null = null;
  let neutralTables: PreviewTables | null = null;
  let frameRequest = 0;

  function tablesOf(settings: LightSettings, isOriginal: boolean): PreviewTables {
    if (isOriginal) return (neutralTables ??= previewTables(NEUTRAL_LIGHT));
    const key = [settings.exposure, settings.contrast, settings.highlights, settings.shadows, settings.saturation, settings.vibrance, settings.warmth, ...mixSettings(settings)].join(',');
    if (tablesFor?.key !== key) tablesFor = { key, tables: previewTables(settings) };
    return tablesFor.tables;
  }

  function draw(): void {
    frameRequest = 0;
    if (!renderer || !natural || !image || !shown || picture.width < 1 || picture.height < 1) return;
    const density = Math.min(2, window.devicePixelRatio || 1);
    const width = Math.ceil(picture.width * density);
    const height = Math.ceil(picture.height * density);
    // The texture follows the canvas: the region on show maps to about one texture pixel per canvas pixel.
    const size = textureSize(natural, shown, width);
    if (uploaded?.image !== image || uploaded.width !== size.width || uploaded.height !== size.height) {
      renderer.setImage(scaledPicture(image, size));
      uploaded = { image, width: size.width, height: size.height };
    }
    renderer.draw(tablesOf(light, original), quadCorners(geometry, natural.width, natural.height, region), width, height);
  }

  $effect(() => {
    // Everything a frame depends on: the sliders, the geometry, the region, Compare, the picture's box and the picture itself.
    void [light, geometry, region, original, picture.width, picture.height, glLive, image];
    if (!glLive || frameRequest) return;
    frameRequest = requestAnimationFrame(draw);
  });

  $effect(() => () => {
    if (frameRequest) cancelAnimationFrame(frameRequest);
  });

  // --- The fallback -------------------------------------------------------------------------------------------------
  const noPreview = $derived(natural !== null && glTried && !glLive);
  const css = $derived(natural ? cssFrameTransform(geometry, natural.width, natural.height, { width: picture.width, height: picture.height }, region) : null);
</script>

<div class="adjust-stage">
  {#if noPreview}
    <div class="ui-alert ui-alert-warning adjust-notice" role="status">
      <Icon name="alert" size={14} />
      <span>Live preview isn't available in this browser; saving still applies your changes.</span>
    </div>
  {/if}
  <div class="stage-fit" bind:this={fitEl} use:hold={{ onhold: () => onpicturehold?.(), ignore: '[data-crop-part]', enabled: onpicturehold !== undefined }}>
    {#if natural && frame}
      <div class="pic" class:ui-checkerboard={transparent} data-testid="adjust-picture" style="left: {picture.left}px; top: {picture.top}px; width: {picture.width}px; height: {picture.height}px">
        <canvas bind:this={canvas} class="pic-canvas" hidden={noPreview}></canvas>
        {#if noPreview && css}
          <img
            data-testid="adjust-fallback"
            {src}
            alt=""
            draggable="false"
            style="width: {natural.width}px; height: {natural.height}px; transform: {css.transform}; clip-path: {css.clip}"
          >
        {/if}
        {#if original}
          <span class="media-tag pic-tag" data-testid="adjust-original">Original</span>
        {/if}
      </div>
      <div class="pic-overlay" style="left: {picture.left}px; top: {picture.top}px; width: {picture.width}px; height: {picture.height}px">
        {@render overlay?.({ frame, scale })}
      </div>
    {:else}
      <div class="stage-loading"><Spinner size="lg" label="Opening the picture" /></div>
    {/if}
  </div>
</div>

<style>
  /* The stage: the picture is fitted to what is left; it never sits under the controls. */
  .adjust-stage { position: relative; display: flex; flex: 1; min-width: 0; min-height: 0; flex-direction: column; align-items: center; justify-content: center; gap: 12px; padding: var(--adjust-stage-pad, 32px); background: var(--color-bg-base); }
  .stage-fit { position: relative; flex: 1; align-self: stretch; min-height: 0; -webkit-touch-callout: none; }
  .stage-loading { position: absolute; inset: 0; display: grid; place-items: center; color: var(--color-text-muted); }
  .pic { position: absolute; overflow: hidden; border-radius: var(--radius-xs); }
  .pic-canvas { display: block; width: 100%; height: 100%; }
  .pic-canvas[hidden] { display: none; }
  .pic img { position: absolute; left: 0; top: 0; display: block; max-width: none; transform-origin: 0 0; user-select: none; }
  .pic-tag { position: absolute; left: 8px; top: 8px; z-index: 3; }
  .pic-overlay { position: absolute; pointer-events: none; }
  /* No WebGL2: the one-line notice sits above the picture, in flow, so the picture fits under it. */
  .adjust-notice { flex: none; max-width: 560px; }
</style>
