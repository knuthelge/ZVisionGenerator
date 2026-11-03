<script lang="ts" module>
  export type MascotMood =
    | 'idle'
    | 'thinking'
    | 'reading'
    | 'creating'
    | 'finishing'
    | 'cheerful'
    | 'waving'
    | 'curious'
    | 'sleeping'
    | 'paused'
    | 'sad'
    | 'surprised'
    | 'nodding';

  /** What the mascot holds or wears: a paintbrush for images, a clapperboard for video, headphones for audio. */
  export type MascotTool = 'brush' | 'clapper' | 'audio';

  /** Report whether the user asked the system to reduce motion. */
  export function prefersReducedMotion(): boolean {
    return window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
  }
</script>

<script lang="ts">
  import { untrack } from 'svelte';
  import { FINISHING_FRACTION } from '$lib/state/mascot';
  import { BODY_PATH, bubbleCells, confettiPieces, noteConfetti, notePixels, risingNotes, TILE_CELLS, TILE_STEP_COUNT, tileCells, tileStep, zGlyphs } from './mascotArt';

  interface Props {
    mood?: MascotMood;
    tool?: MascotTool;
    size?: number;
    /** Share (0 to 1) of the active job's steps done; the creating mood's canvas follows it, and holds its last step while it is absent. */
    progress?: number;
    /** Hidden from assistive technology, for a copy that only decorates (the logo). */
    decorative?: boolean;
    class?: string;
  }

  let {
    mood = 'idle',
    tool = 'brush',
    size = 120,
    progress,
    decorative = false,
    class: extraClass = ''
  }: Props = $props();

  const uid = $props.id();

  const labels: Record<MascotMood, string> = {
    idle: 'Z-Vision mascot',
    thinking: 'Z-Vision mascot is thinking',
    reading: 'Z-Vision mascot is reading your prompt',
    creating: 'Z-Vision mascot is creating an image',
    finishing: 'Z-Vision mascot is adding the finishing touches',
    cheerful: 'Z-Vision mascot is cheering',
    waving: 'Z-Vision mascot is waving hello',
    curious: 'Z-Vision mascot is watching you type',
    sleeping: 'Z-Vision mascot is sleeping',
    paused: 'Z-Vision mascot is waiting for you to resume',
    sad: 'Z-Vision mascot is sad',
    surprised: 'Z-Vision mascot is surprised',
    nodding: 'Z-Vision mascot nods: added to the queue'
  };

  interface Pose {
    transform: string;
    opacity: string;
  }

  const BLEND_MS = 280;
  /** Below this size (px) he is drawn in a simpler form: thicker eyes, no hands, tool or effects. */
  const SIMPLE_BELOW_PX = 48;
  // The viewBox and the point between the eyes, for mapping the pointer into SVG units.
  const VIEW = { x: -14, y: -18, width: 228, height: 218 };
  const EYE = { x: 101, y: 112 };
  // Furthest the iris looks (SVG units), reached once the pointer is this far away (CSS px).
  const LOOK_MAX = 5;
  const LOOK_RANGE_PX = 320;
  // Where the effect tile sits beside his head.
  const TILE = { x: 168, y: 2, size: 40 };
  const EYES = [
    { x: 79, side: -1 },
    { x: 123, side: 1 }
  ] as const;

  const confetti = confettiPieces();
  const zs = zGlyphs();
  const rising = risingNotes();
  const noteBits = noteConfetti();
  const veilCells = tileCells(6);
  const finishedCells = tileCells(null);

  let svg = $state<SVGSVGElement | null>(null);
  let fromPoses: Map<Element, Pose> | null = null;
  let blends: Animation[] = [];
  let blendRun = 0;
  let booping = $state(false);
  // The step the creating canvas shows: the job's latest real step, kept while the job reports none (between images or stages).
  let heldStep = $state(0);

  const simple = $derived(size < SIMPLE_BELOW_PX);
  const bubble = $derived(bubbleCells(tool === 'audio' ? 'note' : 'picture'));
  const tile = $derived(tileCells(mood === 'finishing' ? null : heldStep));
  const dotsLit = $derived(mood === 'finishing' ? TILE_STEP_COUNT : heldStep + 1);

  $effect.pre(() => {
    if (mood !== 'creating' && mood !== 'finishing') heldStep = 0;
    else if (progress !== undefined) heldStep = tileStep(progress, FINISHING_FRACTION);
  });

  /** Read the rendered pose of every styled part, including any running animation. */
  function readPoses(root: SVGSVGElement): Map<Element, Pose> {
    const poses = new Map<Element, Pose>();
    // The click bounce plays on regardless of mood, so it is left out of the blend; so are the effects' many small parts.
    root.querySelectorAll('[class]:not(.boop)').forEach((el) => {
      if (el.closest('.fx') && !el.classList.contains('fx')) return;
      const style = getComputedStyle(el);
      poses.set(el, { transform: style.transform, opacity: style.opacity });
    });
    return poses;
  }

  /** Blend each part from its old pose into the new mood, then let the new loops play. */
  function blendInto(root: SVGSVGElement, from: Map<Element, Pose>): void {
    const run = ++blendRun;
    for (const [el, to] of readPoses(root)) {
      const start = from.get(el);
      if (!start) continue;
      const first: Keyframe = {};
      const last: Keyframe = {};
      if (start.transform !== to.transform) { first.transform = start.transform; last.transform = to.transform; }
      if (start.opacity !== to.opacity) { first.opacity = start.opacity; last.opacity = to.opacity; }
      if (Object.keys(first).length > 0) {
        blends.push(el.animate([first, last], { duration: BLEND_MS, easing: 'ease-in-out' }));
      }
    }
    void Promise.allSettled(blends.map((blend) => blend.finished)).then(() => {
      if (run !== blendRun) return;
      blends = [];
      delete root.dataset.settling;
    });
  }

  // Snapshot the pose before the mood changes, and hold the new mood's loops at their first frame.
  $effect.pre(() => {
    void mood;
    untrack(() => {
      if (!svg || typeof svg.animate !== 'function') return;
      fromPoses = readPoses(svg);
      blends.forEach((blend) => blend.cancel());
      blends = [];
      svg.dataset.settling = '';
    });
  });

  $effect(() => {
    void mood;
    untrack(() => {
      const from = fromPoses;
      fromPoses = null;
      if (svg && from) blendInto(svg, from);
    });
  });

  /** Aim the irises at a pointer position, in SVG units relative to the point between the eyes. */
  function lookAt(root: SVGSVGElement, clientX: number, clientY: number): void {
    const rect = root.getBoundingClientRect();
    const scale = Math.min(rect.width / VIEW.width, rect.height / VIEW.height);
    const eyeX = rect.left + (rect.width - VIEW.width * scale) / 2 + (EYE.x - VIEW.x) * scale;
    const eyeY = rect.top + (rect.height - VIEW.height * scale) / 2 + (EYE.y - VIEW.y) * scale;
    const dx = clientX - eyeX;
    const dy = clientY - eyeY;
    const distance = Math.hypot(dx, dy);
    const reach = distance === 0 ? 0 : (LOOK_MAX * Math.min(1, distance / LOOK_RANGE_PX)) / distance;
    root.style.setProperty('--look-x', `${(dx * reach).toFixed(2)}px`);
    root.style.setProperty('--look-y', `${(dy * reach * 0.8).toFixed(2)}px`);
  }

  // While idle, the eyes follow the pointer around the page; the simple form has no eyes to move.
  $effect(() => {
    if (mood !== 'idle' || simple || !svg || prefersReducedMotion()) return;
    const root = svg;
    let frame = 0;
    let pointer = { x: 0, y: 0 };
    function onPointerMove(event: PointerEvent): void {
      pointer = { x: event.clientX, y: event.clientY };
      if (!frame) frame = requestAnimationFrame(() => { frame = 0; lookAt(root, pointer.x, pointer.y); });
    }
    window.addEventListener('pointermove', onPointerMove, { passive: true });
    return () => {
      window.removeEventListener('pointermove', onPointerMove);
      cancelAnimationFrame(frame);
      root.style.removeProperty('--look-x');
      root.style.removeProperty('--look-y');
    };
  });

  /** Bounce once when clicked; clicks during a bounce are ignored. */
  function boop(): void {
    if (booping || prefersReducedMotion()) return;
    booping = true;
  }

  function boopEnded(event: AnimationEvent): void {
    if (event.target === event.currentTarget) booping = false;
  }
</script>

<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
<!-- The bounce is a pointer-only flourish that carries no information. -->
<svg
  bind:this={svg}
  class="mascot {extraClass}"
  data-mood={mood}
  data-tool={tool}
  data-simple={simple ? '' : undefined}
  data-boop={booping ? '' : undefined}
  onclick={boop}
  width={size}
  height={size}
  viewBox="-14 -18 228 218"
  role={decorative ? undefined : 'img'}
  aria-label={decorative ? undefined : tool === 'audio' && mood === 'creating' ? 'Z-Vision mascot is making a take' : labels[mood]}
  aria-hidden={decorative ? 'true' : undefined}
>
  <defs>
    <radialGradient id="{uid}-body" cx=".34" cy=".3" r=".85">
      <stop offset="0" stop-color="#a5f3e6" />
      <stop offset=".35" stop-color="#4fe1cc" />
      <stop offset=".68" stop-color="#2dd4bf" />
      <stop offset="1" stop-color="#12998b" />
    </radialGradient>
    <linearGradient id="{uid}-rim" x1="0" y1="0" x2="1" y2="1">
      <stop offset=".55" stop-color="#d5fff8" stop-opacity="0" />
      <stop offset=".95" stop-color="#d5fff8" stop-opacity=".9" />
    </linearGradient>
    <linearGradient id="{uid}-beret" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#ffb8aa" />
      <stop offset=".55" stop-color="#fb8f7c" />
      <stop offset="1" stop-color="#e2705d" />
    </linearGradient>
    <clipPath id="{uid}-clip">
      <path d={BODY_PATH} />
    </clipPath>
  </defs>

  <ellipse class="shadow" cx="100" cy="188" rx="56" ry="7" fill="#000" opacity=".32" />

  <!-- Thinking: thought bubbles, and an idea that has half formed out of noise -->
  <g class="fx fx-thinking">
    <circle class="tdot d1" cx="146" cy="46" r="3.2" fill="#b0b5ba" />
    <circle class="tdot d2" cx="155" cy="33" r="4.4" fill="#b0b5ba" />
    <circle cx="176" cy="8" r="23" fill="#393c42" />
    <clipPath id="{uid}-bubble"><circle cx="176" cy="8" r="19.5" /></clipPath>
    <g clip-path="url(#{uid}-bubble)">
      <g transform="translate(157 -11) scale(6.5)" shape-rendering="crispEdges">
        {#each bubble as cell (cell.y * 8 + cell.x)}
          <rect x={cell.x} y={cell.y} width="1.05" height="1.05" fill={cell.fill} />
        {/each}
      </g>
    </g>
  </g>

  <!-- Sleeping: pixel Zs that rise and fade -->
  <g class="fx fx-sleeping">
    {#each zs as glyph, zi (zi)}
      <g class="z" style:--zd="{glyph.delay}s" fill="#d6d8dc">
        {#each glyph.pixels as pixel (pixel.x * 1000 + pixel.y)}
          <rect x={pixel.x} y={pixel.y} width={glyph.cell} height={glyph.cell} />
        {/each}
      </g>
    {/each}
  </g>

  <!-- Audio, creating: pixel notes that rise from behind his head -->
  <g class="fx fx-notes" shape-rendering="crispEdges">
    {#each rising as note, ni (ni)}
      <g class="note" style:--nd="{note.delay}s" style:--rx="{note.rest.x}px" style:--ry="{note.rest.y}px" fill={note.fill}>
        <g class="note-tick">
          {#each note.pixels as pixel, i (i)}
            <rect x={pixel.x} y={pixel.y} width={note.cell + 0.06} height={note.cell + 0.06} />
          {/each}
        </g>
      </g>
    {/each}
  </g>

  <!-- Audio, sad: a grey note sinks behind his shoulder -->
  <g class="fx fx-sadnote" shape-rendering="crispEdges">
    <g class="sad-note" fill="#8f939a">
      {#each notePixels('eighth', 148, 28, 3.8) as pixel, i (i)}
        <rect x={pixel.x} y={pixel.y} width="3.86" height="3.86" />
      {/each}
    </g>
  </g>

  <!-- Paused: pause badge -->
  <g class="fx fx-paused">
    <g class="pause-badge" fill="#b0b5ba">
      <rect x="170" y="4" width="8" height="26" rx="4" />
      <rect x="184" y="4" width="8" height="26" rx="4" />
    </g>
    <!-- Audio: the note the music stopped on -->
    <g class="held-note" fill="#b0b5ba" shape-rendering="crispEdges">
      {#each notePixels('eighth', 146, 4, 3.8) as pixel, i (i)}
        <rect x={pixel.x} y={pixel.y} width="3.86" height="3.86" />
      {/each}
    </g>
  </g>

  <!-- Nodding: check badge -->
  <g class="fx fx-nodding">
    <g class="check">
      <circle cx="178" cy="14" r="13" fill="#2dd4bf" />
      <path d="M171,14 L176,19 L185,9" stroke="#fff" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" fill="none" />
    </g>
  </g>

  <!-- Surprised: exclamation mark -->
  <g class="fx fx-surprised">
    <g class="exclaim">
      <rect x="164" y="2" width="9" height="28" rx="4.5" fill="#fbbf24" />
      <circle cx="168.5" cy="39" r="5" fill="#fbbf24" />
    </g>
  </g>

  <g class="boop" onanimationend={boopEnded}>
    <g class="root">
      <ellipse cx="74" cy="183" rx="17" ry="8" fill="#0f8f81" />
      <ellipse cx="126" cy="183" rx="17" ry="8" fill="#0f8f81" />

      <!-- Tool arm (behind body): a brush for images, a clapperboard for video -->
      <g class="tool detail">
        <g transform="translate(168 140)">
          <g class="tool-brush">
            <line x1="0" y1="8" x2="0" y2="-40" stroke="#a16207" stroke-width="6.5" stroke-linecap="round" />
            <line x1="-1.6" y1="4" x2="-1.6" y2="-38" stroke="#d9a35b" stroke-width="2" stroke-linecap="round" opacity=".7" />
            <rect x="-4.2" y="-49" width="8.4" height="10" rx="1.5" fill="#b4b6bc" />
            <path d="M-5,-48 Q-8,-62 0,-78 Q8,-62 5,-48 Z" fill="#fbbf24" />
            <path d="M-3.4,-64 Q-2,-73 0,-78 Q2,-73 3.4,-64 Z" fill="#fb8f7c" />
          </g>
          <g class="tool-clapper" transform="translate(-6 -2)">
            <rect x="0" y="-26" width="38" height="25" rx="3" fill="#2c2e33" stroke="#b4b6bc" stroke-width="1.6" />
            <path d="M6,-16 H32 M6,-9 H24" stroke="#8f939a" stroke-width="2" stroke-linecap="round" />
            <circle cx="30" cy="-9" r="2.6" fill="#fb8f7c" />
            <g class="clap-arm">
              <rect x="0" y="-36" width="38" height="8.5" rx="2" fill="#2c2e33" stroke="#b4b6bc" stroke-width="1.6" />
              <path d="M6,-35.2 L12,-35.2 L8,-28.8 L2,-28.8 Z M18,-35.2 L24,-35.2 L20,-28.8 L14,-28.8 Z M30,-35.2 L36,-35.2 L32,-28.8 L26,-28.8 Z" fill="#fbbf24" />
            </g>
          </g>
        </g>
      </g>

      <path
        class="body"
        d={BODY_PATH}
        fill="url(#{uid}-body)"
      />
      <g clip-path="url(#{uid}-clip)">
        <ellipse cx="100" cy="200" rx="90" ry="40" fill="#0b3b37" opacity=".28" />
        <path
          class="rim"
          d={BODY_PATH}
          fill="none"
          stroke="url(#{uid}-rim)"
          stroke-width="8"
        />
        <g class="detail">
          <ellipse cx="56" cy="104" rx="10" ry="18" transform="rotate(28 56 104)" fill="#fff" opacity=".28" />
          <circle cx="64" cy="86" r="3.4" fill="#fff" opacity=".8" />
        </g>
      </g>

      <!-- Audio: headphones in front of the body, behind the beret (a bolder form for small sizes) -->
      <g class="phones">
        <g class="phones-full">
          <path d="M31,100 C24,30 177,22 170,100" fill="none" stroke="#5b5f66" stroke-width="7.5" stroke-linecap="round" />
          <path d="M33,92 C30,40 172,32 168,92" fill="none" stroke="#c9ccd1" stroke-width="2" stroke-linecap="round" opacity=".75" />
          <g class="cup-l">
            <g class="cup">
              <ellipse cx="38" cy="113" rx="4.5" ry="13.5" fill="#2c2e33" />
              <ellipse cx="29" cy="113" rx="10.5" ry="17" fill="url(#{uid}-beret)" />
              <ellipse class="detail" cx="25" cy="105" rx="3" ry="6.5" fill="#fff" opacity=".42" />
              <circle class="detail" cx="28" cy="116" r="3.2" fill="#fbbf24" />
            </g>
          </g>
          <g class="cup-r-hinge">
            <g class="cup">
              <ellipse cx="162" cy="113" rx="4.5" ry="13.5" fill="#2c2e33" />
              <ellipse cx="171" cy="113" rx="10.5" ry="17" fill="url(#{uid}-beret)" />
              <ellipse class="detail" cx="167" cy="105" rx="3" ry="6.5" fill="#fff" opacity=".42" />
              <circle class="detail" cx="172" cy="116" r="3.2" fill="#fbbf24" />
            </g>
          </g>
        </g>
        <g class="phones-simple">
          <path d="M30,104 C22,24 178,16 170,104" fill="none" stroke="#b4b6bc" stroke-width="13" stroke-linecap="round" />
          <ellipse cx="27" cy="114" rx="15" ry="22" fill="#fb8f7c" />
          <ellipse cx="173" cy="114" rx="15" ry="22" fill="#fb8f7c" />
        </g>
      </g>

      <circle class="hand-r detail" cx="168" cy="142" r="10.5" fill="#14b8a6" />
      <!-- Reading: the prompt, held up like a recipe -->
      <g class="scroll detail">
        <rect x="6" y="102" width="34" height="32" rx="2" fill="#fef3c7" />
        <rect x="3" y="98" width="40" height="7" rx="3.5" fill="#fde68a" />
        <rect x="3" y="131" width="40" height="7" rx="3.5" fill="#fde68a" />
        <path d="M12,112 H34 M12,118 H32 M12,124 H27" stroke="#b45309" stroke-width="2" stroke-linecap="round" opacity=".6" />
      </g>
      <circle class="hand-l detail" cx="32" cy="146" r="10.5" fill="#14b8a6" />

      <!-- The slouchy beret, worn tilted so the paint flick shows -->
      <g class="beret">
        <g transform="rotate(-18 76 62)">
          <ellipse class="beret-full" cx="76" cy="62" rx="33" ry="11" fill="url(#{uid}-beret)" />
          <circle class="beret-full" cx="76" cy="50" r="5.5" fill="#fb8f7c" />
          <ellipse class="beret-full detail" cx="66" cy="57" rx="12" ry="2.6" fill="#fff" opacity=".4" />
          <ellipse class="beret-simple" cx="76" cy="62" rx="36" ry="13" fill="url(#{uid}-beret)" />
          <circle class="beret-simple" cx="76" cy="48" r="7" fill="#fb8f7c" />
        </g>
      </g>

      <!-- Eyes: open (with a bead-eye form for small sizes), happy, and sleeping -->
      {#each EYES as eye (eye.side)}
        <g class="eye" style:--ex="{eye.x}px">
          <g class="eye-full">
            <clipPath id="{uid}-eye{eye.side}"><ellipse cx={eye.x} cy="112" rx="15" ry="17.1" /></clipPath>
            <ellipse cx={eye.x} cy="112" rx="15" ry="17.1" fill="#fff" />
            <g clip-path="url(#{uid}-eye{eye.side})">
              <g class="iris" style:--ex="{eye.x}px">
                <circle cx={eye.x} cy="112" r="9.9" fill="#134e4a" />
                <circle cx={eye.x} cy="112" r="5.45" fill="#06201d" />
                <circle cx={eye.x + 3.76} cy="107.84" r="3.56" fill="#fff" />
                <circle cx={eye.x - 3.96} cy="116.16" r="1.49" fill="#fff" opacity=".8" />
              </g>
              <rect class="lid" x={eye.x - 16} y="93.9" width="32" height="36.2" fill="#2dd4bf" />
              <path
                class="lid-sad"
                d="M{eye.x - 17},92.9 L{eye.x + 17},92.9 L{eye.x + 17},{eye.side < 0 ? 102.6 : 111.1} L{eye.x - 17},{eye.side < 0 ? 111.1 : 102.6} Z"
                fill="#2dd4bf"
              />
            </g>
            <ellipse cx={eye.x} cy="112" rx="15" ry="17.1" fill="none" stroke="#0b3b37" stroke-width="2.4" />
          </g>
          <g class="eye-bead">
            <circle class="bead-white" cx={eye.x} cy="112" r="17.9" fill="#fff" />
            <ellipse class="bead" cx={eye.x} cy="112" rx="10.5" ry="11.8" fill="#0b3b37" />
          </g>
        </g>
        <path class="arc eye-happy" d="M{eye.x - 13.5},117.3 Q{eye.x},96.2 {eye.x + 13.5},117.3" stroke="#0b3b37" stroke-width="5.4" stroke-linecap="round" fill="none" />
        <path class="arc eye-closed" d="M{eye.x - 13.5},110.5 Q{eye.x},124.8 {eye.x + 13.5},110.5" stroke="#0b3b37" stroke-width="5.4" stroke-linecap="round" fill="none" />
        <path class="brow detail" style:--sg={-eye.side} style:--ex="{eye.x}px" d="M{eye.x - 9},88 H{eye.x + 9}" stroke="#0b3b37" stroke-width="4.5" stroke-linecap="round" />
      {/each}
      <!-- Sad tear -->
      <path class="tear detail" d="M132,124 Q137,134 132,138 Q127,134 132,124 Z" fill="#7dd3fc" />

      <ellipse class="blush detail" cx="62" cy="138" rx="10" ry="5" fill="#ff9cc6" />
      <ellipse class="blush detail" cx="140" cy="138" rx="10" ry="5" fill="#ff9cc6" />

      <!-- Mouths, one visible per mood -->
      <path class="mouth m-smile" d="M93,146 Q101,153.5 109,146" stroke="#0b3b37" stroke-width="3.4" stroke-linecap="round" fill="none" />
      <path class="mouth m-thinking" d="M93,149 Q97,145 101,148 T109,147" stroke="#0b3b37" stroke-width="3.4" stroke-linecap="round" fill="none" />
      <g class="mouth m-creating">
        <ellipse class="detail" cx="105.5" cy="150.2" rx="3" ry="3.6" fill="#f472b6" />
        <path d="M93,146 Q101,151 109,146" stroke="#0b3b37" stroke-width="3.4" stroke-linecap="round" fill="none" />
      </g>
      <path class="mouth m-sad" d="M93,152 Q101,143 109,152" stroke="#0b3b37" stroke-width="3.4" stroke-linecap="round" fill="none" />
      <ellipse class="mouth m-surprised" cx="101" cy="150" rx="4.5" ry="6" fill="#0b3b37" />
      <ellipse class="mouth m-sleeping" cx="101" cy="149" rx="3.6" ry="2.8" fill="#0b3b37" />
      <path class="mouth m-paused" d="M94,148 Q101,150 108,148" stroke="#0b3b37" stroke-width="3.4" stroke-linecap="round" fill="none" />
      <g class="mouth m-cheerful">
        <path d="M91,143 Q101,162 111,143 Z" fill="#0b3b37" stroke="#0b3b37" stroke-width="3" stroke-linejoin="round" />
        <ellipse class="detail" cx="101" cy="153.5" rx="4.5" ry="2.5" fill="#f472b6" />
      </g>
      <!-- Audio: humming mouth, opening on each beat -->
      <g class="mouth m-sing">
        <ellipse cx="101" cy="149" rx="5" ry="5.6" fill="#0b3b37" />
        <ellipse class="detail" cx="101" cy="152.2" rx="3" ry="1.8" fill="#f472b6" />
      </g>
    </g>
  </g>

  <!-- Creating: his canvas denoises one step at a time, following the job; Finishing: it snaps to sharp -->
  <g class="fx fx-tile">
    <rect x={TILE.x - 3} y={TILE.y - 3} width={TILE.size + 6} height={TILE.size + 6} rx="5" fill="#0b0c0d" />
    <clipPath id="{uid}-tile"><rect x={TILE.x} y={TILE.y} width={TILE.size} height={TILE.size} rx="3" /></clipPath>
    <g clip-path="url(#{uid}-tile)">
      <g transform="translate({TILE.x} {TILE.y}) scale({TILE.size / TILE_CELLS})" shape-rendering="crispEdges">
        {#each tile as cell (cell.y * TILE_CELLS + cell.x)}
          <rect x={cell.x} y={cell.y} width="1.05" height="1.05" fill={cell.fill} />
        {/each}
      </g>
      <g class="veil">
        <g transform="translate({TILE.x} {TILE.y}) scale({TILE.size / TILE_CELLS})" shape-rendering="crispEdges">
          {#each veilCells as cell (cell.y * TILE_CELLS + cell.x)}
            <rect x={cell.x} y={cell.y} width="1.05" height="1.05" fill={cell.fill} />
          {/each}
        </g>
      </g>
      <rect class="scan-line" x={TILE.x} y={TILE.y - 1} width={TILE.size} height="3" fill="#ccfbf1" />
      <rect class="flash" x={TILE.x} y={TILE.y} width={TILE.size} height={TILE.size} fill="#fff" />
    </g>
    <g class="tile-dots">
      {#each { length: TILE_STEP_COUNT } as _, i (i)}
        <rect x={TILE.x + i * 5 + 0.8} y={TILE.y + TILE.size + 5} width="3.4" height="4.4" rx="1" fill={i < dotsLit ? '#2dd4bf' : '#393c42'} />
      {/each}
    </g>
  </g>

  <!-- Finishing: the brackets and sparks that frame the snap, and for audio the note that locks in -->
  <g class="fx fx-snap">
    <g class="note-big" fill="#fbbf24" shape-rendering="crispEdges">
      {#each notePixels('beamed', 173.3, 9.4, 4.2) as pixel, i (i)}
        <rect x={pixel.x} y={pixel.y} width="4.26" height="4.26" />
      {/each}
    </g>
    <g class="brackets" stroke="#2dd4bf" stroke-width="3" fill="none" stroke-linecap="square">
      <path d="M164,-2 V4 M164,-2 H170" />
      <path d="M212,-2 V4 M212,-2 H206" />
      <path d="M212,46 V40 M212,46 H206" />
      <path d="M164,46 V40 M164,46 H170" />
    </g>
    <rect class="spark" style:--sx="-1" style:--sy="-1" x="162" y="-4" width="5" height="5" fill="#fbbf24" />
    <rect class="spark" style:--sx="1" style:--sy="-1" x="209" y="-4" width="5" height="5" fill="#fff" />
    <rect class="spark" style:--sx="1" style:--sy="1" x="209" y="43" width="5" height="5" fill="#fb8f7c" />
    <rect class="spark" style:--sx="-1" style:--sy="1" x="162" y="43" width="5" height="5" fill="#2dd4bf" />
  </g>

  <!-- Cheerful: pixel confetti -->
  <g class="fx fx-cheerful">
    {#each confetti as piece, i (i)}
      <g class="confetti{i % 2 === 1 ? ' sq-alt' : ''}" style:--d="{piece.delay}s" style:--sway="{piece.sway}px" style:--ry="{piece.restY}px">
        <rect x={piece.x} y="0" width={piece.size} height={piece.size} fill={piece.fill} />
        {#if piece.pair}
          <rect x={piece.x + piece.size} y="0" width={piece.size} height={piece.size} fill={piece.pair} />
        {/if}
      </g>
    {/each}
    <!-- Audio: half the confetti are pixel notes -->
    <g class="note-confetti" shape-rendering="crispEdges">
      {#each noteBits as piece, ni (ni)}
        <g class="note-c" style:--d="{piece.delay}s" style:--sway="{piece.sway}px" style:--ry="{piece.restY}px" fill={piece.fill}>
          {#each piece.pixels as pixel, i (i)}
            <rect x={pixel.x} y={pixel.y} width={piece.cell + 0.06} height={piece.cell + 0.06} />
          {/each}
        </g>
      {/each}
    </g>
  </g>
</svg>

<style>
  .mascot {
    display: block;
    overflow: visible;
  }

  /* Shared transform origins (SVG user units) */
  .root,
  .boop { transform-origin: 100px 186px; }
  .shadow { transform-origin: 100px 188px; }
  .eye { transform-origin: var(--ex) 112px; }
  .iris { transform-origin: var(--ex) 112px; }
  .lid { transform-origin: var(--ex) 93.9px; transform: scaleY(0); }
  .brow { transform-origin: var(--ex) 88px; }
  .tool { transform-origin: 168px 140px; transform: rotate(16deg); }
  .beret { transform-origin: 76px 62px; }
  .exclaim { transform-origin: 168px 24px; }
  .check { transform-origin: 178px 14px; }
  .clap-arm { transform-origin: 1px -28px; transform: rotate(-20deg); }
  .blush { opacity: 0.55; }

  /* While a mood change blends in (see blendInto), the new loops wait on their first frame. */
  .mascot:global([data-settling]) *:not(.boop) { animation-play-state: paused !important; }

  /* Clicks bounce the mascot, even inside a pointer-events: none dock. */
  .mascot { pointer-events: auto; }
  [data-boop] .boop { animation: boop 0.5s ease-out 1; }

  /* The simple form for small sizes: bead eyes, no hands, tool, brows or effects, a broader beret and rim. */
  .eye-bead,
  .beret-simple,
  .bead-white { display: none; }
  [data-simple] .eye-full,
  [data-simple] .beret-full,
  [data-simple] .detail,
  [data-simple] .fx { display: none; }
  [data-simple] .eye-bead,
  [data-simple] .beret-simple { display: inline; }
  [data-simple] .rim { stroke-width: 12px; }
  [data-simple] .arc { stroke-width: 9.4px; }
  [data-simple] .mouth,
  [data-simple] .mouth path { stroke-width: 6px; }
  [data-simple][data-mood='surprised'] .bead-white { display: inline; }
  [data-simple][data-mood='surprised'] .bead { transform: scale(0.65); transform-origin: var(--ex) 112px; }
  [data-simple][data-mood='creating'] .eye,
  [data-simple][data-mood='finishing'] .eye,
  [data-simple][data-mood='paused'] .eye { transform: scaleY(0.72); }
  [data-simple][data-mood='sad'] .eye { transform: scaleY(0.85); }

  /* Tool: a brush for images, a clapperboard for video, headphones for audio */
  .tool-brush,
  .tool-clapper { transition: opacity 250ms ease; }
  .tool-clapper,
  [data-tool='clapper'] .tool-brush,
  [data-tool='audio'] .tool-brush { opacity: 0; }
  [data-tool='clapper'] .tool-clapper { opacity: 1; }
  .phones { opacity: 0; transition: opacity 250ms ease; transform-origin: 100px 104px; }
  [data-tool='audio'] .phones { opacity: 1; }
  .phones-simple { display: none; }
  [data-simple] .phones-full { display: none; }
  [data-simple] .phones-simple { display: inline; }
  .cup { transform-box: fill-box; transform-origin: center; }
  .cup-r-hinge { transform-origin: 170px 98px; }

  /* Mood-dependent visibility */
  .mouth,
  .arc,
  .tear,
  .scroll,
  .brow,
  .lid-sad,
  .fx { opacity: 0; }

  [data-mood='idle'] .m-smile,
  [data-mood='thinking'] .m-thinking,
  [data-mood='thinking'] .fx-thinking,
  [data-mood='thinking'] .brow,
  [data-mood='reading'] .m-smile,
  [data-mood='reading'] .scroll,
  [data-mood='creating'] .m-creating,
  [data-mood='creating'] .fx-tile,
  [data-mood='creating'] .brow,
  [data-mood='finishing'] .m-creating,
  [data-mood='finishing'] .fx-tile,
  [data-mood='finishing'] .fx-snap,
  [data-mood='finishing'] .brow,
  [data-mood='nodding'] .m-smile,
  [data-mood='nodding'] .eye-happy,
  [data-mood='nodding'] .fx-nodding,
  [data-mood='cheerful'] .m-cheerful,
  [data-mood='cheerful'] .eye-happy,
  [data-mood='cheerful'] .fx-cheerful,
  [data-mood='waving'] .m-smile,
  [data-mood='curious'] .m-smile,
  [data-mood='curious'] .brow,
  [data-mood='sleeping'] .m-sleeping,
  [data-mood='sleeping'] .eye-closed,
  [data-mood='sleeping'] .fx-sleeping,
  [data-mood='paused'] .m-paused,
  [data-mood='paused'] .fx-paused,
  [data-mood='sad'] .m-sad,
  [data-mood='sad'] .lid-sad,
  [data-mood='sad'] .brow,
  [data-mood='surprised'] .m-surprised,
  [data-mood='surprised'] .brow,
  [data-mood='surprised'] .fx-surprised { opacity: 1; }

  [data-mood='cheerful'] .eye,
  [data-mood='nodding'] .eye,
  [data-mood='sleeping'] .eye { opacity: 0; }
  [data-mood='sleeping'] .blush,
  [data-mood='sad'] .blush { opacity: 0.3; }
  [data-mood='paused'] .blush { opacity: 0.4; }
  [data-mood='cheerful'] .blush,
  [data-mood='finishing'] .blush,
  [data-mood='nodding'] .blush { opacity: 0.8; }

  /* Idle: gentle float and blink; the eyes follow the pointer (set by the component) */
  [data-mood='idle'] .root { animation: float 3.2s ease-in-out infinite; }
  [data-mood='idle'] .shadow { animation: shadow-float 3.2s ease-in-out infinite; }
  [data-mood='idle'] .eye { animation: blink 4.5s infinite; }
  [data-mood='idle'] .iris {
    transform: translate(var(--look-x, 0px), var(--look-y, 0px));
    transition: transform 180ms ease-out;
  }

  /* Reading: lean toward the prompt scroll and read it line by line */
  [data-mood='reading'] .root { animation: peek 2.4s ease-in-out infinite; }
  [data-mood='reading'] .iris { animation: scan 1.6s ease-in-out infinite; }
  [data-mood='reading'] .hand-l { transform: translate(6px, -8px); }
  [data-mood='reading'] .tool { transform: rotate(30deg); }
  [data-mood='reading'] .beret { transform: rotate(-6deg); }

  /* Thinking: look up, sway, bubbles, brush tapping, one raised brow */
  [data-mood='thinking'] .root { animation: sway 3s ease-in-out infinite; }
  [data-mood='thinking'] .iris { animation: ponder 3s ease-in-out infinite; }
  [data-mood='thinking'] .eye { animation: blink 5s 1s infinite; }
  [data-mood='thinking'] .tool { animation: tap 1.6s ease-in-out infinite; }
  [data-mood='thinking'] .tdot { animation: bubble 1.8s ease-in-out infinite; }
  [data-mood='thinking'] .d2 { animation-delay: 0.3s; }
  [data-mood='thinking'] .brow { transform: translateY(calc((var(--sg) - 1) * 2.5px)); }

  /* Creating: focused squint, brush strokes; the canvas beside him is drawn by the component */
  [data-mood='creating'] .root { animation: lean 0.9s ease-in-out infinite; }
  [data-mood='creating'] .lid { transform: scaleY(0.36); }
  [data-mood='creating'] .iris { animation: follow 0.9s ease-in-out infinite; }
  [data-mood='creating'] .tool { animation: paint 0.9s ease-in-out infinite; }
  [data-mood='creating'] .brow,
  [data-mood='finishing'] .brow { transform: translateY(2px) rotate(calc(var(--sg) * 12deg)); }

  /* Finishing: the creating pose, faster; the canvas sweeps, flashes and snaps sharp */
  [data-mood='finishing'] .root { animation: lean 0.55s ease-in-out infinite; }
  [data-mood='finishing'] .lid { transform: scaleY(0.36); }
  [data-mood='finishing'] .iris { animation: follow 0.55s ease-in-out infinite; }
  [data-mood='finishing'] .tool { animation: paint 0.55s ease-in-out infinite; }
  [data-mood='creating'] .tile-dots { opacity: 1; }
  [data-mood='finishing'] .veil { animation: veil 1.1s linear infinite; }
  [data-mood='finishing'] .scan-line { animation: scan-line 1.1s linear infinite; }
  [data-mood='finishing'] .flash { animation: flash 1.1s linear infinite; }
  [data-mood='finishing'] .brackets { animation: brackets 1.1s ease-out infinite; transform-origin: 188px 22px; }
  [data-mood='finishing'] .spark { animation: spark 1.1s ease-out infinite; }
  .tile-dots,
  .veil,
  .scan-line,
  .flash,
  .brackets,
  .spark { opacity: 0; }

  /* Video: the clapperboard snaps shut with every stroke */
  [data-tool='clapper'][data-mood='creating'] .clap-arm { animation: clap 0.9s ease-in-out infinite; }
  [data-tool='clapper'][data-mood='finishing'] .clap-arm { animation: clap 0.55s ease-in-out infinite; }

  /* Nodding: two quick nods, a happy squint and a check badge */
  [data-mood='nodding'] .root { animation: nod 0.9s ease-in-out 1; }
  [data-mood='nodding'] .tool { transform: rotate(4deg); }
  [data-mood='nodding'] .check { animation: exclaim 0.6s ease-out 1 both; }

  /* Cheerful: squash-and-stretch hop, waving brush, pixel confetti */
  [data-mood='cheerful'] .root { animation: hop 0.8s ease-in-out infinite; }
  [data-mood='cheerful'] .shadow { animation: shadow-hop 0.8s ease-in-out infinite; }
  [data-mood='cheerful'] .beret { animation: beret 0.8s ease-in-out infinite; }
  [data-mood='cheerful'] .tool { animation: wave 0.4s ease-in-out infinite alternate; }
  .confetti { opacity: 0; transform-box: fill-box; transform-origin: center; }
  [data-mood='cheerful'] .confetti {
    animation: confetti-fall 1.6s linear infinite, confetti-turn 1.6s steps(8) infinite;
    animation-delay: var(--d), var(--d);
  }

  /* Waving: hop in place and wave the free hand */
  [data-mood='waving'] .root { animation: float 1.6s ease-in-out infinite; }
  [data-mood='waving'] .shadow { animation: shadow-float 1.6s ease-in-out infinite; }
  [data-mood='waving'] .hand-l { animation: hand-wave 0.5s ease-in-out infinite alternate; }
  [data-mood='waving'] .beret { transform: rotate(-6deg); }
  [data-mood='waving'] .eye { animation: blink 3s 0.4s infinite; }

  /* Curious: lean toward the prompt (to the left) and glance around it */
  [data-mood='curious'] .root { animation: peek 2.4s ease-in-out infinite; }
  [data-mood='curious'] .iris { animation: read 2.4s ease-in-out infinite; }
  [data-mood='curious'] .eye { transform: scale(1.06); }
  [data-mood='curious'] .beret { transform: rotate(-10deg); }
  [data-mood='curious'] .brow { transform: translateY(-4px); }

  /* Sleeping: slow breathing, lowered brush, pixel Zs that dissolve into noise */
  [data-mood='sleeping'] .root { animation: breathe 3.6s ease-in-out infinite; }
  [data-mood='sleeping'] .tool { transform: rotate(72deg); }
  [data-mood='sleeping'] .beret { transform: rotate(10deg) translate(4px, 4px); }
  .z { opacity: 0; }
  [data-mood='sleeping'] .z { animation: z-rise 3.6s linear infinite; animation-delay: var(--zd); }

  /* Paused: still and patient, heavy-lidded, brush resting, a pulsing pause badge */
  [data-mood='paused'] .root { animation: breathe 4.4s ease-in-out infinite; }
  [data-mood='paused'] .lid { transform: scaleY(0.5); }
  [data-mood='paused'] .tool { transform: rotate(40deg); }
  [data-mood='paused'] .pause-badge { animation: pulse 2.4s ease-in-out infinite; }

  /* Sad: droop, look down, worried brows, a single tear */
  [data-mood='sad'] .root { animation: droop 4s ease-in-out infinite; }
  [data-mood='sad'] .iris { transform: translate(0, 4.5px); }
  [data-mood='sad'] .tool { transform: rotate(62deg); }
  [data-mood='sad'] .beret { transform: rotate(12deg) translateY(4px); }
  [data-mood='sad'] .brow { transform: translateY(3px) rotate(calc(var(--sg) * -22deg)); }
  [data-mood='sad'] .tear { animation: tear 2.4s ease-in infinite; }

  /* Surprised: startled jump, wide eyes, tiny pupils, flying beret, tall brows */
  [data-mood='surprised'] .root { animation: startle 0.6s ease-out 1; }
  [data-mood='surprised'] .eye { transform: scale(1.08, 1.1); }
  [data-mood='surprised'] .iris { transform: scale(0.64); }
  [data-mood='surprised'] .beret { animation: beret-pop 0.6s ease-out 1 forwards; }
  [data-mood='surprised'] .exclaim { animation: exclaim 0.9s ease-out 1 both; }
  [data-mood='surprised'] .tool { transform: rotate(-40deg); }
  [data-mood='surprised'] .brow { transform: translateY(-9px); }

  /*
   * Audio: Blob wears headphones. The image tile never shows (the live preview is the picture), and every effect is
   * a transform or an opacity. The beat is 120 BPM, 0.5 s.
   */
  [data-tool='audio'] .fx-tile { display: none; }
  .m-sing { transform-origin: 101px 145px; }
  .note,
  .note-c,
  .note-big,
  .sad-note,
  .held-note { opacity: 0; }
  .note-tick { transform-box: fill-box; transform-origin: 50% 80%; }
  .note-c,
  .sad-note,
  .note-big { transform-box: fill-box; transform-origin: center; }
  .note-confetti { display: none; }
  [data-tool='audio'] .note-confetti { display: inline; }
  [data-tool='audio'] .confetti.sq-alt { display: none; }

  [data-tool='audio'][data-mood='creating'] .fx-notes,
  [data-tool='audio'][data-mood='sad'] .fx-sadnote,
  [data-tool='audio'][data-mood='paused'] .held-note,
  [data-tool='audio'][data-mood='creating'] .m-sing,
  [data-tool='audio'][data-mood='creating'] .eye-happy { opacity: 1; }
  [data-tool='audio'][data-mood='creating'] .eye,
  [data-tool='audio'][data-mood='creating'] .m-creating,
  [data-tool='audio'][data-mood='creating'] .brow { opacity: 0; }
  [data-tool='audio'][data-mood='creating'] .blush { opacity: 0.8; }

  /* Creating: he grooves, eyes closed in the music and humming, and a note rises every two beats */
  [data-tool='audio'][data-mood='creating'] .root { animation: groove 1s ease-in-out infinite; }
  [data-tool='audio'][data-mood='creating'] .beret { animation: beret-bob 0.5s ease-in-out infinite; }
  [data-tool='audio'][data-mood='creating'] .m-sing { animation: sing 0.5s ease-in-out infinite; }
  [data-tool='audio'][data-mood='creating'] .note { animation: note-rise 3s linear infinite; animation-delay: var(--nd); }
  [data-tool='audio'][data-mood='creating'] .note-tick { animation: note-tick 1s linear infinite; }
  [data-tool='audio'][data-mood='creating'] .cup,
  [data-tool='audio'][data-mood='finishing'] .cup { animation: cup-pump 0.5s ease-out infinite; }
  [data-tool='audio'][data-mood='creating'] .hand-r { animation: hand-tap 0.5s ease-in-out infinite; }

  /* Finishing: the focused face of the image, on the beat; the notes gather into one that locks into the brackets */
  [data-tool='audio'][data-mood='finishing'] .root { animation: beat-bob 0.5s ease-in-out infinite; }
  [data-tool='audio'][data-mood='finishing'] .note-big { animation: note-lock 1.1s ease-out infinite; }

  /* Cheerful: a side-step dance, both hands up, and half the confetti are notes */
  [data-tool='audio'][data-mood='cheerful'] .root { animation: boogie 1s ease-in-out infinite; }
  [data-tool='audio'][data-mood='cheerful'] .shadow { animation: shadow-boogie 1s ease-in-out infinite; }
  [data-tool='audio'][data-mood='cheerful'] .beret { animation: beret 0.5s ease-in-out infinite; }
  [data-tool='audio'][data-mood='cheerful'] .hand-l { animation: hand-wave 0.5s ease-in-out infinite alternate; }
  [data-tool='audio'][data-mood='cheerful'] .hand-r { animation: hand-pump 0.5s ease-in-out infinite alternate; }
  [data-tool='audio'][data-mood='cheerful'] .note-c {
    animation: confetti-fall 1.6s linear infinite, note-wobble 0.5s steps(1) infinite;
    animation-delay: var(--d), 0s;
  }

  /* Paused: the music stops mid-note and one cup lifts off to listen */
  [data-tool='audio'][data-mood='paused'] .held-note { animation: pulse 2.4s ease-in-out infinite; }
  [data-tool='audio'][data-mood='paused'] .cup-r-hinge { transform: rotate(-26deg); }

  /* Sad: a grey note sinks behind his shoulder and the headphones slip */
  [data-tool='audio'][data-mood='sad'] .sad-note { animation: note-sink 3.2s ease-in infinite; }
  [data-tool='audio'][data-mood='sad'] .phones { transform: rotate(5deg) translateY(3px); }

  /* The other moods keep their motion; only the headphones find a pose */
  [data-tool='audio'][data-mood='sleeping'] .phones { transform: rotate(-4deg) translateY(3px); }
  [data-tool='audio'][data-mood='surprised'] .phones { animation: phones-pop 0.6s ease-out 1 forwards; }

  @keyframes float { 50% { transform: translateY(-5px); } }
  @keyframes shadow-float { 50% { transform: scaleX(0.9); } }
  @keyframes blink {
    0%, 92%, 100% { transform: scaleY(1); }
    95% { transform: scaleY(0.08); }
  }

  @keyframes sway {
    0%, 100% { transform: rotate(-3deg); }
    50% { transform: rotate(3deg); }
  }
  @keyframes ponder {
    0%, 100% { transform: translate(5px, -5px); }
    45%, 55% { transform: translate(-3px, -5px); }
  }
  @keyframes tap {
    0%, 100% { transform: rotate(22deg); }
    50% { transform: rotate(30deg); }
  }
  @keyframes bubble {
    0%, 100% { opacity: 0.25; transform: translateY(0); }
    50% { opacity: 1; transform: translateY(-3px); }
  }

  @keyframes lean {
    0%, 100% { transform: rotate(1deg); }
    50% { transform: rotate(4deg) translateX(2px); }
  }
  @keyframes follow {
    0%, 100% { transform: translate(4px, -1px); }
    50% { transform: translate(6px, -3px); }
  }
  @keyframes paint {
    0%, 100% { transform: rotate(-24deg); }
    50% { transform: rotate(-4deg); }
  }

  @keyframes veil {
    0% { opacity: 1; translate: 0 0; }
    50% { opacity: 1; translate: 0 40px; }
    50.1%, 100% { opacity: 0; translate: 0 40px; }
  }
  @keyframes scan-line {
    0% { opacity: 1; translate: 0 0; }
    50% { opacity: 1; translate: 0 40px; }
    50.1%, 100% { opacity: 0; translate: 0 40px; }
  }
  @keyframes flash {
    0%, 50% { opacity: 0; }
    52% { opacity: 0.7; }
    68%, 100% { opacity: 0; }
  }
  @keyframes brackets {
    0%, 50% { opacity: 0; scale: 1.3; }
    62% { opacity: 1; scale: 1; }
    78% { opacity: 1; scale: 1; }
    100% { opacity: 0; scale: 1; }
  }
  @keyframes spark {
    0%, 50% { opacity: 0; translate: 0 0; }
    54% { opacity: 1; }
    100% { opacity: 0; translate: calc(var(--sx) * 14px) calc(var(--sy) * 14px); }
  }

  @keyframes hop {
    0%, 100% { transform: translateY(0) scale(1.06, 0.94); }
    40% { transform: translateY(-18px) scale(0.96, 1.05); }
    60% { transform: translateY(-18px) scale(0.98, 1.02); }
    85% { transform: translateY(0) scale(1.04, 0.96); }
  }
  @keyframes shadow-hop {
    0%, 100% { transform: scaleX(1); opacity: 0.32; }
    50% { transform: scaleX(0.7); opacity: 0.2; }
  }
  @keyframes beret {
    0%, 100% { transform: rotate(0deg); }
    40% { transform: translateY(-5px) rotate(-8deg); }
    60% { transform: translateY(-5px) rotate(6deg); }
  }
  @keyframes wave {
    from { transform: rotate(-36deg); }
    to { transform: rotate(-12deg); }
  }
  @keyframes confetti-fall {
    0% { translate: 0 -26px; opacity: 0; }
    8% { opacity: 1; }
    75% { opacity: 1; }
    100% { translate: var(--sway) 144px; opacity: 0; }
  }
  @keyframes confetti-turn {
    from { rotate: 0deg; }
    to { rotate: 720deg; }
  }

  @keyframes hand-wave {
    from { transform: translate(-1px, -42px); }
    to { transform: translate(-9px, -50px); }
  }
  @keyframes peek {
    0%, 100% { transform: rotate(-5deg) translateX(-3px); }
    50% { transform: rotate(-7deg) translateX(-5px); }
  }
  @keyframes read {
    0%, 100% { transform: translate(-5px, 2px); }
    40%, 60% { transform: translate(-2px, 2px); }
  }
  @keyframes breathe {
    0%, 100% { transform: scale(1, 1); }
    50% { transform: scale(1.02, 1.04); }
  }
  @keyframes z-rise {
    0% { translate: 0 6px; opacity: 0; }
    15% { opacity: 1; }
    80% { opacity: 1; }
    100% { translate: 10px -16px; opacity: 0; }
  }
  @keyframes pulse { 50% { opacity: 0.45; } }
  @keyframes scan {
    0%, 100% { transform: translate(-5px, 2px); }
    30% { transform: translate(-2px, 3px); }
    45% { transform: translate(-5px, 4px); }
    75% { transform: translate(-2px, 5px); }
    90% { transform: translate(-5px, 2px); }
  }
  @keyframes clap {
    0%, 55%, 100% { transform: rotate(-20deg); }
    30% { transform: rotate(0deg); }
  }
  @keyframes nod {
    0%, 45%, 100% { transform: translateY(0) rotate(0deg); }
    20%, 70% { transform: translateY(4px) rotate(3deg) scale(1.02, 0.97); }
  }
  @keyframes boop {
    0%, 100% { transform: scale(1, 1); }
    30% { transform: scale(1.12, 0.86); }
    60% { transform: translateY(-10px) scale(0.94, 1.08); }
  }
  @keyframes droop {
    0%, 100% { transform: scale(1.03, 0.96); }
    50% { transform: scale(1.04, 0.94); }
  }
  @keyframes tear {
    0% { transform: translateY(0); opacity: 0; }
    20% { opacity: 1; }
    100% { transform: translateY(34px); opacity: 0; }
  }
  @keyframes startle {
    0% { transform: translateY(0) scale(1.04, 0.94); }
    35% { transform: translateY(-16px) scale(0.96, 1.06); }
    70% { transform: translateY(0) scale(1.03, 0.97); }
    100% { transform: translateY(0) scale(1, 1); }
  }
  @keyframes beret-pop {
    0% { transform: translateY(0); }
    40% { transform: translateY(-16px) rotate(-14deg); }
    100% { transform: translateY(-3px) rotate(-6deg); }
  }
  @keyframes exclaim {
    0% { transform: scale(0.2); opacity: 0; }
    50% { transform: scale(1.15); opacity: 1; }
    100% { transform: scale(1); opacity: 1; }
  }

  @keyframes groove {
    0%, 100% { transform: translateY(0) rotate(-2deg); }
    14% { transform: translateY(3.5px) rotate(-3.5deg) scale(1.03, 0.96); }
    50% { transform: translateY(0) rotate(2deg); }
    64% { transform: translateY(3.5px) rotate(3.5deg) scale(1.03, 0.96); }
  }
  @keyframes beat-bob {
    0%, 100% { transform: translateY(0) rotate(1deg); }
    18% { transform: translateY(3px) rotate(2deg) scale(1.02, 0.97); }
  }
  @keyframes beret-bob {
    0%, 100% { transform: translateY(0); }
    30% { transform: translateY(-2.5px) rotate(-3deg); }
  }
  @keyframes sing {
    0%, 100% { transform: scaleY(0.45); }
    35% { transform: scaleY(1); }
  }
  @keyframes cup-pump {
    0%, 100% { scale: 1; }
    12% { scale: 1.08; }
  }
  @keyframes hand-tap {
    0%, 100% { transform: translateY(0); }
    15% { transform: translateY(-5px); }
    35% { transform: translateY(1px); }
  }
  @keyframes hand-pump {
    from { transform: translate(-2px, -40px); }
    to { transform: translate(5px, -48px); }
  }
  @keyframes note-rise {
    0% { translate: 0 8px; opacity: 0; }
    10% { opacity: 1; }
    40% { translate: -5px -22px; }
    75% { opacity: 1; }
    100% { translate: 16px -64px; opacity: 0; }
  }
  @keyframes note-tick {
    0%, 49.9% { rotate: -12deg; }
    50%, 100% { rotate: 12deg; }
  }
  @keyframes note-wobble {
    0% { rotate: -14deg; }
    50% { rotate: 14deg; }
  }
  @keyframes note-lock {
    0% { opacity: 0; scale: 0.55; }
    44% { opacity: 0.45; scale: 0.85; }
    52% { opacity: 1; scale: 1.14; }
    62%, 86% { opacity: 1; scale: 1; }
    100% { opacity: 0; scale: 1; }
  }
  @keyframes note-sink {
    0% { translate: 0 -6px; rotate: 0deg; opacity: 0; }
    15% { opacity: 0.9; }
    70% { opacity: 0.8; }
    100% { translate: 6px 34px; rotate: 28deg; opacity: 0; }
  }
  @keyframes boogie {
    0%, 100% { transform: translateY(0) rotate(-5deg) scale(1.05, 0.95); }
    25% { transform: translateY(-14px) rotate(0deg) scale(0.97, 1.04); }
    50% { transform: translateY(0) rotate(5deg) scale(1.05, 0.95); }
    75% { transform: translateY(-14px) rotate(0deg) scale(0.97, 1.04); }
  }
  @keyframes shadow-boogie {
    0%, 50%, 100% { transform: scaleX(1); opacity: 0.32; }
    25%, 75% { transform: scaleX(0.74); opacity: 0.2; }
  }
  @keyframes phones-pop {
    0% { transform: translateY(0); }
    40% { transform: translateY(-10px) rotate(4deg); }
    100% { transform: translateY(-2px) rotate(2deg); }
  }

  /* Reduced motion: no loops, and each effect holds a still frame. */
  @media (prefers-reduced-motion: reduce) {
    .mascot * { animation: none !important; }
    [data-mood='thinking'] .iris { transform: translate(5px, -5px); }
    [data-mood='thinking'] .tool { transform: rotate(26deg); }
    [data-mood='thinking'] .tdot { opacity: 1; }
    [data-mood='creating'] .iris { transform: translate(5px, -2px); }
    [data-mood='creating'] .tool,
    [data-mood='finishing'] .tool { transform: rotate(-14deg); }
    [data-mood='finishing'] .iris { transform: translate(5px, -2px); }
    [data-mood='finishing'] .brackets { opacity: 1; }
    [data-mood='cheerful'] .confetti { opacity: 1; translate: 0 var(--ry); }
    [data-mood='waving'] .hand-l { transform: translate(-1px, -42px); }
    [data-mood='curious'] .iris { transform: translate(-5px, 2px); }
    [data-mood='curious'] .root { transform: rotate(-5deg); }
    [data-mood='reading'] .iris { transform: translate(-4px, 3px); }
    [data-mood='reading'] .root { transform: rotate(-5deg); }
    [data-mood='sleeping'] .z { opacity: 1; }
    [data-mood='sad'] .tear { opacity: 1; }
    [data-tool='audio'][data-mood='creating'] .note { opacity: 1; translate: var(--rx) var(--ry); }
    [data-tool='audio'][data-mood='creating'] .root { transform: rotate(-2deg); }
    [data-tool='audio'][data-mood='creating'] .m-sing { transform: scaleY(0.8); }
    [data-tool='audio'][data-mood='finishing'] .note-big { opacity: 1; }
    [data-tool='audio'][data-mood='cheerful'] .note-c { opacity: 1; translate: 0 var(--ry); }
    [data-tool='audio'][data-mood='cheerful'] .hand-l { transform: translate(-1px, -42px); }
    [data-tool='audio'][data-mood='cheerful'] .hand-r { transform: translate(-2px, -40px); }
    [data-tool='audio'][data-mood='paused'] .held-note { opacity: 0.6; }
    [data-tool='audio'][data-mood='sad'] .sad-note { opacity: 0.8; translate: 3px 12px; rotate: 14deg; }
  }
</style>
