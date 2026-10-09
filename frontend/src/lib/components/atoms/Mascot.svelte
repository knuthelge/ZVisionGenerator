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

  /** What the mascot holds: a paintbrush for images, a clapperboard for video. */
  export type MascotTool = 'brush' | 'clapper';

  /** Report whether the user asked the system to reduce motion. */
  export function prefersReducedMotion(): boolean {
    return window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
  }
</script>

<script lang="ts">
  import { untrack } from 'svelte';

  interface Props {
    mood?: MascotMood;
    tool?: MascotTool;
    size?: number | string;
    class?: string;
  }

  let {
    mood = 'idle',
    tool = 'brush',
    size = 120,
    class: extraClass = ''
  }: Props = $props();

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
  // The viewBox and the eye centre, for mapping the pointer into SVG units.
  const VIEW = { x: -20, y: -20, width: 240, height: 225 };
  const EYE = { x: 100, y: 100 };
  // Furthest the iris looks (SVG units), reached once the pointer is this far away (CSS px).
  const LOOK_MAX = 7;
  const LOOK_RANGE_PX = 320;

  let svg = $state<SVGSVGElement | null>(null);
  let fromPoses: Map<Element, Pose> | null = null;
  let blends: Animation[] = [];
  let blendRun = 0;
  let booping = $state(false);

  /** Read the rendered pose of every styled part, including any running animation. */
  function readPoses(root: SVGSVGElement): Map<Element, Pose> {
    const poses = new Map<Element, Pose>();
    // The click bounce plays on regardless of mood, so it is left out of the blend.
    root.querySelectorAll('[class]:not(.boop)').forEach((el) => {
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

  /** Aim the iris at a pointer position, in SVG units relative to the eye. */
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

  // While idle, the eye follows the pointer around the page.
  $effect(() => {
    if (mood !== 'idle' || !svg || prefersReducedMotion()) return;
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
  data-boop={booping ? '' : undefined}
  onclick={boop}
  width={size}
  height={size}
  viewBox="-20 -20 240 225"
  role="img"
  aria-label={labels[mood]}
>
  <ellipse class="shadow" cx="100" cy="190" rx="50" ry="6" fill="#000" opacity=".35" />

  <!-- Thinking: thought bubbles -->
  <g class="fx fx-thinking">
    <circle class="dot d1" cx="150" cy="32" r="4.5" fill="#b0b5ba" />
    <circle class="dot d2" cx="164" cy="16" r="6" fill="#b0b5ba" />
    <circle class="dot d3" cx="183" cy="-2" r="8" fill="#b0b5ba" />
  </g>

  <!-- Sleeping: drifting Zs -->
  <g class="fx fx-sleeping" font-family="system-ui, sans-serif" font-weight="700" fill="#b0b5ba">
    <text class="z z1" x="146" y="40" font-size="16">z</text>
    <text class="z z2" x="160" y="22" font-size="22">z</text>
    <text class="z z3" x="178" y="0" font-size="28">Z</text>
  </g>

  <!-- Paused: pause badge -->
  <g class="fx fx-paused">
    <g class="pause-badge" fill="#b0b5ba">
      <rect x="158" y="2" width="8" height="26" rx="4" />
      <rect x="172" y="2" width="8" height="26" rx="4" />
    </g>
  </g>

  <!-- Nodding: check badge -->
  <g class="fx fx-nodding">
    <g class="check">
      <circle cx="168" cy="10" r="13" fill="#2dd4bf" />
      <path d="M161,10 L166,15 L175,5" stroke="#fff" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" fill="none" />
    </g>
  </g>

  <!-- Surprised: exclamation mark -->
  <g class="fx fx-surprised">
    <g class="exclaim">
      <rect x="160" y="-6" width="9" height="30" rx="4.5" fill="#fbbf24" />
      <circle cx="164.5" cy="34" r="5" fill="#fbbf24" />
    </g>
  </g>

  <g class="boop" onanimationend={boopEnded}>
    <g class="root">
      <!-- Tool arm (behind body): a brush for images, a clapperboard for video -->
      <g class="brush">
        <line x1="168" y1="126" x2="188" y2="84" stroke="#a16207" stroke-width="6" stroke-linecap="round" />
        <path class="tool-brush" d="M184,86 Q186,68 196,62 Q198,78 192,90 Z" fill="#fbbf24" />
        <g class="tool-clapper">
          <g transform="rotate(-20 190 78)">
            <rect x="176" y="72" width="28" height="19" rx="2.5" fill="#3f4650" stroke="#b0b5ba" stroke-width="1.5" />
            <g class="clap-arm">
              <rect x="176" y="63" width="28" height="7" rx="2" fill="#3f4650" stroke="#b0b5ba" stroke-width="1.5" />
              <path d="M181,63.8 L186,63.8 L182,69.2 L177,69.2 Z M190,63.8 L195,63.8 L191,69.2 L186,69.2 Z M199,63.8 L203,63.8 L200,69.2 L195,69.2 Z" fill="#fbbf24" />
            </g>
          </g>
        </g>
      </g>

      <path
        class="body"
        d="M100,42 C150,42 172,90 170,130 C168,168 140,182 100,182 C60,182 32,168 30,130 C28,90 50,42 100,42 Z"
        fill="#2dd4bf"
      />
      <ellipse cx="100" cy="150" rx="42" ry="24" fill="#99f6e4" opacity=".45" />
      <circle class="hand-r" cx="168" cy="128" r="10" fill="#14b8a6" />
      <!-- Reading: the prompt, held up like a recipe -->
      <g class="scroll">
        <rect x="8" y="88" width="34" height="32" rx="2" fill="#fef3c7" />
        <rect x="5" y="84" width="40" height="7" rx="3.5" fill="#fde68a" />
        <rect x="5" y="117" width="40" height="7" rx="3.5" fill="#fde68a" />
        <path d="M14,98 H36 M14,104 H34 M14,110 H29" stroke="#b45309" stroke-width="2" stroke-linecap="round" opacity=".6" />
      </g>
      <circle class="hand-l" cx="32" cy="132" r="10" fill="#14b8a6" />

      <g class="beret">
        <ellipse cx="100" cy="46" rx="36" ry="10" fill="#fb8f7c" />
        <circle cx="100" cy="35" r="5" fill="#fb8f7c" />
      </g>

      <!-- Open eye -->
      <g class="eye">
        <path d="M60,100 Q100,66 140,100 Q100,134 60,100 Z" fill="#fff" stroke="#134e4a" stroke-width="3" />
        <g class="iris">
          <circle cx="100" cy="100" r="17" fill="#134e4a" />
          <circle cx="107" cy="93" r="6" fill="#fff" />
        </g>
      </g>
      <!-- Sleeping closed eye -->
      <path class="eye-closed" d="M70,100 Q100,122 130,100" stroke="#134e4a" stroke-width="7" stroke-linecap="round" fill="none" />
      <!-- Sad tear -->
      <path class="tear" d="M136,108 Q141,118 136,122 Q131,118 136,108 Z" fill="#60a5fa" />
      <!-- Happy closed eye -->
      <path class="eye-happy" d="M70,106 Q100,74 130,106" stroke="#134e4a" stroke-width="7" stroke-linecap="round" fill="none" />

      <ellipse class="blush" cx="62" cy="126" rx="9" ry="4.5" fill="#f9a8d4" />
      <ellipse class="blush" cx="138" cy="126" rx="9" ry="4.5" fill="#f9a8d4" />

      <!-- Mouths, one visible per mood -->
      <path class="mouth m-idle" d="M89,130 Q100,140 111,130" stroke="#134e4a" stroke-width="3.5" stroke-linecap="round" fill="none" />
      <path class="mouth m-thinking" d="M91,135 Q96,131 101,134 T111,132" stroke="#134e4a" stroke-width="3.5" stroke-linecap="round" fill="none" />
      <g class="mouth m-creating">
        <ellipse cx="105" cy="136" rx="4" ry="5" fill="#f472b6" />
        <path d="M90,131 Q100,137 110,131" stroke="#134e4a" stroke-width="3.5" stroke-linecap="round" fill="none" />
      </g>
      <path class="mouth m-sad" d="M88,138 Q100,128 112,138" stroke="#134e4a" stroke-width="3.5" stroke-linecap="round" fill="none" />
      <ellipse class="mouth m-surprised" cx="100" cy="136" rx="7" ry="9" fill="#134e4a" />
      <ellipse class="mouth m-sleeping" cx="100" cy="135" rx="4" ry="3" fill="#134e4a" />
      <path class="mouth m-paused" d="M92,133 Q100,136 108,133" stroke="#134e4a" stroke-width="3.5" stroke-linecap="round" fill="none" />
      <g class="mouth m-cheerful">
        <path d="M84,126 Q100,152 116,126 Z" fill="#134e4a" stroke="#134e4a" stroke-width="3" stroke-linejoin="round" />
        <ellipse cx="100" cy="140" rx="7" ry="4" fill="#f472b6" />
      </g>
    </g>
  </g>

  <!-- Creating: sparkles and paint specks around the brush tip -->
  <g class="fx fx-creating">
    <g transform="translate(204 58)"><path class="spark s1" d="M0,-9 L2,-2 L9,0 L2,2 L0,9 L-2,2 L-9,0 L-2,-2Z" fill="#fbbf24" /></g>
    <g transform="translate(176 40)"><path class="spark s2" d="M0,-7 L1.6,-1.6 L7,0 L1.6,1.6 L0,7 L-1.6,1.6 L-7,0 L-1.6,-1.6Z" fill="#fff" /></g>
    <g transform="translate(210 96)"><path class="spark s3" d="M0,-6 L1.4,-1.4 L6,0 L1.4,1.4 L0,6 L-1.4,1.4 L-6,0 L-1.4,-1.4Z" fill="#2dd4bf" /></g>
    <circle class="speck p1" cx="194" cy="70" r="3.5" fill="#fb8f7c" />
    <circle class="speck p2" cx="190" cy="76" r="3" fill="#60a5fa" />
    <circle class="speck p3" cx="198" cy="74" r="3" fill="#fbbf24" />
  </g>

  <!-- Finishing: extra sparkles for the last strokes -->
  <g class="fx fx-finishing">
    <g transform="translate(186 26)"><path class="spark s4" d="M0,-8 L1.8,-1.8 L8,0 L1.8,1.8 L0,8 L-1.8,1.8 L-8,0 L-1.8,-1.8Z" fill="#fbbf24" /></g>
    <g transform="translate(214 76)"><path class="spark s5" d="M0,-6 L1.4,-1.4 L6,0 L1.4,1.4 L0,6 L-1.4,1.4 L-6,0 L-1.4,-1.4Z" fill="#fb8f7c" /></g>
  </g>

  <!-- Cheerful: confetti -->
  <g class="fx fx-cheerful">
    <rect class="confetti c1" x="20" y="-10" width="7" height="11" rx="1.5" fill="#fbbf24" />
    <rect class="confetti c2" x="52" y="-18" width="6" height="10" rx="1.5" fill="#fb8f7c" />
    <rect class="confetti c3" x="84" y="-6" width="7" height="11" rx="1.5" fill="#60a5fa" />
    <rect class="confetti c4" x="118" y="-16" width="6" height="10" rx="1.5" fill="#2dd4bf" />
    <rect class="confetti c5" x="150" y="-8" width="7" height="11" rx="1.5" fill="#f9a8d4" />
    <rect class="confetti c6" x="178" y="-14" width="6" height="10" rx="1.5" fill="#fbbf24" />
    <rect class="confetti c7" x="0" y="-4" width="6" height="10" rx="1.5" fill="#2dd4bf" />
    <rect class="confetti c8" x="196" y="-2" width="7" height="11" rx="1.5" fill="#fb8f7c" />
  </g>
</svg>

<style>
  .mascot {
    display: block;
    overflow: visible;
  }

  /* Shared transform origins (SVG user units) */
  .root { transform-origin: 100px 182px; }
  .shadow { transform-origin: 100px 190px; }
  .eye { transform-origin: 100px 100px; }
  .brush { transform-origin: 162px 132px; }
  .beret { transform-origin: 100px 50px; }
  .iris { transform-origin: 100px 100px; }
  .exclaim { transform-origin: 164px 30px; }
  .check { transform-origin: 168px 10px; }
  .clap-arm { transform-origin: 177px 70px; }
  .boop { transform-origin: 100px 182px; }
  .blush { opacity: 0.7; }

  /* While a mood change blends in (see blendInto), the new loops wait on their first frame. */
  .mascot:global([data-settling]) *:not(.boop) { animation-play-state: paused !important; }

  /* Clicks bounce the mascot, even inside a pointer-events: none dock. */
  .mascot { pointer-events: auto; }
  [data-boop] .boop { animation: boop 0.5s ease-out 1; }

  /* Tool: a brush for images, a clapperboard for video */
  .tool-brush,
  .tool-clapper { transition: opacity 250ms ease; }
  .tool-clapper,
  [data-tool='clapper'] .tool-brush { opacity: 0; }
  [data-tool='clapper'] .tool-clapper { opacity: 1; }
  /* Paint specks animate their opacity, so they are hidden rather than faded. */
  [data-tool='clapper'] .speck { visibility: hidden; }

  /* Mood-dependent visibility */
  .mouth,
  .eye-happy,
  .eye-closed,
  .tear,
  .scroll,
  .fx { opacity: 0; }

  [data-mood='idle'] .m-idle,
  [data-mood='thinking'] .m-thinking,
  [data-mood='thinking'] .fx-thinking,
  [data-mood='reading'] .m-idle,
  [data-mood='reading'] .scroll,
  [data-mood='creating'] .m-creating,
  [data-mood='creating'] .fx-creating,
  [data-mood='finishing'] .m-creating,
  [data-mood='finishing'] .fx-creating,
  [data-mood='finishing'] .fx-finishing,
  [data-mood='nodding'] .m-idle,
  [data-mood='nodding'] .eye-happy,
  [data-mood='nodding'] .fx-nodding,
  [data-mood='cheerful'] .m-cheerful,
  [data-mood='cheerful'] .eye-happy,
  [data-mood='cheerful'] .fx-cheerful,
  [data-mood='waving'] .m-idle,
  [data-mood='curious'] .m-idle,
  [data-mood='sleeping'] .m-sleeping,
  [data-mood='sleeping'] .eye-closed,
  [data-mood='sleeping'] .fx-sleeping,
  [data-mood='paused'] .m-paused,
  [data-mood='paused'] .fx-paused,
  [data-mood='sad'] .m-sad,
  [data-mood='surprised'] .m-surprised,
  [data-mood='surprised'] .fx-surprised { opacity: 1; }

  [data-mood='cheerful'] .eye,
  [data-mood='nodding'] .eye,
  [data-mood='sleeping'] .eye { opacity: 0; }
  [data-mood='sleeping'] .blush,
  [data-mood='sad'] .blush { opacity: 0.3; }
  [data-mood='paused'] .blush { opacity: 0.5; }
  [data-mood='cheerful'] .blush,
  [data-mood='finishing'] .blush,
  [data-mood='nodding'] .blush { opacity: 1; }

  /* Idle: gentle float and blink */
  [data-mood='idle'] .root { animation: float 3.2s ease-in-out infinite; }
  [data-mood='idle'] .shadow { animation: shadow-float 3.2s ease-in-out infinite; }
  [data-mood='idle'] .eye { animation: blink 4.5s infinite; }
  /* Idle: the iris follows the pointer (set by the component) */
  [data-mood='idle'] .iris {
    transform: translate(var(--look-x, 0px), var(--look-y, 0px));
    transition: transform 180ms ease-out;
  }

  /* Reading: lean toward the prompt scroll and read it line by line */
  [data-mood='reading'] .root { animation: peek 2.4s ease-in-out infinite; }
  [data-mood='reading'] .iris { animation: scan 1.6s ease-in-out infinite; }
  [data-mood='reading'] .hand-l { transform: translate(6px, -8px); }
  [data-mood='reading'] .brush { transform: rotate(30deg); }
  [data-mood='reading'] .beret { transform: rotate(-6deg); }

  /* Thinking: look up, sway, bubbles, brush tapping */
  [data-mood='thinking'] .root { animation: sway 3s ease-in-out infinite; }
  [data-mood='thinking'] .iris { animation: ponder 3s ease-in-out infinite; }
  [data-mood='thinking'] .eye { animation: blink 5s 1s infinite; }
  [data-mood='thinking'] .brush { animation: tap 1.6s ease-in-out infinite; }
  [data-mood='thinking'] .dot { animation: bubble 1.8s ease-in-out infinite; }
  [data-mood='thinking'] .d2 { animation-delay: 0.3s; }
  [data-mood='thinking'] .d3 { animation-delay: 0.6s; }

  /* Creating: focused squint, brush strokes, sparkles */
  [data-mood='creating'] .root { animation: lean 0.9s ease-in-out infinite; }
  [data-mood='creating'] .eye { transform: scaleY(0.72); }
  [data-mood='creating'] .iris { animation: follow 0.9s ease-in-out infinite; }
  [data-mood='creating'] .brush { animation: paint 0.9s ease-in-out infinite; }
  [data-mood='creating'] .spark { animation: twinkle 1.2s ease-in-out infinite; }
  [data-mood='creating'] .s2 { animation-delay: 0.4s; }
  [data-mood='creating'] .s3 { animation-delay: 0.8s; }
  [data-mood='creating'] .speck { animation: speck 1.2s ease-out infinite; }
  [data-mood='creating'] .p2 { animation-delay: 0.4s; }
  [data-mood='creating'] .p3 { animation-delay: 0.8s; }

  /* Finishing: the creating pose, faster, with extra sparkles */
  [data-mood='finishing'] .root { animation: lean 0.55s ease-in-out infinite; }
  [data-mood='finishing'] .eye { transform: scaleY(0.72); }
  [data-mood='finishing'] .iris { animation: follow 0.55s ease-in-out infinite; }
  [data-mood='finishing'] .brush { animation: paint 0.55s ease-in-out infinite; }
  [data-mood='finishing'] .spark { animation: twinkle 0.7s ease-in-out infinite; }
  [data-mood='finishing'] .s2 { animation-delay: 0.23s; }
  [data-mood='finishing'] .s3 { animation-delay: 0.46s; }
  [data-mood='finishing'] .s4 { animation-delay: 0.12s; }
  [data-mood='finishing'] .s5 { animation-delay: 0.35s; }
  [data-mood='finishing'] .speck { animation: speck 0.7s ease-out infinite; }
  [data-mood='finishing'] .p2 { animation-delay: 0.23s; }
  [data-mood='finishing'] .p3 { animation-delay: 0.46s; }

  /* Video: the clapperboard snaps shut with every stroke */
  [data-tool='clapper'][data-mood='creating'] .clap-arm { animation: clap 0.9s ease-in-out infinite; }
  [data-tool='clapper'][data-mood='finishing'] .clap-arm { animation: clap 0.55s ease-in-out infinite; }

  /* Nodding: two quick nods, a happy squint and a check badge */
  [data-mood='nodding'] .root { animation: nod 0.9s ease-in-out 1; }
  [data-mood='nodding'] .brush { transform: rotate(-14deg); }
  [data-mood='nodding'] .check { animation: exclaim 0.6s ease-out 1 both; }

  /* Cheerful: squash-and-stretch hop, waving brush, confetti */
  [data-mood='cheerful'] .root { animation: hop 0.8s ease-in-out infinite; }
  [data-mood='cheerful'] .shadow { animation: shadow-hop 0.8s ease-in-out infinite; }
  [data-mood='cheerful'] .beret { animation: beret 0.8s ease-in-out infinite; }
  [data-mood='cheerful'] .brush { animation: wave 0.4s ease-in-out infinite alternate; }
  [data-mood='cheerful'] .confetti { animation: confetti 1.6s linear infinite; transform-box: fill-box; transform-origin: center; }
  [data-mood='cheerful'] .c2 { animation-delay: 0.5s; }
  [data-mood='cheerful'] .c3 { animation-delay: 0.2s; }
  [data-mood='cheerful'] .c4 { animation-delay: 0.9s; }
  [data-mood='cheerful'] .c5 { animation-delay: 0.35s; }
  [data-mood='cheerful'] .c6 { animation-delay: 1.1s; }
  [data-mood='cheerful'] .c7 { animation-delay: 0.7s; }
  [data-mood='cheerful'] .c8 { animation-delay: 1.3s; }

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

  /* Sleeping: slow breathing, lowered brush, drifting Zs */
  [data-mood='sleeping'] .root { animation: breathe 3.6s ease-in-out infinite; }
  [data-mood='sleeping'] .brush { transform: rotate(48deg); }
  [data-mood='sleeping'] .beret { transform: rotate(10deg) translate(4px, 4px); }
  [data-mood='sleeping'] .z { animation: drift 3.6s ease-in infinite; }
  [data-mood='sleeping'] .z2 { animation-delay: 1.2s; }
  [data-mood='sleeping'] .z3 { animation-delay: 2.4s; }

  /* Paused: still and patient, heavy-lidded, brush resting, a pulsing pause badge */
  [data-mood='paused'] .root { animation: breathe 4.4s ease-in-out infinite; }
  [data-mood='paused'] .eye { transform: scaleY(0.55); }
  [data-mood='paused'] .brush { transform: rotate(30deg); }
  [data-mood='paused'] .pause-badge { animation: pulse 2.4s ease-in-out infinite; }

  /* Sad: droop, look down, a single tear */
  [data-mood='sad'] .root { animation: droop 4s ease-in-out infinite; }
  [data-mood='sad'] .iris { transform: translate(-3px, 8px); }
  [data-mood='sad'] .eye { transform: scaleY(0.82); }
  [data-mood='sad'] .brush { transform: rotate(52deg); }
  [data-mood='sad'] .beret { transform: rotate(12deg) translateY(4px); }
  [data-mood='sad'] .tear { animation: tear 2.4s ease-in infinite; }

  /* Surprised: startled jump, wide eye, tiny pupil, flying beret */
  [data-mood='surprised'] .root { animation: startle 0.6s ease-out 1; }
  [data-mood='surprised'] .eye { transform: scale(1.14); }
  [data-mood='surprised'] .iris { transform: scale(0.72); }
  [data-mood='surprised'] .beret { animation: beret-pop 0.6s ease-out 1 forwards; }
  [data-mood='surprised'] .exclaim { animation: exclaim 0.9s ease-out 1 both; }
  [data-mood='surprised'] .brush { transform: rotate(-30deg); }

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
    0%, 100% { transform: translate(9px, -9px); }
    45% { transform: translate(-6px, -10px); }
    55% { transform: translate(-6px, -10px); }
  }
  @keyframes tap {
    0%, 100% { transform: rotate(28deg); }
    50% { transform: rotate(36deg); }
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
    0%, 100% { transform: translate(6px, -3px); }
    50% { transform: translate(14px, -7px); }
  }
  @keyframes paint {
    0%, 100% { transform: rotate(-22deg); }
    50% { transform: rotate(14deg); }
  }
  @keyframes twinkle {
    0%, 100% { transform: scale(0.2) rotate(0deg); opacity: 0; }
    50% { transform: scale(1.1) rotate(45deg); opacity: 1; }
  }
  @keyframes speck {
    0% { transform: translate(0, 0); opacity: 1; }
    100% { transform: translate(14px, -26px); opacity: 0; }
  }

  @keyframes hop {
    0%, 100% { transform: translateY(0) scale(1.06, 0.94); }
    40% { transform: translateY(-18px) scale(0.96, 1.05); }
    60% { transform: translateY(-18px) scale(0.98, 1.02); }
    85% { transform: translateY(0) scale(1.04, 0.96); }
  }
  @keyframes shadow-hop {
    0%, 100% { transform: scaleX(1); opacity: 0.35; }
    50% { transform: scaleX(0.7); opacity: 0.2; }
  }
  @keyframes beret {
    0%, 100% { transform: rotate(0deg); }
    40% { transform: translateY(-5px) rotate(-8deg); }
    60% { transform: translateY(-5px) rotate(6deg); }
  }
  @keyframes wave {
    from { transform: rotate(-24deg); }
    to { transform: rotate(10deg); }
  }
  @keyframes confetti {
    0% { transform: translateY(0) rotate(0deg); opacity: 0; }
    10% { opacity: 1; }
    100% { transform: translateY(150px) rotate(540deg); opacity: 0; }
  }

  @keyframes hand-wave {
    from { transform: translate(-4px, -58px); }
    to { transform: translate(-14px, -64px); }
  }
  @keyframes peek {
    0%, 100% { transform: rotate(-5deg) translateX(-3px); }
    50% { transform: rotate(-7deg) translateX(-5px); }
  }
  @keyframes read {
    0%, 100% { transform: translate(-13px, 3px); }
    40% { transform: translate(-6px, 4px); }
    60% { transform: translate(-6px, 4px); }
  }
  @keyframes breathe {
    0%, 100% { transform: scale(1, 1); }
    50% { transform: scale(1.02, 1.04); }
  }
  @keyframes drift {
    0% { transform: translate(0, 6px); opacity: 0; }
    30% { opacity: 1; }
    100% { transform: translate(10px, -14px); opacity: 0; }
  }
  @keyframes pulse { 50% { opacity: 0.45; } }
  @keyframes scan {
    0%, 100% { transform: translate(-14px, 3px); }
    30% { transform: translate(-6px, 4px); }
    45% { transform: translate(-14px, 7px); }
    75% { transform: translate(-6px, 8px); }
    90% { transform: translate(-14px, 3px); }
  }
  @keyframes clap {
    0%, 55%, 100% { transform: rotate(0deg); }
    30% { transform: rotate(-26deg); }
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
    100% { transform: translateY(40px); opacity: 0; }
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

  @media (prefers-reduced-motion: reduce) {
    .mascot * { animation: none !important; }
    [data-mood='thinking'] .iris { transform: translate(9px, -9px); }
    [data-mood='thinking'] .brush { transform: rotate(30deg); }
    [data-mood='thinking'] .dot { opacity: 1; }
    [data-mood='creating'] .spark { opacity: 1; }
    [data-mood='cheerful'] .confetti { opacity: 1; }
    [data-mood='waving'] .hand-l { transform: translate(-4px, -58px); }
    [data-mood='curious'] .iris { transform: translate(-13px, 3px); }
    [data-mood='curious'] .root { transform: rotate(-5deg); }
    [data-mood='reading'] .iris { transform: translate(-10px, 5px); }
    [data-mood='reading'] .root { transform: rotate(-5deg); }
    [data-mood='finishing'] .spark { opacity: 1; }
    [data-mood='sleeping'] .z { opacity: 1; }
    [data-mood='sad'] .tear { opacity: 1; }
  }
</style>
