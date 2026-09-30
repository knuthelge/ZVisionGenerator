<script lang="ts" module>
  export type MascotMood =
    | 'idle'
    | 'thinking'
    | 'creating'
    | 'cheerful'
    | 'waving'
    | 'curious'
    | 'sleeping'
    | 'sad'
    | 'surprised';
</script>

<script lang="ts">
  interface Props {
    mood?: MascotMood;
    size?: number | string;
    class?: string;
  }

  let {
    mood = 'idle',
    size = 120,
    class: extraClass = ''
  }: Props = $props();

  const labels: Record<MascotMood, string> = {
    idle: 'Z-Vision mascot',
    thinking: 'Z-Vision mascot is thinking',
    creating: 'Z-Vision mascot is creating an image',
    cheerful: 'Z-Vision mascot is cheering',
    waving: 'Z-Vision mascot is waving hello',
    curious: 'Z-Vision mascot is watching you type',
    sleeping: 'Z-Vision mascot is sleeping',
    sad: 'Z-Vision mascot is sad',
    surprised: 'Z-Vision mascot is surprised'
  };
</script>

<svg
  class="mascot {extraClass}"
  data-mood={mood}
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

  <!-- Surprised: exclamation mark -->
  <g class="fx fx-surprised">
    <g class="exclaim">
      <rect x="160" y="-6" width="9" height="30" rx="4.5" fill="#fbbf24" />
      <circle cx="164.5" cy="34" r="5" fill="#fbbf24" />
    </g>
  </g>

  <g class="root">
    <!-- Brush arm (behind body) -->
    <g class="brush">
      <line x1="168" y1="126" x2="188" y2="84" stroke="#a16207" stroke-width="6" stroke-linecap="round" />
      <path d="M184,86 Q186,68 196,62 Q198,78 192,90 Z" fill="#fbbf24" />
    </g>

    <path
      class="body"
      d="M100,42 C150,42 172,90 170,130 C168,168 140,182 100,182 C60,182 32,168 30,130 C28,90 50,42 100,42 Z"
      fill="#2dd4bf"
    />
    <ellipse cx="100" cy="150" rx="42" ry="24" fill="#99f6e4" opacity=".45" />
    <circle class="hand-r" cx="168" cy="128" r="10" fill="#14b8a6" />
    <circle class="hand-l" cx="32" cy="132" r="10" fill="#14b8a6" />

    <g class="beret">
      <ellipse cx="100" cy="46" rx="36" ry="10" fill="#f87171" />
      <circle cx="100" cy="35" r="5" fill="#f87171" />
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
    <g class="mouth m-cheerful">
      <path d="M84,126 Q100,152 116,126 Z" fill="#134e4a" stroke="#134e4a" stroke-width="3" stroke-linejoin="round" />
      <ellipse cx="100" cy="140" rx="7" ry="4" fill="#f472b6" />
    </g>
  </g>

  <!-- Creating: sparkles and paint specks around the brush tip -->
  <g class="fx fx-creating">
    <g transform="translate(204 58)"><path class="spark s1" d="M0,-9 L2,-2 L9,0 L2,2 L0,9 L-2,2 L-9,0 L-2,-2Z" fill="#fbbf24" /></g>
    <g transform="translate(176 40)"><path class="spark s2" d="M0,-7 L1.6,-1.6 L7,0 L1.6,1.6 L0,7 L-1.6,1.6 L-7,0 L-1.6,-1.6Z" fill="#fff" /></g>
    <g transform="translate(210 96)"><path class="spark s3" d="M0,-6 L1.4,-1.4 L6,0 L1.4,1.4 L0,6 L-1.4,1.4 L-6,0 L-1.4,-1.4Z" fill="#2dd4bf" /></g>
    <circle class="speck p1" cx="194" cy="70" r="3.5" fill="#f87171" />
    <circle class="speck p2" cx="190" cy="76" r="3" fill="#60a5fa" />
    <circle class="speck p3" cx="198" cy="74" r="3" fill="#fbbf24" />
  </g>

  <!-- Cheerful: confetti -->
  <g class="fx fx-cheerful">
    <rect class="confetti c1" x="20" y="-10" width="7" height="11" rx="1.5" fill="#fbbf24" />
    <rect class="confetti c2" x="52" y="-18" width="6" height="10" rx="1.5" fill="#f87171" />
    <rect class="confetti c3" x="84" y="-6" width="7" height="11" rx="1.5" fill="#60a5fa" />
    <rect class="confetti c4" x="118" y="-16" width="6" height="10" rx="1.5" fill="#2dd4bf" />
    <rect class="confetti c5" x="150" y="-8" width="7" height="11" rx="1.5" fill="#f9a8d4" />
    <rect class="confetti c6" x="178" y="-14" width="6" height="10" rx="1.5" fill="#fbbf24" />
    <rect class="confetti c7" x="0" y="-4" width="6" height="10" rx="1.5" fill="#2dd4bf" />
    <rect class="confetti c8" x="196" y="-2" width="7" height="11" rx="1.5" fill="#f87171" />
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
  .brush { transform-origin: 162px 132px; transition: transform 300ms ease; }
  .beret { transform-origin: 100px 50px; }
  .iris { transform-origin: 100px 100px; transition: transform 300ms ease; }
  .exclaim { transform-origin: 164px 30px; }
  .blush { opacity: 0.7; transition: opacity 300ms ease; }
  .hand-l { transition: transform 300ms ease; }
  .beret { transition: transform 300ms ease; }
  .eye { transition: opacity 250ms ease, transform 250ms ease; }

  /* Mood-dependent visibility */
  .mouth,
  .eye-happy,
  .eye-closed,
  .tear,
  .fx { opacity: 0; transition: opacity 250ms ease; }

  [data-mood='idle'] .m-idle,
  [data-mood='thinking'] .m-thinking,
  [data-mood='thinking'] .fx-thinking,
  [data-mood='creating'] .m-creating,
  [data-mood='creating'] .fx-creating,
  [data-mood='cheerful'] .m-cheerful,
  [data-mood='cheerful'] .eye-happy,
  [data-mood='cheerful'] .fx-cheerful,
  [data-mood='waving'] .m-idle,
  [data-mood='curious'] .m-idle,
  [data-mood='sleeping'] .m-sleeping,
  [data-mood='sleeping'] .eye-closed,
  [data-mood='sleeping'] .fx-sleeping,
  [data-mood='sad'] .m-sad,
  [data-mood='surprised'] .m-surprised,
  [data-mood='surprised'] .fx-surprised { opacity: 1; }

  [data-mood='cheerful'] .eye,
  [data-mood='sleeping'] .eye { opacity: 0; }
  [data-mood='sleeping'] .blush,
  [data-mood='sad'] .blush { opacity: 0.3; }
  [data-mood='cheerful'] .blush { opacity: 1; }

  /* Idle: gentle float and blink */
  [data-mood='idle'] .root { animation: float 3.2s ease-in-out infinite; }
  [data-mood='idle'] .shadow { animation: shadow-float 3.2s ease-in-out infinite; }
  [data-mood='idle'] .eye { animation: blink 4.5s infinite; }

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
    [data-mood='sleeping'] .z { opacity: 1; }
    [data-mood='sad'] .tear { opacity: 1; }
  }
</style>
