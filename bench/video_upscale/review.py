"""A small blind-review site for the video-upscale bench, served on the local network.

Each question shows one large viewer that flips between the source and the candidate upscales (shuffled letters), as
stills (a centre detail or the whole frame) or videos that keep their position when flipped. Media is served under
opaque letter URLs, so nothing on the page names a method. Answers go to ``runs/review/votes.jsonl`` with the letters
resolved to method names on the server.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from bench.video_upscale.sheet import _pixel_double, _read_frame

# Frames shown per clip: mid-motion frames, and for the extended clip the first frame after each join.
REVIEW_FRAMES = {"walk5": (48, 84), "face": (48, 90), "medium5": (48, 84), "walk10": (121, 185)}
# Where the detail crop is centred per clip, as fractions of the frame: on the subject, which is rarely in the middle.
REVIEW_FOCUS = {"walk5": (0.43, 0.55), "face": (0.66, 0.45), "medium5": (0.39, 0.62), "walk10": (0.25, 0.65)}
DETAIL_FRACTION = 0.45
SOURCE = "S"


def detail_crop(frame: np.ndarray, focus: tuple[float, float]) -> np.ndarray:
    """Return a DETAIL_FRACTION crop of *frame* centred on *focus*, moved inside the frame where it would cross an edge."""
    height, width = frame.shape[:2]
    crop_h, crop_w = int(height * DETAIL_FRACTION), int(width * DETAIL_FRACTION)
    top = min(max(int(focus[1] * height) - crop_h // 2, 0), height - crop_h)
    left = min(max(int(focus[0] * width) - crop_w // 2, 0), width - crop_w)
    return frame[top : top + crop_h, left : left + crop_w]


@dataclass(frozen=True)
class Question:
    """One review question: a clip and the candidate videos, each a path under the runs folder."""

    id: str
    title: str
    prompt: str
    clip: str
    candidates: tuple[Path, ...]
    flag_changes: bool

    def letters(self) -> dict[str, Path]:
        """Return the candidates by letter, shuffled by the question id so the key stays fixed between visits."""
        order = list(self.candidates)
        random.Random(self.id).shuffle(order)
        return {chr(ord("A") + index): path for index, path in enumerate(order)}


def build_questions(runs: Path) -> list[Question]:
    """Return the questions the finished runs can answer, in review order."""
    questions = []
    for clip in ("face", "walk5", "walk10", "medium5"):
        videos = _videos(runs / "steps" / clip)
        if len(videos) > 1:
            questions.append(
                Question(
                    id=f"best-{clip}",
                    title=f"Best upscale · {clip}",
                    prompt="Pick the one you'd keep. Compare with the source: a good upscale is sharper without changing faces, people, objects or light.",
                    clip=clip,
                    candidates=tuple(videos),
                    flag_changes=True,
                )
            )
    tiled = runs / "encode" / "medium5" / "refine-1-e512x48.mp4"
    if tiled.exists():
        questions.append(
            Question(
                id="encode-medium5",
                title="Encoder tiles · medium5",
                prompt="Same settings, two ways of reading the source in. Pick the better one, or Can't tell.",
                clip="medium5",
                candidates=(runs / "steps" / "medium5" / "refine-1.mp4", tiled),
                flag_changes=False,
            )
        )
    for clip in ("face", "walk5"):
        empty = runs / "prompt" / clip / "refine-1-noprompt.mp4"
        if empty.exists():
            questions.append(
                Question(
                    id=f"prompt-{clip}",
                    title=f"Prompt · {clip}",
                    prompt="Same settings, with and without the clip's own prompt. Pick the better one, or Can't tell.",
                    clip=clip,
                    candidates=(runs / "steps" / clip / "refine-1.mp4", empty),
                    flag_changes=False,
                )
            )
    return questions


def _videos(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.glob("*.mp4") if not path.stem.endswith(".raw"))


class _Site:
    """The review site's state: the runs folder, the still cache and the votes file."""

    def __init__(self, runs: Path) -> None:
        self.runs = runs
        self.cache = runs / "review" / "stills"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.votes = runs / "review" / "votes.jsonl"

    def questions(self) -> list[Question]:
        return build_questions(self.runs)

    def question(self, question_id: str) -> Question | None:
        return next((question for question in self.questions() if question.id == question_id), None)

    def video(self, question: Question, letter: str) -> Path | None:
        if letter == SOURCE:
            return self.runs / "sources" / f"{question.clip}.mp4"
        return question.letters().get(letter)

    def still(self, question: Question, letter: str, frame: int, detail: bool) -> Path | None:
        """Return a JPEG of *letter* at *frame*: the detail around the clip's subject or the whole frame; the source is pixel-doubled.

        Made on first use and cached by video, not letter: the letters reshuffle when a question gains a candidate.
        """
        video = self.video(question, letter)
        if video is None:
            return None
        target = self.cache / f"{video.parent.parent.name}_{video.parent.name}_{video.stem}_{frame}_{'detail' if detail else 'full'}.jpg"
        if not target.exists():
            from PIL import Image

            image = _read_frame(video, frame)
            if letter == SOURCE:
                image = _pixel_double(image)
            Image.fromarray(detail_crop(image, REVIEW_FOCUS.get(question.clip, (0.5, 0.5))) if detail else image).save(target, quality=90)
        return target

    def answers(self) -> dict[str, dict[str, Any]]:
        """Return the latest saved answer per question."""
        latest: dict[str, dict[str, Any]] = {}
        if self.votes.exists():
            for line in self.votes.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                latest[record["question"]] = record
        return latest

    def save(self, question: Question, answer: dict[str, Any]) -> None:
        letters = question.letters()
        pick = str(answer.get("pick", ""))
        record = {
            "question": question.id,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pick": pick,
            "pick_method": letters[pick].stem if pick in letters else pick,
            "changed_letters": [letter for letter in answer.get("changed", []) if letter in letters],
            "changed": [letters[letter].stem for letter in answer.get("changed", []) if letter in letters],
            "note": str(answer.get("note", "")),
            "letters": {letter: path.stem for letter, path in letters.items()},
        }
        with self.votes.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def listing(self) -> list[dict[str, Any]]:
        """Return the questions for the page, with each one's saved answer in letters only."""
        answers = self.answers()
        return [
            {
                "id": question.id,
                "title": question.title,
                "prompt": question.prompt,
                "letters": list(question.letters()),
                "frames": list(REVIEW_FRAMES.get(question.clip, (48,))),
                "flagChanges": question.flag_changes,
                "answer": _current_answer(question, answers.get(question.id)),
            }
            for question in self.questions()
        ]


def _current_answer(question: Question, record: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the saved answer in letters, or ``None`` when there is none or the question's letters have changed since."""
    current = {letter: path.stem for letter, path in question.letters().items()}
    if record is None or record.get("letters") != current:
        return None
    return {key: record.get(key) for key in ("pick", "changed_letters", "note")}


class _Handler(BaseHTTPRequestHandler):
    site: _Site

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 -- stdlib signature
        return

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/":
            return self._send(HTTPStatus.OK, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
        if path == "/api/questions":
            return self._send(HTTPStatus.OK, json.dumps(self.site.listing()).encode("utf-8"), "application/json")
        if match := re.fullmatch(r"/i/([\w-]+)/([A-Z])/(\d+)/(detail|full)", path):
            question = self.site.question(match[1])
            still = self.site.still(question, match[2], int(match[3]), match[4] == "detail") if question else None
            return self._file(still, "image/jpeg", cache=True) if still else self._missing()
        if match := re.fullmatch(r"/v/([\w-]+)/([A-Z])", path):
            question = self.site.question(match[1])
            video = self.site.video(question, match[2]) if question else None
            return self._file(video, "video/mp4") if video else self._missing()
        return self._missing()

    def do_POST(self) -> None:
        match = re.fullmatch(r"/api/answer/([\w-]+)", self.path)
        question = self.site.question(match[1]) if match else None
        if question is None:
            return self._missing()
        length = int(self.headers.get("Content-Length", 0))
        self.site.save(question, json.loads(self.rfile.read(length) or b"{}"))
        self._send(HTTPStatus.OK, b'{"ok": true}', "application/json")

    def _send(self, status: HTTPStatus, body: bytes, kind: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _missing(self) -> None:
        self._send(HTTPStatus.NOT_FOUND, b"Not found", "text/plain")

    def _file(self, path: Path, kind: str, *, cache: bool = False) -> None:
        """Send *path*, honouring a single byte range (Safari only plays video served with ranges)."""
        size = path.stat().st_size
        start, end = 0, size - 1
        ranged = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if ranged and (ranged[1] or ranged[2]):
            if ranged[1]:
                start = int(ranged[1])
                end = min(int(ranged[2]), size - 1) if ranged[2] else size - 1
            else:
                start = max(0, size - int(ranged[2]))
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", kind)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if cache:
            self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = handle.read(min(1 << 20, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except BrokenPipeError, ConnectionResetError:
                    return
                remaining -= len(chunk)


def serve(runs: Path, host: str, port: int) -> None:
    """Serve the review site for *runs* on *host*:*port* until interrupted."""
    handler = type("Handler", (_Handler,), {"site": _Site(runs)})
    with ThreadingHTTPServer((host, port), handler) as server:
        print(f"Review site on http://{host}:{port}/", flush=True)
        server.serve_forever()


_PAGE = r"""<!doctype html>
<html lang=en>
<meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Upscale review</title>
<style>
:root { color-scheme: dark; --bg: #0e0e10; --panel: #1a1a1d; --line: #2c2c31; --text: #e8e8ea; --dim: #9a9aa2; --accent: #6ea8ff; --good: #4cc38a; --warn: #f0a04b; }
* { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
body { margin: 0; background: var(--bg); color: var(--text); font: 16px/1.4 -apple-system, system-ui, sans-serif; padding-bottom: env(safe-area-inset-bottom); }
header { display: flex; align-items: center; gap: 8px; padding: 10px 12px; position: sticky; top: 0; background: var(--bg); z-index: 2; }
header h1 { font-size: 17px; margin: 0; flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
header .count { color: var(--dim); font-size: 14px; }
.icon { background: var(--panel); border: 1px solid var(--line); color: var(--text); border-radius: 10px; width: 40px; height: 40px; font-size: 20px; }
.icon:disabled { opacity: .3; }
.prompt { color: var(--dim); font-size: 14px; margin: 0 12px 10px; }
.viewer { position: relative; background: #000; touch-action: pan-y pinch-zoom; user-select: none; }
.viewer img, .viewer video { display: block; width: 100%; height: auto; }
.badge { position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,.65); color: #fff; font-weight: 700; font-size: 22px; min-width: 44px; text-align: center; padding: 4px 10px; border-radius: 10px; pointer-events: none; }
.badge.src { color: var(--accent); }
.badge .tag { font-size: 13px; font-weight: 600; margin-left: 6px; }
.hint { position: absolute; bottom: 8px; right: 10px; color: rgba(255,255,255,.55); font-size: 12px; pointer-events: none; }
.tabs { display: flex; gap: 6px; padding: 10px 12px; overflow-x: auto; }
.tab { flex: 1 0 auto; min-width: 52px; padding: 10px 0; border-radius: 10px; border: 1px solid var(--line); background: var(--panel); color: var(--text); font-size: 17px; font-weight: 700; position: relative; }
.tab.on { border-color: var(--accent); background: #22314a; }
.tab.src { color: var(--accent); }
.tab .dot { position: absolute; top: 4px; right: 6px; font-size: 11px; }
.row { display: flex; gap: 6px; padding: 0 12px 10px; flex-wrap: wrap; align-items: center; }
.seg { display: inline-flex; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
.seg button { background: var(--panel); color: var(--dim); border: 0; padding: 8px 12px; font-size: 14px; }
.seg button.on { background: #2a2a30; color: var(--text); }
.answer { margin: 6px 12px 16px; background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 12px; }
.answer h2 { font-size: 14px; color: var(--dim); font-weight: 600; margin: 0 0 8px; text-transform: uppercase; letter-spacing: .04em; }
.chips { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
.chip { min-width: 52px; padding: 10px 14px; border-radius: 22px; border: 1px solid var(--line); background: #111114; color: var(--text); font-size: 16px; font-weight: 700; }
.chip.best.on { background: var(--good); border-color: var(--good); color: #062; }
.chip.changed.on { background: var(--warn); border-color: var(--warn); color: #301800; }
textarea { width: 100%; min-height: 44px; background: #111114; color: var(--text); border: 1px solid var(--line); border-radius: 10px; padding: 8px; font: inherit; }
.status { color: var(--dim); font-size: 13px; margin-top: 8px; min-height: 18px; }
.next { width: 100%; margin-top: 10px; padding: 14px; border-radius: 12px; border: 0; background: var(--accent); color: #061a3a; font-size: 17px; font-weight: 700; }
.list { padding: 0 12px; }
.list a { display: flex; justify-content: space-between; padding: 16px; margin-bottom: 8px; background: var(--panel); border: 1px solid var(--line); border-radius: 12px; color: var(--text); text-decoration: none; }
.list .done { color: var(--good); }
</style>
<div id=app></div>
<script>
const state = { questions: [], index: -1, letter: "S", frame: 0, mode: "still", detail: true };
const app = document.getElementById("app");

async function load() {
  state.questions = await (await fetch("/api/questions")).json();
  route();
}
function route() {
  const id = decodeURIComponent(location.hash.replace(/^#/, ""));
  state.index = state.questions.findIndex(q => q.id === id);
  state.letter = "S"; state.frame = 0;
  state.index < 0 ? renderList() : renderQuestion();
}
window.addEventListener("hashchange", route);

function renderList() {
  const items = state.questions.map(q => `<a href="#${q.id}"><span>${q.title}</span><span class=done>${q.answer && q.answer.pick ? "✓" : ""}</span></a>`).join("");
  app.innerHTML = `<header><h1>Upscale review</h1></header><p class=prompt>Flip between the source and the blind versions in one viewer; answers save as you go.</p><div class=list>${items}</div>`;
}

const q = () => state.questions[state.index];
const order = () => ["S", ...q().letters];
const label = l => l === "S" ? "Source" : l;
const stillUrl = l => `/i/${q().id}/${l}/${q().frames[state.frame]}/${state.detail ? "detail" : "full"}`;

function renderQuestion() {
  const question = q();
  const n = state.questions.length;
  app.innerHTML = `
    <header>
      <button class=icon onclick="location.hash=''" aria-label="All questions">☰</button>
      <h1>${question.title}</h1><span class=count>${state.index + 1}/${n}</span>
      <button class=icon id=prev aria-label="Previous question" ${state.index === 0 ? "disabled" : ""}>‹</button>
      <button class=icon id=nextq aria-label="Next question" ${state.index === n - 1 ? "disabled" : ""}>›</button>
    </header>
    <p class=prompt>${question.prompt}</p>
    <div class=viewer id=viewer></div>
    <div class=tabs id=tabs></div>
    <div class=row>
      <div class=seg id=mode><button data-v=still>Still</button><button data-v=video>Video</button></div>
      <div class=seg id=detail><button data-v=1>Detail</button><button data-v=0>Whole</button></div>
      <div class=seg id=frames>${question.frames.map((f, i) => `<button data-v=${i}>Frame ${f}</button>`).join("")}</div>
    </div>
    <div class=answer>
      <h2>Best</h2><div class=chips id=best></div>
      ${question.flagChanges ? `<h2>Changed faces, people, objects or light</h2><div class=chips id=changed></div>` : ""}
      <textarea id=note placeholder="Note (optional)"></textarea>
      <div class=status id=status></div>
      <button class=next id=go>${state.index === n - 1 ? "Done" : "Next question"}</button>
    </div>`;
  document.getElementById("prev").onclick = () => go(-1);
  document.getElementById("nextq").onclick = () => go(1);
  document.getElementById("go").onclick = () => go(1);
  bindSeg("mode", v => { state.mode = v; showViewer(true); });
  bindSeg("detail", v => { state.detail = v === "1"; showViewer(true); });
  bindSeg("frames", v => { state.frame = +v; showViewer(true); });
  renderAnswer();
  showViewer(true);
  preload();
}

function bindSeg(id, onPick) {
  const seg = document.getElementById(id);
  seg.onclick = e => { const b = e.target.closest("button"); if (b) { onPick(b.dataset.v); syncSegs(); } };
}
function syncSegs() {
  const on = { mode: state.mode, detail: state.detail ? "1" : "0", frames: String(state.frame) };
  for (const [id, v] of Object.entries(on)) document.querySelectorAll(`#${id} button`).forEach(b => b.classList.toggle("on", b.dataset.v === v));
  document.getElementById("detail").style.display = document.getElementById("frames").style.display = state.mode === "still" ? "" : "none";
}

function answerOf() { return q().answer || (q().answer = { pick: "", changed_letters: [], note: "" }); }

function showViewer(rebuild) {
  syncSegs();
  const viewer = document.getElementById("viewer");
  const tag = l => { const a = answerOf(); return (a.pick === l ? "<span class=tag>★ best</span>" : "") + ((a.changed_letters || []).includes(l) ? "<span class=tag>⚠ changed</span>" : ""); };
  const badge = `<div class="badge ${state.letter === "S" ? "src" : ""}">${label(state.letter)}${tag(state.letter)}</div><div class=hint>tap left / right · swipe</div>`;
  if (state.mode === "still") {
    viewer.innerHTML = `<img src="${stillUrl(state.letter)}" alt="${label(state.letter)}">${badge}`;
  } else {
    const old = viewer.querySelector("video");
    const t = old ? old.currentTime : 0, playing = old && !old.paused;
    viewer.innerHTML = `<video src="/v/${q().id}/${state.letter}" muted loop playsinline ${playing || !old ? "autoplay" : ""}></video>${badge}`;
    const v = viewer.querySelector("video");
    v.addEventListener("loadedmetadata", () => { v.currentTime = t; if (playing || !old) v.play().catch(() => {}); }, { once: true });
  }
  document.getElementById("tabs").innerHTML = order().map(l => {
    const a = answerOf(); const dot = a.pick === l ? "★" : ((a.changed_letters || []).includes(l) ? "⚠" : "");
    return `<button class="tab ${l === "S" ? "src" : ""} ${l === state.letter ? "on" : ""}" data-l=${l}>${label(l)}<span class=dot>${dot}</span></button>`;
  }).join("");
  document.getElementById("tabs").onclick = e => { const b = e.target.closest("button"); if (b) { state.letter = b.dataset.l; showViewer(); } };
  bindSwipe(viewer);
}

function step(delta) {
  const o = order(); state.letter = o[(o.indexOf(state.letter) + delta + o.length) % o.length]; showViewer();
}
function bindSwipe(viewer) {
  let x0 = null, y0 = null, moved = false;
  viewer.ontouchstart = e => { if (e.touches.length === 1) { x0 = e.touches[0].clientX; y0 = e.touches[0].clientY; moved = false; } else x0 = null; };
  viewer.ontouchmove = e => { if (x0 !== null && Math.abs(e.touches[0].clientX - x0) > 10) moved = true; };
  viewer.ontouchend = e => {
    if (x0 === null) return;
    const dx = e.changedTouches[0].clientX - x0, dy = e.changedTouches[0].clientY - y0;
    if (moved && Math.abs(dx) > 40 && Math.abs(dx) > Math.abs(dy)) step(dx < 0 ? 1 : -1);
    x0 = null;
  };
  viewer.onclick = e => {
    if (state.mode === "video" && e.target.tagName === "VIDEO" && e.offsetX > viewer.clientWidth / 3 && e.offsetX < viewer.clientWidth * 2 / 3) {
      const v = e.target; v.paused ? v.play() : v.pause(); return;
    }
    if (moved) return;
    step(e.offsetX < viewer.clientWidth / 2 ? -1 : 1);
  };
}
document.addEventListener("keydown", e => {
  if (state.index < 0 || e.target.tagName === "TEXTAREA") return;
  if (e.key === "ArrowRight") step(1);
  if (e.key === "ArrowLeft") step(-1);
});

function renderAnswer() {
  const question = q(), a = answerOf();
  const best = [...question.letters, ...(question.flagChanges ? [] : ["same"])];
  document.getElementById("best").innerHTML = best.map(l => `<button class="chip best ${a.pick === l ? "on" : ""}" data-l=${l}>${l === "same" ? "Can't tell" : l}</button>`).join("");
  document.getElementById("best").onclick = e => { const b = e.target.closest("button"); if (!b) return; a.pick = a.pick === b.dataset.l ? "" : b.dataset.l; save(); };
  const changed = document.getElementById("changed");
  if (changed) {
    changed.innerHTML = question.letters.map(l => `<button class="chip changed ${(a.changed_letters || []).includes(l) ? "on" : ""}" data-l=${l}>${l}</button>`).join("");
    changed.onclick = e => {
      const b = e.target.closest("button"); if (!b) return;
      const set = new Set(a.changed_letters || []); set.has(b.dataset.l) ? set.delete(b.dataset.l) : set.add(b.dataset.l);
      a.changed_letters = [...set]; save();
    };
  }
  const note = document.getElementById("note");
  note.value = a.note || "";
  note.onchange = () => { a.note = note.value; save(); };
}

async function save() {
  const a = answerOf();
  renderAnswer(); showViewer();
  document.getElementById("status").textContent = "Saving…";
  try {
    await fetch(`/api/answer/${q().id}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ pick: a.pick, changed: a.changed_letters || [], note: a.note || "" }) });
    document.getElementById("status").textContent = "Saved ✓";
  } catch { document.getElementById("status").textContent = "Couldn't save; try again."; }
}

function go(delta) {
  const i = state.index + delta;
  location.hash = i >= 0 && i < state.questions.length ? state.questions[i].id : "";
  window.scrollTo(0, 0);
}
function preload() {
  for (const l of order()) for (const d of [true, false]) for (const f of q().frames) {
    const img = new Image(); img.src = `/i/${q().id}/${l}/${f}/${d ? "detail" : "full"}`;
  }
}
load();
</script>
"""
