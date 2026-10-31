"""Serve a blind A/B survey of the bench mattes: the original render and two unnamed cut-outs, pick the better one.

Every question compares two methods on the same render, so the original is the same for both sides. Sides are
shuffled and images are served by question id, so nothing on the page names a method. Answers are appended to
``runs/survey/answers.jsonl`` (a later answer to the same question replaces an earlier one)::

    <bench python> -m bench.transparent.survey --port 8791
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from dataclasses import asdict, dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bench.transparent.__main__ import RUNS_DIR, find_render, load_cases

PHOTO_METHODS = ("birefnet+fg", "ben2+fg", "birefnet-matting+fg", "birefnet+vitmatte+fg")
GRAPHIC_METHODS = {
    "birefnet-matting+fg": "white/birefnet-matting+fg",
    "ink": "white/ink",
    "ink-solid": "white/ink-solid",
    "difference": "pair-edit/difference",
}
_SEED = 20261031
_PAGE = Path(__file__).with_name("survey.html")


@dataclass(frozen=True)
class Question:
    """One comparison: the original render and the two mattes shown as A and B."""

    id: str
    kind: str
    subject: str
    model: str
    original: str
    a_method: str
    a_path: str
    b_method: str
    b_path: str


def _balanced_pairs(methods: list[str], count: int, rng: random.Random) -> list[tuple[str, str]]:
    """Return *count* method pairs cycling through every combination, so each pair comes up about equally often."""
    combos = list(itertools.combinations(methods, 2))
    rng.shuffle(combos)
    return [combos[index % len(combos)] for index in range(count)]


def build_plan(runs: Path, rng: random.Random) -> list[Question]:
    """Return the shuffled questions: one photo pair per grey render, two graphic pairs per klein 4B white render."""
    _, subjects = load_cases()
    renders, mattes = runs / "renders", runs / "mattes"
    questions: list[Question] = []
    photo = [(model.name, name) for model in sorted(renders.iterdir()) if model.is_dir() for name in subjects if find_render(model / "grey", name)]
    for (model, name), pair in zip(photo, _balanced_pairs(list(PHOTO_METHODS), len(photo), rng)):
        paths = [mattes / model / "grey" / method / f"{name}.png" for method in pair]
        if all(path.exists() for path in paths):
            questions.append(_question("photo", model, name, find_render(renders / model / "grey", name), pair, paths, rng))
    graphic = [name for name in subjects if find_render(renders / "klein4b" / "white", name)]
    for name, pair in zip(graphic * 2, _balanced_pairs(list(GRAPHIC_METHODS), len(graphic) * 2, rng)):
        paths = [mattes / "klein4b" / GRAPHIC_METHODS[method] / f"{name}.png" for method in pair]
        if all(path.exists() for path in paths):
            questions.append(_question("graphic", "klein4b", name, find_render(renders / "klein4b" / "white", name), pair, paths, rng))
    rng.shuffle(questions)
    # Plain numbered ids: the page sees them, so they must not name a method.
    return [replace(question, id=f"q{index:03d}") for index, question in enumerate(questions, start=1)]


def _question(kind: str, model: str, name: str, original: Path, pair: tuple[str, str], paths: list[Path], rng: random.Random) -> Question:
    order = [0, 1]
    rng.shuffle(order)
    first, second = order
    return Question("", kind, name, model, str(original), pair[first], str(paths[first]), pair[second], str(paths[second]))


def load_plan(runs: Path) -> list[Question]:
    """Return the survey plan, building and saving it on first use so a restart keeps the same order and sides."""
    path = runs / "survey" / "plan.json"
    if path.exists():
        return [Question(**item) for item in json.loads(path.read_text(encoding="utf-8"))]
    plan = build_plan(runs, random.Random(_SEED))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(question) for question in plan], indent=1), encoding="utf-8")
    return plan


def answered(runs: Path) -> dict[str, str]:
    """Return the latest choice per question id."""
    path = runs / "survey" / "answers.jsonl"
    if not path.exists():
        return {}
    choices: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        choices[record["id"]] = record["choice"]
    return choices


def _handler(runs: Path, plan: list[Question]) -> type[BaseHTTPRequestHandler]:
    by_id = {question.id: question for question in plan}
    public = [{"id": question.id, "kind": question.kind} for question in plan]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

        def _send(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                self._send(200, _PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/plan":
                self._send(200, json.dumps({"questions": public, "answers": answered(runs)}).encode(), "application/json")
            elif self.path.startswith("/img/"):
                _, _, ident, side = self.path.split("/", 3)
                question = by_id.get(ident)
                path = {"o": question.original, "a": question.a_path, "b": question.b_path}.get(side) if question else None
                if path is None:
                    self._send(404, b"not found", "text/plain")
                else:
                    self._send(200, Path(path).read_bytes(), "image/png")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/answer":
                self._send(404, b"not found", "text/plain")
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            question, choice = by_id.get(body.get("id")), body.get("choice")
            if question is None or choice not in ("a", "b", "equal", "bad"):
                self._send(400, b"bad answer", "text/plain")
                return
            record = {
                "id": question.id,
                "choice": choice,
                "kind": question.kind,
                "model": question.model,
                "subject": question.subject,
                "a": question.a_method,
                "b": question.b_method,
                "time": time.time(),
            }
            with (runs / "survey" / "answers.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(record) + "\n")
            self._send(200, b"{}", "application/json")

    return Handler


def main() -> None:
    """Serve the survey on all interfaces."""
    parser = argparse.ArgumentParser(prog="python -m bench.transparent.survey", description="Blind A/B survey of the bench mattes.")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args()
    plan = load_plan(args.runs_dir)
    print(f"{len(plan)} questions; serving on port {args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), _handler(args.runs_dir, plan)).serve_forever()


if __name__ == "__main__":
    main()
