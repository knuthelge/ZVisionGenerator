"""Tests for the prompt-enhancer bench: cases, checks, the runner with a fake model, and the compare report."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from bench.enhance import __main__ as bench_cli
from bench.enhance.cases import case_seeds, load_cases, parse_case, select_cases
from bench.enhance.checks import CHECK_NAMES, case_score, caption_format, count_kept, forbid_clear, keep_share, length_check, length_plan, quotes_kept
from bench.enhance.report import check_rates, compare_runs, load_run
from bench.enhance.runner import META_FILE, RESULTS_FILE, RecordingEnhancer, run_bench, run_case
from zvisiongenerator.utils.prompt_enhance import DEFAULT_LENGTH_CONFIG, LengthPlan

_OPTIONS = {"temperature": 0.7, "max_new_tokens": 700, "idle_release_seconds": 0.0, "length": DEFAULT_LENGTH_CONFIG}
_CONFIG: dict = {}


class _FakeEnhancer:
    """Yields the next scripted reply per call, word by word."""

    def __init__(self, replies: list[str]) -> None:
        self.repo = "fake/model"
        self.revision = None
        self._replies = list(replies)
        self.closed = False

    def generate(self, messages, *, seed, max_tokens, temperature, cancelled: Callable[[], bool] | None = None) -> Iterator[str]:
        reply = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
        for word in reply.split(" "):
            yield word + " "

    def close(self) -> None:
        self.closed = True


def _case(**overrides):
    entry = {"id": "c1", "tags": ["t"], "prompt": "a fox in snow", "settings": {}}
    entry.update(overrides)
    return parse_case(entry)


class TestCases:
    def test_shipped_cases_load(self):
        cases = load_cases()
        assert len(cases) >= 50
        assert {case.mode for case in cases} == {"image", "video", "caption"}

    def test_seeds_are_stable_and_at_least_eight_digits(self):
        seeds = case_seeds("short-default", 3)
        assert seeds == case_seeds("short-default", 3)
        assert all(seed >= 10_000_000 for seed in seeds)
        assert seeds[1] == seeds[0] + 1
        assert case_seeds("other", 1) != seeds[:1]

    @pytest.mark.parametrize(
        "entry",
        [
            {"id": "x", "prompt": "a", "mode": "audio"},
            {"id": "x", "prompt": "a", "settings": {"style": "nope"}},
            {"id": "x", "prompt": "a", "scene": "b"},
            {"id": "x", "prompt": "", "settings": {}},
            {"id": "x", "prompt": "a", "colour": "red"},
            {"id": "x", "prompt": "a", "context": {"speed": 2}},
            {"id": "x", "prompt": "a", "settings": {"details": []}},
        ],
    )
    def test_rejects_malformed_case(self, entry):
        with pytest.raises(ValueError, match="x"):
            parse_case(entry)

    def test_rejects_duplicate_ids(self, tmp_path):
        path = tmp_path / "cases.yaml"
        path.write_text("cases:\n  - {id: a, prompt: x}\n  - {id: a, prompt: y}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Duplicate"):
            load_cases(path)

    def test_select_by_tag_and_id(self):
        cases = (_case(id="a", tags=["x"]), _case(id="b", tags=["y"]))
        assert [case.id for case in select_cases(cases, tags={"y"})] == ["b"]
        assert [case.id for case in select_cases(cases, ids={"a"})] == ["a"]
        assert len(select_cases(cases)) == 2


class TestChecks:
    def test_length_check(self):
        plan = LengthPlan(length="same", target=20, lo=17, hi=23, clamped=False)
        assert length_check(20, plan) == (True, 1.0)
        assert length_check(30, plan) == (False, 1.5)
        assert length_check(5, None) == (None, None)

    def test_quotes_kept(self):
        source = 'a diner with a sign that says "OPEN 24 HOURS"'
        assert quotes_kept(source, 'A diner glows; its sign reads "OPEN  24 HOURS".') is True
        assert quotes_kept(source, 'A diner with a sign reading "Open 24 Hours".') is False
        assert quotes_kept("a fox in snow", "anything") is None

    def test_keep_share_allows_case_and_plurals(self):
        assert keep_share("Two Foxes rest in deep snow.", ("fox", "snow", "owl")) == pytest.approx(2 / 3, abs=0.001)
        assert keep_share("text", ()) is None

    def test_count_kept_as_word_or_digit(self):
        assert count_kept("Three cats sleep.", "three cats") is True
        assert count_kept("3 cats sleep.", "three cats") is True
        assert count_kept("Two cats sleep.", "three cats") is False
        assert count_kept("x", None) is None

    def test_forbid_matches_whole_words(self):
        assert forbid_clear("A woman plays chess.", ("man",)) is True
        assert forbid_clear("A man plays chess.", ("man",)) is False
        assert forbid_clear("x", ()) is None

    def test_caption_format(self):
        elements = [{"type": "obj", "desc": f"thing {index}"} for index in range(3)]
        caption = {"high_level_description": "A fox.", "compositional_deconstruction": {"background": "snow", "elements": elements}}
        assert caption_format(json.dumps(caption)) is True
        caption["compositional_deconstruction"]["elements"] = elements[:2]
        assert caption_format(json.dumps(caption)) is False
        assert caption_format("not json") is False

    def test_case_score(self):
        checks = {name: None for name in CHECK_NAMES} | {"usable": True, "first_try": False, "keep": 0.5}
        assert case_score(checks) == 0.5
        assert case_score({"usable": False}) == 0.0

    def test_length_plan_per_mode(self):
        assert length_plan(_case(), ceiling=300, length_cfg=None)[1] == "prompt"
        assert length_plan(_case(mode="caption"), ceiling=300, length_cfg=None) == (None, "caption")
        action = _case(mode="video", scene="A cat on a counter.", prompt="it jumps", context={"reference_image": True})
        assert length_plan(action, ceiling=300, length_cfg=None)[1] == "action"
        pair = _case(mode="video", prompt="a cat jumps off a counter")
        assert length_plan(pair, ceiling=300, length_cfg=None)[1] == "pair"


class TestRunner:
    def test_recording_enhancer_records_each_call(self):
        ticks = iter(range(100))
        recorder = RecordingEnhancer(_FakeEnhancer(["one two three"]), clock=lambda: float(next(ticks)))
        text = "".join(recorder.generate([{"role": "user", "content": "x"}], seed=1, max_tokens=10, temperature=0.5))
        assert text.split() == ["one", "two", "three"]
        assert len(recorder.calls) == 1
        assert recorder.calls[0]["words"] == 3
        assert recorder.calls[0]["first_text_seconds"] is not None

    def test_run_case_image_scores_output(self):
        recorder = RecordingEnhancer(_FakeEnhancer(["A red fox curls up in deep fresh snow under soft morning light, framed low and wide."]))
        record = run_case(recorder, _case(keep=["fox", "snow"]), 12345678, config=_CONFIG, options=_OPTIONS)
        assert record["error"] is None
        assert record["checks"]["usable"] is True
        assert record["checks"]["first_try"] is True
        assert record["checks"]["keep"] == 1.0
        assert record["system_prompt"]
        assert record["plan"]["measures"] == "prompt"

    def test_run_case_records_retry_and_failure(self):
        recorder = RecordingEnhancer(_FakeEnhancer(["a fox in snow"]))
        record = run_case(recorder, _case(), 12345678, config=_CONFIG, options=_OPTIONS)
        assert record["error"]
        assert record["checks"]["usable"] is False
        assert len(record["attempts"]) == 2

    def test_run_case_video_pair(self):
        reply = "SCENE: A busy kitchen with steel counters, warm light.\nACTION: The chef flips a pancake high and laughs as it lands."
        case = _case(mode="video", prompt="a chef flips a pancake and laughs", context={"audio": True, "frames": 97, "fps": 24}, keep=["chef", "kitchen"])
        record = run_case(RecordingEnhancer(_FakeEnhancer([reply])), case, 12345678, config=_CONFIG, options=_OPTIONS)
        assert record["scene"].startswith("A busy kitchen")
        assert record["checks"]["format"] is True
        assert record["checks"]["keep"] == 1.0

    def test_run_bench_writes_results_and_meta(self, tmp_path):
        fake = _FakeEnhancer(["A red fox curls up in deep fresh snow under soft morning light."])
        out = tmp_path / "base"
        path = run_bench(
            (_case(),), factory=lambda repo, revision: fake, repo="fake/model", revision=None, model_choice="fast", config=_CONFIG, options=_OPTIONS, seeds=2, out_dir=out, echo=lambda _line: None
        )
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert {json.loads(line)["seed"] for line in lines} == set(case_seeds("c1", 2))
        assert json.loads((out / META_FILE).read_text(encoding="utf-8"))["repo"] == "fake/model"
        assert fake.closed
        with pytest.raises(FileExistsError):
            run_bench((_case(),), factory=lambda repo, revision: fake, repo="r", revision=None, model_choice="fast", config=_CONFIG, options=_OPTIONS, seeds=1, out_dir=out, echo=lambda _line: None)


def _write_run(run_dir: Path, reply: str) -> None:
    run_bench(
        (_case(keep=["fox"]),),
        factory=lambda repo, revision: _FakeEnhancer([reply]),
        repo="fake/model",
        revision=None,
        model_choice="fast",
        config=_CONFIG,
        options=_OPTIONS,
        seeds=1,
        out_dir=run_dir,
        echo=lambda _line: None,
    )


class TestCompare:
    def test_compare_reports_checks_and_changed_cases(self, tmp_path):
        _write_run(tmp_path / "base", "A red fox curls up in deep fresh snow.")
        _write_run(tmp_path / "new", "A quiet white field under falling snow at dawn.")
        base, new = load_run(tmp_path / "base"), load_run(tmp_path / "new")
        assert check_rates(base.records)["keep"] == (1.0, 1)
        assert check_rates(new.records)["keep"] == (0.0, 1)
        assert "c1" in compare_runs(base, new)

    def test_cli_compare_writes_report(self, tmp_path):
        _write_run(tmp_path / "base", "A red fox curls up in deep fresh snow.")
        _write_run(tmp_path / "new", "A red fox sleeps in deep fresh snow.")
        assert bench_cli.main(["compare", "base", "new", "--runs-dir", str(tmp_path)]) == 0
        assert (tmp_path / "compare-base-new.md").exists()

    def test_load_run_requires_results(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=RESULTS_FILE):
            load_run(tmp_path)
