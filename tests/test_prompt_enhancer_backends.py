"""Tests for the prompt-enhancer adapters, session manager, and model resolution."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from zvisiongenerator.backends.prompt_enhancer_session import PromptEnhancerSession, ensure_available
from zvisiongenerator.utils.config import resolve_enhancer_model, split_model_revision


# ── mlx-lm adapter ──────────────────────────────────────────────────────────


def _install_fake_mlx(monkeypatch, *, load_side_effect, chunks=("A ", "fox.")):
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "<chat>"
    tokenizer.convert_tokens_to_ids.return_value = 248046
    tokenizer.unk_token_id = None
    load = MagicMock(side_effect=load_side_effect)
    responses = [types.SimpleNamespace(text=chunk) for chunk in chunks]
    stream_generate = MagicMock(side_effect=lambda *a, **k: iter(responses))
    make_sampler = MagicMock(return_value="sampler")
    mx = types.SimpleNamespace(random=types.SimpleNamespace(seed=MagicMock()))
    mlx_pkg = types.ModuleType("mlx")
    mlx_pkg.core = mx
    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.load = load
    mlx_lm.stream_generate = stream_generate
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = make_sampler
    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    return types.SimpleNamespace(tokenizer=tokenizer, load=load, stream_generate=stream_generate, make_sampler=make_sampler, mx=mx)


class TestMlxAdapter:
    def test_generate_contract(self, monkeypatch):
        from zvisiongenerator.backends.prompt_enhancer_mac import MlxPromptEnhancer

        fake = _install_fake_mlx(monkeypatch, load_side_effect=None)
        fake.load.side_effect = None
        fake.load.return_value = ("model", fake.tokenizer)
        enhancer = MlxPromptEnhancer("owner/repo", "abc123")
        fake.load.assert_called_once_with("owner/repo", revision="abc123")
        fake.tokenizer.add_eos_token.assert_called_once_with("<|im_end|>")

        out = list(enhancer.generate([{"role": "user", "content": "x"}], seed=9, max_tokens=50, temperature=0.4))
        assert out == ["A ", "fox."]
        fake.tokenizer.apply_chat_template.assert_called_once_with([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        fake.mx.random.seed.assert_called_once_with(9)
        fake.make_sampler.assert_called_once_with(temp=0.4)
        assert fake.stream_generate.call_args.kwargs["max_tokens"] == 50

    def test_cancel_stops_stream(self, monkeypatch):
        from zvisiongenerator.backends.prompt_enhancer_mac import MlxPromptEnhancer

        fake = _install_fake_mlx(monkeypatch, load_side_effect=None)
        fake.load.side_effect = None
        fake.load.return_value = ("model", fake.tokenizer)
        enhancer = MlxPromptEnhancer("owner/repo", None)
        assert list(enhancer.generate([], seed=1, max_tokens=5, temperature=0.7, cancelled=lambda: True)) == []

    def test_text_model_type_retry(self, monkeypatch):
        from zvisiongenerator.backends.prompt_enhancer_mac import MlxPromptEnhancer

        fake = _install_fake_mlx(monkeypatch, load_side_effect=None)
        fake.load.side_effect = [ValueError("Model type qwen3_5_text not supported."), ("model", fake.tokenizer)]
        MlxPromptEnhancer("owner/repo", None)
        assert fake.load.call_args_list[1].kwargs == {"revision": None, "model_config": {"model_type": "qwen3_5"}}

    def test_other_value_error_names_repo(self, monkeypatch):
        from zvisiongenerator.backends.prompt_enhancer_mac import MlxPromptEnhancer

        fake = _install_fake_mlx(monkeypatch, load_side_effect=None)
        fake.load.side_effect = ValueError("Model type llama9 not supported.")
        with pytest.raises(RuntimeError, match=r"owner/repo@rev"):
            MlxPromptEnhancer("owner/repo", "rev")
        assert fake.load.call_count == 1

    def test_missing_repo_names_repo(self, monkeypatch):
        from zvisiongenerator.backends.prompt_enhancer_mac import MlxPromptEnhancer

        fake = _install_fake_mlx(monkeypatch, load_side_effect=None)
        fake.load.side_effect = OSError("404")
        with pytest.raises(RuntimeError, match="Could not load prompt enhancer model owner/gone: 404"):
            MlxPromptEnhancer("owner/gone", None)

    def test_released_model_refuses(self, monkeypatch):
        from zvisiongenerator.backends import prompt_enhancer_mac

        fake = _install_fake_mlx(monkeypatch, load_side_effect=None)
        fake.load.side_effect = None
        fake.load.return_value = ("model", fake.tokenizer)
        monkeypatch.setattr("zvisiongenerator.backends.memory_mac.release_memory", lambda: None)
        enhancer = prompt_enhancer_mac.MlxPromptEnhancer("owner/repo", None)
        enhancer.close()
        with pytest.raises(RuntimeError, match="released"):
            list(enhancer.generate([], seed=1, max_tokens=5, temperature=0.7))


class TestTransformersAdapter:
    def test_load_failure_names_repo(self, monkeypatch):
        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")
        from zvisiongenerator.backends.prompt_enhancer_win import TransformersPromptEnhancer

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", MagicMock(side_effect=OSError("offline")))
        with pytest.raises(RuntimeError, match="owner/repo@rev: offline"):
            TransformersPromptEnhancer("owner/repo", "rev")

    def test_cpu_load_has_no_quantization(self, monkeypatch):
        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")
        from zvisiongenerator.backends.prompt_enhancer_win import TransformersPromptEnhancer

        tokenizer = MagicMock()
        tokenizer.convert_tokens_to_ids.return_value = 7
        tokenizer.eos_token_id = 3
        tokenizer.unk_token_id = 0
        model_loader = MagicMock()
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", MagicMock(return_value=tokenizer))
        monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", model_loader)
        enhancer = TransformersPromptEnhancer("owner/repo", None)
        kwargs = model_loader.call_args.kwargs
        assert "quantization_config" not in kwargs and kwargs["revision"] is None
        assert sorted(enhancer._eos_ids) == [3, 7]


# ── Session manager ─────────────────────────────────────────────────────────


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _session(clock=None, downloaded=True):
    created: list[MagicMock] = []
    scheduled: list[tuple[float, object]] = []

    def factory(repo, revision):
        enhancer = MagicMock(repo=repo, revision=revision)
        created.append(enhancer)
        return enhancer

    session = PromptEnhancerSession(factory, clock=clock or _Clock(), scheduler=lambda delay, cb: scheduled.append((delay, cb)), downloaded=lambda r, v: downloaded)
    return session, created, scheduled


class TestSession:
    def test_loads_once_and_reuses(self):
        session, created, _ = _session()
        with session.acquire("a/b", None, idle_seconds=None):
            pass
        with session.acquire("a/b", None, idle_seconds=None) as enhancer:
            assert enhancer is created[0]
        assert len(created) == 1

    def test_busy_only_while_held(self):
        session, _, _ = _session()
        with session.acquire("a/b", None, idle_seconds=None):
            assert session.busy()
        assert not session.busy()

    def test_model_change_releases_previous(self):
        session, created, _ = _session()
        with session.acquire("a/b", None, idle_seconds=None):
            pass
        with session.acquire("a/c", "r1", idle_seconds=None):
            pass
        created[0].close.assert_called_once()
        assert session.resident() == ("a/c", "r1")

    def test_idle_release_with_clock(self):
        clock = _Clock()
        session, created, scheduled = _session(clock)
        with session.acquire("a/b", None, idle_seconds=120):
            pass
        assert scheduled[0][0] == 120
        clock.now = 60
        assert session.release_if_idle(120) is False
        clock.now = 121
        assert session.release_if_idle(120) is True
        created[0].close.assert_called_once()
        assert session.resident() is None

    def test_reuse_resets_idle_clock(self):
        clock = _Clock()
        session, created, _ = _session(clock)
        with session.acquire("a/b", None, idle_seconds=120):
            pass
        clock.now = 100
        with session.acquire("a/b", None, idle_seconds=120):
            pass
        clock.now = 150
        assert session.release_if_idle(120) is False
        created[0].close.assert_not_called()

    def test_zero_idle_releases_immediately(self):
        session, created, _ = _session()
        with session.acquire("a/b", None, idle_seconds=0):
            pass
        created[0].close.assert_called_once()

    def test_release_on_failure_inside_block(self):
        session, created, _ = _session()
        with pytest.raises(RuntimeError):
            with session.acquire("a/b", None, idle_seconds=None):
                raise RuntimeError("job failed")
        assert not session.busy()
        session.release()
        created[0].close.assert_called_once()

    def test_idle_release_skips_while_busy(self):
        clock = _Clock()
        session, created, _ = _session(clock)
        with session.acquire("a/b", None, idle_seconds=None):
            clock.now = 1000
            assert session.release_if_idle(1) is False
        created[0].close.assert_not_called()

    @pytest.mark.parametrize(("downloaded", "phase"), [(True, "loading"), (False, "downloading")])
    def test_phase_reported_before_load(self, downloaded, phase):
        session, _, _ = _session(downloaded=downloaded)
        phases: list[str] = []
        with session.acquire("a/b", None, idle_seconds=None, on_phase=phases.append):
            pass
        with session.acquire("a/b", None, idle_seconds=None, on_phase=phases.append):
            pass
        assert phases == [phase]

    def test_factory_error_propagates_and_clears_busy(self):
        session = PromptEnhancerSession(MagicMock(side_effect=RuntimeError("no model")), scheduler=lambda d, c: None, downloaded=lambda r, v: True)
        with pytest.raises(RuntimeError, match="no model"):
            with session.acquire("a/b", None, idle_seconds=120):
                pass
        assert not session.busy() and session.resident() is None


class TestEnsureAvailable:
    def test_downloaded(self):
        assert ensure_available("a/b", None, downloaded=lambda r, v: True, offline=lambda: True) is True

    def test_online_not_downloaded(self):
        assert ensure_available("a/b", None, downloaded=lambda r, v: False, offline=lambda: False) is False

    def test_offline_not_downloaded(self):
        with pytest.raises(RuntimeError, match="a/b@r is not downloaded and Hugging Face is offline"):
            ensure_available("a/b", "r", downloaded=lambda r, v: False, offline=lambda: True)


# ── Model resolution ────────────────────────────────────────────────────────


_CONFIG = {
    "prompt_enhancer": {
        "model": {"darwin": "mac/model", "win32": "win/model"},
        "revision": {"darwin": "macrev", "win32": "winrev"},
    }
}


class TestResolveModel:
    def test_platform_default(self):
        assert resolve_enhancer_model(_CONFIG, platform_key="darwin") == ("mac/model", "macrev")
        assert resolve_enhancer_model(_CONFIG, platform_key="win32") == ("win/model", "winrev")

    def test_user_override_beats_default(self):
        config = {"prompt_enhancer": {**_CONFIG["prompt_enhancer"], "user_model": "me/mine"}}
        assert resolve_enhancer_model(config, platform_key="darwin") == ("me/mine", None)

    def test_cli_beats_everything(self):
        config = {"prompt_enhancer": {**_CONFIG["prompt_enhancer"], "user_model": "me/mine"}}
        assert resolve_enhancer_model(config, platform_key="darwin", cli_model="cli/model@v2") == ("cli/model", "v2")

    def test_missing_platform(self):
        with pytest.raises(ValueError, match="platform 'linux'"):
            resolve_enhancer_model(_CONFIG, platform_key="linux")

    @pytest.mark.parametrize(("value", "expected"), [("a/b", ("a/b", None)), ("a/b@main", ("a/b", "main")), ("/models/llm", ("/models/llm", None))])
    def test_split(self, value, expected):
        assert split_model_revision(value) == expected

    @pytest.mark.parametrize("value", ["a/b@", "@rev"])
    def test_split_invalid(self, value):
        with pytest.raises(ValueError, match="REPO@REVISION"):
            split_model_revision(value)

    def test_bundled_config_has_pinned_defaults(self):
        from zvisiongenerator.utils.config import load_config

        config = load_config()
        for platform_key in ("darwin", "win32", "linux"):
            repo, revision = resolve_enhancer_model({"prompt_enhancer": {k: v for k, v in config["prompt_enhancer"].items() if not k.startswith("user_")}}, platform_key=platform_key)
            assert repo and revision and len(revision) == 40
        assert config["model_presets"]["flux1"]["enhance_max_words"] == 180


class TestReviewFixes:
    def test_reservation_blocks_and_is_consumed_by_acquire(self):
        session, _created, _ = _session()
        token = session.reserve()
        assert token is not None and session.busy()
        assert session.reserve() is None
        with session.acquire("a/b", None, idle_seconds=None):
            pass
        assert not session.busy()
        session.cancel_reservation(token)  # stale token: no effect
        assert session.reserve() is not None

    def test_cancel_only_clears_own_reservation(self):
        session, _created, _ = _session()
        first = session.reserve()
        session.cancel_reservation(first)
        second = session.reserve()
        session.cancel_reservation(first)
        assert session.busy()
        session.cancel_reservation(second)
        assert not session.busy()

    def test_unknown_im_end_token_is_not_a_stop_token(self, monkeypatch):
        from zvisiongenerator.backends.prompt_enhancer_mac import MlxPromptEnhancer

        fake = _install_fake_mlx(monkeypatch, load_side_effect=None)
        fake.load.side_effect = None
        fake.load.return_value = ("model", fake.tokenizer)
        fake.tokenizer.convert_tokens_to_ids.return_value = 0
        fake.tokenizer.unk_token_id = 0
        MlxPromptEnhancer("meta/llama", None)
        fake.tokenizer.add_eos_token.assert_not_called()

    def test_local_path_with_at_is_not_split(self, tmp_path):
        model_dir = tmp_path / "qwen@4bit"
        model_dir.mkdir()
        assert split_model_revision(str(model_dir)) == (str(model_dir), None)

    def test_user_model_may_carry_revision(self):
        config = {"prompt_enhancer": {"user_model": "Qwen/Qwen3-4B@main", "user_revision": "stale"}}
        # A leftover user_revision from an older config is ignored: the model field carries its own revision.
        assert resolve_enhancer_model(config, platform_key="darwin") == ("Qwen/Qwen3-4B", "main")
        config["prompt_enhancer"]["user_model"] = "Qwen/Qwen3-8B"
        assert resolve_enhancer_model(config, platform_key="darwin") == ("Qwen/Qwen3-8B", None)

    def test_transformers_temperature_zero_is_greedy(self, monkeypatch):
        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")
        from zvisiongenerator.backends.prompt_enhancer_win import TransformersPromptEnhancer

        tokenizer = MagicMock(eos_token_id=3, unk_token_id=0, pad_token_id=None)
        tokenizer.convert_tokens_to_ids.return_value = 7
        tokenizer.apply_chat_template.return_value = "<chat>"
        model = MagicMock()
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", MagicMock(return_value=tokenizer))
        monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", MagicMock(return_value=model))

        class _Streamer:
            def __init__(self, *a, **k):
                self.items = ["x"]

            def __iter__(self):
                return iter(self.items)

            def end(self):
                pass

        monkeypatch.setattr(transformers, "TextIteratorStreamer", _Streamer)
        enhancer = TransformersPromptEnhancer("owner/repo", None)
        list(enhancer.generate([], seed=1, max_tokens=5, temperature=0.0))
        assert tokenizer.call_args.kwargs["add_special_tokens"] is False  # chat template already has any BOS
        kwargs = model.generate.call_args.kwargs
        assert kwargs["do_sample"] is False and "temperature" not in kwargs
        list(enhancer.generate([], seed=1, max_tokens=5, temperature=0.7))
        kwargs = model.generate.call_args.kwargs
        assert kwargs["do_sample"] is True and kwargs["temperature"] == 0.7
