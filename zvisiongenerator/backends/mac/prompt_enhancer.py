"""Run the prompt-enhancer LLM on Apple Silicon with mlx-lm."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from typing import Any

from zvisiongenerator.utils.config import model_reference

_UNSUPPORTED_TYPE_RE = re.compile(r"Model type (\S+) not supported")
_END_TOKEN = "<|im_end|>"
# Subtracted once from the logit of every token already generated in this request; stops songs from repeating a verse.
_PRESENCE_PENALTY = 1.5


def _presence_penalty_processor(penalty: float) -> Callable[[Any, Any], Any]:
    """Return an mlx-lm logits processor that subtracts *penalty* once from each token generated so far.

    It runs on the raw logits before mlx-lm's sampler applies temperature, as the CUDA processor does.

    mlx-lm's own presence penalty looks at a sliding window that includes the prompt, so it cannot be used here. The
    processor sees the whole token history on every call; its first call holds only the prompt and fixes where
    generated tokens begin.
    """
    prompt_len: int | None = None

    def processor(tokens: Any, logits: Any) -> Any:
        nonlocal prompt_len
        import mlx.core as mx

        if prompt_len is None:
            prompt_len = len(tokens)
            return logits
        generated = tokens[prompt_len:]
        # Counting then clamping to 1 penalises a repeated token once, not once per occurrence.
        counts = mx.zeros((logits.shape[-1],), dtype=logits.dtype).at[generated].add(1)
        return logits - penalty * mx.minimum(counts, 1)

    return processor


def _load(repo: str, revision: str | None) -> tuple[Any, Any]:
    """Load *repo* with mlx-lm, retrying once for ``*_text`` model types mlx-lm does not register."""
    from mlx_lm import load

    try:
        return load(repo, revision=revision)
    except ValueError as exc:
        match = _UNSUPPORTED_TYPE_RE.search(str(exc))
        if match is None or not match.group(1).endswith("_text"):
            raise RuntimeError(f"Could not load prompt enhancer model {model_reference(repo, revision)}: {exc}") from exc
        base_type = match.group(1).removesuffix("_text")
        try:
            return load(repo, revision=revision, model_config={"model_type": base_type})
        except Exception as retry_exc:  # noqa: BLE001 - surface any loader failure with the model name
            raise RuntimeError(f"Could not load prompt enhancer model {model_reference(repo, revision)} (also tried model type '{base_type}'): {retry_exc}") from retry_exc
    except Exception as exc:  # noqa: BLE001 - surface any loader failure with the model name
        raise RuntimeError(f"Could not load prompt enhancer model {model_reference(repo, revision)}: {exc}") from exc


class MlxPromptEnhancer:
    """A loaded mlx-lm chat model that streams completions with thinking disabled."""

    def __init__(self, repo: str, revision: str | None) -> None:
        self.repo = repo
        self.revision = revision
        self._model, self._tokenizer = _load(repo, revision)
        # ChatML models (Qwen) end turns with <|im_end|>; other tokenizers map it to unk (or nothing), and
        # registering unk as a stop token would cut rewrites short.
        end_id = self._tokenizer.convert_tokens_to_ids(_END_TOKEN)
        if isinstance(end_id, int) and end_id != getattr(self._tokenizer, "unk_token_id", None):
            self._tokenizer.add_eos_token(_END_TOKEN)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        seed: int,
        max_tokens: int,
        temperature: float,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[str]:
        """Yield text deltas for *messages*; stop early when *cancelled* returns True."""
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        from zvisiongenerator.backends.mac.memory import release_memory

        if self._model is None:
            raise RuntimeError("The prompt enhancer model has been released.")
        # A previous image/video pass leaves its buffers in MLX's cache; return them so a tight fit has room to generate.
        release_memory()
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        mx.random.seed(seed)
        for response in stream_generate(
            self._model,
            self._tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
            logits_processors=[_presence_penalty_processor(_PRESENCE_PENALTY)],
        ):
            if cancelled is not None and cancelled():
                return
            if response.text:
                yield response.text

    def close(self) -> None:
        """Drop the weights and return MLX's buffer cache to the system."""
        from zvisiongenerator.backends.mac.memory import release_memory

        self._model = None
        self._tokenizer = None
        release_memory()
