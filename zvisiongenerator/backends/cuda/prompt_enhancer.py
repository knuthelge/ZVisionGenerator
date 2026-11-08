"""Run the prompt-enhancer LLM with transformers (CUDA 4-bit via bitsandbytes, else bf16 on CPU)."""

from __future__ import annotations

import threading
import warnings
from collections.abc import Callable, Iterator
from typing import Any

from zvisiongenerator.utils.config import model_reference

_END_TOKEN = "<|im_end|>"
# Subtracted once from the logit of every token already generated in this request; stops songs from repeating a verse.
# transformers runs custom processors before its temperature and top-k warpers.
_PRESENCE_PENALTY = 1.5
# Generation-config keys that leave decoding at transformers' defaults (the app passes temperature and do_sample itself).
_NEUTRAL_GENERATION_KEYS = frozenset({"eos_token_id", "bos_token_id", "pad_token_id", "use_cache", "temperature", "do_sample", "transformers_version", "_from_model_config"})


def _uses_graph_decode(model_type: str | None, repo_quantized: bool, generation_config: Any) -> bool:
    """Return whether the CUDA-graph decode loop applies to this model.

    The loop uses the same sampling as *generate*, though seeded texts differ. It applies only to the measured setup: a
    Qwen3.5 text model whose repo carries no quantization of its own and whose generation config sets no sampling fields.
    Other families can break under capture.
    """
    return model_type == "qwen3_5_text" and not repo_quantized and set(generation_config.to_diff_dict()) <= _NEUTRAL_GENERATION_KEYS


def _make_stopping_criteria(cancelled: Callable[[], bool] | None) -> Any:
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _CancelCriteria(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            return bool(cancelled is not None and cancelled())

    return StoppingCriteriaList([_CancelCriteria()])


def _make_presence_penalty(prompt_len: int) -> Any:
    """Return a logits processor list that penalises each token generated after the first *prompt_len* ids."""
    from transformers import LogitsProcessor, LogitsProcessorList

    class _PresencePenalty(LogitsProcessor):
        def __call__(self, input_ids: Any, scores: Any) -> Any:
            generated = input_ids[:, prompt_len:]
            if generated.shape[1] == 0:
                return scores
            seen = scores.new_zeros(scores.shape)
            seen.scatter_(1, generated, 1.0)
            return scores - _PRESENCE_PENALTY * seen

    return LogitsProcessorList([_PresencePenalty()])


def _make_text_streamer(tokenizer: Any) -> Any:
    """Return a ``TextStreamer`` that collects finalized text in its ``texts`` list instead of printing it."""
    from transformers import TextStreamer

    class _CollectingStreamer(TextStreamer):
        def __init__(self) -> None:
            super().__init__(tokenizer, skip_special_tokens=True)
            self.texts: list[str] = []

        def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
            if text:
                self.texts.append(text)

    return _CollectingStreamer()


class TransformersPromptEnhancer:
    """A loaded transformers causal LM that streams chat completions with thinking disabled."""

    def __init__(self, repo: str, revision: str | None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from zvisiongenerator.backends.cuda.driver import cuda_driver_hint
        from zvisiongenerator.backends.cuda.memory import configure_allocator

        # Preflight enhancement can be the first thing in the process to start CUDA.
        configure_allocator()
        self.repo = repo
        self.revision = revision
        self.on_cuda = torch.cuda.is_available()
        hint = cuda_driver_hint(self.on_cuda, getattr(torch.version, "cuda", None))
        if hint:
            warnings.warn(f"The prompt enhancer runs on the CPU. {hint}", stacklevel=2)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
            kwargs: dict[str, Any] = {"revision": revision, "dtype": torch.bfloat16}
            if self.on_cuda:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
                kwargs["device_map"] = "cuda"
            self._model = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
        except Exception as exc:  # noqa: BLE001 - surface any loader failure with the model name
            raise RuntimeError(f"Could not load prompt enhancer model {model_reference(repo, revision)}: {exc}") from exc
        self._model.eval()
        self._graph_decode = self.on_cuda and _uses_graph_decode(self._model.config.model_type, self._model.hf_quantizer.pre_quantized, self._model.generation_config)
        end_id = self._tokenizer.convert_tokens_to_ids(_END_TOKEN)
        if end_id == self._tokenizer.unk_token_id:
            end_id = None
        self._eos_ids = [token for token in {end_id, self._tokenizer.eos_token_id} if isinstance(token, int) and token >= 0]

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
        import torch

        if self._model is None:
            raise RuntimeError("The prompt enhancer model has been released.")
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        # The chat template already contains any BOS token; adding special tokens again would duplicate it.
        inputs = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self._model.device)
        torch.manual_seed(seed)
        if self._graph_decode:
            yield from self._stream_graph(inputs["input_ids"], max_tokens=max_tokens, temperature=temperature, cancelled=cancelled)
        else:
            yield from self._stream_generate(inputs, max_tokens=max_tokens, temperature=temperature, cancelled=cancelled)

    def _stream_graph(self, input_ids: Any, *, max_tokens: int, temperature: float, cancelled: Callable[[], bool] | None) -> Iterator[str]:
        """Yield text deltas decoded one CUDA-graph replay per token, on the caller's thread."""
        import torch

        from zvisiongenerator.backends.cuda.enhancer_decode import graph_tokens

        streamer = _make_text_streamer(self._tokenizer)

        try:
            for token in graph_tokens(self._model, input_ids, max_tokens=max_tokens, temperature=temperature, penalty=_PRESENCE_PENALTY):
                if token in self._eos_ids:
                    break
                streamer.put(torch.tensor([token]))
                yield from streamer.texts
                streamer.texts.clear()
                if cancelled is not None and cancelled():
                    break
            streamer.end()
            yield from streamer.texts
            streamer.texts.clear()
        except Exception as exc:  # noqa: BLE001 - one clear error for any failure on the graph path
            raise RuntimeError(f"Prompt enhancement failed: {exc}") from exc

    def _stream_generate(self, inputs: Any, *, max_tokens: int, temperature: float, cancelled: Callable[[], bool] | None) -> Iterator[str]:
        """Yield text deltas from transformers ``generate`` running on a worker thread."""
        import torch
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        errors: list[BaseException] = []

        def _run() -> None:
            try:
                with torch.inference_mode():
                    self._model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        # Temperature 0 means greedy (as on MLX); transformers rejects sampling at 0.
                        **({"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}),
                        eos_token_id=self._eos_ids or None,
                        pad_token_id=self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else (self._eos_ids[0] if self._eos_ids else None),
                        streamer=streamer,
                        stopping_criteria=_make_stopping_criteria(cancelled),
                        logits_processor=_make_presence_penalty(inputs["input_ids"].shape[1]),
                    )
            except BaseException as exc:  # noqa: BLE001 - re-raised on the consumer thread
                errors.append(exc)
                streamer.end()

        worker = threading.Thread(target=_run, name="ziv-prompt-enhancer", daemon=True)
        worker.start()
        for text in streamer:
            if text:
                yield text
        worker.join()
        if errors:
            raise RuntimeError(f"Prompt enhancement failed: {errors[0]}") from errors[0]

    def close(self) -> None:
        """Drop the weights and free CUDA cache."""
        from zvisiongenerator.backends.cuda.memory import release_memory

        self._model = None
        self._tokenizer = None
        release_memory()
