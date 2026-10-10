"""Run the prompt-enhancer LLM with transformers (CUDA 4-bit via bitsandbytes, else bf16 on CPU)."""

from __future__ import annotations

import threading
import warnings
from collections.abc import Callable, Iterator
from typing import Any

from zvisiongenerator.utils.config import model_reference

_END_TOKEN = "<|im_end|>"


def _make_stopping_criteria(cancelled: Callable[[], bool] | None) -> Any:
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _CancelCriteria(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            return bool(cancelled is not None and cancelled())

    return StoppingCriteriaList([_CancelCriteria()])


class TransformersPromptEnhancer:
    """A loaded transformers causal LM that streams chat completions with thinking disabled."""

    def __init__(self, repo: str, revision: str | None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from zvisiongenerator.backends.cuda_driver import cuda_driver_hint
        from zvisiongenerator.backends.memory_cuda import configure_allocator

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
        from transformers import TextIteratorStreamer

        if self._model is None:
            raise RuntimeError("The prompt enhancer model has been released.")
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        # The chat template already contains any BOS token; adding special tokens again would duplicate it.
        inputs = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self._model.device)
        torch.manual_seed(seed)
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
        from zvisiongenerator.backends.memory_cuda import release_memory

        self._model = None
        self._tokenizer = None
        release_memory()
