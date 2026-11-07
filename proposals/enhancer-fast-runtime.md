# A faster prompt enhancer on Windows and Linux

**Status:** Proposed (2026-11-07)

## Problem

Every helper that writes with the prompt enhancer waits about 10 to 15 seconds on Windows and Linux with the Fast model, and longer with Smart. On macOS the same calls feel instant: mlx-lm decodes the 4-bit model fast enough that the owner calls it "extremely fast". The wait is the first thing a Windows or Linux user meets in every box that has a Write button, and it grows as more prompt boxes get helpers.

The goal is a typical writer call (about 300 tokens) in at most about 5 s on Fast and about 8 s on Smart, which is roughly 3x today's decode speed, with no stall on the first call and none when a prompt has a new length.

## Evidence

Measured on this branch on an RTX 3080 10 GB under Linux, **on the Fast model only** (`coder3101/Qwen3.5-4B-heretic`). Smart (`coder3101/Qwen3.5-9B-heretic`) has no eager baseline yet. Both run bitsandbytes NF4 through transformers `generate`, thinking off.

- **Speed.** About 16 to 21 tok/s, with about 24% run-to-run noise.
- **Where the time goes.** About 14 ms of GPU work per token against about 63 ms of wall time: about 2,700 kernel launches per token. The cost is Python and eager launch overhead, not arithmetic. Gated DeltaNet's small eager ops are about half the CPU time; bitsandbytes `Linear4bit` is about 26% (248 calls per token at about 105 µs of CPU each). With no launch overhead at all, Fast tops out near 70 tok/s.
- **Rejected already.** flash-linear-attention plus causal-conv1d gives about 1.1x, and causal-conv1d has no wheel for Python 3.14 with torch 2.14 or for Windows (a 12-minute, 212 MB source build). torchao int4 in place of NF4 gives no gain, which confirms overhead rather than dequantization.
- **CUDA graphs work for decode, not for the whole call.** A static cache with `torch.compile(model.forward, backend="cudagraphs", fullgraph=True)` traces with no graph breaks and decodes at about 32 to 36 tok/s (1.8 to 1.9x). It recompiles the prefill for most new prompt lengths (100 to 140 s), and prefill cannot be captured because the static cache's `cumulative_length` lives on the CPU. `mode="reduce-overhead"` (Inductor) died silently in one try. Greedy output drifts slightly from eager while reading the same. Why decode stops at 1.8x when the ceiling is 3.5x is not known.
- **A constraint on any runtime.** The enhancer applies a presence penalty of 1.5 to generated tokens only, on both backends (`_PRESENCE_PENALTY` in `backends/cuda/prompt_enhancer.py` and `backends/mac/prompt_enhancer.py`).

## Options checked

Checked on 2026-11-07. Speeds other than the measured ones are expectations.

| Option | Expected speed | Windows/Linux parity | Install burden | Extra model files | Presence penalty | Risk |
|---|---|---|---|---|---|---|
| **A. Compile only the decode step** (transformers `compile_config`, static cache, prefill eager) | At most about 3.5x; 1.8 to 1.9x seen | Same code, no install | None | None | Existing processor, outside the graph | **Likely fails the gate**: the ceiling is barely 3x, and compiled generation already stopped at 1.8x |
| **B. llama.cpp `llama-server`** (official release binary, GGUF Q4_K_M) | Far above 3x expected: 2.7 GB of weights, no Python launches, fused CUDA Gated DeltaNet in mainline | Official `win-cuda` and `ubuntu-cuda` builds from one release tag | One pinned binary (170 to 290 MB, plus 390 to 590 MB of CUDA runtime if the system has none); a subprocess to start, stop and health-check | GGUF Q4_K_M, 2.7 GB Fast and 5.6 GB Smart, in place of the 9 and 19 GB bf16 | `presence_penalty` per request; history may include prompt tokens (probe first) | Single-maintainer GGUFs; penalty semantics; CUDA 12.4 or 13.5 build must match the driver |
| B2. llama.cpp Vulkan build | Slower than CUDA on an NVIDIA card; measure inside B | Official `win-vulkan` and `ubuntu-vulkan` builds | Same binary, but no CUDA runtime download | Same GGUFs | Same | Only worth it if the CUDA runtime bundle proves too heavy |
| C. ExLlamaV3 (EXL3, in process) | Fast, with its own Triton Gated DeltaNet; unmeasured | Wheels for Windows and Linux with Python 3.14, but only up to torch 2.13; the project is on 2.14.1 | Source build against torch 2.14, or a second torch; `triton-windows` on Windows | EXL3 per model; a 4B exists for the pinned weights, no pinned 9B | `SS_PresFreqP` sampler step | Torch pin mismatch is the install pain that sank causal-conv1d |
| ONNX Runtime GenAI | Unknown | Qwen3.5 Gated DeltaNet ops are landing (CUDA kernel and model-builder PRs); Windows load of Qwen3.5-9B fails with `CausalConvWithState not registered` (Foundry Local #700, open) | An ONNX export per model | Yes | Not checked | Rejected for now: the Windows failure is the case we need |
| TensorRT-LLM | High | Official support is Linux and containers; Windows is a community source build for RTX 40; Qwen3.5 needed a newer transformers than its container had | Heavy engine build per GPU | Engine per model | Yes | Rejected: breaks parity, engine per card |
| MLC-LLM | Unknown | No shipped Qwen3.5 (Gated DeltaNet needs TVM support; only a WebLLM request found) | Compile per model | Yes | Not checked | Rejected: no Qwen3.5 |
| Ollama | As B | Windows and Linux | A separate service the user installs and keeps running | Its own model store | Per request | Rejected: llama.cpp plus a service we do not control |
| llama-cpp-python | As B | PyPI has only the 0.3.36 sdist; its CUDA wheel index covers Python 3.10 to 3.12 | A compiler and the CUDA toolkit on every machine | As B | As B | Rejected: no Python 3.14 wheel |
| vLLM, SGLang | High | vLLM has no official Windows build (community wheels, Triton under MSVC failures on Qwen3.5); SGLang is Linux and WSL2 only | Their own torch pins | None | Yes | Rejected: breaks parity |

Notes:

- **GGUFs exist for the pinned models.** `FadedRedStar/Qwen3.5-4B-heretic-GGUF` and `FadedRedStar/Qwen3.5-9B-heretic-GGUF` list `coder3101/...` as base, are Apache-2.0, and hold Q4_K_M (2.7 and 5.6 GB), Q5_K_M, Q8_0, vision projectors and `mtp-*.gguf` heads. They were uploaded on 10 July 2026; the pinned revisions (`config.yaml`, `prompt_enhancer`) are the last commits of the `coder3101` repos, from 4 March, so no later weights exist to differ from. The card names the base repo but not a revision, so spike B still confirms by running the bench. The card advertises a separate imatrix repo as better at 4 bits; use it if it checks out. The project hosts nothing of its own, at the cost of the single-maintainer risk [prequantized-model-copies.md](prequantized-model-copies.md) already names.
- **MTP.** llama.cpp merged multi-token-prediction speculative decoding in May 2026, and the GGUF repos ship the heads (0.7 GB Fast, 2.2 GB Smart at Q8_0). It could stack on B later; it is not part of the first step.
- llama.cpp supports Qwen3.5 with fused chunked Gated DeltaNet CUDA kernels since March 2026, still being tuned. Releases in the last weeks: llama.cpp b12189 (7 Nov) with CUDA 12.4 and 13.5 builds for Windows and 12.8 and 13.5 for Linux; ExLlamaV3 1.6.3 to 1.6.6 (17 to 31 Oct) with cp314 wheels, none for torch 2.14. Licences: llama.cpp, ExLlamaV3 and llama-cpp-python MIT; the models Apache-2.0.

## Proposed change

Spikes on the 10 GB card, then one decision. Nothing ships from a spike, and no setting is added.

1. **Measure the 9B eager baseline.** Decode tok/s and the wall time of a 300-token call for Smart, the same way Fast was measured.
2. **Probe llama-server's presence penalty.** One request with `presence_penalty: 1.5` and a prompt whose words are the only candidates to repeat, to see whether the penalty history includes prompt tokens. If it does, prompt words are penalised and quality may suffer; the fallback is penalty off, and B then has to pass the gate that way or fail. There is no advice to widen `repeat_last_n`, which only adds prompt tokens.
3. **Spike B, one day.** Download the two GGUFs, run the official CUDA `llama-server` on the same prompts through the chat endpoint (`chat_template_kwargs: {"enable_thinking": false}`), and read the stream the way the enhancer reads transformers' streamer today. Measure the Vulkan build within it.
4. **Spike A, half a day, last.** Start from transformers' built-in decode-only compile (`compile_config` with the static cache), before any hand-rolled graph; prefill stays eager so no prompt length compiles anything. Expect a fail.

If B passes, it replaces the transformers path on Windows and Linux in one change, and the enhancer's NF4 snapshots in [prequantized-model-copies.md](prequantized-model-copies.md) are dropped in favour of the GGUFs. If only A passes, ship A. macOS stays on mlx-lm. Backend selection stays in `backends/__init__.py`.

## Measurement plan

The same 20 enhancer-bench prompts per model, Fast and Smart, current code against each spike, interleaved, three runs each, every GPU run through the GPU queue.

- Wall time of a typical 300-token writer call, decode tok/s, time to first token and VRAM peak.
- The first call after load and a call with an unseen prompt length.
- `make enhance-bench` checks and compare, plus a count of copied verses.
- B only: server start and stop time (the 120 s idle release must end the process to free the VRAM), and one launch on the owner's Windows machine with the official zip.

**Gate.** A spike passes when a typical 300-token writer call takes at most about 5 s on Fast and about 8 s on Smart, the first call and a new prompt length add no stall beyond the model load, output quality on the enhancer bench is no worse than eager, and copied verses stay at about 0. Decode of at least 3x the eager median is a secondary check. For B the Windows launch must also pass, because parity is a condition, not a later phase.

## Testing

- The enhancer session keeps its contract (streaming text, cancel, idle release); only the runtime under it changes.
- For B: a `llama-server` command builder and stream parser tested with a faked process and canned server-sent events, including cancel and an early exit; the binary release tag and GGUF repos read from config.
- No real models; the enhancer bench is the only place a real model loads.

## Open questions

- Which CUDA build a user's driver needs; the existing driver hint could pick it.
- Whether the binary downloads through the same plan and licence dialog as models.
- Without a GPU the enhancer runs bf16 on CPU today; llama.cpp would do that better, but it is a separate decision.

## Alternatives considered

- **Keep the transformers path and accept 10 to 15 s.** Free, but every helper box carries it, and macOS shows the gap.
- **More kernel packages for the eager path.** flash-linear-attention and causal-conv1d gave 1.1x, and one cannot be installed on the project's Python and torch.
- **ExLlamaV3.** The fallback if A and B both fail, as a torch 2.14 build.

## Sources

- [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) and [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [`FadedRedStar/Qwen3.5-4B-heretic-GGUF`](https://huggingface.co/FadedRedStar/Qwen3.5-4B-heretic-GGUF) and [9B](https://huggingface.co/FadedRedStar/Qwen3.5-9B-heretic-GGUF)
- [llama-cpp-python on PyPI](https://pypi.org/project/llama-cpp-python/) and its [wheel index](https://abetlen.github.io/llama-cpp-python/whl/cu124/llama-cpp-python/)
- [ExLlamaV3 releases](https://github.com/turboderp-org/exllamav3/releases) and [`DBMe/Qwen3.5-4B-heretic-exl3`](https://huggingface.co/DBMe/Qwen3.5-4B-heretic-exl3)
- [Foundry Local #700](https://github.com/microsoft/Foundry-Local/issues/700)
- [TensorRT-LLM #12321](https://github.com/NVIDIA/TensorRT-LLM/issues/12321) and [WebLLM #778](https://github.com/mlc-ai/web-llm/issues/778)
- [Community Windows vLLM build](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.17.1-win)
- [llama.cpp MTP merge benchmarks](https://bayesiansapien.github.io/cere-bro/inference-efficiency/2026-05-17-mtp-llama-cpp-merge-strix-halo-benchmarks)
