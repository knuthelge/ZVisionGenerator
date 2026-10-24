# Third-Party Licenses

## ltx-core-mlx and ltx-pipelines-mlx

Vendored from [dgrauet/ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx) at commit de4a12d3cd14f345b2ec680eb63f4114b927b435.

MIT License

Copyright (c) 2025 dgrauet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## seedvr2_torch

The SeedVR2 inference core (`packages/seedvr2_torch`, used on Windows and Linux) is vendored from [numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) at commit 4490bd1f482e026674543386bb2a4d176da245b9 (v2.5.24), without its ComfyUI nodes. It is based on [SeedVR2](https://github.com/ByteDance-Seed/SeedVR) by the ByteDance Seed team.

Copyright (c) 2025 ByteDance Ltd. and/or its affiliates; portions Copyright (c) 2023 HuggingFace Team.

Licensed under the Apache License, Version 2.0. The full license text ships with the vendored code as `packages/seedvr2_torch/LICENSE`, and at http://www.apache.org/licenses/LICENSE-2.0. Upstream publishes no NOTICE file.

## minimax_h3_mlx

The MiniMax H3 MLX pipeline (`packages/minimax_h3_mlx`, used on macOS) is vendored from [Argus-AiTeam/minimax-h3-mac](https://github.com/Argus-AiTeam/minimax-h3-mac) at commit 3fcc9f8c557cf353400b1acf737b985e524a509f, without its command-line tools, tests, examples and research notes, and with its `scripts/quantize_text_encoder.py` added as `packages/minimax_h3_mlx/quantize_text_encoder.py`. It builds on the early MLX port [PipeNetwork/minimax-h3-mlx](https://github.com/PipeNetwork/minimax-h3-mlx).

Licensed under the Apache License, Version 2.0. The full license text ships with the vendored code as `packages/minimax_h3_mlx/LICENSE`, and at http://www.apache.org/licenses/LICENSE-2.0. Upstream publishes no NOTICE file and names no copyright holder in its license file.

## MiniMax H3 models (downloaded at runtime, not bundled)

The MiniMax H3 weights are downloaded from Hugging Face on first use and are not distributed with this package. They are subject to the [MiniMax-H3 License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) of [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3).

## SeedVR2 models (downloaded at runtime, not bundled)

The SeedVR2 upscale models are downloaded from Hugging Face on first use and are not distributed with this package: [`numz/SeedVR2_comfyUI`](https://huggingface.co/numz/SeedVR2_comfyUI) and [`AInVFX/SeedVR2_comfyUI`](https://huggingface.co/AInVFX/SeedVR2_comfyUI), both derived from [`ByteDance-Seed/SeedVR2-3B`](https://huggingface.co/ByteDance-Seed/SeedVR2-3B) and [`ByteDance-Seed/SeedVR2-7B`](https://huggingface.co/ByteDance-Seed/SeedVR2-7B) and licensed under the Apache License 2.0.

## Prompt enhancer models (downloaded at runtime, not bundled)

The default prompt-enhancer models are downloaded from Hugging Face on first use and are not distributed with this package: [`McG-221/Qwen3.5-4B-heretic-mlx-4Bit`](https://huggingface.co/McG-221/Qwen3.5-4B-heretic-mlx-4Bit) (macOS) and [`coder3101/Qwen3.5-4B-heretic`](https://huggingface.co/coder3101/Qwen3.5-4B-heretic) (Windows/Linux), both derived from [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) and licensed under the Apache License 2.0.
