# Third-Party Licenses

## ltx-core-mlx and ltx-pipelines-mlx

Vendored from [dgrauet/ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx) at commit 895201df379b079c2154738bcf92005b7a371923.

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

## moss_image

The MOSS-Image pipeline (`packages/moss_image`, used on Windows and Linux) is vendored unchanged from [OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) at revision f972bd4b4b9e3852133cc5da954d87bc32cd8fb9.

Licensed under the Apache License, Version 2.0. The full license text ships with the vendored code as `packages/moss_image/LICENSE`, and at http://www.apache.org/licenses/LICENSE-2.0.

## birefnet

The BiRefNet model code (`packages/birefnet`, used for background removal) is vendored unchanged from [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) at revision e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4.

Licensed under the MIT License, which the model card of the upstream repository declares. The repository ships no license file, so none is vendored.

## MiniMax H3 latent upscaler (network code)

The network code of the MiniMax H3 upscale (`zvisiongenerator/backends/cuda/video/h3/latent_upsampler.py`, used on Windows and Linux) is ported from [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) (`nodes/minimax_h3_latent_upscaler_3d.py`), matching upstream commit 98934c0b5f4a213323c4d1d0e47eb3d2f27cb743, without its ComfyUI node code.

MIT License

Copyright (c) 2026 LBH-123-AI

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

## MiniMax H3 models (downloaded at runtime, not bundled)

The MiniMax H3 weights are downloaded from Hugging Face on first use and are not distributed with this package. They are subject to the [MiniMax-H3 License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) of [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3).

## MiniMax H3 latent upscaler weights (downloaded at runtime, not bundled)

The upscaler weights are downloaded from Hugging Face on the first upscale and are not distributed with this package: [`LBH-123-AI/Minimax_h3_latent_Upscaler`](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler), licensed under the Apache License 2.0.

## SeedVR2 models (downloaded at runtime, not bundled)

The SeedVR2 upscale models are downloaded from Hugging Face on first use and are not distributed with this package: [`numz/SeedVR2_comfyUI`](https://huggingface.co/numz/SeedVR2_comfyUI) and [`AInVFX/SeedVR2_comfyUI`](https://huggingface.co/AInVFX/SeedVR2_comfyUI), both derived from [`ByteDance-Seed/SeedVR2-3B`](https://huggingface.co/ByteDance-Seed/SeedVR2-3B) and [`ByteDance-Seed/SeedVR2-7B`](https://huggingface.co/ByteDance-Seed/SeedVR2-7B) and licensed under the Apache License 2.0.

## Background removal models (downloaded at runtime, not bundled)

The models behind Transparent background are downloaded from Hugging Face on first use and are not distributed with this package. Both carry permissive licence labels, but both were trained on datasets whose terms do not allow commercial use, so the feature is offered for non-commercial use only and is marked that way in the app.

- [`ZhengPeng7/BiRefNet`](https://huggingface.co/ZhengPeng7/BiRefNet), which finds the subject. The weights are licensed under the MIT License. They were trained on DIS-TR from DIS5K, whose terms of use state that the dataset is available for non-commercial use in research or educational purposes and that commercial use is prohibited, also after copying, editing or processing it.
- [`hustvl/vitmatte-small-composition-1k`](https://huggingface.co/hustvl/vitmatte-small-composition-1k) (ViTMatte-S), which refines the edge. The model card tags the weights Apache-2.0. They were trained on Composition-1k, which is built from Adobe's Deep Image Matting dataset, so they are treated as non-commercial.

## pymatting

The foreground colour estimation of Transparent background (`zvisiongenerator/processing/matte.py`) is reimplemented in PyTorch from the multilevel foreground estimation of [pymatting](https://github.com/pymatting/pymatting) (PyMatting), which is licensed under the MIT License. pymatting itself is not a dependency.

MIT License

Copyright (c) 2020 PyMatting

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

## Prompt enhancer models (downloaded at runtime, not bundled)

The default prompt-enhancer models are downloaded from Hugging Face on first use and are not distributed with this package: [`McG-221/Qwen3.5-4B-heretic-mlx-4Bit`](https://huggingface.co/McG-221/Qwen3.5-4B-heretic-mlx-4Bit) (macOS) and [`coder3101/Qwen3.5-4B-heretic`](https://huggingface.co/coder3101/Qwen3.5-4B-heretic) (Windows/Linux), both derived from [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) and licensed under the Apache License 2.0.
