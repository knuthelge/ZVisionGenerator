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

## yue2

The YuE2 inference kit (`packages/yue2`, used for music) is vendored from [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE) at commit 3acdec8cf1b97603bb696e4c6c4560348ffbb0b7 (`src/yue2`), without its command-line tool (`cli.py`) and its vLLM module (`fast.py`).

Licensed under the Apache License, Version 2.0. The full license text ships with the vendored code as `packages/yue2/LICENSE`, and at http://www.apache.org/licenses/LICENSE-2.0. The kit's own notices ship as `packages/yue2/THIRD_PARTY_NOTICES.md`: its VAE (`modeling_vae.py`) is derived from [Stability-AI/stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) (Copyright (c) 2023 Stability AI, MIT) and its SnakeBeta activation from NVIDIA's BigVGAN (Copyright (c) 2022 NVIDIA CORPORATION, MIT), with their license texts in `packages/yue2/licenses/`.

The code license does not cover the YuE2 model weights, which are not vendored and are licensed separately by their authors under CC BY-NC 4.0 with additional permission for individual creators.

## stable_audio_3

The Stable Audio 3 inference package (`packages/stable_audio_3`, used for sound and ambience) is vendored from [Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3) at commit 5f5e6ac21ae9ff78a6e8847eb7a1dcbe969ffb84, without its training code, its Gradio interface, its command-line tool and its dataset module. Nothing is taken from Stability AI's gated model repository.

MIT License, Copyright (c) 2026 Stability AI. The full license text ships with the vendored code as `packages/stable_audio_3/LICENSE`. The model weights are not vendored and are licensed separately by Stability AI.

## qwen_tts

The Qwen3-TTS Python package (`packages/qwen_tts`, used for speech on Windows and Linux) is vendored from [andimarafioti/Qwen3-TTS](https://github.com/andimarafioti/Qwen3-TTS), branch `distribution/qwen-tts-hf` (the `qwen-tts-hf` build of [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) for Transformers 5), at commit a466278d04ce9693395947906a949f744acd41e8, without its demo (`cli/` and `__main__.py`).

Licensed under the Apache License, Version 2.0, Copyright 2026 Alibaba Cloud. The full license text ships with the vendored code as `packages/qwen_tts/LICENSE`, and at http://www.apache.org/licenses/LICENSE-2.0.

## faster_qwen3_tts

The faster-qwen3-tts package (`packages/faster_qwen3_tts`, used for speech on Windows and Linux) is vendored from [andimarafioti/faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) at commit 1094aabd83b367b1b29942769141b0524bed245d (release v0.5.4), without its command-line tool (`cli.py`).

MIT License, Copyright (c) 2026 Andres Marafioti. The full license text ships with the vendored code as `packages/faster_qwen3_tts/LICENSE`.

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

## mlx-audio (speech on macOS)

On macOS, speech models run on [mlx-audio](https://github.com/Neywa-Labs/mlx-audio), installed as a dependency (MIT License, Copyright (c) 2024 Prince Canuma). The app calls its public `mlx_audio.tts` API and does not copy its code.

## Qwen3-TTS models (downloaded at runtime, not bundled)

The Qwen3-TTS weights are downloaded from Hugging Face on first use and are not distributed with this package: [`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) for the built-in voices and [`Qwen/Qwen3-TTS-12Hz-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) for cloning a voice recording, both licensed under the Apache License 2.0.

## Stable Audio 3 models (downloaded at runtime, not bundled)

The Stable Audio 3 Medium weights are downloaded from Hugging Face on first use and are not distributed with this package: the checkpoint from [`Comfy-Org/stable-audio-3`](https://huggingface.co/Comfy-Org/stable-audio-3), its model configuration from [`cocktailpeanut/stable-audio-3-medium`](https://huggingface.co/cocktailpeanut/stable-audio-3-medium) and the T5Gemma text encoder from [`mlx-community/stable-audio-3-small-sfx`](https://huggingface.co/mlx-community/stable-audio-3-small-sfx). The model is released by Stability AI in [`stabilityai/stable-audio-3-medium`](https://huggingface.co/stabilityai/stable-audio-3-medium) under the Stability AI Community License, which allows free use below $1M of annual revenue and asks for a free registration for commercial use. It requires this notice to accompany the model:

> This Stability AI Model is licensed under the Stability AI Community License, Copyright © Stability AI Ltd. All Rights Reserved

and the credit "Powered by Stability AI" on a related website, user interface or product page. The text encoder is a Gemma model, provided under and subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms):

> Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms

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

## uv (downloaded at runtime by the desktop launcher, not bundled)

The desktop launcher downloads a pinned release of [uv](https://github.com/astral-sh/uv) from Astral's GitHub releases when it installs or repairs the app, and uses it to install Python and the app. uv is not distributed with this package or inside the launcher. It is dual-licensed under the MIT License or the Apache License 2.0, at your option; the license texts ship with each uv release and are at https://github.com/astral-sh/uv/blob/main/LICENSE-MIT and https://github.com/astral-sh/uv/blob/main/LICENSE-APACHE.

## FFmpeg (downloaded at runtime by the desktop launcher, not bundled)

The desktop launcher downloads pinned GPL builds of `ffmpeg` and `ffprobe` when it installs or repairs the app, and puts them where the app looks for them. They are not distributed with this package or inside the launcher. [FFmpeg](https://ffmpeg.org/) is licensed under the GNU General Public License for these builds, which include GPL components such as libx264; its license text is at https://www.gnu.org/licenses/gpl-3.0.html and its source at https://ffmpeg.org/download.html and https://github.com/FFmpeg/FFmpeg.

- Windows and Linux: the `gpl` builds of [BtbN's FFmpeg-Builds](https://github.com/FFmpeg-Builds/FFmpeg-Builds), release `autobuild-2026-10-31-12-59` (FFmpeg n9.0.2). The repository holds the build scripts and the versions of every library in the build. The builds are GPLv3 because they are configured with `--enable-version3`.
- macOS (Apple silicon): FFmpeg 9.0.2 from [ffmpeg.martin-riedl.de](https://ffmpeg.martin-riedl.de/), a signed static build by Martin Riedl, also GPLv3 with `--enable-version3`.

## Nunito (bundled in the desktop launcher)

The launcher's window bundles the Nunito typeface (the Latin subset of the variable font, as packaged by [`@fontsource-variable/nunito`](https://github.com/fontsource/font-files)), so that its headings look the same on every platform. Nunito is licensed under the SIL Open Font License, Version 1.1. The license text also ships with the font as `frontend/src/launcher/fonts/OFL.txt`, and is reproduced here:

```text
Copyright 2014 The Nunito Project Authors (https://github.com/googlefonts/nunito) Nunito-Italic[wght].ttf: Copyright 2014 The Nunito Project Authors (https://github.com/googlefonts/nunito)

This Font Software is licensed under the SIL Open Font License, Version 1.1.
This license is copied below, and is also available with a FAQ at:
http://scripts.sil.org/OFL


-----------------------------------------------------------
SIL OPEN FONT LICENSE Version 1.1 - 26 February 2007
-----------------------------------------------------------

PREAMBLE
The goals of the Open Font License (OFL) are to stimulate worldwide
development of collaborative font projects, to support the font creation
efforts of academic and linguistic communities, and to provide a free and
open framework in which fonts may be shared and improved in partnership
with others.

The OFL allows the licensed fonts to be used, studied, modified and
redistributed freely as long as they are not sold by themselves. The
fonts, including any derivative works, can be bundled, embedded,
redistributed and/or sold with any software provided that any reserved
names are not used by derivative works. The fonts and derivatives,
however, cannot be released under any other type of license. The
requirement for fonts to remain under this license does not apply
to any document created using the fonts or their derivatives.

DEFINITIONS
"Font Software" refers to the set of files released by the Copyright
Holder(s) under this license and clearly marked as such. This may
include source files, build scripts and documentation.

"Reserved Font Name" refers to any names specified as such after the
copyright statement(s).

"Original Version" refers to the collection of Font Software components as
distributed by the Copyright Holder(s).

"Modified Version" refers to any derivative made by adding to, deleting,
or substituting -- in part or in whole -- any of the components of the
Original Version, by changing formats or by porting the Font Software to a
new environment.

"Author" refers to any designer, engineer, programmer, technical
writer or other person who contributed to the Font Software.

PERMISSION & CONDITIONS
Permission is hereby granted, free of charge, to any person obtaining
a copy of the Font Software, to use, study, copy, merge, embed, modify,
redistribute, and sell modified and unmodified copies of the Font
Software, subject to the following conditions:

1) Neither the Font Software nor any of its individual components,
in Original or Modified Versions, may be sold by itself.

2) Original or Modified Versions of the Font Software may be bundled,
redistributed and/or sold with any software, provided that each copy
contains the above copyright notice and this license. These can be
included either as stand-alone text files, human-readable headers or
in the appropriate machine-readable metadata fields within text or
binary files as long as those fields can be easily viewed by the user.

3) No Modified Version of the Font Software may use the Reserved Font
Name(s) unless explicit written permission is granted by the corresponding
Copyright Holder. This restriction only applies to the primary font name as
presented to the users.

4) The name(s) of the Copyright Holder(s) or the Author(s) of the Font
Software shall not be used to promote, endorse or advertise any
Modified Version, except to acknowledge the contribution(s) of the
Copyright Holder(s) and the Author(s) or with their explicit written
permission.

5) The Font Software, modified or unmodified, in part or in whole,
must be distributed entirely under this license, and must not be
distributed under any other license. The requirement for fonts to
remain under this license does not apply to any document created
using the Font Software.

TERMINATION
This license becomes null and void if any of the above conditions are
not met.

DISCLAIMER
THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
OF COPYRIGHT, PATENT, TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL THE
COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
INCLUDING ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL
DAMAGES, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF THE USE OR INABILITY TO USE THE FONT SOFTWARE OR FROM
OTHER DEALINGS IN THE FONT SOFTWARE.
```
