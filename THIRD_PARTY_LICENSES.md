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

The code license does not cover the YuE2 model weights, which are not vendored and are licensed separately by their authors under CC BY-NC 4.0.

## stable_audio_3

The Stable Audio 3 inference package (`packages/stable_audio_3`, used for sound and ambience) is vendored from [Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3) at commit 5f5e6ac21ae9ff78a6e8847eb7a1dcbe969ffb84, without its training code, its Gradio interface, its command-line tool and its dataset module. Nothing is taken from Stability AI's gated model repository.

MIT License, Copyright (c) 2026 Stability AI. The full license text ships with the vendored code as `packages/stable_audio_3/LICENSE`. The model weights are not vendored and are licensed separately by Stability AI.

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

## sglang-omni (Higgs TTS 3 model code)

Parts of the Higgs TTS 3 speech code are adapted from [sgl-project/sglang-omni](https://github.com/sgl-project/sglang-omni), matching upstream commit d80161dc34d7b900e7a0869481fa79d35ed618d9:

- `zvisiongenerator/backends/audio/higgs3_model.py`: the fused multi-codebook embedding and head, from `sglang_omni/models/higgs_tts/modeling.py`.
- `zvisiongenerator/backends/audio/higgs3_codes.py`: the delay pattern from `sglang_omni/models/higgs_tts/utils.py` (`apply_delay_pattern`), the de-delay from `sglang_omni/utils/codec_delay.py` (`reverse_delay_pattern`), and the sampler and the frame rule from `sglang_omni/models/higgs_tts/sampler.py`.
- `zvisiongenerator/backends/audio/higgs3_prompt.py`: the prompt layout, from `sglang_omni/models/higgs_tts/text_tokenizer.py`.

Each file's header names its source and what was changed. The network around them is transformers' Apache-2.0 `Qwen3Model`.

Apache License 2.0, Copyright 2025-2026 SGLang-Omni Team. Upstream publishes no NOTICE file. The full license text, copied from upstream's `LICENSE` at that commit, follows.

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright 2025-2026 SGLang-Omni Team

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

## Higgs TTS 3 models (downloaded at runtime, not bundled)

The Higgs TTS 3 weights are downloaded from Hugging Face on first use and are not distributed with this package: [`bosonai/higgs-tts-3-4b`](https://huggingface.co/bosonai/higgs-tts-3-4b), licensed under the [Boson Higgs TTS 3 Research and Non-Commercial License](https://huggingface.co/bosonai/higgs-tts-3-4b/blob/239f63fb7b02b1aa085f98d9efae5e35cc5523e8/LICENSE), which allows research and non-commercial use only, with a limited exception for digital creators. A copy of that license ships with this package as `licenses/boson-higgs-tts-3.txt`. The audio codec, [`bosonai/higgs-audio-v2-tokenizer`](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer), is likewise downloaded, and is licensed under the [Boson Higgs Audio 2 Community License Agreement](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer/blob/815a03f1b57dcb5ae918db415f22cd0c8a6f0f54/LICENSE). That license is based upon the Meta Llama 3 Community License Agreement and incorporates Meta's Acceptable Use Policy. It requires this credit line on a related website, user interface, blogpost, about page or product documentation:

> Built with Higgs Materials licensed from Boson AI USA, Inc., Copyright Boson AI USA, Inc., All Rights Reserved and Meta Llama 3 licensed under the Meta Llama 3 Community License, Copyright Meta Platforms, Inc., All Right Reserved

This file serves as the NOTICE file the Boson license asks for:

> Boson Higgs TTS 3 is licensed under the Boson Higgs TTS 3 Research and Non-Commercial License, Copyright (c) Boson AI USA, Inc. All Rights Reserved.

The Higgs TTS 3 code of this application is "Derived from Higgs TTS 3, licensed from Boson AI USA, Inc."

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
