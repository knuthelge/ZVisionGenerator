# LoRAs

A LoRA is a small add-on file that teaches a model a style, a character or a subject. It works on top of the model it was trained for: a FLUX.2 Klein LoRA on Klein, an LTX LoRA on LTX.

## Get one

LoRAs live in `~/.ziv/loras/` as `.safetensors` files. To add one:

- **Discover LoRAs** in the Workspace: see [Discover LoRAs](#discover-loras);
- copy the file into that folder;
- import it from a file or from Hugging Face on the [Models page](../web-ui/models.md#import-a-lora) or with [`ziv-model lora`](../cli/models.md#import-a-lora).

## Discover LoRAs

**Discover LoRAs** in the Workspace finds LoRAs for your model and downloads them into `loras/`. See [Workspace](../web-ui/workspace.md#discover-loras) for how to use it.

- **Curated** LoRAs were checked by us: the card, the previews, the licence, and that the file works with the model. The list ships with the app; its pictures load from Hugging Face. A **Tool** does something, such as a transition, instead of giving a look. **Non-commercial** marks a licence that forbids commercial use.
- **Hugging Face** lists the LoRAs authors publish for the model, unchecked. You turn it on once for this ziv, and **Turn off** hides it again.
- Discover never sends your Hugging Face login.

A Hugging Face file is saved under the repository's name, with `-2` added when that name is taken.

### Content filter

Hugging Face leaves content filtering to its community, so Discover filters out what it can. It hides repos that Hugging Face tags as adult, and repos whose names, tags or file names use explicit terms. The filter does its best but can't catch everything, so some adult content still shows up. Now and then it may also hide a LoRA that is fine. If it hides one you want to use, set `ui.discover_content_filter` to `off` in the [config](configuration.md#ui). Repos tagged as adult stay hidden.

## Trigger words

Some LoRAs need a trigger word in the prompt. The app never adds one for you: pick it from the **Triggers** menu in the prompt box, which lists the words of the LoRAs in use and counts the ones the prompt doesn't have yet. The word goes at the start of the text Generate sends. The menu is hidden for a JSON caption and a prompt file.

A LoRA's trigger comes from its curated entry, its Hugging Face model card or the file itself. **Edit** in the LoRA's detail in Discover sets your own.

## Use one

=== "Web UI"

    Pick LoRAs under **LoRAs** in the Workspace toolbar and set each one's weight. The Add LoRA menu lists the LoRAs made for the selected model first, with the ones you used with it recently on top, and folds the rest, including ones whose model can't be detected, under **Other models (n)**. Which model a LoRA was made for is read from the file's header the first time the file is seen, and remembered.

    LoRAs are remembered per model family, and per size for FLUX.2 Klein: switch to another family and you get the LoRAs you last used with it.

=== "CLI"

    ```bash
    ziv-image -m klein9b --lora myStyle --prompt "a portrait"                # weight 1.0
    ziv-image -m klein9b --lora myStyle:0.8 --prompt "a portrait"            # weight 0.8
    ziv-image -m klein9b --lora style1:0.8,detail:0.5 --prompt "a portrait"  # two, stacked
    ```

    A bare name is found in `~/.ziv/loras/`; a full path works too.

The **weight** sets how strongly the LoRA applies: 1.0 is full strength, lower is subtler. Several LoRAs stack.

## What works

- **Images:** standard LoRAs and LyCORIS LoKr files work on every platform and quant level, and can be combined. Other LyCORIS formats aren't supported; those parts of a file are skipped with a warning.
- **MOSS-Image (Windows and Linux):** LoRAs in kohya, ai-toolkit and diffusers/PEFT formats, and LyCORIS LoKr files, on `moss` and `moss-sft`. They are applied unmerged, next to the model's weights, at any quant level. LyCORIS LoHa files and full `diff` weights in a file are skipped with a warning. On `moss` (Turbo) the Turbo LoRA stays on and yours is added on top.
- **Video on macOS:** LTX LoRAs as MLX loads them.
- **Video on Windows and Linux:** LTX LoRAs in a diffusers-compatible format. An unsupported file or scale fails before generation with a message saying why.
- **MiniMax H3:** H3 LoRAs in PEFT, ai-toolkit, kohya and ComfyUI formats. A LoRA that changes the model's timestep embedding is refused, and on macOS so is one that changes its AdaLN projections or anything outside its transformer blocks and text refiner. A LoRA adds a few percent to the time, and can break the seam of an extension.

LoRAs from other trainers, such as PEFT adapter names (`lora_A.default.weight`) or Kohya names (`lora_unet_blocks_0_attn_wq`), are renamed to the names the model reads when they load; your file isn't changed. On a Mac this takes a temporary copy in `~/.ziv/cache/loras/`, as large as the LoRA, which is deleted after the job.

A LoRA must match the model: LTX LoRAs for the LTX models, MiniMax H3 LoRAs for `h3`, MOSS-Image LoRAs for `moss` and `moss-sft`.

With [quantization](quantization.md), the LoRA is applied on top of the quantized model when the job loads.

## Recorded with the file

The LoRAs and weights an image or video was made with are stored in the file. **Reuse settings** and **Extend** use exactly those LoRAs; a source made without LoRAs clears them.
