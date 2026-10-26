# LoRAs

A LoRA is a small add-on file that teaches a model a style, a character or a subject. It works on top of the model it was trained for: a FLUX.2 Klein LoRA on Klein, an LTX LoRA on LTX.

## Get one

LoRAs live in `~/.ziv/loras/` as `.safetensors` files. To add one:

- copy the file into that folder;
- import it from a file or from Hugging Face on the [Models page](../web-ui/models.md#import-a-lora) or with [`ziv-model lora`](../cli/models.md#import-a-lora).

A LoRA on Hugging Face must be imported before you can use it.

## Use one

=== "Web UI"

    Pick LoRAs under **LoRAs** in the Workspace sidebar and set each one's weight. The Add LoRA menu lists the LoRAs made for the selected model first, with the ones you used with it recently on top, and folds the rest, including ones whose model can't be detected, under **Other models (n)**. Which model a LoRA was made for is read from the file's header the first time the file is seen, and remembered.

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

A LoRA must match the model: LTX LoRAs for the LTX models, MiniMax H3 LoRAs for `h3`, MOSS-Image LoRAs for `moss` and `moss-sft`.

With [quantization](quantization.md), the LoRA is applied on top of the quantized model when the job loads.

## Recorded with the file

The LoRAs and weights an image or video was made with are stored in the file. **Reuse settings** and **Extend** use exactly those LoRAs; a source made without LoRAs clears them.
