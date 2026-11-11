# LoRAs

A LoRA is a small add-on file that teaches a model a style, a character or a subject. It works on top of the model it was trained for: a FLUX.2 Klein LoRA on Klein, an LTX LoRA on LTX.

Image and video models take LoRAs (Z-Image, FLUX.2 Klein, Krea 2, Ideogram 4, MOSS-Image, LTX and MiniMax H3). The audio models don't. A LoRA can't be used with a model it wasn't made for; the rules per model are in [What works](#what-works).

## Get one { #get-one }

LoRAs live in `loras/` in the [data folder](../reference/data-folder.md) (`~/.ziv` by default) as `.safetensors` files. To add one:

- **Discover LoRAs** in the Workspace finds LoRAs for your model and downloads them ([below](#discover-loras));
- **Import** a file, or a Hugging Face repository, on the [Models page](../web-ui/models.md) or with `ziv-model lora`: `ziv-model lora --hf owner/repo --name my-lora` downloads one from Hugging Face, and `ziv-model lora -i style.safetensors` copies a local file. See [Import a LoRA](../guides/models.md#import-a-lora);
- **Copy** the file into `loras/` yourself, for example a `.safetensors` LoRA downloaded from Civitai.

`--lora` takes a name or a path, never a Hugging Face repository (`--lora owner/repo` is refused): import it first, then use its name.

### Discover LoRAs { #discover-loras }

Discover finds LoRAs for the model you have picked and downloads them into `loras/`. It needs a model whose LoRA family is known. See [Workspace](../web-ui/workspace.md) for where it is.

- **Curated** LoRAs were checked by us: the card, the previews, the licence, and that the file works with the model. The list ships with the app; its pictures load from Hugging Face. A **Tool** does something, such as a transition, instead of giving a look. **Non-commercial** marks a licence that forbids commercial use.
- **Hugging Face** lists the LoRAs authors publish for the model, unchecked. You turn it on once for this ziv, and **Turn off** hides it again.
- Discover never sends your Hugging Face login.

A Hugging Face file is saved under the repository's name, with `-2` added when that name is taken.

### Content filter { #content-filter }

Discover hides repositories that Hugging Face tags as adult and ones whose names use explicit terms. The filter is best effort: some adult content can still show up, and now and then a fine LoRA is hidden. To see the ones the name filter hides, set `ui.discover_content_filter` to `off` in the [config](../reference/configuration.md#ui); repositories tagged as adult stay hidden.

## Use one { #use-one }

Pick LoRAs for a job and give each a **weight**. The weight sets how strongly the LoRA applies: 1.0 is full strength, lower is subtler. Several LoRAs stack, each at its own weight.

- **Which LoRAs are offered.** The Workspace lists the LoRAs made for the selected model first, with the ones you used with it recently on top, and folds the rest, including ones whose model can't be detected, under **Other models (n)**. Which model a LoRA was made for is read from the file's header the first time the file is seen, and remembered.
- **Remembered per model family,** and per size for FLUX.2 Klein (4B and 9B keep separate LoRAs): switch to another family and you get the LoRAs you last used with it.
- **Recorded with the file.** The LoRAs and weights an image or video was made with are stored in it. **Reuse settings** and **Extend** use exactly those LoRAs; a source made without LoRAs clears them.

### Trigger words { #trigger-words }

Some LoRAs need a trigger word in the prompt. The app never adds one for you: pick it from the **Triggers** menu (the tag icon) in the prompt box, which lists the words of the LoRAs in use and counts the ones the prompt doesn't have yet. The word goes at the start of the text Generate sends. The menu is hidden for a JSON caption and a prompt file.

A LoRA's trigger comes from its curated entry, its Hugging Face model card or the file itself. **Edit** in the LoRA's detail in Discover sets your own.

## What works { #what-works }

Each family reads its own LoRAs, and the formats are listed in [LoRA formats](../reference/models.md#lora-formats):

- **Images:** standard LoRAs and LyCORIS LoKr files, on every platform and quant level, and they can be combined.
- **Ideogram 4:** Ideogram 4 LoRAs load onto Instant, but check the results.
- **MOSS-Image:** MOSS-Image LoRAs on `moss` and `moss-sft`, at any quant level. On `moss` (Turbo) the Turbo LoRA stays on and yours is added on top.
- **LTX video:** LTX LoRAs. An unsupported file or scale fails before generation with a message saying why.
- **MiniMax H3:** H3 LoRAs. A LoRA can add a few percent to the time and break the seam of an extension.

Parts of a LoRA the model can't use are skipped with a warning, and your file isn't changed. With [quantization](memory.md), the LoRA is applied on top of the quantized model when the job loads.

## Where you set it

- **Web UI:** **LoRAs** in the Workspace toolbar, with a weight for each; **Triggers** in the prompt box ([Workspace](../web-ui/workspace.md)).
- **Command line:** `--lora name:weight,name2:weight` (see the [command reference](../cli/reference.md)), as in `ziv-image -m klein9b --lora style1:0.8,detail:0.5 --prompt "a portrait"`. A bare name is found in `loras/` in the data folder; a full path works too.
