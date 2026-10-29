# The curated LoRA list

The curated list is a hand-checked index of LoRAs on Hugging Face that ships inside the package. It lives in `zvisiongenerator/utils/curated_loras.json`, read by `utils/curated_loras.py`. The pictures on its tiles are not bundled: they load from Hugging Face. The list changes only with a release.

No hub offers safe-only results, so safety comes from someone looking at what is listed. An entry is curated only when all of these hold.

## Criteria

**Safe.** Someone looked at the card and at the preview images. Hugging Face's own adult-content tag is not trusted: it is set by authors, and explicit uploads go untagged. A repository with explicit, hateful or otherwise unsuitable content is not listed, and neither is a gated one.

**A working family match.** The file's safetensors header, read with `detect_lora_family`, names the family the entry is listed under (and the size, for Klein). A repository whose tags say one family and whose weights say another is not listed. LTX-2.3 and LTX-2.5 share one detected family, so an LTX entry carries its `version` and is offered only for that version's model. The Workspace context gives each LTX video model its `version`, read by `utils/ltx_version.py` without loading the model: the config's version blocks (aliases), the known-repo table, then a local folder's converted manifest, its transformer config's `ff_bias` (`false` is 2.5) or its diffusers text encoder (Gemma 4 is 2.5); a folder with none of them is 2.3.

**Real previews.** The repository shows what the LoRA does: two to four preview images that come from the LoRA itself, not a trainer's generic sample prompts. A repository whose previews are only videos lists two to four of them instead. A repository without previews is not listed.

**A licence that was read.** The licence is one of the ids in `LICENCES`. A non-commercial licence is listed, with `non_commercial` set, so the UI can badge it. A licence not in the table is read first, then added to `LICENCES`.

**Tested platforms.** Content is looked at once. The file format (kohya, PEFT, ai-toolkit, LoKr, ComfyUI) is loaded on both macOS and Windows or Linux. MOSS-Image runs only on Windows and Linux, so its entries list `win32` and `linux` and nothing else in `platforms`.

**One per series.** Per-step uploads, snapshots of one training run, near-identical sets and templated series keep their best entry. A repository with several files lists the one the author marks as final, or the one without a step number.

**Tools only when the app can use them.** A tool (it does something, as opposed to a style, which looks like something) is listed only when the app can apply it. Its `workflow` says which workflow it needs (`txt2img`, `img2img`, `txt2vid`, `img2vid` or `extend`; none when any works), and `what_it_does` says what it does in one line. A tool that needs nodes, masks or a pipeline the app does not have is not listed.

## What an entry holds

| Field | Rule |
|---|---|
| `id` | Stable across releases; the repository path as a slug |
| `name` | A readable `snake_case` slug, unique across the list. It becomes the file name on disk and what presets, Reuse and `--lora` refer to. Never a generic file name such as `pytorch_lora_weights` |
| `family`, `size_variant`, `version` | The family as `utils/lora_family.py` and the video presets name it, with a size (`4b`, `9b`) for Klein and a version (`2.3`, `2.5`) for LTX |
| `display_name`, `author`, `description` | What people read. The description is one short, factual line written from the card and the previews, without claims the card does not make |
| `section`, `what_it_does`, `workflow` | `style` or `tool`. A tool says in one line what it does (`what_it_does`, the line its tile shows) and which workflow it needs |
| `repo`, `revision`, `file`, `size_bytes`, `sha256` | The file at a pinned 40-character commit, with the size and SHA-256 the Hub reports (`lfs.sha256`). The preview URLs are pinned to the same commit |
| `trigger` | The trigger words the card names for the prompt, comma-separated (`Pixel Art, PixArFK`), or none. A card that gives a sentence or a prompt keeps the one word that triggers the LoRA. No word ends in a full stop and none repeats. Never derived from the file name or from training metadata such as `ss_output_name` |
| `weight` | 1.0, unless the card clearly recommends another value |
| `licence`, `non_commercial` | The licence id and whether it forbids commercial use |
| `downloads` | The repository's Hugging Face download count, shown on the cards. `make check-curated` refreshes it; a new entry starts at `0` |
| `platforms`, `note` | The platforms it was tested on, and a remark about the file format when it matters |
| `previews`, `preview_videos` | Two to four URLs in all, on `huggingface.co` |

## Adding or changing an entry

1. Check the repository against the criteria above. Read the card, look at every preview, and load the file on each platform that applies.
2. Take the repository's current commit and the file's size and hash from the Hub:

    ```bash
    curl -s "https://huggingface.co/api/models/<owner>/<repo>?blobs=true"
    ```

    The response holds `sha` (the commit) and, for each file under `siblings`, its `size` and `lfs.sha256`. Pin the full commit, and write the preview URLs against it (`…/resolve/<commit>/<path>`).
3. Add the entry to `curated_loras.json` in the order the list is shown: best first within a family.
4. Run `uv run pytest tests/contracts/test_curated_loras.py`. It fails on a duplicate `id` or `name`, a family or size the app does not know, a malformed commit or hash, a file that is not `.safetensors`, a preview off the Hub, and a licence outside the allowed set.
5. Before a release, run `make check-curated` (see [Releasing](releasing.md#curated-list)). It also refreshes every entry's `downloads`.
