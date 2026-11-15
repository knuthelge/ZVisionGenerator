# Add a Hugging Face token from the Web UI

**Status:** Done (2026-11-15; built on `feat/hf-token`)

## Problem

Every message about a gated model tells the user to run `hf auth login`, but a fresh user can't. `uv tool install z-vision-generator` (the install page, and the launcher's `uv.rs::install_args`) puts only our own `[project.scripts]` on PATH, not `huggingface_hub`'s `hf`. Launcher users don't have `uv` on PATH either. Setting `HF_TOKEN` means restarting the app with an environment variable.

None of the built-in models is gated. This hits users who add a gated model or LoRA by repo: few people, but they currently have no way forward that works.

## What the user sees

The existing "Hugging Face token" row on the Config and Models pages gets a button:

| Token | Value | Hint | Button |
|---|---|---|---|
| None | Not set | Gives higher download limits and access to gated models. | **Add token** |
| Saved (by the app or `hf auth login`) | Available | Saved on this computer. | **Replace token** |
| From `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | Available | From HF_TOKEN. Unset it to use a token saved here. | none |

The button opens a small dialog:

- Two lines on what a token does: higher download limits (big downloads are less likely to be slowed down or stopped) and gated models.
- A link, **Create a token on Hugging Face** (`https://huggingface.co/settings/tokens`, new tab), followed by "and choose the Read token type." A plain text field, not a password field, so browsers don't offer to save the token as a site password.
- A password field and **Save** (disabled while the field is empty or saving).
- On success the dialog closes and a "Hugging Face token saved" toast appears. On failure it stays open with one error line.

On desktop and tablet the dialog is `Modal size="sm"`. On phones `KeyValueList` already puts the value under the label, so the button sits under the value. `Modal` turns into a bottom sheet on its own, and Save goes full width.

The LoRA import warning on the Models page changes to "Add a Hugging Face token above to download gated models."

## How it works

**Backend.** `POST /api/huggingface/token` in `web/server.py` reads `await request.json()`, like `api_save_config`. `web/request_guard.py` already guards it.

1. Strip the token. If it's empty, return 400 "Paste a token."
2. Call `huggingface_hub.login(token=token, add_to_git_credential=False)` in the thread pool. That function checks the token with Hugging Face and writes `HF_HOME/token`, the file `get_token()` reads on every call. Downloads, running jobs and the CLI then use the new token without a restart.
3. Map errors with three branches:
   - `ValueError` → 400 with its message (Hugging Face's text for an organisation token).
   - `HfHubHTTPError` with status 401 → 400 "Hugging Face doesn't accept this token."
   - Anything else → 502 "Couldn't check the token with Hugging Face. Try again when you're online."

   The last branch uses fixed text and never `str(exc)`. An untrimmed token raises an httpx header error that contains the token.
4. Return the new `huggingface_token_source`.

**Token source and messages** (`downloads/hub_access.py`):

- `token_source()` returns `"saved"` instead of `"hf auth login"` for a token in the file.
- The login hint in the refusal lines becomes "add a token on the Web UI's Config page or run `uvx hf auth login`". `hf` on PyPI is Hugging Face's own alias package, and it writes the same token file (checked). One wording serves the CLI and the Web UI, because the same code produces the refusal line in both, including in the download child process.

**Frontend:**

- An optional `action` field on `KeyValueList` items, rendered as a small button after the value.
- One shared helper that builds the token row for both pages.
- A new `HfTokenDialog`, the same shape as `SaveVoiceDialog`.
- `saveHfToken()` in `lib/api/config.ts`.
- After a save, each page reloads only its own data.

The token is never stored in `config.yaml` and never logged. The server never sends it back to the browser.

## Not in v1

- **Remove.** A wrong token is fixed by pasting a new one. A token is revoked on huggingface.co, or removed locally with `uvx hf auth logout`. Removing from the app can't use `logout()`: it always runs `git credential reject`, which fails without git and opens the "install developer tools" dialog on a Mac.
- **An Add token action on the refusal toast.** Refusals reach the frontend as plain text over three paths, and the toast's one action slot is already Retry.
- **Sign in with Hugging Face** (a device-code login) instead of pasting a token.

## Security

With `--host local`, any device that can open the Web UI can replace the token, and the token crosses the LAN unencrypted. Anyone on the LAN can already browse the computer's folders, so `docs/web-ui/devices.md` gets one sentence about this.

## Alternatives considered

- **Store the token in `config.yaml`.** Rejected: it creates a second source of truth next to `HF_HOME/token`, puts a secret in a file the Config page round-trips, and every Hub call would have to pass the token.
- **Only change the wording to `uvx hf auth login`.** Launcher users have no `uv` on PATH, so this can't be the only fix. The new wording uses it as the terminal option.
- **`--with-executables-from huggingface_hub` on the install commands.** It does nothing for the launcher, and every install, upgrade and reinstall command would have to carry it.

## Tests

- `tests/web/test_huggingface_token_api.py`:
  - A trimmed token reaches a mocked `login` with `add_to_git_credential=False`.
  - An empty or whitespace-only token returns 400 without calling `login`.
  - Parametrized over `ValueError`, `HfHubHTTPError` 401, `httpx.ConnectError` and an `httpx.LocalProtocolError` whose text contains the token: each returns the right status, and no response body contains the token.
- `tests/downloads/test_downloads_hub_access.py`: the new `token_source` value and the new hint wording.
- Frontend:
  - The dialog closes and reports success on a save, and shows the error line on a 400.
  - The row shows a button for no token and for a saved token, and none for `HF_TOKEN`.

## Docs

- `docs/reference/data-folder.md#hugging-face-token`: add a token from the Config page, with `uvx hf auth login` and `HF_TOKEN` as the other ways. Add one line saying a launcher started from a desktop menu doesn't see an `HF_HOME` set in the shell.
- `docs/reference/troubleshooting.md`: the steps under the refusal messages.
- `docs/web-ui/config.md` and `docs/web-ui/models.md`: one clause each for the button.
- `docs/web-ui/devices.md`: the security sentence.
- `docs/cli/reference.md`: `HF_TOKEN`'s note.
- The CHANGELOG.
