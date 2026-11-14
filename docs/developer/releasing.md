---
search:
  exclude: true
---

# Releasing

Releases are cut by the maintainer. Pushing a `v*` tag runs the Release workflow, which resolves the wheel, builds the package, publishes it to PyPI and creates a GitHub Release. PyPI never accepts the same version twice, so a broken release needs a new tag, and a pushed tag cannot be undone. Never create or push a `v*` tag unless you are releasing.

## Version numbers

The package version comes from the tag (`hatch-vcs`), so the tag must be a valid PEP 440 version:

- **Final releases:** `vX.Y.Z`, for example `v0.13.0`.
- **Betas:** `vX.Y.ZbN`, for example `v0.13.0b1`, then `b2`, `b3` and so on before the final `vX.Y.Z`. Use `aN` and `rcN` the same way for alphas and release candidates.

Pre-release tags are published as GitHub pre-releases, and uv installs them only when asked: `uv tool install --prerelease-package z-vision-generator=allow z-vision-generator`, or an exact pin such as `uv tool install z-vision-generator==0.13.0b1`.

Final release tags also run the Deploy Docs workflow, which publishes the docs site from the tagged commit. Pre-release tags leave the published docs unchanged; run the Deploy Docs workflow by hand to publish between releases. A manual run publishes `main`, so the docs of every feature not yet released go live with it.

## Checklist

Do these in order, on the commit you will tag.

1. **Land everything.** The branch to release is on the newest `main`, and `make check` passes.
2. **Check the curated LoRA list.** Run `make check-curated`. It asks the Hub for every entry at its pinned commit, with no token, and compares each file's size and `lfs.sha256` with the list. It exits with an error and names each entry whose file is missing, changed or unreachable; fix or drop those entries first. The same run rewrites each entry's `downloads` with the repository's current Hugging Face download count in `curated_loras.json`, so commit the file afterwards (`--no-write` checks without writing). It needs the network, so it is not part of `make check`. See [The curated LoRA list](curated-loras.md).
3. **Check that the wheel resolves.** `scripts/check_wheel_resolves.sh` builds the wheel and resolves it from an empty directory with `uv pip compile --no-config` for three fixed targets, from any host: Linux (`x86_64-unknown-linux-gnu`, PyPI only), Windows (`x86_64-pc-windows-msvc`, with `--torch-backend cu132`, as on the [install page](../get-started/install.md)) and macOS (`aarch64-apple-darwin` with `MACOSX_DEPLOYMENT_TARGET=14.0`, because uv assumes macOS 13 for that target while `mlx` ships only macOS 14 and newer wheels). A user installs the wheel with `uv tool install z-vision-generator`, where nothing of the repository's `uv.lock` or `[tool.uv]` settings applies; the script fails when a dependency has no solution next to ours, or when `gradio`, `qwen-tts-hf` or `faster-qwen3-tts` comes back. CI runs it on every pull request, on its macOS runner, and the Release workflow runs it on an Ubuntu runner before it publishes; the script resolves all three targets from either host.
4. **Write the release commit.** Move the `[Unreleased]` entries of `CHANGELOG.md` under a `## [<version>] - <date>` heading, in a commit named `chore(release): v<version>` whose body lists the release highlights.
5. **Check the install guide's launcher links.** If [Install](../get-started/install.md) links a `launcher-v*` release, that release must exist before you tag, because the docs deploy on every final app tag and would otherwise publish dead download links.
6. **Tag and push.** Create the tag on that commit and push the commit and the tag. The Release workflow takes it from there.
7. **Check the result.** The workflow's run is green, the version is on PyPI, the GitHub Release exists, and for a final release the docs site shows the new version.

## Launcher releases { #launcher-releases }

The desktop launcher is released apart from the app. It has its own version, in `launcher/src-tauri/Cargo.toml`, and its own workflow, started by `launcher-v*` tags. A `v*` tag never builds it, and a `launcher-v*` tag never publishes to PyPI. The app's version reads only `v*` tags, so `launcher-v*` tags never affect it. Which app version it installs comes from PyPI, so an app release needs no launcher release, except when the app needs something the launcher doesn't provide, such as a newer Python.

The launcher hard-codes things the app owns: the Python version, the Windows `--torch-backend cu132` and the uv version. When the app changes any of these, ship a launcher release first, because older launchers' **Update** will fail or change nothing. See [Desktop launcher](launcher.md) for how it is built.

The release is not marked latest, so app releases stay on top; that is why the install guide links to the exact `launcher-v…` release and the asset names stay fixed. A pushed `launcher-v*` tag cannot be undone.

Do these in order.

1. **Land everything.** The branch is on the newest `main`, and `make launcher-test` and `make check` pass.
2. **Bump the version.** Set `version` in `launcher/src-tauri/Cargo.toml`. The workflow fails a tag whose version differs from it.
3. **Update the pins** if a tool has moved or a download has gone, as in [Pins](launcher.md#pins).
4. **Run a test build.** Start the Launcher release workflow by hand from the branch. It builds the installers for macOS, Windows and Linux and uploads them as artifacts, and publishes nothing.
5. **Check each platform** with the installers from the test build: a fresh install, start, Quit, Update, Repair and Uninstall, and the first-open steps on the install page. Include a Windows account whose name has spaces or non-ASCII characters, the tray on stock GNOME and the AppImage on a fresh Ubuntu. These need a real machine:
    - **Windows:** turn on **Start at login**, then sign out and in. The launcher starts hidden and opens no browser.
    - **Windows:** Quit, Update and Uninstall with other apps open. Only the launcher's own processes stop.
    - **macOS:** the first install on a Mac without the Command Line Tools shows no "install developer tools" popup.
    - **macOS:** open the launcher from the disk image. The window says to quit and move it; after the move, the copy in Applications opens.
    - **macOS:** log out with **Reopen windows when logging back in** ticked. The launcher does not reopen its window.
    - **macOS:** after **Open Anyway**, install and start both work, so Gatekeeper does not kill the managed Python.
6. **Update the install guide.** Point the download links in [Install](../get-started/install.md#get-the-launcher) at the new `launcher-v<version>` release.
7. **Tag and push.** Create `launcher-v<version>` on the commit and push the commit and the tag. The workflow builds the installers again and creates the GitHub release with the five files.
8. **Deploy the docs** once the release exists, so that the install page's links work: run the Deploy Docs workflow by hand, since a `launcher-v*` tag does not run it.
9. **Check the result.** The release is not marked latest, each of the five files downloads, and the install page links to them.
