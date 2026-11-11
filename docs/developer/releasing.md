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

Pre-release tags are published as GitHub pre-releases, and uv installs them only when asked: `uv tool install --prerelease allow z-vision-generator`, or an exact pin such as `uv tool install z-vision-generator==0.13.0b1`.

Final release tags also run the Deploy Docs workflow, which publishes the docs site from the tagged commit. Pre-release tags leave the published docs unchanged; run the Deploy Docs workflow by hand to publish between releases.

## Checklist

Do these in order, on the commit you will tag.

1. **Land everything.** The branch to release is on the newest `main`, and `make check` passes.
2. **Check the curated LoRA list.** Run `make check-curated`. It asks the Hub for every entry at its pinned commit, with no token, and compares each file's size and `lfs.sha256` with the list. It exits with an error and names each entry whose file is missing, changed or unreachable; fix or drop those entries first. The same run rewrites each entry's `downloads` with the repository's current Hugging Face download count in `curated_loras.json`, so commit the file afterwards (`--no-write` checks without writing). It needs the network, so it is not part of `make check`. See [The curated LoRA list](curated-loras.md).
3. **Check that the wheel resolves.** `scripts/check_wheel_resolves.sh` builds the wheel and resolves it from an empty directory with `uv pip compile --no-config` for three fixed targets, from any host: Linux (`x86_64-unknown-linux-gnu`, PyPI only), Windows (`x86_64-pc-windows-msvc`, with the CUDA index and `--index-strategy unsafe-best-match` of the [install page](../get-started/install.md)) and macOS (`aarch64-apple-darwin` with `MACOSX_DEPLOYMENT_TARGET=14.0`, because uv assumes macOS 13 for that target while `mlx` ships only macOS 14 and newer wheels). A user installs the wheel with `uv tool install z-vision-generator`, where nothing of the repository's `uv.lock` or `[tool.uv]` settings applies; the script fails when a dependency has no solution next to ours, or when `gradio`, `qwen-tts-hf` or `faster-qwen3-tts` comes back. CI runs it on every pull request, on its macOS runner, and the Release workflow runs it on an Ubuntu runner before it publishes; the script resolves all three targets from either host.
4. **Write the release commit.** Move the `[Unreleased]` entries of `CHANGELOG.md` under a `## [<version>] - <date>` heading, in a commit named `chore(release): v<version>` whose body lists the release highlights.
5. **Tag and push.** Create the tag on that commit and push the commit and the tag. The Release workflow takes it from there.
6. **Check the result.** The workflow's run is green, the version is on PyPI, the GitHub Release exists, and for a final release the docs site shows the new version.
