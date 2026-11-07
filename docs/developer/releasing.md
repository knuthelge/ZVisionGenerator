# Releasing

Pushing a `v*` tag runs the Release workflow, which builds the package, publishes it to PyPI and creates a GitHub Release. The package version comes from the tag (`hatch-vcs`), so the tag must be a valid PEP 440 version:

- **Final releases:** `vX.Y.Z`, e.g. `v0.13.0`.
- **Betas:** `vX.Y.ZbN`, e.g. `v0.13.0b1`, then `b2`, `b3`, … before the final `vX.Y.Z`. Use `aN` and `rcN` the same way for alphas and release candidates.

Final release tags also run the Deploy Docs workflow, which publishes the docs site from the tagged commit; pre-release tags leave the published docs unchanged.

Pre-release tags are published as GitHub pre-releases, and uv installs them only when asked: `uv tool install --prerelease allow z-vision-generator`, or an exact pin, `uv tool install z-vision-generator==0.13.0b1`. Before tagging, move the `[Unreleased]` CHANGELOG entries under a `## [<version>] - <date>` heading in a `chore(release): v<version>` commit whose body lists the release highlights. PyPI never accepts the same version twice, so a broken release needs a new tag.

## Curated list

The curated LoRA list pins each file to a commit of its Hugging Face repository (see [The curated LoRA list](curated-loras.md)). Before tagging, run `make check-curated`. It asks the Hub for every entry at its pinned commit, with no token, and compares the file's size and `lfs.sha256` with the list. It exits with an error and names each entry whose file is missing, changed or unreachable; fix or drop those entries before the release. The same run rewrites each entry's `downloads` with the repository's current Hugging Face download count in `curated_loras.json`, so commit the file after it (`--no-write` checks without writing). It needs the network, so it is not part of `make check`.

## Wheel resolve check

A user installs the wheel with `uv tool install z-vision-generator`, where nothing of the repository's `uv.lock` or `[tool.uv]` settings applies. `scripts/check_wheel_resolves.sh` builds the wheel and resolves it from an empty directory with `uv pip compile --no-config` for three fixed targets, from any host: Linux (`x86_64-unknown-linux-gnu`, PyPI only), Windows (`x86_64-pc-windows-msvc`, with the CUDA index and `--index-strategy unsafe-best-match` of the [install page](../install.md)) and macOS (`aarch64-apple-darwin` with `MACOSX_DEPLOYMENT_TARGET=14.0`, because uv assumes macOS 13 for that target while `mlx` ships only macOS 14 and newer wheels). It fails when a dependency has no solution next to ours, or when `gradio`, `qwen-tts-hf` or `faster-qwen3-tts` comes back. CI runs it on every pull request and the Release workflow runs it, on an Ubuntu runner, before it publishes.

