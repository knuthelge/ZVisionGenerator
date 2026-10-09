# Dependency refresh: shared packages and the frontend

**Status:** Proposed (2026-10-09)

This is the first of three dependency proposals. It covers what every platform installs and runs the same way: the web server, the general Python libraries, the Hugging Face hub client, the dev tools and the frontend toolchain. The platform-specific engines are in [macOS: mflux 0.22 and a fresh LTX](deps-macos.md) and [Windows and Linux: torch 2.14, CUDA 13 and diffusers 0.41](deps-cuda.md).

## Problem

The lockfile was last refreshed broadly in the spring. As of 2026-10-09, most direct dependencies are a few minor versions behind. That isn't urgent on its own, but three things make it worth doing now:

1. **`hf-transfer` is a dead dependency.** huggingface-hub 1.x no longer uses it. Downloads go through `hf-xet`, which the hub already pulls in (1.6.0 in `uv.lock`). Our code never imports `hf_transfer` or sets `HF_HUB_ENABLE_HF_TRANSFER`. Worse, a user who sets that variable from old guides gets a `FutureWarning` from the hub at import time (`huggingface_hub/constants.py`, v1.33.0).
2. **The platform upgrades need new floors here.** mflux 0.22 requires `pillow>=12.3.0` and diffusers 0.41 requires `huggingface-hub>=1.32.0`. Raising those first keeps the platform proposals small.
3. **The frontend toolchain has moved two majors** (Vite 8, Vitest 5). Taking the patch-level updates now and the majors as a separate step is cheaper than one big jump later.

## Evidence

Output of `uv tree --outdated --depth 1` (Linux and `--python-platform aarch64-apple-darwin`) and the npm registry, 2026-10-09.

### Python

| Package | Floor in `pyproject.toml` | Locked | Latest | Notes |
|---|---|---|---|---|
| `hf-transfer` | `>=0.1.9` | 0.1.9 | 0.1.9 | Unused. Remove. |
| `huggingface-hub` | `>=1.23.0` | 1.33.0 | 2.2.0 | 2.x is blocked: diffusers 0.41 and mflux 0.22 both require `<2.0`. |
| `fastapi` | `>=0.116.0` | 0.136.1 | 0.143.0 | |
| `uvicorn` | `>=0.35.0` | 0.46.0 | 0.54.0 | |
| `python-multipart` | `>=0.0.20` | 0.0.28 | 0.0.32 | Parses uploads, so worth staying current. |
| `numpy` | `>=2.0.0` | 2.4.4 | 2.5.3 | mflux needs `<3.0`. |
| `scipy` | `>=1.17.0` | 1.17.1 | 1.18.1 | Only `schedulers/beta_scheduler.py`. |
| `Pillow` | `>=12.0.0` | 12.2.0 | 12.3.0 | mflux 0.22 needs `>=12.3.0`. |
| `tqdm` | `>=4.66.0` | 4.67.3 | 4.70.1 | |
| `pyyaml`, `ruamel-yaml`, `safetensors` | | current | | No change. |
| `ruff` (dev) | `>=0.11.0` | 0.15.12 | 0.16.10 | A new minor can enable new default rules. |
| `pytest` (dev) | `>=9.0.2` | 9.0.3 | 9.1.1 | |
| `mkdocs-material` (dev) | `>=9.6` | 9.7.6 | 9.7.7 | |

`torch`, `transformers`, `accelerate`, `bitsandbytes` and `diffusers` are in the CUDA proposal. `mflux`, `mlx`, `mlx-lm` and the vendored LTX packages are in the macOS proposal.

### Frontend (`frontend/package.json`)

| Package | Locked | Latest | Step |
|---|---|---|---|
| `svelte` | 5.55.4 | 5.57.2 | 1 |
| `svelte-check` | 4.4.6 | 4.7.6 | 1 |
| `tailwindcss`, `@tailwindcss/vite` | 4.2.4 | 4.3.3 | 1 |
| `vite` | 7.3.2 | 8.3.4 | 2 (with the Svelte plugin) |
| `@sveltejs/vite-plugin-svelte` | 6.2.4 | 7.3.1 | 2 (needs Vite 8) |
| `vitest` | 3.2.4 | 5.0.3 | 2 (with jsdom) |
| `jsdom` | 26.1.0 | 30.1.2 | 2 |
| `typescript` | 5.9.3 | 7.0.2 | Not now |
| `@types/node` | 22.19.17 | 26.6.5 | Not now |
| `pnpm` (`packageManager`) | 11.1.2 | 12.10.1 | Not now |

## Proposed change

### 1. Python: one lock refresh

- Remove `hf-transfer` from `dependencies`.
- Raise floors to what we test against: `huggingface-hub>=1.32.0,<2.0`, `Pillow>=12.3.0`, `fastapi>=0.143.0`, `uvicorn>=0.54.0`, `python-multipart>=0.0.32`. Other floors stay as they are. They describe the oldest version that works, and the lock decides what gets installed.
- Cap `huggingface-hub` at `<2.0` explicitly. Two of our engines already cap it, so the cap documents a fact rather than adding a restriction. It also turns a future resolver conflict into a clear pin to revisit.
- `uv lock --upgrade-package` for each package above, plus the dev group. Don't run a blanket `uv lock --upgrade`: that would also move torch, transformers and mflux, which belong to the platform proposals and need hardware to verify.
- Run `make lint-fix && make format` under ruff 0.16 and commit any rule fixes separately (`style:` or `chore(lint):`), so the dependency commit stays readable.

### 2. Frontend, step 1: minor updates

Update svelte, svelte-check and Tailwind within their current majors. Run `make frontend-test`, then `make frontend-build`, and commit the rebuilt `zvisiongenerator/web/static/app/`. The visible output should not change. Check the Workspace, Gallery, Prompts and Config pages in a browser, since Tailwind minors sometimes change how utilities resolve.

### 3. Frontend, step 2: Vite 8 and Vitest 5

A separate PR, because the bundler changes:

- Vite 8 replaces the esbuild and Rollup pair with Rolldown. Chunk names and file hashes in the packaged SPA will change, so the drift check in `make check` will flag the whole `static/app/` folder once. That is expected.
- `@sveltejs/vite-plugin-svelte` 7 goes with it. Review `frontend/vite.config.ts` against the Vite 8 migration guide, in particular `build.rollupOptions.input`, which Vite 8 deprecates in favour of `build.rolldownOptions`.
- Vitest 5 and jsdom 30 together. Review `frontend/vitest.config.ts`, which relies on `globals: true`, the `browser` resolve condition and `svelte({ hot: false })`, and also `src/test-setup.ts`.
- Compare the packaged bundle size before and after and note it in the PR.

### 4. Not now

- **TypeScript 7** is the new native compiler. Wait until `svelte-check` supports it officially.
- **`@types/node`** stays on 22 because `engines.node` is `>=22.0.0`. The types should describe the oldest runtime we support, not the newest.
- **pnpm 12** is a major release of the package manager everyone has to install. Take it only when there is a reason to.

### Docs and changelog

- `CHANGELOG.md` under `[Unreleased]`: "hf-transfer is no longer installed. Downloads use Xet. Set `HF_XET_HIGH_PERFORMANCE=1` instead of `HF_HUB_ENABLE_HF_TRANSFER`."
- Mention `HF_XET_HIGH_PERFORMANCE` wherever the docs talk about download speed. There is currently no such section, and adding one is optional.

## Order across the three proposals

The lock is shared, so every proposal moves packages for all platforms. CI runs only on `macos-latest`, which means CUDA changes have to be checked by hand.

1. **This proposal.** It verifies on CI and in the browser.
2. **[Windows and Linux](deps-cuda.md).** It moves torch to 2.14, which mflux 0.22 requires (`torch>=2.13.0`).
3. **[macOS](deps-macos.md).** It builds on the torch and Pillow floors from 1 and 2.

## Alternatives considered

- **Upgrade everything in one PR.** Rejected: a broken generation on one platform would be hard to bisect between a dozen bumps, and the CUDA half can't be verified in CI.
- **Allow huggingface-hub 2.x now.** Not possible: diffusers 0.41.0 (`huggingface-hub>=1.32.0,<2.0`) and mflux 0.22.0 (`<2.0`) block it. Our own use (`hf_hub_download`, `snapshot_download`, `list_repo_files`, `constants`) wouldn't need code changes. The breaking changes are the httpx2 swap and removed APIs we don't call. Revisit when both engines allow it.
- **Keep `hf-transfer` for users who set the variable.** Rejected: the hub ignores it, so keeping it only adds a Rust wheel to every install.
- **Renovate or Dependabot for Python.** Worth considering separately. It would keep these refreshes small, but the CUDA verification gap stays the same.

## Open questions

- Whether the first-run download proposal ([first-run.md](first-run.md)) should set `HF_XET_HIGH_PERFORMANCE` for its downloads. It uses more connections and memory, and it may help most on fast links.
