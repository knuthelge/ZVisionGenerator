<div class="zb zb--developer" markdown>

# Developer

Thanks for helping with Z-Vision Generator. This section is for working on the code: setting up, the rules every change follows, and how the app is built.

</div>

## Ways to help

- **Report a bug** with a [bug report](https://github.com/knuthelge/ZVisionGenerator/issues/new?template=bug_report.yml): steps to reproduce, what you expected and what happened, your platform and Python version, and the error output.
- **Suggest a feature** with a [feature request](https://github.com/knuthelge/ZVisionGenerator/issues/new?template=feature_request.yml), describing the use case.
- **Send a change** as a pull request, described below.

Security issues go through [SECURITY.md](https://github.com/knuthelge/ZVisionGenerator/blob/main/SECURITY.md), not public issues. Everyone taking part follows the [Code of Conduct](https://github.com/knuthelge/ZVisionGenerator/blob/main/CODE_OF_CONDUCT.md).

## Set up

You need:

- Python 3.14 or newer and [uv](https://docs.astral.sh/uv/);
- Node.js 22.22.2+ (22.x), 24.15.0+ (24.x) or 26+, and pnpm 11 or newer, for the Web UI;
- a Mac with Apple Silicon, or Windows or Linux with an NVIDIA GPU, to run real generation. The test suite runs anywhere, since it never loads a model.

```bash
git clone https://github.com/knuthelge/ZVisionGenerator && cd ZVisionGenerator
make install
```

`make install` runs `uv sync` and installs the frontend's packages. Then:

```bash
uv run ziv ui          # the Web UI from your checkout
make frontend-dev      # the Vite dev server, for frontend work
make docs-serve        # these docs, with live reload
```

## Make a change

1. Look in `proposals/` for a write-up of larger work before you start. See [Proposals](#proposals).
2. Create a branch.
3. Make the change, following the [conventions](conventions.md). Add or update tests.
4. Iterate with targeted checks:

    ```bash
    uv run pytest tests/test_<module>.py -k <name>
    make lint-fix && make format
    make frontend-test
    ```

5. Update the docs and `CHANGELOG.md` for anything a user would notice. See [Writing docs](docs.md).
6. Run the full gate, which CI runs too:

    ```bash
    make check
    ```

7. Commit with a [Conventional Commits](#commits) message, and open a pull request against `main`.

## Commits

`type(scope): imperative summary`, with a body that lists the high-level changes:

```text
feat(web): save a Workspace prompt to a prompt file

- Add Save… to the prompt box's tools row
- Append to the `prompts` set, keeping the file's comments
```

- **Types:** `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `build`, `ci`, `chore`.
- **Scope** is the product area, such as `web`, `image`, `video` or `enhance`.
- Mark a breaking change with `!` after the scope and a `BREAKING CHANGE:` line in the body.

## Repository rules

- **Vendored packages.** `packages/ltx_*` and `packages/seedvr2_torch` are copies of upstream code. Never edit them by hand; see [Vendored packages](vendored.md).
- **The built Web UI is committed.** After any change in `frontend/`, run `make frontend-build` and commit the rebuilt files in `zvisiongenerator/web/static/app/`. `make check` fails when they are out of date.
- **User-facing changes** get an entry under `[Unreleased]` in `CHANGELOG.md` and updated docs.
- **Releases** are cut by the maintainer. Never push a `v*` tag: it publishes to PyPI. See [Releasing](releasing.md).

## Proposals

Future work is written up in `proposals/` before it is built: the problem, the evidence, the proposed change and the alternatives. Its `README.md` explains the format and lists the open proposals. When a proposal ships or is rejected, its status is updated and it moves to `proposals/archive/`. The docs never link to proposals.

## Make targets

| Target | Does |
|---|---|
| `make install` | Install Python and frontend dependencies |
| `make check` | The full CI gate: lint, format check, pytest, frontend checks, packaged SPA drift check, strict docs build |
| `make test` | Run pytest, one worker per CPU core |
| `make lint` / `make lint-fix` | Run ruff, or let it fix what it can |
| `make format` / `make format-check` | Format with ruff, or check formatting |
| `make frontend-install` | Install only the frontend dependencies |
| `make frontend-dev` | Start the Vite dev server |
| `make frontend-build` | Build the Svelte app into `zvisiongenerator/web/static/app/` |
| `make frontend-test` | Frontend type checks and Vitest |
| `make frontend-static-check` | Rebuild the Svelte app and fail if the committed build differs |
| `make frontend-palette-check` | Fail on raw colour classes or literals in Svelte files (`PALETTE_STRICT=0` only warns) |
| `make frontend-hover-check` | Fail on a `:hover` rule outside `@media (hover: hover)` |
| `make docs-check` | Build the docs strictly; fails on broken links and anchors |
| `make docs-serve` | Serve the docs with live reload on `localhost:8000` |
| `make lock` | Regenerate `uv.lock` |
| `make build` | Build the wheel and sdist |
| `make clean` | Remove build artifacts, caches and the venv |
| `make run` / `make model` | Run `ziv-image` / `ziv-model` with `ARGS="..."` |
| `make enhance-bench` | Run the [prompt-enhancer bench](enhancer-bench.md) with `ARGS="..."`; loads a real model |
| `make update-ltx` / `make update-seedvr2` / `make update-h3-mlx` | Re-vendor LTX, SeedVR2 or the MiniMax H3 MLX port. See [Vendored packages](vendored.md) |
| `make e2e` / `make e2e-image` / `make e2e-video` | Smoke tests that load real models and write tiny outputs to `output/`; not part of `make check` |

## License

Contributions are licensed under the [AGPL-3.0-or-later](https://github.com/knuthelge/ZVisionGenerator/blob/main/LICENSE), like the rest of the project.
