# Developer

Thanks for helping with Z-Vision Generator. This section is for working on the code: setting up, making a first change, the rules every change follows, and how the app is built.

## Ways to help

- **Report a bug** with a [bug report](https://github.com/knuthelge/ZVisionGenerator/issues/new?template=bug_report.yml): steps to reproduce, what you expected and what happened, your platform and Python version, and the error output.
- **Suggest a feature** with a [feature request](https://github.com/knuthelge/ZVisionGenerator/issues/new?template=feature_request.yml), describing the use case.
- **Send a change** as a pull request, as in [Your first change](#your-first-change).

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
uv run ziv ui          # the Web UI from your checkout, on port 8080
make docs-serve        # these docs, with live reload on localhost:8000
```

Run `make help` for every target. The ones you will use most:

| Target | Does |
|---|---|
| `make check` | The full CI gate; run it before you finish |
| `make lint-fix` and `make format` | Let ruff fix and format your Python |
| `make frontend-test` | Frontend type checks and Vitest |
| `make frontend-build` | Build the Web UI into the package; commit the result |
| `make docs-check` | Build these docs strictly |
| `make launcher-dev`, `make launcher-test`, `make launcher-build`, `make launcher-clean` | Run, check, build and clean the desktop launcher; see [Desktop launcher](launcher.md) |

### Working on the frontend

`make frontend-dev` starts the Vite dev server with hot reload. It proxies `/api`, `/jobs`, `/thumbs` and `/media` to a Web UI server on port 8765 (`frontend/vite.config.ts`), but `ziv ui` listens on 8080 by default, so start the server on the port Vite expects, in a second terminal:

```bash
uv run ziv ui --port 8765 --no-browser
make frontend-dev
```

Open the address Vite prints. Edits to `frontend/src/` appear at once. The dev server is for iterating; the package serves the committed build, so run `make frontend-build` before you commit. See [Frontend](frontend.md).

## Your first change

This walks through a small change from branch to merge. The example adds a `--vignette [AMOUNT]` flag to `ziv-image` that darkens the image edges, modelled on the existing `--contrast`. It is a command-line flag only: the Web UI form, **Reuse settings** and presets need more, and the [Add a CLI flag](how-to.md#add-a-cli-flag) recipe lists every place a flag touches. Any change of this size follows the same loop; [Testing](testing.md) says which check to run when.

**1. Branch from the newest main.**

```bash
git fetch origin
git switch -c feat/image-vignette origin/main
```

Look in `proposals/` first for a write-up of larger work. See [Proposals](#proposals).

**2. Write the tests first.** Three places hold the contract of a flag like this, and each has a sibling to copy:

- The parser: add `("vignette", False, 0.4)` to the cases of `test_flag_forms` in `TestPostProcessingFlags` (`tests/image/test_cli_parsing.py`), and `["--vignette", "1.5"]` to the refused amounts. The second one already passes, because argparse refuses an unknown flag; it starts to mean something once the flag exists.
- The processing function: a new test file beside `tests/image/test_image_adjustments.py`, named `test_vignette`, where the corners come out darker than the centre, an amount of 0 returns the image unchanged and an amount above 1 raises `ValueError`.
- The workflow: add `{"vignette": 0.4}` expecting `[vignette_stage, sharpen_stage]` to `TestBuildWorkflow` in `tests/image/test_workflow.py`.

Run them; they fail because none of it exists yet.

```bash
uv run pytest tests/image/test_cli_parsing.py -k vignette
```

```text
FAILED tests/image/test_cli_parsing.py::TestPostProcessingFlags::test_flag_forms[vignette-False-0.4] - AttributeError: 'Namespace' object has no attribute 'vignette'
```

**3. Make the change.** Follow `contrast` through the code (`git grep -n contrast -- zvisiongenerator`) and mirror each place:

- A new `vignette` module beside `processing/contrast.py`: the pure function `apply_vignette(image, amount)`.
- `workflows/image_stages.py`: `vignette_stage`, a thin adapter over it, and `build_workflow()` in `workflows/__init__.py`: append it `if args.vignette is not False`, before `sharpen_stage`.
- `image_cli.py`: the `--vignette` and `--no-vignette` arguments, and the range check beside the other `parser.error(...)` calls.
- `core/image_types.py`: `vignette` and `vignette_amount` on `ImageGenerationRequest`. `image_runner.py` fills them where it builds the request, resolving an amount that was not given from `config["vignette"]["default_amount"]`, as `contrast_amount` does.
- `config.yaml`: the `vignette:` section with `default_amount: 0.4`, and `"vignette"` in the mapping check of `utils/config.load_config()`. The key is a setting, so list it in [Configuration](../reference/configuration.md).
- `utils/provenance.py`: record the amount in the output's `generation` block, so the file carries its settings.
- `web/server.py`: add `vignette=False` to the `argparse.Namespace` that `_parse_image_job()` builds. The runner reads `args.vignette`, so without it every Web UI image job fails with an `AttributeError`. The tests' own namespaces need it too: `_make_args()` in `tests/_factories.py` and `_default_args()` in `tests/image/test_workflow.py`.

Then the same tests pass:

```bash
uv run pytest tests/image -k vignette
```

```text
5 passed, 811 deselected in 1.79s
```

**4. Keep the style clean.**

```bash
make lint-fix && make format
```

```text
All checks passed!
690 files left unchanged
```

**5. Document it.** Add the flag to the table in [the command reference](../cli/reference.md), the `vignette.default_amount` key to [Configuration](../reference/configuration.md) and a line under `[Unreleased]` in `CHANGELOG.md`. A test fails when a flag or a config key is missing from the reference: `uv run pytest tests/common/test_docs_reference.py`.

**6. Run the full gate.**

```bash
make check
```

It runs lint, format check, pytest, the frontend checks and a strict docs build, and takes about three minutes. It stops at the first failing step. A passing run prints these lines along the way (the counts and times differ), and ends with the docs build:

```text
All checks passed!
690 files already formatted
=========== 16440 passed, 164 skipped, 44 warnings in 102.16s (0:01:42) ===========
COMPLETED 891 FILES 0 ERRORS 0 WARNINGS 0 FILES_WITH_PROBLEMS
✓ built in 5.50s
No issues found
```

**7. Commit** with a [Conventional Commits](#commits) message:

```text
feat(image): add --vignette to darken the image corners

- Add --vignette [AMOUNT] and --no-vignette to ziv-image
- Add the vignette stage, its config default and its record in the output
```

**8. Rebase and merge.** Land the branch on the newest main as a series of commits, with no merge commit. In this repository, fast-forward main to the branch:

```bash
git fetch origin
git rebase origin/main
make check                      # again, if the rebase brought changes
git switch main
git merge --ff-only feat/image-vignette
```

On a fork, push the branch and open a pull request against `main` instead. Never create or push a `v*` tag; see [Releasing](releasing.md).

## Commits

`type(scope): imperative summary`, with a body that lists the high-level changes, as in step 7.

- **Types:** `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `build`, `ci`, `chore`.
- **Scope** is the product area, such as `web`, `image`, `video` or `enhance`.
- Mark a breaking change with `!` after the scope and a `BREAKING CHANGE:` line in the body.

## Repository rules

- **Vendored packages.** Everything under `packages/` is a copy of upstream code. Never edit them by hand; see [Vendored packages](vendored.md).
- **The built Web UI is committed.** After any change in `frontend/`, run `make frontend-build` and commit the rebuilt files in `zvisiongenerator/web/static/app/`. `make check` fails when they are out of date.
- **User-facing changes** get an entry under `[Unreleased]` in `CHANGELOG.md` and updated docs. See [Writing docs](docs.md).
- **Releases** are cut by the maintainer. Never push a `v*` tag: it publishes to PyPI. See [Releasing](releasing.md).

## Proposals

Future work is written up in `proposals/` before it is built: the problem, the evidence, the proposed change and the alternatives. Its `README.md` explains the format and lists the open proposals. When a proposal ships or is rejected, its status is updated and it moves to `proposals/archive/`. The docs never link to proposals.

## License

Contributions are licensed under the [AGPL-3.0-or-later](https://github.com/knuthelge/ZVisionGenerator/blob/main/LICENSE), like the rest of the project.
