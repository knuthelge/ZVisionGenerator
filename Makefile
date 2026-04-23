.DEFAULT_GOAL := help

ARGS ?=
ZIMAGE_MODEL ?= zit
KLEIN_MODEL  ?= klein9b
LTX_MODEL    ?= ltx-8

# ——— Setup ———————————————————————————————————————————————

.PHONY: help install lock

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	uv sync

lock: ## Regenerate uv.lock
	uv lock

# ——— Quality ——————————————————————————————————————————————

.PHONY: lint lint-fix format format-check test check

lint: ## Run ruff linter
	uv run --frozen ruff check zvisiongenerator/ tests/

lint-fix: ## Run ruff linter with auto-fix
	uv run --frozen ruff check --fix zvisiongenerator/ tests/

format: ## Format code with ruff
	uv run --frozen ruff format zvisiongenerator/ tests/

format-check: ## Check formatting without changes
	uv run --frozen ruff format --check zvisiongenerator/ tests/

test: ## Run tests with pytest
	uv run --frozen pytest

check: lint format-check test ## Run lint + format-check + test (full CI gate)

# ——— Vendored Dependencies ————————————————————————————————

LTX_REPO   ?= https://github.com/dgrauet/ltx-2-mlx.git
LTX_COMMIT ?= de4a12d3cd14f345b2ec680eb63f4114b927b435

.PHONY: update-ltx

update-ltx: ## Update vendored ltx-core-mlx and ltx-pipelines-mlx
	$(eval TMP := $(shell mktemp -d))
	git clone --quiet $(LTX_REPO) $(TMP)
	cd $(TMP) && git checkout --quiet $(LTX_COMMIT)
	rm -rf packages/ltx_core_mlx packages/ltx_pipelines_mlx
	cp -r $(TMP)/packages/ltx-core-mlx/src/ltx_core_mlx packages/ltx_core_mlx
	cp -r $(TMP)/packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx packages/ltx_pipelines_mlx
	rm -f packages/ltx_pipelines_mlx/cli.py packages/ltx_pipelines_mlx/__main__.py
	rm -rf $(TMP)
	@echo "✓ Vendored ltx packages updated to $(LTX_COMMIT)"

# ——— Frontend ————————————————————————————————————————————

.PHONY: frontend-install frontend-build frontend-test frontend-dev

frontend-install: ## Install frontend npm dependencies
	npm install --prefix frontend

frontend-build: ## Build the Svelte frontend into the static dir
	npm run --prefix frontend build

frontend-test: ## TypeScript check + vitest tests
	npm run --prefix frontend check
	npm run --prefix frontend test

frontend-dev: ## Start Vite dev server (for development)
	npm run --prefix frontend dev

# ——— Build ————————————————————————————————————————————————

.PHONY: build clean

build: frontend-build ## Build wheel and sdist
	uv build

clean: ## Remove build artifacts, caches, and venv
	rm -rf build/ dist/ *.egg-info output/ .agent-work/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .venv

# ——— Run ——————————————————————————————————————————————————

.PHONY: run model

run: ## Run ziv-image CLI (use ARGS="..." for arguments)
	uv run ziv-image $(ARGS)

model: ## Run ziv-model CLI (use ARGS="..." for arguments)
	uv run ziv-model $(ARGS)

# ——— E2E Image ————————————————————————————————————————————

.PHONY: e2e-image e2e-image-zimage e2e-image-klein

e2e-image-zimage: ## Run Z-Image E2E tests
	@mkdir -p output
	@echo "=== E2E: zimage — smoke test ==="
	uv run ziv-image -m $(ZIMAGE_MODEL) --prompt "a red circle on white background" -W 64 -H 64 -o output --steps 1 --seed 42

e2e-image-klein: ## Run FLUX.2-klein E2E tests
	@mkdir -p output
	@echo "=== E2E: klein — smoke test ==="
	uv run ziv-image -m $(KLEIN_MODEL) --prompt "a red circle on white background" -W 64 -H 64 -o output --steps 2 --seed 42

e2e-image: e2e-image-zimage e2e-image-klein ## Run all image E2E tests

# ——— E2E Video —————————————————————————————————————————————

.PHONY: e2e-video e2e-video-ltx

e2e-video-ltx: ## Run LTX video E2E tests
	@mkdir -p output
	@echo "=== E2E: LTX T2V — smoke test ==="
	uv run ziv-video -m $(LTX_MODEL) --prompt "a red circle moving" -W 128 -H 128 --frames 9 --steps 2 --seed 42 -o output

e2e-video: e2e-video-ltx ## Run all video E2E tests

# ——— E2E All ——————————————————————————————————————————————

.PHONY: e2e

e2e: e2e-image e2e-video ## Run all E2E tests (image + video)
