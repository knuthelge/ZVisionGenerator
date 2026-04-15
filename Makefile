.DEFAULT_GOAL := help

ARGS ?=
ZIMAGE_MODEL ?= Tongyi-MAI/Z-Image-Turbo
KLEIN_MODEL  ?= black-forest-labs/FLUX.2-klein-9B
LTX_MODEL    ?= dgrauet/ltx-2.3-mlx-q4

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

# ——— Build ————————————————————————————————————————————————

.PHONY: build clean

build: ## Build wheel and sdist
	uv build

clean: ## Remove build artifacts, caches, and venv
	rm -rf build/ dist/ *.egg-info output/ .agent-work/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .venv

# ——— Run ——————————————————————————————————————————————————

.PHONY: run convert

run: ## Run ziv CLI (use ARGS="..." for arguments)
	uv run ziv $(ARGS)

convert: ## Run ziv-convert CLI (use ARGS="..." for arguments)
	uv run ziv-convert $(ARGS)

# ——— E2E Image ————————————————————————————————————————————

.PHONY: e2e-image e2e-image-zimage e2e-image-klein

e2e-image-zimage: ## Run Z-Image E2E tests
	@mkdir -p output
	@echo "=== E2E: zimage — basic generation ==="
	uv run ziv -m $(ZIMAGE_MODEL) --prompt "a red circle on white background" -W 64 -H 64 -o output
	@echo "=== E2E: zimage — custom steps and seed ==="
	uv run ziv -m $(ZIMAGE_MODEL) --prompt "a blue square" -W 64 -H 64 -o output --steps 6 --seed 42
	@echo "=== E2E: zimage — guidance scale and contrast ==="
	uv run ziv -m $(ZIMAGE_MODEL) --prompt "a green triangle" -W 64 -H 64 -o output --guidance 1.0 --contrast
	@echo "=== E2E: zimage — saturation and no-sharpen ==="
	uv run ziv -m $(ZIMAGE_MODEL) --prompt "sunset over ocean" -W 64 -H 64 -o output --saturation --no-sharpen
	@echo "=== E2E: zimage — beta scheduler ==="
	uv run ziv -m $(ZIMAGE_MODEL) --prompt "mountain landscape" -W 64 -H 64 -o output --scheduler beta
	@echo "=== E2E: zimage — ratio/size default (2:3 @ m) ==="
	uv run ziv -m $(ZIMAGE_MODEL) --prompt "test" -o output --steps 1
	@echo "=== E2E: zimage — explicit ratio + size ==="
	uv run ziv -m $(ZIMAGE_MODEL) --prompt "test" --ratio 1:1 --size xs -o output --steps 1
	@echo "=== E2E: zimage — only ratio (size defaults to m) ==="
	uv run ziv -m $(ZIMAGE_MODEL) --prompt "test" --ratio 16:9 -o output --steps 1
	@echo "=== E2E: zimage — only size (ratio defaults to 2:3) ==="
	uv run ziv -m $(ZIMAGE_MODEL) --prompt "test" -s xl -o output --steps 1
	@echo "=== E2E: zimage — ratio + size combination ==="
	uv run ziv -m $(ZIMAGE_MODEL) --prompt "test" --ratio 9:16 -s s -o output --steps 1
	@echo "=== E2E: all Z-Image tests passed ==="

e2e-image-klein: ## Run FLUX.2-klein E2E tests
	@mkdir -p output
	@echo "=== E2E: klein — basic generation ==="
	uv run ziv -m $(KLEIN_MODEL) --prompt "a red circle on white background" -W 64 -H 64 -o output
	@echo "=== E2E: klein — custom steps and seed ==="
	uv run ziv -m $(KLEIN_MODEL) --prompt "a blue square" -W 64 -H 64 -o output --steps 2 --seed 123
	@echo "=== E2E: klein — guidance scale ==="
	uv run ziv -m $(KLEIN_MODEL) --prompt "a green triangle" -W 64 -H 64 -o output --guidance 2.0
	@echo "=== E2E: klein — multiple runs ==="
	uv run ziv -m $(KLEIN_MODEL) --prompt "abstract pattern" -W 64 -H 64 -o output -r 2
	@echo "=== E2E: all FLUX.2-klein tests passed ==="

e2e-image: e2e-image-zimage e2e-image-klein ## Run all image E2E tests

# ——— E2E Video —————————————————————————————————————————————

.PHONY: e2e-video e2e-video-ltx

e2e-video-ltx: ## Run LTX video E2E tests
	@mkdir -p output
	@echo "=== E2E: LTX T2V — basic generation ==="
	uv run ziv-video -m $(LTX_MODEL) --prompt "a red circle moving" -W 128 -H 128 --frames 9 --steps 2 -o output
	@echo "=== E2E: LTX T2V — custom seed ==="
	uv run ziv-video -m $(LTX_MODEL) --prompt "ocean waves" -W 160 -H 128 --frames 9 --steps 2 --seed 42 -o output
	@echo "=== E2E: LTX T2V — resolution alignment (non-32-divisible) ==="
	uv run ziv-video -m $(LTX_MODEL) --prompt "abstract shapes" -W 140 -H 140 --frames 10 --steps 2 -o output
	@echo "=== E2E: LTX T2V — low-memory disabled ==="
	uv run ziv-video -m $(LTX_MODEL) --prompt "sunset" -W 128 -H 128 --frames 9 --steps 2 --no-low-memory -o output
	@echo "=== E2E: all LTX video tests passed ==="

e2e-video: e2e-video-ltx ## Run all video E2E tests

# ——— E2E All ——————————————————————————————————————————————

.PHONY: e2e

e2e: e2e-image e2e-video ## Run all E2E tests (image + video)
