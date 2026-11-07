.DEFAULT_GOAL := help

ARGS ?=
ZIMAGE_MODEL ?= zit
KLEIN_MODEL  ?= klein9b
IDEOGRAM_MODEL ?= ideo
LTX_MODEL    ?= ltx-8

# ——— Setup ———————————————————————————————————————————————

.PHONY: help install lock

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: frontend-install ## Install all dependencies
	uv sync

lock: ## Regenerate uv.lock
	uv lock

# ——— Quality ——————————————————————————————————————————————

.PHONY: lint lint-fix format format-check test docs-check docs-serve docs-screenshots check check-curated

lint: ## Run ruff linter
	uv run --frozen ruff check zvisiongenerator/ tests/ bench/

lint-fix: ## Run ruff linter with auto-fix
	uv run --frozen ruff check --fix zvisiongenerator/ tests/ bench/

format: ## Format code with ruff
	uv run --frozen ruff format zvisiongenerator/ tests/ bench/

format-check: ## Check formatting without changes
	uv run --frozen ruff format --check zvisiongenerator/ tests/ bench/

test: ## Run tests with pytest, one worker per CPU core
	uv run --frozen pytest -n auto

docs-check: ## Build documentation strictly with Zensical
	uv run --frozen zensical build --clean --strict

docs-serve: ## Serve documentation with live reload on localhost:8000
	uv run --frozen zensical serve

docs-screenshots: ## Retake the docs screenshots from a running throwaway Web UI (URL=..., optional EMPTY_URL=..., ONLY=...)
	uv run --frozen --with playwright python scripts/docs_screenshots.py --url $(URL) $(if $(EMPTY_URL),--empty-url $(EMPTY_URL)) $(if $(ONLY),--only $(ONLY))

check-curated: ## Compare the curated LoRA list's pinned files with the Hugging Face Hub (network; run before a release)
	uv run --frozen python scripts/check_curated_loras.py

check: lint format-check test frontend-test frontend-palette-check frontend-hover-check frontend-static-check docs-check ## Run lint + format-check + pytest + frontend checks + docs build + packaged SPA static gate (full CI gate)

# ——— Vendored Dependencies ————————————————————————————————

LTX_REPO   ?= https://github.com/dgrauet/ltx-2-mlx.git
# v0.18.0 + 4 commits (2026-10-26)
LTX_COMMIT ?= 895201df379b079c2154738bcf92005b7a371923

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

SEEDVR2_REPO   ?= https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git
# v2.5.24 (2025-12-24)
SEEDVR2_COMMIT ?= 4490bd1f482e026674543386bb2a4d176da245b9

.PHONY: update-seedvr2

update-seedvr2: ## Update vendored SeedVR2 PyTorch inference core (Windows/Linux)
	$(eval TMP := $(shell mktemp -d))
	git clone --quiet $(SEEDVR2_REPO) $(TMP)
	cd $(TMP) && git checkout --quiet $(SEEDVR2_COMMIT)
	rm -rf packages/seedvr2_torch
	mkdir -p packages/seedvr2_torch
	cp -r $(TMP)/src $(TMP)/configs_3b $(TMP)/configs_7b $(TMP)/pos_emb.pt $(TMP)/neg_emb.pt $(TMP)/LICENSE packages/seedvr2_torch/
	rm -rf packages/seedvr2_torch/src/interfaces
	find packages/seedvr2_torch -name __pycache__ -type d -prune -exec rm -rf {} +
	printf '"""SeedVR2 inference core vendored from numz/ComfyUI-SeedVR2_VideoUpscaler at %s, without its ComfyUI nodes."""\n' $(SEEDVR2_COMMIT) > packages/seedvr2_torch/__init__.py
	rm -rf $(TMP)
	@echo "✓ Vendored seedvr2_torch updated to $(SEEDVR2_COMMIT)"

H3_MLX_REPO   ?= https://github.com/Argus-AiTeam/minimax-h3-mac.git
# main (2026-08-12)
H3_MLX_COMMIT ?= 3fcc9f8c557cf353400b1acf737b985e524a509f

.PHONY: update-h3-mlx

update-h3-mlx: ## Update vendored minimax_h3_mlx (MiniMax H3 on macOS)
	$(eval TMP := $(shell mktemp -d))
	git clone --quiet $(H3_MLX_REPO) $(TMP)
	cd $(TMP) && git checkout --quiet $(H3_MLX_COMMIT)
	rm -rf packages/minimax_h3_mlx
	cp -r $(TMP)/minimax_h3_mlx packages/minimax_h3_mlx
	cp $(TMP)/scripts/quantize_text_encoder.py packages/minimax_h3_mlx/quantize_text_encoder.py
	cp $(TMP)/LICENSE packages/minimax_h3_mlx/LICENSE
	rm -f packages/minimax_h3_mlx/generate_cli.py packages/minimax_h3_mlx/doctor.py packages/minimax_h3_mlx/cleanroom_smoke.py
	find packages/minimax_h3_mlx -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf $(TMP)
	@echo "✓ Vendored minimax_h3_mlx updated to $(H3_MLX_COMMIT)"

MOSS_REPO     ?= https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT
# main (2026-10-21)
MOSS_REVISION ?= f972bd4b4b9e3852133cc5da954d87bc32cd8fb9

.PHONY: update-moss

update-moss: ## Update vendored moss_image pipeline (MOSS-Image on Windows/Linux)
	rm -rf packages/moss_image
	mkdir -p packages/moss_image
	curl -fsSL -o packages/moss_image/pipeline_moss_image.py $(MOSS_REPO)/resolve/$(MOSS_REVISION)/pipeline_moss_image.py
	curl -fsSL -o packages/moss_image/LICENSE $(MOSS_REPO)/resolve/$(MOSS_REVISION)/LICENSE
	printf '"""MOSS-Image pipeline vendored from OpenMOSS-Team/MOSS-Image-SFT at revision %s."""\n' $(MOSS_REVISION) > packages/moss_image/__init__.py
	@echo "✓ Vendored moss_image updated to $(MOSS_REVISION)"

BIREFNET_REPO     ?= https://huggingface.co/ZhengPeng7/BiRefNet
# main (2026-02-04)
BIREFNET_REVISION ?= e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4

.PHONY: update-birefnet

update-birefnet: ## Update vendored birefnet model code (background removal)
	rm -rf packages/birefnet
	mkdir -p packages/birefnet
	curl -fsSL -o packages/birefnet/birefnet.py $(BIREFNET_REPO)/resolve/$(BIREFNET_REVISION)/birefnet.py
	curl -fsSL -o packages/birefnet/BiRefNet_config.py $(BIREFNET_REPO)/resolve/$(BIREFNET_REVISION)/BiRefNet_config.py
	printf '"""BiRefNet model code vendored from ZhengPeng7/BiRefNet at revision %s."""\n' $(BIREFNET_REVISION) > packages/birefnet/__init__.py
	@echo "✓ Vendored birefnet updated to $(BIREFNET_REVISION)"

YUE2_REPO   ?= https://github.com/multimodal-art-projection/YuE.git
# main (2026-09-29); src/yue2 is byte-identical to the yue2-infer 0.1.5 wheel except its version string and cli.py
YUE2_COMMIT ?= 3acdec8cf1b97603bb696e4c6c4560348ffbb0b7

.PHONY: update-yue2

update-yue2: ## Update vendored yue2 inference kit (YuE2 music)
	$(eval TMP := $(shell mktemp -d))
	git clone --quiet $(YUE2_REPO) $(TMP)
	cd $(TMP) && git checkout --quiet $(YUE2_COMMIT)
	rm -rf packages/yue2
	cp -r $(TMP)/src/yue2 packages/yue2
	rm -f packages/yue2/cli.py packages/yue2/fast.py
	cp $(TMP)/LICENSE $(TMP)/THIRD_PARTY_NOTICES.md packages/yue2/
	cp -r $(TMP)/licenses packages/yue2/licenses
	printf '%s\n' $(YUE2_COMMIT) > packages/yue2/UPSTREAM_COMMIT
	find packages/yue2 -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf $(TMP)
	@echo "✓ Vendored yue2 updated to $(YUE2_COMMIT)"

SA3_REPO   ?= https://github.com/Stability-AI/stable-audio-3.git
# main (2026-10-30)
SA3_COMMIT ?= 5f5e6ac21ae9ff78a6e8847eb7a1dcbe969ffb84

.PHONY: update-sa3

update-sa3: ## Update vendored stable_audio_3 inference package (Stable Audio 3)
	$(eval TMP := $(shell mktemp -d))
	git clone --quiet $(SA3_REPO) $(TMP)
	cd $(TMP) && git checkout --quiet $(SA3_COMMIT)
	rm -rf packages/stable_audio_3
	cp -r $(TMP)/stable_audio_3 packages/stable_audio_3
	rm -rf packages/stable_audio_3/training packages/stable_audio_3/interface packages/stable_audio_3/data/dataset.py packages/stable_audio_3/cli.py
	cp $(TMP)/LICENSE packages/stable_audio_3/LICENSE
	printf '%s\n' $(SA3_COMMIT) > packages/stable_audio_3/UPSTREAM_COMMIT
	find packages/stable_audio_3 -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf $(TMP)
	@echo "✓ Vendored stable_audio_3 updated to $(SA3_COMMIT)"

QWEN_TTS_REPO   ?= https://github.com/andimarafioti/Qwen3-TTS.git
# distribution/qwen-tts-hf, "Prepare qwen-tts-hf 0.1.1.post1": the sources of the qwen-tts-hf 0.1.1.post1 wheel
QWEN_TTS_COMMIT ?= a466278d04ce9693395947906a949f744acd41e8
FASTER_QWEN3_TTS_REPO   ?= https://github.com/andimarafioti/faster-qwen3-tts.git
# v0.5.4 (release commit of the faster-qwen3-tts 0.5.4 wheel)
FASTER_QWEN3_TTS_COMMIT ?= 1094aabd83b367b1b29942769141b0524bed245d

.PHONY: update-qwen3-tts

update-qwen3-tts: ## Update vendored qwen_tts and faster_qwen3_tts (Qwen3-TTS speech on Windows/Linux)
	$(eval TMP := $(shell mktemp -d))
	git clone --quiet $(QWEN_TTS_REPO) $(TMP)/qwen_tts
	cd $(TMP)/qwen_tts && git checkout --quiet $(QWEN_TTS_COMMIT)
	git clone --quiet $(FASTER_QWEN3_TTS_REPO) $(TMP)/faster_qwen3_tts
	cd $(TMP)/faster_qwen3_tts && git checkout --quiet $(FASTER_QWEN3_TTS_COMMIT)
	rm -rf packages/qwen_tts packages/faster_qwen3_tts
	cp -r $(TMP)/qwen_tts/qwen_tts packages/qwen_tts
	rm -rf packages/qwen_tts/cli packages/qwen_tts/__main__.py
	cp $(TMP)/qwen_tts/LICENSE packages/qwen_tts/LICENSE
	printf '%s\n' $(QWEN_TTS_COMMIT) > packages/qwen_tts/UPSTREAM_COMMIT
	cp -r $(TMP)/faster_qwen3_tts/faster_qwen3_tts packages/faster_qwen3_tts
	rm -f packages/faster_qwen3_tts/cli.py
	cp $(TMP)/faster_qwen3_tts/LICENSE packages/faster_qwen3_tts/LICENSE
	printf '%s\n' $(FASTER_QWEN3_TTS_COMMIT) > packages/faster_qwen3_tts/UPSTREAM_COMMIT
	find packages/qwen_tts packages/faster_qwen3_tts -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf $(TMP)
	@echo "✓ Vendored qwen_tts updated to $(QWEN_TTS_COMMIT) and faster_qwen3_tts to $(FASTER_QWEN3_TTS_COMMIT)"

# ——— Frontend ————————————————————————————————————————————

.PHONY: frontend-install frontend-build frontend-static-check frontend-palette-check frontend-hover-check frontend-test frontend-dev

frontend-install: ## Install frontend pnpm dependencies
	pnpm --dir frontend install --frozen-lockfile

frontend-build: ## Build the Svelte frontend into the static dir
	pnpm --dir frontend run build

frontend-static-check: ## Verify packaged SPA artifacts match the frontend build
	$(eval TMP := $(shell mktemp -d))
	@mkdir -p $(TMP)/app
	@if [ -d zvisiongenerator/web/static/app ]; then cp -R zvisiongenerator/web/static/app/. $(TMP)/app/; fi
	pnpm --dir frontend run build
	@diff -qr $(TMP)/app zvisiongenerator/web/static/app >/dev/null || { \
		echo "Packaged SPA artifacts are stale. Run 'make frontend-build' and commit the updated zvisiongenerator/web/static/app files."; \
		diff -qr $(TMP)/app zvisiongenerator/web/static/app; \
		rm -rf $(TMP); \
		exit 1; \
	}
	@test -z "$$(git ls-files --others --exclude-standard -- zvisiongenerator/web/static/app)" || { \
		echo "Packaged SPA artifacts include untracked files:"; \
		git ls-files --others --exclude-standard -- zvisiongenerator/web/static/app; \
		rm -rf $(TMP); \
		exit 1; \
	}
	@rm -rf $(TMP)

# Raw colours in Svelte files bypass the design tokens in frontend/src/app/global.css.
# The mascot artwork is exempt. Set PALETTE_STRICT=0 to warn instead of fail.
PALETTE_STRICT ?= 1
PALETTE_PATTERN := ((^|[^-[:alnum:]])(text|bg|border|ring|divide|placeholder|from|to|via|fill|stroke|outline|accent|shadow|decoration)-(zinc|red|green|emerald|amber|yellow|blue|sky|teal|cyan|rose|pink|orange|purple|violet|indigo|slate|gray|neutral|stone|black|white)(-[0-9]{2,3})?(/[0-9]+)?\b)|(\#[0-9a-fA-F]{3,8}\b)|(rgba?\()

frontend-palette-check: ## Report raw colour classes and literals in Svelte files
	@hits=$$(grep -rnE '$(PALETTE_PATTERN)' frontend/src --include='*.svelte' | grep -v '/Mascot[A-Za-z]*\.svelte:' || true); \
	if [ -n "$$hits" ]; then \
		count=$$(printf '%s\n' "$$hits" | wc -l | tr -d ' '); \
		echo "$$count raw colour use(s) in Svelte files; use the tokens in frontend/src/app/global.css:"; \
		printf '%s\n' "$$hits" | head -20; \
		if [ "$(PALETTE_STRICT)" = "1" ]; then exit 1; fi; \
	fi

frontend-hover-check: ## Fail on :hover styles a touch screen would keep after a tap
	node frontend/scripts/check-hover.mjs

frontend-test: ## TypeScript check + vitest tests
	pnpm --dir frontend run check
	pnpm --dir frontend run test

frontend-dev: ## Start Vite dev server (for development)
	pnpm --dir frontend run dev

# ——— Build ————————————————————————————————————————————————

.PHONY: build clean

build: frontend-build ## Build wheel and sdist
	uv build

clean: ## Remove build artifacts, caches, and venv
	rm -rf build/ dist/ *.egg-info output/ .agent-work/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .venv

# ——— Run ——————————————————————————————————————————————————

.PHONY: run model enhance-bench

run: ## Run ziv-image CLI (use ARGS="..." for arguments)
	uv run ziv-image $(ARGS)

model: ## Run ziv-model CLI (use ARGS="..." for arguments)
	uv run ziv-model $(ARGS)

enhance-bench: ## Run the prompt-enhancer bench (loads a real model; use ARGS="run --label NAME" or ARGS="compare BASE NEW")
	uv run python -m bench.enhance $(ARGS)

# ——— E2E Image ————————————————————————————————————————————

.PHONY: e2e-image e2e-image-zimage e2e-image-klein e2e-image-ideogram

e2e-image-zimage: ## Run Z-Image E2E tests
	@mkdir -p output
	@echo "=== E2E: zimage — smoke test ==="
	uv run ziv-image -m $(ZIMAGE_MODEL) --prompt "a red circle on white background" -W 64 -H 64 -o output --steps 1 --seed 42

e2e-image-klein: ## Run FLUX.2-klein E2E tests
	@mkdir -p output
	@echo "=== E2E: klein — smoke test ==="
	uv run ziv-image -m $(KLEIN_MODEL) --prompt "a red circle on white background" -W 64 -H 64 -o output --steps 2 --seed 42

e2e-image-ideogram: ## Run Ideogram 4 E2E tests
	@mkdir -p output
	@echo "=== E2E: ideogram — smoke test ==="
	uv run ziv-image -m $(IDEOGRAM_MODEL) --prompt "a red circle on white background" -W 256 -H 256 -o output --steps 2 --seed 42

e2e-image: e2e-image-zimage e2e-image-klein e2e-image-ideogram ## Run all image E2E tests

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
