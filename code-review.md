# Code Review for ZVisionGenerator

ZVisionGenerator is a Python 3.14+ CLI tool and FastAPI web server with a Svelte 5 frontend for local AI image and video generation. The tool is entirely local: the server binds to `127.0.0.1` by default, operates without multi-user semantics, and performs all file access on the host machine. The codebase demonstrates strong architectural discipline with clean protocol-based abstractions, atomic stage composition, and consistent modern framework patterns. However, several security gaps related to path validation and DNS rebinding, plus critical configuration issues, require remediation before any network-adjacent deployment. All findings below have been verified against the actual source code.

---

# Security

## 🔧 Defense-in-depth gap: `/docs/assets/` handler lacks containment check
* **Priority**: ⚠️ High
* **File**: `zvisiongenerator/web/server.py` (line ~65)
* **Details**: The `/docs/assets/{asset_name}` route accepts a `str` parameter and constructs a file path without full path resolution and containment validation:
```python
@app.get("/docs/assets/{asset_name}")
async def docs_asset(asset_name: str) -> FileResponse:
    asset_path = Path(__file__).resolve().parents[2] / "docs" / "assets" / asset_name
    if not asset_path.is_file():
        raise HTTPException(status_code=404, detail=f"Unknown asset: {asset_name}")
    return FileResponse(asset_path)
```
The `/media/` route in the same file correctly validates with `resolve_output_asset_path()`, which uses `.resolve()` followed by `.is_relative_to()` to guarantee the final path stays within the configured root. This route does neither. While the `str` parameter type prevents FastAPI from allowing `/` in `asset_name` (blocking multi-hop traversal like `../../etc/passwd` at the HTTP router layer), Python's `pathlib.Path` does not collapse `..` components without an explicit `.resolve()` call. If this parameter were ever changed to a `path:` type to support subdirectories, or if a library version change altered path-matching, the route would become exploitable. The pattern inconsistency is a code-quality concern even if immediate exploitation risk is nil. The same containment pattern used throughout the codebase should apply here for defense-in-depth.
* **Suggested Change**:
```python
@app.get("/docs/assets/{asset_name}")
async def docs_asset(asset_name: str) -> FileResponse:
    asset_root = (Path(__file__).resolve().parents[2] / "docs" / "assets").resolve()
    asset_path = (asset_root / asset_name).resolve()
    if not asset_path.is_relative_to(asset_root) or not asset_path.is_file():
        raise HTTPException(status_code=404, detail="Unknown asset")
    return FileResponse(asset_path)
```

## 🔧 Arbitrary file read via prompt-file read API
* **Priority**: ⚠️ High
* **File**: `zvisiongenerator/web/prompt_files.py` (line ~55, `normalize_prompt_file_path`)
* **Details**: The `normalize_prompt_file_path()` function accepts any filesystem path and validates only the file extension, existence, and no `://` substring:
```python
def normalize_prompt_file_path(path: str, *, accepted_extensions: tuple[str, ...]) -> Path:
    ...
    if candidate.suffix.lower() not in accepted_extensions:
        raise ValueError(...)
    if not candidate.exists():
        raise ValueError(...)
    if not candidate.is_file():
        raise ValueError(...)
    return candidate
```
There is no check that the path lives inside any configured boundary directory. `POST /api/prompt-files/read {"path": "~/.kube/config.yaml"}` returns the full file content as `{"raw_text": "..."}`. The same `normalize_prompt_file_path()` is used by `POST /api/prompt-files/write`, which atomically replaces the file. A DNS rebinding attack (see SEC-04) can trigger this. An attacker can read and overwrite arbitrary YAML files on the user's machine: API tokens stored in YAML format, custom configuration files, and other secrets. The HuggingFace token file (`~/.cache/huggingface/token`) is not YAML so it is safe, but any YAML-formatted secret the user stores is readable and overwritable. OWASP Path Traversal guidance (CWE-22) identifies directory confinement (not extension filtering alone) as the required primary control for file APIs.
* **Suggested Change**:
```python
def normalize_prompt_file_path(path: str, *, accepted_extensions: tuple[str, ...], allowed_root: Path | None = None) -> Path:
    if allowed_root is None:
        allowed_root = (Path.home() / ".ziv" / "prompts").resolve()
    else:
        allowed_root = allowed_root.resolve()
    
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_relative_to(allowed_root):
        raise ValueError(f"Path '{path}' is outside allowed root '{allowed_root}'")
    if candidate.suffix.lower() not in accepted_extensions:
        raise ValueError(...)
    if not candidate.exists():
        raise ValueError(...)
    if not candidate.is_file():
        raise ValueError(...)
    return candidate
```

## 🔧 Client-controlled output directory: generated files written to attacker-chosen path
* **Priority**: ⚠️ High
* **File**: `zvisiongenerator/web/server.py` (line ~316 in `_submit_image_job`, line ~439 in `_submit_video_job`)
* **Details**: Both job submission endpoints accept an `output` form field that overrides the configured output directory without any validation:
```python
output_dir = _resolve_output_dir(_text_or_default(form, "output", web_config.output_dir))
```
The `_resolve_output_dir()` function (config_contract.py ~line 103) expands the path and creates it if missing:
```python
output_dir = Path(text).expanduser()
if not output_dir.is_absolute():
    output_dir = (Path.cwd() / output_dir).resolve()
...
output_dir.mkdir(parents=True, exist_ok=True)
return output_dir
```
There is no check that the resolved path is inside the configured `web_config.output_dir`. This allows every submission to override where generated files land. Combined with DNS rebinding (SEC-04), an attacker-controlled webpage can: (1) write generated image files to arbitrary paths the server process can write to (`~/Documents/`, `~/.ssh/`, etc.), (2) create directories anywhere, and (3) write the uploaded reference image (stored in `.web_uploads` subdirectory) inside the attacker-controlled `output_dir`. The `output_dir` setting is already configurable through the legitimate `GET /api/config` endpoint and is presented in the UI — there is no UX reason for it to also be a per-request parameter.
* **Suggested Change**:
```python
# In _submit_image_job and _submit_video_job:
# Remove the form field lookup entirely
output_dir = _resolve_output_dir(web_config.output_dir)  # Never from form
```

## 💭 No authentication on any endpoint; DNS rebinding attack surface
* **Priority**: ⚠️ High
* **File**: `zvisiongenerator/web/server.py` (all routes)
* **Details**: Every route in the application is entirely unauthenticated: `/api/generate`, `/api/prompt-files/read`, `/api/prompt-files/write`, `/api/gallery/delete`, `POST /api/config`, `/api/picker`, `/api/models/convert`, `/api/models/import-lora-*`. The default bind address (`127.0.0.1`) provides implicit network-layer isolation in the common case, but DNS rebinding removes that protection. An attacker-controlled website visited by the user can instruct the browser to make requests to `http://127.0.0.1:<port>/api/...` once the browser has resolved the attacker's domain to `127.0.0.1` via a crafted TTL-0 DNS response. This is a documented, actively exploited technique (example: Glances CVE GHSA-hhcg-r27j-fhv9 exploited a FastAPI local tool via DNS rebinding). The exploitable chain: DNS rebinding → SEC-02 (read arbitrary YAML files) + SEC-03 (write generated files to arbitrary path). The minimum mitigation for a local-only tool is `TrustedHostMiddleware` that validates the `Host` header, rejecting requests whose `Host` is not a localhost address. This breaks DNS rebinding by ensuring the browser's post-rebind request fails because the `Host` header will be the attacker's domain, not `localhost` or `127.0.0.1`.
* **Suggested Change**:
```python
from starlette.middleware.trustedhost import TrustedHostMiddleware

# Add this in the startup sequence, before defining routes
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "::1"]
)

# Emit a startup warning if --host is not 127.0.0.1
if host != "127.0.0.1":
    warnings.warn(
        "ZVisionGenerator is starting on a non-localhost address. "
        "DNS rebinding attacks may be possible. For local use, bind to 127.0.0.1.",
        stacklevel=2
    )
```

## 🔧 AppleScript injection via newline character in `initial_path`
* **Priority**: ⚠️ High
* **File**: `zvisiongenerator/web/path_picker.py` (line ~97, `_pick_macos` and `_macos_default_location`)
* **Details**: The `_macos_default_location()` function sanitizes double-quote characters but not newlines:
```python
def _macos_default_location(initial_path: str | None) -> str | None:
    ...
    return str(candidate.resolve()).replace('"', '\\"')
```
This sanitized string is then interpolated into an AppleScript:
```python
def _pick_macos():
    default_location = _macos_default_location(initial_path)
    script = f'set defaultLocation to POSIX file "{default_location}"\nset chosenItem to {command} ...'
    result = subprocess.run(["osascript", "-e", script], ...)
```
Unix filesystems allow directory names to contain literal newline characters. If `initial_path` resolves to a directory with a newline in its name, the `default_location` string will contain `\n`. The f-string interpolation places this unescaped newline into the AppleScript source, splitting the script into multiple statements. For example: `set defaultLocation to POSIX file "/Users/knut/dir\nwith\nnewlines"` becomes:
```
set defaultLocation to POSIX file "/Users/knut/dir
with
newlines"
```
The newline becomes a statement separator in AppleScript; anything injected after the first statement executes as a separate AppleScript command. Combined with an attacker-controlled `initial_path` via DNS rebinding to `POST /api/picker`, and assuming a directory with a newline in its name exists (or can be created on a shared filesystem), arbitrary AppleScript execution is possible. The practical severity is low because: (1) the precondition requires a directory with a literal newline name, which is unusual, and (2) `_macos_default_location()` only returns non-None if the directory already exists on the filesystem. However, the vulnerability is straightforward to fix.
* **Suggested Change**:
```python
def _macos_default_location(initial_path: str | None) -> str | None:
    ...
    resolved = str(candidate.resolve())
    # Reject paths containing control characters
    if any(ord(c) < 32 for c in resolved):
        return None
    return resolved.replace('"', '\\"')
```

## 🔧 No size limit on reference image uploads
* **Priority**: ⚠️ High
* **File**: `zvisiongenerator/web/server.py` (line ~673, `_save_uploaded_reference_image`)
* **Details**: The reference image upload handler reads the entire multipart body into memory without a cap:
```python
uploaded_file.file.seek(0)
candidate.write_bytes(uploaded_file.file.read())
```
There is no size limit passed to `read()`. A 2 GB upload allocates 2 GB of process memory before writing to disk. Because `multipart/form-data` decoding happens before the route handler runs (at the Starlette framework layer), there is no streaming from the application's perspective — the full body is buffered. A DOS attacker (or user error) can exhaust the server process's memory. Note: Pillow's decompression bomb protection IS active by default (`Image.MAX_IMAGE_PIXELS = 89,478,485`), and a crafted PNG inflating to a huge uncompressed size will trigger `DecompressionBombError` inside the `image.verify()` call. However, this check runs *after* the raw bytes are already allocated in memory and written to disk. The raw read is the immediate issue. A practical reference image (photographs, screenshots) is typically 1–20 MB; there is no UX reason to accept unlimited sizes.
* **Suggested Change**:
```python
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
uploaded_file.file.seek(0)
data = uploaded_file.file.read(MAX_UPLOAD_BYTES + 1)
if len(data) > MAX_UPLOAD_BYTES:
    raise ValueError("Reference image must not exceed 50 MB.")
candidate.write_bytes(data)
```

## 📝 Internal stack details exposed in HTTP error responses
* **Priority**: 🟢 Low
* **File**: `zvisiongenerator/web/server.py` (multiple handlers throughout)
* **Details**: All caught exceptions forward their string representation verbatim in HTTP responses. Application-level exceptions (`ValueError`, `FileNotFoundError`) have carefully written messages (e.g., `"Path '...' for 'image_path' must be an existing host-local file"`). However, `RuntimeError` and other exceptions from library code (mflux, diffusers, Pillow) can expose absolute host paths (`/Users/knut/.ziv/models/`), model directory layouts, CUDA device details, and Python tracebacks in `str(exc)`. When combined with DNS rebinding, an attacker reading the gallery API (SEC-04) can map the host filesystem from error messages. This is information disclosure that aids post-exploitation reconnaissance.
* **Suggested Change**:
```python
_SAFE_EXCEPTION_TYPES = (ValueError, FileNotFoundError)
detail = str(exc) if isinstance(exc, _SAFE_EXCEPTION_TYPES) else f"Internal error [{type(exc).__name__}]. Check server logs."
```

## 📝 Absolute host paths exposed in gallery API responses
* **Priority**: 🟢 Low
* **File**: `zvisiongenerator/web/gallery.py` (line ~175, `gallery_asset_to_json`)
* **Details**: The `gallery_asset_to_json()` function returns the `reference_image_path` field from the generation sidecar JSON verbatim:
```python
return {
    ...
    "image_path": asset.reference_image_path,  # raw absolute host path from sidecar
    ...
}
```
The sidecar file is created at generation time and contains absolute paths (e.g., `/Users/knut/Pictures/reference.jpg`). Any browser tab connected to the `/api/gallery` endpoint can read this path. On its own this is low severity (the path is visible to the local user anyway), but it contributes to filesystem mapping in a DNS rebinding scenario.
* **Suggested Change**:
```python
# In gallery_asset_to_json():
image_path = None
if asset.reference_image_path:
    try:
        ref_path = Path(asset.reference_image_path).resolve()
        output_root = web_config.output_dir.resolve()
        if ref_path.is_relative_to(output_root):
            # Return only the root-relative portion
            image_path = ref_path.relative_to(output_root).as_posix()
    except (ValueError, RuntimeError):
        # Path is outside output root or invalid; omit it
        pass
return {
    ...
    "image_path": image_path,
    ...
}
```

---

# Code Quality

## 🔧 `torch` installed unconditionally on macOS (~2 GB wasted)
* **Priority**: ⚠️ High
* **File**: `pyproject.toml` (line ~35)
* **Details**: The dependency specification includes `"torch>=2.11.0",` with no platform marker:
```toml
[project.dependencies]
"torch>=2.11.0",
```
The macOS backend (`backends/image_mac.py`) uses `mflux`, which runs on Apple's MLX framework — a completely separate tensor library from PyTorch. The Windows backend (`backends/image_win.py`) uses `diffusers` + `accelerate`, which require PyTorch. The platform dispatch in `backends/__init__.py` checks `sys.platform == "darwin"` and loads only the macOS backend, so PyTorch code is never imported on macOS. Yet `torch` is always installed on macOS because the dependency has no `sys_platform != 'darwin'` marker. The `[tool.uv.sources]` table correctly routes torch to the CUDA index on non-darwin platforms but does not exclude it on darwin. The macOS CPU wheel for PyTorch 2.11 is approximately 190 MB compressed, expanding to ~600 MB. Every `uv sync` on macOS fetches and installs unused weight. The contrast to nearby dependencies is stark: `bitsandbytes` is already correctly gated with `; sys_platform != 'darwin'` in the same file, setting a clear pattern. This is a one-character fix.
* **Suggested Change**:
```toml
[project.dependencies]
"torch>=2.11.0; sys_platform != 'darwin'",
```

## ♻️ `run_batch()` is ~350 lines mixing 8+ unrelated concerns
* **Priority**: ⚠️ High
* **File**: `zvisiongenerator/image_runner.py` (line ~47)
* **Details**: The project's own architecture guidelines (`.github/copilot-instructions.md`) mandate: "Each function does one thing. Each module owns one concept. If a docstring needs 'and', split the unit." The `run_batch()` function violates this principle on every dimension. Inside ~350 lines, it handles: (1) resolving config scalars (seed range, sharpening, contrast/saturation amounts), (2) resolving dimensions and per-preset upscale drift warnings, (3) negative-prompt suppression logic, (4) interactive-controls lifecycle (`SkipSignal.start()` / `skip.stop()`), (5) triple-nested loop bookkeeping (run × set × prompt indices), (6) seed generation per iteration, (7) ETA calculation, (8) building both a display `_display_request` AND a separate real `request` dataclass, (9) the retry loop on skip/repeat signals, and (10) progress callback emission. Understanding whether a given variable belongs to "display preparation" or "actual generation" requires reading the entire function. Testing any single behavior (e.g., ETA calculation) in isolation is impossible — every test must bring along the full generation infrastructure. This is a prime candidate for decomposition.
* **Suggested Change**:
```python
def _resolve_batch_config(args, config, model_info) -> dict[str, Any]:
    """Resolve all configuration scalars from args, config, and model_info."""
    # Return seed_mode, sharpening_amount, contrast_amount, etc.
    ...

def _run_single_generation(
    backend, model, prompt, seed, ..., 
    *, display_config, real_config
) -> tuple[Image.Image, float]:
    """Run a single prompt with retry logic. Returns generated image and generation time."""
    ...

def run_batch(backend, model, args, ...):
    """Orchestrate batch generation: resolve config, loop over iterations, report progress."""
    config = _resolve_batch_config(args, config, model_info)
    image_times = []
    for run_idx in range(num_runs):
        for set_idx in range(num_sets):
            for prompt_idx, prompt in enumerate(prompts):
                img, elapsed = _run_single_generation(backend, model, prompt, ...)
                image_times.append(elapsed)
                _emit_progress(...)
    ...
```

## 🔧 `strip_audio_stage` swallows `FileNotFoundError` from missing ffmpeg
* **Priority**: ⚠️ High
* **File**: `zvisiongenerator/workflows/video_stages.py` (line ~124)
* **Details**: The `strip_audio_stage()` function catches `subprocess.CalledProcessError` but not `FileNotFoundError`:
```python
def strip_audio_stage(request, artifacts):
    if not request.no_audio:
        return StageOutcome.success
    if artifacts.video_path is None:
        return StageOutcome.success
    try:
        strip_audio(artifacts.video_path)
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode("utf-8", "replace").strip() if e.stderr else str(e)
        warnings.warn(f"ffmpeg strip-audio failed: {err_msg}", stacklevel=2)
        return StageOutcome.failed
    return StageOutcome.success
```
If `ffmpeg` is not on `PATH`, `subprocess.run(["ffmpeg", ...])` raises `FileNotFoundError` before the process is launched. This exception propagates out of `strip_audio_stage()` entirely because it is not caught. The exception bubbles up through `GenerationWorkflow.run()` (which only handles `StageOutcome` return values) and into `web_runner._run_target()`, where a bare `except Exception` publishes the failure. The job fails cleanly from the server's perspective, but:
1. The error message is the raw `FileNotFoundError` string: `"No such file or directory: 'ffmpeg'"` — not user-friendly.
2. The intended `StageOutcome.failed` path with its `warnings.warn()` call is bypassed.
3. The behavior is inconsistent: intentional errors use `StageOutcome.failed` with a warning; unexpected errors produce raw Python exceptions.
* **Suggested Change**:
```python
try:
    strip_audio(artifacts.video_path)
except (subprocess.CalledProcessError, FileNotFoundError) as e:
    if isinstance(e, FileNotFoundError):
        err_msg = "ffmpeg not found on PATH"
    else:
        err_msg = e.stderr.decode("utf-8", "replace").strip() if e.stderr else str(e)
    warnings.warn(f"ffmpeg strip-audio failed: {err_msg}", stacklevel=2)
    return StageOutcome.failed
```

## 🔧 `backend: Any` in `run_video_batch()` — `VideoBackend` protocol ignored
* **Priority**: ⚠️ High
* **File**: `zvisiongenerator/video_runner.py` (line 22)
* **Details**: The function signature uses `Any` instead of the explicit `VideoBackend` protocol:
```python
def run_video_batch(backend: Any, ...):  # Bad
```
Compare to the image runner (line 47):
```python
def run_batch(backend: ImageBackend, ...):  # Good
```
The codebase defines a `VideoBackend` Protocol in `core/video_backend.py` with explicit methods (`name`, `load_model()`, `text_to_video()`, `image_to_video()`). The docstring for `run_video_batch()` even claims "satisfies VideoBackend Protocol." Using `Any` means: (1) static analysis (mypy, pyright, ruff) cannot verify that any backend passed here implements the Protocol, (2) IDE auto-complete and cross-reference navigation break, and (3) the asymmetry between image and video runners is confusing. The Protocol is runtime-checkable (uses `typing.Protocol`) and there is no circular import preventing this change.
* **Suggested Change**:
```python
from zvisiongenerator.core.video_backend import VideoBackend

def run_video_batch(backend: VideoBackend, ...):
```

## 🔧 `confirm()` blocks the JavaScript event loop
* **Priority**: ⚠️ High
* **File**: `frontend/src/features/gallery/GalleryPage.svelte` (line ~123 in `deleteSelected`, line ~132 in `deleteSingle`)
* **Details**: Gallery deletion uses the synchronous `window.confirm()` dialog:
```typescript
async function deleteSelected(): Promise<void> {
    if (!confirm(`Delete ${selectedCount} selected asset${selectedCount !== 1 ? 's' : ''}?`)) return;
```
`window.confirm()` is a synchronous API that halts all JavaScript execution until the user responds. While the dialog is open:
- Any ongoing network requests are not processed (their microtasks are queued but blocked).
- SSE events from an active generation job stop updating the UI.
- Svelte's reactivity system is frozen — `$state` updates are queued but the DOM is not refreshed.
- In some browsers, other tabs to the same origin are also blocked.

The larger UX concern is inconsistency: the rest of the application uses `addToast()` for all feedback. The same `deleteSingle` function calls `addToast('Deleted', 'success')` on completion — the only departure is this modal confirmation. MDN explicitly recommends against `confirm()` and suggests `<dialog>` as the modern replacement. The expected pattern for this codebase is a `<dialog>` element or a lightweight inline component.
* **Suggested Change**:
```typescript
async function deleteSelected(): Promise<void> {
    // Use a reusable ConfirmDialog component instead of window.confirm()
    const confirmed = await showConfirmDialog({
        title: 'Delete Selected Assets',
        message: `Delete ${selectedCount} selected asset${selectedCount !== 1 ? 's' : ''}?`
    });
    if (!confirmed) return;
    // ... proceed with deletion
}
```

## 🔧 `loadMorePages` silently drops all load errors
* **Priority**: ⚠️ High
* **File**: `frontend/src/features/gallery/GalleryPage.svelte` (line ~85, `loadMorePages`)
* **Details**: The infinite-scroll pagination handler silently ignores errors:
```typescript
async function loadMorePages(): Promise<void> {
    if (loadingMore || !hasMore) return;
    loadingMore = true;
    try {
        const result = await getGallery(page + 1, mediaFilter, sortOrder);
        assets = [...assets, ...result.assets];
        page = result.page;
        totalPages = result.total_pages;
    } catch {
        // ignore load-more errors silently
    } finally {
        loadingMore = false;
    }
}
```
When `getGallery()` fails (network timeout, server 500, backend crash), the catch block sets nothing. The `loadingMore` flag is reset to `false`, and no items load. The `IntersectionObserver` that triggers `loadMorePages()` immediately fires again (the sentinel element is still in the viewport), causing another silent failure loop. The gallery appears to have loaded all items when it has actually stalled, with no indication to the user. Contrast this with `loadPage()` (initial load), which correctly sets `error = e instanceof Error ? e.message : 'Failed to load gallery'` and renders it to the UI. The asymmetry is a bug.
* **Suggested Change**:
```typescript
} catch (e) {
    addToast(
        'Failed to load more assets. Try scrolling back up and down.',
        'error'
    );
}
```

## ♻️ Duplicate `ImageGenerationRequest` construction per batch iteration (display vs. real)
* **Priority**: ⚠️ High
* **File**: `zvisiongenerator/image_runner.py` (line ~194)
* **Details**: The batch loop constructs `ImageGenerationRequest` twice per iteration — once for display, once for actual generation:
```python
# Build a lightweight request just for display info
_display_request = ImageGenerationRequest(
    backend=backend,
    model=model,
    # ... many fields
)
format_generation_info(_display_request, ...)

# ... later ...
request = ImageGenerationRequest(
    # ... same fields plus seed, scheduler class, etc.
)
```
`ImageGenerationRequest` is a frozen dataclass — all fields are validated and stored on construction. Creating a full `_display_request` instance whose only purpose is to pass to `format_generation_info()` doubles allocation per iteration. More importantly, the two construction sites can drift: if a field is added to the dataclass, it must be kept in sync at both sites. The comment "the real request is built below" warns of this danger, which is itself a code smell. Looking at `format_generation_info()` in `utils/console.py`, it accepts an `ImageGenerationRequest` and formats only a small subset of its fields. The correct fix is to pass those specific fields directly to `format_generation_info()` rather than constructing a full frozen dataclass just to destructure it.
* **Suggested Change**: Modify `format_generation_info()` to accept only the fields it needs:
```python
def format_generation_info(
    model: str,
    workflow: str,
    prompt: str,
    # ... other specific fields
) -> str:
    # Format without needing the full ImageGenerationRequest
    ...
```
Then in the batch loop, call it with just the needed fields:
```python
format_generation_info(model, workflow, prompt, ...)
```

## ♻️ Four private wrapper functions in `server.py` delegate nothing
* **Priority**: ⚠️ High
* **File**: `zvisiongenerator/web/server.py` (lines ~284–295)
* **Details**: Four functions are pure pass-throughs with no added logic:
```python
def _canonicalize_workflow(value: Any, *, fallback: str | None = None) -> str | None:
    return canonicalize_workflow(value, fallback=fallback)

def _default_workflow_for_mode(mode: str) -> str:
    return default_workflow_for_mode(mode)

def _build_workflow_contract() -> dict[str, Any]:
    return build_workflow_contract()

def _build_workspace_bootstrap_view(web_config: WebUiConfig) -> dict[str, Any]:
    return build_workspace_bootstrap_view(web_config)
```
These appear throughout route handlers (e.g., `_canonicalize_workflow(...)` is called 5+ times) as private namespace wrappers around imported symbols, adding no validation, argument transformation, or monkeypatching seam. They expose confusion: callers must trace through two function definitions to understand what happens. A new contributor would reasonably assume the private prefix adds logic — but does not. Compare this to `_build_workspace_response()` and `_build_models_response()` in the same file, which DO add value by threading additional arguments. Only the four listed are pure delegates. The cost is unnecessary cognitive load.
* **Suggested Change**: Remove the wrapper functions entirely and call the imported functions directly:
```python
# Before:
_canonicalize_workflow(value, fallback=fallback)

# After:
canonicalize_workflow(value, fallback=fallback)
```

## 🔧 Video stages use deferred local import; image stages do not
* **Priority**: ⚠️ High
* **File**: `zvisiongenerator/workflows/__init__.py` (line ~44)
* **Details**: Video stages are imported inside `build_video_workflow()`:
```python
def build_video_workflow(args: argparse.Namespace) -> GenerationWorkflow:
    from zvisiongenerator.workflows.video_stages import (
        resolve_prompt_stage as video_resolve_prompt,
        generate_filename_stage,
        text_to_video_stage,
        image_to_video_stage,
        strip_audio_stage,
        log_video_stage,
    )
```
While image stages are imported at the module top level:
```python
from zvisiongenerator.workflows.image_stages import (
    resolve_prompt_stage,
    # ...
)
```
Deferred local imports are used to break circular dependencies or defer heavy runtime imports. Neither applies here: `video_stages.py` imports `VideoGenerationRequest` from `core.video_types` and utilities — no circular dependency. The stages themselves do not import heavy runtime (torch, mflux, diffusers). The asymmetry is unexplained and creates ambiguity: is there a circular import being avoided? Is `video_stages` intentionally lazy-loaded? Future developers cannot answer from the code alone. The inconsistency is a maintainability cost.
* **Suggested Change**: Move the video stages import to the module top level to match image stages:
```python
from zvisiongenerator.workflows import video_stages  # or import individual stages
```
Or add a comment explaining the deferral if there is an actual reason.

## 🔧 `successful_generations` is initialized but never incremented
* **Priority**: 🟢 Low
* **File**: `zvisiongenerator/image_runner.py` (line ~133)
* **Details**: The variable is declared but never used:
```python
image_times: list[float] = []
successful_generations = 0
failed_generations = 0
```
`image_times` is appended to on each successful generation. `failed_generations` is incremented in the exception handler. `successful_generations` is never incremented. The batch summary uses `len(image_times)` as the success count. The variable is dead code — it holds `0` for the entire function execution. A reader trying to understand success-tracking will be confused by this unused variable, creating the false impression that `successful_generations` and `len(image_times)` track the same thing independently.
* **Suggested Change**: Remove the line entirely:
```python
# deleted: successful_generations = 0
failed_generations = 0  # Keep this; it IS used
```

## ⛏️ `random.Random()` instance allocated per `expand_random_choices()` call
* **Priority**: 🟢 Low
* **File**: `zvisiongenerator/utils/prompt_compose.py` (line ~74)
* **Details**: A new `Random` instance is allocated on every call:
```python
def expand_random_choices(text: str) -> str:
    rng = random.Random()
    while _INNERMOST_RE.search(text):
        text = _INNERMOST_RE.sub(
            lambda m: rng.choice(m.group(1).split("|")),
            text,
        )
    return text.strip()
```
`random.Random()` with no seed uses the system time (or `os.urandom()`) as its seed — the same source as the module-level `random` singleton. A new instance is allocated per call purely to call `.choice()` a few times, then it is discarded. The module-level `random.choice()` does exactly the same thing. The only reason to create a private `Random` instance would be for deterministic seeding (e.g., `random.Random(seed)` for reproducible expansion). That is not the case here. Idiomatic Python is to use `random.choice()` directly.
* **Suggested Change**:
```python
def expand_random_choices(text: str) -> str:
    while _INNERMOST_RE.search(text):
        text = _INNERMOST_RE.sub(
            lambda m: random.choice(m.group(1).split("|")),
            text,
        )
    return text.strip()
```

## 💭 `job_id` and `id` are the same field in the job snapshot; `??` fallback is dead code
* **Priority**: 🟢 Low
* **File**: `frontend/src/lib/state/job.svelte.ts` (line ~116) and `zvisiongenerator/web/job_contract.py` (line ~46)
* **Details**: The backend always populates both fields with the same value:
```python
# job_contract.py
snapshot = {
    "id": job_id,
    "job_id": job_id,
    ...
}
```
The frontend's coalescing operator is dead code:
```typescript
// job.svelte.ts
writeActiveJobId(snapshot.job_id ?? snapshot.id);
```
The `??` fallback will never reach `snapshot.id` because `snapshot.job_id` is always present and non-null. The dual field exists (presumably to satisfy two different consumers expecting different names), but both are always set to the same value. The fallback misleads future maintainers into thinking there is a scenario where `job_id` can be absent — there is not. If the backend contract is ever cleaned up to remove one field, this fallback becomes actively wrong.
* **Suggested Change**: Use only the primary field, without the fallback:
```typescript
writeActiveJobId(snapshot.job_id);  // or snapshot.id, consistently
```

## ⛏️ `_prevWorkflow` mutation inside `$effect` is a Svelte 4 idiom
* **Priority**: 🟢 Low
* **File**: `frontend/src/features/workspace/WorkspacePage.svelte` (line ~133)
* **Details**: The code mutates a `let` variable inside `$effect` to track the previous iteration:
```typescript
let _prevWorkflow: Workflow | null = null;
$effect(() => {
    const currentWorkflow = draft.state.workflow;
    if (_prevWorkflow !== null && context !== null && currentWorkflow !== _prevWorkflow) {
        draft.onWorkflowChange(currentWorkflow, context);
    }
    _prevWorkflow = currentWorkflow;
});
```
This is the Svelte 4 `afterUpdate` pattern. In Svelte 5, the `$effect` dependency-tracking means this works correctly (the `_prevWorkflow` mutation doesn't re-trigger the effect because `_prevWorkflow` is not a `$state` value), but it uses an older idiom rather than the modern Svelte 5 approach. Svelte 5 documentation notes that `$effect` is an "escape hatch" — prefer `$derived` for reactive values and event handlers for user actions. The real intent here is: "when the user changes workflow, apply clearing logic." An `onchange` handler on the workflow dropdown would be more semantically clear. Note: this is a nitpick. The current code is not incorrect. Svelte 5 has no built-in previous-value primitive, so the mutation-in-effect pattern is the only available mechanism when event handlers aren't applicable.
* **Suggested Change**: If semantically appropriate, use an event handler instead:
```typescript
function handleWorkflowChange(event: Event) {
    const newWorkflow = (event.target as HTMLSelectElement).value;
    draft.onWorkflowChange(newWorkflow, context);
}
```

## ⛏️ Silent `catch {}` block masks backend SSE serialization bugs
* **Priority**: 🟢 Low
* **File**: `frontend/src/lib/api/sse.ts` (line ~63)
* **Details**: `JSON.parse` errors are silently ignored:
```typescript
es.addEventListener(type, (event: Event) => {
    try {
        const data = JSON.parse((event as MessageEvent).data) as SSEEvent;
        handleEvent(type, data);
    } catch {
        // ignore malformed events
    }
});
```
`JSON.parse` throws `SyntaxError` when the event data is not valid JSON. The silent catch means that if the backend accidentally emits a plain-text debug line instead of a JSON frame (which can happen if a `print()` statement is introduced in a worker thread, or if an exception escapes before the SSE formatter), the frontend silently ignores it. During development, this makes backend bugs hard to find because the SSE stream appears to silently stop progressing without any visible error in DevTools. The `_ThreadAwareTextStream` in `web_runner.py` mutes worker-thread stdout to prevent this, but the catch is masking the symptom. In production, silent failures are acceptable, but in development, logging makes debugging easier without affecting behavior.
* **Suggested Change**:
```typescript
} catch (e) {
    if (import.meta.env.DEV) {
        console.warn('[SSE] malformed event ignored', type, (event as MessageEvent).data, e);
    }
}
```

---

# Testing Gaps

## 🔧 No tests for `gallery.py` — path-traversal guards and serialization logic entirely uncovered
* **Priority**: ⚠️ High
* **File**: N/A (missing test file: `tests/test_gallery.py`)
* **Details**: `zvisiongenerator/web/gallery.py` contains the security-critical `normalize_asset_id()` and `resolve_output_asset_path()` functions. These are the primary defense mechanisms against path traversal for the `/media/` and `/api/gallery/delete` routes. `normalize_asset_id()` rejects `..` components, absolute paths, Windows drive letters, and staging directory names (`_STAGING_DIR_NAMES`). `resolve_output_asset_path()` calls `.resolve().is_relative_to()` to guarantee confinement. A future refactor that accidentally removes one of these checks (e.g., dropping the `..` validation) would silently introduce a path traversal vulnerability with zero automated detection. Beyond security: `gallery_asset_to_json()` contains a multi-step `reuse_state` resolution involving `workflow_media_mismatch`, `missing_reference_image`, and `model_not_configured` fallback chains — each conditional branch affects the reuse URL. `build_gallery_page_json()` has pagination arithmetic with `max(1, ...)` boundary guards. Both are stateful logic requiring unit tests. No `tests/test_gallery.py` exists.
* **Suggested Change**: Create `tests/test_gallery.py` with comprehensive tests:
```python
import pytest
from zvisiongenerator.web.gallery import normalize_asset_id, resolve_output_asset_path

def test_normalize_asset_id_rejects_traversal():
    with pytest.raises(ValueError):
        normalize_asset_id("../etc/passwd")
    with pytest.raises(ValueError):
        normalize_asset_id("..\\..\\windows\\system32")

def test_resolve_output_asset_path_confines_to_root(tmp_path):
    output_root = tmp_path / "output"
    output_root.mkdir()
    
    # Valid asset inside root
    valid_id = "abc123.png"
    result = resolve_output_asset_path(valid_id, output_root)
    assert result.is_relative_to(output_root)
    
    # Traversal attempt rejected
    with pytest.raises(ValueError):
        resolve_output_asset_path("../../../etc/passwd", output_root)
```

## 🔧 `strip_audio_stage` has zero test coverage
* **Priority**: ⚠️ High
* **File**: `tests/test_video_stages.py`
* **Details**: `strip_audio_stage` is not imported or tested in `tests/test_video_stages.py`. The function has three code paths that require coverage: (1) `request.no_audio == False` → immediate `StageOutcome.success` return (no-op), (2) normal `strip_audio` invocation succeeds → `StageOutcome.success` with progress callback, (3) `subprocess.CalledProcessError` → `StageOutcome.failed` with warning. CQ-03 identifies a fourth path (`FileNotFoundError` from missing ffmpeg) that is currently unhandled. Without a test for this stage, the CQ-03 fix will go unvalidated. The fix must include a corresponding test.
* **Suggested Change**: Add tests to `tests/test_video_stages.py`:
```python
from unittest.mock import MagicMock, patch
from zvisiongenerator.workflows.video_stages import strip_audio_stage
from zvisiongenerator.core.runner_outcome import StageOutcome

def test_strip_audio_stage_no_audio_request_skips():
    request = MagicMock(no_audio=False)
    artifacts = MagicMock()
    outcome = strip_audio_stage(request, artifacts)
    assert outcome == StageOutcome.success

def test_strip_audio_stage_no_video_path_skips():
    request = MagicMock(no_audio=True)
    artifacts = MagicMock(video_path=None)
    outcome = strip_audio_stage(request, artifacts)
    assert outcome == StageOutcome.success

@patch('zvisiongenerator.workflows.video_stages.strip_audio')
def test_strip_audio_stage_succeeds(mock_strip):
    request = MagicMock(no_audio=True)
    artifacts = MagicMock(video_path="/path/to/video.mp4")
    outcome = strip_audio_stage(request, artifacts)
    assert outcome == StageOutcome.success
    mock_strip.assert_called_once_with("/path/to/video.mp4")

@patch('zvisiongenerator.workflows.video_stages.strip_audio')
def test_strip_audio_stage_ffmpeg_not_found(mock_strip):
    mock_strip.side_effect = FileNotFoundError("ffmpeg not found")
    request = MagicMock(no_audio=True)
    artifacts = MagicMock(video_path="/path/to/video.mp4")
    with pytest.warns(UserWarning, match="ffmpeg not found"):
        outcome = strip_audio_stage(request, artifacts)
    assert outcome == StageOutcome.failed
```

## 🔧 `/media/` path-traversal protection is never integration-tested
* **Priority**: ⚠️ High
* **File**: `tests/test_web_server.py`
* **Details**: The `normalize_asset_id()` + `resolve_output_asset_path()` guards are the only runtime protection against traversal attempts on the `/media/{asset_id}` route. No integration test sends a traversal payload and asserts a `404`. The expected test cases: `GET /media/../../../etc/passwd`, `GET /media/%2e%2e%2f..%2fetc%2fpasswd` (URL-encoded traversal), `GET /media//etc/passwd` (absolute path). Each should return `404 Not Found`. Without these tests, a future refactor that accidentally bypasses the guard (e.g., removing the `.resolve().is_relative_to()` check) ships undetected.
* **Suggested Change**: Add to `tests/test_web_server.py`:
```python
@pytest.mark.asyncio
async def test_media_route_rejects_traversal_relative(client):
    response = await client.get("/media/../../../etc/passwd")
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_media_route_rejects_traversal_absolute(client):
    response = await client.get("/media//etc/passwd")
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_media_route_rejects_traversal_urlencoded(client):
    response = await client.get("/media/%2e%2e%2f..%2fetc%2fpasswd")
    assert response.status_code == 404
```

## 🔧 Gallery API routes untested at the route level
* **Priority**: ⚠️ High
* **File**: `tests/test_web_server.py`
* **Details**: The `/api/gallery` (paginated listing with `filter` and `sort_order` query params) and `/api/gallery/delete` (multi-asset delete) routes have no dedicated integration test. The nearest test (`test_phase_a_routes_share_config_and_path_authority`) monkeypatches `list_gallery_assets` to return `[]` but never exercises: (1) pagination arithmetic, (2) filter parameter forwarding to the backend, (3) sort order forwarding, (4) multi-asset delete behavior through the HTTP layer. The `gallery_asset_to_json()` serialization (including reuse-state chains) is not tested at the route level.
* **Suggested Change**: Add route-level tests:
```python
@pytest.mark.asyncio
async def test_gallery_route_pagination(client, mock_backend):
    response = await client.get("/api/gallery?page=1&media_filter=image&sort_order=newest")
    assert response.status_code == 200
    data = response.json()
    assert "assets" in data
    assert "page" in data
    assert "total_pages" in data

@pytest.mark.asyncio
async def test_gallery_delete_route(client, mock_backend):
    response = await client.post("/api/gallery/delete", json={"ids": ["asset1", "asset2"]})
    assert response.status_code == 200
```

## 🌱 `StageOutcome.skipped` not covered for the video batch runner
* **Priority**: 🟡 Medium
* **File**: `tests/test_video_runner.py`
* **Details**: `tests/test_runner_outcome.py` includes coverage for the image runner's handling of `StageOutcome.skipped` (stage should warn and continue to the next iteration, incrementing the skip count). The equivalent case for `run_video_batch()` is absent. Video runners should handle skip outcomes the same way as image runners. Without a test, a refactor that accidentally changes skip handling ships undetected.
* **Suggested Change**: Add to `tests/test_video_runner.py`:
```python
@patch('zvisiongenerator.video_runner.get_video_backend')
def test_run_video_batch_handles_skipped_stage(mock_backend_getter):
    # Set up a workflow with a stage that returns StageOutcome.skipped
    # Assert that the runner warns and continues to the next iteration
    ...
```

## ⛏️ `JobCard.test.ts` asserts formatted time/progress strings instead of behavioral values
* **Priority**: 🟢 Low
* **File**: `frontend/src/lib/components/molecules/JobCard.test.ts` (line ~60)
* **Details**: Assertions check the rendered text format rather than underlying state:
```typescript
expect(text).toContain('02:05')  // Asserts mm:ss time format
expect(text).toContain('2 / 3')  // Asserts "X / Y" count format
```
These test the rendered formatting (mm:ss, `X / Y` delimiters) rather than the semantic values. A change from `"2 / 3"` to `"2 of 3"` or localization to `"2 av 3"` (Norwegian) breaks the test without any behavioral regression. Per the project's testing guidelines: "Write tests against behavior and machine-readable contracts, not help text wording or source-code text." Legitimate string assertions cover routes, config keys, event names, enum values, and accessibility/control names when operability depends on them. Formatted time and counts do not meet this bar.
* **Suggested Change**: Expose semantic values via `data-*` attributes and assert those:
```typescript
// In JobCard.svelte:
<div data-job-id={job.id} data-elapsed-seconds={elapsedSeconds} data-progress-current={current} data-progress-total={total}>
  {formatTime(elapsedSeconds)} {current} / {total}
</div>

// In test:
const elapsedSeconds = element.getAttribute('data-elapsed-seconds');
const progressCurrent = element.getAttribute('data-progress-current');
const progressTotal = element.getAttribute('data-progress-total');
expect(elapsedSeconds).toBe('125');  // 2:05
expect(progressCurrent).toBe('2');
expect(progressTotal).toBe('3');
```

## ⛏️ `_make_mock_backend()` helper duplicated across test files
* **Priority**: 🟢 Low
* **File**: `tests/test_workflow.py` vs `tests/conftest.py`
* **Details**: The image mock backend factory `_make_mock_backend()` is defined inline in `tests/test_workflow.py`. The video equivalent `_make_mock_video_backend()` lives in `conftest.py`. Per the project's "reuse over duplication" guideline, the image helper should also be centralized in `conftest.py` so both can be imported uniformly. This reduces duplication and makes the helper available to all test modules.
* **Suggested Change**: Move `_make_mock_backend()` from `test_workflow.py` to `conftest.py` and import it in tests:
```python
# conftest.py
def _make_mock_backend(text_to_image_output=None, ...):
    # shared mock factory
    ...

# test_workflow.py
from conftest import _make_mock_backend
```

---

# Positive Highlights

## 👍 Correct and consistent path-traversal guard on `/media/` route
* **Priority**: 🟢 Low
* **File**: `zvisiongenerator/web/gallery.py`
* **Details**: The `resolve_output_asset_path()` function applies the two-step containment check correctly: `(root / normalized).resolve()` followed by `.is_relative_to(root)`. The `normalize_asset_id()` function provides a first defense layer that rejects `..` components, absolute paths, Windows drive letters, URL-encoded values (via `unquote()`), and staging directory names. Together these give defense-in-depth against traversal. SEC-01 exists only because the `/docs/assets/` route does not apply the same pattern — the contrast highlights that the `/media/` protection is the correct model.

## 👍 Platform dispatch is cleanly isolated in a single module
* **Priority**: 🟢 Low
* **File**: `zvisiongenerator/backends/__init__.py`
* **Details**: All platform-specific logic (`sys.platform == "darwin"` vs. Windows) is checked in one location only. All routes through the codebase that need a backend call `get_backend()` or `get_video_backend()`. There are no scattered `if sys.platform == "darwin"` guards in CLI code, runner code, or workflow stages. This makes the platform boundary auditable (one file to check), testable (mock `backends/__init__.py` to inject backends), and maintainable (single point to add support for a new platform).

## 👍 Atomic stage design with uniform signatures
* **Priority**: 🟢 Low
* **File**: `zvisiongenerator/workflows/image_stages.py`, `zvisiongenerator/workflows/video_stages.py`
* **Details**: Every image generation stage has the uniform signature `(ImageGenerationRequest, ImageWorkingArtifacts) -> StageOutcome`. Every video generation stage has `(VideoGenerationRequest, VideoWorkingArtifacts) -> StageOutcome`. Stages are pure functions — no global state, no I/O side effects except through the `backend` field in the request. `build_workflow()` and `build_video_workflow()` dynamically assemble ordered lists of stages from CLI flags. This architecture makes stages individually testable (mock the input, verify the output), composable (stages can be reordered or added/removed), and replaceable without touching the runner. This is exemplary of the project's stated Atomic Code Design principle.

## 👍 Svelte 5 runes used consistently; no legacy store patterns
* **Priority**: 🟢 Low
* **File**: `frontend/src/lib/state/`, `frontend/src/features/`
* **Details**: The frontend uses `$state`, `$derived`, and `$effect` runes throughout with singleton module-level state stores (`draft.svelte.ts`, `job.svelte.ts`, `history.svelte.ts`, `router.svelte.ts`). There are no legacy Svelte 4 `writable`/`readable` stores mixed in, no inconsistent reactivity patterns, and state management is centralized rather than scattered across components. The codebase demonstrates a clean migration to Svelte 5's declarative reactive model.

## 👍 Atomic prompt-file writes prevent partial corruption
* **Priority**: 🟢 Low
* **File**: `zvisiongenerator/web/prompt_files.py` (`write_prompt_file()`)
* **Details**: The function writes content to a UUID-named temporary file and then renames it atomically over the target using `Path.rename()`. On POSIX systems, `rename()` is atomic within the same filesystem. A crash mid-write leaves a temporary file behind, not a partially-written target. The original file is always either fully present (pre-crash state) or fully absent (allowing a clean retry). This is defensive programming that prevents data corruption under fault conditions.

---

# Summary

ZVisionGenerator is a well-architected, locally-scoped AI generation tool with clean separation between protocol-based backends, atomic stage composition, and modern frontend patterns. The codebase adheres to its stated design principles (Atomic Code Design, reuse over duplication) and demonstrates solid engineering practices.

However, several security findings require immediate remediation:

**Critical (resolve before network-adjacent deployment):**
- **SEC-02**: Arbitrary YAML file read/write via prompt-file API — requires directory confinement.
- **SEC-03**: Client-controlled output directory — generated files writable to arbitrary paths. Remove the form field; use only the config.
- **SEC-04**: DNS rebinding attack surface — add `TrustedHostMiddleware` to validate the `Host` header.

**High-priority defects:**
- **SEC-01**: `/docs/assets/` lacks path-traversal containment check — apply the same pattern as `/media/`.
- **SEC-05**: AppleScript injection via newline in `initial_path` — reject control characters.
- **SEC-06**: Unbounded reference image uploads — cap at 50 MB.
- **CQ-01**: PyTorch unconditionally installed on macOS (~2 GB waste) — add `sys_platform != 'darwin'` marker.
- **CQ-03**: `strip_audio_stage` swallows `FileNotFoundError` — catch and return `StageOutcome.failed` with warning.

**Testing gaps (highest priority):**
- **T-01**: No tests for `gallery.py` path-traversal guards.
- **T-03**: No integration tests for `/media/` traversal protection.
- **T-02**: `strip_audio_stage` not tested.

The remaining findings (CQ-02, CQ-04 through CQ-14, T-04 through T-07, SEC-07, SEC-08) are improvements to code quality, consistency, test coverage, and information disclosure that enhance maintainability and do not block a locally-run release.

**Recommendation: CHANGES REQUIRED**

Resolve SEC-01 through SEC-06, CQ-01, CQ-03, and the three testing gaps (T-01, T-02, T-03) before any network-adjacent deployment or production use. All security findings should be addressed in a dedicated security-hardening PR. The remaining code quality and testing improvements can be addressed in follow-up PRs or incorporated into the development roadmap.
