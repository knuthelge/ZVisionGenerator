"""Release asset manifest, verification, and safe fetch utilities.

The asset manager is intentionally non-generative. It never imports MLX, never
loads tensor payloads, and never writes outside the caller-selected cache root.
Fetches are license-gated and land through a resumable ``.part`` file that is
hash-checked before an atomic rename.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


OK = 0
ASSET_FAILURE = 2
USAGE_ERROR = 64
LICENSE_FAILURE = 65
FETCH_FAILURE = 66

LICENSE_ENV_VAR = "MINIMAX_H3_ACCEPT_LICENSE"
DEFAULT_CACHE_ENV_VAR = "MINIMAX_H3_ASSET_CACHE"
_TRUE_VALUES = {"1", "true", "yes", "y", "on", "accepted"}
_CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class FetchResult:
    """Summary for one fetched/reused asset."""

    ok: bool
    exit_code: int
    asset_id: str
    status: str
    files: list[dict[str, Any]]
    issues: list[dict[str, Any]]


def _package_manifest_resource():
    return resources.files("minimax_h3_mlx").joinpath("release_assets.json")


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Load a release asset manifest from a path or the package resource."""

    if path is None:
        resource = _package_manifest_resource()
        with resource.open() as handle:
            manifest = json.load(handle)
        manifest.setdefault("_manifest_path", "package:minimax_h3_mlx/release_assets.json")
        return manifest

    manifest_path = Path(path)
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    manifest.setdefault("_manifest_path", str(manifest_path))
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the manifest schema fields needed for safe verification/fetch."""

    issues: list[dict[str, Any]] = []
    assets = manifest.get("assets")
    if manifest.get("schema_version") != 1:
        issues.append({"code": "unsupported_manifest_schema", "message": "schema_version must be 1"})
    if not isinstance(manifest.get("manifest_version"), str) or not manifest.get("manifest_version"):
        issues.append({"code": "missing_manifest_version", "message": "manifest_version is required"})
    if not isinstance(assets, list) or not assets:
        issues.append({"code": "missing_assets", "message": "manifest assets must be a non-empty list"})
        assets = []

    seen: set[str] = set()
    for asset in assets:
        if not isinstance(asset, dict):
            issues.append({"code": "invalid_asset", "message": "asset entries must be objects"})
            continue
        asset_id = str(asset.get("id") or "")
        if not asset_id:
            issues.append({"code": "missing_asset_id", "message": "asset id is required"})
            continue
        if asset_id in seen:
            issues.append({"code": "duplicate_asset_id", "asset_id": asset_id, "message": "asset ids must be unique"})
        seen.add(asset_id)
        if not asset.get("repo_id"):
            issues.append({"code": "missing_repo_id", "asset_id": asset_id, "message": "repo_id is required"})
        if not isinstance(asset.get("license"), dict) or not asset["license"].get("name"):
            issues.append({"code": "missing_license", "asset_id": asset_id, "message": "license.name is required"})

        fetch_enabled = bool(asset.get("fetch_enabled"))
        files = asset.get("files")
        if fetch_enabled:
            if not asset.get("revision"):
                issues.append({"code": "missing_revision", "asset_id": asset_id, "message": "fetchable assets require a pinned revision"})
            if not isinstance(files, list) or not files:
                issues.append({"code": "missing_files", "asset_id": asset_id, "message": "fetchable assets require files"})
                files = []
            for item in files:
                if not isinstance(item, dict):
                    issues.append({"code": "invalid_file", "asset_id": asset_id, "message": "file entries must be objects"})
                    continue
                missing = [key for key in ("path", "size_bytes", "sha256") if item.get(key) in (None, "")]
                if missing:
                    issues.append(
                        {
                            "code": "incomplete_file_pin",
                            "asset_id": asset_id,
                            "file": item.get("path"),
                            "missing": missing,
                            "message": "fetchable files require path, size_bytes, and sha256",
                        }
                    )
        elif files not in ([], None):
            issues.append({"code": "disabled_asset_has_files", "asset_id": asset_id, "message": "non-fetchable records must not carry partial file pins"})

    return {"ok": not issues, "issues": issues, "asset_count": len(assets)}


def default_cache_dir(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    if env.get(DEFAULT_CACHE_ENV_VAR):
        return Path(env[DEFAULT_CACHE_ENV_VAR]).expanduser()
    base = env.get("XDG_CACHE_HOME") or str(Path(env.get("HOME", str(Path.home()))) / ".cache")
    return Path(base).expanduser() / "minimax-h3" / "assets"


def license_accepted(env: Mapping[str, str] | None = None, *, explicit: bool = False) -> bool:
    env = os.environ if env is None else env
    return explicit or env.get(LICENSE_ENV_VAR, "").strip().lower() in _TRUE_VALUES


def _asset_names(asset: Mapping[str, Any]) -> set[str]:
    names = {str(asset.get("id"))}
    names.update(str(alias) for alias in asset.get("aliases", []) if str(alias))
    return names


def select_assets(manifest: Mapping[str, Any], requested: Iterable[str] | None = None) -> list[dict[str, Any]]:
    assets = [dict(asset) for asset in manifest.get("assets", []) if isinstance(asset, dict)]
    requested_ids = list(requested or [])
    if not requested_ids:
        return assets
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for item in requested_ids:
        match = next((asset for asset in assets if item in _asset_names(asset)), None)
        if match is None:
            missing.append(item)
        else:
            selected.append(match)
    if missing:
        raise KeyError(f"unknown asset id/alias: {', '.join(missing)}")
    return selected


def asset_root(cache_dir: str | Path, asset: Mapping[str, Any]) -> Path:
    subdir = str(asset.get("local_subdir") or asset.get("id"))
    return Path(cache_dir).expanduser() / subdir


def _safe_join(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe manifest file path: {relative!r}")
    return root / rel


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_status(path: Path, spec: Mapping[str, Any], *, hash_mode: bool) -> dict[str, Any]:
    expected_size = int(spec.get("size_bytes") or 0)
    expected_sha = str(spec.get("sha256") or "")
    record: dict[str, Any] = {
        "path": str(path),
        "manifest_path": spec.get("path"),
        "exists": path.exists(),
        "size_bytes": None,
        "expected_size_bytes": expected_size,
        "size_ok": False,
        "sha256_checked": False,
        "sha256": None,
        "expected_sha256": expected_sha,
        "sha256_ok": None,
        "ok": False,
    }
    if not path.exists():
        record["status"] = "missing"
        return record
    if not path.is_file():
        record["status"] = "not_file"
        return record
    size = path.stat().st_size
    record["size_bytes"] = size
    record["size_ok"] = size == expected_size
    if not record["size_ok"]:
        record["status"] = "size_mismatch"
        return record
    if hash_mode:
        record["sha256_checked"] = True
        observed = _sha256_file(path)
        record["sha256"] = observed
        record["sha256_ok"] = observed == expected_sha
        if not record["sha256_ok"]:
            record["status"] = "hash_mismatch"
            return record
    record["ok"] = True
    record["status"] = "ok"
    return record


def verify_asset(asset: Mapping[str, Any], cache_dir: str | Path, *, hash_mode: bool = True) -> dict[str, Any]:
    """Verify one asset in a cache root using size and, by default, sha256."""

    root = asset_root(cache_dir, asset)
    issues: list[dict[str, Any]] = []
    files = asset.get("files") or []
    if not asset.get("fetch_enabled"):
        issues.append(
            {
                "severity": "warning",
                "code": "asset_not_fetchable",
                "asset_id": asset.get("id"),
                "message": "asset record is present for provenance only and lacks pinned revision/file hashes",
            }
        )
    if not files:
        return {
            "asset_id": asset.get("id"),
            "ok": False,
            "status": "no_pinned_files",
            "root": str(root),
            "hash_mode": hash_mode,
            "files": [],
            "issues": issues,
            "summary": {"files": 0, "ok_files": 0, "missing_files": 0, "hash_mismatches": 0},
        }

    records: list[dict[str, Any]] = []
    for spec in files:
        try:
            target = _safe_join(root, str(spec["path"]))
        except ValueError as exc:
            issues.append({"severity": "error", "code": "unsafe_manifest_path", "asset_id": asset.get("id"), "message": str(exc)})
            continue
        record = _file_status(target, spec, hash_mode=hash_mode)
        records.append(record)
        if record["status"] == "missing":
            issues.append({"severity": "error", "code": "missing_asset_file", "asset_id": asset.get("id"), "file": spec.get("path")})
        elif record["status"] == "size_mismatch":
            issues.append(
                {
                    "severity": "error",
                    "code": "asset_size_mismatch",
                    "asset_id": asset.get("id"),
                    "file": spec.get("path"),
                    "observed_size_bytes": record["size_bytes"],
                    "expected_size_bytes": record["expected_size_bytes"],
                }
            )
        elif record["status"] == "hash_mismatch":
            issues.append(
                {
                    "severity": "error",
                    "code": "asset_hash_mismatch",
                    "asset_id": asset.get("id"),
                    "file": spec.get("path"),
                    "observed_sha256": record["sha256"],
                    "expected_sha256": record["expected_sha256"],
                }
            )
        elif record["status"] == "not_file":
            issues.append({"severity": "error", "code": "asset_path_not_file", "asset_id": asset.get("id"), "file": spec.get("path")})

    ok = bool(records) and all(record["ok"] for record in records)
    summary = {
        "files": len(records),
        "ok_files": sum(1 for record in records if record["ok"]),
        "missing_files": sum(1 for record in records if record["status"] == "missing"),
        "hash_mismatches": sum(1 for record in records if record["status"] == "hash_mismatch"),
        "size_mismatches": sum(1 for record in records if record["status"] == "size_mismatch"),
    }
    return {
        "asset_id": asset.get("id"),
        "ok": ok,
        "status": "ok" if ok else "asset_error",
        "root": str(root),
        "hash_mode": hash_mode,
        "files": records,
        "issues": issues,
        "summary": summary,
    }


def verify_assets(
    manifest: Mapping[str, Any],
    cache_dir: str | Path,
    *,
    asset_ids: Iterable[str] | None = None,
    hash_mode: bool = True,
) -> dict[str, Any]:
    validation = validate_manifest(manifest)
    issues: list[dict[str, Any]] = [{"severity": "error", **issue} for issue in validation["issues"]]
    assets: list[dict[str, Any]] = []
    requested_ids = list(asset_ids or [])
    try:
        assets = select_assets(manifest, requested_ids)
        if not requested_ids:
            assets = [asset for asset in assets if asset.get("fetch_enabled")]
    except KeyError as exc:
        issues.append({"severity": "error", "code": "unknown_asset", "message": str(exc)})
    results = [verify_asset(asset, cache_dir, hash_mode=hash_mode) for asset in assets]
    for result in results:
        issues.extend(result.get("issues", []))
    errors = [issue for issue in issues if issue.get("severity") == "error"]
    ok = not errors and all(result.get("ok") for result in results)
    return {
        "schema_version": 1,
        "tool": "minimax-h3-assets",
        "action": "verify",
        "ok": ok,
        "exit_code": OK if ok else ASSET_FAILURE,
        "manifest_version": manifest.get("manifest_version"),
        "cache_dir": str(Path(cache_dir).expanduser()),
        "hash_mode": hash_mode,
        "assets": results,
        "issues": issues,
        "summary": {
            "assets": len(results),
            "ok_assets": sum(1 for result in results if result.get("ok")),
            "errors": len(errors),
            "warnings": sum(1 for issue in issues if issue.get("severity") == "warning"),
        },
    }


def _source_url(asset: Mapping[str, Any], spec: Mapping[str, Any]) -> str:
    if spec.get("url"):
        return str(spec["url"])
    if spec.get("source_path"):
        return Path(str(spec["source_path"])).expanduser().resolve().as_uri()
    repo_id = asset.get("repo_id")
    revision = asset.get("revision")
    if not repo_id or not revision:
        raise ValueError(f"asset {asset.get('id')} is not fetchable without repo_id and revision")
    quoted_path = urllib.parse.quote(str(spec["path"]))
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{quoted_path}"


def _copy_file_url(source: Path, part: Path, *, resume_from: int) -> int:
    total = resume_from
    with source.open("rb") as src, part.open("ab") as dst:
        if resume_from:
            src.seek(resume_from)
        for chunk in iter(lambda: src.read(_CHUNK_SIZE), b""):
            dst.write(chunk)
            total += len(chunk)
    return total


def _download_url(url: str, part: Path, *, resume_from: int, timeout: float = 60.0) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme in {"", "file"}:
        source = Path(urllib.request.url2pathname(parsed.path if parsed.scheme == "file" else url))
        total = _copy_file_url(source, part, resume_from=resume_from)
        return {"url": url, "resumed_from_bytes": resume_from, "downloaded_bytes": total - resume_from, "status": "copied_local"}

    headers = {}
    mode = "ab"
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response, part.open(mode) as dst:  # nosec: URL comes from manifest/operator.
        status = getattr(response, "status", None)
        if resume_from and status not in (206, None):
            dst.close()
            part.write_bytes(b"")
            resume_from = 0
            return _download_url(url, part, resume_from=0, timeout=timeout)
        copied = 0
        while True:
            chunk = response.read(_CHUNK_SIZE)
            if not chunk:
                break
            dst.write(chunk)
            copied += len(chunk)
    return {"url": url, "resumed_from_bytes": resume_from, "downloaded_bytes": copied, "status": "downloaded"}


def _fetch_file(asset: Mapping[str, Any], spec: Mapping[str, Any], root: Path, *, retries: int) -> dict[str, Any]:
    target = _safe_join(root, str(spec["path"]))
    part = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = []
    last_error: str | None = None
    url = _source_url(asset, spec)
    for attempt in range(max(1, retries + 1)):
        try:
            resume_from = part.stat().st_size if part.exists() else 0
            event = _download_url(url, part, resume_from=resume_from)
            event["attempt"] = attempt + 1
            events.append(event)
            status = _file_status(part, spec, hash_mode=True)
            if status["ok"]:
                os.replace(part, target)
                final_status = _file_status(target, spec, hash_mode=True)
                final_status["fetch_events"] = events
                final_status["status"] = "fetched"
                return final_status
            last_error = status["status"]
            if status["status"] in {"hash_mismatch", "size_mismatch"}:
                try:
                    part.unlink()
                except FileNotFoundError:
                    pass
                break
        except (OSError, urllib.error.URLError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            events.append({"attempt": attempt + 1, "status": "error", "error": last_error})
            if attempt < retries:
                time.sleep(min(1.0, 0.1 * (attempt + 1)))
    try:
        if part.exists() and last_error in {"hash_mismatch", "size_mismatch"}:
            part.unlink()
    except FileNotFoundError:
        pass
    return {
        "path": str(target),
        "manifest_path": spec.get("path"),
        "ok": False,
        "status": "fetch_failed",
        "error": last_error,
        "part_exists_after_failure": part.exists(),
        "fetch_events": events,
    }


def fetch_asset(
    asset: Mapping[str, Any],
    cache_dir: str | Path,
    *,
    accept_license: bool = False,
    env: Mapping[str, str] | None = None,
    offline: bool = False,
    retries: int = 2,
) -> FetchResult:
    """Fetch or reuse one asset under ``cache_dir`` without touching other paths."""

    root = asset_root(cache_dir, asset)
    reused = verify_asset(asset, cache_dir, hash_mode=True)
    if reused["ok"]:
        return FetchResult(True, OK, str(asset.get("id")), "reused", reused["files"], [])

    if offline:
        return FetchResult(False, ASSET_FAILURE, str(asset.get("id")), "offline_missing_or_invalid", reused["files"], reused["issues"])
    if not asset.get("fetch_enabled"):
        return FetchResult(
            False,
            ASSET_FAILURE,
            str(asset.get("id")),
            "asset_not_fetchable",
            reused["files"],
            [{"severity": "error", "code": "asset_not_fetchable", "asset_id": asset.get("id")}],
        )
    if not license_accepted(env, explicit=accept_license):
        return FetchResult(
            False,
            LICENSE_FAILURE,
            str(asset.get("id")),
            "license_not_accepted",
            reused["files"],
            [
                {
                    "severity": "error",
                    "code": "model_license_not_accepted",
                    "asset_id": asset.get("id"),
                    "env_var": asset.get("license", {}).get("acceptance_env_var", LICENSE_ENV_VAR),
                    "message": "Set the license acceptance environment variable or pass --accept-license before downloading weights.",
                }
            ],
        )

    files: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for spec in asset.get("files", []):
        target = _safe_join(root, str(spec["path"]))
        existing = _file_status(target, spec, hash_mode=True)
        if existing["ok"]:
            existing["status"] = "reused"
            files.append(existing)
            continue
        if target.exists() and not existing["ok"]:
            files.append(existing)
            issues.append(
                {
                    "severity": "error",
                    "code": "existing_asset_file_invalid",
                    "asset_id": asset.get("id"),
                    "file": spec.get("path"),
                    "status": existing["status"],
                    "message": "Existing asset file is invalid; refusing to overwrite without operator intervention.",
                }
            )
            continue
        fetched = _fetch_file(asset, spec, root, retries=retries)
        files.append(fetched)
        if not fetched.get("ok"):
            issues.append({"severity": "error", "code": "asset_fetch_failed", "asset_id": asset.get("id"), "file": spec.get("path"), "error": fetched.get("error")})
            break
    ok = all(item.get("ok") for item in files) and not issues
    return FetchResult(ok, OK if ok else FETCH_FAILURE, str(asset.get("id")), "fetched" if ok else "fetch_failed", files, issues)


def fetch_assets(
    manifest: Mapping[str, Any],
    cache_dir: str | Path,
    *,
    asset_ids: Iterable[str],
    accept_license: bool = False,
    env: Mapping[str, str] | None = None,
    offline: bool = False,
    retries: int = 2,
) -> dict[str, Any]:
    validation = validate_manifest(manifest)
    issues: list[dict[str, Any]] = [{"severity": "error", **issue} for issue in validation["issues"]]
    try:
        assets = select_assets(manifest, asset_ids)
    except KeyError as exc:
        assets = []
        issues.append({"severity": "error", "code": "unknown_asset", "message": str(exc)})
    results = [fetch_asset(asset, cache_dir, accept_license=accept_license, env=env, offline=offline, retries=retries) for asset in assets]
    for result in results:
        issues.extend(result.issues)
    exit_code = OK
    if any(result.exit_code == LICENSE_FAILURE for result in results):
        exit_code = LICENSE_FAILURE
    elif issues or any(not result.ok for result in results):
        exit_code = FETCH_FAILURE
    return {
        "schema_version": 1,
        "tool": "minimax-h3-assets",
        "action": "fetch",
        "ok": exit_code == OK,
        "exit_code": exit_code,
        "manifest_version": manifest.get("manifest_version"),
        "cache_dir": str(Path(cache_dir).expanduser()),
        "assets": [result.__dict__ for result in results],
        "issues": issues,
    }


def asset_status_report(
    manifest: Mapping[str, Any],
    cache_dir: str | Path,
    *,
    asset_ids: Iterable[str] | None = None,
    hash_mode: bool = False,
    include_preflight: bool = True,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    """Build a non-generative manifest/doctor status report."""

    validation = validate_manifest(manifest)
    issues: list[dict[str, Any]] = [{"severity": "error", **issue} for issue in validation["issues"]]
    try:
        assets = select_assets(manifest, asset_ids)
    except KeyError as exc:
        assets = []
        issues.append({"severity": "error", "code": "unknown_asset", "message": str(exc)})

    records: list[dict[str, Any]] = []
    for asset in assets:
        verify = verify_asset(asset, cache_dir, hash_mode=hash_mode)
        root = Path(verify["root"])
        preflight: dict[str, Any] | None = None
        if include_preflight and root.exists():
            from minimax_h3_mlx.asset_preflight import run_preflight

            preflight = run_preflight([root], cwd=cwd or Path.cwd())
        record = {
            "asset_id": asset.get("id"),
            "kind": asset.get("kind"),
            "precision": asset.get("precision"),
            "repo_id": asset.get("repo_id"),
            "revision": asset.get("revision"),
            "fetch_enabled": bool(asset.get("fetch_enabled")),
            "pinning_status": asset.get("pinning_status"),
            "license": asset.get("license", {}),
            "root": str(root),
            "verification": verify,
            "header_preflight": preflight,
            "streaming_24gb_notes": asset.get("streaming_24gb_notes", {}),
            "quality_claim_status": asset.get("quality_claim_status"),
        }
        if preflight is not None:
            record["shard_header_coverage"] = {
                "preflight_ok": preflight.get("ok"),
                "indexes": preflight.get("summary", {}).get("indexes"),
                "declared_shards": preflight.get("summary", {}).get("declared_shards"),
                "standalone_safetensors": preflight.get("summary", {}).get("standalone_safetensors"),
                "issues": preflight.get("issues", []),
            }
        else:
            record["shard_header_coverage"] = {"preflight_ok": None, "reason": "asset_root_missing_or_preflight_disabled"}
        records.append(record)
        issues.extend(verify.get("issues", []))

    errors = [issue for issue in issues if issue.get("severity") == "error"]
    return {
        "schema_version": 1,
        "tool": "minimax-h3-assets",
        "action": "doctor-report",
        "ok": not errors,
        "exit_code": OK if not errors else ASSET_FAILURE,
        "manifest_version": manifest.get("manifest_version"),
        "manifest_id": manifest.get("manifest_id"),
        "cache_dir": str(Path(cache_dir).expanduser()),
        "hash_mode": hash_mode,
        "runs_generation": False,
        "downloads_models": False,
        "assets": records,
        "issues": issues,
        "summary": {
            "assets": len(records),
            "fetch_enabled_assets": sum(1 for asset in records if asset["fetch_enabled"]),
            "ok_assets": sum(1 for asset in records if asset["verification"].get("ok")),
            "header_preflight_roots": sum(1 for asset in records if asset.get("header_preflight") is not None),
            "errors": len(errors),
            "warnings": sum(1 for issue in issues if issue.get("severity") == "warning"),
        },
    }


def quant_provenance_report(
    manifest: Mapping[str, Any],
    cache_dir: str | Path,
    *,
    hash_mode: bool = False,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    report = asset_status_report(
        manifest,
        cache_dir,
        asset_ids=["dit-mlx-4bit", "dit-mlx-6bit", "dit-mlx-8bit"],
        hash_mode=hash_mode,
        include_preflight=True,
        cwd=cwd,
    )
    report["action"] = "quant-provenance-report"
    report["evidence_boundary"] = {
        "non_generative": True,
        "no_quality_or_pareto_claim": True,
        "four_bit_status": "run baseline only; not a high-quality baseline",
        "six_eight_bit_status": "repo names and README size estimates are recorded, but revision/hash-pinned staging is still required before generation or profile claims",
    }
    return report


def _summarize_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_manifest(manifest)
    return {
        "schema_version": 1,
        "tool": "minimax-h3-assets",
        "action": "list",
        "ok": validation["ok"],
        "exit_code": OK if validation["ok"] else USAGE_ERROR,
        "manifest_version": manifest.get("manifest_version"),
        "manifest_id": manifest.get("manifest_id"),
        "contract": manifest.get("contract", {}),
        "assets": [
            {
                "id": asset.get("id"),
                "aliases": asset.get("aliases", []),
                "kind": asset.get("kind"),
                "precision": asset.get("precision"),
                "repo_id": asset.get("repo_id"),
                "revision": asset.get("revision"),
                "fetch_enabled": bool(asset.get("fetch_enabled")),
                "pinning_status": asset.get("pinning_status"),
                "file_count": len(asset.get("files") or []),
                "license": asset.get("license", {}),
                "streaming_24gb_notes": asset.get("streaming_24gb_notes", {}),
                "quality_claim_status": asset.get("quality_claim_status"),
            }
            for asset in manifest.get("assets", [])
            if isinstance(asset, dict)
        ],
        "issues": [{"severity": "error", **issue} for issue in validation["issues"]],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List, verify, fetch, and report MiniMax-H3 release assets.")
    parser.add_argument("--manifest", default=None, help="manifest JSON path; defaults to package release_assets.json")
    parser.add_argument("--cache-dir", default=None, help=f"asset cache root; defaults to ${DEFAULT_CACHE_ENV_VAR} or ~/.cache/minimax-h3/assets")
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON output")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list manifest assets without touching the cache")

    verify = sub.add_parser("verify", help="verify local asset files against manifest size/hash pins")
    verify.add_argument("--asset", action="append", default=[], help="asset id or alias; may be repeated; defaults to fetch-enabled assets")
    verify.add_argument("--no-hash", action="store_true", help="check only presence and byte sizes")

    fetch = sub.add_parser("fetch", help="fetch missing asset files with license gate, resume, retry, and atomic landing")
    fetch.add_argument("--asset", action="append", required=True, help="asset id or alias; may be repeated")
    fetch.add_argument("--accept-license", action="store_true", help=f"confirm upstream license acceptance for this invocation (or set {LICENSE_ENV_VAR}=1)")
    fetch.add_argument("--offline", action="store_true", help="do not download; succeed only if all selected assets already verify")
    fetch.add_argument("--retries", type=int, default=2, help="retry count after the initial attempt")

    doctor = sub.add_parser("doctor-report", help="emit manifest asset status for release doctor/preflight")
    doctor.add_argument("--asset", action="append", default=[], help="asset id or alias; defaults to all")
    doctor.add_argument("--hash", action="store_true", help="include full sha256 checks; default is header/size-only")
    doctor.add_argument("--no-preflight", action="store_true", help="skip header-only safetensors preflight")

    quant = sub.add_parser("quant-report", help="emit non-generative 4/6/8-bit DiT provenance and streaming-risk report")
    quant.add_argument("--hash", action="store_true", help="include full sha256 checks; default is header/size-only")
    quant.add_argument("--out", default=None, help="optional JSON output path in addition to stdout")
    return parser


def _cache_from_args(args: argparse.Namespace) -> Path:
    return Path(args.cache_dir).expanduser() if args.cache_dir else default_cache_dir()


def _dump(payload: Mapping[str, Any], *, pretty: bool, out: str | None = None) -> None:
    text = json.dumps(payload, indent=2 if pretty else None, sort_keys=True) + "\n"
    if out:
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    sys.stdout.write(text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "tool": "minimax-h3-assets",
            "ok": False,
            "exit_code": USAGE_ERROR,
            "issues": [{"severity": "error", "code": "manifest_load_failed", "message": f"{type(exc).__name__}: {exc}"}],
        }
        _dump(payload, pretty=args.pretty)
        return USAGE_ERROR

    cache_dir = _cache_from_args(args)
    try:
        if args.command == "list":
            payload = _summarize_manifest(manifest)
        elif args.command == "verify":
            payload = verify_assets(manifest, cache_dir, asset_ids=args.asset, hash_mode=not args.no_hash)
        elif args.command == "fetch":
            payload = fetch_assets(
                manifest,
                cache_dir,
                asset_ids=args.asset,
                accept_license=args.accept_license,
                env=os.environ,
                offline=args.offline,
                retries=max(0, args.retries),
            )
        elif args.command == "doctor-report":
            payload = asset_status_report(
                manifest,
                cache_dir,
                asset_ids=args.asset,
                hash_mode=args.hash,
                include_preflight=not args.no_preflight,
            )
        elif args.command == "quant-report":
            payload = quant_provenance_report(manifest, cache_dir, hash_mode=args.hash)
        else:  # pragma: no cover - argparse prevents this.
            raise AssertionError(args.command)
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "tool": "minimax-h3-assets",
            "ok": False,
            "exit_code": FETCH_FAILURE,
            "issues": [{"severity": "error", "code": "asset_manager_failed", "message": f"{type(exc).__name__}: {exc}"}],
        }
    _dump(payload, pretty=args.pretty, out=getattr(args, "out", None))
    return int(payload.get("exit_code", FETCH_FAILURE))


if __name__ == "__main__":  # pragma: no cover - exercised by tests/console entrypoint.
    raise SystemExit(main())
