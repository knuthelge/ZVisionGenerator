"""Header-only model asset preflight for local MiniMax-H3 deployments.

The checks in this module are intentionally read-only and header-only: safetensors
files are opened with ``safetensors.safe_open(..., framework="np")`` so the tensor
metadata is parsed without materializing large model weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from safetensors import safe_open

OK = 0
ASSET_FAILURE = 2
USAGE_ERROR = 64

_DIT_EXPECTED = {
    "_class_name": "MiniMaxH3DiTModel",
    "hidden_size": 5376,
    "num_layers": 50,
    "num_attention_heads": 56,
    "attention_head_dim": 128,
    "latents_dim": 24,
    "audio_latents_dim": 32,
    "text_dim": 5120,
    "patch_size": [1, 2, 2],
}


class _IssueSink:
    def __init__(self, cwd: Path):
        self.cwd = cwd
        self.issues: list[dict[str, Any]] = []
        self._seen: set[tuple[str, str, str]] = set()

    def path(self, path: str | Path) -> str:
        path = Path(path)
        try:
            return str(path.resolve().relative_to(self.cwd))
        except ValueError:
            return str(path)

    def add(self, severity: str, code: str, path: str | Path, message: str, **extra: Any) -> None:
        rendered = self.path(path)
        key = (severity, code, rendered)
        if key in self._seen:
            return
        self._seen.add(key)
        issue = {
            "severity": severity,
            "code": code,
            "path": rendered,
            "message": message,
        }
        issue.update(extra)
        self.issues.append(issue)


def _read_json(path: Path, sink: _IssueSink) -> dict[str, Any] | None:
    try:
        with path.open() as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        sink.add("error", "missing_json", path, "required JSON file is missing")
        return None
    except json.JSONDecodeError as exc:
        sink.add("error", "invalid_json", path, f"JSON is not parseable: {exc}")
        return None
    if not isinstance(raw, dict):
        sink.add("error", "invalid_json", path, "top-level JSON value must be an object")
        return None
    return raw


def _check_safetensors_header(path: Path, sink: _IssueSink, *, code_prefix: str = "") -> dict[str, Any]:
    rendered = sink.path(path)
    entry: dict[str, Any] = {
        "path": rendered,
        "ok": False,
        "status": "missing",
        "size_bytes": None,
        "tensor_count": None,
    }
    if not path.exists():
        sink.add("error", f"missing_{code_prefix}safetensors", path, "safetensors file is missing")
        return entry
    if not path.is_file():
        entry["status"] = "not_file"
        sink.add("error", f"invalid_{code_prefix}safetensors", path, "safetensors path is not a file")
        return entry

    size = path.stat().st_size
    entry["size_bytes"] = size
    if size <= 0:
        entry["status"] = "empty"
        sink.add("error", f"empty_{code_prefix}safetensors", path, "safetensors file is empty")
        return entry

    try:
        with safe_open(str(path), framework="np") as handle:
            keys = list(handle.keys())
            metadata = handle.metadata()
    except Exception as exc:  # safetensors raises backend-specific parse exceptions.
        entry["status"] = "bad_header"
        entry["error"] = f"{type(exc).__name__}: {exc}"
        sink.add(
            "error",
            f"bad_{code_prefix}safetensors_header",
            path,
            "safetensors header is not readable",
            error=entry["error"],
        )
        return entry

    entry.update(
        {
            "ok": True,
            "status": "ok",
            "tensor_count": len(keys),
            "metadata_keys": sorted(metadata.keys()) if metadata else [],
        }
    )
    if not keys:
        entry["ok"] = False
        entry["status"] = "no_tensors"
        sink.add("error", f"empty_{code_prefix}safetensors_header", path, "safetensors header declares no tensors")
    return entry


def _index_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.name.endswith(".safetensors.index.json") else []
    return sorted(root.rglob("*.safetensors.index.json"))


def _config_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.name in {"config.json", "model_index.json", "metadata.json"} else []
    paths: list[Path] = []
    for name in ("model_index.json", "config.json", "metadata.json"):
        paths.extend(root.rglob(name))
    return sorted(set(paths))


def _safetensors_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix == ".safetensors" else []
    return sorted(root.rglob("*.safetensors"))


def _check_index(index_path: Path, sink: _IssueSink) -> tuple[dict[str, Any], set[Path]]:
    record: dict[str, Any] = {
        "path": sink.path(index_path),
        "ok": False,
        "declared_shard_count": 0,
        "weight_count": 0,
        "shards": [],
    }
    declared: set[Path] = set()
    raw = _read_json(index_path, sink)
    if raw is None:
        record["status"] = "invalid_json"
        return record, declared

    weight_map = raw.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        sink.add("error", "invalid_safetensors_index", index_path, "index must contain a non-empty weight_map")
        record["status"] = "invalid_weight_map"
        return record, declared

    shard_names = sorted({str(value) for value in weight_map.values()})
    record["declared_shard_count"] = len(shard_names)
    record["weight_count"] = len(weight_map)
    for shard_name in shard_names:
        shard_path = (index_path.parent / shard_name).resolve()
        declared.add(shard_path)
        entry = _check_safetensors_header(shard_path, sink, code_prefix="shard_")
        entry["declared_by"] = sink.path(index_path)
        entry["name_in_index"] = shard_name
        record["shards"].append(entry)

    record["ok"] = all(entry["ok"] for entry in record["shards"])
    record["status"] = "ok" if record["ok"] else "asset_error"
    return record, declared


def _nested(raw: dict[str, Any], *keys: str) -> Any:
    current: Any = raw
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _classify_config(path: Path, raw: dict[str, Any]) -> str | None:
    class_name = raw.get("_class_name")
    if path.name == "model_index.json" or class_name == "MiniMaxH3Pipeline":
        return "minimax_h3_pipeline"
    if class_name == "MiniMaxH3DiTModel" or {"adaln_out_features", "audio_latents_dim", "text_dim"} <= set(raw):
        return "minimax_h3_dit"
    if raw.get("model_type") == "qwen3_vl" or "Qwen3VLForConditionalGeneration" in raw.get("architectures", []):
        return "qwen3_vl_text_encoder"
    if class_name == "MiniMaxH3VideoVAE":
        return "minimax_h3_video_vae"
    if class_name == "AutoencoderKLLegacy":
        return "minimax_h3_video_vae_source"
    if class_name == "MiniMaxH3AudioVAE":
        return "minimax_h3_audio_vae"
    if isinstance(_nested(raw, "metadata", "kwargs"), dict) and "vae_latent_channels" in raw["metadata"]["kwargs"]:
        return "minimax_h3_audio_vae_metadata"
    return None


def _expect(record: dict[str, Any], sink: _IssueSink, path: Path, raw: dict[str, Any], key: str, expected: Any) -> None:
    observed = raw.get(key)
    if observed != expected:
        record["ok"] = False
        sink.add(
            "error",
            "config_family_mismatch",
            path,
            f"expected {key}={expected!r} for {record['family']}, found {observed!r}",
            key=key,
            expected=expected,
            observed=observed,
            family=record["family"],
        )


def _expect_nested(
    record: dict[str, Any],
    sink: _IssueSink,
    path: Path,
    raw: dict[str, Any],
    keys: tuple[str, ...],
    expected: Any,
) -> None:
    observed = _nested(raw, *keys)
    if observed != expected:
        record["ok"] = False
        dotted = ".".join(keys)
        sink.add(
            "error",
            "config_family_mismatch",
            path,
            f"expected {dotted}={expected!r} for {record['family']}, found {observed!r}",
            key=dotted,
            expected=expected,
            observed=observed,
            family=record["family"],
        )


def _expect_len(
    record: dict[str, Any],
    sink: _IssueSink,
    path: Path,
    raw: dict[str, Any],
    key: str,
    expected: int,
) -> None:
    observed = raw.get(key)
    if not isinstance(observed, list) or len(observed) != expected:
        record["ok"] = False
        sink.add(
            "error",
            "config_family_mismatch",
            path,
            f"expected {key} to be a list of length {expected} for {record['family']}",
            key=key,
            expected_length=expected,
            observed_length=len(observed) if isinstance(observed, list) else None,
            family=record["family"],
        )


def _validate_config(path: Path, raw: dict[str, Any], family: str, sink: _IssueSink) -> dict[str, Any]:
    record: dict[str, Any] = {"path": sink.path(path), "family": family, "ok": True, "referenced_safetensors": []}

    if family == "minimax_h3_pipeline":
        _expect(record, sink, path, raw, "_class_name", "MiniMaxH3Pipeline")
        partition = _nested(raw, "_minimax_h3", "partition")
        if partition not in {"fl2va", "ref2va"}:
            record["ok"] = False
            sink.add(
                "error",
                "config_family_mismatch",
                path,
                "MiniMax-H3 pipeline partition must be fl2va or ref2va",
                key="_minimax_h3.partition",
                expected=["fl2va", "ref2va"],
                observed=partition,
                family=family,
            )
        for component in ("text_encoder", "tokenizer", "video_vae", "audio_vae", "transformer", "processor"):
            if component not in raw:
                record["ok"] = False
                sink.add(
                    "error",
                    "config_family_mismatch",
                    path,
                    f"MiniMax-H3 pipeline model_index.json is missing component {component}",
                    key=component,
                    family=family,
                )

    elif family == "minimax_h3_dit":
        for key, expected in _DIT_EXPECTED.items():
            _expect(record, sink, path, raw, key, expected)

    elif family == "qwen3_vl_text_encoder":
        _expect(record, sink, path, raw, "model_type", "qwen3_vl")
        _expect_nested(record, sink, path, raw, ("text_config", "hidden_size"), 5120)
        _expect_nested(record, sink, path, raw, ("vision_config", "out_hidden_size"), 5120)
        layers = _nested(raw, "text_config", "num_hidden_layers")
        if not isinstance(layers, int) or layers <= 50:
            record["ok"] = False
            sink.add(
                "error",
                "config_family_mismatch",
                path,
                "MiniMax-H3 reads Qwen3-VL hidden_states[50], so the text encoder must have more than 50 layers",
                key="text_config.num_hidden_layers",
                expected="> 50",
                observed=layers,
                family=family,
            )

    elif family == "minimax_h3_video_vae":
        _expect(record, sink, path, raw, "_class_name", "MiniMaxH3VideoVAE")
        _expect(record, sink, path, raw, "latent_channels", 24)
        _expect_len(record, sink, path, raw, "latents_mean", 24)
        _expect_len(record, sink, path, raw, "latents_std", 24)
        source_name = raw.get("source_safetensors_path")
        if isinstance(source_name, str):
            source_dir = path.parent / str(raw.get("source_path", "source"))
            target = source_dir / source_name
            record["referenced_safetensors"].append(_check_safetensors_header(target, sink))
        else:
            record["ok"] = False
            sink.add(
                "error",
                "config_family_mismatch",
                path,
                "video VAE config must declare source_safetensors_path",
                key="source_safetensors_path",
                family=family,
            )

    elif family == "minimax_h3_video_vae_source":
        _expect(record, sink, path, raw, "_class_name", "AutoencoderKLLegacy")
        _expect(record, sink, path, raw, "z_channels", 24)
        _expect(record, sink, path, raw, "in_channels", 3)
        _expect(record, sink, path, raw, "out_ch", 3)
        _expect(record, sink, path, raw, "vae_ratio", 16)
        _expect(record, sink, path, raw, "vae_ratio_t", 4)

    elif family == "minimax_h3_audio_vae":
        _expect(record, sink, path, raw, "_class_name", "MiniMaxH3AudioVAE")
        _expect(record, sink, path, raw, "latent_channels", 32)
        _expect(record, sink, path, raw, "sample_rate", 32000)
        _expect(record, sink, path, raw, "output_channel", 2)
        _expect_len(record, sink, path, raw, "latents_mean", 32)
        _expect_len(record, sink, path, raw, "latents_std", 32)
        source_name = raw.get("source_safetensors_path")
        if isinstance(source_name, str):
            record["referenced_safetensors"].append(_check_safetensors_header(path.parent / source_name, sink))
        else:
            record["ok"] = False
            sink.add(
                "error",
                "config_family_mismatch",
                path,
                "audio VAE config must declare source_safetensors_path",
                key="source_safetensors_path",
                family=family,
            )

    elif family == "minimax_h3_audio_vae_metadata":
        _expect_nested(record, sink, path, raw, ("metadata", "kwargs", "vae_latent_channels"), 32)
        _expect_nested(record, sink, path, raw, ("metadata", "kwargs", "sample_rate"), 32000)

    if record["referenced_safetensors"] and not all(item["ok"] for item in record["referenced_safetensors"]):
        record["ok"] = False
    return record


def run_preflight(paths: Iterable[str | Path] = ("models",), *, cwd: str | Path | None = None) -> dict[str, Any]:
    """Run the local asset preflight and return a machine-readable result."""
    cwd_path = Path(cwd).resolve() if cwd is not None else Path.cwd().resolve()
    sink = _IssueSink(cwd_path)
    roots = [Path(path) for path in paths]
    if not roots:
        roots = [Path("models")]

    index_records: list[dict[str, Any]] = []
    config_records: list[dict[str, Any]] = []
    standalone_records: list[dict[str, Any]] = []
    declared_shards: set[Path] = set()

    for root in roots:
        if not root.exists():
            sink.add("error", "missing_root", root, "preflight root does not exist")
            continue

        for index_path in _index_paths(root):
            record, declared = _check_index(index_path, sink)
            index_records.append(record)
            declared_shards.update(declared)

        for config_path in _config_paths(root):
            raw = _read_json(config_path, sink)
            if raw is None:
                continue
            family = _classify_config(config_path, raw)
            if family is None:
                continue
            config_records.append(_validate_config(config_path, raw, family, sink))

        for tensor_path in _safetensors_paths(root):
            if tensor_path.resolve() in declared_shards:
                continue
            standalone_records.append(_check_safetensors_header(tensor_path, sink))

    errors = sum(1 for issue in sink.issues if issue["severity"] == "error")
    warnings = sum(1 for issue in sink.issues if issue["severity"] == "warning")
    exit_code = OK if errors == 0 else ASSET_FAILURE
    result = {
        "schema_version": 1,
        "ok": errors == 0,
        "exit_code": exit_code,
        "roots": [sink.path(root) for root in roots],
        "summary": {
            "indexes": len(index_records),
            "declared_shards": sum(record.get("declared_shard_count", 0) for record in index_records),
            "configs": len(config_records),
            "standalone_safetensors": len(standalone_records),
            "errors": errors,
            "warnings": warnings,
        },
        "issues": sink.issues,
        "indexes": index_records,
        "configs": config_records,
        "standalone_safetensors": standalone_records,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Header-only MiniMax-H3 model asset preflight")
    parser.add_argument("paths", nargs="*", default=["models"], help="model roots or files to inspect")
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON output")
    args = parser.parse_args(argv)

    result = run_preflight(args.paths)
    json.dump(result, sys.stdout, indent=2 if args.pretty else None, sort_keys=True)
    sys.stdout.write("\n")
    return int(result["exit_code"])


if __name__ == "__main__":  # pragma: no cover - exercised through scripts/preflight_assets.py.
    raise SystemExit(main())
