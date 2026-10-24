"""Stream-convert H3's truncated Qwen3-VL conditioner to MLX 4-bit weights."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx


def _wanted(key: str, num_layers: int) -> bool:
    prefix = "model.language_model."
    if not key.startswith(prefix) or key.startswith("lm_head"):
        return False
    rest = key[len(prefix) :]
    if rest.startswith("layers."):
        return int(rest.split(".")[1]) < num_layers
    return rest.startswith("embed_tokens.") or rest.startswith("norm.")


def convert(
    source: Path,
    output: Path,
    *,
    bits: int = 4,
    group_size: int = 64,
    num_layers: int = 50,
) -> dict[str, int | float]:
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("source and output directories must differ")
    with (source / "model.safetensors.index.json").open() as handle:
        source_index = json.load(handle)
    weight_map = source_index["weight_map"]
    selected = {key: shard for key, shard in weight_map.items() if _wanted(key, num_layers)}
    by_shard: dict[str, list[str]] = {}
    for key, shard in selected.items():
        by_shard.setdefault(shard, []).append(key)

    missing_shards = [name for name in by_shard if not (source / name).is_file()]
    if missing_shards:
        raise FileNotFoundError(
            f"{len(missing_shards)} required source shards are missing, e.g. {missing_shards[:3]}"
        )

    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "config.json", output / "config.json")
    output_map: dict[str, str] = {}
    source_bytes = output_bytes = quantized_tensors = 0

    for shard_name, keys in sorted(by_shard.items()):
        arrays = mx.load(str(source / shard_name))
        converted: dict[str, mx.array] = {}
        for key in sorted(keys):
            tensor = arrays[key]
            source_bytes += tensor.nbytes
            if tensor.ndim == 2 and tensor.shape[-1] % group_size == 0:
                weight, scales, biases = mx.quantize(
                    tensor,
                    group_size=group_size,
                    bits=bits,
                )
                converted[key] = weight
                stem = key[: -len("weight")]
                converted[stem + "scales"] = scales
                converted[stem + "biases"] = biases
                quantized_tensors += 1
            else:
                converted[key] = tensor

        target = output / shard_name
        mx.save_safetensors(str(target), converted)
        for key, tensor in converted.items():
            output_map[key] = shard_name
            output_bytes += tensor.nbytes
        del arrays, converted
        gc.collect()
        mx.clear_cache()
        print(f"wrote {target.name}", flush=True)

    (output / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": output_bytes},
                "weight_map": output_map,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output / "quant_config.json").write_text(
        json.dumps(
            {
                "bits": bits,
                "group_size": group_size,
                "num_layers": num_layers,
                "source_bytes": source_bytes,
                "output_bytes": output_bytes,
                "quantized_tensors": quantized_tensors,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {
        "source_bytes": source_bytes,
        "output_bytes": output_bytes,
        "quantized_tensors": quantized_tensors,
        "compression": source_bytes / output_bytes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bits", type=int, default=4, choices=(2, 3, 4, 6, 8))
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=50)
    args = parser.parse_args(argv)
    result = convert(
        args.source,
        args.output,
        bits=args.bits,
        group_size=args.group_size,
        num_layers=args.num_layers,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())