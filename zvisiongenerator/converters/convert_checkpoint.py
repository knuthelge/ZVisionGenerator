"""
Convert single-file safetensors checkpoints to HuggingFace diffusers
directory format. The primary input format is safetensors / LDLM
single-file checkpoints; the output is a standard diffusers model
directory that mflux and other diffusers-based tools can load.

Supported model types:
  - zimage      : Z-Image-Turbo (manual key remapping)
  - flux2-klein-4b : FLUX.2 Klein 4B (manual key remapping)
  - flux2-klein-9b : FLUX.2 Klein 9B (manual key remapping)

Only the transformer weights are converted from the checkpoint.
Text encoder, VAE, tokenizer, and scheduler are downloaded from the
base HuggingFace repo since they are identical to the base model.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

from zvisiongenerator.utils.paths import get_ziv_data_dir


# ── FLUX.2 Klein HuggingFace repos ──────────────────────────────────────────

FLUX2_KLEIN_REPOS = {
    "flux2-klein-4b": "black-forest-labs/FLUX.2-klein-4B",
    "flux2-klein-9b": "black-forest-labs/FLUX.2-klein-9B",
}


# ── Transformer key conversion ──────────────────────────────────────────────

# String replacements applied in order after stripping the prefix
KEY_REPLACEMENTS = [
    ("final_layer.", "all_final_layer.2-1."),
    ("x_embedder.", "all_x_embedder.2-1."),
    (".attention.out.bias", ".attention.to_out.0.bias"),
    (".attention.out.weight", ".attention.to_out.0.weight"),
    (".attention.q_norm.weight", ".attention.norm_q.weight"),
    (".attention.k_norm.weight", ".attention.norm_k.weight"),
]

TRANSFORMER_PREFIX = "model.diffusion_model."

# Key prefixes indicating the state dict is already in diffusers format
DIFFUSERS_PREFIXES = (
    "transformer_blocks.",
    "single_transformer_blocks.",
    "x_embedder.",
    "context_embedder.",
    "norm_out.",
    "proj_out.",
    "time_text_embed.",
    "time_guidance_embed.",
)

# Key prefixes for unprefixed safetensors format (missing model.diffusion_model.)
UNPREFIXED_SAFETENSORS_PREFIXES = (
    "double_blocks.",
    "single_blocks.",
    "img_in.",
    "txt_in.",
    "time_in.",
    "guidance_in.",
    "final_layer.",
    "double_stream_modulation_img.",
    "double_stream_modulation_txt.",
    "single_stream_modulation.",
)


def _ensure_bfloat16(state_dict: dict) -> dict:
    """Cast any non-standard dtypes (e.g., float8) to bfloat16 for mflux compatibility."""
    import torch

    result = {}
    casted = 0
    for key, tensor in state_dict.items():
        if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            result[key] = tensor.to(torch.bfloat16)
            casted += 1
        else:
            result[key] = tensor
    if casted:
        print(f"  Cast {casted} tensors from non-standard dtype to bfloat16")
    return result


def convert_transformer_keys(state_dict: dict) -> dict:
    """Convert safetensors transformer keys to HF diffusers format."""
    converted = {}

    for key, tensor in state_dict.items():
        if not key.startswith(TRANSFORMER_PREFIX):
            continue

        # Step 1: Strip prefix
        new_key = key[len(TRANSFORMER_PREFIX) :]

        # Step 4: Drop norm_final.weight
        if new_key == "norm_final.weight":
            print(f"  Dropping: {key}")
            continue

        # Step 2: Apply string replacements
        for old, new in KEY_REPLACEMENTS:
            new_key = new_key.replace(old, new)

        # Step 3: Split merged QKV weights
        if ".attention.qkv.weight" in new_key:
            q, k, v = tensor.chunk(3, dim=0)
            q_key = new_key.replace(".attention.qkv.weight", ".attention.to_q.weight")
            k_key = new_key.replace(".attention.qkv.weight", ".attention.to_k.weight")
            v_key = new_key.replace(".attention.qkv.weight", ".attention.to_v.weight")
            converted[q_key] = q
            converted[k_key] = k
            converted[v_key] = v
            continue

        converted[new_key] = tensor

    return converted


def download_base_components(base_model: str, output_dir: Path, use_symlinks: bool):
    """Download text_encoder, vae, tokenizer, scheduler, and config files from the base HF repo."""
    from huggingface_hub import hf_hub_download, snapshot_download

    # Download non-transformer components via snapshot_download
    print(f"Downloading base model components from {base_model}...")
    cache_dir = snapshot_download(
        repo_id=base_model,
        allow_patterns=[
            "model_index.json",
            "text_encoder/*",
            "vae/*",
            "tokenizer/*",
            "scheduler/*",
        ],
    )

    # Download transformer config.json separately
    print("Downloading transformer config.json...")
    transformer_config = hf_hub_download(
        repo_id=base_model,
        filename="transformer/config.json",
    )

    # Link or copy base model files into output directory
    action = "Copying" if not use_symlinks else "Symlinking"
    print(f"{action} base model files to output directory...")

    cache_path = Path(cache_dir)

    # Directories/files to copy from the snapshot
    items_to_link = [
        "model_index.json",
        "text_encoder",
        "vae",
        "tokenizer",
        "scheduler",
    ]

    for item in items_to_link:
        src = cache_path / item
        dst = output_dir / item
        if not src.exists():
            print(f"  Skipping {item} (not found in base repo)")
            continue
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()

        if use_symlinks:
            try:
                os.symlink(src.resolve(), dst)
            except OSError:
                if src.is_dir():
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
        else:
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        print(f"  {item}")

    # Handle transformer/config.json
    transformer_dir = output_dir / "transformer"
    transformer_dir.mkdir(exist_ok=True)
    config_dst = transformer_dir / "config.json"
    if config_dst.exists() or config_dst.is_symlink():
        config_dst.unlink()
    if use_symlinks:
        try:
            os.symlink(Path(transformer_config).resolve(), config_dst)
        except OSError:
            shutil.copy2(transformer_config, config_dst)
    else:
        shutil.copy2(transformer_config, config_dst)
    print("  transformer/config.json")


# ── FLUX.2 Klein key conversion ──────────────────────────────────────────────

# Simple top-level renames (applied first, after stripping prefix)
FLUX2_TOP_RENAMES = {
    "img_in": "x_embedder",
    "txt_in": "context_embedder",
    "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
    "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
    "guidance_in.in_layer": "time_guidance_embed.guidance_embedder.linear_1",
    "guidance_in.out_layer": "time_guidance_embed.guidance_embedder.linear_2",
    "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
    "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
    "single_stream_modulation.lin": "single_stream_modulation.linear",
    "final_layer.linear": "proj_out",
}

# Renames within double blocks (after "double_blocks.{N}." is replaced with "transformer_blocks.{N}.")
FLUX2_DOUBLE_BLOCK_RENAMES = {
    "img_attn.norm.query_norm": "attn.norm_q",
    "img_attn.norm.key_norm": "attn.norm_k",
    "img_attn.proj": "attn.to_out.0",
    "img_mlp.0": "ff.linear_in",
    "img_mlp.2": "ff.linear_out",
    "txt_attn.norm.query_norm": "attn.norm_added_q",
    "txt_attn.norm.key_norm": "attn.norm_added_k",
    "txt_attn.proj": "attn.to_add_out",
    "txt_mlp.0": "ff_context.linear_in",
    "txt_mlp.2": "ff_context.linear_out",
}

# Renames within single blocks (after "single_blocks.{N}." is replaced with "single_transformer_blocks.{N}.")
FLUX2_SINGLE_BLOCK_RENAMES = {
    "linear1": "attn.to_qkv_mlp_proj",
    "linear2": "attn.to_out",
    "norm.query_norm": "attn.norm_q",
    "norm.key_norm": "attn.norm_k",
}


def convert_flux2_transformer_keys(state_dict: dict) -> dict:
    """Convert safetensors FLUX.2 Klein transformer keys to HF diffusers format.

    Supports three input formats:
    - Prefixed safetensors keys (model.diffusion_model.*)
    - Unprefixed safetensors keys (double_blocks.*, single_blocks.*, etc.)
    - Already-converted diffusers keys (transformer_blocks.*, etc.)
    """
    import torch

    # ── Auto-detect source format ──
    has_prefixed = any(k.startswith(TRANSFORMER_PREFIX) for k in state_dict)
    has_diffusers = any(k.startswith(DIFFUSERS_PREFIXES) for k in state_dict)
    has_unprefixed = not has_prefixed and any(k.startswith(UNPREFIXED_SAFETENSORS_PREFIXES) for k in state_dict)

    if has_diffusers and not has_prefixed:
        # Already in diffusers format — pass through transformer keys only
        print("  Detected format: diffusers (already converted)")
        result = {k: v for k, v in state_dict.items() if k.startswith(DIFFUSERS_PREFIXES)}
        print(f"  Passing through {len(result)} transformer keys as-is")
        return result

    if has_unprefixed:
        # Unprefixed safetensors — add prefix so the main loop can process them
        print("  Detected format: unprefixed safetensors — adding prefix before conversion")
        state_dict = {f"{TRANSFORMER_PREFIX}{k}": v for k, v in state_dict.items()}
    elif has_prefixed:
        print("  Detected format: prefixed safetensors")

    converted = {}

    for key, tensor in state_dict.items():
        if not key.startswith(TRANSFORMER_PREFIX):
            continue

        # Strip prefix
        new_key = key[len(TRANSFORMER_PREFIX) :]

        # Handle final_layer.adaLN_modulation.1 — scale/shift swap
        if new_key.startswith("final_layer.adaLN_modulation.1."):
            suffix = new_key.split("final_layer.adaLN_modulation.1.")[-1]
            shift, scale = tensor.chunk(2, dim=0)
            converted[f"norm_out.linear.{suffix}"] = torch.cat([scale, shift], dim=0)
            continue

        # Handle double blocks — QKV split + renames
        double_match = re.match(r"double_blocks\.(\d+)\.(.*)", new_key)
        if double_match:
            block_idx = double_match.group(1)
            rest = double_match.group(2)
            hf_prefix = f"transformer_blocks.{block_idx}"

            # QKV split for image attention
            if rest.startswith("img_attn.qkv."):
                suffix = rest.split("img_attn.qkv.")[-1]  # "weight" or "bias"
                q, k, v = tensor.chunk(3, dim=0)
                converted[f"{hf_prefix}.attn.to_q.{suffix}"] = q
                converted[f"{hf_prefix}.attn.to_k.{suffix}"] = k
                converted[f"{hf_prefix}.attn.to_v.{suffix}"] = v
                continue

            # QKV split for text attention
            if rest.startswith("txt_attn.qkv."):
                suffix = rest.split("txt_attn.qkv.")[-1]
                q, k, v = tensor.chunk(3, dim=0)
                converted[f"{hf_prefix}.attn.add_q_proj.{suffix}"] = q
                converted[f"{hf_prefix}.attn.add_k_proj.{suffix}"] = k
                converted[f"{hf_prefix}.attn.add_v_proj.{suffix}"] = v
                continue

            # Apply double block renames
            for old_part, new_part in FLUX2_DOUBLE_BLOCK_RENAMES.items():
                if rest.startswith(old_part):
                    rest = rest.replace(old_part, new_part, 1)
                    break

            # .scale → .weight in norm contexts
            rest = re.sub(r"\.scale$", ".weight", rest)

            converted[f"{hf_prefix}.{rest}"] = tensor
            continue

        # Handle single blocks — renames only (no QKV split)
        single_match = re.match(r"single_blocks\.(\d+)\.(.*)", new_key)
        if single_match:
            block_idx = single_match.group(1)
            rest = single_match.group(2)
            hf_prefix = f"single_transformer_blocks.{block_idx}"

            # Apply single block renames
            for old_part, new_part in FLUX2_SINGLE_BLOCK_RENAMES.items():
                if rest.startswith(old_part):
                    rest = rest.replace(old_part, new_part, 1)
                    break

            # .scale → .weight in norm contexts
            rest = re.sub(r"\.scale$", ".weight", rest)

            converted[f"{hf_prefix}.{rest}"] = tensor
            continue

        # Handle top-level renames
        matched = False
        for old_name, new_name in FLUX2_TOP_RENAMES.items():
            if new_key.startswith(old_name):
                suffix = new_key[len(old_name) :]
                converted[f"{new_name}{suffix}"] = tensor
                matched = True
                break

        if not matched:
            # Pass through any unrecognized keys as-is
            converted[new_key] = tensor

    return converted


def convert_flux2_klein(input_path: Path, output_dir: Path, model_type: str, use_symlinks: bool):
    """Convert a safetensors FLUX.2 Klein checkpoint to HF diffusers format (manual key remapping)."""
    from safetensors.torch import load_file, save_file

    repo_id = FLUX2_KLEIN_REPOS[model_type]

    # Step 1: Load checkpoint
    print(f"Loading checkpoint: {input_path}")
    state_dict = load_file(str(input_path))
    print(f"  Loaded {len(state_dict)} keys")

    transformer_keys = [k for k in state_dict if k.startswith(TRANSFORMER_PREFIX)]
    print(f"  Transformer: {len(transformer_keys)} keys")

    # Step 2: Convert transformer keys
    print("\nConverting FLUX.2 Klein transformer weights...")
    converted = convert_flux2_transformer_keys(state_dict)
    converted = _ensure_bfloat16(converted)
    print(f"  Converted to {len(converted)} keys (QKV splits: {len(transformer_keys)} → {len(converted)})")

    # Validate — never save an empty transformer
    if not converted:
        sample_keys = list(state_dict.keys())[:5]
        raise RuntimeError(f"Conversion produced 0 transformer keys. The input checkpoint may use an unsupported key format. Sample input keys: {sample_keys}")

    # Step 3: Save converted transformer
    transformer_dir = output_dir / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    transformer_path = transformer_dir / "diffusion_pytorch_model.safetensors"
    print(f"\nSaving transformer weights to {transformer_path}")
    save_file(converted, str(transformer_path))
    print("  Done")

    # Step 4: Download non-transformer components from HF repo
    print()
    download_base_components(repo_id, output_dir, use_symlinks)


def _cmd_model(args):
    """Handle the 'model' subcommand — convert a checkpoint to diffusers format."""
    from safetensors.torch import load_file, save_file

    input_path = Path(args.input).expanduser().resolve()

    # Validate input
    if not input_path.is_file():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output directory
    output_dir = get_ziv_data_dir() / "models"
    if args.name is not None:
        from zvisiongenerator.converters.lora_import import _validate_name

        try:
            model_name = _validate_name(args.name)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        model_name = input_path.stem.removesuffix(".safetensors")
    output_dir = output_dir / model_name

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    use_symlinks = not args.copy

    if args.model_type in FLUX2_KLEIN_REPOS:
        # ── FLUX.2 Klein conversion path ──
        convert_flux2_klein(input_path, output_dir, args.model_type, use_symlinks)
    else:
        # ── Z-Image conversion path (unchanged) ──
        # Step 1: Load checkpoint
        print(f"Loading checkpoint: {input_path}")
        state_dict = load_file(str(input_path))
        print(f"  Loaded {len(state_dict)} keys")

        # Count keys by component
        transformer_keys = [k for k in state_dict if k.startswith(TRANSFORMER_PREFIX)]
        te_keys = [k for k in state_dict if k.startswith("text_encoders.")]
        vae_keys = [k for k in state_dict if k.startswith("vae.")]
        print(f"  Transformer: {len(transformer_keys)} keys")
        print(f"  Text encoder: {len(te_keys)} keys (using base model)")
        print(f"  VAE: {len(vae_keys)} keys (using base model)")

        # Step 2: Convert transformer keys
        print("\nConverting transformer weights...")
        converted = convert_transformer_keys(state_dict)
        converted = _ensure_bfloat16(converted)
        print(f"  Converted to {len(converted)} keys (QKV split: {len(transformer_keys)} → {len(converted)})")

        # Validate — never save an empty transformer
        if not converted:
            print(
                "\nError: Conversion produced 0 transformer keys. The input checkpoint may use an unsupported key format.",
                file=sys.stderr,
            )
            sample_keys = list(state_dict.keys())[:5]
            print(f"  Sample input keys: {sample_keys}", file=sys.stderr)
            sys.exit(1)

        # Step 3: Save converted transformer
        transformer_dir = output_dir / "transformer"
        transformer_dir.mkdir(exist_ok=True)
        transformer_path = transformer_dir / "diffusion_pytorch_model.safetensors"
        print(f"\nSaving transformer weights to {transformer_path}")
        save_file(converted, str(transformer_path))
        print("  Done")

        # Step 4: Download and link base model components
        print()
        download_base_components(args.base_model, output_dir, use_symlinks)

    print(f"\nConversion complete! Output directory: {output_dir}")
    print("\nDirectory structure:")
    for item in sorted(output_dir.rglob("*")):
        if item.is_file() or item.is_symlink():
            rel = item.relative_to(output_dir)
            suffix = " → " + str(os.readlink(item)) if item.is_symlink() else ""
            print(f"  {rel}{suffix}")


def _cmd_lora(args):
    """Handle the 'lora' subcommand — import a LoRA file."""
    from zvisiongenerator.converters.lora_import import import_lora_hf, import_lora_local

    dest_dir = get_ziv_data_dir() / "loras"

    try:
        if args.input:
            source = Path(args.input).expanduser().resolve()
            dest = import_lora_local(source, dest_dir, name=args.name)
        else:
            dest = import_lora_hf(args.hf, dest_dir, filename=args.file, name=args.name)
    except (FileNotFoundError, ValueError, FileExistsError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"LoRA imported: {dest}")


def _cmd_list(args):
    """Handle the 'list' subcommand — list installed models, video models, LoRAs, and aliases."""
    from zvisiongenerator.converters.list_assets import format_asset_table, list_loras, list_models, list_video_models
    from zvisiongenerator.utils.config import load_config
    from zvisiongenerator.utils.platform import get_all_platform_labels

    data_dir = get_ziv_data_dir()

    if not args.models and not args.loras:
        models = list_models(data_dir)
        video_models = list_video_models(data_dir)
        loras = list_loras(data_dir)
    else:
        models = list_models(data_dir) if args.models else None
        video_models = list_video_models(data_dir) if args.models else None
        loras = list_loras(data_dir) if args.loras else None

    try:
        config = load_config()
    except ValueError:
        config = {}
    aliases = config.get("model_aliases", {})
    platform_labels = get_all_platform_labels(config) if config else None

    print(format_asset_table(models=models, video_models=video_models, loras=loras, aliases=aliases or None, platform_labels=platform_labels))


def _build_model_parser(*, prog: str = "ziv-model") -> argparse.ArgumentParser:
    """Build the argument parser for the model/LoRA management CLI."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Import and manage models and LoRAs for Z-Vision Generator",
    )
    subparsers = parser.add_subparsers(dest="command")

    # model subcommand
    model_parser = subparsers.add_parser("model", help="Convert a checkpoint to diffusers format")
    model_parser.add_argument("-i", "--input", required=True, help="Path to .safetensors checkpoint file")
    model_parser.add_argument("--name", default=None, help="Custom model folder name (default: checkpoint filename)")
    model_parser.add_argument(
        "--model-type",
        choices=["zimage", "flux2-klein-4b", "flux2-klein-9b"],
        default="zimage",
        help="Type of model to convert (default: zimage)",
    )
    model_parser.add_argument(
        "--base-model",
        default="Tongyi-MAI/Z-Image-Turbo",
        help="HuggingFace repo ID for the base model (default: Tongyi-MAI/Z-Image-Turbo, only used for zimage)",
    )
    model_parser.add_argument("--copy", action="store_true", help="Copy base model files instead of symlinking")

    # lora subcommand
    lora_parser = subparsers.add_parser("lora", help="Import a LoRA file")
    lora_source = lora_parser.add_mutually_exclusive_group(required=True)
    lora_source.add_argument("-i", "--input", help="Path to local .safetensors file")
    lora_source.add_argument("--hf", help="HuggingFace repo ID to download from")
    lora_parser.add_argument("--file", default=None, help="Specific .safetensors filename in HF repo")
    lora_parser.add_argument("--name", default=None, help="Custom LoRA name (without .safetensors extension)")

    # list subcommand
    list_parser = subparsers.add_parser("list", help="List installed models, video models, and LoRAs")
    list_parser.add_argument("--models", action="store_true", help="Show only models")
    list_parser.add_argument("--loras", action="store_true", help="Show only LoRAs")

    return parser


def main(*, prog: str = "ziv-model") -> None:
    """Entry point for the model/LoRA management CLI."""
    parser = _build_model_parser(prog=prog)
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "model":
        _cmd_model(args)
    elif args.command == "lora":
        _cmd_lora(args)
    elif args.command == "list":
        _cmd_list(args)


if __name__ == "__main__":
    main()
