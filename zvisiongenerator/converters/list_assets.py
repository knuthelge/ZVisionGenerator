"""Asset listing — scan directories, detect model types, format output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from zvisiongenerator.utils.image_model_detect import ImageModelInfo, detect_image_model
from zvisiongenerator.utils.video_model_detect import detect_video_model

_AliasPlatformValue = str | dict[str, str]
_AliasMap = dict[str, str | dict[str, _AliasPlatformValue]]


@dataclass
class ModelEntry:
    name: str
    family: str  # "zimage", "flux2_klein", etc.
    size: str | None  # "4b", "9b", or None
    is_distilled: bool


@dataclass(frozen=True)
class VideoModelEntry:
    name: str
    family: str  # "ltx"
    supports_i2v: bool


@dataclass
class LoraEntry:
    name: str
    file_size_mb: float


def list_models(data_dir: Path) -> list[ModelEntry]:
    """Scan data_dir/models/ and return detected model entries sorted by name."""
    models_dir = data_dir / "models"
    if not models_dir.is_dir():
        return []

    entries: list[ModelEntry] = []
    for child in models_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            info: ImageModelInfo = detect_image_model(str(child))
        except Exception:
            info = ImageModelInfo(family="unknown", is_distilled=False, size=None)
        entries.append(
            ModelEntry(
                name=child.name,
                family=info.family,
                size=info.size,
                is_distilled=info.is_distilled,
            )
        )
    entries.sort(key=lambda e: e.name)
    return entries


def list_video_models(data_dir: Path) -> list[VideoModelEntry]:
    """Scan data_dir/models/ and return detected video model entries sorted by name."""
    models_dir = data_dir / "models"
    if not models_dir.is_dir():
        return []

    entries: list[VideoModelEntry] = []
    for child in models_dir.iterdir():
        if not child.is_dir():
            continue
        info = detect_video_model(str(child))
        if info.family == "unknown":
            continue
        entries.append(
            VideoModelEntry(
                name=child.name,
                family=info.family,
                supports_i2v=info.supports_i2v,
            )
        )
    entries.sort(key=lambda e: e.name)
    return entries


def list_loras(data_dir: Path) -> list[LoraEntry]:
    """Scan data_dir/loras/ and return LoRA entries sorted by name."""
    loras_dir = data_dir / "loras"
    if not loras_dir.is_dir():
        return []

    entries: list[LoraEntry] = []
    for child in loras_dir.iterdir():
        if child.suffix == ".safetensors" and child.is_file():
            size_mb = child.stat().st_size / 1024 / 1024
            entries.append(LoraEntry(name=child.stem, file_size_mb=round(size_mb, 1)))
    entries.sort(key=lambda e: e.name)
    return entries


def format_asset_table(
    models: list[ModelEntry] | None = None,
    video_models: list[VideoModelEntry] | None = None,
    loras: list[LoraEntry] | None = None,
    aliases: _AliasMap | None = None,
    platform_labels: dict[str, str] | None = None,
) -> str:
    """Format models, video models, LoRAs, and/or aliases as a human-readable table string."""
    sections: list[str] = []

    if models is not None:
        sections.append(_format_models(models))
    if video_models is not None:
        sections.append(_format_video_models(video_models))
    if loras is not None:
        sections.append(_format_loras(loras))
    if aliases is not None:
        sections.append(_format_aliases(aliases, platform_labels=platform_labels))

    return "\n\n".join(sections)


def _format_models(models: list[ModelEntry]) -> str:
    header = "Models:"
    if not models:
        return f"{header}\n  (none)"

    col_name = "Name"
    col_family = "Family"
    col_size = "Size"

    w_name = max(len(col_name), *(len(m.name) for m in models))
    w_family = max(len(col_family), *(len(m.family) for m in models))
    w_size = max(len(col_size), *(len(m.size or "-") for m in models))

    hdr = f"  {col_name:<{w_name}}  {col_family:<{w_family}}  {col_size:>{w_size}}"
    sep = f"  {'-' * w_name}  {'-' * w_family}  {'-' * w_size}"

    lines = [header, hdr, sep]
    for m in models:
        size_str = m.size or "-"
        lines.append(f"  {m.name:<{w_name}}  {m.family:<{w_family}}  {size_str:>{w_size}}")
    return "\n".join(lines)


def _format_video_models(video_models: list[VideoModelEntry]) -> str:
    header = "Video Models:"
    if not video_models:
        return f"{header}\n  (none)"

    col_name = "Name"
    col_family = "Family"
    col_i2v = "I2V"

    w_name = max(len(col_name), *(len(m.name) for m in video_models))
    w_family = max(len(col_family), *(len(m.family) for m in video_models))
    w_i2v = max(len(col_i2v), 3)

    hdr = f"  {col_name:<{w_name}}  {col_family:<{w_family}}  {col_i2v:>{w_i2v}}"
    sep = f"  {'-' * w_name}  {'-' * w_family}  {'-' * w_i2v}"

    lines = [header, hdr, sep]
    for m in video_models:
        i2v_str = "yes" if m.supports_i2v else "no"
        lines.append(f"  {m.name:<{w_name}}  {m.family:<{w_family}}  {i2v_str:>{w_i2v}}")
    return "\n".join(lines)


def _format_loras(loras: list[LoraEntry]) -> str:
    header = "LoRAs:"
    if not loras:
        return f"{header}\n  (none)"

    col_name = "Name"
    col_size = "Size (MB)"

    w_name = max(len(col_name), *(len(lora.name) for lora in loras))
    w_size = max(len(col_size), *(len(f"{lora.file_size_mb:.1f}") for lora in loras))

    hdr = f"  {col_name:<{w_name}}  {col_size:>{w_size}}"
    sep = f"  {'-' * w_name}  {'-' * w_size}"

    lines = [header, hdr, sep]
    for lora in loras:
        lines.append(f"  {lora.name:<{w_name}}  {f'{lora.file_size_mb:.1f}':>{w_size}}")
    return "\n".join(lines)


def _format_aliases(aliases: _AliasMap, platform_labels: dict[str, str] | None = None) -> str:
    header = "Model Aliases:"
    if not aliases:
        return f"{header}\n  (none)"

    if platform_labels is None:
        from zvisiongenerator.utils.config import load_config
        from zvisiongenerator.utils.platform import get_all_platform_labels

        platform_labels = get_all_platform_labels(load_config())

    w_name = max(len(a) for a in aliases)
    lines = [header]
    for alias, target in sorted(aliases.items()):
        if isinstance(target, dict):
            supported = {k: v for k, v in target.items() if isinstance(v, str)}
            parts = [f"{v} ({platform_labels.get(k, k)})" for k, v in sorted(supported.items())]
            display = " / ".join(parts)
            missing = set(platform_labels.keys()) - set(supported.keys())
            if missing:
                missing_labels = ", ".join(platform_labels.get(p, p) for p in sorted(missing))
                display += f"  ({missing_labels} coming soon)"
        else:
            display = target
        lines.append(f"  {alias:<{w_name}}  → {display}")
    return "\n".join(lines)
