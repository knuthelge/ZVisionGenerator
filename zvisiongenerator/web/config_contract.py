"""Own writable Web config semantics and path readback contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from zvisiongenerator.utils.paths import get_ziv_data_dir
from zvisiongenerator.web.defaults import resolve_image_ratio_size_defaults


type ConfigValue = str | int | float | bool | None


@dataclass(frozen=True)
class WritableConfigField:
    """Describe one backend-owned writable config key."""

    key: str
    value_type: str
    clearable: bool
    empty_string: str
    validation_rules: tuple[str, ...]
    persisted_value_shape: str
    effective_value_shape: str
    default_source: str
    owning_consumer: str

    def to_schema(self, *, value: ConfigValue, effective_value: ConfigValue) -> dict[str, Any]:
        """Serialize the field inventory using the public snake_case wire shape."""
        return {
            "key": self.key,
            "type": self.value_type,
            "clearable": self.clearable,
            "empty_string": self.empty_string,
            "omitted": "unchanged",
            "null": "clear" if self.clearable else "reject",
            "value": value,
            "persisted_value": value,
            "persisted_value_shape": self.persisted_value_shape,
            "effective_value": effective_value,
            "effective_value_shape": self.effective_value_shape,
            "default_source": self.default_source,
            "validation_rules": list(self.validation_rules),
            "owning_consumer": self.owning_consumer,
        }


WRITABLE_CONFIG_FIELDS: tuple[WritableConfigField, ...] = (
    WritableConfigField(
        key="ui.default_models.image",
        value_type="string",
        clearable=True,
        empty_string="clear",
        validation_rules=("When set, value must be one of image_model_options.",),
        persisted_value_shape="string model id or null when no user override is persisted",
        effective_value_shape="string model id or null when no image model is available",
        default_source="ui.default_models.image, then first discovered image model",
        owning_consumer="Config page and workspace bootstrap",
    ),
    WritableConfigField(
        key="ui.default_models.video",
        value_type="string",
        clearable=True,
        empty_string="clear",
        validation_rules=("When set, value must be one of video_model_options.",),
        persisted_value_shape="string model id or null when no user override is persisted",
        effective_value_shape="string model id or null when no video model is available",
        default_source="ui.default_models.video, then first discovered video model",
        owning_consumer="Config page and workspace bootstrap",
    ),
    WritableConfigField(
        key="generation.default_size",
        value_type="string",
        clearable=True,
        empty_string="clear",
        validation_rules=("When set, value must be valid for the effective generation.default_ratio.",),
        persisted_value_shape="string size key or null when no user override is persisted",
        effective_value_shape="string size key",
        default_source="generation.default_size from layered config",
        owning_consumer="Config page, image CLI defaults, and workspace bootstrap",
    ),
    WritableConfigField(
        key="ui.output_dir",
        value_type="string",
        clearable=True,
        empty_string="clear",
        validation_rules=("When set, path is expanded, made absolute relative to the current process, and created if needed.",),
        persisted_value_shape="absolute host directory string or null when no user override is persisted",
        effective_value_shape="absolute host directory string",
        default_source="ui.output_dir from layered config, then ZIV data directory outputs folder",
        owning_consumer="Config page, workspace submissions, media serving, gallery, and history",
    ),
)


def resolve_output_dir(value: Any, *, data_dir: Path | None = None, create: bool = True) -> Path:
    """Resolve the effective output directory with the backend-owned path rules."""
    if value is None:
        output_dir = (data_dir or get_ziv_data_dir()) / "outputs"
    else:
        text = str(value).strip()
        if not text:
            output_dir = (data_dir or get_ziv_data_dir()) / "outputs"
        else:
            output_dir = Path(text).expanduser()
            if not output_dir.is_absolute():
                output_dir = (Path.cwd() / output_dir).resolve()
    if create:
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_models_dir(*, data_dir: Path | None = None) -> Path:
    """Return the canonical installed-model directory."""
    return (data_dir or get_ziv_data_dir()) / "models"


def resolve_loras_dir(*, data_dir: Path | None = None) -> Path:
    """Return the canonical installed-LoRA directory."""
    return (data_dir or get_ziv_data_dir()) / "loras"


def normalize_user_directory(value: str) -> str:
    """Normalize and create a user-provided directory before persistence."""
    return str(resolve_output_dir(value, create=True))


def build_writable_config_schema(web_config: Any) -> dict[str, Any]:
    """Build the public writable config contract from current effective state."""
    override_config = read_user_config_override()
    fields = [
        field.to_schema(
            value=_get_nested_value(override_config, tuple(field.key.split("."))),
            effective_value=_effective_field_value(web_config, field.key),
        )
        for field in WRITABLE_CONFIG_FIELDS
    ]
    return {
        "version": 1,
        "semantics": {
            "omitted": "unchanged",
            "null": "clear for clearable fields",
            "empty_string": "normalized by field empty_string behavior before persistence",
        },
        "fields": fields,
    }


def persist_writable_config_patch(patch: dict[str, Any], current: Any) -> None:
    """Apply a writable config patch using omitted/null/empty-string semantics."""
    override_config = read_user_config_override()
    fields_by_key = {field.key: field for field in WRITABLE_CONFIG_FIELDS}
    for key, raw_value in patch.items():
        field = fields_by_key.get(key)
        if field is None:
            raise ValueError(f"Unknown writable config field: {key}")
        path = tuple(key.split("."))
        value = _normalize_patch_value(field, raw_value)
        if value is None:
            _delete_nested_mapping_value(override_config, path)
            continue
        _set_nested_mapping_value(override_config, path, _validate_config_value(field, value, current))
    write_user_config_override(override_config)


def read_user_config_override() -> dict[str, Any]:
    """Read the mutable user config override file, if present."""
    config_path = get_ziv_data_dir() / "config.yaml"
    if not config_path.is_file():
        return {}
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Failed to read user config override: {exc}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("User config override must be a YAML mapping.")
    return payload


def write_user_config_override(payload: dict[str, Any]) -> None:
    """Write the mutable user config override file."""
    config_path = get_ziv_data_dir() / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _normalize_patch_value(field: WritableConfigField, raw_value: Any) -> ConfigValue:
    if raw_value is None:
        if field.clearable:
            return None
        raise ValueError(f"Field '{field.key}' cannot be cleared.")
    if field.value_type == "string":
        if not isinstance(raw_value, str):
            raise ValueError(f"Field '{field.key}' must be a string.")
        text = raw_value.strip()
        if not text:
            if field.empty_string == "clear" and field.clearable:
                return None
            raise ValueError(f"Field '{field.key}' cannot be empty.")
        return text
    raise ValueError(f"Unsupported writable config field type: {field.value_type}")


def _validate_config_value(field: WritableConfigField, value: ConfigValue, current: Any) -> ConfigValue:
    if not isinstance(value, str):
        raise ValueError(f"Field '{field.key}' must be a string.")
    if field.key == "ui.default_models.image":
        if value not in current.image_model_options:
            raise ValueError("Default image model must be one of the discovered image models.")
        return value
    if field.key == "ui.default_models.video":
        if value not in current.video_model_options:
            raise ValueError("Default video model must be one of the discovered video models.")
        return value
    if field.key == "generation.default_size":
        default_ratio, _default_size = resolve_image_ratio_size_defaults(current)
        valid_sizes = current.image_size_options.get(default_ratio, ())
        if value not in valid_sizes:
            raise ValueError(f"Default base resolution must be one of {list(valid_sizes)} for ratio '{default_ratio}'.")
        return value
    if field.key == "ui.output_dir":
        return normalize_user_directory(value)
    raise ValueError(f"Unknown writable config field: {field.key}")


def _effective_field_value(web_config: Any, key: str) -> ConfigValue:
    if key == "ui.default_models.image":
        return web_config.default_models.image
    if key == "ui.default_models.video":
        return web_config.default_models.video
    if key == "generation.default_size":
        _default_ratio, default_size = resolve_image_ratio_size_defaults(web_config)
        return default_size
    if key == "ui.output_dir":
        return web_config.output_dir
    raise ValueError(f"Unknown writable config field: {key}")


def _get_nested_value(payload: dict[str, Any], path: tuple[str, ...]) -> ConfigValue:
    cursor: Any = payload
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
    if isinstance(cursor, (str, int, float, bool)) or cursor is None:
        return cursor
    return None


def _set_nested_mapping_value(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = payload
    for key in path[:-1]:
        next_value = cursor.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[key] = next_value
        cursor = next_value
    cursor[path[-1]] = value


def _delete_nested_mapping_value(payload: dict[str, Any], path: tuple[str, ...]) -> None:
    parents: list[tuple[dict[str, Any], str]] = []
    cursor = payload
    for key in path[:-1]:
        next_value = cursor.get(key)
        if not isinstance(next_value, dict):
            return
        parents.append((cursor, key))
        cursor = next_value
    cursor.pop(path[-1], None)
    for parent, key in reversed(parents):
        child = parent.get(key)
        if isinstance(child, dict) and not child:
            parent.pop(key, None)
