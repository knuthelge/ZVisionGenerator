"""Own writable Web config semantics and path readback contracts."""

from __future__ import annotations

import io
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.error import CommentMark, YAMLError
from ruamel.yaml.tokens import CommentToken

from zvisiongenerator.utils.atomic_write import write_text_atomic
from zvisiongenerator.utils.config import model_reference
from zvisiongenerator.utils.paths import get_ziv_data_dir, is_huggingface_repo_id
from zvisiongenerator.utils.yaml_round_trip import attach_following, delete_key, round_trip_yaml, set_key, yaml11_safe
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
    WritableConfigField(
        key="prompt_enhancer.user_model",
        value_type="string",
        clearable=True,
        empty_string="clear",
        validation_rules=("When set, value must be a Hugging Face repo id (owner/name, optionally @revision) or an existing local directory.",),
        persisted_value_shape="string repo id (optionally @revision) or absolute directory, or null when the platform default is used",
        effective_value_shape="string repo id or directory, or null when no enhancer is configured",
        default_source="prompt_enhancer.model for this platform",
        owning_consumer="Workspace prompt enhancement, auto-enhance jobs, and CLI --enhance",
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


_USER_CONFIG_WRITE_LOCK = threading.RLock()


def persist_writable_config_patch(patch: dict[str, Any], current: Any) -> None:
    """Apply a writable config patch using omitted/null/empty-string semantics."""
    # Web handlers run in worker threads; serialise read-modify-write so concurrent saves cannot drop each other's changes.
    with _USER_CONFIG_WRITE_LOCK:
        override_config = _load_user_config_document()
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


def persist_first_run_skipped() -> None:
    """Record in the user config override that the first-run dialog was skipped, keeping every other key."""
    with _USER_CONFIG_WRITE_LOCK:
        override_config = _load_user_config_document()
        _set_nested_mapping_value(override_config, ("ui", "first_run_skipped"), True)
        write_user_config_override(override_config)


def discover_huggingface_enabled() -> bool:
    """Return whether the user chose to show Hugging Face LoRAs in Discover LoRAs (``ui.discover_huggingface``)."""
    ui = read_user_config_override().get("ui")
    return isinstance(ui, dict) and ui.get("discover_huggingface") is True


def discover_content_filter_setting() -> object:
    """Return the user's ``ui.discover_content_filter`` value, or None when the user config does not set it."""
    ui = read_user_config_override().get("ui")
    return ui.get("discover_content_filter") if isinstance(ui, dict) else None


def persist_discover_huggingface(enabled: bool) -> None:
    """Record in the user config override whether Discover LoRAs shows Hugging Face LoRAs, keeping every other key."""
    with _USER_CONFIG_WRITE_LOCK:
        override_config = _load_user_config_document()
        _set_nested_mapping_value(override_config, ("ui", "discover_huggingface"), enabled)
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


def _load_user_config_document() -> CommentedMap:
    """Load the user config override as a round-trip document that keeps its comments, for editing and saving.

    Reads still go through :func:`read_user_config_override` (PyYAML); this copy is only what a save edits.

    Returns:
        The document; an empty one when the file is missing or empty. A file with only comments keeps them.

    Raises:
        ValueError: When the file cannot be read, is not valid YAML (a repeated key counts), or is not a mapping.
            The file is not changed.
    """
    config_path = get_ziv_data_dir() / "config.yaml"
    if not config_path.is_file():
        return CommentedMap()
    try:
        text = config_path.read_text(encoding="utf-8")
        payload = round_trip_yaml().load(text)
    except (OSError, YAMLError) as exc:
        raise ValueError(f"Failed to read user config override: {exc}") from exc
    if payload is None:
        return _comment_only_document(text)
    if not isinstance(payload, CommentedMap):
        raise ValueError("User config override must be a YAML mapping.")
    return payload


def _comment_only_document(text: str) -> CommentedMap:
    """Return an empty document that writes *text* (a file of comments and blank lines) above its first key."""
    document = CommentedMap()
    if text.strip():
        document.ca.comment = [None, [CommentToken(text if text.endswith("\n") else text + "\n", CommentMark(0))]]
    return document


def write_user_config_override(document: CommentedMap) -> None:
    """Atomically replace the mutable user config override file with *document*, keeping its comments.

    Readers (config loads on other request threads) see either the old or the new file, never a truncated one;
    a symlinked config.yaml and its permissions are preserved.

    Raises:
        TypeError: When *document* is not a :class:`CommentedMap` from :func:`_load_user_config_document`.
    """
    if not isinstance(document, CommentedMap):
        raise TypeError("The user config override must be written from a round-trip document (CommentedMap).")
    config_path = get_ziv_data_dir() / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    round_trip_yaml().dump(document, buffer)
    with _USER_CONFIG_WRITE_LOCK:
        write_text_atomic(config_path, buffer.getvalue())


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
    if field.key == "prompt_enhancer.user_model":
        local = Path(value).expanduser()
        if local.is_dir():
            return str(local.resolve())
        if not is_huggingface_repo_id(value):
            raise ValueError("Enhancer model must be a Hugging Face repo id (owner/name, optionally @revision) or an existing local directory.")
        return value
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
    if key == "prompt_enhancer.user_model":
        model = getattr(web_config, "enhancer_model", None)
        return model_reference(model, getattr(web_config, "enhancer_revision", None)) if model else None
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


def _set_nested_mapping_value(document: CommentedMap, path: tuple[str, ...], value: Any) -> None:
    """Set the value at *path*, creating the mappings it needs; strings PyYAML would read as another type are quoted."""
    cursor = document
    for depth, key in enumerate(path[:-1]):
        child = cursor.get(key)
        if not isinstance(child, CommentedMap):
            set_key(cursor, key, _nested_chain(path[depth + 1 :], value))
            return
        cursor = child
    set_key(cursor, path[-1], yaml11_safe(value) if isinstance(value, str) else value)


def _nested_chain(path: tuple[str, ...], value: Any) -> CommentedMap:
    """Return new mappings that hold *value* at *path*."""
    node: Any = yaml11_safe(value) if isinstance(value, str) else value
    for key in reversed(path):
        wrapper = CommentedMap()
        wrapper[key] = node
        node = wrapper
    return node


def _delete_nested_mapping_value(document: CommentedMap, path: tuple[str, ...]) -> None:
    """Remove the value at *path* and the mappings it leaves empty; comments that followed it stay in the file."""
    chain = [document]
    for key in path[:-1]:
        child = chain[-1].get(key)
        if not isinstance(child, CommentedMap):
            return
        chain.append(child)
    depth = len(chain) - 1
    following, previous = _remove_key(chain[depth], path[depth])
    while depth > 0 and not chain[depth]:
        depth -= 1
        removed, previous = _remove_key(chain[depth], path[depth])
        following = (removed or "") + (following or "") or None
    if following:
        _reattach(chain, path, depth, previous, following)


def _remove_key(node: CommentedMap, key: str) -> tuple[str | None, str | None]:
    """Delete *key* from *node* if present; return the lines that followed it and the key before it."""
    keys = list(node)
    if key not in keys:
        return None, None
    position = keys.index(key)
    return delete_key(node, key), keys[position - 1] if position else None


def _reattach(chain: list[CommentedMap], path: tuple[str, ...], depth: int, previous: str | None, following: str) -> None:
    """Put *following* back below the key before the removed one, else at the start of its mapping.

    A removed first key leaves the lines at the top of its parent key (after ``ui:``); a removed first top-level
    key leaves them in the file header. An emptied document drops them.
    """
    node = chain[depth]
    if previous is not None:
        attach_following(node, previous, following)
    elif depth > 0:
        attach_following(chain[depth - 1], path[depth - 1], following, at_key=True)
    elif node:
        node.ca.comment = node.ca.comment or [None, []]
        node.ca.comment[1] = [*(node.ca.comment[1] or []), CommentToken(following, CommentMark(0))]
