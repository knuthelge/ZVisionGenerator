"""The audio part of a gallery asset: the facts the viewer and the tile read, and the settings a Reuse restores.

The functions read the settings embedded in a take (the config the save step wrote) and the facts the gallery read from
the file, and return plain values. They never read a file themselves.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from zvisiongenerator.utils.audio_preset import AUDIO_KINDS, parse_audio_presets
from zvisiongenerator.utils.paths import expand_home
from zvisiongenerator.utils.provenance import optional_float, optional_int, optional_text
from zvisiongenerator.web.audio_models import runnable_audio_models
from zvisiongenerator.web.workspace_contract import AUDIO_WORKFLOW_VALUES

if TYPE_CHECKING:
    from zvisiongenerator.web.config import WebUiConfig
    from zvisiongenerator.web.gallery import GalleryAsset

__all__ = ["audio_asset_facts", "audio_asset_json", "is_reusable_audio_config"]


def is_reusable_audio_config(config: Mapping[str, Any] | None) -> bool:
    """Return whether *config*, the settings embedded in an audio file, are a take of this app (they name a known kind) and so can be reused."""
    return isinstance(config, Mapping) and config.get("kind") in AUDIO_KINDS


def audio_asset_facts(config: Mapping[str, Any] | None, facts: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return what a gallery asset keeps of an audio file, from its embedded *config* and the *facts* read from the file.

    Keys: ``kind`` (``music``, ``speech``, ``sound``, or ``None`` for a file without this app's settings), ``duration``
    (seconds, from the file, else the recorded one), ``peaks`` (100 values of 0-255, or ``None`` while they are not
    computed yet: the gallery computes them on the first request, see ``gallery.audio_peaks``), ``channels``, ``sample_rate``, ``has_reference_voice``, ``lyrics``, ``seconds`` (the length asked for),
    ``voice_path`` (``~`` expanded), ``voice_text``, ``cut`` and ``parts``.
    """
    reusable = is_reusable_audio_config(config)
    recorded = config if reusable and config is not None else {}
    generation = recorded.get("generation") if isinstance(recorded.get("generation"), Mapping) else {}
    read = facts or {}
    voice_path = optional_text(recorded.get("voice_path"))
    return {
        "kind": recorded.get("kind") if reusable else None,
        "duration": optional_float(read.get("duration")) or optional_float(generation.get("duration")),
        "peaks": [int(peak) for peak in read["peaks"]] if isinstance(read.get("peaks"), list) else None,
        "channels": optional_int(read.get("channels")),
        "sample_rate": optional_int(read.get("sample_rate")),
        "has_reference_voice": voice_path is not None,
        "lyrics": optional_text(recorded.get("lyrics")),
        "seconds": optional_float(recorded.get("seconds")),
        "voice_path": expand_home(voice_path) if voice_path is not None else None,
        "voice_text": optional_text(recorded.get("voice_text")),
        "cut": generation.get("cut") is True,
        "parts": optional_int(generation.get("parts")),
    }


def audio_asset_json(asset: GalleryAsset, web_config: WebUiConfig) -> dict[str, Any]:
    """Convert an audio gallery asset to the SPA JSON shape: the common keys, ``media_type: "audio"``, ``model_label`` and the audio facts.

    ``reuse_workspace_url`` restores the recorded settings and never the speaker's consent.
    """
    audio = asset.audio
    reuse = _reuse_state(asset, web_config)
    return {
        "id": asset.id,
        "url": asset.media_url,
        "thumbnail_url": asset.media_url,
        "filename": asset.name,
        "created_at": datetime.fromtimestamp(asset.modified_at, tz=timezone.utc).isoformat(),
        "workflow": reuse["state"]["requested_workflow"],
        "prompt": asset.prompt,
        "model": asset.model_label,
        "model_label": _model_label(asset.model_label, web_config),
        "file_path": asset.filesystem_path,
        "seed": asset.seed,
        "media_type": "audio",
        "kind": audio["kind"],
        "duration": audio["duration"],
        "peaks": audio["peaks"],
        "channels": audio["channels"],
        "sample_rate": audio["sample_rate"],
        "file_bytes": audio.get("bytes"),
        "has_reference_voice": audio["has_reference_voice"],
        "lyrics": audio["lyrics"],
        "cut": audio["cut"],
        "has_reusable_config": asset.has_reusable_config,
        "reuse_state": reuse["state"],
        "reuse_workspace_url": f"#/workspace?{urlencode(reuse['params'])}",
        "details": {"recorded_workflow": asset.workflow, "generation": _generation(audio)},
    }


def _model_label(model_id: str, web_config: WebUiConfig) -> str | None:
    """Return the label of the configured audio preset *model_id*, or ``None`` when there is no such preset."""
    return next((preset.label for preset in parse_audio_presets(web_config.app_config) if preset.id == model_id), None)


def _generation(audio: Mapping[str, Any]) -> dict[str, Any]:
    """Return the recorded facts of how the take was made that the viewer shows."""
    generation = {key: audio[key] for key in ("duration", "parts") if audio[key] is not None}
    if audio["cut"]:
        generation["cut"] = True
    return generation


def _reuse_state(asset: GalleryAsset, web_config: WebUiConfig) -> dict[str, Any]:
    """Return the reuse state and Reuse query params of an audio asset.

    The workflow is the take's kind; a file without this app's settings opens the first audio workflow. The recorded model
    is carried only when it still runs on this machine, and the workflow is unavailable when no model of its kind does.
    """
    kind = asset.audio["kind"]
    workflow = kind or AUDIO_WORKFLOW_VALUES[0]
    runnable = runnable_audio_models(web_config.app_config)
    model_available = any(preset.id == asset.model_label for preset in runnable)
    requested_model = None if asset.model_label == "Unavailable" else asset.model_label
    resolved_model = requested_model if model_available else None
    workflow_available = any(preset.kind == workflow for preset in runnable)
    fallback_reasons: list[str] = []
    if not workflow_available:
        fallback_reasons.append("no_audio_model")
    if asset.has_reusable_config and requested_model is not None and not model_available:
        fallback_reasons.append("model_not_configured")
    params: dict[str, str] = {"workflow": workflow}
    if asset.has_reusable_config:
        params.update(_reuse_params(asset, resolved_model))
    state = {
        "requested_workflow": workflow,
        "resolved_workflow": workflow,
        "workflow_available": workflow_available,
        "requested_model": requested_model,
        "resolved_model": resolved_model,
        "model_available": model_available,
        "fallback_reasons": fallback_reasons,
    }
    return {"state": state, "params": params}


def _reuse_params(asset: GalleryAsset, model: str | None) -> dict[str, str]:
    """Return the Reuse query params of a take with settings: the fields of the form, never the consent."""
    audio = asset.audio
    params: dict[str, str] = {}
    if model:
        params["model"] = model
    params["prompt"] = asset.prompt
    if audio["lyrics"]:
        params["lyrics"] = audio["lyrics"]
    if audio["seconds"] is not None:
        params["seconds"] = f"{audio['seconds']:g}"
    if audio["voice_path"]:
        params["voice_path"] = audio["voice_path"]
        if audio["voice_text"]:
            params["voice_text"] = audio["voice_text"]
    if asset.seed is not None:
        params["seed"] = str(asset.seed)
    return params
