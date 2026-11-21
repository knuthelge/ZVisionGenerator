"""Read the fields a Web UI request names, with the defaults and choices a job form resolves them by.

Both a submitted job form and a JSON body such as ``POST /api/memory`` read their ratio, size, model and steps through these
functions, so a request the memory check accepts is one the job accepts, and each refuses a bad value with the same message.
"""

from __future__ import annotations

from typing import Any

from zvisiongenerator.core.video_family import UNDESCRIBED_UPSCALE_STEPS, SettledSteps, settle_video_steps
from zvisiongenerator.utils.config import video_family
from zvisiongenerator.utils.video_model_detect import VideoModelInfo
from zvisiongenerator.web.config import WebUiConfig
from zvisiongenerator.web.defaults import default_image_size_for_ratio, default_video_size_for_ratio

__all__ = [
    "image_ratio",
    "image_size_name",
    "optional_text",
    "settled_video_steps",
    "text_or_default",
    "video_model_name",
    "video_ratio_and_size",
]


def optional_text(form: Any, key: str) -> str | None:
    """Return the stripped text of *key*, or ``None`` when it is absent or blank."""
    value = form.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def text_or_default(form: Any, key: str, default: str | None) -> str | None:
    """Return the text of *key*, else *default*."""
    return optional_text(form, key) or default


def _choice_or_default(form: Any, key: str, choices: tuple[str, ...], default: str) -> str:
    """Return the text of *key*, else *default*, which must be one of *choices*.

    Raises:
        ValueError: If the value is not one of *choices*.
    """
    value = optional_text(form, key) or default
    if value not in choices:
        raise ValueError(f"Field '{key}' must be one of {list(choices)}.")
    return value


def image_ratio(form: Any, web_config: WebUiConfig) -> str:
    """Return the image ratio a request names, else the configured default.

    Raises:
        ValueError: If it is not one the Web UI offers.
    """
    return _choice_or_default(form, "ratio", web_config.image_ratios, web_config.app_config["generation"].get("default_ratio", "2:3"))


def image_size_name(form: Any, web_config: WebUiConfig, ratio: str) -> str:
    """Return the image size name a request names for *ratio*, else the configured default.

    Raises:
        ValueError: If it is not one the Web UI offers for *ratio*.
    """
    return _choice_or_default(form, "size", web_config.image_size_options[ratio], default_image_size_for_ratio(web_config, ratio))


def video_model_name(form: Any, web_config: WebUiConfig) -> str:
    """Return the video model a request names, else the configured default.

    Raises:
        ValueError: If neither names one.
    """
    model_name = text_or_default(form, "model", web_config.default_models.video)
    if not model_name:
        raise ValueError("A video model is required.")
    return model_name


def video_ratio_and_size(form: Any, web_config: WebUiConfig) -> tuple[str, str]:
    """Return the ratio and size a request names, else the configured defaults.

    Raises:
        ValueError: If a value is not one the Web UI offers.
    """
    ratio = _choice_or_default(form, "ratio", web_config.video_ratios, web_config.app_config.get("video_generation", {}).get("default_ratio", "16:9"))
    return ratio, _choice_or_default(form, "size", web_config.video_size_options[ratio], default_video_size_for_ratio(web_config, ratio))


def settled_video_steps(steps: int | None, upscale: bool, config: dict[str, Any], model_info: VideoModelInfo) -> SettledSteps:
    """Settle the requested *steps* of a video job by the upscale and the model family's step rules."""
    family = video_family(config, model_info.family)
    return settle_video_steps(steps, upscale=upscale, family=family, default_upscale_steps=family.default_upscale_steps if family is not None else UNDESCRIBED_UPSCALE_STEPS)
