"""Hugging Face access: where the token comes from and one actionable line for a Hub refusal."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping

TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
LOGIN_SOURCE = "hf auth login"
_HUB_ROOT = "https://huggingface.co"
_LOGIN_HINT = f"run `{LOGIN_SOURCE}`"


def token_source(*, read_token: Callable[[], str | None] | None = None, environ: Mapping[str, str] | None = None) -> str | None:
    """Return where the Hugging Face token comes from, or ``None`` when there is none.

    Args:
        read_token: Returns the token like ``huggingface_hub.get_token``; defaults to it.
        environ: The environment to match the token against; defaults to ``os.environ``.

    Returns:
        The environment variable holding the token, ``"hf auth login"`` for a saved token, or ``None``.
        A token reader that raises (for example an unavailable OIDC token) counts as no token.
    """
    try:
        if read_token is None:
            from huggingface_hub import get_token as read_token
        token = read_token()
    except Exception:  # noqa: BLE001 - an unreadable token is the same as no token
        return None
    if not token:
        return None
    env = os.environ if environ is None else environ
    for name in TOKEN_ENV_VARS:
        if env.get(name) == token:
            return name
    return LOGIN_SOURCE


def refusal_message(exc: BaseException, *, token_present: bool, repo_id: str | None = None) -> str | None:
    """Return one line saying what to do about a Hub refusal in *exc* (or its cause), or ``None`` for anything else.

    A refusal is a gated repo, a repo the Hub reports as not found (which includes private and gated repos
    without access), or a 401/403 answer. Server errors, missing revisions and non-Hub errors are not refusals.

    Args:
        exc: The error raised by a Hub call; its ``__cause__`` is checked too.
        token_present: Whether a Hugging Face token is configured.
        repo_id: The repo that was requested; defaults to the one the error names.
    """
    refusal = _find_refusal(exc)
    if refusal is None:
        return None
    kind, error = refusal
    repo = repo_id or getattr(error, "repo_id", None)
    subject = repo or "This model"
    target = repo or "this model"
    link = f"{_HUB_ROOT}/{repo}" if repo else _HUB_ROOT
    license_step = f"accept its license at {link}"
    if kind == "unauthorized" and token_present:
        return f"Hugging Face rejected your token: {_LOGIN_HINT} again or unset HF_TOKEN."
    if kind == "forbidden" and token_present:
        return f"Your Hugging Face token has no access to {target}: {license_step} with that account."
    if kind == "gated" and token_present:
        return f"{subject} is gated: {license_step} with the Hugging Face account your token belongs to."
    if kind == "not_found" and token_present:
        return f"{subject} was not found, or your Hugging Face token has no access to it: check the name, or {license_step}."
    if kind == "not_found":
        return f"{subject} was not found, or it needs a Hugging Face token: check the name, or {license_step}, then {_LOGIN_HINT}."
    return f"{subject} needs a Hugging Face token: {license_step}, then {_LOGIN_HINT}."


def describe_refusal(exc: BaseException, repo_id: str | None = None) -> str | None:
    """Return the one-line refusal message for *exc*, reading the token only when *exc* is a refusal.

    Args:
        exc: The error raised by a Hub call; its ``__cause__`` is checked too.
        repo_id: The repo that was requested; defaults to the one the error names.

    Returns:
        The line, or ``None`` when *exc* is not a Hub refusal.
    """
    if refusal_message(exc, token_present=False, repo_id=repo_id) is None:
        return None
    return refusal_message(exc, token_present=token_source() is not None, repo_id=repo_id)


def _find_refusal(exc: BaseException) -> tuple[str, BaseException] | None:
    """Return ``(kind, error)`` for the first Hub refusal in *exc* or its cause, or ``None``."""
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    for error in (exc, exc.__cause__):
        if isinstance(error, GatedRepoError):
            return "gated", error
        if isinstance(error, RepositoryNotFoundError):
            return "not_found", error
        if isinstance(error, HfHubHTTPError):
            status = getattr(error.response, "status_code", None)
            if status == 401:
                return "unauthorized", error
            if status == 403:
                return "forbidden", error
    return None
