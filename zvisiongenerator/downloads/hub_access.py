"""Hugging Face access: where the token comes from, saving one, and one actionable line for a Hub refusal."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping

TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
SAVED_SOURCE = "saved"
_HUB_ROOT = "https://huggingface.co"
_TOKEN_STEPS = "add a token on the Web UI's Config page or run `uvx hf auth login`"
_REJECTED_TOKEN = "Hugging Face doesn't accept this token."
_NOT_A_TOKEN = "That doesn't look like a Hugging Face token. Copy it again from Hugging Face."
_ORGANISATION_TOKEN = "Use a token from your personal account, not an organization's."
_UNCHECKED = "Couldn't check the token with Hugging Face. Try again in a minute."


def token_source(*, read_token: Callable[[], str | None] | None = None, environ: Mapping[str, str] | None = None) -> str | None:
    """Return where the Hugging Face token comes from, or ``None`` when there is none.

    Args:
        read_token: Returns the token like ``huggingface_hub.get_token``; defaults to it.
        environ: The environment to match the token against; defaults to ``os.environ``.

    Returns:
        The environment variable holding the token, ``"saved"`` for a token saved on this computer, or ``None``.
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
    return SAVED_SOURCE


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
        return "Hugging Face rejected your token: replace it on the Web UI's Config page or run `uvx hf auth login` again, or unset HF_TOKEN."
    if kind == "forbidden" and token_present:
        return f"Your Hugging Face token has no access to {target}: {license_step} with that account."
    if kind == "gated" and token_present:
        return f"{subject} is gated: {license_step} with the Hugging Face account your token belongs to."
    if kind == "not_found" and token_present:
        return f"{subject} was not found, or your Hugging Face token has no access to it: check the name, or {license_step}."
    if kind == "not_found":
        return f"{subject} was not found, or it needs a Hugging Face token: check the name, or {license_step}, then {_TOKEN_STEPS}."
    return f"{subject} needs a Hugging Face token: {license_step}, then {_TOKEN_STEPS}."


def save_token(token: str, *, login: Callable[..., object] | None = None) -> None:
    """Check *token* with Hugging Face and save it where every Hub call reads it.

    Args:
        token: The pasted token; surrounding whitespace is dropped.
        login: Validates and saves like ``huggingface_hub.login``; defaults to it.

    Raises:
        ValueError: The paste is empty or not a token, the token belongs to an organisation, or Hugging Face rejects it.
        RuntimeError: The token could not be checked or saved. Messages are fixed text and never contain the token.
    """
    token = token.strip()
    if not token:
        raise ValueError("Paste a token.")
    if not token.isascii() or not token.isprintable() or any(char.isspace() for char in token):
        raise ValueError(_NOT_A_TOKEN)
    if token.startswith("api_org"):
        raise ValueError(_ORGANISATION_TOKEN)
    from huggingface_hub.errors import HfHubHTTPError

    if login is None:
        from huggingface_hub import login
    try:
        login(token=token, add_to_git_credential=False)
    except HfHubHTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        raise (ValueError(_REJECTED_TOKEN) if status == 401 else RuntimeError(_UNCHECKED)) from None
    except Exception:  # noqa: BLE001 - transport errors can quote the token, so only fixed text leaves
        raise RuntimeError(_UNCHECKED) from None


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
