from __future__ import annotations

import base64
import binascii
import logging
from collections.abc import Callable

from fastapi import Header, HTTPException

logger = logging.getLogger(__name__)


def parse_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


def parse_basic(authorization: str | None) -> tuple[str | None, str | None]:
    """Parse HTTP Basic. Returns (username, password); either may be None on failure."""
    if not authorization:
        return None, None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "basic":
        return None, None
    try:
        decoded = base64.b64decode(parts[1], validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return None, None
    username, sep, password = decoded.partition(":")
    if not sep:
        return None, None
    return username or None, password or None


def _unauthorized(detail: str = "invalid or missing bearer token") -> HTTPException:
    return HTTPException(
        status_code=401,
        detail=detail,
        headers={"WWW-Authenticate": 'Basic realm="orchestro-mesh", Bearer'},
    )


def make_requester_dep(
    *,
    mesh_token: str | None,
    api_tokens: dict[str, str],
) -> Callable[..., str]:
    """Build a FastAPI dependency that authenticates a request and returns the requester name.

    Accepts ``Authorization: Bearer <token>`` or ``Authorization: Basic <b64(user:token)>``.
    The Basic form is intended for browser access (dashboard) — paste the token as the
    password; the username is informational and, when set, takes precedence over the
    ``X-Orchestro-Requester`` header in mesh_token mode.
    """
    if not mesh_token and not api_tokens:
        logger.warning(
            "orchestro-mesh auth is disabled: no mesh_token or api_tokens configured. "
            "Anyone reaching this endpoint can claim any requester identity."
        )

    def dep(
        authorization: str | None = Header(default=None),
        x_orchestro_requester: str | None = Header(default=None),
    ) -> str:
        token = parse_bearer(authorization)
        basic_user, basic_pw = parse_basic(authorization)
        if token is None and basic_pw is not None:
            token = basic_pw

        if api_tokens:
            if token and token in api_tokens:
                return api_tokens[token]
            raise _unauthorized()
        if mesh_token:
            if token == mesh_token:
                return basic_user or x_orchestro_requester or "anonymous"
            raise _unauthorized()
        return basic_user or x_orchestro_requester or "anonymous"

    return dep


def make_shared_token_dep(token: str | None) -> Callable[..., None]:
    """Simple shared-secret bearer guard for worker endpoints."""
    if not token:
        logger.warning("orchestro-mesh worker auth is disabled: no worker_token configured.")

    def dep(authorization: str | None = Header(default=None)) -> None:
        if not token:
            return
        received = parse_bearer(authorization)
        if received is None:
            _, received = parse_basic(authorization)
        if received != token:
            raise _unauthorized()

    return dep
