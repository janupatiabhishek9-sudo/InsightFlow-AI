"""API-key authentication with roles.

API_KEYS="alice:<key>:approver:restricted,bob:<key>:analyst:internal"
Roles are ordered: viewer (read) < analyst (upload, investigate) < approver (+ approve/reject).
The clearance caps which knowledge documents an investigation may retrieve.
With API_KEYS empty, auth is off (local development) and every caller is a local approver.
"""

from __future__ import annotations

import hmac
import logging
from typing import Literal

from fastapi import Depends, Header, HTTPException, Request
from pydantic import BaseModel

from app.config import AccessLevel, Settings

log = logging.getLogger(__name__)
Role = Literal["viewer", "analyst", "approver"]
ROLE_RANK: dict[str, int] = {"viewer": 0, "analyst": 1, "approver": 2}


class Principal(BaseModel):
    name: str
    role: Role
    clearance: AccessLevel
    authenticated: bool = True


def parse_api_keys(raw: str) -> dict[str, Principal]:
    keys: dict[str, Principal] = {}
    for entry in filter(None, (e.strip() for e in raw.split(","))):
        parts = entry.split(":")
        if len(parts) != 4:
            raise ValueError("API_KEYS entries must look like name:key:role:clearance")
        name, key, role, clearance = (p.strip() for p in parts)
        if len(key) < 16:
            raise ValueError(f"API key for '{name}' is too short (minimum 16 characters)")
        keys[key] = Principal(name=name, role=role, clearance=clearance)
    return keys


def _settings(request: Request) -> Settings:
    return request.app.state.service.settings


def _keys(request: Request) -> dict[str, Principal]:
    cached = getattr(request.app.state, "api_keys", None)
    if cached is None:
        raw = _settings(request).api_keys
        cached = parse_api_keys(raw.get_secret_value() if raw else "")
        request.app.state.api_keys = cached
        if not cached:
            log.warning("API_KEYS not set: authentication is disabled (local development mode)")
    return cached


def current_principal(request: Request, x_api_key: str | None = Header(None)) -> Principal:
    keys = _keys(request)
    if not keys:
        return Principal(name="local-dev", role="approver", clearance=_settings(request).default_user_clearance,
                         authenticated=False)
    if x_api_key:
        for key, principal in keys.items():
            if hmac.compare_digest(key, x_api_key):  # constant-time comparison
                return principal
    raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")


def require(role: Role):
    def dependency(principal: Principal = Depends(current_principal)) -> Principal:
        if ROLE_RANK[principal.role] < ROLE_RANK[role]:
            raise HTTPException(status_code=403, detail=f"requires role '{role}' (you are '{principal.role}')")
        return principal

    return dependency
