"""Access control for the hosted app: project ON/OFF switch, coupon codes, AI switch, admin audit log.

Defaults come from configuration (on Streamlit Cloud: the app's Secrets page), so they survive
restarts. Changes made in the admin page are saved to ADMIN_STATE_PATH (atomic writes, one lock);
free hosts wipe their disk on restart, after which the configured defaults apply again.

COUPONS format:  CODE[:max_uses[:YYYY-MM-DD]], comma-separated, e.g. "PYTHON2026:100:2026-12-31,DEMO"
"""

from __future__ import annotations

import hmac
import os
import re
import threading
from datetime import date, datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import Settings

CODE_PATTERN = re.compile(r"^[A-Z0-9_-]{3,40}$")
MAX_AUDIT_ENTRIES = 200


class Coupon(BaseModel):
    code: str
    max_uses: int | None = Field(None, ge=1, description="None = unlimited")
    expires: date | None = None
    active: bool = True
    uses: int = 0
    investigations: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def problem(self, today: date | None = None) -> str | None:
        """Why this coupon cannot be used right now, or None if it can."""
        today = today or date.today()
        if not self.active:
            return "This code has been deactivated."
        if self.expires is not None and today > self.expires:
            return "This code has expired."
        if self.max_uses is not None and self.uses >= self.max_uses:
            return "This code has reached its usage limit."
        return None


class AuditEntry(BaseModel):
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event: str


class AccessState(BaseModel):
    project_enabled: bool = True
    ai_enabled: bool = True
    groq_key_override: str | None = None  # stored only on the server's disk; never shown in full
    coupons: dict[str, Coupon] = Field(default_factory=dict)
    audit: list[AuditEntry] = Field(default_factory=list)


def normalise_code(code: str) -> str:
    return (code or "").strip().upper()


def parse_coupon_seed(raw: str) -> dict[str, Coupon]:
    coupons: dict[str, Coupon] = {}
    for entry in filter(None, (e.strip() for e in (raw or "").split(","))):
        parts = [p.strip() for p in entry.split(":")]
        code = normalise_code(parts[0])
        if not CODE_PATTERN.match(code):
            raise ValueError(f"invalid coupon code '{parts[0]}' (3-40 letters, digits, - or _)")
        max_uses = int(parts[1]) if len(parts) > 1 and parts[1] else None
        expires = date.fromisoformat(parts[2]) if len(parts) > 2 and parts[2] else None
        coupons[code] = Coupon(code=code, max_uses=max_uses, expires=expires)
    return coupons


class AccessStore:
    """Thread-safe store shared by every browser session of the app process."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.path = settings.resolve(Path(settings.admin_state_path))
        self._lock = threading.RLock()
        self._mtime: float | None = None
        self._state = self._load()
        self._mtime = self._file_mtime()

    def _file_mtime(self) -> float | None:
        return self.path.stat().st_mtime if self.path.exists() else None

    def _sync(self) -> None:
        """Pick up changes written by another process (e.g. the UI and the API sharing one file)."""
        mtime = self._file_mtime()
        if mtime != self._mtime:
            self._state = self._load()
            self._mtime = mtime

    # ---- persistence ------------------------------------------------------------------------
    def _defaults(self) -> AccessState:
        return AccessState(project_enabled=self.settings.project_enabled,
                           coupons=parse_coupon_seed(self.settings.coupons))

    def _load(self) -> AccessState:
        state = self._defaults()
        if self.path.exists():
            try:
                saved = AccessState.model_validate_json(self.path.read_text(encoding="utf-8"))
            except ValueError:
                return state  # corrupt file: fall back to configured defaults (fail safe, not open)
            # Coupons from configuration are always present; saved ones add to / update them.
            saved.coupons = {**state.coupons, **saved.coupons}
            return saved
        return state

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(self._state.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
        self._mtime = self._file_mtime()

    def _log(self, event: str) -> None:
        self._state.audit.append(AuditEntry(event=event))
        self._state.audit = self._state.audit[-MAX_AUDIT_ENTRIES:]

    # ---- admin authentication ---------------------------------------------------------------
    @property
    def admin_configured(self) -> bool:
        return bool(self.settings.admin_password and self.settings.admin_password.get_secret_value())

    def check_admin_password(self, password: str) -> bool:
        if not self.admin_configured:
            return False
        ok = hmac.compare_digest(self.settings.admin_password.get_secret_value().encode(), (password or "").encode())
        with self._lock:
            self._log("admin login" if ok else "failed admin login")
            self._save()
        return ok

    # ---- project switch ---------------------------------------------------------------------
    @property
    def project_enabled(self) -> bool:
        with self._lock:
            self._sync()
            return self._state.project_enabled

    def set_project_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._state.project_enabled = enabled
            self._log(f"project switched {'ON' if enabled else 'OFF'}")
            self._save()

    @property
    def coupon_required(self) -> bool:
        return self.settings.require_coupon

    # ---- coupons ----------------------------------------------------------------------------
    def coupons(self) -> list[Coupon]:
        with self._lock:
            self._sync()
            return [c.model_copy() for c in sorted(self._state.coupons.values(), key=lambda c: c.code)]

    def check(self, code: str) -> str | None:
        """Problem with a code for a session that already unlocked with it (no usage counted)."""
        with self._lock:
            self._sync()
            if not self._state.project_enabled:
                return "InsightFlow AI is switched off by the administrator."
            coupon = self._state.coupons.get(normalise_code(code))
            if coupon is None:
                return "Unknown code."
            if not coupon.active:
                return "This code has been deactivated."
            if coupon.expires is not None and date.today() > coupon.expires:
                return "This code has expired."
            return None

    def redeem(self, code: str) -> str | None:
        """Unlock a session with a code: returns an error message, or None on success (counts one use)."""
        code = normalise_code(code)
        with self._lock:
            self._sync()
            if not self._state.project_enabled:
                return "InsightFlow AI is switched off by the administrator."
            coupon = self._state.coupons.get(code)
            if coupon is None:
                self._log("rejected unknown code")
                self._save()
                return "Unknown code."
            problem = coupon.problem()
            if problem:
                self._log(f"rejected code {code}: {problem}")
                self._save()
                return problem
            coupon.uses += 1
            self._log(f"code {code} redeemed ({coupon.uses}{'/' + str(coupon.max_uses) if coupon.max_uses else ''})")
            self._save()
            return None

    def record_investigation(self, code: str | None) -> None:
        with self._lock:
            coupon = self._state.coupons.get(normalise_code(code or ""))
            if coupon is not None:
                coupon.investigations += 1
                self._save()

    def add_coupon(self, code: str, max_uses: int | None = None, expires: date | None = None) -> Coupon:
        code = normalise_code(code)
        if not CODE_PATTERN.match(code):
            raise ValueError("Codes need 3-40 characters: letters, digits, - or _.")
        with self._lock:
            if code in self._state.coupons:
                raise ValueError(f"Code {code} already exists.")
            coupon = Coupon(code=code, max_uses=max_uses, expires=expires)
            self._state.coupons[code] = coupon
            self._log(f"code {code} created")
            self._save()
            return coupon.model_copy()

    def set_coupon_active(self, code: str, active: bool) -> None:
        with self._lock:
            self._state.coupons[normalise_code(code)].active = active
            self._log(f"code {normalise_code(code)} {'activated' if active else 'deactivated'}")
            self._save()

    def delete_coupon(self, code: str) -> None:
        with self._lock:
            self._state.coupons.pop(normalise_code(code), None)
            self._log(f"code {normalise_code(code)} deleted")
            self._save()

    # ---- AI (LLM) switch --------------------------------------------------------------------
    @property
    def ai_enabled(self) -> bool:
        return self._state.ai_enabled

    @property
    def groq_key_override(self) -> str | None:
        return self._state.groq_key_override

    def set_ai(self, enabled: bool, groq_key: str | None = None, clear_key: bool = False) -> None:
        with self._lock:
            self._state.ai_enabled = enabled
            if clear_key:
                self._state.groq_key_override = None
            elif groq_key:
                self._state.groq_key_override = groq_key.strip()
            key_note = "; Groq key replaced" if groq_key else ("; Groq key override removed" if clear_key else "")
            self._log(f"AI switched {'ON' if enabled else 'OFF'}{key_note}")
            self._save()

    def audit(self, limit: int = 50) -> list[AuditEntry]:
        with self._lock:
            return list(reversed(self._state.audit[-limit:]))


def mask(secret: str | None) -> str:
    if not secret:
        return "not set"
    return secret[:4] + "…" + secret[-4:] if len(secret) > 12 else "set"
