"""Tool permissions and authorization decisions.

Permissions are granted by configuration and by recorded human approvals - never by the model.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.domain.governance import RiskLevel


class Permission(str, Enum):
    READ_DATA = "READ_DATA"
    RUN_ANALYSIS = "RUN_ANALYSIS"
    READ_KNOWLEDGE = "READ_KNOWLEDGE"
    WRITE_FILE = "WRITE_FILE"
    ACCESS_EXTERNAL_SYSTEM = "ACCESS_EXTERNAL_SYSTEM"
    SEND_EXTERNAL_MESSAGE = "SEND_EXTERNAL_MESSAGE"
    MODIFY_DATA = "MODIFY_DATA"


# What every investigation gets by default.
DEFAULT_GRANTS: frozenset[Permission] = frozenset(
    {Permission.READ_DATA, Permission.RUN_ANALYSIS, Permission.READ_KNOWLEDGE}
)
# Permissions a human reviewer may grant for a single approved step.
APPROVAL_GRANTABLE: frozenset[Permission] = frozenset({Permission.MODIFY_DATA})


class ToolPolicy(BaseModel):
    name: str
    permissions: list[Permission]
    risk_level: RiskLevel
    read_only: bool
    requires_approval: bool = False
    timeout_seconds: float = Field(30, gt=0)


class AuthorizationDecision(BaseModel):
    allowed: bool
    reason: str
    needs_approval: bool = False


def authorize(
    policy: ToolPolicy,
    grants: frozenset[Permission],
    step_approved: bool = False,
) -> AuthorizationDecision:
    """Decide whether a tool call may run.

    A step approved by a human temporarily receives the APPROVAL_GRANTABLE permissions;
    anything outside that set can never be unlocked at runtime.
    """
    effective = set(grants)
    if step_approved:
        effective |= APPROVAL_GRANTABLE
    missing = [p.value for p in policy.permissions if p not in effective]
    if missing:
        grantable = all(Permission(m) in APPROVAL_GRANTABLE for m in missing)
        return AuthorizationDecision(
            allowed=False,
            needs_approval=grantable and not step_approved,
            reason=f"tool '{policy.name}' requires permission(s) {missing}",
        )
    if policy.requires_approval and not step_approved:
        return AuthorizationDecision(
            allowed=False, needs_approval=True, reason=f"tool '{policy.name}' requires human approval"
        )
    return AuthorizationDecision(allowed=True, reason="authorized")
