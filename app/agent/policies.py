"""工具权限策略注册表。

ISSUE-007 在 ToolManager 和真实工具之间增加一层轻量权限判断：按 tool_name、
tenant_id、user_id 和可选 session_id 决定工具是否可用。这里不引入认证系统、
RBAC 管理 API 或业务接入，只提供可独立测试和回滚的内部策略边界。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from app.config import config
from app.core.errors import JsonObject
from app.core.request_context import RequestContext

PolicyReason: TypeAlias = Literal[
    "allowed",
    "policy_disabled",
    "tool_not_allowed",
    "disabled",
    "tenant_denied",
    "user_denied",
    "session_denied",
]


@dataclass(frozen=True)
class PolicyDecision:
    """一次工具权限判断的稳定结果。

    决策对象只暴露稳定 reason_code，不携带允许租户/用户列表。这样 ToolManager 可以写
    trace 方便排障，但未授权 ToolResult 不会泄漏内部策略细节。
    """

    tool_name: str
    allowed: bool
    reason_code: PolicyReason
    timeout_ms: int | None = None
    max_result_chars: int | None = None

    def to_trace_fields(self) -> JsonObject:
        """转换为 trace 字段，便于定位策略允许或拒绝的原因。"""

        return {
            "tool_name": self.tool_name,
            "policy_allowed": self.allowed,
            "policy_reason": self.reason_code,
            "policy_timeout_ms": self.timeout_ms,
            "policy_max_result_chars": self.max_result_chars,
        }


@dataclass(frozen=True)
class ToolPolicy:
    """单个工具的权限和执行覆盖配置。

    `allowed_tenants/allowed_users` 默认只允许当前工程的 anonymous/default 演示身份；
    `allowed_sessions=()` 表示不按 session 限制，避免在没有认证和会话策略前误伤旧 API。
    可用 `("*",)` 表达显式全允许，但默认配置不会这样做，以便未来多租户边界可收紧。
    """

    tool_name: str
    allowed_tenants: tuple[str, ...] = ("default",)
    allowed_users: tuple[str, ...] = ("anonymous",)
    allowed_sessions: tuple[str, ...] = ()
    enabled: bool = True
    timeout_ms: int | None = None
    max_result_chars: int | None = None

    def decide(self, ctx: RequestContext) -> PolicyDecision:
        """根据 RequestContext 判断当前工具是否允许调用。"""

        if not self.enabled:
            return self._deny("disabled")
        if not _matches_scope(self.allowed_tenants, ctx.tenant_id):
            return self._deny("tenant_denied")
        if not _matches_scope(self.allowed_users, ctx.user_id):
            return self._deny("user_denied")
        if self.allowed_sessions and not _matches_scope(
            self.allowed_sessions,
            ctx.session_id or "",
        ):
            return self._deny("session_denied")
        return PolicyDecision(
            tool_name=self.tool_name,
            allowed=True,
            reason_code="allowed",
            timeout_ms=_positive_int_or_none(self.timeout_ms),
            max_result_chars=_positive_int_or_none(self.max_result_chars),
        )

    def _deny(self, reason_code: PolicyReason) -> PolicyDecision:
        return PolicyDecision(
            tool_name=self.tool_name,
            allowed=False,
            reason_code=reason_code,
            timeout_ms=None,
            max_result_chars=None,
        )


class PolicyRegistry:
    """内存策略注册表。

    注册表启用时默认“未列入即拒绝”，防止后续新增工具绕过权限；配置
    `tool_policy_enabled=false` 时直接全允许，用于紧急回滚到 ISSUE-006 的工具行为。
    """

    def __init__(
        self,
        policies: Iterable[ToolPolicy] | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self._policies: dict[str, ToolPolicy] = {}
        for policy in policies or ():
            self.register(policy)

    @classmethod
    def from_config(cls, settings: object | None = None) -> PolicyRegistry:
        """从应用配置构造默认策略。

        默认 allowlist 覆盖当前本地工具和 mock MCP server 工具，确保 ISSUE-007 接入后
        anonymous/default 演示身份仍可使用现有能力；没有列入的新工具会被拒绝，等后续
        issue 显式补策略后再开放。
        """

        source = settings or config
        enabled = bool(getattr(source, "tool_policy_enabled", True))
        allowlist = _string_sequence(getattr(source, "tool_default_allowlist", ()))
        tenants = _string_sequence(getattr(source, "tool_policy_default_allowed_tenants", ()))
        users = _string_sequence(getattr(source, "tool_policy_default_allowed_users", ()))
        policies = [
            ToolPolicy(
                tool_name=tool_name,
                allowed_tenants=tenants or ("default",),
                allowed_users=users or ("anonymous",),
            )
            for tool_name in allowlist
        ]
        return cls(policies, enabled=enabled)

    def register(self, policy: ToolPolicy) -> ToolPolicy:
        """注册或覆盖单个工具策略。"""

        tool_name = policy.tool_name.strip()
        if not tool_name:
            raise ValueError("tool policy name must not be empty")
        self._policies[tool_name] = policy
        return policy

    def check(self, tool_name: str, ctx: RequestContext) -> PolicyDecision:
        """返回工具调用权限决策。"""

        normalized_name = tool_name.strip()
        if not self.enabled:
            return PolicyDecision(
                tool_name=normalized_name,
                allowed=True,
                reason_code="policy_disabled",
            )

        policy = self._policies.get(normalized_name) or self._policies.get("*")
        if policy is None:
            return PolicyDecision(
                tool_name=normalized_name,
                allowed=False,
                reason_code="tool_not_allowed",
            )
        decision = policy.decide(ctx)
        if decision.tool_name == normalized_name:
            return decision
        # 通配策略的决策仍要回填真实 tool_name，trace 和 ToolResult 才能定位到具体工具。
        return PolicyDecision(
            tool_name=normalized_name,
            allowed=decision.allowed,
            reason_code=decision.reason_code,
            timeout_ms=decision.timeout_ms,
            max_result_chars=decision.max_result_chars,
        )

    def is_allowed(self, tool_name: str, ctx: RequestContext) -> bool:
        """简化布尔入口，供后续 adapter 或测试使用。"""

        return self.check(tool_name, ctx).allowed

    def names(self) -> tuple[str, ...]:
        """返回已注册策略名快照。"""

        return tuple(self._policies.keys())


def _matches_scope(allowed_values: Sequence[str], actual_value: str) -> bool:
    """匹配 tenant/user/session 作用域。

    空序列表示不限制该维度；`*` 表示显式全允许。两者都保留，是为了让默认 session
    不限制和未来配置全允许表达不同的治理含义。
    """

    normalized = tuple(value.strip() for value in allowed_values if value.strip())
    if not normalized:
        return True
    return "*" in normalized or actual_value in normalized


def _positive_int_or_none(value: int | None) -> int | None:
    if value is None:
        return None
    return value if value > 0 else None


def _string_sequence(value: object) -> tuple[str, ...]:
    """把配置中的列表/元组规整为非空字符串元组。"""

    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, Iterable):
        items: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                items.append(item.strip())
        return tuple(items)
    return ()
