"""
通用 Plan-Execute-Replan 服务
基于 LangGraph 官方教程实现
"""

import asyncio
from typing import Any, AsyncGenerator
from loguru import logger

from app.config import config
from app.core.errors import AgentMaxStepExceededError, AppError, LLMTimeoutError
from app.core.request_context import RequestContext, get_request_context_or_none
from app.observability.tracing import TraceLogger


# 节点名称常量
NODE_PLANNER = "planner"
NODE_EXECUTOR = "executor"
NODE_REPLANNER = "replanner"
NODE_CRITIC = "critic"


class AIOpsService:
    """通用 Plan-Execute-Replan 服务"""

    def __init__(self):
        """初始化服务"""
        from app.memory.checkpointer_factory import create_checkpointer

        # 与 RAG 服务共享同一 Redis checkpointer 单例：会话状态跨服务一致。
        self.checkpointer = create_checkpointer()
        self.graph = self._build_graph()
        self.trace_logger = TraceLogger(
            trace_jsonl_path=config.trace_jsonl_path,
            enabled=config.trace_enabled,
        )
        logger.info("Plan-Execute-Replan Service 初始化完成")

    def _build_graph(self):
        """构建 Plan-Execute-Replan 工作流"""
        logger.info("构建工作流图...")

        # 创建状态图
        from langgraph.graph import END, StateGraph

        from app.agent.aiops import PlanExecuteState, critic, executor, planner, replanner

        workflow = StateGraph(PlanExecuteState)

        # 添加节点
        workflow.add_node(NODE_PLANNER, planner)      # 制定计划
        workflow.add_node(NODE_EXECUTOR, executor)  # 执行步骤
        workflow.add_node(NODE_REPLANNER, replanner)  # 重新规划
        workflow.add_node(NODE_CRITIC, critic)  # 答案级自我批判（ISSUE-B）

        # 设置入口点
        workflow.set_entry_point(NODE_PLANNER)

        # 定义边
        workflow.add_edge(NODE_PLANNER, NODE_EXECUTOR)     # planner -> executor
        workflow.add_edge(NODE_EXECUTOR, NODE_REPLANNER)   # executor -> replanner
        # critic 修订后直接结束：节点内部完成有界修订，不回 executor/replanner，
        # 图上没有新增环，agent_max_steps 与 recursion_limit 语义不变。
        workflow.add_edge(NODE_CRITIC, END)

        # replanner 的条件边（路由函数为模块级 _should_continue，便于单测）
        workflow.add_conditional_edges(
            NODE_REPLANNER,
            _should_continue,
            {
                NODE_EXECUTOR: NODE_EXECUTOR,
                NODE_CRITIC: NODE_CRITIC,
                END: END
            }
        )

        # 编译工作流
        compiled_graph = workflow.compile(checkpointer=self.checkpointer)

        logger.info("工作流图构建完成")
        return compiled_graph

    async def execute(
        self,
        user_input: str,
        session_id: str = "default",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        执行 Plan-Execute-Replan 流程

        Args:
            user_input: 用户的任务描述
            session_id: 会话ID

        Yields:
            Dict[str, Any]: 流式事件
        """
        logger.info(f"[会话 {session_id}] 开始执行任务: {user_input}")

        try:
            # 初始化状态
            initial_state: dict[str, Any] = {
                "input": user_input,
                "plan": [],
                "past_steps": [],
                "response": "",
                # ISSUE-A 证据链通道：executor 每步累加工具证据摘要，供 Critic 核对。
                "tool_evidence": [],
                # ISSUE-B：响应尚未经过 Critic 审查；critic 节点完成后置 True。
                "critic_reviewed": False,
            }

            # 流式执行工作流
            # recursion_limit 是 LangGraph 自身的节点递归保护，不等同于业务步骤数。
            # ISSUE-009 要求两个上限同时存在：replanner 继续控制业务步骤，graph config
            # 负责防止异常环路无限调度节点；配置放在顶层以符合 LangGraph 调用约定。
            recursion_limit = int(getattr(config, "agent_recursion_limit", 30))
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                },
                "recursion_limit": recursion_limit,
            }
            self._record_agent_event(
                "agent.start",
                session_id=session_id,
                recursion_limit=recursion_limit,
                max_steps=int(getattr(config, "agent_max_steps", 8)),
            )
            self._record_agent_event(
                "agent.recursion_limit",
                session_id=session_id,
                recursion_limit=recursion_limit,
            )

            request_ctx = _context_with_session(session_id)
            async for event in _iterate_with_request_timeout(
                self.graph.astream(
                    input=initial_state,
                    config=config_dict,
                    stream_mode="updates",
                ),
                request_ctx,
            ):
                # 解析事件
                for node_name, node_output in event.items():
                    logger.info(f"节点 '{node_name}' 输出事件")

                    # 根据节点类型生成不同的事件
                    if node_name == NODE_PLANNER:
                        formatted_event = self._format_planner_event(node_output)

                    elif node_name == NODE_EXECUTOR:
                        formatted_event = self._format_executor_event(node_output)

                    elif node_name == NODE_REPLANNER:
                        formatted_event = self._format_replanner_event(node_output)

                    elif node_name == NODE_CRITIC:
                        formatted_event = self._format_critic_event(node_output)

                    else:
                        continue

                    yield formatted_event
                    if formatted_event.get("type") == "error":
                        return

            # 获取最终状态
            final_state = self.graph.get_state(config_dict)
            final_response = ""

            # 安全地获取响应（处理 values 可能为 None 的情况）
            if final_state and final_state.values:
                final_response = final_state.values.get("response", "")

            # 发送完成事件
            yield {
                "type": "complete",
                "stage": "complete",
                "message": "任务执行完成",
                "response": final_response
            }
            self._record_agent_event(
                "agent.end",
                session_id=session_id,
                recursion_limit=recursion_limit,
            )

            logger.info(f"[会话 {session_id}] 任务执行完成")

        except Exception as e:
            app_error = _map_agent_exception(e)
            self._record_agent_error(app_error, session_id=session_id)
            logger.error(
                f"[会话 {session_id}] 任务执行失败: {app_error.code}",
                exc_info=True,
            )
            # 这里不返回 str(e)：LangGraph recursion_limit、LLM 或工具异常可能包含
            # 内部类名、URL 或参数。服务层直接输出稳定 error envelope，API 层可继续
            # 追加 trace/request 并保持 SSE `data.type=error` 兼容。
            yield app_error.to_sse_payload(**_trace_kwargs(get_request_context_or_none()))

    async def diagnose(
        self,
        session_id: str = "default"
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        AIOps 诊断接口（兼容旧接口）

        Args:
            session_id: 会话ID

        Yields:
            Dict[str, Any]: 诊断过程的流式事件
        """
        # 使用固定的 AIOps 任务描述
        from textwrap import dedent
        aiops_task = dedent("""诊断当前系统是否存在告警，如果存在告警请详细分析告警原因并生成诊断报告，诊断报告输出格式要求：
                ```
                # 告警分析报告

                ---

                ## 📋 活跃告警清单

                | 告警名称 | 级别 | 目标服务 | 首次触发时间 | 最新触发时间 | 状态 |
                |---------|------|----------|-------------|-------------|------|
                | [告警1名称] | [级别] | [服务名] | [时间] | [时间] | 活跃 |
                | [告警2名称] | [级别] | [服务名] | [时间] | [时间] | 活跃 |

                ---

                ## 🔍 告警根因分析1 - [告警名称]

                ### 告警详情
                - **告警级别**: [级别]
                - **受影响服务**: [服务名]
                - **持续时间**: [X分钟]

                ### 症状描述
                [根据监控指标描述症状]

                ### 日志证据
                [引用查询到的关键日志]

                ### 根因结论
                [基于证据得出的根本原因]

                ---

                ## 🛠️ 处理方案执行1 - [告警名称]

                ### 已执行的排查步骤
                1. [步骤1]
                2. [步骤2]

                ### 处理建议
                [给出具体的处理建议]

                ### 预期效果
                [说明预期的效果]

                ---

                ## 🔍 告警根因分析2 - [告警名称]
                [如果有第2个告警，重复上述格式]

                ---

                ## 📊 结论

                ### 整体评估
                [总结所有告警的整体情况]

                ### 关键发现
                - [发现1]
                - [发现2]

                ### 后续建议
                1. [建议1]
                2. [建议2]

                ### 风险评估
                [评估当前风险等级和影响范围]
                ```

                **重要提醒**：
                - 最终输出必须是纯 Markdown 文本，不要包含 JSON 结构
                - 所有内容必须基于工具查询的真实数据，严禁编造
                - 如果某个步骤失败，在结论中如实说明，不要跳过""")

        async for event in self.execute(aiops_task, session_id):
            # 转换事件格式以兼容旧的 API
            if event.get("type") == "complete":
                # 将 response 包装为 diagnosis 格式
                yield {
                    "type": "complete",
                    "stage": "diagnosis_complete",
                    "message": "诊断流程完成",
                    "diagnosis": {
                        "status": "completed",
                        "report": event.get("response", "")
                    }
                }
            else:
                yield event

    def _format_planner_event(self, state: dict | None) -> dict:
        """格式化 Planner 节点事件"""
        if not state:
            return {
                "type": "status",
                "stage": "planner",
                "message": "规划节点执行中"
            }

        plan = state.get("plan", [])

        return {
            "type": "plan",
            "stage": "plan_created",
            "message": f"执行计划已制定，共 {len(plan)} 个步骤",
            "plan": plan
        }

    def _format_executor_event(self, state: dict | None) -> dict:
        """格式化 Executor 节点事件"""
        if not state:
            return {
                "type": "status",
                "stage": "executor",
                "message": "执行节点运行中"
            }

        plan = state.get("plan", [])
        past_steps = state.get("past_steps", [])

        if past_steps:
            last_step, _ = past_steps[-1]
            return {
                "type": "step_complete",
                "stage": "step_executed",
                "message": f"步骤执行完成 ({len(past_steps)}/{len(past_steps) + len(plan)})",
                "current_step": last_step,
                "remaining_steps": len(plan)
            }
        else:
            return {
                "type": "status",
                "stage": "executor",
                "message": "开始执行步骤"
            }

    def _format_replanner_event(self, state: dict | None) -> dict:
        """格式化 Replanner 节点事件"""
        if not state:
            return {
                "type": "status",
                "stage": "replanner",
                "message": "评估节点运行中"
            }

        error_event = state.get("error_event")
        if isinstance(error_event, dict):
            return dict(error_event)

        response = state.get("response", "")
        plan = state.get("plan", [])

        if response:
            # 已生成最终响应
            return {
                "type": "report",
                "stage": "final_report",
                "message": "最终报告已生成",
                "report": response
            }
        else:
            # 重新规划
            return {
                "type": "status",
                "stage": "replanner",
                "message": f"评估完成，{'继续执行剩余步骤' if plan else '准备生成最终响应'}",
                "remaining_steps": len(plan)
            }

    def _format_critic_event(self, state: dict | None) -> dict:
        """格式化 Critic 节点事件（ISSUE-B）。

        修订发生时节点更新携带 response，推送一条 report 更新事件——SSE 流中
        replanner 已先发过旧版 report，客户端以最后一条 report/complete 为准；
        未修订时只发 status，不重复推送报告内容，保持旧客户端事件形态兼容。
        """

        if not state:
            return {
                "type": "status",
                "stage": "critic",
                "message": "答案审查中"
            }

        revised = state.get("response")
        if isinstance(revised, str) and revised:
            return {
                "type": "report",
                "stage": "critic_revised",
                "message": "答案已按证据链修订",
                "report": revised
            }
        return {
            "type": "status",
            "stage": "critic",
            "message": "答案审查完成"
        }

    def _record_agent_event(
        self,
        name: str,
        *,
        session_id: str,
        recursion_limit: int,
        max_steps: int | None = None,
    ) -> None:
        """记录 Agent 执行边界事件。

        AIOpsService 也可能在单元测试中通过 `__new__` 构造，此时没有 trace_logger；
        helper 做宽松判断，保证测试和脚本不会因为观测组件缺失而影响业务事件。
        """

        ctx = _context_with_session(session_id)
        trace_logger = getattr(self, "trace_logger", None)
        if ctx is None or trace_logger is None:
            return
        trace_logger.record_event(
            name,
            ctx,
            session_id=session_id,
            recursion_limit=recursion_limit,
            max_steps=max_steps,
        )

    def _record_agent_error(self, error: AppError, *, session_id: str) -> None:
        """记录 Agent 错误，不把原始异常文本写入对外事件。"""

        ctx = _context_with_session(session_id)
        trace_logger = getattr(self, "trace_logger", None)
        if ctx is None or trace_logger is None:
            return
        trace_logger.record_event(
            "agent.error",
            ctx,
            status="error",
            error_code=error.code,
            session_id=session_id,
        )


# 全局单例
class _LazyAIOpsService:
    """Delay LangGraph construction until the AIOps service is actually used."""

    def __init__(self) -> None:
        self._instance: AIOpsService | None = None

    def _get(self) -> AIOpsService:
        if self._instance is None:
            self._instance = AIOpsService()
        return self._instance

    def execute(
        self,
        user_input: str,
        session_id: str = "default",
    ) -> AsyncGenerator[dict[str, Any], None]:
        return self._get().execute(user_input, session_id=session_id)

    def diagnose(self, session_id: str = "default") -> AsyncGenerator[dict[str, Any], None]:
        return self._get().diagnose(session_id=session_id)


# Keep the historical singleton name while avoiding import-time LangGraph construction.
aiops_service = _LazyAIOpsService()


def _should_continue(state: "dict[str, Any]") -> str:
    """replanner 之后的路由决策（ISSUE-B 提为模块级函数便于单测）。

    优先级：error_event > response 路由（critic 审查或 END）> plan 继续 > END。
    response 已生成且未审查、且 critic_enabled 开启时进入 critic；开关关闭或已
    审查完成时直接 END，与接入 critic 前的旧图行为完全一致。
    """

    from langgraph.graph import END

    if state.get("error_event"):
        logger.info("检测到标准化错误事件，结束流程")
        return END

    # 如果已经生成了最终响应：先判断是否需要 Critic 答案审查
    if state.get("response"):
        if not state.get("critic_reviewed") and _critic_enabled():
            logger.info("响应已生成，进入 Critic 答案审查")
            return NODE_CRITIC
        logger.info("已生成最终响应，结束流程")
        return END

    # 如果还有计划步骤，继续执行
    plan = state.get("plan", [])
    if plan:
        logger.info(f"继续执行，剩余 {len(plan)} 个步骤")
        return NODE_EXECUTOR

    # 计划为空但没有响应，返回 replanner 生成响应
    logger.info("计划执行完毕，生成最终响应")
    return END


def _critic_enabled() -> bool:
    """运行时读取 Critic 开关：配置回滚不需要重建图，测试可 monkeypatch。"""

    return bool(getattr(config, "critic_enabled", False))


def _map_agent_exception(exc: Exception) -> AppError:
    """把 Agent/LangGraph 异常映射为稳定 AppError。

    LangGraph recursion_limit 的异常类型在不同版本中可能不完全一致，因此这里用
    类名和安全的内部消息特征做兼容判断；对外仍只返回 `AGENT_MAX_STEP_EXCEEDED`
    的固定文案，不泄漏原始异常全文。
    """

    if isinstance(exc, AppError):
        return exc
    if isinstance(exc, TimeoutError):
        return LLMTimeoutError()
    marker = f"{exc.__class__.__name__}: {exc}".lower()
    if "recursion" in marker and "limit" in marker:
        return AgentMaxStepExceededError()
    return AppError.from_exception(exc, origin_module="app.services.aiops_service")


def _context_with_session(session_id: str) -> RequestContext | None:
    ctx = get_request_context_or_none()
    if ctx is None:
        return None
    if ctx.session_id == session_id:
        return ctx
    return ctx.with_session(session_id)


def _trace_kwargs(ctx: RequestContext | None) -> dict[str, str | None]:
    if ctx is None:
        return {"trace_id": None, "request_id": None}
    return {"trace_id": ctx.trace_id, "request_id": ctx.request_id}


async def _iterate_with_request_timeout(
    async_iterable: object,
    ctx: RequestContext | None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Pull graph updates under the remaining request deadline and close on timeout."""

    iterator = async_iterable.__aiter__()  # type: ignore[attr-defined]
    try:
        while True:
            try:
                yield await asyncio.wait_for(
                    iterator.__anext__(),
                    timeout=_request_timeout_seconds(ctx),
                )
            except StopAsyncIteration:
                return
            except TimeoutError as exc:
                raise LLMTimeoutError() from exc
    finally:
        await _close_async_iterator(iterator)


def _request_timeout_seconds(ctx: RequestContext | None) -> float:
    timeout_ms = (
        ctx.remaining_ms()
        if ctx is not None
        else int(getattr(config, "request_timeout_ms", 60_000))
    )
    return max(timeout_ms, 0) / 1000


async def _close_async_iterator(iterator: object) -> None:
    close = getattr(iterator, "aclose", None)
    if callable(close):
        await close()
