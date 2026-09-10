"""
通用 Plan-Execute-Replan 框架
基于 LangGraph 官方教程实现
"""

from .state import PlanExecuteState


async def planner(*args: object, **kwargs: object) -> object:
    from .planner import planner as _planner

    return await _planner(*args, **kwargs)


async def executor(*args: object, **kwargs: object) -> object:
    from .executor import executor as _executor

    return await _executor(*args, **kwargs)


async def replanner(*args: object, **kwargs: object) -> object:
    from .replanner import replanner as _replanner

    return await _replanner(*args, **kwargs)


async def critic(*args: object, **kwargs: object) -> object:
    from .critic import critic as _critic

    return await _critic(*args, **kwargs)


__all__ = [
    "PlanExecuteState",
    "planner",
    "executor",
    "replanner",
    "critic",
]
