"""动态收缩完工时间上界（DSUB）及其 DAG 删弧插件。

适用阶段：基础或节点局部 SWPT DAG 建立后，以及分支/变量固定删除弧后。
按照项目规则，DAG 弧删除与由删除触发的 DSUB 传播统一在本脚本中实现。

Input:
    ``PricingProblem``；全局机器完工时间上界；可选已有 ``DynamicUpperBoundState``；
    本轮需要从局部 DAG 删除的弧集合。删除弧来自后续分支或变量固定模块。
Output:
    不可变 ``DynamicUpperBoundUpdate``，其中包含更新后的局部 DAG、逐节点 DSUB、
    决定性前驱、不可达节点、实际删除弧和上界变化节点；逐作业上界供时间桶重构，
    局部 DAG 供 Bucket Graph 与标签算法使用。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import fsum
from typing import Iterable, Protocol

from src.model.PP import PricingProblem
from src.model.dataload import SchedulingInstance


@dataclass(frozen=True, slots=True)
class VertexCompletionBound:
    """一个 DAG 节点的 DSUB、决定性前驱及可达状态。"""

    vertex: int
    upper_bound: float
    max_predecessor: int | None
    reachable: bool


@dataclass(frozen=True, slots=True)
class DynamicUpperBoundState:
    """单台机器局部 DAG 与当前 DSUB 的不可变快照。"""

    machine_id: int
    global_upper_bound: float
    dag_arcs: tuple[tuple[int, int], ...]
    bounds: tuple[VertexCompletionBound, ...]

    def bound(self, vertex: int) -> VertexCompletionBound:
        return next(bound for bound in self.bounds if bound.vertex == vertex)

    def completion_time_upper_bounds(self) -> dict[int, float]:
        """输出作业节点上界，直接作为 ``time_bucket.py`` 的输入。"""

        return {
            bound.vertex: bound.upper_bound
            for bound in self.bounds
            if 0 < bound.vertex < len(self.bounds) - 1
        }


@dataclass(frozen=True, slots=True)
class DynamicUpperBoundUpdate:
    """一次 DAG 删弧与 DSUB 传播的结果。"""

    state: DynamicUpperBoundState
    removed_arcs: tuple[tuple[int, int], ...]
    changed_vertices: tuple[int, ...]
    unreachable_vertices: tuple[int, ...]

    @property
    def upper_bounds_updated(self) -> bool:
        return bool(self.changed_vertices)


class DynamicUpperBoundUpdater(Protocol):
    """可替换的 DSUB 与 DAG 删弧联合插件接口。"""

    def initialize(
        self,
        problem: PricingProblem,
        global_upper_bound: float,
    ) -> DynamicUpperBoundState:
        """输入基础 DAG 与全局上界，输出初始 DSUB 状态。"""

    def delete_arcs(
        self,
        problem: PricingProblem,
        state: DynamicUpperBoundState,
        arcs_to_remove: Iterable[tuple[int, int]],
    ) -> DynamicUpperBoundUpdate:
        """输入旧状态和删弧集合，输出新局部 DAG 与传播后的 DSUB。"""


@dataclass(frozen=True, slots=True)
class RecurrenceDynamicUpperBoundUpdater:
    """实现论文递推式的默认 DSUB 插件。"""

    tolerance: float = 1e-9

    def initialize(
        self,
        problem: PricingProblem,
        global_upper_bound: float,
    ) -> DynamicUpperBoundState:
        return DynamicUpperBoundState(
            machine_id=problem.machine_id,
            global_upper_bound=global_upper_bound,
            dag_arcs=problem.dag_arcs,
            bounds=self._recalculate(
                problem,
                problem.dag_arcs,
                global_upper_bound,
            ),
        )

    def delete_arcs(
        self,
        problem: PricingProblem,
        state: DynamicUpperBoundState,
        arcs_to_remove: Iterable[tuple[int, int]],
    ) -> DynamicUpperBoundUpdate:
        requested = frozenset(arcs_to_remove)
        removed_arcs = tuple(arc for arc in state.dag_arcs if arc in requested)
        remaining_arcs = tuple(arc for arc in state.dag_arcs if arc not in requested)
        new_bounds = self._recalculate(
            problem,
            remaining_arcs,
            state.global_upper_bound,
        )
        old_by_vertex = {bound.vertex: bound for bound in state.bounds}
        changed_vertices = tuple(
            bound.vertex
            for bound in new_bounds
            if (
                old_by_vertex[bound.vertex].reachable != bound.reachable
                or bound.upper_bound
                < old_by_vertex[bound.vertex].upper_bound - self.tolerance
            )
        )
        new_state = replace(
            state,
            dag_arcs=remaining_arcs,
            bounds=new_bounds,
        )
        return DynamicUpperBoundUpdate(
            state=new_state,
            removed_arcs=removed_arcs,
            changed_vertices=changed_vertices,
            unreachable_vertices=tuple(
                bound.vertex for bound in new_bounds if not bound.reachable
            ),
        )

    def _recalculate(
        self,
        problem: PricingProblem,
        dag_arcs: tuple[tuple[int, int], ...],
        global_upper_bound: float,
    ) -> tuple[VertexCompletionBound, ...]:
        topological_position = {
            vertex: position
            for position, vertex in enumerate(problem.dag_vertices)
        }
        calculated = {
            problem.source_node: VertexCompletionBound(
                vertex=problem.source_node,
                upper_bound=0.0,
                max_predecessor=None,
                reachable=True,
            )
        }
        for vertex in problem.dag_vertices[1:]:
            predecessors = tuple(
                tail
                for tail, head in dag_arcs
                if head == vertex and calculated[tail].reachable
            )
            if not predecessors:
                processing_time = (
                    0.0
                    if vertex == problem.sink_node
                    else problem.processing_times[vertex - 1]
                )
                calculated[vertex] = VertexCompletionBound(
                    vertex=vertex,
                    upper_bound=processing_time,
                    max_predecessor=None,
                    reachable=False,
                )
                continue
            max_predecessor = max(
                predecessors,
                key=lambda predecessor: (
                    calculated[predecessor].upper_bound,
                    topological_position[predecessor],
                ),
            )
            processing_time = (
                0.0
                if vertex == problem.sink_node
                else problem.processing_times[vertex - 1]
            )
            calculated[vertex] = VertexCompletionBound(
                vertex=vertex,
                upper_bound=min(
                    global_upper_bound,
                    calculated[max_predecessor].upper_bound + processing_time,
                ),
                max_predecessor=max_predecessor,
                reachable=True,
            )
        return tuple(calculated[vertex] for vertex in problem.dag_vertices)


def calculate_global_completion_upper_bound(
    instance: SchedulingInstance,
) -> float:
    """按论文公式输出跨机器通用的完工时间上界 ``ub*``。"""

    largest_processing_time_by_job = tuple(
        max(instance.processing_times[job_id - 1])
        for job_id in instance.job_ids
    )
    return (
        fsum(largest_processing_time_by_job) / instance.num_machines
        + max(largest_processing_time_by_job)
    )


def initialize_dynamic_upper_bounds(
    problem: PricingProblem,
    global_upper_bound: float,
    *,
    updater: DynamicUpperBoundUpdater | None = None,
) -> DynamicUpperBoundState:
    """使用给定插件初始化 DSUB；默认使用论文递推式。"""

    selected = RecurrenceDynamicUpperBoundUpdater() if updater is None else updater
    return selected.initialize(problem, global_upper_bound)


def delete_dag_arcs_and_update_bounds(
    problem: PricingProblem,
    state: DynamicUpperBoundState,
    arcs_to_remove: Iterable[tuple[int, int]],
    *,
    updater: DynamicUpperBoundUpdater | None = None,
) -> DynamicUpperBoundUpdate:
    """在同一插件调用中删除局部 DAG 弧并传播更新 DSUB。"""

    selected = RecurrenceDynamicUpperBoundUpdater() if updater is None else updater
    return selected.delete_arcs(problem, state, arcs_to_remove)
