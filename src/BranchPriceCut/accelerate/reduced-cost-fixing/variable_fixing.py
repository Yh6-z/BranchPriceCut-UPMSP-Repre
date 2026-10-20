"""用经过证明的逐弧约化成本界固定节点局部弧为零。

适用阶段：节点列—割生成收敛、真实对偶与整数 incumbent 均已获得后。
输入：实例、单机 ``PricingProblem``、DSUB、已证实的 LB/UB、固定参数及列池。
输出：删弧证明、更新后的 DAG/DSUB 和过滤后的列，供桶图重建与后续节点求解。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Sequence

from src.BranchPriceCut.Cut.lm_subset_row_cut import LimitedMemorySubsetRowCut
from src.BranchPriceCut.Price.dynamic_upper_bound import (
    DynamicUpperBoundState,
    DynamicUpperBoundUpdate,
    delete_dag_arcs_and_update_bounds,
)
from src.BranchPriceCut.Price.forward_labeling import collect_forward_labels
from src.model.PP import PricingProblem
from src.model.dataload import SchedulingInstance
from src.model.schedulemodel import ScheduleColumn

from .backward_labeling import collect_backward_labels
from .label_joining import ArcAssessment, assess_arcs


@dataclass(frozen=True, slots=True)
class ReducedCostFixingResult:
    """固定状态、逐弧证明、局部图更新与保留的列。"""

    status: str
    gap: float
    assessments: tuple[ArcAssessment, ...]
    graph_update: DynamicUpperBoundUpdate | None
    retained_columns: tuple[ScheduleColumn, ...]
    removed_columns: tuple[ScheduleColumn, ...]

    @property
    def removed_arcs(self) -> tuple[tuple[int, int], ...]:
        """读取本次实际删除的机器弧。

        输入：无。
        输出：``tuple[tuple[int, int], ...]``，元素为尾、头节点编号。
        """

        return () if self.graph_update is None else self.graph_update.removed_arcs


def fix_arcs_by_reduced_cost(
    instance: SchedulingInstance,
    problem: PricingProblem,
    state: DynamicUpperBoundState,
    *,
    lower_bound: float,
    upper_bound: float,
    full_master_dual_certified: bool,
    relative_gap_limit: float,
    fixing_tolerance: float,
    columns: Sequence[ScheduleColumn] = (),
    use_completion_pruning: bool = True,
) -> ReducedCostFixingResult:
    """用精确双向标签判定并执行单机节点局部删弧。

    输入：实例、当前单机定价请求和 DSUB；已证实的节点 ``LB``、整数
    ``UB``、全机器精确定价证书、来自 ``config.py`` 的间隙阈值与容差，
    可选节点列池及完成界剪枝开关。
    输出：``ReducedCostFixingResult``，含安全删弧、DSUB 更新和列过滤。
    """

    original_columns = tuple(columns)
    gap = upper_bound - lower_bound
    skipped = ReducedCostFixingResult(
        status="skipped_no_certificate",
        gap=gap,
        assessments=(),
        graph_update=None,
        retained_columns=original_columns,
        removed_columns=(),
    )
    if (
        not full_master_dual_certified
        or not isfinite(lower_bound)
        or not isfinite(upper_bound)
        or gap < 0.0
        or fixing_tolerance < 0.0
    ):
        return skipped
    if gap / max(1.0, abs(upper_bound)) >= relative_gap_limit:
        return replace(skipped, status="skipped_gap")
    if (
        state.machine_id != problem.machine_id
        or problem.sink_node != instance.num_jobs + 1
        or not frozenset(problem.dag_arcs).issuperset(state.dag_arcs)
    ):
        return replace(skipped, status="skipped_snapshot")

    definitions = {cut.cut_id: cut for cut in problem.cut_definitions}
    if (
        len(problem.cut_ids) != len(problem.cut_duals)
        or len(set(problem.cut_ids)) != len(problem.cut_ids)
        or any(cut_id not in definitions for cut_id in problem.cut_ids)
        or any(
            not isinstance(definitions[cut_id], LimitedMemorySubsetRowCut)
            or definitions[cut_id].sense != "<="
            for cut_id in problem.cut_ids
        )
        or any(dual > 0.0 for dual in problem.cut_duals)
    ):
        return replace(skipped, status="skipped_unsupported_cuts")

    local_problem = replace(problem, dag_arcs=state.dag_arcs)
    completion_bounds = state.completion_time_upper_bounds()
    completion_bounds[problem.sink_node] = state.global_upper_bound
    forward = collect_forward_labels(
        instance,
        local_problem,
        completion_upper_bounds=completion_bounds,
        dominance_tolerance=0.0,
    )
    sink_labels = forward.labels_at(problem.sink_node)
    if (
        not forward.is_exact
        or not sink_labels
        or min(label.reduced_cost for label in sink_labels)
        < -problem.reduced_cost_tolerance
    ):
        return replace(skipped, status="skipped_uncertified_pricing")

    threshold = gap + fixing_tolerance
    backward = collect_backward_labels(
        local_problem,
        state,
        forward=forward if use_completion_pruning else None,
        threshold=threshold if use_completion_pruning else None,
    )
    assessments = assess_arcs(
        instance,
        local_problem,
        state,
        forward,
        backward,
        threshold=threshold,
    )
    to_remove = tuple(
        assessment.arc for assessment in assessments if assessment.can_delete
    )
    update = delete_dag_arcs_and_update_bounds(local_problem, state, to_remove)
    removed = frozenset(update.removed_arcs)
    retained_columns = tuple(
        column
        for column in original_columns
        if column.machine_id != problem.machine_id
        or removed.isdisjoint(column.arcs)
    )
    removed_columns = tuple(
        column
        for column in original_columns
        if column.machine_id == problem.machine_id
        and not removed.isdisjoint(column.arcs)
    )
    return ReducedCostFixingResult(
        status="applied",
        gap=gap,
        assessments=assessments,
        graph_update=update,
        retained_columns=retained_columns,
        removed_columns=removed_columns,
    )
