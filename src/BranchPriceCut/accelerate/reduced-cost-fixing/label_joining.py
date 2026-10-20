"""前后向标签按 DAG 弧拼接，证明弧是否可能进入当前上界内的调度。

适用阶段：同一局部图上前向和后向标签生成之后。
输入：实例、单机定价请求、DSUB、两侧标签及 $UB-LB$ 阈值。
输出：逐弧排除证明和路径证据，供 ``variable_fixing.py`` 删弧。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf

from src.BranchPriceCut.Price.dynamic_upper_bound import DynamicUpperBoundState
from src.BranchPriceCut.Price.forward_labeling import ForwardLabelSearch
from src.model.PP import PricingProblem, evaluate_schedule
from src.model.dataload import SchedulingInstance

from .backward_labeling import BackwardLabelSearch


@dataclass(frozen=True, slots=True)
class ArcAssessment:
    """一条弧的乐观界、可保留路径和阈值排除证明。"""

    arc: tuple[int, int]
    optimistic_lower_bound: float | None
    qualifying_job_ids: tuple[int, ...] | None
    can_delete: bool
    compared_pairs: int


def assess_arcs(
    instance: SchedulingInstance,
    problem: PricingProblem,
    state: DynamicUpperBoundState,
    forward: ForwardLabelSearch,
    backward: BackwardLabelSearch,
    *,
    threshold: float,
    tolerance: float = 0.0,
) -> tuple[ArcAssessment, ...]:
    """以两侧标签求每条现存弧的安全阈值排除结论。

    输入：``SchedulingInstance``、当前 ``PricingProblem``/DSUB、精确前向和
    完整或安全剪枝的后向标签、约化成本阈值与容差 ``float``。
    输出：逐弧 ``ArcAssessment`` 元组；无数值界时 ``None`` 表示仅有阈值证明。
    """

    assessments: list[ArcAssessment] = []
    if not forward.is_exact or not backward.is_exact_for_threshold:
        return tuple(
            ArcAssessment(
                arc=arc,
                optimistic_lower_bound=None,
                qualifying_job_ids=None,
                can_delete=False,
                compared_pairs=0,
            )
            for arc in state.dag_arcs
        )
    bounds_are_complete = backward.pruned_labels == 0
    local_arcs = frozenset(state.dag_arcs)
    for arc in state.dag_arcs:
        tail, head = arc
        prefixes = forward.labels_at(tail)
        suffixes = backward.labels_at(head)
        if not prefixes or not suffixes:
            assessments.append(
                ArcAssessment(
                    arc=arc,
                    optimistic_lower_bound=inf if bounds_are_complete else None,
                    qualifying_job_ids=None,
                    can_delete=True,
                    compared_pairs=0,
                )
            )
            continue

        aggregate_bound = (
            min(label.reduced_cost for label in prefixes)
            + min(label.optimistic_base for label in suffixes)
            + min(label.time for label in prefixes)
            * min(label.cumulative_weight for label in suffixes)
        )
        if aggregate_bound > threshold + tolerance:
            assessments.append(
                ArcAssessment(
                    arc=arc,
                    optimistic_lower_bound=(
                        aggregate_bound if bounds_are_complete else None
                    ),
                    qualifying_job_ids=None,
                    can_delete=True,
                    compared_pairs=0,
                )
            )
            continue

        minimum_bound = inf
        qualifying_jobs: tuple[int, ...] | None = None
        compared_pairs = 0
        for prefix in prefixes:
            for suffix in suffixes:
                if prefix.time > suffix.latest_prefix_time + tolerance:
                    continue
                optimistic = (
                    prefix.reduced_cost
                    + suffix.optimistic_base
                    + prefix.time * suffix.cumulative_weight
                )
                minimum_bound = min(minimum_bound, optimistic)
                if optimistic > threshold + tolerance:
                    continue
                compared_pairs += 1
                jobs = (*prefix.job_ids, *suffix.job_ids)
                evaluation = evaluate_schedule(instance, problem, jobs)
                if (
                    arc in evaluation.column.arcs
                    and local_arcs.issuperset(evaluation.column.arcs)
                    and all(
                        completion <= state.bound(job_id).upper_bound + tolerance
                        for job_id, completion in zip(
                            evaluation.column.job_ids,
                            evaluation.column.completion_times,
                        )
                    )
                    and evaluation.reduced_cost <= threshold + tolerance
                ):
                    qualifying_jobs = jobs
                    break
            if qualifying_jobs is not None:
                break

        assessments.append(
            ArcAssessment(
                arc=arc,
                optimistic_lower_bound=minimum_bound if bounds_are_complete else None,
                qualifying_job_ids=qualifying_jobs,
                can_delete=qualifying_jobs is None,
                compared_pairs=compared_pairs,
            )
        )
    return tuple(assessments)
