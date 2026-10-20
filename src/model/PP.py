"""按机器分解的定价问题规格和候选列约化成本计算。

适用阶段：RMP 得到对偶解后，为每台机器构造一个独立定价请求。
上游输入：实例、``RMP.DualSolution``、SWPT 顺序和可选割对偶。
下游输出： :class:`PricingProblem` 与 :class:`PricingEvaluation`，供后续
启发式或精确标签算法消费，再把负约化成本列送回 RMP。

本模块不搜索最小约化成本路径，也不能给出“无负列”的最优性证明。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Iterable, Sequence

from src.model.dataload import SchedulingInstance
from src.model.RMP import ColumnCutDefinition, DualSolution, RestrictedMasterProblem
from src.model.schedulemodel import ScheduleColumn, build_schedule_column, swpt_order


@dataclass(frozen=True, slots=True)
class PricingProblem:
    """单台机器定价子问题的只读输入规格。"""

    machine_id: int
    num_eligible_jobs: int
    source_node: int
    sink_node: int
    job_ids: tuple[int, ...]
    dag_vertices: tuple[int, ...]
    weights: tuple[int, ...]
    processing_times: tuple[float, ...]
    swpt_job_order: tuple[int, ...]
    dag_arcs: tuple[tuple[int, int], ...]
    job_duals: tuple[float, ...]
    machine_dual: float
    cut_ids: tuple[str, ...]    # [lm-src(C_1), lm-src(C_2), ...]
    cut_duals: tuple[float, ...]
    cut_definitions: tuple[ColumnCutDefinition, ...]
    reduced_cost_tolerance: float


@dataclass(frozen=True, slots=True)
class PricingEvaluation:
    """一个给定候选调度列的约化成本计算结果。"""

    column: ScheduleColumn
    cut_coefficients: tuple[float, ...]
    reduced_cost: float
    reduced_cost_tolerance: float

    @property
    def is_improving(self) -> bool:
        """是否满足进入 RMP 的负约化成本容差。"""

        return self.reduced_cost < -self.reduced_cost_tolerance


def build_pricing_problems(
    rmp: RestrictedMasterProblem,
    dual_solution: DualSolution,
    *,
    reduced_cost_tolerance: float = 1e-6,
) -> tuple[PricingProblem, ...]:
    """把一个 RMP 对偶解拆成恰好 ``m`` 个相互独立的机器定价请求。"""

    instance = rmp.instance
    problems: list[PricingProblem] = []
    for machine_id in instance.machine_ids:
        order = swpt_order(instance, machine_id)
        num_eligible_jobs = len(order)
        source_node = 0
        sink_node = num_eligible_jobs + 1   # 尾节点
        dag_vertices = (source_node, *order, sink_node)
        problems.append(
            PricingProblem(
                machine_id=machine_id,
                num_eligible_jobs=num_eligible_jobs,
                source_node=source_node,
                sink_node=sink_node,
                job_ids=instance.job_ids,
                dag_vertices=dag_vertices,
                weights=instance.weights,
                processing_times=tuple(
                    instance.processing_time(job_id, machine_id)
                    for job_id in instance.job_ids
                ),
                swpt_job_order=order,
                dag_arcs=build_full_swpt_dag_arcs(instance, order),
                
                job_duals=tuple(dual_solution.job_duals),
                machine_dual=dual_solution.machine_duals[machine_id - 1],
                cut_ids=tuple(dual_solution.cut_ids),
                cut_duals=tuple(dual_solution.cut_duals),
                cut_definitions=tuple(rmp.cut_definitions),
                reduced_cost_tolerance=reduced_cost_tolerance,
            )
        )
    return tuple(problems)


def build_full_swpt_dag_arcs(
    instance: SchedulingInstance,
    order: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    """构造节点 ``0`` 到节点 ``N_k + 1`` 的完整 SWPT 前向 DAG。"""

    num_eligible_jobs = len(order)
    vertices = (0, *order, num_eligible_jobs + 1)
    return tuple(
        (tail, head)
        for tail_position, tail in enumerate(vertices[:-1])
        for head in vertices[tail_position + 1 :]
    )


def evaluate_column_reduced_cost(
    problem: PricingProblem,
    column: ScheduleColumn,
    *,
    cut_coefficients: Iterable[float] | None = None,
) -> PricingEvaluation:
    r"""按 $c-a\pi-\mu-g\gamma$ 计算一个给定候选列的约化成本。"""

    if cut_coefficients is None:
        definitions = {cut.cut_id: cut for cut in problem.cut_definitions}
        coefficients = tuple(
            definitions[cut_id].coefficient(column) if cut_id in definitions else 0.0
            for cut_id in problem.cut_ids
        )
    else:
        coefficients = tuple(cut_coefficients)

    job_dual_contribution = fsum(
        coefficient * dual
        for coefficient, dual in zip(column.coverage, problem.job_duals)
    )
    cut_dual_contribution = fsum(
        coefficient * dual
        for coefficient, dual in zip(coefficients, problem.cut_duals)
    )
    reduced_cost = (
        column.cost
        - job_dual_contribution
        - problem.machine_dual
        - cut_dual_contribution
    )
    return PricingEvaluation(
        column=column,
        cut_coefficients=coefficients,
        reduced_cost=reduced_cost,
        reduced_cost_tolerance=problem.reduced_cost_tolerance,
    )


def evaluate_schedule(
    instance: SchedulingInstance,
    problem: PricingProblem,
    job_ids: Sequence[int],
    *,
    cut_coefficients: Iterable[float] | None = None,
) -> PricingEvaluation:
    """把指定作业序列转为合法列并计算约化成本，但不进行路径搜索。"""

    column = build_schedule_column(instance, problem.machine_id, job_ids)
    return evaluate_column_reduced_cost(
        problem,
        column,
        cut_coefficients=cut_coefficients,
    )
