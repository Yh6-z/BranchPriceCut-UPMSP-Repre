"""按 C# 的内存门槛、10/3 候选和两轮试算选择作业弧。

输入：已完成列—割求解的父节点、其 y 解和目标值；可注入 LP 与启发式试算。
输出：所选作业弧、两侧独立子节点、各阶段得分与更新后的历史候选。
试探目标值只用于排序，不充当子节点的已证明下界。
"""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass, replace
from random import Random
from typing import Callable, Sequence

import cplex

from src.BranchPriceCut.Branch.branching import (
    BranchCandidate,
    BranchKind,
    BranchNode,
    aggregate_job_arcs,
    create_child_node,
    ordinary_branch_candidate,
    pricing_requests_for_node,
)
from src.BranchPriceCut.Price.Bucket.bucket_graph import build_bucket_graph
from src.BranchPriceCut.Price.Bucket.time_bucket import build_time_buckets
from src.BranchPriceCut.Price.rmp_integration import rebuild_rmp_with_columns
from src.heuristic.neighborhood_search import solve_neighborhood_pricing
from src.model.RMP import DualSolution, RestrictedMasterProblem


@dataclass(frozen=True, slots=True)
class TrialLPResult:
    """一次有限列 RMP LP 试算的目标值、列解与对偶。"""

    objective_value: float
    column_values: tuple[float, ...]
    dual_solution: DualSolution


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """某候选两侧相对父节点目标值的乘积得分。"""

    candidate: BranchCandidate
    zero_objective: float
    one_objective: float
    score: float


@dataclass(frozen=True, slots=True)
class StrongBranchingResult:
    """选弧结果、试算证据、历史池和待正式求解的两个子节点。"""

    method: str
    candidate: BranchCandidate | None
    children: tuple[BranchNode, ...]
    phase0_candidates: tuple[BranchCandidate, ...]
    phase1_scores: tuple[CandidateScore, ...]
    phase2_scores: tuple[CandidateScore, ...]
    history: tuple[BranchCandidate, ...]


class _ProcessMemoryCounters(ctypes.Structure):
    """Windows GetProcessMemoryInfo 的内存计数器布局。"""

    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def process_peak_working_set_bytes() -> int:
    """读取本进程峰值常驻内存，供强分支门槛判断。

    输入：无。
    输出：峰值工作集的字节数 ``int``。
    """

    if os.name == "nt":
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        current_process = ctypes.windll.kernel32.GetCurrentProcess
        current_process.restype = ctypes.c_void_p
        memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        memory_info.argtypes = (
            ctypes.c_void_p, ctypes.POINTER(_ProcessMemoryCounters), ctypes.c_uint32
        )
        memory_info.restype = ctypes.c_int
        assert memory_info(current_process(), ctypes.byref(counters), counters.cb)
        return int(counters.PeakWorkingSetSize)
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def solve_trial_rmp(rmp: RestrictedMasterProblem) -> TrialLPResult | None:
    """求解带分支行的有限列 RMP LP。

    输入：``RestrictedMasterProblem``，其变量仍为调度列 y。
    输出：含目标、列解和对偶的 ``TrialLPResult``；不可行时为 ``None``。
    """

    model = cplex.Cplex()
    model.objective.set_sense(model.objective.sense.minimize)
    model.parameters.lpmethod.set(1)
    model.parameters.simplex.tolerances.optimality.set(1e-6)
    model.parameters.emphasis.numerical.set(1)
    model.set_log_stream(None)
    model.set_error_stream(None)
    model.set_warning_stream(None)
    model.set_results_stream(None)
    names = [f"y[{index}]" for index in range(rmp.num_variables)]
    model.variables.add(
        obj=list(rmp.objective_coefficients),
        lb=[rmp.variable_lower_bound] * rmp.num_variables,
        ub=[rmp.variable_upper_bound] * rmp.num_variables,
        names=names,
    )
    rows = (
        *((f"job[{index}]", coefficients, rmp.job_constraint_sense, rhs)
          for index, (coefficients, rhs) in enumerate(zip(rmp.job_cover_matrix, rmp.job_rhs))),
        *((f"machine[{index}]", coefficients, rmp.machine_constraint_sense, rhs)
          for index, (coefficients, rhs) in enumerate(zip(rmp.machine_capacity_matrix, rmp.machine_rhs))),
        *((f"cut[{cut.cut_id}]", cut.coefficients, cut.sense, cut.rhs) for cut in rmp.cut_rows),
    )
    for row_name, coefficients, sense, rhs in rows:
        model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=names, val=[float(v) for v in coefficients])],
            senses=[{">=": "G", "<=": "L", "=": "E"}[sense]],
            rhs=[float(rhs)],
            names=[row_name],
        )
    model.solve()
    if not model.solution.is_primal_feasible():
        model.end()
        return None
    objective = model.solution.get_objective_value()
    values = tuple(model.solution.get_values())
    duals = tuple(model.solution.get_dual_values())
    job_end = rmp.instance.num_jobs
    machine_end = job_end + rmp.instance.num_machines
    result = TrialLPResult(
        objective,
        values,
        DualSolution(
            job_duals=duals[:job_end],
            machine_duals=duals[job_end:machine_end],
            cut_ids=tuple(cut.cut_id for cut in rmp.cut_rows),
            cut_duals=duals[machine_end:],
            rmp_objective_value=objective,
        ),
    )
    model.end()
    return result


def run_heuristic_column_generation(
    node: BranchNode,
    *,
    max_rounds: int = 3,
    num_buckets_per_job: int = 4,
    seed: int = 0,
) -> TrialLPResult | None:
    """在子节点调用邻域启发式并重新求解有限列 RMP。

    输入：``BranchNode``、轮数/时间桶预算和随机种子 ``int``。
    输出：最终试算 ``TrialLPResult``；初始受限 RMP 不可行时为 ``None``。
    """

    rmp = node.rmp
    rng = Random(seed)
    for _ in range(max_rounds):
        solution = solve_trial_rmp(rmp)
        if solution is None:
            return None
        current_keys = frozenset(column.key for column in rmp.columns)
        additions = []
        local_node = replace(node, rmp=rmp)
        requests = pricing_requests_for_node(local_node, solution.dual_solution)
        for local, state in zip(requests, node.bounds):
            buckets = build_time_buckets(
                local, state.completion_time_upper_bounds(),
                num_buckets_per_job=num_buckets_per_job,
            )
            graph = build_bucket_graph(local, buckets)
            result = solve_neighborhood_pricing(
                rmp.instance, local, graph, rng=rng,
                existing_column_keys=current_keys,
            )
            if result.negative_evaluation is not None:
                additions.append(result.negative_evaluation.column)
        if not additions:
            return solution
        rmp = rebuild_rmp_with_columns(rmp, (*rmp.columns, *additions))
    return solve_trial_rmp(rmp)


def _score_candidate(
    candidate: BranchCandidate,
    parent_objective: float,
    zero: TrialLPResult | None,
    one: TrialLPResult | None,
) -> CandidateScore | None:
    """由两侧 LP 目标值计算乘积得分。

    输入：``BranchCandidate``、父目标 ``float``、两侧可选 ``TrialLPResult``。
    输出：两侧可比较时的 ``CandidateScore``，否则为 ``None``。
    """

    if zero is None or one is None:
        return None
    left = max(0.0, zero.objective_value - parent_objective)
    right = max(0.0, one.objective_value - parent_objective)
    return CandidateScore(candidate, zero.objective_value, one.objective_value, left * right)


def select_branch_candidate(
    node: BranchNode,
    column_values: Sequence[float],
    parent_objective: float,
    *,
    incumbent_objective: float | None = None,
    use_strong_branching: bool = True,
    ordinary_kind: BranchKind = "x",
    phase0_size: int = 10,
    phase1_size: int = 3,
    history: Sequence[BranchCandidate] = (),
    peak_working_set_bytes: int | None = None,
    memory_limit_bytes: int = 2 * 1024**3,
    lp_solver: Callable[[RestrictedMasterProblem], TrialLPResult | None] = solve_trial_rmp,
    heuristic_runner: Callable[[BranchNode], TrialLPResult | None] = run_heuristic_column_generation,
) -> StrongBranchingResult:
    """按 C# 内存条件选择普通或两阶段强分支作业弧。

    输入：父 ``BranchNode``、列值 ``Sequence[float]``、目标值 ``float``，
    可选 incumbent、开关、候选预算、历史候选、峰值内存/字节门槛 int 及试算函数。
    输出：含选择方法、阶段得分、历史池和两个子节点的 ``StrongBranchingResult``。
    """

    candidates = aggregate_job_arcs(node.rmp, column_values)
    if incumbent_objective is not None and parent_objective >= incumbent_objective - 1e-9:
        return StrongBranchingResult("pruned_by_bound", None, (), (), (), (), tuple(history))
    if all(value <= 1e-5 or value >= 1.0 - 1e-5 for value in column_values):
        return StrongBranchingResult("integer_y", None, (), (), (), (), tuple(history))
    if not candidates:
        return StrongBranchingResult("no_fractional_arc", None, (), (), (), (), tuple(history))
    strong = use_strong_branching and (
        process_peak_working_set_bytes() if peak_working_set_bytes is None
        else peak_working_set_bytes
    ) < memory_limit_bytes
    if not strong:
        selected = ordinary_branch_candidate(candidates, kind=ordinary_kind)
        children = () if selected is None else (
            create_child_node(node, selected, 0), create_child_node(node, selected, 1)
        )
        return StrongBranchingResult("ordinary", selected, children, (), (), (), tuple(history))

    by_key = {candidate.key: candidate for candidate in candidates}
    current = list(candidates[:phase0_size if not history else phase0_size // 2])
    if history:
        for past in history:
            candidate = by_key.get(past.key)
            if candidate is not None and candidate.key not in {item.key for item in current}:
                current.append(candidate)
            if len(current) >= phase0_size:
                break

    first_scores = []
    for candidate in current:
        zero = lp_solver(create_child_node(node, candidate, 0).rmp)
        one = lp_solver(create_child_node(node, candidate, 1).rmp)
        score = _score_candidate(candidate, parent_objective, zero, one)
        if score is not None:
            first_scores.append(score)
    first_scores.sort(key=lambda item: (-item.score, item.candidate.key))
    shortlisted = first_scores[:phase1_size]
    next_history = tuple(
        {candidate.key: candidate for candidate in (
            *(score.candidate for score in first_scores[phase1_size:]), *history
        )}.values()
    )[:phase0_size]

    second_scores = []
    for first in shortlisted:
        candidate = first.candidate
        zero = heuristic_runner(create_child_node(node, candidate, 0))
        one = heuristic_runner(create_child_node(node, candidate, 1))
        score = _score_candidate(candidate, parent_objective, zero, one)
        if score is not None:
            second_scores.append(score)
    second_scores.sort(key=lambda item: (-item.score, item.candidate.key))
    selected = (
        second_scores[0].candidate if second_scores
        else ordinary_branch_candidate(candidates, kind=ordinary_kind)
    )
    children = () if selected is None else (
        create_child_node(node, selected, 0), create_child_node(node, selected, 1)
    )
    return StrongBranchingResult(
        "strong" if second_scores else "ordinary_fallback",
        selected, children, tuple(current), tuple(first_scores),
        tuple(second_scores), next_history,
    )
