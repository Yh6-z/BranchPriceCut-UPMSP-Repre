"""UPMSP 的 Branch-Price-Cut 实例入口。

输入：JD Excel 或内存 SchedulingInstance，以及 config.py 的统一配置。
流程：初始列 -> 启发式/精确列生成 <-> lm-SRC -> 根节点固定 -> IP -> 作业弧分支。
输出：经核验的整数调度、证书支持的全局界和完整终止原因；按规模保存 JSON。
受限 RMP 不可行时遵循作者流程停止该节点，同时保留其未解决的证明责任。
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from math import comb, fsum
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Sequence

from config import get_config
from src.BranchPriceCut.Branch.branching import (
    BranchCandidate, BranchNode, SolveInformation, frontier_lower_bound,
    make_branch_node, node_pruning_reason, pricing_requests_for_node,
    record_incumbent, record_node_stage, register_node_information,
    validate_integer_schedule,
)
from src.BranchPriceCut.Cut.cut_pool import update_cut_pool
from src.BranchPriceCut.Cut.lm_subset_row_cut import LimitedMemorySubsetRowCut
from src.BranchPriceCut.Cut.separation import separate_lm_subset_row_cuts
from src.BranchPriceCut.Price.Bucket.bucket_forward_labeling import solve_bucket_forward_labeling_with_bounds
from src.BranchPriceCut.Price.Bucket.bucket_graph import build_bucket_graph
from src.BranchPriceCut.Price.Bucket.time_bucket import build_time_buckets
from src.BranchPriceCut.Price.dynamic_upper_bound import (
    DynamicUpperBoundState, calculate_global_completion_upper_bound, initialize_dynamic_upper_bounds,
)
from src.BranchPriceCut.Price.forward_labeling import ExactPricingResult, solve_forward_labeling
from src.BranchPriceCut.Price.rmp_integration import rebuild_rmp_with_columns, rebuild_rmp_with_cut_pool
from src.BranchPriceCut.accelerate.RootNodeColumnInitial import initialize_root_columns
from src.BranchPriceCut.accelerate.enumeration import solve_schedule_enumeration
from src.BranchPriceCut.accelerate.strong_branching import (
    TrialLPResult, select_branch_candidate,
)
from src.BranchPriceCut.solve import RMPSolveResult, solve_restricted_master
from src.heuristic.neighborhood_search import solve_neighborhood_pricing
from src.model.PP import PricingProblem, build_pricing_problems
from src.model.RMP import DualSolution, RestrictedMasterProblem, build_restricted_master
from src.model.dataload import SchedulingInstance, build_instance_path, load_instance
from src.model.schedulemodel import ScheduleColumn, build_schedule_column


fix_arcs_by_reduced_cost = importlib.import_module(
    "src.BranchPriceCut.accelerate.reduced-cost-fixing.variable_fixing"
).fix_arcs_by_reduced_cost


@dataclass(frozen=True, slots=True)
class BPCResult:
    """实例结果；optimality_proven 表示达到配置的绝对或相对证明间隙。"""

    status: str
    lower_bound: float
    upper_bound: float | None
    absolute_gap: float | None
    relative_gap: float | None
    optimality_proven: bool
    elapsed_seconds: float
    information: SolveInformation
    statistics: dict[str, int]
    diagnostics: tuple[dict, ...]
    unresolved_node_ids: tuple[int, ...]
    result_path: Path | None = None
    source_path: Path | None = None


@dataclass(frozen=True, slots=True)
class NodeRelaxation:
    """节点快照、LP 解、定价证书及割扫描完整性；扫描限额不影响已有 LP 下界。"""

    node: BranchNode
    solution: RMPSolveResult | None
    pricing_lower_bound: float | None
    status: str
    cut_separation_complete: bool = False


def instance_scale(num_jobs: int, configuration: argparse.Namespace) -> str:
    """将实例映射到保存目录。

    输入：作业数 int 和包含两级规模阈值的 argparse.Namespace。
    输出：low、medium 或 high 字符串。
    """

    if num_jobs <= configuration.low_max_jobs:
        return "low"
    return "medium" if num_jobs <= configuration.medium_max_jobs else "high"


def discover_instances(configuration: argparse.Namespace) -> tuple[argparse.Namespace, ...]:
    """发现当前数据精度下的全部 JD 实例，供批量验收使用。

    输入：argparse.Namespace；scale 筛选规模，data_dir 指定数据根目录。
    输出：按作业数、机器数和实例号排序的独立配置元组。
    """

    directory = build_instance_path(configuration).parent
    configurations = []
    for path in directory.glob("*_TWCT_*.xlsx"):
        machine, jobs, _, identity = path.stem.split("_")
        current = argparse.Namespace(**vars(configuration))
        current.num_machine, current.num_job, current.instance_id = int(machine), int(jobs), int(identity)
        if current.scale == "all" or instance_scale(current.num_job, current) == current.scale:
            configurations.append(current)
    return tuple(sorted(configurations, key=lambda item: (item.num_job, item.num_machine, item.instance_id)))


class BPCSolver:
    """只负责跨模块状态传递、搜索和证书管理；具体算法保留在原模块中。"""

    def __init__(self, instance: SchedulingInstance, configuration: argparse.Namespace):
        """建立单实例求解上下文。

        输入：SchedulingInstance 和 config.get_config 产生的 argparse.Namespace。
        输出：初始化的 BPCSolver；此时不调用求解器。
        """

        self.instance = instance
        self.config = configuration
        self.information = SolveInformation(instance)
        self.started = perf_counter()
        self.deadline = None if configuration.time_limit is None else self.started + configuration.time_limit
        self.rng = Random(configuration.seed)
        self.history: tuple[BranchCandidate, ...] = ()
        self.diagnostics: list[dict] = []
        self.statistics = dict.fromkeys((
            "rmp_solves", "heuristic_calls", "exact_calls", "bucket_exact_calls",
            "generated_labels", "added_columns", "added_cuts", "fixing_calls",
            "removed_arcs", "enumeration_calls", "branch_calls", "strong_branches",
            "explored_nodes",
        ), 0)

    def _remaining(self) -> float | None:
        """读取剩余时间预算。

        输入：当前 BPCSolver。
        输出：剩余秒数 float；未设置预算时为 None。
        """

        return None if self.config.time_limit is None else max(
            0.0, self.config.time_limit - (perf_counter() - self.started)
        )

    def _timed_out(self) -> bool:
        """检查阶段边界是否达到总时间预算。

        输入：当前 BPCSolver。输出：bool。
        """

        return self._remaining() == 0.0

    def _solve_rmp(self, rmp: RestrictedMasterProblem) -> RMPSolveResult:
        """把统一配置中的 CPLEX 参数传入主线及强分支试算。

        输入：RestrictedMasterProblem。输出：RMPSolveResult，受实例剩余预算约束。
        """

        return solve_restricted_master(
            rmp, log_output=self.config.solver_log, time_limit=self._remaining(),
            threads=self.config.solver_threads, lp_method=self.config.rmp_lp_method,
            optimality_tolerance=self.config.solver_optimality_tolerance,
            numerical_emphasis=self.config.solver_numerical_emphasis,
        )

    def _record(self, identity: int, node: BranchNode, stage: str, **kwargs) -> None:
        """记录当前快照的阶段事件。

        输入：节点号 int、BranchNode、阶段 str 及 record_node_stage 的关键字参数。
        输出：None；更新 SolveInformation，不改变旧快照。
        """

        self.information = record_node_stage(self.information, identity, node, stage, **kwargs)
        if self.config.progress:
            record = self.information.nodes[identity]
            solution = kwargs.get("solution")
            print(json.dumps(dict(
                event="progress", instance=f"{self.instance.num_machines}_{self.instance.num_jobs}_TWCT_{self.instance.instance_id}",
                node_id=identity, stage=stage, status=kwargs.get("status"),
                columns=len(node.rmp.columns), cuts=len(node.rmp.cut_rows),
                rmp_objective=None if solution is None else solution.objective_value,
                node_LB=record.lower_bound, UB=self.information.upper_bound,
                seconds=round(perf_counter() - self.started, 3),
            ), ensure_ascii=False), flush=True)

    def _configure_bounds(self, node: BranchNode) -> BranchNode:
        """关闭 DSUB 时放宽逐作业时间界，同时保留局部 DAG。

        输入：BranchNode。输出：配置生效后的独立 BranchNode。
        """

        if self.config.use_DSUB:
            return node
        states = tuple(replace(state, bounds=tuple(
            replace(bound, upper_bound=0.0 if bound.vertex == problem.source_node else state.global_upper_bound)
            for bound in state.bounds
        )) for problem, state in zip(node.pricing_problems, node.bounds))
        return replace(node, bounds=states)

    def _initial_node(self) -> BranchNode:
        """构造根 RMP、局部 DAG/DSUB，登记初始整数调度。

        输入：当前 BPCSolver。输出：根 BranchNode；空列启动由 solve 显式停止。
        """

        initial = initialize_root_columns(self.instance, enabled=self.config.use_root_node_column_initial)
        rmp = build_restricted_master(self.instance, initial.columns)
        dual = DualSolution((0.0,) * self.instance.num_jobs, (0.0,) * self.instance.num_machines)
        problems = build_pricing_problems(rmp, dual, reduced_cost_tolerance=self.config.pricing_tolerance)
        horizon = calculate_global_completion_upper_bound(self.instance) if self.config.use_DSUB else max(
            fsum(problem.processing_times) for problem in problems
        )
        states = tuple(initialize_dynamic_upper_bounds(problem, horizon) for problem in problems)
        node = self._configure_bounds(make_branch_node(rmp, problems, states))
        self.information = register_node_information(self.information, node, 0)
        # 每个作业至少经历自身加工时间，得到无需列生成证书的原问题下界。
        baseline = fsum(self.instance.weight(job) * min(self.instance.processing_times[job - 1])
                        for job in self.instance.job_ids)
        self.information = replace(self.information, nodes=(replace(self.information.nodes[0], lower_bound=baseline),))
        if initial.columns:
            self.information = record_incumbent(self.information, initial.columns,
                                                elapsed_seconds=perf_counter() - self.started)
        self._record(0, node, "initialization", status="ready" if initial.columns else "empty_root_pool")
        return node

    def _exact_price(self, problem: PricingProblem, state: DynamicUpperBoundState,
                     keys: frozenset[tuple[int, tuple[int, ...]]]) -> ExactPricingResult:
        """复用已有标签定价，在同一次搜索的汇点标签中最多选入配置数量的新列。

        输入：PricingProblem、DynamicUpperBoundState 和旧列键 frozenset。
        输出：ExactPricingResult；不新增定价、支配或搜索技术。
        """

        self.statistics["exact_calls"] += 1
        if self.config.use_time_bucket and self.config.use_bucket_graph:
            self.statistics["bucket_exact_calls"] += 1
            exact = solve_bucket_forward_labeling_with_bounds(
                self.instance, problem, state, num_buckets_per_job=self.config.num_time_buckets,
                use_bucket_dominance=self.config.use_bucket_dominance, deadline=self.deadline,
                max_columns=self.config.max_added_columns, existing_column_keys=keys,
            )
        else:
            limits = state.completion_time_upper_bounds()
            limits[problem.sink_node] = state.global_upper_bound
            exact = solve_forward_labeling(self.instance, problem, completion_upper_bounds=limits,
                deadline=self.deadline, max_columns=self.config.max_added_columns, existing_column_keys=keys)
        self.statistics["generated_labels"] += exact.generated_labels
        return exact

    def _price(self, identity: int, node: BranchNode, solution: RMPSolveResult,
               *, use_heuristic: bool = True) -> tuple[tuple[ScheduleColumn, ...], bool, float]:
        """在同一 LP 对偶下逐机执行启发式和精确定价。

        输入：节点号 int、BranchNode、最优 RMPSolveResult 和启发式阶段开关 bool。
        输出：新列元组、全机精确无负列 bool、包含定价残差修正的 float 下界。
        """

        requests = pricing_requests_for_node(node, solution.dual_solution)
        keys = frozenset(column.key for column in node.rmp.columns)
        additions: dict[tuple, ScheduleColumn] = {}
        certificates = []
        correction = 0.0
        bucket_mode = self.config.use_time_bucket and self.config.use_bucket_graph
        for problem, state in zip(requests, node.bounds):
            if self._timed_out():
                return tuple(additions.values()), False, 0.0
            problem = replace(problem, reduced_cost_tolerance=self.config.pricing_tolerance)
            limits = state.completion_time_upper_bounds()
            limits[problem.sink_node] = state.global_upper_bound
            candidates = ()
            if self.config.use_neighborhood_search and use_heuristic:
                self.statistics["heuristic_calls"] += 1
                heuristic_problem = replace(problem, reduced_cost_tolerance=self.config.heuristic_pricing_tolerance)
                if bucket_mode:
                    partition = build_time_buckets(problem, limits, num_buckets_per_job=self.config.num_time_buckets)
                    graph = build_bucket_graph(problem, partition)
                    heuristic = solve_neighborhood_pricing(
                        self.instance, heuristic_problem, graph, rng=self.rng,
                        max_iterations=self.config.heuristic_iterations,
                        num_neighbors=self.config.heuristic_neighbors, existing_column_keys=keys,
                    )
                    candidates = heuristic.negative_evaluations
                else:
                    limited = solve_forward_labeling(self.instance, heuristic_problem, max_labels_per_node=20,
                        completion_upper_bounds=limits, deadline=self.deadline,
                        max_columns=self.config.max_added_columns, existing_column_keys=keys)
                    candidates = limited.negative_evaluations
                if candidates:
                    additions.update((item.column.key, item.column) for item in candidates)
                    continue
            if self._timed_out():
                return tuple(additions.values()), False, 0.0
            exact = self._exact_price(problem, state, keys)
            # C# 的 0.01 是常规入列阈值，保留原有严格精确定价以产生证明。
            strict_check = (exact.is_exact and not exact.negative_evaluations
                and exact.best_evaluation is not None
                and exact.best_evaluation.reduced_cost < -self.config.certification_tolerance)
            if strict_check and not self._timed_out():
                problem = replace(problem, reduced_cost_tolerance=self.config.certification_tolerance)
                exact = self._exact_price(problem, state, keys)
            evaluation = exact.best_evaluation
            reduced_cost = None if evaluation is None else evaluation.reduced_cost
            certificates.append(exact.is_exact and (reduced_cost is None
                                or reduced_cost >= -self.config.certification_tolerance))
            correction += min(0.0, reduced_cost if reduced_cost is not None else 0.0)
            self.diagnostics.append(dict(node_id=identity, stage="exact_pricing", machine_id=problem.machine_id,
                                         pricing_round=self.statistics["rmp_solves"], column_count=len(node.rmp.columns),
                                         cut_ids=problem.cut_ids, status=exact.status, is_exact=exact.is_exact,
                                         reduced_cost=reduced_cost, strict_check=strict_check,
                                         added_columns=len(exact.negative_evaluations)))
            additions.update((item.column.key, item.column) for item in exact.negative_evaluations)
        certified = len(certificates) == self.instance.num_machines and all(certificates) and not additions
        return tuple(additions.values()), certified, solution.objective_value + correction

    def _converge(self, identity: int, node: BranchNode) -> NodeRelaxation:
        """循环求解 RMP、加列和受 C# 参数限额控制的三作业割分离。

        输入：节点号 int 和 BranchNode。
        输出：NodeRelaxation；converged 表示定价已证明，割是否全扫描单独记录。
        """

        cut_rounds = 0
        last_objective = None
        same_objective = 0
        while not self._timed_out():
            started = perf_counter()
            solution = self._solve_rmp(node.rmp)
            self.statistics["rmp_solves"] += 1
            self._record(identity, node, "rmp", solution=solution, status=solution.status,
                         elapsed_seconds=perf_counter() - started)
            if solution.status != "optimal":
                return NodeRelaxation(node, solution, None, "restricted_rmp_" + solution.status)
            started = perf_counter()
            same_objective = same_objective + 1 if solution.objective_value == last_objective else 0
            last_objective = solution.objective_value
            additions, certified, lower = self._price(identity, node, solution,
                use_heuristic=same_objective <= self.config.heuristic_max_same_objective)
            if additions:
                node = replace(node, rmp=rebuild_rmp_with_columns(node.rmp, (*node.rmp.columns, *additions)))
                self.statistics["added_columns"] += len(additions)
                self._record(identity, node, "column_generation", status="columns_added",
                             elapsed_seconds=perf_counter() - started)
                continue
            if not certified:
                status = "time_limit" if self._timed_out() else "pricing_not_certified"
                self._record(identity, node, "pricing", status=status, elapsed_seconds=perf_counter() - started)
                return NodeRelaxation(node, solution, None, status)
            self._record(identity, node, "pricing", solution=solution, exact_pricing_complete=True,
                         pricing_lower_bound=lower, status="exact_converged", elapsed_seconds=perf_counter() - started)
            if self._timed_out():
                return NodeRelaxation(node, solution, lower, "time_limit")
            started = perf_counter()
            if self.config.use_cuts and cut_rounds > self.config.max_cut_rounds:
                self._record(identity, node, "cut_separation", status="cut_round_limit")
                return NodeRelaxation(node, solution, lower, "converged", False)
            full_scan = self.config.max_cut_candidates >= comb(self.instance.num_jobs, 3)
            candidates = separate_lm_subset_row_cuts(
                node.rmp, solution.column_values, max_candidates=self.config.max_cut_candidates,
                max_added=self.config.max_added_cuts, violation_tolerance=self.config.cut_violation_tolerance,
            ) if self.config.use_cuts else ()
            if not candidates:
                status = ("full_scan_no_new_cut" if full_scan else "candidate_limit_no_new_cut") if self.config.use_cuts else "disabled"
                self._record(identity, node, "cut_separation", status=status,
                             elapsed_seconds=perf_counter() - started)
                return NodeRelaxation(node, solution, lower, "converged", full_scan and self.config.use_cuts)
            pool = tuple(cut for cut in node.rmp.cut_definitions if isinstance(cut, LimitedMemorySubsetRowCut))
            update = update_cut_pool(pool, candidates)
            node = replace(node, rmp=rebuild_rmp_with_cut_pool(node.rmp, update.cuts))
            self.statistics["added_cuts"] += len(update.added_cut_ids)
            cut_rounds += 1
            self._record(identity, node, "cut_separation", status="cuts_added", elapsed_seconds=perf_counter() - started)
        return NodeRelaxation(node, None, None, "time_limit")

    def _fix_root(self, node: BranchNode, relaxation: NodeRelaxation) -> BranchNode:
        """在根节点同一对偶/图版本上逐机固定，再统一重建 RMP。

        输入：BranchNode 和已收敛的 NodeRelaxation。
        输出：删除安全弧、过滤旧列并更新 DSUB 的 BranchNode。
        """

        if not (self.config.use_reduced_cost_fixing and self.config.use_backward_labeling and self.config.use_label_joining):
            return node
        started = perf_counter()
        requests = pricing_requests_for_node(node, relaxation.solution.dual_solution)
        states, problems = [], []
        columns = node.rmp.columns
        for problem, state in zip(requests, node.bounds):
            if self._timed_out():
                return node
            self.statistics["fixing_calls"] += 1
            outcome = fix_arcs_by_reduced_cost(
                self.instance, problem, state, lower_bound=relaxation.pricing_lower_bound,
                upper_bound=self.information.upper_bound, full_master_dual_certified=True,
                relative_gap_limit=self.config.reduced_cost_fixing_gap,
                fixing_tolerance=self.config.reduced_cost_fixing_tolerance, columns=columns,
            )
            state = state if outcome.graph_update is None else outcome.graph_update.state
            states.append(state)
            problems.append(replace(problem, dag_arcs=state.dag_arcs))
            columns = outcome.retained_columns
            self.statistics["removed_arcs"] += len(outcome.removed_arcs)
            self.diagnostics.append(dict(node_id=0, stage="reduced_cost_fixing", machine_id=problem.machine_id,
                                         status=outcome.status, removed_arcs=outcome.removed_arcs))
        changed = tuple(states) != node.bounds or columns != node.rmp.columns
        updated = self._configure_bounds(replace(node, rmp=rebuild_rmp_with_columns(node.rmp, columns),
                                                 pricing_problems=tuple(problems), bounds=tuple(states))) if changed else node
        self._record(0, updated, "reduced_cost_fixing", status="completed" if changed else "unchanged",
                     elapsed_seconds=perf_counter() - started)
        return updated

    def _integer_candidate(self, identity: int, node: BranchNode, solution: RMPSolveResult) -> None:
        """核验整数 LP，覆盖型解有重复作业时去重构造全局可行候选。

        输入：节点号 int、BranchNode、RMPSolveResult。
        输出：None；仅通过真实覆盖/成本检查的候选会改善 incumbent。
        """

        tolerance = self.config.integer_tolerance
        if any(abs(value - round(value)) > tolerance for value in solution.column_values):
            return
        selected = tuple(column for column, value in zip(node.rmp.columns, solution.column_values) if value > 0.5)
        if validate_integer_schedule(self.instance, selected, node=node, tolerance=tolerance):
            source = identity
        else:
            used: set[int] = set()
            repaired = []
            for column in selected:
                jobs = tuple(job for job in column.job_ids if job not in used)
                used.update(jobs)
                if jobs:
                    repaired.append(build_schedule_column(self.instance, column.machine_id, jobs))
            selected, source = tuple(repaired), None
        self.information = record_incumbent(self.information, selected, source_node_id=source,
                                            elapsed_seconds=perf_counter() - self.started, tolerance=tolerance)

    def _enumerate(self, identity: int, node: BranchNode, relaxation: NodeRelaxation) -> BranchNode:
        """在作者的间隙条件下求有限列 IP，并核验/记录新整数上界。

        输入：节点号 int、BranchNode 和已收敛 NodeRelaxation。
        输出：合入枚举补充列后的 BranchNode；有限列 IP 不更新证明下界。
        """

        if not self.config.use_schedule_enumeration:
            return node
        gap = (self.information.upper_bound - relaxation.pricing_lower_bound) / max(1.0, abs(self.information.upper_bound))
        if gap >= self.config.enumeration_gap or self._timed_out():
            return node
        started = perf_counter()
        self.statistics["enumeration_calls"] += 1
        outcome = solve_schedule_enumeration(
            node, mode=self.config.schedule_enumeration_mode, dual_solution=relaxation.solution.dual_solution,
            lower_bound=relaxation.pricing_lower_bound, incumbent_objective=self.information.upper_bound,
            max_labels_per_machine=self.config.schedule_enumeration_max_labels,
            tolerance=self.config.integer_tolerance, time_limit=self._remaining(),
        )
        if outcome.added_columns:
            node = replace(node, rmp=rebuild_rmp_with_columns(node.rmp, outcome.columns))
            self.statistics["added_columns"] += len(outcome.added_columns)
        self._record(identity, node, "schedule_enumeration", status=outcome.status, elapsed_seconds=perf_counter() - started)
        self.information = record_incumbent(self.information, outcome.selected_columns, source_node_id=identity,
                                            elapsed_seconds=perf_counter() - self.started)
        self.diagnostics.append(dict(node_id=identity, stage="schedule_enumeration", mode=outcome.mode,
                                     status=outcome.status, enumeration_complete=outcome.enumeration_complete,
                                     objective_value=outcome.objective_value))
        return node

    def _trial_lp(self, rmp) -> TrialLPResult | None:
        """为强分支提供受总预算约束的 RMP 试算。

        输入：RestrictedMasterProblem。输出：TrialLPResult 或 None；不写入证明记录。
        """

        if self._timed_out():
            return None
        result = self._solve_rmp(rmp)
        if result.status != "optimal":
            return None
        return TrialLPResult(result.objective_value, result.column_values, result.dual_solution)

    def _trial_heuristic(self, node: BranchNode) -> TrialLPResult | None:
        """在强分支第二阶段执行现有启发式列生成，以无新列或目标停滞停止。

        输入：独立试探 BranchNode。输出：TrialLPResult 或 None，不产生证明下界。
        """

        last_objective = None
        same_objective = 0
        while not self._timed_out():
            solution = self._trial_lp(node.rmp)
            if solution is None:
                return None
            same_objective = same_objective + 1 if solution.objective_value == last_objective else 0
            last_objective = solution.objective_value
            if same_objective > self.config.heuristic_max_same_objective:
                return solution
            additions = []
            requests = pricing_requests_for_node(node, solution.dual_solution)
            keys = frozenset(column.key for column in node.rmp.columns)
            for problem, state in zip(requests, node.bounds):
                if self._timed_out():
                    return None
                limits = state.completion_time_upper_bounds()
                problem = replace(problem, reduced_cost_tolerance=self.config.heuristic_pricing_tolerance)
                limits[problem.sink_node] = state.global_upper_bound
                if self.config.use_neighborhood_search and self.config.use_bucket_graph and self.config.use_time_bucket:
                    partition = build_time_buckets(problem, limits, num_buckets_per_job=self.config.num_time_buckets)
                    trial = solve_neighborhood_pricing(
                        self.instance, problem, build_bucket_graph(problem, partition),
                        rng=self.rng, max_iterations=self.config.heuristic_iterations,
                        num_neighbors=self.config.heuristic_neighbors, existing_column_keys=keys,
                    )
                    candidates = trial.negative_evaluations
                else:
                    trial = solve_forward_labeling(self.instance, problem, max_labels_per_node=20,
                        completion_upper_bounds=limits, deadline=self.deadline,
                        max_columns=self.config.max_added_columns, existing_column_keys=keys)
                    candidates = trial.negative_evaluations
                additions.extend(item.column for item in candidates)
            if not additions:
                return solution
            node = replace(node, rmp=rebuild_rmp_with_columns(node.rmp, (*node.rmp.columns, *additions)))
        return None

    def _global_lower(self, identities: Sequence[int], closed: Sequence[float]) -> float:
        """汇总完整未解决前沿和已关闭叶节点的有效下界。

        输入：未解决节点号 Sequence[int]、已关闭叶界 Sequence[float]。
        输出：float 全局下界；停止但未证明的节点始终保留在前沿。
        """

        records = tuple(self.information.nodes[identity] for identity in identities)
        frontier = frontier_lower_bound(records, upper_bound=self.information.upper_bound)
        bounds = list(closed)
        if frontier is not None:
            bounds.append(frontier)
        if self.information.upper_bound is not None:
            bounds.append(self.information.upper_bound)
        return min(bounds) if bounds else 0.0

    def _gap_satisfied(self, lower: float) -> bool:
        """按统一绝对/相对阈值判断是否具有足够小的证明间隙。

        输入：全局下界 float。输出：bool。
        """

        upper = self.information.upper_bound
        return upper is not None and upper - lower <= max(
            self.config.absolute_gap, self.config.relative_gap * max(1.0, abs(upper))
        )

    def solve(self) -> BPCResult:
        """执行节点搜索，直至间隙达标、预算耗尽或作者流程停止。

        输入：初始化的 BPCSolver。输出：含证书边界与调度的 BPCResult。
        """

        self._initial_node()
        pending, stopped, closed = [0], [], []
        status = "author_flow_stopped"
        if not self.config.use_root_node_column_initial:
            pending, stopped, status = [], [0], "unsupported_empty_root_pool"
        while pending:
            if self._timed_out():
                status = "time_limit"
                break
            if self.config.max_nodes is not None and self.statistics["explored_nodes"] >= self.config.max_nodes:
                status = "node_limit"
                break
            identity = pending.pop()  # 与 C# 一致，按深度优先规则搜索。
            record = self.information.nodes[identity]
            node = record.node
            self.statistics["explored_nodes"] += 1
            if node_pruning_reason(record, self.information.upper_bound, tolerance=self.config.absolute_gap):
                self._record(identity, node, "pruning", status="bound")
                closed.append(record.lower_bound)
                continue
            relaxation = self._converge(identity, node)
            node = relaxation.node
            if relaxation.status != "converged":
                stopped.append(identity)
                self._record(identity, node, "node_stopped", status=relaxation.status)
                if relaxation.status == "restricted_rmp_solver_size_limit":
                    status = "solver_size_limit"
                if self._timed_out():
                    status = "time_limit"
                    break
                continue
            self._integer_candidate(identity, node, relaxation.solution)
            lower = self.information.nodes[identity].lower_bound
            if not self._gap_satisfied(lower):
                if identity == 0:
                    updated = self._fix_root(node, relaxation)
                    if updated != node:
                        node = updated
                        relaxation = self._converge(identity, node)
                        node = relaxation.node
                if relaxation.status == "converged":
                    self._integer_candidate(identity, node, relaxation.solution)
                    updated = self._enumerate(identity, node, relaxation)
                    if updated != node:
                        node = updated
                        relaxation = self._converge(identity, node)
                        node = relaxation.node
                if relaxation.status != "converged" or self._timed_out():
                    stopped.append(identity)
                    self._record(identity, node, "node_stopped", status=relaxation.status if not self._timed_out() else "time_limit")
                    if relaxation.status == "restricted_rmp_solver_size_limit":
                        status = "solver_size_limit"
                    if self._timed_out():
                        status = "time_limit"
                        break
                    continue
            lower = self.information.nodes[identity].lower_bound
            if self._gap_satisfied(lower):
                closed.append(lower)
                self._record(identity, node, "pruning", status="gap_bound")
            else:
                started = perf_counter()
                outcome = select_branch_candidate(
                    node, relaxation.solution.column_values, relaxation.solution.objective_value,
                    incumbent_objective=self.information.upper_bound, use_strong_branching=self.config.use_strong_branching,
                    ordinary_kind=self.config.branch_variable, phase0_size=self.config.strong_branch_phase0,
                    phase1_size=self.config.strong_branch_phase1, history=self.history,
                    memory_limit_bytes=int(self.config.strong_branch_memory_gb * 1024**3),
                    lp_solver=self._trial_lp, heuristic_runner=self._trial_heuristic,
                )
                self.history = outcome.history
                self.statistics["branch_calls"] += 1
                self.statistics["strong_branches"] += int(outcome.method == "strong")
                self._record(identity, node, "branching", status=outcome.method, elapsed_seconds=perf_counter() - started)
                self.diagnostics.append(dict(node_id=identity, stage="branching", method=outcome.method,
                                             candidate=None if outcome.candidate is None else asdict(outcome.candidate),
                                             phase1_scores=[asdict(score) for score in outcome.phase1_scores],
                                             phase2_scores=[asdict(score) for score in outcome.phase2_scores]))
                if not outcome.children:
                    stopped.append(identity)
                for child in reversed(outcome.children):
                    child = self._configure_bounds(child)
                    child_id = len(self.information.nodes)
                    self.information = register_node_information(self.information, child, child_id, parent_id=identity)
                    pending.append(child_id)
            lower = self._global_lower((*pending, *stopped), closed)
            if self._gap_satisfied(lower):
                status = "optimal"
                break
        unresolved = tuple(sorted((*pending, *stopped)))
        lower = self._global_lower(unresolved, closed)
        proven = self._gap_satisfied(lower)
        if proven:
            status = "optimal"
        upper = self.information.upper_bound
        gap = None if upper is None else max(0.0, upper - lower)
        return BPCResult(status, lower, upper, gap, None if gap is None else gap / max(1.0, abs(upper)),
                         proven, perf_counter() - self.started, self.information, dict(self.statistics),
                         tuple(self.diagnostics), unresolved)


def save_result(result: BPCResult, configuration: argparse.Namespace) -> Path:
    """按规模保存输入数据、调度、配置、界及紧凑的节点/证书记录。

    输入：BPCResult 和 argparse.Namespace。
    输出：唯一结果 JSON 的绝对 Path；每次运行独立保存，便于比较历史结果。
    """

    instance = result.information.instance
    root = Path(configuration.results_dir).resolve()
    for scale in ("low", "medium", "high"):
        (root / scale).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = f"{instance.num_machines}_{instance.num_jobs}_TWCT_{instance.instance_id}_d{configuration.processing_time_decimals}_{stamp}.json"
    path = root / instance_scale(instance.num_jobs, configuration) / name
    nodes = []
    for record in result.information.nodes:
        events = []
        for event in record.stages:
            solution = event.solution
            events.append(dict(
                stage=event.stage, status=event.status, elapsed_seconds=event.elapsed_seconds,
                lower_bound=event.lower_bound, upper_bound=event.upper_bound,
                exact_pricing_complete=event.exact_pricing_complete, infeasibility_proven=event.infeasibility_proven,
                added_columns=event.added_columns, removed_columns=event.removed_columns,
                added_cut_rows=event.added_cut_rows, removed_dag_arcs=event.removed_dag_arcs,
                lp_objective=None if solution is None else solution.objective_value,
                dual=None if solution is None or solution.dual_solution is None else asdict(solution.dual_solution),
                active_columns=[] if solution is None else [dict(machine_id=column.machine_id, job_ids=column.job_ids, value=value)
                    for column, value in zip(event.node.rmp.columns, solution.column_values) if value > configuration.integer_tolerance],
            ))
        nodes.append(dict(node_id=record.node_id, parent_id=record.parent_id, depth=record.depth,
                          lower_bound=record.lower_bound, decisions=[asdict(item) for item in record.node.decisions], stages=events))
    payload = dict(
        schema_version=1, instance=asdict(instance),
        source_path=None if result.source_path is None else str(result.source_path),
        configuration={key: str(value) if isinstance(value, Path) else value for key, value in vars(configuration).items()},
        status=result.status, optimality_proven=result.optimality_proven,
        lower_bound=result.lower_bound, upper_bound=result.upper_bound,
        absolute_gap=result.absolute_gap, relative_gap=result.relative_gap, elapsed_seconds=result.elapsed_seconds,
        schedules=[] if not result.information.incumbent_history else [asdict(column) for column in result.information.incumbent_history[-1].columns],
        incumbent_history=[dict(objective_value=item.objective_value, source_node_id=item.source_node_id,
                                elapsed_seconds=item.elapsed_seconds) for item in result.information.incumbent_history],
        statistics=result.statistics, unresolved_node_ids=result.unresolved_node_ids,
        diagnostics=result.diagnostics, nodes=nodes,
    )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return path


def solve_instance(instance: SchedulingInstance | None = None, configuration: argparse.Namespace | None = None) -> BPCResult:
    """读取或接收一个实例并执行完整 BPC，可由测试直接调用。

    输入：可选 SchedulingInstance、可选 config.get_config 的 argparse.Namespace。
    输出：BPCResult；save_results 开启时同时保存 JSON 并设置 result_path。
    """

    configuration = get_config([]) if configuration is None else configuration
    source = build_instance_path(configuration) if instance is None else None
    instance = load_instance(configuration) if instance is None else instance
    configuration = argparse.Namespace(**vars(configuration))
    configuration.num_machine = instance.num_machines
    configuration.num_job = instance.num_jobs
    configuration.instance_id = instance.instance_id
    result = BPCSolver(instance, configuration).solve()
    result = replace(result, source_path=source)
    return replace(result, result_path=save_result(result, configuration)) if configuration.save_results else result


def main(argv: Sequence[str] | None = None) -> tuple[BPCResult, ...]:
    """提供单实例/批量命令行入口，并打印可核对的运行摘要。

    输入：可选 Sequence[str] 命令行参数，None 时读取实际命令行。
    输出：按运行顺序排列的 BPCResult 元组；文件由 solve_instance 保存。
    """

    configuration = get_config(argv)
    configurations = discover_instances(configuration) if configuration.batch else (configuration,)
    results = []
    for current in configurations:
        result = solve_instance(configuration=current)
        results.append(result)
        print(json.dumps(dict(instance=f"{current.num_machine}_{current.num_job}_TWCT_{current.instance_id}",
                              status=result.status, UB=result.upper_bound, LB=result.lower_bound,
                              gap=result.relative_gap, seconds=round(result.elapsed_seconds, 3),
                              result_path=None if result.result_path is None else str(result.result_path)), ensure_ascii=False))
    return tuple(results)


if __name__ == "__main__":
    main()
