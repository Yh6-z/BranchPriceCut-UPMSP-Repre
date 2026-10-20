"""在完整 SWPT DAG 上执行精确前向标签定价。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from time import perf_counter

from src.BranchPriceCut.Price.cut_resource import (
    extend_cut_resources,
    initial_cut_states,
)
from src.BranchPriceCut.Price.labels import ForwardLabel, insert_nondominated
from src.model.PP import PricingEvaluation, PricingProblem, evaluate_schedule
from src.model.dataload import SchedulingInstance


@dataclass(frozen=True, slots=True)
class ExactPricingResult:
    machine_id: int
    status: str
    best_evaluation: PricingEvaluation | None
    is_exact: bool
    generated_labels: int
    retained_labels: int
    negative_evaluations: tuple[PricingEvaluation, ...] = ()

    @property
    def has_improving_column(self) -> bool:
        """判断是否存在负约化成本列。

        输入：当前 ExactPricingResult。
        输出：bool；局部图没有可行源汇路径时返回 False。
        """

        return self.best_evaluation is not None and self.best_evaluation.is_improving


@dataclass(frozen=True, slots=True)
class ForwardLabelSearch:
    """前向搜索的逐节点标签快照、割支持状态及搜索统计。"""

    labels_by_node: tuple[tuple[int, tuple[ForwardLabel, ...]], ...]
    is_exact: bool
    generated_labels: int

    def labels_at(self, vertex: int) -> tuple[ForwardLabel, ...]:
        """读取一个节点的前向标签。

        输入：节点编号 ``int``。
        输出：该节点保留的 ``tuple[ForwardLabel, ...]``。
        """

        return next(labels for node, labels in self.labels_by_node if node == vertex)

    @property
    def retained_labels(self) -> int:
        """统计搜索后保留的标签。

        输入：无。
        输出：所有节点保留标签的总数 ``int``。
        """

        return sum(len(labels) for _, labels in self.labels_by_node)


def evaluate_pricing_label(
    instance: SchedulingInstance,
    problem: PricingProblem,
    label: ForwardLabel,
) -> PricingEvaluation:
    """用已有标签资源复核完整调度的约化成本。

    输入：SchedulingInstance、PricingProblem 和完整 ForwardLabel。
    输出：PricingEvaluation，割系数顺序与当前定价请求一致。
    """

    cuts = tuple(cut for cut in problem.cut_definitions
                 if cut.cut_id in problem.cut_ids and hasattr(cut, "transition"))
    coefficients = {cut.cut_id: value for cut, value in zip(cuts, label.cut_coefficients)}
    return evaluate_schedule(instance, problem, label.job_ids,
        cut_coefficients=tuple(coefficients.get(identity, 0.0) for identity in problem.cut_ids))


def select_negative_labels(
    instance: SchedulingInstance,
    problem: PricingProblem,
    labels: tuple[ForwardLabel, ...] | list[ForwardLabel],
    *,
    max_columns: int,
    existing_column_keys: frozenset[tuple[int, tuple[int, ...]]],
) -> tuple[PricingEvaluation, ...]:
    """从已有汇点标签按约化成本排序选取多列，不另行搜索或枚举。

    输入：实例、定价请求、已有标签序列、入列上限 int 与旧列键 frozenset。
    输出：先排除旧列及重复列后选出的负约化成本 PricingEvaluation 元组。
    """

    selected = []
    seen = set(existing_column_keys)
    for label in sorted(labels, key=lambda item: (item.reduced_cost, item.job_ids)):
        if len(selected) >= max_columns:
            break
        key = problem.machine_id, label.job_ids
        if not label.job_ids or key in seen or label.reduced_cost >= -problem.reduced_cost_tolerance:
            continue
        evaluation = evaluate_pricing_label(instance, problem, label)
        if evaluation.is_improving:
            selected.append(evaluation)
            seen.add(key)
    return tuple(selected)


def collect_forward_labels(
    instance: SchedulingInstance,
    problem: PricingProblem,
    *,
    completion_upper_bounds: Mapping[int, float] | None = None,
    dominance_tolerance: float = 1e-9,
    max_labels_per_node: int | None = None,
    deadline: float | None = None,
) -> ForwardLabelSearch:
    """运行可供定价和删弧复用的前向标签搜索。

    输入：``SchedulingInstance``、含当前 DAG/真实对偶的 ``PricingProblem``，
    可选逐节点完工时间上界、数值/标签预算及 perf_counter 截止时刻 float。
    输出：``ForwardLabelSearch``，含所有节点标签和精确性标志。
    """

    upper_bounds = {} if completion_upper_bounds is None else completion_upper_bounds
    dual_by_cut_id = dict(zip(problem.cut_ids, problem.cut_duals))
    cuts = tuple(
        cut
        for cut in problem.cut_definitions
        if cut.cut_id in dual_by_cut_id and hasattr(cut, "transition")
    )
    cut_duals = tuple(dual_by_cut_id[cut.cut_id] for cut in cuts)

    labels_by_node: dict[int, list[ForwardLabel]] = {
        vertex: [] for vertex in problem.dag_vertices
    }
    source_label = ForwardLabel(
        last_node=problem.source_node,
        time=0.0,
        reduced_cost=-problem.machine_dual,
        job_ids=(),
        cut_states=initial_cut_states(cuts),
        cut_coefficients=(0.0,) * len(cuts),
    )
    labels_by_node[problem.source_node].append(source_label)

    adjacency: dict[int, list[int]] = {
        vertex: [] for vertex in problem.dag_vertices
    }
    for tail, head in problem.dag_arcs:
        adjacency[tail].append(head)

    generated_labels = 1
    interrupted = False
    for tail in problem.dag_vertices[:-1]:
        if interrupted:
            break
        for label in tuple(labels_by_node[tail]):
            if deadline is not None and perf_counter() >= deadline:
                interrupted = True
                break
            for head in adjacency[tail]:
                if head == problem.sink_node:
                    if label.time <= upper_bounds.get(head, float("inf")):
                        insert_nondominated(
                            labels_by_node[head],
                            ForwardLabel(
                                last_node=head,
                                time=label.time,
                                reduced_cost=label.reduced_cost,
                                job_ids=label.job_ids,
                                cut_states=label.cut_states,
                                cut_coefficients=label.cut_coefficients,
                            ),
                            tolerance=dominance_tolerance,
                        )
                        generated_labels += 1
                    continue

                processing_time = instance.processing_time(head, problem.machine_id)
                completion_time = label.time + processing_time
                if completion_time > upper_bounds.get(head, float("inf")):
                    continue
                states, coefficients, cut_increment = extend_cut_resources(
                    cuts,
                    cut_duals,
                    problem.machine_id,
                    head,
                    label.cut_states,
                    label.cut_coefficients,
                )
                candidate = ForwardLabel(
                    last_node=head,
                    time=completion_time,
                    reduced_cost=(
                        label.reduced_cost
                        + instance.weight(head) * completion_time
                        - problem.job_duals[head - 1]
                        + cut_increment
                    ),
                    job_ids=(*label.job_ids, head),
                    cut_states=states,
                    cut_coefficients=coefficients,
                )
                generated_labels += 1
                insert_nondominated(
                    labels_by_node[head],
                    candidate,
                    tolerance=dominance_tolerance,
                )
                if (
                    max_labels_per_node is not None
                    and len(labels_by_node[head]) > max_labels_per_node
                ):
                    labels_by_node[head].sort(
                        key=lambda item: (item.reduced_cost, item.time)
                    )
                    del labels_by_node[head][max_labels_per_node:]

    return ForwardLabelSearch(
        labels_by_node=tuple(
            (vertex, tuple(labels_by_node[vertex]))
            for vertex in problem.dag_vertices
        ),
        is_exact=not interrupted and max_labels_per_node is None and len(cuts) == len(problem.cut_ids),
        generated_labels=generated_labels,
    )


def solve_forward_labeling(
    instance: SchedulingInstance,
    problem: PricingProblem,
    *,
    dominance_tolerance: float = 1e-9,
    max_labels_per_node: int | None = None,
    completion_upper_bounds: Mapping[int, float] | None = None,
    deadline: float | None = None,
    max_columns: int = 1,
    existing_column_keys: frozenset[tuple[int, tuple[int, ...]]] = frozenset(),
) -> ExactPricingResult:
    """求解单机最小约化成本调度。

    输入：实例、定价请求、时间界/截止时刻，以及入列上限 int 和旧列键 frozenset。
    输出：含最优列及精确性统计的 ``ExactPricingResult``。
    """

    search = collect_forward_labels(
        instance,
        problem,
        dominance_tolerance=dominance_tolerance,
        max_labels_per_node=max_labels_per_node,
        completion_upper_bounds=completion_upper_bounds,
        deadline=deadline,
    )
    if not search.labels_at(problem.sink_node):
        return ExactPricingResult(
            problem.machine_id, "no_feasible_schedule", None, search.is_exact,
            search.generated_labels, search.retained_labels,
        )
    best_label = min(
        search.labels_at(problem.sink_node),
        key=lambda label: label.reduced_cost,
    )
    evaluation = evaluate_pricing_label(instance, problem, best_label)
    supports_all_cuts = all(any(cut.cut_id == identity and hasattr(cut, "transition")
                                for cut in problem.cut_definitions) for identity in problem.cut_ids)
    is_exact = search.is_exact
    if not supports_all_cuts:
        status = "unsupported_cut_definition"
    elif evaluation.is_improving:
        status = "negative_column"
    elif is_exact:
        status = "optimal_no_negative_column"
    else:
        status = "heuristic_no_negative_column"
    return ExactPricingResult(
        machine_id=problem.machine_id,
        status=status,
        best_evaluation=evaluation,
        is_exact=is_exact,
        generated_labels=search.generated_labels,
        retained_labels=search.retained_labels,
        negative_evaluations=select_negative_labels(
            instance, problem, search.labels_at(problem.sink_node),
            max_columns=max_columns, existing_column_keys=existing_column_keys,
        ),
    )
