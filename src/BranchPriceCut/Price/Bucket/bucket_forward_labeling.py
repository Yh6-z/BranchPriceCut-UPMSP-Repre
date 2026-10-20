"""组合时间桶、Bucket Graph 与桶支配插件的精确前向标签算法。

适用阶段：RMP 取得对偶值后的逐机器精确定价。该实现与原有非桶
``forward_labeling.py`` 并列，避免加速模块污染正确性基线。

Input:
    调度实例、``PricingProblem``、逐作业完工时间上界；可选时间桶构造器、
    Bucket Graph 构造器、桶支配策略和每作业桶数。
Output:
    ``ExactPricingResult``，包含最优列评价、精确性标志和标签统计；
    负约化成本列交给后续列生成，精确无负列状态用于 RMP 收敛证明。
"""

from __future__ import annotations

from math import inf
from typing import Mapping
from time import perf_counter

from src.BranchPriceCut.Price.Bucket.bucket_dominance import (
    BucketDominancePolicy,
    SafeBucketDominance,
)
from src.BranchPriceCut.Price.Bucket.bucket_graph import (
    BucketGraphBuilder,
    build_bucket_graph,
)
from src.BranchPriceCut.Price.cut_resource import (
    extend_cut_resources,
    initial_cut_states,
)
from src.BranchPriceCut.Price.forward_labeling import ExactPricingResult, select_negative_labels
from src.BranchPriceCut.Price.labels import ForwardLabel, insert_nondominated
from src.BranchPriceCut.Price.Bucket.time_bucket import (
    TimeBucket,
    TimeBucketBuilder,
    build_time_buckets,
)
from src.BranchPriceCut.Price.dynamic_upper_bound import DynamicUpperBoundState
from src.model.PP import PricingProblem, evaluate_schedule
from src.model.dataload import SchedulingInstance


def solve_bucket_forward_labeling(
    instance: SchedulingInstance,
    problem: PricingProblem,
    completion_time_upper_bounds: Mapping[int, float],
    *,
    num_buckets_per_job: int = 4,
    time_bucket_builder: TimeBucketBuilder | None = None,
    bucket_graph_builder: BucketGraphBuilder | None = None,
    dominance_policy: BucketDominancePolicy | None = None,
    use_bucket_dominance: bool = True,
    deadline: float | None = None,
    max_columns: int = 1,
    existing_column_keys: frozenset[tuple[int, tuple[int, ...]]] = frozenset(),
) -> ExactPricingResult:
    """组合桶插件求解 PSP。

    输入：实例、请求、时间界、桶参数、截止时刻及入列上限 int / 旧列键 frozenset。
    输出：ExactPricingResult；时间预算中断时 is_exact=False，不生成收敛证书。
    """

    partition = build_time_buckets(
        problem,
        completion_time_upper_bounds,
        num_buckets_per_job=num_buckets_per_job,
        builder=time_bucket_builder,
    )
    graph = build_bucket_graph(
        problem,
        partition,
        builder=bucket_graph_builder,
    )
    dominance = SafeBucketDominance() if dominance_policy is None else dominance_policy

    dual_by_cut_id = dict(zip(problem.cut_ids, problem.cut_duals))
    cuts = tuple(
        cut
        for cut in problem.cut_definitions
        if cut.cut_id in dual_by_cut_id and hasattr(cut, "transition")
    )
    cut_duals = tuple(dual_by_cut_id[cut.cut_id] for cut in cuts)
    labels_by_bucket: dict[TimeBucket, list[ForwardLabel]] = {
        bucket: [] for bucket in partition.buckets
    }
    best_reduced_cost: dict[TimeBucket, float] = {
        bucket: inf for bucket in partition.buckets
    }

    source_bucket = partition.for_vertex(problem.source_node)[0]
    labels_by_bucket[source_bucket].append(
        ForwardLabel(
            last_node=problem.source_node,
            time=0.0,
            reduced_cost=-problem.machine_dual,
            job_ids=(),
            cut_states=initial_cut_states(cuts),
            cut_coefficients=(0.0,) * len(cuts),
        )
    )

    generated_labels = 1
    for bucket in partition.buckets[:-1]:
        for label in tuple(labels_by_bucket[bucket]):
            if deadline is not None and perf_counter() >= deadline:
                return ExactPricingResult(problem.machine_id, "time_limit", None, False,
                                          generated_labels, sum(len(labels) for labels in labels_by_bucket.values()))
            if use_bucket_dominance and dominance.dominated_in_smaller_buckets(
                label,
                bucket,
                graph,
                labels_by_bucket,
                best_reduced_cost,
            ):
                continue

            for bucket_arc in graph.outgoing(bucket):
                head = bucket_arc.dag_arc[1]
                if head == problem.sink_node:
                    candidate = ForwardLabel(
                        last_node=head,
                        time=label.time,
                        reduced_cost=label.reduced_cost,
                        job_ids=label.job_ids,
                        cut_states=label.cut_states,
                        cut_coefficients=label.cut_coefficients,
                    )
                    generated_labels += 1
                    sink_bucket = partition.for_vertex(head)[0]
                    if use_bucket_dominance:
                        dominance.insert(labels_by_bucket[sink_bucket], candidate)
                    else:
                        labels_by_bucket[sink_bucket].append(candidate)
                    continue

                completion_time = max(
                    label.time + bucket_arc.processing_time,
                    partition.for_vertex(head)[0].lower_bound,
                )
                if completion_time > partition.for_vertex(head)[-1].upper_bound:
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
                actual_bucket = graph.locate(head, completion_time)
                if use_bucket_dominance:
                    dominance.insert(labels_by_bucket[actual_bucket], candidate)
                else:
                    labels_by_bucket[actual_bucket].append(candidate)

        best_reduced_cost[bucket] = dominance.update_best_reduced_cost(
            bucket,
            graph,
            labels_by_bucket[bucket],
            best_reduced_cost,
        )

    sink_bucket = partition.for_vertex(problem.sink_node)[0]
    sink_labels = labels_by_bucket[sink_bucket]
    if not sink_labels:
        return ExactPricingResult(
            problem.machine_id, "no_feasible_schedule", None,
            len(cuts) == len(problem.cut_ids), generated_labels,
            sum(len(labels) for labels in labels_by_bucket.values()),
        )
    if not use_bucket_dominance:
        nondominated_sink: list[ForwardLabel] = []
        for label in sink_labels:
            insert_nondominated(nondominated_sink, label)
        sink_labels = nondominated_sink
    best_label = min(sink_labels, key=lambda label: label.reduced_cost)

    coefficient_by_id = {
        cut.cut_id: coefficient
        for cut, coefficient in zip(cuts, best_label.cut_coefficients)
    }
    evaluation = evaluate_schedule(
        instance,
        problem,
        best_label.job_ids,
        cut_coefficients=tuple(
            coefficient_by_id.get(cut_id, 0.0) for cut_id in problem.cut_ids
        ),
    )
    supports_all_cuts = len(cuts) == len(problem.cut_ids)
    if not supports_all_cuts:
        status = "unsupported_cut_definition"
    elif evaluation.is_improving:
        status = "negative_column"
    else:
        status = "optimal_no_negative_column"
    return ExactPricingResult(
        machine_id=problem.machine_id,
        status=status,
        best_evaluation=evaluation,
        is_exact=supports_all_cuts,
        generated_labels=generated_labels,
        retained_labels=sum(len(labels) for labels in labels_by_bucket.values()),
        negative_evaluations=select_negative_labels(
            instance, problem, sink_labels, max_columns=max_columns,
            existing_column_keys=existing_column_keys,
        ),
    )


def solve_bucket_forward_labeling_with_bounds(
    instance: SchedulingInstance,
    problem: PricingProblem,
    bounds: DynamicUpperBoundState,
    *,
    num_buckets_per_job: int = 4,
    time_bucket_builder: TimeBucketBuilder | None = None,
    bucket_graph_builder: BucketGraphBuilder | None = None,
    dominance_policy: BucketDominancePolicy | None = None,
    use_bucket_dominance: bool = True,
    deadline: float | None = None,
    max_columns: int = 1,
    existing_column_keys: frozenset[tuple[int, tuple[int, ...]]] = frozenset(),
) -> ExactPricingResult:
    """以 DSUB 的局部 DAG 和作业上界运行桶版定价。

    输入：实例、请求、DSUB、桶参数、截止时刻、入列上限 int 和旧列键 frozenset。
    输出：ExactPricingResult；转发全部预算和桶开关。
    """

    local_problem = PricingProblem(
        machine_id=problem.machine_id,
        num_eligible_jobs=problem.num_eligible_jobs,
        source_node=problem.source_node,
        sink_node=problem.sink_node,
        job_ids=problem.job_ids,
        dag_vertices=problem.dag_vertices,
        weights=problem.weights,
        processing_times=problem.processing_times,
        swpt_job_order=problem.swpt_job_order,
        dag_arcs=bounds.dag_arcs,
        job_duals=problem.job_duals,
        machine_dual=problem.machine_dual,
        cut_ids=problem.cut_ids,
        cut_duals=problem.cut_duals,
        cut_definitions=problem.cut_definitions,
        reduced_cost_tolerance=problem.reduced_cost_tolerance,
    )
    return solve_bucket_forward_labeling(
        instance,
        local_problem,
        bounds.completion_time_upper_bounds(),
        num_buckets_per_job=num_buckets_per_job,
        time_bucket_builder=time_bucket_builder,
        bucket_graph_builder=bucket_graph_builder,
        dominance_policy=dominance_policy,
        use_bucket_dominance=use_bucket_dominance,
        deadline=deadline,
        max_columns=max_columns,
        existing_column_keys=existing_column_keys,
    )
