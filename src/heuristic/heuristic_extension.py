"""按 C# HeuristicPricing 的随机桶弧扩展构造完整机器调度。"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from src.BranchPriceCut.Price.Bucket.bucket_graph import BucketArc, BucketGraph
from src.BranchPriceCut.Price.Bucket.time_bucket import TimeBucket
from src.BranchPriceCut.Price.cut_resource import extend_cut_resources, initial_cut_states
from src.BranchPriceCut.Price.labels import ForwardLabel
from src.model.PP import PricingProblem
from src.model.dataload import SchedulingInstance


@dataclass(frozen=True, slots=True)
class BucketWalk:
    """完整路径上的桶弧，以及每次扩展前后的标签和实际桶。"""

    arcs: tuple[BucketArc, ...]
    states: tuple[tuple[ForwardLabel, TimeBucket], ...]

    @property
    def label(self) -> ForwardLabel:
        """取得路径终点标签。

        输入：当前 ``BucketWalk``。
        输出：``labels.py`` 的 ``ForwardLabel``。
        """

        return self.states[-1][0]


def initial_heuristic_label(problem: PricingProblem) -> ForwardLabel:
    """建立源标签。

    输入：``PP.py`` 的单机 ``PricingProblem``。
    输出：``labels.py`` 的 ``ForwardLabel``，含机器对偶与初始 lm-SRC 状态。
    """

    cuts = tuple(
        cut for cut in problem.cut_definitions
        if cut.cut_id in problem.cut_ids and hasattr(cut, "transition")
    )
    return ForwardLabel(
        last_node=problem.source_node,
        time=0.0,
        reduced_cost=-problem.machine_dual,
        job_ids=(),
        cut_states=initial_cut_states(cuts),
        cut_coefficients=(0.0,) * len(cuts),
    )


def extend_heuristic_label(
    instance: SchedulingInstance,
    problem: PricingProblem,
    graph: BucketGraph,
    label: ForwardLabel,
    bucket: TimeBucket,
    arc: BucketArc,
) -> tuple[ForwardLabel, TimeBucket] | None:
    """沿一条桶弧执行单标签扩展，拒绝时间或路径不合法的弧。

    输入：``dataload.py`` 的 ``SchedulingInstance``、``PP.py`` 的
    ``PricingProblem``、``bucket_graph.py`` 的图/弧、``labels.py`` 的标签，
    以及 ``time_bucket.py`` 的当前桶。
    输出：可行时返回新 ``ForwardLabel`` 与实际 ``TimeBucket``；否则返回 ``None``。
    """

    if arc.tail_bucket != bucket or arc.dag_arc[0] != label.last_node:
        return None
    head = arc.dag_arc[1]
    if head != problem.sink_node and head in label.job_ids:
        return None
    if head == problem.sink_node:
        sink_label = ForwardLabel(
            last_node=head,
            time=label.time,
            reduced_cost=label.reduced_cost,
            job_ids=label.job_ids,
            cut_states=label.cut_states,
            cut_coefficients=label.cut_coefficients,
        )
        return sink_label, graph.partition.for_vertex(head)[0]

    completion_time = label.time + instance.processing_time(head, problem.machine_id)
    buckets = graph.partition.for_vertex(head)
    if completion_time < buckets[0].lower_bound - 1e-9:
        return None
    if completion_time > buckets[-1].upper_bound + 1e-9:
        return None

    # 割定义和对偶来自 PP.py；资源转移由 Price/cut_resource.py 提供。
    dual_by_cut_id = dict(zip(problem.cut_ids, problem.cut_duals))
    cuts = tuple(
        cut for cut in problem.cut_definitions
        if cut.cut_id in dual_by_cut_id and hasattr(cut, "transition")
    )
    states, coefficients, cut_increment = extend_cut_resources(
        cuts,
        tuple(dual_by_cut_id[cut.cut_id] for cut in cuts),
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
    return candidate, graph.locate(head, completion_time)


def random_bucket_extension(
    instance: SchedulingInstance,
    problem: PricingProblem,
    graph: BucketGraph,
    *,
    rng: Random | None = None,
    prefix: BucketWalk | None = None,
    forbidden_first_arc: BucketArc | None = None,
) -> BucketWalk | None:
    """随机遍历桶弧直至得到完整调度；死路回溯后继续搜索。

    输入：``dataload.py`` 的实例、``PP.py`` 的定价请求、
    ``bucket_graph.py`` 的图；可选随机源、已有标签路径及首步禁用弧。
    输出：有可行路径时返回含完整 ``ForwardLabel`` 的 ``BucketWalk``，
    图上无可行源汇路径时返回 ``None``。随机只改变遍历顺序，不影响可行性。
    """

    random_source = Random() if rng is None else rng
    initial = prefix or BucketWalk(
        arcs=(),
        states=((
            initial_heuristic_label(problem),
            graph.partition.for_vertex(problem.source_node)[0],
        ),),
    )
    stack = [initial]
    while stack:
        walk = stack.pop()
        label, bucket = walk.states[-1]
        if label.last_node == problem.sink_node and label.job_ids:
            return walk
        # 桶图已有相同顺序的不可变出弧索引；邻域变异不需反复扫描全部图弧。
        arcs = list(graph.outgoing(bucket))
        random_source.shuffle(arcs)
        for arc in arcs:
            if forbidden_first_arc is not None and len(walk.arcs) == len(initial.arcs) and arc == forbidden_first_arc:
                continue
            extended = extend_heuristic_label(instance, problem, graph, label, bucket, arc)
            if extended is not None:
                stack.append(BucketWalk(
                    arcs=(*walk.arcs, arc),
                    states=(*walk.states, extended),
                ))
    return None
