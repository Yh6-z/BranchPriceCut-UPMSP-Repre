"""在节点局部 DAG 上构造约化成本固定所需的后向仿射标签。

适用阶段：列—割收敛、取得真实对偶及当前 DSUB 之后。
输入：``PricingProblem``、``DynamicUpperBoundState``、可选精确前向标签及间隙。
输出：逐节点后向标签，供 ``label_joining.py`` 检查每条现存弧。
"""

from __future__ import annotations

from dataclasses import dataclass
from src.BranchPriceCut.Cut.lm_subset_row_cut import LimitedMemorySubsetRowCut
from src.BranchPriceCut.Price.dynamic_upper_bound import DynamicUpperBoundState
from src.BranchPriceCut.Price.forward_labeling import ForwardLabelSearch
from src.BranchPriceCut.Price.labels import ForwardLabel
from src.model.PP import PricingProblem


@dataclass(frozen=True, slots=True)
class BackwardLabel:
    """后缀的历时、权重、仿射截距、割资源与允许的前缀结束时间。"""

    first_node: int
    duration: float
    cumulative_weight: float
    base_reduced_cost: float
    optimistic_base: float
    job_ids: tuple[int, ...]
    cut_states: tuple[float, ...]
    cut_coefficients: tuple[float, ...]
    latest_prefix_time: float


@dataclass(frozen=True, slots=True)
class BackwardLabelSearch:
    """逐节点后向标签与安全完成界剪枝统计。"""

    labels_by_node: tuple[tuple[int, tuple[BackwardLabel, ...]], ...]
    generated_labels: int
    pruned_labels: int
    is_exact_for_threshold: bool

    def labels_at(self, vertex: int) -> tuple[BackwardLabel, ...]:
        """读取一个节点的后向标签。

        输入：节点编号 ``int``。
        输出：该节点的 ``tuple[BackwardLabel, ...]``。
        """

        return next(labels for node, labels in self.labels_by_node if node == vertex)


def collect_backward_labels(
    problem: PricingProblem,
    state: DynamicUpperBoundState,
    *,
    forward: ForwardLabelSearch | None = None,
    threshold: float | None = None,
    tolerance: float = 0.0,
) -> BackwardLabelSearch:
    """沿反向 DAG 生成后缀标签并可选执行安全完成界剪枝。

    输入：单机 ``PricingProblem``、同一局部图的 ``DynamicUpperBoundState``，
    可选精确 ``ForwardLabelSearch``、约化成本阈值 ``float`` 和容差。
    输出：``BackwardLabelSearch``；剪枝仅在割对偶非正时启用。
    """

    definitions = {cut.cut_id: cut for cut in problem.cut_definitions}
    cuts = tuple(definitions[cut_id] for cut_id in problem.cut_ids)
    predecessor_labels: dict[int, tuple[ForwardLabel, ...]] = {
        vertex: () for vertex in problem.dag_vertices
    }
    if forward is not None:
        predecessor_labels = {
            vertex: tuple(
                label
                for tail, head in state.dag_arcs
                if head == vertex
                for label in forward.labels_at(tail)
            )
            for vertex in problem.dag_vertices
        }

    labels_by_node: dict[int, list[BackwardLabel]] = {
        vertex: [] for vertex in problem.dag_vertices
    }
    labels_by_node[problem.sink_node].append(
        BackwardLabel(
            first_node=problem.sink_node,
            duration=0.0,
            cumulative_weight=0.0,
            base_reduced_cost=0.0,
            optimistic_base=0.0,
            job_ids=(),
            cut_states=(0.0,) * len(cuts),
            cut_coefficients=(0.0,) * len(cuts),
            latest_prefix_time=state.global_upper_bound,
        )
    )
    generated = 1
    pruned = 0
    can_prune = (
        forward is not None
        and forward.is_exact
        and threshold is not None
        and all(dual <= 0.0 for dual in problem.cut_duals)
        and all(isinstance(cut, LimitedMemorySubsetRowCut) for cut in cuts)
    )

    for vertex in reversed(problem.dag_vertices[1:-1]):
        processing_time = problem.processing_times[vertex - 1]
        weight = problem.weights[vertex - 1]
        for tail, head in state.dag_arcs:
            if tail != vertex:
                continue
            for suffix in labels_by_node[head]:
                jobs = (vertex, *suffix.job_ids)
                cumulative_weight = weight + suffix.cumulative_weight
                duration = processing_time + suffix.duration
                optimistic_base = (
                    suffix.optimistic_base
                    + processing_time * cumulative_weight
                    - problem.job_duals[vertex - 1]
                )
                latest_prefix_time = min(
                    state.bound(vertex).upper_bound - processing_time,
                    suffix.latest_prefix_time - processing_time,
                )
                generated += 1
                if latest_prefix_time < -tolerance:
                    pruned += 1
                    continue

                if can_prune:
                    prefixes = predecessor_labels[vertex]
                    if not prefixes:
                        pruned += 1
                        continue
                    minimum_time = min(label.time for label in prefixes)
                    minimum_cost = min(label.reduced_cost for label in prefixes)
                    if (
                        minimum_time > latest_prefix_time + tolerance
                        or minimum_cost
                        + optimistic_base
                        + minimum_time * cumulative_weight
                        > threshold + tolerance
                    ):
                        pruned += 1
                        continue

                cut_states: list[float] = []
                cut_coefficients: list[float] = []
                for cut in cuts:
                    cut_state = 0.0
                    coefficient = 0.0
                    for job_id in jobs:
                        cut_state, increment = cut.transition(
                            problem.machine_id, job_id, cut_state
                        )
                        coefficient += increment
                    cut_states.append(cut_state)
                    cut_coefficients.append(coefficient)
                base_reduced_cost = optimistic_base - sum(
                    dual * coefficient
                    for dual, coefficient in zip(problem.cut_duals, cut_coefficients)
                )
                labels_by_node[vertex].append(
                    BackwardLabel(
                        first_node=vertex,
                        duration=duration,
                        cumulative_weight=cumulative_weight,
                        base_reduced_cost=base_reduced_cost,
                        optimistic_base=optimistic_base,
                        job_ids=jobs,
                        cut_states=tuple(cut_states),
                        cut_coefficients=tuple(cut_coefficients),
                        latest_prefix_time=latest_prefix_time,
                    )
                )

    return BackwardLabelSearch(
        labels_by_node=tuple(
            (vertex, tuple(labels_by_node[vertex]))
            for vertex in problem.dag_vertices
        ),
        generated_labels=generated,
        pruned_labels=pruned,
        is_exact_for_threshold=True,
    )
