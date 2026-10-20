"""列生成使用的启发式定价入口和精确定价回退。"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from src.BranchPriceCut.Price.rmp_integration import rebuild_rmp_with_columns
from src.BranchPriceCut.Price.Bucket.bucket_graph import BucketGraph
from src.BranchPriceCut.Price.forward_labeling import (
    ExactPricingResult,
    solve_forward_labeling,
)
from src.heuristic.neighborhood_search import (
    NeighborhoodPricingResult,
    solve_neighborhood_pricing,
)
from src.model.RMP import RestrictedMasterProblem
from src.model.PP import PricingProblem
from src.model.dataload import SchedulingInstance


def solve_heuristic_pricing(
    instance: SchedulingInstance,
    problem: PricingProblem,
    *,
    max_labels_per_node: int = 20,
) -> ExactPricingResult:
    """保留原有的标签数限制启发式定价。

    输入：``dataload.py`` 的实例、``PP.py`` 的定价请求和单节点标签上限。
    输出：``forward_labeling.py`` 的 ``ExactPricingResult``；限制标签时不作
    无负列的精确证明。
    """

    return solve_forward_labeling(
        instance,
        problem,
        max_labels_per_node=max_labels_per_node,
    )


@dataclass(frozen=True, slots=True)
class PricingDispatchResult:
    """一次定价决策后的 RMP、启发式信息及可选精确结果。"""

    rmp: RestrictedMasterProblem
    heuristic_result: NeighborhoodPricingResult
    exact_result: ExactPricingResult | None
    added_column: bool


def price_with_neighborhood_fallback(
    instance: SchedulingInstance,
    rmp: RestrictedMasterProblem,
    problem: PricingProblem,
    graph: BucketGraph,
    *,
    max_iterations: int = 50,
    num_neighbors: int = 10,
    rng: Random | None = None,
) -> PricingDispatchResult:
    """先搜索邻域；无新负列时运行精确定价，并将新列加入 RMP。

    输入：``dataload.py`` 的实例、``RMP.py`` 的现有主问题、``PP.py`` 的
    定价请求、``bucket_graph.py`` 的桶图和可选随机搜索参数。
    输出：``PricingDispatchResult``；新负列由 ``Price/rmp_integration.py``
    重建 RMP（同步重算语义割系数）。精确结果来自 ``forward_labeling.py``。
    """

    keys = frozenset(column.key for column in rmp.columns)
    heuristic = solve_neighborhood_pricing(
        instance,
        problem,
        graph,
        max_iterations=max_iterations,
        num_neighbors=num_neighbors,
        rng=rng,
        existing_column_keys=keys,
    )
    if heuristic.negative_evaluation is not None:
        column = heuristic.negative_evaluation.column
        return PricingDispatchResult(
            rmp=rebuild_rmp_with_columns(rmp, (*rmp.columns, column)),
            heuristic_result=heuristic,
            exact_result=None,
            added_column=True,
        )

    exact = solve_forward_labeling(instance, problem)
    if exact.has_improving_column and exact.best_evaluation.column.key not in keys:
        return PricingDispatchResult(
            rmp=rebuild_rmp_with_columns(
                rmp, (*rmp.columns, exact.best_evaluation.column)
            ),
            heuristic_result=heuristic,
            exact_result=exact,
            added_column=True,
        )
    return PricingDispatchResult(
        rmp=rmp,
        heuristic_result=heuristic,
        exact_result=exact,
        added_column=False,
    )
