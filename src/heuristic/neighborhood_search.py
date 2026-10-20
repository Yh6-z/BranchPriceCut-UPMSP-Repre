"""C# NeighborhoodSearch 的单路径桶弧变异版启发式定价。"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from src.BranchPriceCut.Price.Bucket.bucket_graph import BucketGraph
from src.heuristic.heuristic_extension import BucketWalk, random_bucket_extension
from src.model.PP import PricingEvaluation, PricingProblem, evaluate_schedule
from src.model.dataload import SchedulingInstance


EXACT_PRICER_SCRIPT = "src/BranchPriceCut/Price/forward_labeling.py:solve_forward_labeling"
NO_NEGATIVE_MESSAGE = (
    "启发式算法没有找到负reduced cost列，启动精确算法："
    + EXACT_PRICER_SCRIPT
)


@dataclass(frozen=True, slots=True)
class NeighborhoodPricingResult:
    """包含已有邻域搜索找到的全部新负列，以及兼容旧接口的最佳单列。"""

    machine_id: int
    status: str
    negative_evaluation: PricingEvaluation | None
    best_feasible_evaluation: PricingEvaluation | None
    message: str
    sampled_schedules: int
    negative_evaluations: tuple[PricingEvaluation, ...] = ()


def _evaluate_walk(
    instance: SchedulingInstance,
    problem: PricingProblem,
    walk: BucketWalk,
) -> PricingEvaluation:
    """复核一条完整标签路径的真实列约化成本。

    输入：``dataload.py`` 的实例、``PP.py`` 的定价请求，以及
    ``heuristic_extension.py`` 返回的完整 ``BucketWalk``。
    输出：``PP.py`` 的 ``PricingEvaluation``，含完整调度列与割系数。
    """

    return evaluate_schedule(instance, problem, walk.label.job_ids)


def solve_neighborhood_pricing(
    instance: SchedulingInstance,
    problem: PricingProblem,
    graph: BucketGraph,
    *,
    max_iterations: int = 50,
    num_neighbors: int = 10,
    rng: Random | None = None,
    existing_column_keys: frozenset[tuple[int, tuple[int, ...]]] = frozenset(),
) -> NeighborhoodPricingResult:
    """随机改动路径中一条桶弧，并保留约化成本最小的完整邻居。

    输入：``dataload.py`` 的实例、``PP.py`` 的定价请求、
    ``bucket_graph.py`` 的桶图；可选迭代数、邻居数、随机源和 RMP 已有列键。
    标签路径由 ``heuristic_extension.py`` 提供，约化成本由 ``PP.py`` 复核。
    输出：NeighborhoodPricingResult；negative_evaluations 保存同次搜索中的所有新负列，
    negative_evaluation 保留最佳单列；无新列时给出应执行精确定价的提示。
    """

    random_source = Random() if rng is None else rng
    current = random_bucket_extension(instance, problem, graph, rng=random_source)
    if current is None:
        return NeighborhoodPricingResult(
            machine_id=problem.machine_id,
            status="heuristic_no_negative_column",
            negative_evaluation=None,
            best_feasible_evaluation=None,
            message=NO_NEGATIVE_MESSAGE,
            sampled_schedules=0,
        )

    best = _evaluate_walk(instance, problem, current)
    best_negative = (
        best if best.is_improving and best.column.key not in existing_column_keys else None
    )
    sampled = 1
    negatives = {} if best_negative is None else {best_negative.column.key: best_negative}
    for _ in range(max_iterations):
        next_walk = current
        next_evaluation = _evaluate_walk(instance, problem, current)
        for _ in range(num_neighbors):
            mutation_index = random_source.randrange(len(current.arcs))
            prefix = BucketWalk(
                arcs=current.arcs[:mutation_index],
                states=current.states[:mutation_index + 1],
            )
            neighbor = random_bucket_extension(
                instance,
                problem,
                graph,
                rng=random_source,
                prefix=prefix,
                forbidden_first_arc=current.arcs[mutation_index],
            )
            if neighbor is None:
                continue
            evaluation = _evaluate_walk(instance, problem, neighbor)
            if evaluation.is_improving and evaluation.column.key not in existing_column_keys:
                negatives[evaluation.column.key] = evaluation
            sampled += 1
            if evaluation.reduced_cost < best.reduced_cost:
                best = evaluation
            if (
                evaluation.is_improving
                and evaluation.column.key not in existing_column_keys
                and (
                    best_negative is None
                    or evaluation.reduced_cost < best_negative.reduced_cost
                )
            ):
                best_negative = evaluation
            if evaluation.reduced_cost < next_evaluation.reduced_cost:
                next_walk = neighbor
                next_evaluation = evaluation
        current = next_walk

    return NeighborhoodPricingResult(
        machine_id=problem.machine_id,
        status="negative_column" if best_negative is not None else "heuristic_no_negative_column",
        negative_evaluation=best_negative,
        best_feasible_evaluation=best,
        message="向 RMP 添加负约化成本列" if best_negative is not None else NO_NEGATIVE_MESSAGE,
        sampled_schedules=sampled,
        negative_evaluations=tuple(sorted(negatives.values(), key=lambda item: item.reduced_cost)),
    )
