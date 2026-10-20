from __future__ import annotations

import unittest
from random import Random

from src.BranchPriceCut.Price import (
    add_lm_subset_row_cuts,
    build_bucket_graph,
    build_time_buckets,
)
from src.heuristic import (
    extend_heuristic_label,
    price_with_neighborhood_fallback,
    random_bucket_extension,
    solve_neighborhood_pricing,
)
from src.BranchPriceCut.Price.Bucket.bucket_graph import BucketGraph
from src.heuristic.heuristic_extension import initial_heuristic_label
from src.heuristic.neighborhood_search import NO_NEGATIVE_MESSAGE
from src.BranchPriceCut.Cut import LimitedMemorySubsetRowCut
from src.model.PP import build_pricing_problems, evaluate_schedule
from src.model.RMP import DualSolution, build_restricted_master
from src.model.dataload import SchedulingInstance


def sample_problem(job_duals: tuple[float, ...]):
    """建立有多条源汇路径的三作业单机定价问题。

    输入：``tuple[float, ...]``，逐作业对偶值。
    输出：``dataload.py`` 的实例、``RMP.py`` 的主问题、``PP.py`` 的定价
    请求，以及 ``Bucket/bucket_graph.py`` 的桶图。
    """

    instance = SchedulingInstance(
        num_machines=1,
        num_jobs=3,
        instance_id=1,
        job_ids=(1, 2, 3),
        machine_ids=(1,),
        weights=(1, 1, 1),
        processing_times=((1.0,), (2.0,), (3.0,)),
    )
    rmp = build_restricted_master(instance, ())
    problem = build_pricing_problems(
        rmp,
        DualSolution(job_duals=job_duals, machine_duals=(0.0,)),
    )[0]
    partition = build_time_buckets(problem, {job: 6.0 for job in instance.job_ids})
    return instance, rmp, problem, build_bucket_graph(problem, partition)


class HeuristicNeighborhoodPricingTests(unittest.TestCase):
    def test_random_extension_preserves_seeded_search_paths(self) -> None:
        """核对复用桶图索引前记录的随机搜索路径，避免改变候选访问顺序。

        输入：TestCase；十个固定随机种子 int。
        输出：None；调度作业序列与原实现的记录不同时断言失败。
        """

        instance, _, problem, graph = sample_problem((10.0, 10.0, 10.0))
        expected = ((2,), (2,), (1, 3), (2,), (2,), (3,), (1, 2, 3), (3,), (2, 3), (3,))
        for seed, jobs in enumerate(expected):
            with self.subTest(seed=seed):
                walk = random_bucket_extension(instance, problem, graph, rng=Random(seed))
                self.assertEqual(walk.label.job_ids, jobs)

    def test_random_extension_returns_complete_feasible_schedule(self) -> None:
        instance, _, problem, graph = sample_problem((10.0, 10.0, 10.0))
        walk = random_bucket_extension(instance, problem, graph, rng=Random(3))

        self.assertIsNotNone(walk)
        self.assertEqual(walk.label.last_node, problem.sink_node)
        self.assertEqual(len(walk.states), len(walk.arcs) + 1)
        self.assertEqual(len(walk.label.job_ids), len(set(walk.label.job_ids)))
        evaluation = evaluate_schedule(instance, problem, walk.label.job_ids)
        self.assertEqual(evaluation.column.job_ids, walk.label.job_ids)
        self.assertAlmostEqual(evaluation.reduced_cost, walk.label.reduced_cost)

    def test_single_label_extension_preserves_cut_state_and_real_time(self) -> None:
        instance, rmp, _, _ = sample_problem((10.0, 10.0, 10.0))
        cut = LimitedMemorySubsetRowCut(
            "lm-src[1,2,3]", (1, 2, 3), (frozenset((1, 2, 3)),),
        )
        rmp = add_lm_subset_row_cuts(rmp, (cut,))
        problem = build_pricing_problems(
            rmp,
            DualSolution(
                job_duals=(10.0, 10.0, 10.0), machine_duals=(0.0,),
                cut_ids=(cut.cut_id,), cut_duals=(-3.0,),
            ),
        )[0]
        partition = build_time_buckets(problem, {job: 6.0 for job in instance.job_ids})
        graph = build_bucket_graph(problem, partition)
        label = initial_heuristic_label(problem)
        bucket = partition.for_vertex(0)[0]
        for head in (1, 2, 4):
            arc = next(arc for arc in graph.outgoing(bucket) if arc.dag_arc[1] == head)
            label, bucket = extend_heuristic_label(
                instance, problem, graph, label, bucket, arc,
            )

        evaluation = evaluate_schedule(instance, problem, label.job_ids)
        self.assertEqual(label.job_ids, (1, 2))
        self.assertEqual(label.cut_coefficients, (1.0,))
        self.assertAlmostEqual(label.reduced_cost, evaluation.reduced_cost)

    def test_random_extension_returns_none_without_nonempty_path(self) -> None:
        instance, _, problem, graph = sample_problem((10.0, 10.0, 10.0))
        empty_arc = next(arc for arc in graph.arcs if arc.dag_arc == (0, 4))
        sparse_graph = BucketGraph(
            machine_id=graph.machine_id,
            partition=graph.partition,
            arcs=(empty_arc,),
            adjacent_smaller_buckets=graph.adjacent_smaller_buckets,
        )

        self.assertIsNone(random_bucket_extension(
            instance, problem, sparse_graph, rng=Random(1),
        ))

    def test_negative_neighborhood_column_is_added_to_rmp(self) -> None:
        instance, rmp, problem, graph = sample_problem((10.0, 10.0, 10.0))
        result = price_with_neighborhood_fallback(
            instance, rmp, problem, graph, max_iterations=5, num_neighbors=3,
            rng=Random(2),
        )

        self.assertTrue(result.added_column)
        self.assertIsNone(result.exact_result)
        self.assertEqual(result.heuristic_result.status, "negative_column")
        self.assertLess(result.rmp.columns[0].cost - sum(
            problem.job_duals[job - 1]
            for job in result.rmp.columns[0].job_ids
        ), -problem.reduced_cost_tolerance)

    def test_added_column_recomputes_lm_src_row(self) -> None:
        instance, rmp, _, _ = sample_problem((10.0, 10.0, 10.0))
        cut = LimitedMemorySubsetRowCut(
            "lm-src[1,2,3]", (1, 2, 3), (frozenset((1, 2, 3)),),
        )
        rmp = add_lm_subset_row_cuts(rmp, (cut,))
        problem = build_pricing_problems(
            rmp,
            DualSolution(
                job_duals=(10.0, 10.0, 10.0), machine_duals=(0.0,),
                cut_ids=(cut.cut_id,), cut_duals=(-3.0,),
            ),
        )[0]
        partition = build_time_buckets(problem, {job: 6.0 for job in instance.job_ids})
        graph = build_bucket_graph(problem, partition)
        result = price_with_neighborhood_fallback(
            instance, rmp, problem, graph, max_iterations=4, num_neighbors=2,
            rng=Random(2),
        )

        self.assertTrue(result.added_column)
        self.assertEqual(
            result.rmp.cut_rows[0].coefficients,
            (cut.coefficient(result.rmp.columns[0]),),
        )

    def test_no_negative_triggers_exact_pricing(self) -> None:
        instance, rmp, problem, graph = sample_problem((0.0, 0.0, 0.0))
        result = price_with_neighborhood_fallback(
            instance, rmp, problem, graph, max_iterations=3, num_neighbors=2,
            rng=Random(4),
        )

        self.assertEqual(result.heuristic_result.message, NO_NEGATIVE_MESSAGE)
        self.assertIsNone(result.heuristic_result.negative_evaluation)
        self.assertIsNotNone(result.exact_result)
        self.assertFalse(result.added_column)
        self.assertTrue(result.exact_result.is_exact)

    def test_exact_adds_column_when_heuristic_graph_only_has_empty_path(self) -> None:
        instance, rmp, problem, graph = sample_problem((10.0, 10.0, 10.0))
        empty_arc = next(arc for arc in graph.arcs if arc.dag_arc == (0, 4))
        sparse_graph = BucketGraph(
            machine_id=graph.machine_id,
            partition=graph.partition,
            arcs=(empty_arc,),
            adjacent_smaller_buckets=graph.adjacent_smaller_buckets,
        )
        result = price_with_neighborhood_fallback(
            instance, rmp, problem, sparse_graph, max_iterations=1, num_neighbors=1,
            rng=Random(1),
        )

        self.assertEqual(result.heuristic_result.status, "heuristic_no_negative_column")
        self.assertIsNotNone(result.exact_result)
        self.assertTrue(result.added_column)
        self.assertTrue(result.exact_result.best_evaluation.is_improving)

    def test_existing_negative_column_does_not_stop_exact_fallback(self) -> None:
        instance, _, problem, graph = sample_problem((10.0, 10.0, 10.0))
        keys = frozenset(
            (1, jobs)
            for jobs in ((), (1,), (2,), (3,), (1, 2), (1, 3),
                         (2, 3), (1, 2, 3))
        )
        result = solve_neighborhood_pricing(
            instance, problem, graph, max_iterations=2, num_neighbors=2,
            rng=Random(1), existing_column_keys=keys,
        )

        self.assertIsNone(result.negative_evaluation)
        self.assertEqual(result.status, "heuristic_no_negative_column")


if __name__ == "__main__":
    unittest.main()
