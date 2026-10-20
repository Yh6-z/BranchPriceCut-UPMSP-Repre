from __future__ import annotations

import unittest
from itertools import combinations
from math import inf

from src.BranchPriceCut.Cut import LimitedMemorySubsetRowCut
from src.BranchPriceCut.Price import (
    add_lm_subset_row_cuts,
    SafeBucketDominance,
    build_bucket_graph,
    build_time_buckets,
    solve_bucket_forward_labeling,
    solve_forward_labeling,
)
from src.BranchPriceCut.Price.labels import ForwardLabel
from src.model.PP import build_pricing_problems, evaluate_schedule
from src.model.RMP import DualSolution, build_restricted_master
from src.model.dataload import SchedulingInstance
from src.model.schedulemodel import build_schedule_column


def small_instance() -> SchedulingInstance:
    return SchedulingInstance(
        num_machines=1,
        num_jobs=4,
        instance_id=1,
        job_ids=(1, 2, 3, 4),
        machine_ids=(1,),
        weights=(3, 2, 4, 1),
        processing_times=((2.0,), (3.0,), (1.0,), (4.0,)),
    )


def problem_with_cut():
    instance = small_instance()
    columns = (
        build_schedule_column(instance, 1, (1, 2)),
        build_schedule_column(instance, 1, (3, 4)),
    )
    cut = LimitedMemorySubsetRowCut(
        "lm-src[1,2,3]",
        (1, 2, 3),
        (frozenset((1, 2, 3, 4)),),
    )
    rmp = add_lm_subset_row_cuts(build_restricted_master(instance, columns), (cut,))
    dual = DualSolution(
        job_duals=(18.0, 12.0, 9.0, 3.0),
        machine_duals=(-2.0,),
        cut_ids=(cut.cut_id,),
        cut_duals=(-3.0,),
    )
    return instance, build_pricing_problems(rmp, dual)[0]


def label(time: float, reduced_cost: float) -> ForwardLabel:
    return ForwardLabel(
        last_node=1,
        time=time,
        reduced_cost=reduced_cost,
        job_ids=(1,),
        cut_states=(0.0,),
        cut_coefficients=(0.0,),
    )


class BucketDominanceTests(unittest.TestCase):
    def test_candidate_can_be_dominated_by_adjacent_smaller_bucket(self) -> None:
        instance, problem = problem_with_cut()
        partition = build_time_buckets(
            problem,
            {job_id: 10.0 for job_id in instance.job_ids},
            num_buckets_per_job=2,
        )
        graph = build_bucket_graph(problem, partition)
        smaller, current = partition.for_vertex(1)
        labels_by_bucket = {bucket: [] for bucket in partition.buckets}
        labels_by_bucket[smaller].append(label(2.0, -5.0))
        best = {bucket: inf for bucket in partition.buckets}
        best[smaller] = -5.0

        self.assertTrue(
            SafeBucketDominance().dominated_in_smaller_buckets(
                label(7.0, -4.0),
                current,
                graph,
                labels_by_bucket,
                best,
            )
        )

    def test_r_best_stops_scan_when_no_smaller_label_can_dominate(self) -> None:
        instance, problem = problem_with_cut()
        partition = build_time_buckets(
            problem,
            {job_id: 14.0 for job_id in instance.job_ids},
            num_buckets_per_job=3,
        )
        graph = build_bucket_graph(problem, partition)
        first, second, third = partition.for_vertex(1)
        labels_by_bucket = {bucket: [] for bucket in partition.buckets}
        labels_by_bucket[first].append(label(2.0, 3.0))
        labels_by_bucket[second].append(label(6.0, 2.0))
        best = {bucket: inf for bucket in partition.buckets}
        best[first] = 3.0
        best[second] = 2.0

        self.assertFalse(
            SafeBucketDominance().dominated_in_smaller_buckets(
                label(10.0, 0.0),
                third,
                graph,
                labels_by_bucket,
                best,
            )
        )


class BucketForwardLabelingTests(unittest.TestCase):
    def test_bucket_labeling_matches_baseline_and_complete_enumeration(self) -> None:
        instance, problem = problem_with_cut()
        upper_bounds = {job_id: 10.0 for job_id in instance.job_ids}

        bucket_result = solve_bucket_forward_labeling(
            instance,
            problem,
            upper_bounds,
            num_buckets_per_job=4,
        )
        baseline_result = solve_forward_labeling(instance, problem)
        enumeration = [
            evaluate_schedule(instance, problem, jobs)
            for size in range(instance.num_jobs + 1)
            for jobs in combinations(instance.job_ids, size)
        ]
        best = min(enumeration, key=lambda evaluation: evaluation.reduced_cost)

        self.assertAlmostEqual(
            bucket_result.best_evaluation.reduced_cost,
            best.reduced_cost,
        )
        self.assertAlmostEqual(
            bucket_result.best_evaluation.reduced_cost,
            baseline_result.best_evaluation.reduced_cost,
        )
        self.assertTrue(bucket_result.is_exact)

    def test_disabling_bucket_dominance_preserves_exact_result(self) -> None:
        instance, problem = problem_with_cut()
        upper_bounds = {job_id: 10.0 for job_id in instance.job_ids}
        with_dominance = solve_bucket_forward_labeling(
            instance,
            problem,
            upper_bounds,
        )
        without_dominance = solve_bucket_forward_labeling(
            instance,
            problem,
            upper_bounds,
            use_bucket_dominance=False,
        )
        self.assertAlmostEqual(
            with_dominance.best_evaluation.reduced_cost,
            without_dominance.best_evaluation.reduced_cost,
        )


if __name__ == "__main__":
    unittest.main()
