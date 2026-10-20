from __future__ import annotations

import unittest

from config import get_config
from src.BranchPriceCut.Price import build_bucket_graph, build_time_buckets
from src.model.PP import build_pricing_problems
from src.model.RMP import DualSolution, build_restricted_master
from src.model.dataload import SchedulingInstance


def bucket_instance() -> SchedulingInstance:
    return SchedulingInstance(
        num_machines=1,
        num_jobs=3,
        instance_id=1,
        job_ids=(1, 2, 3),
        machine_ids=(1,),
        weights=(1, 1, 1),
        processing_times=((2.0,), (3.0,), (1.0,)),
    )


def pricing_problem():
    instance = bucket_instance()
    rmp = build_restricted_master(instance, ())
    dual = DualSolution(job_duals=(0.0, 0.0, 0.0), machine_duals=(0.0,))
    return build_pricing_problems(rmp, dual)[0]


class AccelerationSwitchTests(unittest.TestCase):
    def test_acceleration_switches_default_to_true_and_can_be_disabled(self) -> None:
        defaults = get_config([])
        for name in (
            "use_time_bucket",
            "use_bucket_graph",
            "use_bucket_dominance",
            "use_DSUB",
            "use_backward_labeling",
            "use_label_joining",
        ):
            self.assertTrue(getattr(defaults, name))

        disabled = get_config(
            [
                "--no-use-time-bucket",
                "--no-use-bucket-graph",
                "--no-use-bucket-dominance",
                "--no-use-dsub",
                "--no-use-backward-labeling",
                "--no-use-label-joining",
            ]
        )
        for name in (
            "use_time_bucket",
            "use_bucket_graph",
            "use_bucket_dominance",
            "use_DSUB",
            "use_backward_labeling",
            "use_label_joining",
        ):
            self.assertFalse(getattr(disabled, name))


class TimeBucketTests(unittest.TestCase):
    def test_equal_width_buckets_cover_continuous_completion_window(self) -> None:
        problem = pricing_problem()
        partition = build_time_buckets(
            problem,
            {1: 8.0, 2: 9.0, 3: 5.0},
            num_buckets_per_job=2,
        )

        job_one = partition.for_vertex(1)
        self.assertEqual(
            tuple((bucket.lower_bound, bucket.upper_bound) for bucket in job_one),
            ((2.0, 5.0), (5.0, 8.0)),
        )
        self.assertEqual(partition.bucket_for_time(1, 5.0).index, 1)
        self.assertEqual(partition.bucket_for_time(1, 8.0).index, 1)
        self.assertEqual(len(partition.for_vertex(problem.source_node)), 1)
        self.assertEqual(len(partition.for_vertex(problem.sink_node)), 1)

    def test_zero_width_window_uses_one_effective_bucket(self) -> None:
        problem = pricing_problem()
        partition = build_time_buckets(
            problem,
            {1: 2.0, 2: 3.0, 3: 1.0},
            num_buckets_per_job=4,
        )
        self.assertTrue(
            all(len(partition.for_vertex(job_id)) == 1 for job_id in problem.job_ids)
        )


class BucketGraphTests(unittest.TestCase):
    def test_graph_lifts_every_dag_arc_to_tail_buckets(self) -> None:
        problem = pricing_problem()
        partition = build_time_buckets(
            problem,
            {1: 8.0, 2: 9.0, 3: 5.0},
            num_buckets_per_job=2,
        )
        graph = build_bucket_graph(problem, partition)

        expected_count = sum(
            len(partition.for_vertex(tail))
            for tail, _ in problem.dag_arcs
        )
        self.assertEqual(len(graph.arcs), expected_count)
        self.assertEqual(
            {arc.dag_arc for arc in graph.arcs},
            set(problem.dag_arcs),
        )
        self.assertEqual(len(graph.adjacent_smaller_buckets), 3)

        source_to_job_three = next(
            arc
            for arc in graph.arcs
            if arc.dag_arc == (problem.source_node, 3)
        )
        self.assertEqual(source_to_job_three.head_bucket.index, 0)

        later_job_three_bucket = partition.for_vertex(3)[1]
        three_to_one = next(
            arc
            for arc in graph.outgoing(later_job_three_bucket)
            if arc.dag_arc == (3, 1)
        )
        self.assertEqual(three_to_one.head_bucket.index, 1)
        self.assertEqual(graph.locate(1, 8.0).index, 1)

    def test_rebuild_returns_new_graph_without_mutating_previous_partition(self) -> None:
        problem = pricing_problem()
        first_partition = build_time_buckets(
            problem,
            {1: 8.0, 2: 9.0, 3: 5.0},
            num_buckets_per_job=2,
        )
        first_graph = build_bucket_graph(problem, first_partition)
        second_partition = build_time_buckets(
            problem,
            {1: 6.0, 2: 7.0, 3: 4.0},
            num_buckets_per_job=2,
        )
        second_graph = build_bucket_graph(problem, second_partition)

        self.assertEqual(first_graph.partition.for_vertex(1)[0].upper_bound, 5.0)
        self.assertEqual(second_graph.partition.for_vertex(1)[0].upper_bound, 4.0)

    def test_graph_omits_bucket_arc_beyond_head_completion_upper_bound(self) -> None:
        problem = pricing_problem()
        partition = build_time_buckets(
            problem,
            {1: 4.0, 2: 5.0, 3: 5.0},
            num_buckets_per_job=2,
        )
        graph = build_bucket_graph(problem, partition)
        late_job_three_bucket = partition.for_vertex(3)[1]

        self.assertFalse(
            any(
                arc.dag_arc == (3, 1)
                for arc in graph.outgoing(late_job_three_bucket)
            )
        )


if __name__ == "__main__":
    unittest.main()
