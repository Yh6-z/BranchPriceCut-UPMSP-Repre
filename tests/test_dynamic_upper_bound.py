from __future__ import annotations

import unittest
from itertools import combinations

from src.BranchPriceCut.Price import (
    calculate_global_completion_upper_bound,
    delete_dag_arcs_and_update_bounds,
    initialize_dynamic_upper_bounds,
    solve_bucket_forward_labeling_with_bounds,
)
from src.model.PP import build_pricing_problems, evaluate_schedule
from src.model.RMP import DualSolution, build_restricted_master
from src.model.dataload import SchedulingInstance


def dsub_instance() -> SchedulingInstance:
    return SchedulingInstance(
        num_machines=1,
        num_jobs=4,
        instance_id=1,
        job_ids=(1, 2, 3, 4),
        machine_ids=(1,),
        weights=(1, 1, 1, 1),
        processing_times=((2.0,), (3.0,), (1.0,), (4.0,)),
    )


def dsub_problem():
    instance = dsub_instance()
    rmp = build_restricted_master(instance, ())
    dual = DualSolution(
        job_duals=(6.0, 7.0, 4.0, 8.0),
        machine_duals=(0.0,),
    )
    return instance, build_pricing_problems(rmp, dual)[0]


class DynamicUpperBoundTests(unittest.TestCase):
    def test_initial_recurrence_follows_swpt_order(self) -> None:
        _, problem = dsub_problem()
        state = initialize_dynamic_upper_bounds(problem, global_upper_bound=10.0)

        self.assertEqual(problem.swpt_job_order, (3, 1, 2, 4))
        self.assertEqual(
            tuple(state.bound(vertex).upper_bound for vertex in problem.dag_vertices),
            (0.0, 1.0, 3.0, 6.0, 10.0, 10.0),
        )
        self.assertEqual(state.bound(1).max_predecessor, 3)
        self.assertEqual(state.bound(2).max_predecessor, 1)

    def test_removing_decisive_arc_shrinks_downstream_bounds(self) -> None:
        _, problem = dsub_problem()
        state = initialize_dynamic_upper_bounds(problem, global_upper_bound=10.0)
        update = delete_dag_arcs_and_update_bounds(problem, state, ((3, 1),))

        self.assertEqual(update.removed_arcs, ((3, 1),))
        self.assertEqual(update.state.bound(1).upper_bound, 2.0)
        self.assertEqual(update.state.bound(2).upper_bound, 5.0)
        self.assertEqual(update.state.bound(4).upper_bound, 9.0)
        self.assertEqual(update.changed_vertices, (1, 2, 4, problem.sink_node))
        self.assertNotIn((3, 1), update.state.dag_arcs)

    def test_removing_nondecisive_arc_does_not_change_bounds(self) -> None:
        _, problem = dsub_problem()
        state = initialize_dynamic_upper_bounds(problem, global_upper_bound=10.0)
        update = delete_dag_arcs_and_update_bounds(
            problem,
            state,
            ((problem.source_node, 1),),
        )
        self.assertFalse(update.upper_bounds_updated)

    def test_unreachable_vertex_is_reported_after_all_incoming_arcs_removed(self) -> None:
        _, problem = dsub_problem()
        state = initialize_dynamic_upper_bounds(problem, global_upper_bound=10.0)
        incoming_to_three = tuple(
            arc for arc in state.dag_arcs if arc[1] == 3
        )
        update = delete_dag_arcs_and_update_bounds(
            problem,
            state,
            incoming_to_three,
        )
        self.assertIn(3, update.unreachable_vertices)
        self.assertFalse(update.state.bound(3).reachable)

    def test_global_upper_bound_matches_paper_formula(self) -> None:
        instance = dsub_instance()
        self.assertEqual(calculate_global_completion_upper_bound(instance), 14.0)

    def test_dsub_local_graph_pricing_matches_feasible_enumeration(self) -> None:
        instance, problem = dsub_problem()
        state = initialize_dynamic_upper_bounds(problem, global_upper_bound=10.0)
        update = delete_dag_arcs_and_update_bounds(problem, state, ((3, 1),))
        result = solve_bucket_forward_labeling_with_bounds(
            instance,
            problem,
            update.state,
            num_buckets_per_job=3,
        )

        feasible = []
        for size in range(instance.num_jobs + 1):
            for jobs in combinations(instance.job_ids, size):
                evaluation = evaluate_schedule(instance, problem, jobs)
                if (3, 1) not in evaluation.column.arcs:
                    feasible.append(evaluation)
        best = min(feasible, key=lambda evaluation: evaluation.reduced_cost)
        self.assertAlmostEqual(result.best_evaluation.reduced_cost, best.reduced_cost)


if __name__ == "__main__":
    unittest.main()
