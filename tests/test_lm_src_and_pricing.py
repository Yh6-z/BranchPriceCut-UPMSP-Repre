from __future__ import annotations

import unittest
from itertools import combinations

from src.BranchPriceCut.Cut import (
    LimitedMemorySubsetRowCut,
    separate_lm_subset_row_cuts,
    update_cut_pool,
)
from src.BranchPriceCut.Price import (
    add_lm_subset_row_cuts,
    rebuild_rmp_with_columns,
    solve_forward_labeling,
)
from src.model.PP import build_pricing_problems, evaluate_schedule
from src.model.RMP import DualSolution, build_restricted_master
from src.model.dataload import SchedulingInstance
from src.model.schedulemodel import build_schedule_column


def small_instance(num_machines: int = 1) -> SchedulingInstance:
    processing_times = (
        (2.0,) * num_machines,
        (3.0,) * num_machines,
        (1.0,) * num_machines,
        (4.0,) * num_machines,
    )
    return SchedulingInstance(
        num_machines=num_machines,
        num_jobs=4,
        instance_id=1,
        job_ids=(1, 2, 3, 4),
        machine_ids=tuple(range(1, num_machines + 1)),
        weights=(3, 2, 4, 1),
        processing_times=processing_times,
    )


class LimitedMemorySubsetRowCutTests(unittest.TestCase):
    def test_memory_reset_changes_column_coefficient(self) -> None:
        instance = small_instance()
        column = build_schedule_column(instance, 1, (1, 4, 2), enforce_swpt=False)
        full_memory = LimitedMemorySubsetRowCut(
            "full", (1, 2, 3), (frozenset((1, 2, 3, 4)),)
        )
        limited_memory = LimitedMemorySubsetRowCut(
            "limited", (1, 2, 3), (frozenset((1, 2, 3)),)
        )
        self.assertEqual(full_memory.coefficient(column), 1.0)
        self.assertEqual(limited_memory.coefficient(column), 0.0)

    def test_separation_and_rmp_row_generation(self) -> None:
        instance = small_instance(num_machines=2)
        columns = (
            build_schedule_column(instance, 1, (1, 2)),
            build_schedule_column(instance, 1, (1, 3)),
            build_schedule_column(instance, 2, (2, 3)),
        )
        rmp = build_restricted_master(instance, columns)
        cuts = separate_lm_subset_row_cuts(
            rmp,
            (0.5, 0.5, 0.5),
            max_added=1,
        )
        self.assertEqual(cuts[0].subset, (1, 2, 3))
        self.assertAlmostEqual(cuts[0].violation, 0.5)

        cut_rmp = add_lm_subset_row_cuts(rmp, cuts)
        self.assertEqual(cut_rmp.cut_rows[0].coefficients, (1.0, 1.0, 1.0))
        self.assertEqual(cut_rmp.cut_rows[0].rhs, 1.0)

        extra = build_schedule_column(instance, 2, (1, 2, 3))
        rebuilt = rebuild_rmp_with_columns(cut_rmp, (*columns, extra))
        self.assertEqual(rebuilt.cut_rows[0].coefficients, (1.0, 1.0, 1.0, 1.0))

    def test_cut_pool_expands_memory_for_an_existing_subset(self) -> None:
        first = LimitedMemorySubsetRowCut(
            "c", (1, 2, 3), (frozenset((1, 2, 3)),)
        )
        second = LimitedMemorySubsetRowCut(
            "new-id", (1, 2, 3), (frozenset((1, 2, 3, 4)),)
        )
        update = update_cut_pool((first,), (second,))
        self.assertEqual(update.cuts[0].memory_sets[0], frozenset((1, 2, 3, 4)))
        self.assertEqual(update.added_cut_ids, ())
        self.assertEqual(update.expanded_cut_ids, ("c",))


class ForwardPricingTests(unittest.TestCase):
    def test_exact_labeling_matches_complete_enumeration(self) -> None:
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
        problem = build_pricing_problems(rmp, dual)[0]
        result = solve_forward_labeling(instance, problem)

        enumeration = []
        for size in range(instance.num_jobs + 1):
            for jobs in combinations(instance.job_ids, size):
                enumeration.append(evaluate_schedule(instance, problem, jobs))
        best = min(enumeration, key=lambda evaluation: evaluation.reduced_cost)

        self.assertAlmostEqual(
            result.best_evaluation.reduced_cost,
            best.reduced_cost,
        )
        self.assertTrue(result.is_exact)


if __name__ == "__main__":
    unittest.main()
