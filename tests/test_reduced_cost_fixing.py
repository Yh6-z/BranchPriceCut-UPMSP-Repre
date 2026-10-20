"""用小型完整路径枚举核对约化成本固定的安全性。"""

from __future__ import annotations

import importlib
import unittest
from dataclasses import replace
from itertools import combinations
from math import inf
from random import Random

from config import get_config
from src.BranchPriceCut.Cut.lm_subset_row_cut import LimitedMemorySubsetRowCut
from src.BranchPriceCut.Price.dynamic_upper_bound import (
    delete_dag_arcs_and_update_bounds,
    initialize_dynamic_upper_bounds,
)
from src.model.PP import build_pricing_problems, evaluate_schedule
from src.model.RMP import DualSolution, build_restricted_master
from src.model.dataload import SchedulingInstance


fixing = importlib.import_module(
    "src.BranchPriceCut.accelerate.reduced-cost-fixing.variable_fixing"
)
backward_labeling = importlib.import_module(
    "src.BranchPriceCut.accelerate.reduced-cost-fixing.backward_labeling"
)


def small_problem(with_cut: bool = False):
    """输入是否启用 lm-SRC ``bool``，输出实例、定价请求及初始 DSUB。"""

    instance = SchedulingInstance(
        num_machines=1,
        num_jobs=3,
        instance_id=1,
        job_ids=(1, 2, 3),
        machine_ids=(1,),
        weights=(1, 1, 1),
        processing_times=((1.0,), (2.0,), (3.0,)),
    )
    cuts = (
        LimitedMemorySubsetRowCut(
            cut_id="c123",
            subset=(1, 2, 3),
            memory_sets=(frozenset((1, 2, 3)),),
        ),
    ) if with_cut else ()
    rmp = build_restricted_master(instance, (), cut_definitions=cuts)
    dual = DualSolution(
        job_duals=(3.0, 5.0, 6.0),
        machine_duals=(-3.0 if with_cut else -4.0,),
        cut_ids=("c123",) if with_cut else (),
        cut_duals=(-1.0,) if with_cut else (),
    )
    problem = build_pricing_problems(rmp, dual)[0]
    state = initialize_dynamic_upper_bounds(problem, 10.0)
    return instance, problem, state


def enumerate_arc_costs(instance, problem, state):
    """输入小实例及局部图，输出 ``dict[arc, min reduced cost]``。"""

    by_arc = {arc: inf for arc in state.dag_arcs}
    active = frozenset(state.dag_arcs)
    for size in range(instance.num_jobs + 1):
        for jobs in combinations(instance.job_ids, size):
            evaluation = evaluate_schedule(instance, problem, jobs)
            if not active.issuperset(evaluation.column.arcs):
                continue
            if any(
                completion > state.bound(job_id).upper_bound
                for job_id, completion in zip(
                    evaluation.column.job_ids,
                    evaluation.column.completion_times,
                )
            ):
                continue
            for arc in evaluation.column.arcs:
                by_arc[arc] = min(by_arc[arc], evaluation.reduced_cost)
    return by_arc


class ReducedCostFixingTests(unittest.TestCase):
    def test_fixing_configuration_can_be_disabled(self) -> None:
        """检查根配置中的固定开关、间隙阈值和容差。

        输入：三个约化成本固定命令行选项 ``str``。
        输出：断言 ``argparse.Namespace`` 含指定配置值。
        """

        options = get_config([
            "--no-use-reduced-cost-fixing",
            "--reduced-cost-fixing-gap", "0.2",
            "--reduced-cost-fixing-tolerance", "0.0001",
        ])
        self.assertFalse(options.use_reduced_cost_fixing)
        self.assertEqual(options.reduced_cost_fixing_gap, 0.2)
        self.assertEqual(options.reduced_cost_fixing_tolerance, 0.0001)

    def test_all_arc_decisions_match_complete_enumeration(self) -> None:
        """核对无割时所有弧的固定结论。

        输入：由 ``small_problem`` 构造的三作业算例。
        输出：断言与逐弧完整枚举一致。
        """

        instance, problem, state = small_problem()
        result = fixing.fix_arcs_by_reduced_cost(
            instance, problem, state,
            lower_bound=10.0, upper_bound=10.0,
            full_master_dual_certified=True,
            relative_gap_limit=0.1, fixing_tolerance=1e-7,
            use_completion_pruning=False,
        )
        truth = enumerate_arc_costs(instance, problem, state)
        self.assertEqual(result.status, "applied")
        self.assertEqual(
            set(result.removed_arcs),
            {arc for arc, value in truth.items() if value > 1e-7},
        )
        for assessment in result.assessments:
            self.assertLessEqual(
                assessment.optimistic_lower_bound,
                truth[assessment.arc] + 1e-9,
            )
        self.assertIn((1, 2), result.graph_update.state.dag_arcs)
        self.assertNotIn((0, 3), result.graph_update.state.dag_arcs)

    def test_lm_src_join_and_completion_pruning_match_enumeration(self) -> None:
        """核对有割时的跨界拼接及完成界剪枝。

        输入：三作业 lm-SRC 算例和两种剪枝开关。
        输出：断言两种结果均与完整枚举一致。
        """

        instance, problem, state = small_problem(with_cut=True)
        truth = enumerate_arc_costs(instance, problem, state)
        for use_pruning in (False, True):
            result = fixing.fix_arcs_by_reduced_cost(
                instance, problem, state,
                lower_bound=10.0, upper_bound=10.0,
                full_master_dual_certified=True,
                relative_gap_limit=0.1, fixing_tolerance=1e-7,
                use_completion_pruning=use_pruning,
            )
            self.assertEqual(
                set(result.removed_arcs),
                {arc for arc, value in truth.items() if value > 1e-7},
            )
            joined = next(
                item for item in result.assessments if item.arc == (1, 2)
            )
            self.assertFalse(joined.can_delete)

    def test_crossing_lm_src_state_is_added_at_join(self) -> None:
        """检查跨前后缀边界产生的 lm-SRC 系数。

        输入：前缀作业 1 与后缀作业 2 的标签。
        输出：断言完整路径约化成本比两段独立和多一个割贡献。
        """

        from src.BranchPriceCut.Price.forward_labeling import collect_forward_labels

        instance, problem, state = small_problem(with_cut=True)
        forward = collect_forward_labels(instance, problem)
        backward = backward_labeling.collect_backward_labels(problem, state)
        prefix = next(
            label for label in forward.labels_at(1) if label.job_ids == (1,)
        )
        suffix = next(
            label for label in backward.labels_at(2) if label.job_ids == (2,)
        )
        direct = evaluate_schedule(instance, problem, (1, 2)).reduced_cost
        independent = (
            prefix.reduced_cost
            + suffix.base_reduced_cost
            + prefix.time * suffix.cumulative_weight
        )
        self.assertAlmostEqual(direct - independent, 1.0)

    def test_local_graph_and_column_filter_follow_fixed_arcs(self) -> None:
        """核对分支后局部图、列过滤及 DSUB 更新。

        输入：已删除一条分支弧的局部状态和两个调度列。
        输出：断言固定结果保留合法列、过滤不相容列。
        """

        instance, problem, state = small_problem()
        branch_update = delete_dag_arcs_and_update_bounds(
            problem, state, ((1, 3),)
        )
        bad = evaluate_schedule(instance, problem, (3,)).column
        good = evaluate_schedule(instance, problem, (1, 2, 3)).column
        result = fixing.fix_arcs_by_reduced_cost(
            instance, problem, branch_update.state,
            lower_bound=10.0, upper_bound=10.0,
            full_master_dual_certified=True,
            relative_gap_limit=0.1, fixing_tolerance=1e-7,
            columns=(bad, good),
        )
        self.assertNotIn((1, 3), result.graph_update.state.dag_arcs)
        self.assertIn(bad, result.removed_columns)
        self.assertIn(good, result.retained_columns)
        self.assertEqual(
            result.graph_update.state.bound(3).upper_bound,
            branch_update.state.bound(3).upper_bound,
        )

    def test_missing_certificate_or_wide_gap_does_not_change_graph(self) -> None:
        """核对缺证书及大间隙时跳过固定。

        输入：三作业算例及两种不满足触发条件的参数。
        输出：断言图保持不变且返回跳过状态。
        """

        instance, problem, state = small_problem()
        missing = fixing.fix_arcs_by_reduced_cost(
            instance, problem, state,
            lower_bound=10.0, upper_bound=10.0,
            full_master_dual_certified=False,
            relative_gap_limit=0.1, fixing_tolerance=1e-7,
        )
        wide = fixing.fix_arcs_by_reduced_cost(
            instance, problem, state,
            lower_bound=0.0, upper_bound=10.0,
            full_master_dual_certified=True,
            relative_gap_limit=0.1, fixing_tolerance=1e-7,
        )
        self.assertEqual(missing.status, "skipped_no_certificate")
        self.assertEqual(wide.status, "skipped_gap")
        self.assertIsNone(missing.graph_update)
        self.assertEqual(state.dag_arcs, small_problem()[2].dag_arcs)

    def test_varied_duals_and_memory_sets_never_remove_affordable_path(self) -> None:
        """用不同对偶与记忆集检查安全删弧不变量。

        输入：固定随机种子产生的十二组四作业实例。
        输出：断言每条被删弧的真实最小约化成本均超过间隙。
        """

        rng = Random(19)
        for _ in range(12):
            processing = tuple(float(rng.randint(1, 4)) for _ in range(4))
            weights = tuple(rng.randint(1, 3) for _ in range(4))
            instance = SchedulingInstance(
                num_machines=1,
                num_jobs=4,
                instance_id=1,
                job_ids=(1, 2, 3, 4),
                machine_ids=(1,),
                weights=weights,
                processing_times=tuple((value,) for value in processing),
            )
            cut = LimitedMemorySubsetRowCut(
                cut_id="c123",
                subset=(1, 2, 3),
                memory_sets=(
                    frozenset((1, 2, 3)).union(rng.choice(((), (4,)))),
                ),
            )
            rmp = build_restricted_master(instance, (), cut_definitions=(cut,))
            job_duals = tuple(float(rng.randint(1, 7)) for _ in range(4))
            provisional = build_pricing_problems(
                rmp,
                DualSolution(
                    job_duals=job_duals,
                    machine_duals=(0.0,),
                    cut_ids=("c123",),
                    cut_duals=(-1.0,),
                ),
            )[0]
            all_evaluations = tuple(
                evaluate_schedule(instance, provisional, jobs)
                for size in range(5)
                for jobs in combinations(instance.job_ids, size)
            )
            machine_dual = min(item.reduced_cost for item in all_evaluations)
            problem = build_pricing_problems(
                rmp,
                DualSolution(
                    job_duals=job_duals,
                    machine_duals=(machine_dual,),
                    cut_ids=("c123",),
                    cut_duals=(-1.0,),
                ),
            )[0]
            upper_bound = evaluate_schedule(
                instance, problem, instance.job_ids
            ).column.cost
            lower_bound = sum(job_duals) + machine_dual - cut.rhs
            state = initialize_dynamic_upper_bounds(problem, sum(processing))
            truth = enumerate_arc_costs(instance, problem, state)
            result = fixing.fix_arcs_by_reduced_cost(
                instance, problem, state,
                lower_bound=lower_bound, upper_bound=upper_bound,
                full_master_dual_certified=True,
                relative_gap_limit=2.0, fixing_tolerance=1e-7,
            )
            self.assertEqual(result.status, "applied")
            self.assertTrue(
                all(truth[arc] > result.gap + 1e-7 for arc in result.removed_arcs)
            )

    def test_stale_graph_or_unsupported_cut_skips_fixing(self) -> None:
        """检查图快照不符或割对偶符号不支持时保留全部弧。

        输入：不匹配的机器图和正割对偶定价请求。
        输出：断言均返回跳过状态及原列池。
        """

        instance, problem, state = small_problem(with_cut=True)
        stale = fixing.fix_arcs_by_reduced_cost(
            instance, problem, replace(state, machine_id=2),
            lower_bound=10.0, upper_bound=10.0,
            full_master_dual_certified=True,
            relative_gap_limit=0.1, fixing_tolerance=1e-7,
        )
        unsupported = fixing.fix_arcs_by_reduced_cost(
            instance, replace(problem, cut_duals=(1.0,)), state,
            lower_bound=10.0, upper_bound=10.0,
            full_master_dual_certified=True,
            relative_gap_limit=0.1, fixing_tolerance=1e-7,
        )
        self.assertEqual(stale.status, "skipped_snapshot")
        self.assertEqual(unsupported.status, "skipped_unsupported_cuts")


if __name__ == "__main__":
    unittest.main()
