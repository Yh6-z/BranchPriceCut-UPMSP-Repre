"""用小规模集合划分实例检查两种调度枚举模式和局部节点约束。"""

from __future__ import annotations

import unittest

from config import get_config
from src.BranchPriceCut.Branch.branching import (
    BranchCandidate,
    create_child_node,
    make_branch_node,
)
from src.BranchPriceCut.Price.dynamic_upper_bound import initialize_dynamic_upper_bounds
from src.BranchPriceCut.accelerate.enumeration import solve_schedule_enumeration
from src.model.PP import build_pricing_problems, evaluate_column_reduced_cost
from src.model.RMP import DualSolution, build_restricted_master
from src.model.dataload import SchedulingInstance
from src.model.schedulemodel import build_schedule_column


def enumeration_fixture():
    """构造当前列池只含一条双作业调度的两机器节点。

    输入：无。
    输出：实例 ``SchedulingInstance``、节点 ``BranchNode`` 和对偶 ``DualSolution``。
    """

    instance = SchedulingInstance(
        num_machines=2,
        num_jobs=2,
        instance_id=1,
        job_ids=(1, 2),
        machine_ids=(1, 2),
        weights=(1, 1),
        processing_times=((1.0, 1.0), (1.0, 1.0)),
    )
    columns = (build_schedule_column(instance, 1, (1, 2)),)
    rmp = build_restricted_master(instance, columns)
    dual = DualSolution((1.0, 1.0), (0.0, 0.0))
    problems = build_pricing_problems(rmp, dual)
    states = tuple(initialize_dynamic_upper_bounds(problem, 10.0)
                   for problem in problems)
    return instance, make_branch_node(rmp, problems, states), dual


class ScheduleEnumerationTests(unittest.TestCase):
    def test_mode_zero_uses_existing_columns_only(self) -> None:
        """检查模式 0 不产生新调度。

        输入：当前 ``unittest.TestCase`` 与内置有限列节点。
        输出：无；通过断言核验 ``ScheduleEnumerationResult``。
        """

        _, node, _ = enumeration_fixture()
        result = solve_schedule_enumeration(node, mode=0, incumbent_objective=4.0)
        self.assertEqual(result.objective_value, 3.0)
        self.assertEqual(result.added_columns, ())
        self.assertEqual(result.machine_searches, ())
        self.assertEqual(result.selected_columns, node.rmp.columns)

    def test_empty_current_pool_is_not_reported_as_infeasible_node(self) -> None:
        """检查空列池只报告没有有限列整数解。

        输入：当前 ``unittest.TestCase`` 与空列 ``BranchNode``。
        输出：无；通过断言核验目标值与选中列。
        """

        instance, node, _ = enumeration_fixture()
        empty_rmp = build_restricted_master(instance, ())
        empty_node = make_branch_node(empty_rmp, node.pricing_problems, node.bounds)
        result = solve_schedule_enumeration(empty_node, mode=0)
        self.assertIsNone(result.objective_value)
        self.assertEqual(result.selected_columns, ())

    def test_mode_one_adds_positive_reduced_cost_columns(self) -> None:
        """检查标签搜索可补充严格正约化成本的单作业列。

        输入：当前 ``unittest.TestCase``、有限列节点、对偶与上下界。
        输出：无；通过断言核验新增列及整数目标。
        """

        _, node, _ = enumeration_fixture()
        dual = DualSolution((0.5, 0.5), (0.0, 0.0))
        result = solve_schedule_enumeration(
            node, mode=1, dual_solution=dual,
            lower_bound=1.0, incumbent_objective=3.0,
        )
        self.assertEqual(result.objective_value, 2.0)
        self.assertTrue(result.improves_incumbent)
        self.assertTrue(result.enumeration_complete)
        self.assertIn((2, (2,)), {column.key for column in result.added_columns})
        extra = next(column for column in result.added_columns if column.key == (2, (2,)))
        request = build_pricing_problems(node.rmp, dual)[1]
        self.assertGreater(evaluate_column_reduced_cost(request, extra).reduced_cost, 0.0)
        self.assertEqual(len(result.selected_columns), 2)

    def test_mode_one_retains_all_small_paths_below_threshold(self) -> None:
        """对照完整小图路径集，检查枚举没有定价支配造成的遗漏。

        输入：当前 ``unittest.TestCase``、两机器两作业节点和宽阈值。
        输出：无；通过断言核验每台机器全部非空调度键。
        """

        _, node, dual = enumeration_fixture()
        result = solve_schedule_enumeration(
            node, mode=1, dual_solution=dual,
            lower_bound=2.0, incumbent_objective=3.0,
        )
        self.assertEqual(
            {column.key for column in result.columns},
            {(machine_id, jobs)
             for machine_id in (1, 2)
             for jobs in ((1,), (2,), (1, 2))},
        )

    def test_budget_and_branch_rows_limit_claims_and_ip(self) -> None:
        """检查预算和分支行在模式 1 的行为。

        输入：当前 ``unittest.TestCase``、小预算与 x=1 子节点。
        输出：无；通过断言核验完整性及分支约束。
        """

        _, node, dual = enumeration_fixture()
        limited = solve_schedule_enumeration(
            node, mode=1, dual_solution=dual,
            lower_bound=2.0, incumbent_objective=3.0,
            max_labels_per_machine=2,
        )
        self.assertFalse(limited.enumeration_complete)
        self.assertEqual(limited.objective_value, 3.0)

        child = create_child_node(node, BranchCandidate("x", 1, 2, 0.5, 1), 1)
        child_dual = DualSolution(
            dual.job_duals, dual.machine_duals,
            cut_ids=tuple(row.cut_id for row in child.rmp.cut_rows),
            cut_duals=(0.0,),
        )
        branched = solve_schedule_enumeration(
            child, mode=1, dual_solution=child_dual,
            lower_bound=2.0, incumbent_objective=3.0,
        )
        self.assertEqual(branched.objective_value, 3.0)
        self.assertTrue(all((1, 2) in column.arcs
                            for column in branched.selected_columns))

        stale = solve_schedule_enumeration(
            child, mode=1, dual_solution=dual,
            lower_bound=2.0, incumbent_objective=3.0,
        )
        self.assertEqual(stale.status, "skipped_unsupported_snapshot")

    def test_configuration_modes(self) -> None:
        """检查命令行启动模式和标签预算。

        输入：当前 ``unittest.TestCase`` 与 0/1 命令行参数。
        输出：无；通过断言核验 ``argparse.Namespace``。
        """

        self.assertEqual(get_config([]).schedule_enumeration_mode, 0)
        self.assertEqual(get_config(["--schedule-enumeration-mode", "1"]).schedule_enumeration_mode, 1)
        self.assertEqual(get_config([]).schedule_enumeration_max_labels, 10000)


if __name__ == "__main__":
    unittest.main()
