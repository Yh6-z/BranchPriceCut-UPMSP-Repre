"""验证阶段记录与证书隔离、父子下界继承及 incumbent 合法性。"""

import unittest
from dataclasses import replace

from src.BranchPriceCut.Branch import (
    BranchCandidate, SolveInformation, create_child_node, frontier_lower_bound,
    make_branch_node, node_pruning_reason, record_incumbent, record_node_stage,
    register_node_information, validate_integer_schedule,
)
from src.BranchPriceCut.Price.dynamic_upper_bound import initialize_dynamic_upper_bounds
from src.BranchPriceCut.accelerate.RootNodeColumnInitial import initialize_root_columns
from src.BranchPriceCut.accelerate.enumeration import solve_schedule_enumeration
from src.BranchPriceCut.solve import solve_restricted_master
from src.model.PP import build_pricing_problems
from src.model.RMP import DualSolution, build_restricted_master
from src.model.dataload import SchedulingInstance
from src.model.schedulemodel import build_schedule_column


def information_fixture():
    """构造单条已知调度的节点和空求解记录。

    输入：无。
    输出：SchedulingInstance、BranchNode、SolveInformation 三元组。
    """

    instance = SchedulingInstance(2, 2, 1, (1, 2), (1, 2), (1, 1),
                                  ((1.0, 1.0), (1.0, 1.0)))
    column = build_schedule_column(instance, 1, (1, 2))
    rmp = build_restricted_master(instance, (column,))
    problems = build_pricing_problems(rmp, DualSolution((0.0, 0.0), (0.0, 0.0)))
    bounds = tuple(initialize_dynamic_upper_bounds(problem, 10.0) for problem in problems)
    node = make_branch_node(rmp, problems, bounds)
    info = register_node_information(SolveInformation(instance), node, 0)
    return instance, node, info


class SolveInformationTests(unittest.TestCase):
    def test_rmp_solution_is_not_a_proven_bound_without_exact_pricing(self) -> None:
        """保存真实 LP 解，但不把缺列 RMP 的目标当作节点证明下界。

        输入：内置节点；RMP 目标为 3，完整主问题可达到 2。
        输出：无；断言目标/对偶被记录而下界和剪枝仍为空。
        """

        _, node, info = information_fixture()
        lp = solve_restricted_master(node.rmp)
        recorded = record_node_stage(info, 0, node, "rmp", solution=lp, elapsed_seconds=0.25)
        event = recorded.nodes[0].stages[0]
        self.assertEqual(event.solution.objective_value, 3.0)
        self.assertEqual(event.solution.column_values, (1.0,))
        self.assertEqual(event.elapsed_seconds, 0.25)
        self.assertIsNone(recorded.nodes[0].lower_bound)
        self.assertIsNone(node_pruning_reason(recorded.nodes[0], 2.0))
        self.assertEqual(info.nodes[0].stages, ())

    def test_proven_bound_and_child_history_keep_independent_snapshots(self) -> None:
        """完整列定价证书才能登记 LP 下界，子节点只继承下界。

        输入：两机器的完整非空列集、真实最优 LP、分支子节点。
        输出：无；断言下界、父子身份、阶段列/图统计及历史隔离。
        """

        instance, node, info = information_fixture()
        columns = tuple(build_schedule_column(instance, machine, jobs)
                        for machine in instance.machine_ids for jobs in ((1,), (2,), (1, 2)))
        complete = replace(node, rmp=build_restricted_master(instance, columns))
        lp = solve_restricted_master(complete.rmp)
        info = record_node_stage(info, 0, complete, "exact_pricing", solution=lp,
                                 exact_pricing_complete=True, elapsed_seconds=0.5)
        root = info.nodes[0]
        self.assertEqual(root.lower_bound, 2.0)
        self.assertEqual(root.stages[0].added_columns, 5)
        self.assertEqual(node_pruning_reason(root, 2.0), "bound")
        child = create_child_node(complete, BranchCandidate("x", 1, 2, 0.5, 1), 0)
        info = register_node_information(info, child, 1, parent_id=0)
        self.assertEqual(info.nodes[1].lower_bound, 2.0)
        self.assertEqual((info.nodes[1].parent_id, info.nodes[1].depth), (0, 1))
        self.assertEqual(info.nodes[1].stages, ())
        info = record_node_stage(info, 0, child, "test_snapshot_change")
        self.assertEqual(info.nodes[0].stages[1].removed_columns, 1)
        self.assertEqual(info.nodes[0].stages[1].removed_dag_arcs, 1)
        self.assertEqual(info.nodes[0].stages[0].node, complete)
        self.assertIsNone(info.nodes[0].stages[1].solution)

    def test_incumbent_rejects_overlap_wrong_cost_and_branch_violation(self) -> None:
        """核验完整调度后才记录 UB，不能直接接受整数 y 或伪造目标。

        输入：初始调度、重叠作业、错误成本、x=0 子节点及更优拆分调度。
        输出：无；断言仅真实合法改善进入 incumbent 历史。
        """

        instance, node, info = information_fixture()
        initial = initialize_root_columns(instance)
        info = record_incumbent(info, node.rmp.columns, source_node_id=0)
        self.assertEqual(info.upper_bound, 3.0)
        overlap = (*node.rmp.columns, build_schedule_column(instance, 2, (1,)))
        self.assertIs(record_incumbent(info, overlap), info)
        forged = (replace(node.rmp.columns[0], cost=0.0),)
        self.assertIs(record_incumbent(info, forged), info)
        child = create_child_node(node, BranchCandidate("x", 1, 2, 0.5, 1), 0)
        info = register_node_information(info, child, 1, parent_id=0)
        self.assertFalse(validate_integer_schedule(instance, node.rmp.columns, node=child))
        self.assertIs(record_incumbent(info, node.rmp.columns, source_node_id=1), info)
        improved = record_incumbent(info, initial.columns, source_node_id=1, elapsed_seconds=1.25)
        self.assertEqual(improved.upper_bound, 2.0)
        self.assertEqual(len(improved.incumbent_history), 2)
        self.assertEqual(improved.incumbent_history[-1].source_node_id, 1)
        self.assertEqual(improved.incumbent_history[-1].elapsed_seconds, 1.25)
        self.assertIsNone(improved.nodes[1].lower_bound)

    def test_finite_ip_optimum_and_infeasible_rmp_do_not_prove_node(self) -> None:
        """有限列 IP 仅更新 UB；有限列不可行只记录状态，另需完整不可行证书。

        输入：有限列 IP 结果以及有无证书的节点状态。
        输出：无；断言不误剪枝，且前沿缺少下界时不产生全局 LB。
        """

        _, node, info = information_fixture()
        ip = solve_schedule_enumeration(node, mode=0)
        info = record_incumbent(info, ip.selected_columns, source_node_id=0)
        info = record_node_stage(info, 0, node, "restricted_rmp", status="infeasible")
        self.assertIsNone(node_pruning_reason(info.nodes[0], info.upper_bound))
        self.assertIsNone(frontier_lower_bound(info.nodes, upper_bound=info.upper_bound))
        proved = record_node_stage(info, 0, node, "feasibility_pricing",
                                   infeasibility_proven=True, status="infeasible")
        self.assertEqual(node_pruning_reason(proved.nodes[0], None), "proven_infeasible")
        changed = replace(node, rmp=build_restricted_master(node.rmp.instance, ()))
        changed_info = record_node_stage(proved, 0, changed, "new_snapshot")
        self.assertIsNone(node_pruning_reason(changed_info.nodes[0], None))
        certified = replace(info.nodes[0], lower_bound=2.0)
        self.assertEqual(frontier_lower_bound((certified,), upper_bound=3.0), 2.0)
        self.assertIsNone(frontier_lower_bound((certified, info.nodes[0])))
        self.assertIsNone(frontier_lower_bound(()))


if __name__ == "__main__":
    unittest.main()
