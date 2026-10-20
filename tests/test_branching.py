"""作业弧聚合、子节点图/列/行一致性和两阶段强分支测试。"""

from __future__ import annotations

import unittest
from dataclasses import replace

from config import get_config
from src.BranchPriceCut.Branch import (
    BranchCandidate,
    aggregate_job_arcs,
    create_child_node,
    make_branch_node,
    ordinary_branch_candidate,
    pricing_requests_for_node,
)
from src.BranchPriceCut.accelerate.strong_branching import (
    TrialLPResult,
    process_peak_working_set_bytes,
    run_heuristic_column_generation,
    select_branch_candidate,
    solve_trial_rmp,
)
from src.BranchPriceCut.Price.dynamic_upper_bound import initialize_dynamic_upper_bounds
from src.BranchPriceCut.Price.forward_labeling import solve_forward_labeling
from src.BranchPriceCut.Price.rmp_integration import (
    add_lm_subset_row_cuts,
    rebuild_rmp_with_columns,
    rebuild_rmp_with_cut_pool,
)
from src.BranchPriceCut.Cut.lm_subset_row_cut import LimitedMemorySubsetRowCut
from src.model.PP import build_pricing_problems, evaluate_column_reduced_cost
from src.model.RMP import DualSolution, build_restricted_master
from src.model.dataload import SchedulingInstance
from src.model.schedulemodel import build_schedule_column


def branch_fixture():
    """构造两机器三作业的有限列节点。

    输入：无。
    输出：``SchedulingInstance`` 与 ``BranchNode`` 的二元组。
    """

    instance = SchedulingInstance(
        num_machines=2,
        num_jobs=3,
        instance_id=1,
        job_ids=(1, 2, 3),
        machine_ids=(1, 2),
        weights=(3, 2, 1),
        processing_times=((1.0, 1.0), (2.0, 2.0), (3.0, 3.0)),
    )
    columns = tuple(
        build_schedule_column(instance, machine_id, jobs)
        for machine_id in instance.machine_ids
        for jobs in ((1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3))
    )
    rmp = build_restricted_master(instance, columns)
    dual = DualSolution((0.0,) * 3, (0.0,) * 2)
    problems = build_pricing_problems(rmp, dual)
    bounds = tuple(initialize_dynamic_upper_bounds(problem, 10.0) for problem in problems)
    return instance, make_branch_node(rmp, problems, bounds)


class BranchingTests(unittest.TestCase):
    def test_aggregate_y_to_x_q_and_prefer_half(self) -> None:
        """检查真实作业弧聚合、x/q 差异和 0.5 排序。

        输入：当前 ``unittest.TestCase``。
        输出：无；通过断言报告结果。
        """

        _, node = branch_fixture()
        values = [0.0] * len(node.rmp.columns)
        values[3] = 0.4  # 机器 1 的 (1,2)
        values[10] = 0.3  # 机器 2 的 (1,2)
        values[4] = 0.5  # 机器 1 的 (1,3)
        candidates = aggregate_job_arcs(node.rmp, values)
        lookup = {candidate.key: candidate.value for candidate in candidates}
        self.assertAlmostEqual(lookup["x", 1, 1, 2], 0.4)
        self.assertAlmostEqual(lookup["x", 2, 1, 2], 0.3)
        self.assertAlmostEqual(lookup["q", 0, 1, 2], 0.7)
        self.assertEqual(ordinary_branch_candidate(candidates).key, ("x", 1, 1, 3))
        self.assertTrue(all(0 not in candidate.key[2:] for candidate in candidates))

    def test_x_children_filter_columns_graph_and_preserve_parent(self) -> None:
        """检查 x=0/1 两侧局部 DAG、旧列、分支行及父节点隔离。

        输入：当前 ``unittest.TestCase``。
        输出：无；通过断言报告结果。
        """

        _, parent = branch_fixture()
        candidate = BranchCandidate("x", 1, 2, 0.5, 1)
        zero = create_child_node(parent, candidate, 0)
        one = create_child_node(parent, candidate, 1)
        self.assertIn((1, 2), parent.bounds[0].dag_arcs)
        self.assertNotIn((1, 2), zero.bounds[0].dag_arcs)
        self.assertIn((1, 2), zero.bounds[1].dag_arcs)
        self.assertTrue(all((1, 2) not in c.arcs for c in zero.rmp.columns if c.machine_id == 1))
        self.assertEqual(one.bounds[0].dag_arcs.count((1, 2)), 1)
        self.assertTrue(all(
            head == 2 for tail, head in one.bounds[0].dag_arcs if tail == 1
        ))
        self.assertTrue(all(
            tail == 1 for tail, head in one.bounds[0].dag_arcs if head == 2
        ))
        self.assertTrue(all(1 not in c.job_ids and 2 not in c.job_ids
                            for c in one.rmp.columns if c.machine_id == 2))
        self.assertEqual(one.rmp.cut_rows[-1].sense, "=")
        self.assertEqual(one.rmp.cut_rows[-1].rhs, 1.0)
        self.assertEqual(len(parent.rmp.cut_rows), 0)
        self.assertEqual(len(one.decisions), 1)

    def test_q_one_row_and_dual_work_through_pricing(self) -> None:
        """检查 q=1 的 RMP 等式、列更新、割池更新和定价对偶系数。

        输入：当前 ``unittest.TestCase``。
        输出：无；通过断言报告结果。
        """

        instance, parent = branch_fixture()
        child = create_child_node(parent, BranchCandidate("q", 1, 2, 0.5), 1)
        result = solve_trial_rmp(child.rmp)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(sum(
            coefficient * value
            for coefficient, value in zip(child.rmp.cut_rows[-1].coefficients, result.column_values)
        ), 1.0)
        self.assertEqual(child.rmp.cut_rows[-1].coefficients.count(1.0), 4)

        extra = build_schedule_column(instance, 1, (1, 2, 3))
        rebuilt = rebuild_rmp_with_columns(child.rmp, (*child.rmp.columns, extra))
        self.assertEqual(rebuilt.cut_rows[-1].coefficients[-1], 1.0)
        cut = LimitedMemorySubsetRowCut("lm", (1, 2, 3),
                                        (frozenset((1, 2, 3)),) * 2)
        with_cut = add_lm_subset_row_cuts(child.rmp, (cut,))
        refreshed = rebuild_rmp_with_cut_pool(with_cut, (cut,))
        self.assertEqual(len(refreshed.cut_rows), 2)
        self.assertEqual(refreshed.cut_rows[0].sense, "=")

        local = pricing_requests_for_node(child, result.dual_solution)[0]
        pricing = solve_forward_labeling(instance, local)
        evaluated = evaluate_column_reduced_cost(local, pricing.best_evaluation.column)
        self.assertAlmostEqual(pricing.best_evaluation.reduced_cost, evaluated.reduced_cost)
        self.assertTrue(pricing.is_exact)

        branch_dual = DualSolution(
            job_duals=(5.0, 5.0, 0.0), machine_duals=(0.0, 0.0),
            cut_ids=(child.rmp.cut_rows[-1].cut_id,), cut_duals=(10.0,),
        )
        request = pricing_requests_for_node(child, branch_dual)[0]
        best = solve_forward_labeling(instance, request)
        self.assertIn((1, 2), best.best_evaluation.column.arcs)
        self.assertAlmostEqual(
            best.best_evaluation.reduced_cost,
            evaluate_column_reduced_cost(request, best.best_evaluation.column).reduced_cost,
        )

    def test_q_zero_and_descendant_keep_ancestor_restrictions(self) -> None:
        """检查 q=0 全机器删弧及孙节点的祖先限制。

        输入：当前 ``unittest.TestCase``。
        输出：无；通过断言报告结果。
        """

        _, parent = branch_fixture()
        zero = create_child_node(parent, BranchCandidate("q", 1, 2, 0.5), 0)
        self.assertTrue(all((1, 2) not in state.dag_arcs for state in zero.bounds))
        self.assertTrue(all((1, 2) not in column.arcs for column in zero.rmp.columns))
        grandchild = create_child_node(zero, BranchCandidate("x", 1, 3, 0.5, 1), 1)
        self.assertEqual(len(grandchild.decisions), 2)
        self.assertEqual(len(grandchild.rmp.cut_rows), 2)
        self.assertTrue(all((1, 2) not in state.dag_arcs for state in grandchild.bounds))

    def test_infeasible_restricted_rmp_is_not_a_child_proof(self) -> None:
        """检查缺少强制弧列时试探返回 None，节点仍保留决定。

        输入：当前 ``unittest.TestCase``。
        输出：无；通过断言报告结果。
        """

        _, parent = branch_fixture()
        reduced = replace(parent, rmp=build_restricted_master(
            parent.rmp.instance,
            tuple(c for c in parent.rmp.columns if (1, 2) not in c.arcs),
        ))
        child = create_child_node(reduced, BranchCandidate("q", 1, 2, 0.5), 1)
        self.assertIsNone(solve_trial_rmp(child.rmp))
        self.assertEqual(child.decisions[-1].value, 1)

    def test_strong_branching_gate_scores_and_fallback(self) -> None:
        """检查门槛、两轮得分及无可评分候选时的回退。

        输入：当前 ``unittest.TestCase``。
        输出：无；通过断言报告结果。
        """

        _, node = branch_fixture()
        values = [0.0] * len(node.rmp.columns)
        values[3] = 0.5
        values[4] = 0.4
        values[5] = 0.3
        ordinary = select_branch_candidate(
            node, values, 0.0, peak_working_set_bytes=2 * 1024**3
        )
        self.assertEqual(ordinary.method, "ordinary")
        self.assertEqual(len(ordinary.children), 2)

        def trial(rmp):
            """从末尾分支行产生确定的 LP 试算。

            输入：``RestrictedMasterProblem``。
            输出：带目标和零对偶的 ``TrialLPResult``。
            """

            decision = rmp.cut_definitions[-1].decision
            objective = float(decision.head + (1 if decision.value else 0))
            dual = DualSolution((0.0,) * 3, (0.0,) * 2)
            return TrialLPResult(objective, (0.0,) * rmp.num_variables, dual)

        def heuristic(child):
            """模拟一次启发式列生成。

            输入：``BranchNode`` 子节点。
            输出：可比较的 ``TrialLPResult``。
            """

            decision = child.decisions[-1]
            objective = float(10 - decision.head + decision.value)
            dual = DualSolution((0.0,) * 3, (0.0,) * 2)
            return TrialLPResult(objective, (0.0,) * child.rmp.num_variables, dual)

        strong = select_branch_candidate(
            node, values, 0.0, phase0_size=3, phase1_size=2,
            peak_working_set_bytes=0, lp_solver=trial, heuristic_runner=heuristic,
        )
        self.assertEqual(strong.method, "strong")
        self.assertEqual(len(strong.phase0_candidates), 3)
        self.assertEqual(len(strong.phase1_scores), 3)
        self.assertEqual(len(strong.phase2_scores), 2)
        self.assertEqual(len(strong.children), 2)

        fallback = select_branch_candidate(
            node, values, 0.0, phase0_size=2, phase1_size=2,
            peak_working_set_bytes=0, lp_solver=lambda rmp: None,
        )
        self.assertEqual(fallback.method, "ordinary_fallback")
        self.assertEqual(fallback.candidate.kind, "x")

    def test_real_heuristic_trial_and_peak_memory(self) -> None:
        """检查真实进程内存读数及受限子节点的一次启发式试算。

        输入：当前 ``unittest.TestCase``。
        输出：无；通过断言报告结果。
        """

        self.assertGreater(process_peak_working_set_bytes(), 0)
        _, parent = branch_fixture()
        child = create_child_node(parent, BranchCandidate("x", 1, 2, 0.5, 1), 1)
        result = run_heuristic_column_generation(child, max_rounds=1)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.objective_value, 0.0)

    def test_config_and_parent_node_gate(self) -> None:
        """检查统一配置、整数节点和界剪枝均阻止试探。

        输入：当前 ``unittest.TestCase``。
        输出：无；通过断言报告结果。
        """

        config = get_config([])
        self.assertTrue(config.use_strong_branching)
        self.assertEqual((config.branch_variable, config.strong_branch_phase0,
                          config.strong_branch_phase1), ("x", 10, 3))
        self.assertFalse(get_config(["--no-use-strong-branching"]).use_strong_branching)
        _, node = branch_fixture()
        integer = select_branch_candidate(
            node, (0.0,) * node.rmp.num_variables, 1.0, peak_working_set_bytes=0,
        )
        self.assertEqual(integer.method, "integer_y")
        values = [0.0] * node.rmp.num_variables
        values[3] = 0.5
        pruned = select_branch_candidate(
            node, values, 2.0, incumbent_objective=2.0, peak_working_set_bytes=0,
        )
        self.assertEqual(pruned.method, "pruned_by_bound")


if __name__ == "__main__":
    unittest.main()
