"""主流程回归、完整枚举对照和 JD 批量实例验收。

快速测试：python -m unittest discover -s tests -v
指定/批量 JD：python tests/test_main.py --benchmark --batch --scale low --time-limit 360
真实 JD 目标回归：设置环境变量 BPC_RUN_JD_TESTS=1 后运行 unittest。
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest
from collections import defaultdict
from dataclasses import replace
from itertools import product
from pathlib import Path
from random import Random
from unittest.mock import patch
import cplex

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openpyxl import Workbook

import main as workflow
from config import get_config
from src.BranchPriceCut.Branch.branching import validate_integer_schedule
from src.BranchPriceCut.Price.Bucket.bucket_forward_labeling import solve_bucket_forward_labeling_with_bounds
from src.BranchPriceCut.Price.dynamic_upper_bound import initialize_dynamic_upper_bounds
from src.BranchPriceCut.Price.forward_labeling import solve_forward_labeling
from src.BranchPriceCut.Price.labels import ForwardLabel, insert_nondominated
from src.BranchPriceCut.solve import RMPSolveResult, solve_restricted_master
from src.model.PP import build_pricing_problems
from src.model.RMP import DualSolution, build_restricted_master
from src.model.dataload import SchedulingInstance, build_instance_path
from src.model.schedulemodel import build_schedule_column


def fixture(seed: int = 25, jobs: int = 9, machines: int = 3) -> SchedulingInstance:
    """生成可复现的人工测试实例，默认实例实际触发割和强分支。

    输入：随机种子、作业数和机器数 int。输出：SchedulingInstance。
    """

    rng = Random(seed)
    return SchedulingInstance(
        machines, jobs, seed, tuple(range(1, jobs + 1)), tuple(range(1, machines + 1)),
        tuple(rng.randint(1, 3) for _ in range(jobs)),
        tuple(tuple(float(rng.randint(1, 8)) for _ in range(machines)) for _ in range(jobs)),
    )


def exhaustive_objective(instance: SchedulingInstance) -> float:
    """独立枚举所有机器分配，SWPT 给定每台机器上的最优顺序。

    输入：小 SchedulingInstance。输出：全局最优目标 float；不使用 BPC 的图或时间界。
    """

    best = float("inf")
    for assignment in product(instance.machine_ids, repeat=instance.num_jobs):
        objective = sum(build_schedule_column(instance, machine,
            tuple(job for job, assigned in zip(instance.job_ids, assignment) if assigned == machine)).cost
            for machine in instance.machine_ids)
        best = min(best, objective)
    return best


def configuration(*options: str):
    """提供快速且确定性的回归配置。

    输入：命令行选项 str 序列。输出：argparse.Namespace，默认关闭随机邻域和文件保存。
    """

    return get_config(["--no-save-results", "--no-use-neighborhood-search", *options])


class MainWorkflowTests(unittest.TestCase):
    def assert_valid_result(self, result, expected: float) -> None:
        """核对界、最优状态以及原问题调度可行性。

        输入：BPCResult、独立最优值 float。输出：None，失败时报告断言。
        """

        self.assertTrue(result.optimality_proven)
        self.assertEqual(result.status, "optimal")
        self.assertAlmostEqual(result.upper_bound, expected, places=5)
        self.assertLessEqual(result.lower_bound, expected + 1e-6)
        self.assertLessEqual(result.relative_gap, 1e-6)
        incumbent = result.information.incumbent_history[-1]
        self.assertTrue(validate_integer_schedule(result.information.instance, incumbent.columns))
        self.assertAlmostEqual(sum(column.cost for column in incumbent.columns), expected)

    def test_small_instances_match_independent_assignment_enumeration(self) -> None:
        """对多种机器/作业数量与两条定价路径作完整枚举对照。

        输入：TestCase。输出：None。
        """

        for machines, jobs, seed in ((1, 4, 1), (2, 5, 2), (3, 6, 3), (2, 6, 7)):
            instance = fixture(seed, jobs, machines)
            expected = exhaustive_objective(instance)
            for switches in ((), ("--no-use-dsub", "--no-use-bucket-graph", "--no-use-time-bucket",
                                 "--no-use-bucket-dominance", "--no-use-reduced-cost-fixing", "--no-use-strong-branching")):
                with self.subTest(machines=machines, jobs=jobs, switches=switches):
                    self.assert_valid_result(workflow.solve_instance(instance, configuration(*switches)), expected)

    def test_neighborhood_exact_fallback_and_snapshot_certificates(self) -> None:
        """实际执行邻域和桶精确定价，并检查证书均来自同一 RMP 版本。

        输入：TestCase。输出：None。
        """

        instance = fixture(3, 5, 2)
        result = workflow.solve_instance(instance, configuration("--use-neighborhood-search",
            "--heuristic-iterations", "1", "--heuristic-neighbors", "2"))
        self.assert_valid_result(result, exhaustive_objective(instance))
        self.assertGreater(result.statistics["heuristic_calls"], 0)
        self.assertGreater(result.statistics["bucket_exact_calls"], 0)
        groups = defaultdict(list)
        for diagnostic in result.diagnostics:
            if diagnostic["stage"] == "exact_pricing":
                groups[diagnostic["pricing_round"]].append(diagnostic)
        complete = [items for items in groups.values() if len(items) == instance.num_machines
                    and all(item["is_exact"] and item["reduced_cost"] >= -1e-7 for item in items)]
        self.assertTrue(complete)
        for items in complete:
            self.assertEqual(len({item["column_count"] for item in items}), 1)
            self.assertEqual(len({item["cut_ids"] for item in items}), 1)

    def test_cut_addition_returns_to_column_generation(self) -> None:
        """实际触发两条 lm-SRC，加割后重新定价再取得最优证书。

        输入：TestCase。输出：None。
        """

        result = workflow.solve_instance(fixture(), configuration())
        self.assert_valid_result(result, 64.0)
        self.assertGreater(result.statistics["added_cuts"], 0)
        stages = result.information.nodes[0].stages
        cut_positions = [index for index, event in enumerate(stages) if event.status == "cuts_added"]
        for index in cut_positions:
            self.assertEqual(stages[index + 1].stage, "rmp")
            self.assertFalse(stages[index].exact_pricing_complete)
            self.assertTrue(any(event.exact_pricing_complete for event in stages[index + 1:]))

    def test_fixing_and_multistage_strong_branching_preserve_optimum(self) -> None:
        """实际删弧并执行两阶段强分支，最后与独立枚举核对。

        输入：TestCase。输出：None。
        """

        instance = fixture()
        expected = exhaustive_objective(instance)
        result = workflow.solve_instance(instance, configuration("--no-use-cuts", "--enumeration-gap", "0",
            "--reduced-cost-fixing-gap", "1", "--rmp-lp-method", "0"))
        self.assert_valid_result(result, expected)
        self.assertGreater(result.statistics["removed_arcs"], 0)
        self.assertGreater(result.statistics["strong_branches"], 0)
        evidence = [item for item in result.diagnostics if item["stage"] == "branching"]
        self.assertTrue(all(item["phase1_scores"] and item["phase2_scores"] for item in evidence))
        for record in result.information.nodes[1:]:
            parent = result.information.nodes[record.parent_id]
            self.assertEqual(record.depth, parent.depth + 1)
            self.assertGreaterEqual(record.lower_bound, parent.lower_bound)
            self.assertEqual(len(record.node.decisions), len(parent.node.decisions) + 1)
        root_stages = result.information.nodes[0].stages
        fixed = next(event for event in root_stages if event.stage == "reduced_cost_fixing")
        self.assertGreater(fixed.removed_dag_arcs, 0)
        self.assertFalse(fixed.exact_pricing_complete)
        self.assertTrue(any(event.stage == "pricing" and event.exact_pricing_complete
                            for event in root_stages[root_stages.index(fixed) + 1:]))

    def test_enumeration_modes_and_ordinary_q_branch(self) -> None:
        """两种枚举模式以及普通 q 弧分支均保持目标值。

        输入：TestCase。输出：None。
        """

        for mode in (0, 1):
            result = workflow.solve_instance(fixture(), configuration("--no-use-cuts", "--enumeration-gap", "1",
                "--use-schedule-enumeration", "--schedule-enumeration-mode", str(mode),
                "--no-use-strong-branching", "--branch-variable", "q"))
            self.assert_valid_result(result, 64.0)
            self.assertGreater(result.statistics["enumeration_calls"], 0)
            if mode == 1:
                self.assertTrue(any(item.get("enumeration_complete") for item in result.diagnostics))
        ordinary = workflow.solve_instance(fixture(), configuration("--no-use-cuts", "--enumeration-gap", "0",
            "--max-added-columns", "1", "--rmp-lp-method", "0", "--no-use-strong-branching", "--branch-variable", "q"))
        self.assert_valid_result(ordinary, 64.0)
        self.assertGreater(ordinary.statistics["branch_calls"], 0)

    def test_restricted_child_infeasibility_does_not_prove_optimality(self) -> None:
        """用真实根求解和受控子 RMP 停止检查全局证明责任。

        输入：TestCase。输出：None；不可行停止节点必须留在前沿中。
        """

        def stop_children(rmp, **kwargs):
            """仅将带分支行的试探/正式子节点返回为受限不可行。

            输入：RestrictedMasterProblem 和求解参数。输出：RMPSolveResult。
            """

            if any(cut.cut_id.startswith("branch[") for cut in rmp.cut_rows):
                return RMPSolveResult("infeasible", None, (), (), None)
            return solve_restricted_master(rmp, **kwargs)

        with patch.object(workflow, "solve_restricted_master", side_effect=stop_children):
            result = workflow.solve_instance(fixture(), configuration("--no-use-cuts", "--enumeration-gap", "0"))
        self.assertEqual(result.status, "author_flow_stopped")
        self.assertFalse(result.optimality_proven)
        self.assertEqual(len(result.unresolved_node_ids), 2)
        self.assertAlmostEqual(result.lower_bound, 64.0)
        self.assertGreater(result.upper_bound, 64.0)
        for identity in result.unresolved_node_ids:
            record = result.information.nodes[identity]
            self.assertEqual(record.stages[-1].status, "restricted_rmp_infeasible")
            self.assertFalse(any(event.infeasibility_proven for event in record.stages))

    def test_limits_and_disabled_initialization_keep_honest_bounds(self) -> None:
        """时间/节点预算和空列启动不会产生虚假的最优状态。

        输入：TestCase。输出：None。
        """

        for args, status in ((("--time-limit", "0"), "time_limit"),
                             (("--max-nodes", "0"), "node_limit"),
                             (("--no-use-root-node-column-initial",), "unsupported_empty_root_pool")):
            result = workflow.solve_instance(fixture(), configuration(*args))
            self.assertEqual(result.status, status)
            self.assertFalse(result.optimality_proven)
            self.assertEqual(result.unresolved_node_ids, (0,))
            self.assertLessEqual(result.lower_bound, 64.0)

    def test_empty_machine_graph_and_interrupted_pricing(self) -> None:
        """固定后无路径和超时定价分别返回精确空集与不完整证书。

        输入：TestCase。输出：None。
        """

        instance = fixture(2, 3, 1)
        rmp = build_restricted_master(instance, ())
        problem = build_pricing_problems(rmp, DualSolution((0.0,) * 3, (0.0,)))[0]
        problem = replace(problem, dag_arcs=())
        bounds = initialize_dynamic_upper_bounds(problem, 30.0)
        for result in (solve_forward_labeling(instance, problem),
                       solve_bucket_forward_labeling_with_bounds(instance, problem, bounds)):
            self.assertTrue(result.is_exact)
            self.assertIsNone(result.best_evaluation)
            self.assertFalse(result.has_improving_column)
        normal = build_pricing_problems(rmp, DualSolution((100.0,) * 3, (0.0,)))[0]
        bounds = initialize_dynamic_upper_bounds(normal, 30.0)
        result = solve_bucket_forward_labeling_with_bounds(instance, normal, bounds, deadline=0.0)
        self.assertFalse(result.is_exact)

    def test_rmp_failure_and_equivalent_labels(self) -> None:
        """真实空 RMP 不可行可安全返回，等价标签合并不改变扩展状态。

        输入：TestCase。输出：None。
        """

        result = solve_restricted_master(build_restricted_master(fixture(2, 3, 1), ()))
        self.assertEqual(result.status, "infeasible")
        self.assertIsNone(result.dual_solution)
        labels = [ForwardLabel(3, 5.0, -2.0, (1, 3), (), ())]
        self.assertFalse(insert_nondominated(labels, ForwardLabel(3, 5.0, -2.0, (2, 3), (), ())))
        self.assertEqual(len(labels), 1)

    def test_save_input_results_and_scale_directories(self) -> None:
        """保存全部输入与结果，三档规模分别路由并保留每次运行。

        输入：TestCase。输出：None；临时目录由测试自动清理。
        """

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "tmp") as directory:
            for scale, thresholds in (("low", (5, 6)), ("medium", (2, 5)), ("high", (1, 2))):
                config = configuration("--save-results", "--results-dir", directory,
                    "--low-max-jobs", str(thresholds[0]), "--medium-max-jobs", str(thresholds[1]))
                result = workflow.solve_instance(fixture(3, 3, 2), config)
                self.assertEqual(result.result_path.parent.name, scale)
                saved = json.loads(result.result_path.read_text(encoding="utf-8"))
                self.assertEqual(saved["instance"]["num_jobs"], 3)
                self.assertEqual(saved["configuration"]["num_job"], 3)
                self.assertEqual(saved["instance"]["processing_times"], [[float(value) for value in row]
                    for row in result.information.instance.processing_times])
                self.assertEqual(saved["upper_bound"], result.upper_bound)
                self.assertTrue(saved["nodes"] and saved["schedules"])
                self.assertIsNone(saved["source_path"])
            for scale in ("low", "medium", "high"):
                self.assertTrue((Path(directory) / scale).is_dir())
            second = workflow.solve_instance(fixture(3, 3, 2), config)
            self.assertNotEqual(result.result_path, second.result_path)

    def test_excel_loading_discovery_and_batch_entry(self) -> None:
        """构造真实 Excel，通过批量入口核对读取、目标及保存的路径。

        输入：TestCase。输出：None。
        """

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "tmp") as directory:
            args = ["--batch", "--data-dir", directory, "--results-dir", str(Path(directory) / "results"),
                    "--num-machine", "2", "--num-job", "3", "--instance-id", "1", "--no-use-neighborhood-search"]
            config = get_config(args)
            path = build_instance_path(config)
            path.parent.mkdir(parents=True)
            workbook = Workbook()
            weights = workbook.active
            weights.title = "Weights"
            weights.append(["job", "weight"])
            for job in (1, 2, 3):
                weights.append([job, job])
            times = workbook.create_sheet("Processing times")
            times.append(["machine", 1, 2, 3])
            times.append([1, 1.0, 2.0, 3.0])
            times.append([2, 3.0, 1.0, 2.0])
            workbook.save(path)
            workbook.close()
            self.assertEqual(len(workflow.discover_instances(config)), 1)
            results = workflow.main(args)
            self.assertEqual(len(results), 1)
            self.assert_valid_result(results[0], exhaustive_objective(results[0].information.instance))
            loaded = results[0].information.instance
            self.assertEqual(loaded.weights, (1, 2, 3))
            self.assertEqual(loaded.processing_times, ((1.0, 3.0), (2.0, 1.0), (3.0, 2.0)))
            saved = json.loads(results[0].result_path.read_text(encoding="utf-8"))
            self.assertEqual(Path(saved["source_path"]), path)
            config.scale = "high"
            self.assertEqual(workflow.discover_instances(config), ())

    @unittest.skipUnless(os.environ.get("BPC_RUN_JD_TESTS") == "1", "设置 BPC_RUN_JD_TESTS=1 运行真实 JD 基准")
    def test_jd_reference_objectives(self) -> None:
        """验收 12×60 第一个 JD 实例的两种加工时间精度和保存结果。

        输入：TestCase。输出：None；数据缺失时跳过，实际求解受配置预算约束。
        """

        for decimals, expected in ((1, 881.0), (2, 857.69)):
            options = ["--processing-time-decimals", str(decimals)]
            if "BPC_JD_TIME_LIMIT" in os.environ:
                options.extend(("--time-limit", os.environ["BPC_JD_TIME_LIMIT"]))
            config = get_config(options)
            if not build_instance_path(config).is_file():
                self.skipTest("本机没有 JD 数据文件")
            result = workflow.solve_instance(configuration=config)
            self.assert_valid_result(result, expected)
            self.assertTrue(result.result_path.is_file())

    def test_defaults_match_available_csharp_parameters(self) -> None:
        """直接读取作者参数，防止 Python 默认值与 C# 再次漂移。

        输入：TestCase。输出：None；独立运行项目缺少 C# 源码时跳过源码对照。
        """

        defaults = get_config([])
        self.assertIsNone(defaults.time_limit)
        self.assertIsNone(defaults.max_nodes)
        self.assertTrue(defaults.use_neighborhood_search)
        self.assertFalse(defaults.use_schedule_enumeration)
        self.assertEqual((defaults.pricing_tolerance, defaults.heuristic_pricing_tolerance), (0.01, 1e-4))
        self.assertEqual((defaults.rmp_lp_method, defaults.solver_threads), (1, 0))
        self.assertEqual(defaults.solver_optimality_tolerance, 1e-6)
        self.assertTrue(defaults.solver_numerical_emphasis)
        source = Path(__file__).resolve().parents[2] / "2024.0704-main" / "src" / "UPMSP - Branch-Cut-and-Price Algorithm" / "Parameters.cs"
        if not source.is_file():
            self.skipTest("C# 参考项目不在相邻目录")
        text = source.read_text(encoding="utf-8-sig")
        for python_name, csharp_name in (
            ("num_time_buckets", "numOfBucketOnOneVertex"), ("max_added_columns", "numAddedColumns"),
            ("max_cut_candidates", "numOfSubsets"), ("max_added_cuts", "numAddedLmSRCs"),
            ("max_cut_rounds", "maxIterationsLmSRCs"), ("heuristic_neighbors", "numNeighbors"),
            ("heuristic_iterations", "maxIterationsNeighborhoodSearch"),
            ("heuristic_max_same_objective", "maxNumSameObjValue"),
            ("strong_branch_phase0", "numCandidatesInPhase0"), ("strong_branch_phase1", "numCandidatesInPhase1"),
            ("strong_branch_memory_gb", "maxGigaBytes"), ("schedule_enumeration_max_labels", "maxNumEnumeratedLabels"),
        ):
            match = re.search(r"this\." + csharp_name + r"\s*=\s*([\d.]+)\s*;", text)
            self.assertIsNotNone(match, csharp_name)
            self.assertEqual(getattr(defaults, python_name), float(match.group(1)), python_name)

    def test_existing_label_search_supplies_multiple_new_columns(self) -> None:
        """两条已有标签路径均可一次返回多列，旧列在应用数量上限前剔除。

        输入：TestCase。输出：None；验证最小约化成本和定价证书不受入列批量影响。
        """

        instance = fixture(5, 5, 1)
        rmp = build_restricted_master(instance, ())
        problem = build_pricing_problems(rmp, DualSolution((100.0,) * 5, (0.0,)))[0]
        bounds = initialize_dynamic_upper_bounds(problem, 100.0)
        for bucket in (False, True):
            def price(**kwargs):
                """调用当前测试选择的已有标签实现。

                输入：定价关键字参数 dict。输出：ExactPricingResult。
                """

                return (solve_bucket_forward_labeling_with_bounds(instance, problem, bounds, **kwargs)
                        if bucket else solve_forward_labeling(instance, problem, **kwargs))

            many = price(max_columns=8)
            limited = price(max_columns=2)
            self.assertGreater(len(many.negative_evaluations), 2)
            self.assertEqual(len(limited.negative_evaluations), 2)
            self.assertAlmostEqual(many.best_evaluation.reduced_cost, limited.best_evaluation.reduced_cost)
            excluded = frozenset(item.column.key for item in limited.negative_evaluations)
            next_batch = price(max_columns=2, existing_column_keys=excluded)
            self.assertEqual(len(next_batch.negative_evaluations), 2)
            self.assertTrue(excluded.isdisjoint(item.column.key for item in next_batch.negative_evaluations))
            costs = [item.reduced_cost for item in many.negative_evaluations]
            self.assertEqual(costs, sorted(costs))

    def test_coarse_csharp_threshold_keeps_strict_pricing_certificate(self) -> None:
        """0.01 入列阈值不能掩盖 -0.005 的遗漏列，仍调用原严格精确定价。

        输入：TestCase。输出：None；真实标签结果同时验证多列进入主线。
        """

        instance = SchedulingInstance(1, 1, 1, (1,), (1,), (1,), ((1.0,),))
        solver = workflow.BPCSolver(instance, configuration())
        node = solver._initial_node()
        node = replace(node, rmp=build_restricted_master(instance, ()))
        solution = RMPSolveResult("optimal", 1.005, (), (), DualSolution((1.005,), (0.0,)))
        additions, certified, _ = solver._price(0, node, solution)
        self.assertEqual(len(additions), 1)
        self.assertFalse(certified)
        self.assertEqual(solver.statistics["exact_calls"], 2)
        self.assertTrue(solver.diagnostics[-1]["strict_check"])

    def test_cut_limits_and_enumeration_switch_remain_explicit(self) -> None:
        """扫描/轮数限额不会被记成完整割证明，枚举只有显式开启才调用。

        输入：TestCase。输出：None。
        """

        for options, expected in ((("--max-cut-candidates", "0"), "candidate_limit_no_new_cut"),
                                  (("--max-cut-rounds", "0"), "cut_round_limit")):
            solver = workflow.BPCSolver(fixture(), configuration(*options))
            root = solver._initial_node()
            relaxation = solver._converge(0, root)
            self.assertEqual(relaxation.status, "converged")
            self.assertFalse(relaxation.cut_separation_complete)
            self.assertEqual(solver.information.nodes[0].stages[-1].status, expected)
            self.assertTrue(any(event.exact_pricing_complete for event in solver.information.nodes[0].stages))
        result = workflow.solve_instance(fixture(), configuration("--no-use-cuts", "--enumeration-gap", "1"))
        self.assertEqual(result.statistics["enumeration_calls"], 0)
        self.assert_valid_result(result, 64.0)

    def test_observed_cplex_size_limit_preserves_unresolved_result(self) -> None:
        """默认八列验收实际暴露的 1016 错误应保存为规模限制，不伪造证明。

        输入：TestCase。输出：None；仅该已观测错误被转换，保留节点的合法初始界。
        """

        error = cplex.exceptions.CplexSolverError("Community Edition size limit", None, 1016)
        with patch("cplex.Cplex.solve", side_effect=error):
            result = workflow.solve_instance(fixture(), configuration())
        self.assertEqual(result.status, "solver_size_limit")
        self.assertFalse(result.optimality_proven)
        self.assertEqual(result.unresolved_node_ids, (0,))
        self.assertEqual(result.information.nodes[0].stages[-1].status, "restricted_rmp_solver_size_limit")
        self.assertLess(result.lower_bound, result.upper_bound)


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        workflow.main([argument for argument in sys.argv[1:] if argument != "--benchmark"])
    else:
        unittest.main()
