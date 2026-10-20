"""验证列表调度规则、初始 RMP 可行性及关闭后的空列池。"""

import unittest

from config import get_config
from src.BranchPriceCut.accelerate.RootNodeColumnInitial import initialize_root_columns
from src.BranchPriceCut.solve import solve_restricted_master
from src.model.RMP import build_restricted_master
from src.model.dataload import SchedulingInstance


class RootColumnInitializationTests(unittest.TestCase):
    def test_round_robin_schedule_is_a_feasible_rmp_start(self) -> None:
        """核对 C# 轮转规则及初始列的覆盖和实际目标。

        输入：内置两机器四作业实例。
        输出：无；断言机器序列、整数上界及真实 RMP 的可行目标。
        """

        instance = SchedulingInstance(2, 4, 1, (1, 2, 3, 4), (1, 2), (1, 1, 1, 1),
                                      ((1.0, 8.0), (4.0, 1.0), (2.0, 4.0), (3.0, 2.0)))
        result = initialize_root_columns(instance)
        self.assertEqual(tuple(schedule.job_ids for schedule in result.schedules), ((1, 3), (2, 4)))
        self.assertEqual(result.upper_bound, 8.0)
        self.assertEqual(tuple(sum(column.coverage[j] for column in result.columns)
                               for j in range(4)), (1, 1, 1, 1))
        lp = solve_restricted_master(build_restricted_master(instance, result.columns))
        self.assertAlmostEqual(lp.objective_value, result.upper_bound)

    def test_idle_machine_and_disabled_initialization(self) -> None:
        """检查相同比值的确定性、空闲机器及关闭开关的真实语义。

        输入：三机器一作业实例与配置参数。
        输出：无；断言空闲调度、无虚构上界的空列池及开关。
        """

        instance = SchedulingInstance(3, 1, 1, (1,), (1, 2, 3), (1,), ((2.0, 2.0, 2.0),))
        result = initialize_root_columns(instance)
        self.assertEqual(tuple(schedule.job_ids for schedule in result.schedules), ((1,), (), ()))
        self.assertEqual(len(result.columns), 1)
        self.assertEqual(result.upper_bound, 2.0)
        empty = initialize_root_columns(instance, enabled=False)
        self.assertEqual(empty.columns, ())
        self.assertIsNone(empty.upper_bound)
        self.assertTrue(get_config([]).use_root_node_column_initial)
        self.assertFalse(get_config(["--no-use-root-node-column-initial"]).use_root_node_column_initial)


if __name__ == "__main__":
    unittest.main()
