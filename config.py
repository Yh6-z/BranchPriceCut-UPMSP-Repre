"""BPC 复现项目的统一命令行配置入口。

适用阶段：程序启动时集中定义公共参数。
上游输入：命令行参数，或测试传入的字符串序列。
下游输出：``argparse.Namespace``，供数据加载和后续算法模块读取。

本模块只保存和解析配置，不读取算例、不构造模型、不启动求解器。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def get_config(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析算例与算法配置。

    输入：命令行参数 ``Sequence[str]`` 或 ``None``。
    输出：含数据规模、加速开关和分支参数的 ``argparse.Namespace``。
    """

    parser = argparse.ArgumentParser(
        description="BPC reproduction configuration for UPMSP-WCT",
    )
    parser.add_argument(
        "--num_machine",
        "--num-machine",
        dest="num_machine",
        type=int,
        default=12,
        help="机器数量，默认值为 12",
    )
    parser.add_argument(
        "--num_job",
        "--num-job",
        dest="num_job",
        type=int,
        default=60,
        help="作业数量，默认值为 60",
    )
    parser.add_argument(
        "--instance_id",
        "--instance-id",
        dest="instance_id",
        type=int,
        default=1,
        help="当前机器数和作业数下的算例编号，默认值为 1",
    )
    parser.add_argument(
        "--num_time_buckets",
        "--num-time-buckets",
        dest="num_time_buckets",
        type=int,
        default=4,
        help="每个作业的时间桶数量，默认值为 4",
    )
    _add_acceleration_switch(
        parser,
        "root_node_column_initial",
        help_text="是否使用列表调度初始化根节点列池和整数上界；作者主流程默认开启，不支持空列池启动",
    )
    _add_acceleration_switch(
        parser,
        "time_bucket",
        help_text="是否启用连续时间桶",
    )
    _add_acceleration_switch(
        parser,
        "bucket_graph",
        help_text="是否启用 Bucket Graph",
    )
    _add_acceleration_switch(
        parser,
        "bucket_dominance",
        help_text="是否启用桶内与跨桶支配",
    )
    _add_acceleration_switch(
        parser,
        "DSUB",
        cli_name="dsub",
        help_text="是否启用动态收缩上界及其 DAG 删弧",
    )
    _add_acceleration_switch(
        parser,
        "backward_labeling",
        help_text="是否启用后向标签",
    )
    _add_acceleration_switch(
        parser,
        "label_joining",
        help_text="是否启用前后向标签拼接",
    )
    _add_acceleration_switch(
        parser,
        "reduced_cost_fixing",
        help_text="是否启用约化成本固定删弧",
    )
    parser.add_argument(
        "--schedule_enumeration_mode", "--schedule-enumeration-mode",
        type=int, choices=(0, 1), default=0,
        help="0 仅对当前调度列池求缩减 IP；1 先用标签枚举补充列再求 IP",
    )
    parser.add_argument(
        "--schedule_enumeration_max_labels", "--schedule-enumeration-max-labels",
        type=int, default=10000,
        help="模式 1 每台机器的标签生成上限，默认 10000",
    )
    parser.add_argument(
        "--reduced_cost_fixing_gap",
        "--reduced-cost-fixing-gap",
        type=float,
        default=0.10,
        help="触发约化成本固定的最大相对原始—对偶间隙，默认 0.10",
    )
    parser.add_argument(
        "--reduced_cost_fixing_tolerance",
        "--reduced-cost-fixing-tolerance",
        type=float,
        default=1e-5,
        help="约化成本固定的保守比较容差，默认 1e-5",
    )
    _add_acceleration_switch(
        parser,
        "strong_branching",
        help_text="是否在进程峰值工作集低于 2 GiB 时启用多阶段强分支",
    )
    parser.add_argument(
        "--branch_variable", "--branch-variable",
        choices=("x", "q"), default="x",
        help="普通作业弧分支的表达式类型，默认 x",
    )
    parser.add_argument(
        "--strong_branch_phase0", "--strong-branch-phase0",
        type=int, default=10,
        help="多阶段强分支初筛候选上限，默认 10",
    )
    parser.add_argument(
        "--strong_branch_phase1", "--strong-branch-phase1",
        type=int, default=3,
        help="进入启发式试算的候选上限，默认 3",
    )
    parser.add_argument("--max-added-columns", type=int, default=8,
                        help="每台机器每轮精确定价最多加入的新列数，对应 C# numAddedColumns")
    parser.add_argument("--max-cut-candidates", type=int, default=10000,
                        help="每次割分离扫描的三作业基集数量，对应 C# numOfSubsets")
    parser.add_argument("--max-cut-rounds", type=int, default=30,
                        help="对应 C# maxIterationsLmSRCs；作者在 iter>30 后停止")
    parser.add_argument("--strong-branch-memory-gb", type=float, default=2.0)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data" / "JD" / "JD Instances")
    parser.add_argument("--processing-time-decimals", type=int, choices=(1, 2), default=1)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--save-results", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch", action="store_true", help="运行数据目录中匹配规模的全部实例")
    parser.add_argument("--scale", choices=("all", "low", "medium", "high"), default="all")
    parser.add_argument("--low-max-jobs", type=int, default=60)
    parser.add_argument("--medium-max-jobs", type=int, default=100)
    parser.add_argument("--relative-gap", type=float, default=1e-6)
    parser.add_argument("--absolute-gap", type=float, default=1e-6)
    parser.add_argument("--pricing-tolerance", type=float, default=0.01,
                        help="常规精确定价入列阈值，对应 C# reducedCost < -0.01")
    parser.add_argument("--heuristic-pricing-tolerance", type=float, default=1e-4)
    parser.add_argument("--certification-tolerance", type=float, default=1e-7,
                        help="保留原有严格定价复核容差，不能用 0.01 入列阈值替代最优性证明")
    parser.add_argument("--integer-tolerance", type=float, default=1e-5)
    parser.add_argument("--cut-violation-tolerance", type=float, default=1e-6)
    parser.add_argument("--time-limit", type=float, default=None, help="实例总时间预算（秒），在阶段边界检查")
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--heuristic-iterations", type=int, default=50)
    parser.add_argument("--heuristic-neighbors", type=int, default=10)
    parser.add_argument("--heuristic-max-same-objective", type=int, default=5)
    parser.add_argument("--use-neighborhood-search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-cuts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-schedule-enumeration", action=argparse.BooleanOptionalAction, default=False,
                        help="对应 C# routeEnumeration，默认关闭；模式 0/1 仅在开启时执行")
    parser.add_argument("--max-added-cuts", type=int, default=10)
    parser.add_argument("--enumeration-gap", type=float, default=0.10)
    parser.add_argument("--solver-log", action="store_true")
    parser.add_argument("--progress", action="store_true", help="逐阶段打印列数、节点证明下界和当前上界")
    parser.add_argument("--solver-threads", type=int, default=0, help="0 沿用 C# 未显式限制线程的 CPLEX 自动设置")
    parser.add_argument("--rmp-lp-method", type=int, choices=(0, 1, 2, 3, 4, 5, 6), default=1,
                        help="对应 C# ProduceModel 最终使用的 primal simplex (1)")
    parser.add_argument("--solver-optimality-tolerance", type=float, default=1e-6)
    parser.add_argument("--solver-numerical-emphasis", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def _add_acceleration_switch(
    parser: argparse.ArgumentParser,
    technology_name: str,
    *,
    help_text: str,
    cli_name: str | None = None,
) -> None:
    """注册一个默认开启且可通过 ``--no-use-*`` 关闭的加速技术开关。"""

    option_name = technology_name if cli_name is None else cli_name
    parser.add_argument(
        f"--use_{technology_name}",
        f"--use-{option_name.replace('_', '-')}",
        dest=f"use_{technology_name}",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=help_text,
    )
