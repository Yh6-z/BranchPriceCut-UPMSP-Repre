"""根节点的可选列表调度初始化。

适用阶段：读取实例后、建立根节点 RMP 和 DSUB 之前。
输入：dataload 提供的 SchedulingInstance，以及 config 中的初始化开关。
输出：逐机器完整调度、初始列池与可行整数上界，交给主流程。
关闭时返回空列池和空上界；作者主流程使用开启模式，不实现空列池可行性恢复。
"""

from dataclasses import dataclass
from math import fsum

from src.model.dataload import SchedulingInstance
from src.model.schedulemodel import MachineSchedule, ScheduleColumn, build_schedule_column, swpt_order


@dataclass(frozen=True, slots=True)
class RootColumnInitialization:
    """初始列、各机器调度及可选整数上界；空闲机器保留空调度。"""

    enabled: bool
    columns: tuple[ScheduleColumn, ...]
    schedules: tuple[MachineSchedule, ...]
    upper_bound: float | None


def initialize_root_columns(
    instance: SchedulingInstance,
    *,
    enabled: bool = True,
) -> RootColumnInitialization:
    """按 C# 的逐机器轮转、最小 p/w 规则构造根节点初始列。

    输入：实例 SchedulingInstance 和是否启用列表调度的 bool。
    输出：RootColumnInitialization；开启时每个作业恰好分配一次，
    每台非空闲机器提供一列，upper_bound 为实际总加权完工时间。
    关闭时 columns/schedules 为空元组，upper_bound 为 None。
    """

    if not enabled:
        return RootColumnInitialization(False, (), (), None)

    orders = {machine: swpt_order(instance, machine) for machine in instance.machine_ids}
    remaining = set(instance.job_ids)
    assigned: dict[int, list[int]] = {machine: [] for machine in instance.machine_ids}
    while remaining:
        for machine in instance.machine_ids:
            if not remaining:
                break
            job = next(job for job in orders[machine] if job in remaining)
            assigned[machine].append(job)
            remaining.remove(job)

    schedules = tuple(
        MachineSchedule(machine, tuple(assigned[machine])) for machine in instance.machine_ids
    )
    columns = tuple(
        build_schedule_column(instance, schedule.machine_id, schedule.job_ids)
        for schedule in schedules if schedule.job_ids
    )
    return RootColumnInitialization(True, columns, schedules, fsum(column.cost for column in columns))
