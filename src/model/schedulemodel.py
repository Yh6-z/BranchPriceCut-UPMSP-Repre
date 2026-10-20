"""UPMSP-WCT 原问题、调度列和集合划分模型的数据规格。

`dataload.py`执行算例读取后, 构造机器调度、SWPT 顺序、路径列以及完整集合划分模型的有限表示。

输入: ``dataload.SchedulingInstance``。
输出: schedual column与 :class:`SetPartitioningModel`，供 ``RMP.py`` 使用。

"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import fsum
from typing import Iterable, Sequence

from src.model.dataload import SchedulingInstance

# frozen表示数据不可修改; slots表示属性不可添加; 
@dataclass(frozen=True, slots=True)
class MachineSchedule:
    """
    一台机器上的有序作业序列，编号均为从 1 开始的外部编号。
    机器k上的调度集合Omega_k
    """

    machine_id: int
    job_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True) 
class ScheduleColumn:
    """
    集合划分模型中的一条机器调度列。
    """

    machine_id: int
    job_ids: tuple[int, ...]
    completion_times: tuple[float, ...]
    cost: float
    coverage: tuple[int, ...]
    arcs: tuple[tuple[int, int], ...]

    @property
    def key(self) -> tuple[int, tuple[int, ...]]:
        return self.machine_id, self.job_ids


@dataclass(frozen=True, slots=True)
class SetPartitioningModel:
    """
    完整集合划分模型在当前显式列集合上的系数规格。

    每台机器上的可行调度为 ``Omega_k``；这里的 ``columns``只是一组可检查的显式列，不代表已经枚举完整模型。
    """

    instance: SchedulingInstance
    columns: tuple[ScheduleColumn, ...]
    objective_coefficients: tuple[float, ...]
    job_cover_matrix: tuple[tuple[int, ...], ...]
    machine_capacity_matrix: tuple[tuple[int, ...], ...]
    job_constraint_sense: str = "="
    machine_constraint_sense: str = "<="
    variable_domain: str = "binary"

    @property
    def num_variables(self) -> int:
        return len(self.columns)

    @property
    def num_constraints(self) -> int:
        return self.instance.num_jobs + self.instance.num_machines


def swpt_order(
    instance: SchedulingInstance,
    machine_id: int,
    job_ids: Iterable[int] | None = None,
) -> tuple[int, ...]:
    """按非递减 $p_{jk}/w_j$ 排序，相同比值时按作业编号排序。"""

    selected = tuple(instance.job_ids if job_ids is None else job_ids)

    def ratio_key(job_id: int) -> tuple[Fraction, int]:
        processing_time = Fraction(str(instance.processing_time(job_id, machine_id)))
        return processing_time / instance.weight(job_id), job_id

    return tuple(sorted(selected, key=ratio_key))


def schedule_cost(
    instance: SchedulingInstance,
    schedule: MachineSchedule,
) -> tuple[tuple[float, ...], float]:
    """计算一个调度序列内各作业完工时间和总加权完工时间。"""

    elapsed = 0.0
    completion_times: list[float] = []
    contributions: list[float] = []
    for job_id in schedule.job_ids:
        elapsed += instance.processing_time(job_id, schedule.machine_id)
        completion_times.append(elapsed)
        contributions.append(instance.weight(job_id) * elapsed)
    return tuple(completion_times), fsum(contributions)


def build_schedule_column(
    instance: SchedulingInstance,
    machine_id: int,
    job_ids: Sequence[int],
    *,
    enforce_swpt: bool = True,
) -> ScheduleColumn:
    """把一条机器调度转换为目标系数、覆盖向量和路径弧。"""

    ordered_jobs = (
        swpt_order(instance, machine_id, job_ids)
        if enforce_swpt
        else tuple(job_ids)
    )

    schedule = MachineSchedule(machine_id=machine_id, job_ids=ordered_jobs)
    completion_times, cost = schedule_cost(instance, schedule)
    selected = set(ordered_jobs)
    coverage = tuple(1 if job_id in selected else 0 for job_id in instance.job_ids)  # 覆盖系数alpha_jk

    sink_id = instance.num_jobs + 1
    path_vertices = (0, *ordered_jobs, sink_id)
    # zip()函数将两个序列依次配对；zip((1,3,5),(2,4,6)) -> (1,2),(3,4),(5,6);
    # 如果遇到长度不等的序列，则以最短的为准，如zip((1,3,5),(2,4)) -> (1,2),(3,4)
    arcs = tuple(zip(path_vertices, path_vertices[1:]))     
    return ScheduleColumn(
        machine_id=machine_id,
        job_ids=ordered_jobs,
        completion_times=completion_times,
        cost=cost,
        coverage=coverage,
        arcs=arcs,
    )


def build_set_partitioning_model(
    instance: SchedulingInstance,
    columns: Iterable[ScheduleColumn],
) -> SetPartitioningModel:
    """构造有限显式列上的集合划分系数，作为完整 SP 的可检查规格。"""

    ordered_columns = tuple(columns)
    objective = tuple(column.cost for column in ordered_columns)
    job_cover_matrix = tuple(
        tuple(column.coverage[job_index] for column in ordered_columns)
        for job_index in range(instance.num_jobs)
    )
    machine_capacity_matrix = tuple(
        tuple(1 if column.machine_id == machine_id else 0 for column in ordered_columns)
        for machine_id in instance.machine_ids
    )
    return SetPartitioningModel(
        instance=instance,
        columns=ordered_columns,
        objective_coefficients=objective,
        job_cover_matrix=job_cover_matrix,
        machine_capacity_matrix=machine_capacity_matrix,
    )
