"""受限主问题及其求解结果的数据规格。

适用阶段：调度列产生后构造有限列上的 RMP，并接收 CPLEX 返回的对偶值。
输入：``schedulemodel.ScheduleColumn`` 与不可变实例。
输出： :class:`RestrictedMasterProblem` 和 :class:`DualSolution`，供 CPLEX 接口与 ``PP.py`` 使用。

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from src.model.dataload import SchedulingInstance
from src.model.schedulemodel import ScheduleColumn, build_set_partitioning_model


@dataclass(frozen=True, slots=True)
class CutRow:
    """已生成割在当前列集合上的系数；本阶段不负责分离该割。"""

    cut_id: str
    coefficients: tuple[float, ...]
    rhs: float = 1.0
    sense: str = "<="


class ColumnCutDefinition(Protocol):
    """
    针对割的语义定义; 并不涉及具体系数计算, 类似于将需要什么东西都交代出来。
    """

    cut_id: str
    rhs: float
    sense: str

    def coefficient(self, column: ScheduleColumn) -> float:
        """系数$\omega(C, M_k, \rho, s, k)$对应调度列的割系数。"""

    def transition(
        self,
        machine_id: int,
        job_id: int,
        state: float,
    ) -> tuple[float, int]:
        """返回标签扩展后的资源状态与割系数增量。"""


@dataclass(frozen=True, slots=True)
class RestrictedMasterProblem:
    """作者实现所用集合覆盖型 RMP 的完整系数规格。"""

    instance: SchedulingInstance
    columns: tuple[ScheduleColumn, ...]
    objective_coefficients: tuple[float, ...]
    job_cover_matrix: tuple[tuple[int, ...], ...]
    machine_capacity_matrix: tuple[tuple[int, ...], ...]
    cut_rows: tuple[CutRow, ...]
    job_rhs: tuple[float, ...]
    machine_rhs: tuple[float, ...]
    cut_definitions: tuple[ColumnCutDefinition, ...] = ()
    job_constraint_sense: str = ">="
    machine_constraint_sense: str = "<="
    variable_lower_bound: float = 0.0
    variable_upper_bound: float = 1.0

    @property
    def num_variables(self) -> int:
        return len(self.columns)

    @property
    def num_constraints(self) -> int:
        return self.instance.num_jobs + self.instance.num_machines + len(self.cut_rows)


@dataclass(frozen=True, slots=True)
class DualSolution:
    """CPLEX 求解 RMP 的 LP 松弛后返回的对偶值。"""

    job_duals: tuple[float, ...]
    machine_duals: tuple[float, ...]
    cut_ids: tuple[str, ...] = ()
    cut_duals: tuple[float, ...] = ()
    rmp_objective_value: float | None = None
    tolerance: float = 1e-9


def build_restricted_master(
    instance: SchedulingInstance,
    columns: Iterable[ScheduleColumn],
    *,
    cut_rows: Iterable[CutRow] = (),
    cut_definitions: Iterable[ColumnCutDefinition] = (),
) -> RestrictedMasterProblem:
    """由当前有限列构造 RMP，不判断该列集是否已经列生成最优。"""

    set_partitioning = build_set_partitioning_model(instance, columns)
    definitions = tuple(cut_definitions)
    cuts = tuple(cut_rows) + tuple(
        CutRow(
            cut_id=cut.cut_id,
            ## lm_subset_row_cut.py中coefficient()方法计算当前 lm-SRC 在一条完整调度列上的系数
            coefficients=tuple(cut.coefficient(column) for column in set_partitioning.columns), 
            rhs=cut.rhs,
            sense=cut.sense,
        )
        for cut in definitions
    )

    return RestrictedMasterProblem(
        instance=instance,
        columns=set_partitioning.columns,
        objective_coefficients=set_partitioning.objective_coefficients,
        job_cover_matrix=set_partitioning.job_cover_matrix,
        machine_capacity_matrix=set_partitioning.machine_capacity_matrix,
        cut_rows=cuts,
        cut_definitions=definitions,
        job_rhs=(1.0,) * instance.num_jobs,
        machine_rhs=(1.0,) * instance.num_machines,
    )
