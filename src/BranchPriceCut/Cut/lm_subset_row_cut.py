"""Limited-memory subset-row cut 的定义、记忆集和列系数。"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Iterable, Sequence

from src.model.schedulemodel import ScheduleColumn


@dataclass(frozen=True, slots=True)
class LimitedMemorySubsetRowCut:
    """一条 lm-SRC: 基集 ``C``、乘子 ``\rho`` 与逐机器记忆集 ``M_k``。

    主线算法使用 ``|C|=3`` 和 ``rho=1/2``。记忆状态在路径离开 ``M_k``
    时清零；访问 ``C`` 中的作业时增加 ``rho``，每累计到 1 产生一个割系数。
    """

    cut_id: str
    subset: tuple[int, ...]
    memory_sets: tuple[frozenset[int], ...]
    multiplier: float = 0.5
    violation: float = 0.0
    sense: str = "<="

    @property
    def rhs(self) -> float:
        # rho * |C| 向下取整
        return float(floor(len(self.subset) * self.multiplier))     

    def memory_set(self, machine_id: int) -> frozenset[int]:
        # 返回不同机器的记忆集M_k
        return self.memory_sets[machine_id - 1]

    def transition(
        self,
        machine_id: int,
        job_id: int,
        state: float,
    ) -> tuple[float, int]:
        """沿一个作业节点扩展有限记忆状态，返回新状态和系数增量。
        附录算法5中作业的state变化功能;
        """

        if job_id not in self.memory_set(machine_id):
            return 0.0, 0
        if job_id not in self.subset:
            return state, 0

        new_state = state + self.multiplier
        increment = 0
        while new_state >= 1.0 - 1e-12:
            increment += 1
            new_state -= 1.0
        return new_state, increment

    def coefficient(self, column: ScheduleColumn) -> float:
        """
        附录算法5: 计算当前 lm-SRC 在一条完整调度列上的系数 $\omega(C, M_k, \rho, s, k)$。
        
        """

        state = 0.0
        coefficient = 0
        for job_id in column.job_ids:
            state, increment = self.transition(column.machine_id, job_id, state)
            coefficient += increment
        return float(coefficient)


def build_machine_memory_set(
    columns: Sequence[ScheduleColumn],
    column_values: Sequence[float],
    subset: Iterable[int],
    *,
    multiplier: float = 0.5,
    positive_tolerance: float = 1e-9,   # 接近0
) -> frozenset[int]:
    """附录算法6"""

    base = frozenset(subset)
    memory = set(base)
    for column, value in zip(columns, column_values):
        if value <= positive_tolerance or base.isdisjoint(column.job_ids):
            continue

        state = 0.0
        between: set[int] = set()
        for job_id in column.job_ids:
            if job_id in base:
                state += multiplier
                if state >= 1.0 - 1e-12:
                    memory.update(between)
                    between.clear()
                    state -= 1.0
            elif state > 0.0:
                between.add(job_id)
    return frozenset(memory)


def merge_memory_sets(
    current: LimitedMemorySubsetRowCut,
    candidate: LimitedMemorySubsetRowCut,
) -> LimitedMemorySubsetRowCut:
    """同一基集再次被分离时，逐机器扩充而不缩小已有记忆集。
    原文并未提及, 维护cut_pool的方法; 
    """

    return LimitedMemorySubsetRowCut(
        cut_id=current.cut_id,
        subset=current.subset,
        memory_sets=tuple(
            old.union(new)
            for old, new in zip(current.memory_sets, candidate.memory_sets)
        ),
        multiplier=current.multiplier,
        violation=max(current.violation, candidate.violation),
        sense=current.sense,
    )
