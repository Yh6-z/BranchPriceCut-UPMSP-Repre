"""定价标签中的 lm-SRC 有限记忆资源更新。"""

from __future__ import annotations

from typing import Sequence

from src.model.RMP import ColumnCutDefinition


def initial_cut_states(cuts: Sequence[ColumnCutDefinition]) -> tuple[float, ...]:
    return (0.0,) * len(cuts) # ————> (0.0, 0.0, 0.0, ...) 每个割的初始状态为0.0


def extend_cut_resources(
    cuts: Sequence[ColumnCutDefinition],
    cut_duals: Sequence[float],         # 割的对偶值列表
    machine_id: int,
    job_id: int,
    states: Sequence[float],
    coefficients: Sequence[float],      # 一个调度的割系数列表，[w_1, w_2, w_3, ...]
) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    """扩展所有非鲁棒割资源，并返回约化成本的割增量。"""

    new_states: list[float] = []
    new_coefficients = list(coefficients)
    reduced_cost_increment = 0.0
    for index, (cut, dual, state) in enumerate(zip(cuts, cut_duals, states)):
        new_state, coefficient_increment = cut.transition(machine_id, job_id, state)
        new_states.append(new_state)
        new_coefficients[index] += coefficient_increment
        reduced_cost_increment -= dual * coefficient_increment
    return tuple(new_states), tuple(new_coefficients), reduced_cost_increment
