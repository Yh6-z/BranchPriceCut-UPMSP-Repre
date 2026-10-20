"""精确定价使用的前向标签与节点内支配。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ForwardLabel:
    last_node: int
    time: float
    reduced_cost: float
    job_ids: tuple[int, ...]
    cut_states: tuple[float, ...]
    cut_coefficients: tuple[float, ...]


def dominates(
    left: ForwardLabel,
    right: ForwardLabel,
    *,
    tolerance: float = 1e-9,
) -> bool:
    return (
        left.last_node == right.last_node
        and left.cut_states == right.cut_states
        and left.time <= right.time + tolerance
        and left.reduced_cost <= right.reduced_cost + tolerance
        and (
            left.time < right.time - tolerance
            or left.reduced_cost < right.reduced_cost - tolerance
        )
    )


def insert_nondominated(
    labels: list[ForwardLabel],
    candidate: ForwardLabel,
    *,
    tolerance: float = 1e-9,
) -> bool:
    """插入不可支配标签，并合并未来扩展等价的状态。

    输入：ForwardLabel 列表、候选 ForwardLabel、比较容差 float。
    输出：插入成功的 bool；相同末节点/割状态/时间/成本只保留一个代表。
    SWPT DAG 上这些标签具有相同可用后缀，路径历史不影响后续定价。
    """

    if any(label.last_node == candidate.last_node and label.cut_states == candidate.cut_states
           and label.time == candidate.time and label.reduced_cost == candidate.reduced_cost
           for label in labels):
        return False
    if any(dominates(label, candidate, tolerance=tolerance) for label in labels):
        return False
    labels[:] = [
        label
        for label in labels
        if not dominates(candidate, label, tolerance=tolerance)
    ]
    labels.append(candidate)
    return True
