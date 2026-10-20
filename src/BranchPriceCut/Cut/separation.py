"""在当前分数 RMP 解上分离三作业 limited-memory SRC。"""

from __future__ import annotations

from itertools import combinations, islice
from typing import Iterable

from src.BranchPriceCut.Cut.lm_subset_row_cut import (
    LimitedMemorySubsetRowCut,
    build_machine_memory_set,
)
from src.model.RMP import RestrictedMasterProblem


def separate_lm_subset_row_cuts(
    rmp: RestrictedMasterProblem,
    column_values: Iterable[float],
    *,
    max_candidates: int | None = None,
    max_added: int = 10,
    violation_tolerance: float = 1e-6,
    multiplier: float = 0.5,
) -> tuple[LimitedMemorySubsetRowCut, ...]:
    """
    枚举三作业基集，返回 SRC 违反度最大的至多 ``max_added`` 条新割。
    lhs 和 rhs 分别表示割的左端和右端，违反度为 lhs - rhs。违反度小于 ``violation_tolerance`` 的割不返回。
    """

    values = tuple(column_values)
    subsets = combinations(rmp.instance.job_ids, 3)
    if max_candidates is not None:
        subsets = islice(subsets, max_candidates)
    existing = {
        tuple(sorted(cut.subset))
        for cut in rmp.cut_definitions
        if hasattr(cut, "subset")
    }
    candidates: list[LimitedMemorySubsetRowCut] = []
    for subset in subsets:
        if subset in existing:
            continue

        lhs = sum(
            value
            for column, value in zip(rmp.columns, values)
            if len(set(column.job_ids).intersection(subset)) > 1
        )
        violation = lhs - 1.0
        if violation <= violation_tolerance:
            continue

        memory_sets = tuple(
            build_machine_memory_set(
                tuple(
                    column
                    for column in rmp.columns
                    if column.machine_id == machine_id
                ),
                tuple(
                    value
                    for column, value in zip(rmp.columns, values)
                    if column.machine_id == machine_id
                ),
                subset,
                multiplier=multiplier,
            )
            for machine_id in rmp.instance.machine_ids
        )
        candidates.append(
            LimitedMemorySubsetRowCut(
                cut_id=f"lm-src[{','.join(map(str, subset))}]",
                subset=subset,
                memory_sets=memory_sets,
                multiplier=multiplier,
                violation=violation,
            )
        )

    candidates.sort(key=lambda cut: (-cut.violation, cut.subset))
    return tuple(candidates[:max_added])
