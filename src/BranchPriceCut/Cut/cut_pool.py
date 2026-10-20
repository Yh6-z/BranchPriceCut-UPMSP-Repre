"""lm-SRC 去重以及动态记忆集扩充。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.BranchPriceCut.Cut.lm_subset_row_cut import (
    LimitedMemorySubsetRowCut,
    merge_memory_sets,
)


@dataclass(frozen=True, slots=True)
class CutPoolUpdate:
    cuts: tuple[LimitedMemorySubsetRowCut, ...]
    added_cut_ids: tuple[str, ...]
    expanded_cut_ids: tuple[str, ...]


def update_cut_pool(
    current_cuts: Iterable[LimitedMemorySubsetRowCut],
    candidates: Iterable[LimitedMemorySubsetRowCut],
) -> CutPoolUpdate:
    """按基集去重；重复割合并记忆集，新基集追加到池尾。"""

    pool = list(current_cuts)
    index_by_subset = {cut.subset: index for index, cut in enumerate(pool)}
    added: list[str] = []
    expanded: list[str] = []
    for candidate in candidates:
        index = index_by_subset.get(candidate.subset)
        if index is None:
            index_by_subset[candidate.subset] = len(pool)
            pool.append(candidate)
            added.append(candidate.cut_id)
            continue

        merged = merge_memory_sets(pool[index], candidate)
        if merged.memory_sets != pool[index].memory_sets:
            pool[index] = merged
            expanded.append(merged.cut_id)
    return CutPoolUpdate(tuple(pool), tuple(added), tuple(expanded))