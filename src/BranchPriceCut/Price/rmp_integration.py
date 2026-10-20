"""把语义化割定义转换成 RMP 行并重建当前有限模型。"""

from __future__ import annotations

from typing import Iterable

from src.BranchPriceCut.Cut.lm_subset_row_cut import LimitedMemorySubsetRowCut
from src.model.RMP import RestrictedMasterProblem, build_restricted_master
from src.model.schedulemodel import ScheduleColumn


def add_lm_subset_row_cuts(
    rmp: RestrictedMasterProblem,
    cuts: Iterable[LimitedMemorySubsetRowCut],
) -> RestrictedMasterProblem:
    """
    - 返回加入新 lm-SRC 后的 RMP; 
    - 原rmp中的变量集和割行集保持不变, 仅在原有基础上增加新割行;
    """

    raw_row_count = len(rmp.cut_rows) - len(rmp.cut_definitions)
    return build_restricted_master(
        rmp.instance,
        rmp.columns,
        cut_rows=rmp.cut_rows[:raw_row_count],
        cut_definitions=(*rmp.cut_definitions, *tuple(cuts)),
    )


def rebuild_rmp_with_cut_pool(
    rmp: RestrictedMasterProblem,
    cuts: Iterable[LimitedMemorySubsetRowCut],
) -> RestrictedMasterProblem:
    """
    用去重或扩充记忆集后的完整 lm-SRC 池替换原语义割。
    """

    raw_row_count = len(rmp.cut_rows) - len(rmp.cut_definitions)
    branch_and_other_rows = tuple(
        definition for definition in rmp.cut_definitions
        if not isinstance(definition, LimitedMemorySubsetRowCut)
    )
    return build_restricted_master(
        rmp.instance,
        rmp.columns,
        cut_rows=rmp.cut_rows[:raw_row_count],
        cut_definitions=(*branch_and_other_rows, *tuple(cuts)),
    )


def rebuild_rmp_with_columns(
    rmp: RestrictedMasterProblem,
    columns: Iterable[ScheduleColumn],
) -> RestrictedMasterProblem:
    """
    - 列集合更新后需要重新计算 RMP 的目标系数；
    - 原rmp中的割行集保持不变, 仅在原有基础上增加新列;
    """

    return build_restricted_master(
        rmp.instance,
        tuple(columns),
        cut_definitions=rmp.cut_definitions,
    )
