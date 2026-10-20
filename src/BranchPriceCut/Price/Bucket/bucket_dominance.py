"""Bucket Graph 标签的桶内与跨桶安全支配插件。

适用阶段：桶版前向标签搜索。该模块只判断和维护标签支配，不扩展标签、
不构造桶图，也不读取求解器状态。

Input:
    候选 ``ForwardLabel``、所在 ``TimeBucket``、``BucketGraph``、逐桶标签集合，
    以及已完成桶累计的最小约化成本 ``r_best``。
Output:
    支配判断布尔值、候选是否成功插入，以及某桶更新后的累计 ``r_best``，
    供桶版标签算法决定剪枝、保留和跨桶搜索提前停止。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Mapping, Protocol, Sequence

from src.BranchPriceCut.Price.Bucket.bucket_graph import BucketGraph
from src.BranchPriceCut.Price.labels import (
    ForwardLabel,
    dominates,
    insert_nondominated,
)
from src.BranchPriceCut.Price.Bucket.time_bucket import TimeBucket


class BucketDominancePolicy(Protocol):
    """可替换的桶支配策略接口。"""

    def dominated_in_bucket(
        self,
        candidate: ForwardLabel,
        labels: Sequence[ForwardLabel],
    ) -> bool:
        """输入候选与同桶标签，输出候选是否被安全支配。"""

    def dominated_in_smaller_buckets(
        self,
        candidate: ForwardLabel,
        bucket: TimeBucket,
        graph: BucketGraph,
        labels_by_bucket: Mapping[TimeBucket, Sequence[ForwardLabel]],
        best_reduced_cost: Mapping[TimeBucket, float],
    ) -> bool:
        """输入桶图运行状态，输出候选是否被较小桶中的标签支配。"""

    def insert(
        self,
        labels: list[ForwardLabel],
        candidate: ForwardLabel,
    ) -> bool:
        """输入同桶标签和候选，删除被支配项并输出候选是否插入。"""

    def update_best_reduced_cost(
        self,
        bucket: TimeBucket,
        graph: BucketGraph,
        labels: Sequence[ForwardLabel],
        best_reduced_cost: Mapping[TimeBucket, float],
    ) -> float:
        """输入已处理桶状态，输出当前桶累计的 ``r_best``。"""


@dataclass(frozen=True, slots=True)
class SafeBucketDominance:
    """复用非桶安全支配规则的默认桶支配插件。"""

    tolerance: float = 1e-9

    def dominated_in_bucket(
        self,
        candidate: ForwardLabel,
        labels: Sequence[ForwardLabel],
    ) -> bool:
        return any(
            dominates(label, candidate, tolerance=self.tolerance)
            for label in labels
        )

    def dominated_in_smaller_buckets(
        self,
        candidate: ForwardLabel,
        bucket: TimeBucket,
        graph: BucketGraph,
        labels_by_bucket: Mapping[TimeBucket, Sequence[ForwardLabel]],
        best_reduced_cost: Mapping[TimeBucket, float],
    ) -> bool:
        smaller = graph.adjacent_smaller(bucket)
        while smaller is not None:
            if (
                best_reduced_cost.get(smaller, inf)
                > candidate.reduced_cost + self.tolerance
            ):
                break
            if self.dominated_in_bucket(
                candidate,
                labels_by_bucket.get(smaller, ()),
            ):
                return True
            smaller = graph.adjacent_smaller(smaller)
        return False

    def insert(
        self,
        labels: list[ForwardLabel],
        candidate: ForwardLabel,
    ) -> bool:
        return insert_nondominated(
            labels,
            candidate,
            tolerance=self.tolerance,
        )

    def update_best_reduced_cost(
        self,
        bucket: TimeBucket,
        graph: BucketGraph,
        labels: Sequence[ForwardLabel],
        best_reduced_cost: Mapping[TimeBucket, float],
    ) -> float:
        own_best = min(
            (label.reduced_cost for label in labels),
            default=inf,
        )
        smaller = graph.adjacent_smaller(bucket)
        if smaller is None:
            return own_best
        return min(own_best, best_reduced_cost.get(smaller, inf))
