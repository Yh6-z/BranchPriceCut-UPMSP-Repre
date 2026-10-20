"""基于静态 SWPT DAG 和时间桶构造可插拔的前向 Bucket Graph。

适用阶段：时间桶构造之后、桶版标签算法之前。本模块只建立桶弧与同节点
component-wise-smaller 邻接关系，不保存标签、不执行支配，也不修改原 DAG。

Input:
    ``PricingProblem`` 的 DAG 弧及加工时间；``TimeBucketPartition``。
Output:
    不可变 ``BucketGraph``，包含桶弧、逐桶出弧查询、相邻较小桶关系和
    实际完成时间的重新定位接口，供桶版标签扩展和跨桶支配使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from src.BranchPriceCut.Price.Bucket.time_bucket import (
    TimeBucket,
    TimeBucketPartition,
)
from src.model.PP import PricingProblem


@dataclass(frozen=True, slots=True)
class BucketArc:
    """由一个尾桶、原 DAG 弧和按尾桶下界确定的基础头桶组成。"""

    tail_bucket: TimeBucket
    head_bucket: TimeBucket
    dag_arc: tuple[int, int]
    processing_time: float


@dataclass(frozen=True, slots=True)
class BucketGraph:
    """单台机器的不可变前向 Bucket Graph。"""

    machine_id: int
    partition: TimeBucketPartition
    arcs: tuple[BucketArc, ...]
    adjacent_smaller_buckets: tuple[tuple[TimeBucket, TimeBucket], ...]
    _outgoing: dict = field(init=False, repr=False, compare=False)
    _smaller: dict = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """建立一次性邻接索引，避免每次标签扩展扫描全部桶弧。

        输入：构造后的 BucketGraph。输出：None；派生索引不参与图的值比较。
        """

        outgoing = {bucket: [] for bucket in self.partition.buckets}
        for arc in self.arcs:
            outgoing[arc.tail_bucket].append(arc)
        object.__setattr__(self, "_outgoing", {bucket: tuple(arcs) for bucket, arcs in outgoing.items()})
        object.__setattr__(self, "_smaller", dict(self.adjacent_smaller_buckets))

    def outgoing(self, bucket: TimeBucket) -> tuple[BucketArc, ...]:
        """读取某桶出弧。

        输入：TimeBucket。输出：对应的 BucketArc 元组。
        """

        return self._outgoing[bucket]

    def adjacent_smaller(self, bucket: TimeBucket) -> TimeBucket | None:
        """读取同节点相邻较小桶。

        输入：TimeBucket。输出：TimeBucket 或 None。
        """

        return self._smaller.get(bucket)

    def locate(self, vertex: int, time: float) -> TimeBucket:
        """按标签实际完成时间定位桶，而非强制使用弧的基础头桶。"""

        return self.partition.bucket_for_time(vertex, time)


class BucketGraphBuilder(Protocol):
    """Bucket Graph 插件接口，可独立替换图构造策略。"""

    def build(
        self,
        problem: PricingProblem,
        partition: TimeBucketPartition,
    ) -> BucketGraph:
        """输入定价 DAG 和时间桶，输出不可变 Bucket Graph。"""


@dataclass(frozen=True, slots=True)
class ForwardBucketGraphBuilder:
    """论文前向 Bucket Graph 的默认构造插件。"""

    def build(
        self,
        problem: PricingProblem,
        partition: TimeBucketPartition,
    ) -> BucketGraph:
        arcs: list[BucketArc] = []
        for tail, head in problem.dag_arcs:
            processing_time = (
                0.0
                if head == problem.sink_node
                else problem.processing_times[head - 1]
            )
            for tail_bucket in partition.for_vertex(tail):
                if head == problem.sink_node:
                    head_bucket = partition.for_vertex(head)[0]
                else:
                    arrival_at_lower_bound = (
                        tail_bucket.lower_bound + processing_time
                    )
                    head_buckets = partition.for_vertex(head)
                    if arrival_at_lower_bound > head_buckets[-1].upper_bound:
                        continue
                    head_bucket = partition.bucket_for_time(
                        head,
                        arrival_at_lower_bound,
                    )
                arcs.append(
                    BucketArc(
                        tail_bucket=tail_bucket,
                        head_bucket=head_bucket,
                        dag_arc=(tail, head),
                        processing_time=processing_time,
                    )
                )

        adjacent_smaller = tuple(
            (vertex_buckets[index], vertex_buckets[index - 1])
            for vertex in problem.swpt_job_order
            for vertex_buckets in (partition.for_vertex(vertex),)
            for index in range(1, len(vertex_buckets))
        )
        return BucketGraph(
            machine_id=problem.machine_id,
            partition=partition,
            arcs=tuple(arcs),
            adjacent_smaller_buckets=adjacent_smaller,
        )


def build_bucket_graph(
    problem: PricingProblem,
    partition: TimeBucketPartition,
    *,
    builder: BucketGraphBuilder | None = None,
) -> BucketGraph:
    """使用给定插件构造 Bucket Graph；未指定时使用默认前向构造器。"""

    selected_builder = ForwardBucketGraphBuilder() if builder is None else builder
    return selected_builder.build(problem, partition)
