"""连续完工时间桶的可插拔定义与默认等宽实现。

适用阶段：Bucket Graph 构造前，把每个作业的连续完工时间区间划分为
固定数量的桶；本模块不构造桶弧，也不执行标签扩展或支配。

Input:
    ``PricingProblem``；逐作业完工时间上界；每作业桶数；可选下界。
    上界将由后续 DSUB 模块提供，下界默认等于该作业在当前机器上的加工时间。
Output:
    不可变 ``TimeBucketPartition``，包含全部桶、逐节点桶索引和时间到桶的映射，
    供 ``bucket_graph.py``、桶支配和桶版标签算法使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from src.model.PP import PricingProblem


@dataclass(frozen=True, slots=True)
class TimeBucket:
    """一个节点上的连续完工时间区间。"""

    vertex: int
    index: int
    lower_bound: float
    upper_bound: float
    is_last: bool

    @property
    def key(self) -> tuple[int, int]:
        return self.vertex, self.index

    def contains(self, time: float, *, tolerance: float = 1e-9) -> bool:
        """按左闭右开规则判断时间；最后一个桶包含全局上界。"""

        lower_ok = time >= self.lower_bound - tolerance
        if self.is_last:
            return lower_ok and time <= self.upper_bound + tolerance
        return lower_ok and time < self.upper_bound - tolerance


@dataclass(frozen=True, slots=True)
class TimeBucketPartition:
    """单台机器全部节点的不可变时间桶划分。"""

    machine_id: int
    source_node: int
    sink_node: int
    buckets: tuple[TimeBucket, ...]
    buckets_by_vertex: tuple[tuple[int, tuple[TimeBucket, ...]], ...]

    def for_vertex(self, vertex: int) -> tuple[TimeBucket, ...]:
        return dict(self.buckets_by_vertex)[vertex]

    def bucket_for_time(
        self,
        vertex: int,
        time: float,
        *,
        tolerance: float = 1e-9,
    ) -> TimeBucket:
        """返回包含给定时间的桶；调用方负责保证时间处于节点窗口内。"""

        vertex_buckets = self.for_vertex(vertex)
        return next(
            bucket
            for bucket in vertex_buckets
            if bucket.contains(time, tolerance=tolerance)
        )


class TimeBucketBuilder(Protocol):
    """时间桶插件接口，可替换而不改变 Bucket Graph 或标签算法。"""

    def build(
        self,
        problem: PricingProblem,
        completion_time_upper_bounds: Mapping[int, float],
        *,
        num_buckets_per_job: int,
        completion_time_lower_bounds: Mapping[int, float] | None = None,
    ) -> TimeBucketPartition:
        """输入定价图和时间界，输出不可变时间桶划分。"""


@dataclass(frozen=True, slots=True)
class EqualWidthTimeBucketBuilder:
    """论文等宽划分 ``d_j=(dsub_j-lb_j)/R_j`` 的默认插件。"""

    tolerance: float = 1e-9

    def build(
        self,
        problem: PricingProblem,
        completion_time_upper_bounds: Mapping[int, float],
        *,
        num_buckets_per_job: int,
        completion_time_lower_bounds: Mapping[int, float] | None = None,
    ) -> TimeBucketPartition:
        lower_bounds = (
            {
                job_id: problem.processing_times[job_id - 1]
                for job_id in problem.job_ids
            }
            if completion_time_lower_bounds is None
            else dict(completion_time_lower_bounds)
        )
        horizon = max(completion_time_upper_bounds.values(), default=0.0)
        by_vertex: list[tuple[int, tuple[TimeBucket, ...]]] = [
            (
                problem.source_node,
                (TimeBucket(problem.source_node, 0, 0.0, 0.0, True),),
            )
        ]

        for job_id in problem.swpt_job_order:
            lower_bound = lower_bounds[job_id]
            upper_bound = completion_time_upper_bounds[job_id]
            bucket_count = (
                1
                if upper_bound <= lower_bound + self.tolerance
                else num_buckets_per_job
            )
            step_size = (upper_bound - lower_bound) / bucket_count
            job_buckets = tuple(
                TimeBucket(
                    vertex=job_id,
                    index=index,
                    lower_bound=lower_bound + index * step_size,
                    upper_bound=(
                        upper_bound
                        if index == bucket_count - 1
                        else lower_bound + (index + 1) * step_size
                    ),
                    is_last=index == bucket_count - 1,
                )
                for index in range(bucket_count)
            )
            by_vertex.append((job_id, job_buckets))

        by_vertex.append(
            (
                problem.sink_node,
                (TimeBucket(problem.sink_node, 0, 0.0, horizon, True),),
            )
        )
        buckets_by_vertex = tuple(by_vertex)
        return TimeBucketPartition(
            machine_id=problem.machine_id,
            source_node=problem.source_node,
            sink_node=problem.sink_node,
            buckets=tuple(
                bucket
                for _, vertex_buckets in buckets_by_vertex
                for bucket in vertex_buckets
            ),
            buckets_by_vertex=buckets_by_vertex,
        )


def build_time_buckets(
    problem: PricingProblem,
    completion_time_upper_bounds: Mapping[int, float],
    *,
    num_buckets_per_job: int = 4,
    completion_time_lower_bounds: Mapping[int, float] | None = None,
    builder: TimeBucketBuilder | None = None,
) -> TimeBucketPartition:
    """使用给定插件构造时间桶；未指定时使用论文的等宽划分。"""

    selected_builder = EqualWidthTimeBucketBuilder() if builder is None else builder
    return selected_builder.build(
        problem,
        completion_time_upper_bounds,
        num_buckets_per_job=num_buckets_per_job,
        completion_time_lower_bounds=completion_time_lower_bounds,
    )
