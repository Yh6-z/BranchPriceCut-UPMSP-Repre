"""从调度列解聚合作业弧，并构造保持 RMP、DAG 和列池一致的子节点。

输入：节点的有限列 RMP、各机器局部定价图与 DSUB、列变量解及分支决定。
输出：分数作业弧候选、带 0/1 分支行及局部图的独立子节点，或节点/全局求解记录。
记录接口供主流程接收各方法结果，保存阶段快照、用时、LB/UB 和 incumbent 历史。
RMP 的决策变量仍为 y；弧表达式只用于选枝和施加分支约束。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import fsum, isfinite
from typing import TYPE_CHECKING, Literal, Sequence

from src.BranchPriceCut.Price.dynamic_upper_bound import (
    DynamicUpperBoundState,
    delete_dag_arcs_and_update_bounds,
)
from src.model.PP import PricingProblem, build_pricing_problems
from src.model.RMP import DualSolution, RestrictedMasterProblem, build_restricted_master
from src.model.dataload import SchedulingInstance
from src.model.schedulemodel import ScheduleColumn, build_schedule_column

if TYPE_CHECKING:
    from src.BranchPriceCut.solve import RMPSolveResult


BranchKind = Literal["x", "q"]


@dataclass(frozen=True, slots=True)
class BranchCandidate:
    """一个分数作业弧表达式及其当前 LP 值。"""

    kind: BranchKind
    tail: int
    head: int
    value: float
    machine_id: int | None = None

    @property
    def key(self) -> tuple[str, int, int, int]:
        """返回候选的稳定身份键。

        输入：本候选 ``BranchCandidate``。
        输出：类型、机器、弧尾和弧头组成的 ``tuple[str, int, int, int]``。
        """

        return self.kind, self.machine_id or 0, self.tail, self.head


@dataclass(frozen=True, slots=True)
class BranchDecision:
    """一条祖先或当前节点已生效的 0/1 作业弧决定。"""

    kind: BranchKind
    tail: int
    head: int
    value: Literal[0, 1]
    machine_id: int | None = None

    @property
    def row_id(self) -> str:
        """返回唯一分支行标识。

        输入：本决定 ``BranchDecision``。
        输出：包含类型、机器、弧和取值的 ``str``。
        """

        machine = self.machine_id or 0
        return f"branch[{self.kind},{machine},{self.tail},{self.head},{self.value}]"


@dataclass(frozen=True, slots=True)
class ArcBranchRow:
    """使用调度列弧系数表达 x/q=0/1 的语义 RMP 行。"""

    decision: BranchDecision
    sense: str = "="

    @property
    def cut_id(self) -> str:
        """返回 RMP 行名。

        输入：本行 ``ArcBranchRow``。
        输出：与决定对应的 ``str``。
        """

        return self.decision.row_id

    @property
    def rhs(self) -> float:
        """返回分支右端值。

        输入：本行 ``ArcBranchRow``。
        输出：0 或 1 的 ``float``。
        """

        return float(self.decision.value)

    def coefficient(self, column: ScheduleColumn) -> float:
        """计算调度列在分支行上的系数。

        输入：机器调度列 ``ScheduleColumn``。
        输出：目标机器及目标弧均匹配时为 1，否则为 0 的 ``float``。
        """

        applies = self.decision.kind == "q" or column.machine_id == self.decision.machine_id
        return float(applies and (self.decision.tail, self.decision.head) in column.arcs)

    def transition(
        self,
        machine_id: int,
        job_id: int,
        state: float,
    ) -> tuple[float, int]:
        """沿路径记住前一作业并累计目标弧。

        输入：机器编号 ``int``、当前作业编号 ``int``、前驱作业状态 ``float``。
        输出：新状态和 0/1 系数增量组成的 ``tuple[float, int]``。
        """

        applies = self.decision.kind == "q" or machine_id == self.decision.machine_id
        increment = int(applies and int(state) == self.decision.tail and job_id == self.decision.head)
        return float(job_id), increment


@dataclass(frozen=True, slots=True)
class BranchNode:
    """分支节点的 RMP、局部 DAG/DSUB 与全部祖先决定的不可变快照。"""

    rmp: RestrictedMasterProblem
    pricing_problems: tuple[PricingProblem, ...]
    bounds: tuple[DynamicUpperBoundState, ...]
    decisions: tuple[BranchDecision, ...] = ()


def make_branch_node(
    rmp: RestrictedMasterProblem,
    pricing_problems: Sequence[PricingProblem],
    bounds: Sequence[DynamicUpperBoundState],
) -> BranchNode:
    """封装已建立的局部状态。

    输入：``RestrictedMasterProblem``、机器 ``PricingProblem`` 序列和 DSUB 状态序列。
    输出：不可变的 ``BranchNode`` 快照。
    """

    problems = tuple(pricing_problems)
    states = tuple(bounds)
    assert len(problems) == len(states) == rmp.instance.num_machines
    assert all(problem.machine_id == state.machine_id for problem, state in zip(problems, states))
    return BranchNode(rmp, problems, states)


def pricing_requests_for_node(
    node: BranchNode,
    dual_solution: DualSolution,
) -> tuple[PricingProblem, ...]:
    """用当前 RMP 对偶及局部 DAG 构造逐机器定价请求。

    输入：``BranchNode`` 和当前 LP 的 ``DualSolution``。
    输出：按机器顺序排列的 ``tuple[PricingProblem, ...]``。
    """

    requests = build_pricing_problems(node.rmp, dual_solution)
    return tuple(
        replace(request, dag_arcs=state.dag_arcs)
        for request, state in zip(requests, node.bounds)
    )


def aggregate_job_arcs(
    rmp: RestrictedMasterProblem,
    column_values: Sequence[float],
    *,
    tolerance: float = 1e-5,
) -> tuple[BranchCandidate, ...]:
    """从列变量解聚合分数作业弧。

    输入：``RestrictedMasterProblem``、与列顺序一致的 ``Sequence[float]`` 和容差。
    输出：按 0.5 距离及身份键排序的 ``tuple[BranchCandidate, ...]``。
    """

    assert len(column_values) == len(rmp.columns)
    n = rmp.instance.num_jobs
    x_values: dict[tuple[int, int, int], float] = {}
    for column, value in zip(rmp.columns, column_values):
        if value <= tolerance:
            continue
        for tail, head in column.arcs:
            if 1 <= tail <= n and 1 <= head <= n:
                key = column.machine_id, tail, head
                x_values[key] = x_values.get(key, 0.0) + value

    q_values: dict[tuple[int, int], float] = {}
    candidates: list[BranchCandidate] = []
    for (machine_id, tail, head), value in x_values.items():
        q_values[tail, head] = q_values.get((tail, head), 0.0) + value
        if tolerance < value < 1.0 - tolerance:
            candidates.append(BranchCandidate("x", tail, head, value, machine_id))
    for (tail, head), value in q_values.items():
        if tolerance < value < 1.0 - tolerance:
            candidates.append(BranchCandidate("q", tail, head, value))
    return tuple(sorted(candidates, key=lambda item: (abs(item.value - 0.5), item.key)))


def ordinary_branch_candidate(
    candidates: Sequence[BranchCandidate],
    *,
    kind: BranchKind = "x",
) -> BranchCandidate | None:
    """选择指定类型中最靠近 0.5 的分数弧。

    输入：已排序的 ``Sequence[BranchCandidate]`` 和 ``BranchKind``。
    输出：``BranchCandidate``；无该类分数弧时为 ``None``。
    """

    return next((candidate for candidate in candidates if candidate.kind == kind), None)


def _removed_arcs(
    problem: PricingProblem,
    decision: BranchDecision,
) -> tuple[tuple[int, int], ...]:
    """按 0/1 决定计算本机待删 DAG 弧。

    输入：当前 ``PricingProblem`` 和 ``BranchDecision``。
    输出：待删除弧的 ``tuple[tuple[int, int], ...]``。
    """

    target = decision.tail, decision.head
    if decision.value == 0:
        if decision.kind == "x" and problem.machine_id != decision.machine_id:
            return ()
        return tuple(arc for arc in problem.dag_arcs if arc == target)
    if decision.kind == "x" and problem.machine_id != decision.machine_id:
        return tuple(
            arc for arc in problem.dag_arcs
            if decision.tail in arc or decision.head in arc
        )
    return tuple(
        (tail, head) for tail, head in problem.dag_arcs
        if (tail == decision.tail and head != decision.head)
        or (head == decision.head and tail != decision.tail)
    )


def create_child_node(
    parent: BranchNode,
    candidate: BranchCandidate,
    value: Literal[0, 1],
) -> BranchNode:
    """施加一个作业弧决定并构造子节点。

    输入：父 ``BranchNode``、分数 ``BranchCandidate`` 和 0/1 整数取值。
    输出：RMP、列池、局部 DAG/DSUB 和决定链独立的 ``BranchNode``。
    """

    decision = BranchDecision(candidate.kind, candidate.tail, candidate.head, value, candidate.machine_id)
    problems: list[PricingProblem] = []
    states: list[DynamicUpperBoundState] = []
    for problem, state in zip(parent.pricing_problems, parent.bounds):
        update = delete_dag_arcs_and_update_bounds(
            problem, state, _removed_arcs(problem, decision)
        )
        states.append(update.state)
        problems.append(replace(problem, dag_arcs=update.state.dag_arcs))

    allowed_by_machine = {
        problem.machine_id: frozenset(problem.dag_arcs) for problem in problems
    }
    columns = tuple(
        column for column in parent.rmp.columns
        if set(column.arcs) <= allowed_by_machine[column.machine_id]
    )
    raw_count = len(parent.rmp.cut_rows) - len(parent.rmp.cut_definitions)
    rmp = build_restricted_master(
        parent.rmp.instance,
        columns,
        cut_rows=parent.rmp.cut_rows[:raw_count],
        cut_definitions=(*parent.rmp.cut_definitions, ArcBranchRow(decision)),
    )
    return BranchNode(rmp, tuple(problems), tuple(states), (*parent.decisions, decision))


@dataclass(frozen=True, slots=True)
class NodeStageRecord:
    """一次节点操作及其只读快照；LP 解和对偶始终绑定当次列/割/局部图。"""

    stage: str
    status: str
    node: BranchNode
    elapsed_seconds: float
    solution: RMPSolveResult | None
    exact_pricing_complete: bool
    infeasibility_proven: bool
    lower_bound: float | None
    upper_bound: float | None
    added_columns: int
    removed_columns: int
    added_cut_rows: int
    removed_dag_arcs: int


@dataclass(frozen=True, slots=True)
class NodeSolveRecord:
    """节点身份、父子关系、当前快照、已证实下界和按时间顺序保存的阶段记录。"""

    node_id: int
    parent_id: int | None
    depth: int
    node: BranchNode
    lower_bound: float | None = None
    stages: tuple[NodeStageRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class IncumbentSolution:
    """经核验的完整整数调度、实际成本、来源节点和发现时的累计用时。"""

    columns: tuple[ScheduleColumn, ...]
    objective_value: float
    source_node_id: int | None
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class SolveInformation:
    """一个实例的节点求解记录及历次改善的 incumbent；不持有求解器对象。"""

    instance: SchedulingInstance
    nodes: tuple[NodeSolveRecord, ...] = ()
    incumbent_history: tuple[IncumbentSolution, ...] = ()

    @property
    def upper_bound(self) -> float | None:
        """读取当前全局整数上界。

        输入：本实例的 SolveInformation。
        输出：最后一次合法改善的目标值 float，无 incumbent 时为 None。
        """

        return self.incumbent_history[-1].objective_value if self.incumbent_history else None


def register_node_information(
    information: SolveInformation,
    node: BranchNode,
    node_id: int,
    *,
    parent_id: int | None = None,
) -> SolveInformation:
    """登记根节点或子节点，并仅继承父节点的已证实下界。

    输入：SolveInformation、BranchNode 快照、唯一 int 节点编号、可选 int 父编号。
    输出：新增 NodeSolveRecord 后的 SolveInformation；子节点不继承父 LP 解或对偶。
    """

    assert node.rmp.instance == information.instance
    assert all(record.node_id != node_id for record in information.nodes)
    parent = None if parent_id is None else next(
        record for record in information.nodes if record.node_id == parent_id
    )
    record = NodeSolveRecord(
        node_id, parent_id, 0 if parent is None else parent.depth + 1,
        node, None if parent is None else parent.lower_bound,
    )
    return replace(information, nodes=(*information.nodes, record))


def record_node_stage(
    information: SolveInformation,
    node_id: int,
    node: BranchNode,
    stage: str,
    *,
    solution: RMPSolveResult | None = None,
    exact_pricing_complete: bool = False,
    infeasibility_proven: bool = False,
    status: str = "processing",
    elapsed_seconds: float = 0.0,
    pricing_lower_bound: float | None = None,
) -> SolveInformation:
    """记录一次 RMP、定价、割、固定或分支操作及当次状态。

    输入：SolveInformation、int 节点号、当次 BranchNode、str 阶段/状态、
    可选 RMPSolveResult、全机器精确定价/完整不可行性证明 bool 和本次 float 用时。
    输出：追加阶段后的 SolveInformation。仅当前最优 LP 及其全机器精确定价证书
    同时存在时更新 LB；受限 RMP 的 infeasible 状态本身不构成不可行性证明。
    证书由调用方针对本次快照提供；图、列或割变化后的证明不可跨快照复用。
    可选 pricing_lower_bound: float 为计入定价数值残差后的保守下界。
    """

    record = next(record for record in information.nodes if record.node_id == node_id)
    assert node.rmp.instance == information.instance
    lower_bound = record.lower_bound
    if exact_pricing_complete:
        assert solution is not None and solution.status == "optimal"
        assert isfinite(solution.objective_value)
        assert len(solution.column_values) == len(node.rmp.columns)
        assert solution.dual_solution.cut_ids == tuple(row.cut_id for row in node.rmp.cut_rows)
        certified = solution.objective_value if pricing_lower_bound is None else pricing_lower_bound
        assert isfinite(certified) and certified <= solution.objective_value + 1e-9
        lower_bound = max(
            certified,
            lower_bound if lower_bound is not None else float("-inf"),
        )
    old_keys = {column.key for column in record.node.rmp.columns}
    new_keys = {column.key for column in node.rmp.columns}
    old_rows = {row.cut_id for row in record.node.rmp.cut_rows}
    new_rows = {row.cut_id for row in node.rmp.cut_rows}
    old_arcs = {(state.machine_id, *arc) for state in record.node.bounds for arc in state.dag_arcs}
    new_arcs = {(state.machine_id, *arc) for state in node.bounds for arc in state.dag_arcs}
    event = NodeStageRecord(
        stage, status, node, elapsed_seconds, solution,
        exact_pricing_complete, infeasibility_proven, lower_bound, information.upper_bound,
        len(new_keys - old_keys), len(old_keys - new_keys), len(new_rows - old_rows),
        len(old_arcs - new_arcs),
    )
    updated = replace(record, node=node, lower_bound=lower_bound, stages=(*record.stages, event))
    return replace(information, nodes=tuple(
        updated if item.node_id == node_id else item for item in information.nodes
    ))


def validate_integer_schedule(
    instance: SchedulingInstance,
    columns: Sequence[ScheduleColumn],
    *,
    node: BranchNode | None = None,
    tolerance: float = 1e-6,
) -> bool:
    """核验候选是否为原问题及可选节点上的完整整数调度。

    输入：SchedulingInstance、选中列 Sequence[ScheduleColumn]、可选 BranchNode 和 float 容差。
    输出：bool；检查机器唯一、作业恰好一次、真实成本/路径、局部 DAG/DSUB、割和分支行。
    """

    selected = tuple(columns)
    if len({column.machine_id for column in selected}) != len(selected):
        return False
    jobs = [job for column in selected for job in column.job_ids]
    if sorted(jobs) != sorted(instance.job_ids):
        return False
    for column in selected:
        if column.machine_id not in instance.machine_ids:
            return False
        canonical = build_schedule_column(instance, column.machine_id, column.job_ids)
        if column != canonical:
            return False
    if node is None:
        return True
    if node.rmp.instance != instance:
        return False
    states = {state.machine_id: state for state in node.bounds}
    for column in selected:
        state = states[column.machine_id]
        if not set(column.arcs) <= set(state.dag_arcs):
            return False
        if column.completion_times and column.completion_times[-1] > state.global_upper_bound + tolerance:
            return False
        if any(time > state.bound(job).upper_bound + tolerance
               for job, time in zip(column.job_ids, column.completion_times)):
            return False
    for decision in node.decisions:
        if abs(fsum(ArcBranchRow(decision).coefficient(column) for column in selected)
               - decision.value) > tolerance:
            return False
    definitions = {definition.cut_id: definition for definition in node.rmp.cut_definitions}
    positions = {column.key: i for i, column in enumerate(node.rmp.columns)}
    for row in node.rmp.cut_rows:
        definition = definitions.get(row.cut_id)
        if definition is None and any(column.key not in positions for column in selected):
            return False
        value = fsum(
            definition.coefficient(column) if definition is not None
            else row.coefficients[positions[column.key]] for column in selected
        )
        if ((row.sense == "=" and abs(value - row.rhs) > tolerance)
                or (row.sense == "<=" and value > row.rhs + tolerance)
                or (row.sense == ">=" and value < row.rhs - tolerance)):
            return False
    return True


def record_incumbent(
    information: SolveInformation,
    columns: Sequence[ScheduleColumn],
    *,
    source_node_id: int | None = None,
    elapsed_seconds: float = 0.0,
    tolerance: float = 1e-6,
) -> SolveInformation:
    """统一接收初始调度、整数 LP 或有限列 IP 产生的 incumbent。

    输入：SolveInformation、选中列序列、可选 int 来源节点、float 累计用时与容差。
    输出：候选合法且改善 UB 时追加历史的 SolveInformation，否则返回原对象。
    有限列 IP 的最优状态不会在此转化为节点下界或原问题最优性证明。
    """

    node = None if source_node_id is None else next(
        record.node for record in information.nodes if record.node_id == source_node_id
    )
    if not validate_integer_schedule(information.instance, columns, node=node, tolerance=tolerance):
        return information
    objective = fsum(column.cost for column in columns)
    if not isfinite(objective) or (
        information.upper_bound is not None and objective >= information.upper_bound - tolerance
    ):
        return information
    incumbent = IncumbentSolution(tuple(columns), objective, source_node_id, elapsed_seconds)
    return replace(information, incumbent_history=(*information.incumbent_history, incumbent))


def node_pruning_reason(
    record: NodeSolveRecord,
    upper_bound: float | None,
    *,
    tolerance: float = 1e-6,
) -> str | None:
    """根据已有证书判断节点是否可剪枝。

    输入：NodeSolveRecord、可选 float 全局 UB 和比较容差。
    输出：proven_infeasible、bound 或 None；试算目标与有限列不可行状态不用于剪枝。
    """

    if any(event.infeasibility_proven and event.node == record.node for event in record.stages):
        return "proven_infeasible"
    if (record.lower_bound is not None and upper_bound is not None
            and record.lower_bound >= upper_bound - tolerance):
        return "bound"
    return None


def frontier_lower_bound(
    records: Sequence[NodeSolveRecord],
    *,
    upper_bound: float | None = None,
) -> float | None:
    """汇总尚未解决节点的已证实下界。

    输入：主流程提供的完整前沿 Sequence[NodeSolveRecord]（包括正在处理的节点）和可选 UB。
    输出：float 全局下界；任一节点缺少 LB 或前沿为空时返回 None。
    搜索完成时的最终证明由主流程处理，不从空前沿自动推断最优。
    """

    if not records or any(record.lower_bound is None for record in records):
        return None
    lower_bound = min(record.lower_bound for record in records if record.lower_bound is not None)
    return lower_bound if upper_bound is None else min(lower_bound, upper_bound)
