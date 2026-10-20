"""在节点局部列池上求缩减 IP，或先逐机器枚举调度再求 IP。

适用阶段：节点列—割生成收敛后的可选整数解加速。
输入：节点的 RMP/局部 DAG/DSUB、当前对偶与整数上下界、模式和标签预算。
输出：有限列整数解、补充列、逐机器搜索统计及枚举完整性。
模式 0 只使用已有合法列；模式 1 按约化成本阈值补充完整调度。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import cplex

from src.BranchPriceCut.Branch.branching import BranchNode, pricing_requests_for_node
from src.BranchPriceCut.Price.cut_resource import extend_cut_resources, initial_cut_states
from src.BranchPriceCut.Price.labels import ForwardLabel
from src.model.PP import PricingProblem, evaluate_column_reduced_cost
from src.model.RMP import DualSolution, RestrictedMasterProblem
from src.model.dataload import SchedulingInstance
from src.model.schedulemodel import ScheduleColumn, build_schedule_column


@dataclass(frozen=True, slots=True)
class MachineEnumeration:
    """单机枚举所得列、标签数和完整性。"""

    machine_id: int
    columns: tuple[ScheduleColumn, ...]
    generated_labels: int
    complete: bool


@dataclass(frozen=True, slots=True)
class ScheduleEnumerationResult:
    """缩减 IP 解及可供主流程判断的枚举状态。"""

    mode: int
    status: str
    columns: tuple[ScheduleColumn, ...]
    added_columns: tuple[ScheduleColumn, ...]
    selected_columns: tuple[ScheduleColumn, ...]
    objective_value: float | None
    machine_searches: tuple[MachineEnumeration, ...]
    enumeration_complete: bool
    improves_incumbent: bool


def _column_is_local(
    column: ScheduleColumn,
    problem: PricingProblem,
    completion_bounds: dict[int, float],
    *,
    tolerance: float,
) -> bool:
    """检查旧列是否仍属于当前机器的局部图和时间窗口。

    输入：``ScheduleColumn``、``PricingProblem``、逐作业上界 ``dict[int, float]``、容差 ``float``。
    输出：该列仍可在当前局部状态使用时为 ``bool=True``。
    """

    return (
        column.machine_id == problem.machine_id
        and set(column.arcs) <= set(problem.dag_arcs)
        and (not column.completion_times
             or column.completion_times[-1]
             <= completion_bounds.get(problem.sink_node, float("inf")) + tolerance)
        and all(
            time <= completion_bounds.get(job_id, float("inf")) + tolerance
            for job_id, time in zip(column.job_ids, column.completion_times)
        )
    )


def enumerate_machine_schedules(
    instance: SchedulingInstance,
    problem: PricingProblem,
    completion_bounds: dict[int, float],
    *,
    reduced_cost_limit: float,
    max_labels: int = 10000,
    tolerance: float = 1e-6,
) -> MachineEnumeration:
    """不使用定价支配，枚举阈值内的单机完整调度。

    输入：实例、当前机器 DAG/真实对偶 ``PricingProblem``、DSUB 上界、
    约化成本阈值 ``float``、生成标签上限 ``int`` 和容差 ``float``。
    输出：``MachineEnumeration``，包含去重列、生成标签数和预算完整性。
    """

    definitions = {definition.cut_id: definition for definition in problem.cut_definitions}
    cuts = tuple(definitions[cut_id] for cut_id in problem.cut_ids)
    labels = {vertex: [] for vertex in problem.dag_vertices}
    labels[problem.source_node].append(ForwardLabel(
        last_node=problem.source_node,
        time=0.0,
        reduced_cost=-problem.machine_dual,
        job_ids=(),
        cut_states=initial_cut_states(cuts),
        cut_coefficients=(0.0,) * len(cuts),
    ))
    adjacency = {vertex: [] for vertex in problem.dag_vertices}
    for tail, head in problem.dag_arcs:
        adjacency[tail].append(head)

    columns: dict[tuple[int, tuple[int, ...]], ScheduleColumn] = {}
    generated = 1
    complete = True
    for tail in problem.dag_vertices[:-1]:
        if not complete:
            break
        for label in labels[tail]:
            if not complete:
                break
            for head in adjacency[tail]:
                if head == problem.sink_node:
                    if not label.job_ids:
                        continue
                    if label.time > completion_bounds.get(head, float("inf")) + tolerance:
                        continue
                    if generated >= max_labels:
                        complete = False
                        break
                    generated += 1
                    column = build_schedule_column(instance, problem.machine_id, label.job_ids)
                    evaluation = evaluate_column_reduced_cost(problem, column)
                    if evaluation.reduced_cost <= reduced_cost_limit + tolerance:
                        columns[column.key] = column
                    continue

                finish = label.time + instance.processing_time(head, problem.machine_id)
                if finish > completion_bounds.get(head, float("inf")) + tolerance:
                    continue
                if generated >= max_labels:
                    complete = False
                    break
                states, coefficients, cut_increment = extend_cut_resources(
                    cuts,
                    problem.cut_duals,
                    problem.machine_id,
                    head,
                    label.cut_states,
                    label.cut_coefficients,
                )
                labels[head].append(ForwardLabel(
                    last_node=head,
                    time=finish,
                    reduced_cost=(label.reduced_cost
                                  + instance.weight(head) * finish
                                  - problem.job_duals[head - 1]
                                  + cut_increment),
                    job_ids=(*label.job_ids, head),
                    cut_states=states,
                    cut_coefficients=coefficients,
                ))
                generated += 1

    return MachineEnumeration(problem.machine_id, tuple(columns.values()), generated, complete)


def _solve_schedule_ip(
    rmp: RestrictedMasterProblem,
    columns: tuple[ScheduleColumn, ...],
    time_limit: float | None = None,
) -> tuple[str, float | None, tuple[ScheduleColumn, ...]]:
    """按作业等式和机器容量在有限列上求二进制集合划分 IP。

    输入：当前节点 RestrictedMasterProblem、去重列元组及可选秒数 float。
    输出：求解状态 ``str``、目标 ``float | None``、所选调度列元组。
    """

    model = cplex.Cplex()
    model.objective.set_sense(model.objective.sense.minimize)
    model.parameters.emphasis.numerical.set(1)
    model.parameters.simplex.tolerances.optimality.set(1e-6)
    if time_limit is not None:
        model.parameters.timelimit.set(max(0.001, time_limit))
    model.set_log_stream(None)
    model.set_error_stream(None)
    model.set_warning_stream(None)
    model.set_results_stream(None)
    names = [f"y[{index}]" for index in range(len(columns))]
    model.variables.add(
        obj=[column.cost for column in columns],
        types=[model.variables.type.binary] * len(columns),
        names=names,
    )
    for job_id in rmp.instance.job_ids:
        indices = [index for index, column in enumerate(columns) if job_id in column.job_ids]
        model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=[names[index] for index in indices],
                                       val=[1.0] * len(indices))],
            senses=["E"], rhs=[1.0], names=[f"job[{job_id}]"],
        )
    for machine_id in rmp.instance.machine_ids:
        indices = [index for index, column in enumerate(columns)
                   if column.machine_id == machine_id]
        model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=[names[index] for index in indices],
                                       val=[1.0] * len(indices))],
            senses=["L"], rhs=[1.0], names=[f"machine[{machine_id}]"],
        )

    definitions = {definition.cut_id: definition for definition in rmp.cut_definitions}
    original_index = {column.key: index for index, column in enumerate(rmp.columns)}
    for row in rmp.cut_rows:
        definition = definitions.get(row.cut_id)
        coefficients = [
            row.coefficients[original_index[column.key]]
            if column.key in original_index
            else definition.coefficient(column)
            for column in columns
        ]
        model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=names, val=coefficients)],
            senses=[{"=": "E", "<=": "L", ">=": "G"}[row.sense]],
            rhs=[row.rhs], names=[row.cut_id],
        )

    try:
        model.solve()
    except cplex.exceptions.CplexSolverError as error:
        model.end()
        if error.args[-1] == 1016:
            return "solver_size_limit", None, ()
        raise
    status = model.solution.get_status_string()
    if model.solution.is_primal_feasible():
        values = model.solution.get_values()
        selected = tuple(column for column, value in zip(columns, values) if value > 0.5)
        objective = sum(column.cost for column in selected)
    else:
        selected = ()
        objective = None
    model.end()
    return status, objective, selected


def solve_schedule_enumeration(
    node: BranchNode,
    *,
    mode: int = 0,
    dual_solution: DualSolution | None = None,
    lower_bound: float | None = None,
    incumbent_objective: float | None = None,
    max_labels_per_machine: int = 10000,
    tolerance: float = 1e-6,
    time_limit: float | None = None,
) -> ScheduleEnumerationResult:
    """按启动模式求当前列池 IP，或先用标签补充稍逊列再求 IP。

    输入：局部 ``BranchNode``、模式 ``int``（0/1），可选真实对偶、
    已证实下界、整数上界、单机标签预算、容差和可选 IP 秒数 float。
    输出：``ScheduleEnumerationResult``；模式 1 缺少有效上下界/对偶时
    返回跳过状态，不改变节点；预算耗尽时有限列 IP 仍可用于寻优。
    """

    if mode not in (0, 1):
        return ScheduleEnumerationResult(
            mode, "skipped_invalid_mode", (), (), (), None, (), False, False
        )
    if mode == 1 and (
        dual_solution is None
        or lower_bound is None
        or incumbent_objective is None
        or not isfinite(lower_bound)
        or not isfinite(incumbent_objective)
        or incumbent_objective < lower_bound
    ):
        return ScheduleEnumerationResult(
            mode, "skipped_missing_bounds_or_dual", (), (), (), None, (), False, False
        )

    if mode == 1:
        assert dual_solution is not None
        assert lower_bound is not None
        assert incumbent_objective is not None
        definitions = {definition.cut_id: definition
                       for definition in node.rmp.cut_definitions}
        if (
            len(dual_solution.job_duals) != node.rmp.instance.num_jobs
            or len(dual_solution.machine_duals) != node.rmp.instance.num_machines
            or tuple(row.cut_id for row in node.rmp.cut_rows) != dual_solution.cut_ids
            or len(dual_solution.cut_ids) != len(dual_solution.cut_duals)
            or any(row.cut_id not in definitions or
                   not hasattr(definitions[row.cut_id], "transition")
                   for row in node.rmp.cut_rows)
            or any(problem.dag_arcs != state.dag_arcs
                   for problem, state in zip(node.pricing_problems, node.bounds))
        ):
            return ScheduleEnumerationResult(
                mode, "skipped_unsupported_snapshot", (), (), (),
                None, (), False, False,
            )
        requests = pricing_requests_for_node(node, dual_solution)
    else:
        requests = node.pricing_problems
    bounds_by_machine = {state.machine_id: state for state in node.bounds}
    problems_by_machine = {problem.machine_id: problem for problem in requests}
    valid_columns_list: list[ScheduleColumn] = []
    for column in node.rmp.columns:
        problem = problems_by_machine[column.machine_id]
        state = bounds_by_machine[column.machine_id]
        completion_bounds = state.completion_time_upper_bounds()
        completion_bounds[problem.sink_node] = state.global_upper_bound
        if _column_is_local(column, problem, completion_bounds, tolerance=tolerance):
            valid_columns_list.append(column)
    valid_columns = tuple(valid_columns_list)
    unique = {column.key: column for column in valid_columns}
    searches: list[MachineEnumeration] = []
    added: list[ScheduleColumn] = []
    if mode == 1:
        for request in requests:
            state = bounds_by_machine[request.machine_id]
            completion_bounds = state.completion_time_upper_bounds()
            completion_bounds[request.sink_node] = state.global_upper_bound
            search = enumerate_machine_schedules(
                node.rmp.instance,
                request,
                completion_bounds,
                reduced_cost_limit=incumbent_objective - lower_bound + 1.0,
                max_labels=max_labels_per_machine,
                tolerance=tolerance,
            )
            searches.append(search)
            for column in search.columns:
                if column.key not in unique:
                    unique[column.key] = column
                    added.append(column)

    columns = tuple(unique.values())
    status, objective, selected = _solve_schedule_ip(node.rmp, columns, time_limit)
    return ScheduleEnumerationResult(
        mode=mode,
        status=status,
        columns=columns,
        added_columns=tuple(added),
        selected_columns=selected,
        objective_value=objective,
        machine_searches=tuple(searches),
        enumeration_complete=mode == 1 and all(search.complete for search in searches),
        improves_incumbent=(objective is not None and incumbent_objective is not None
                            and objective < incumbent_objective - tolerance),
    )
