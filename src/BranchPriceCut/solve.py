"""使用 CPLEX 求解受限主问题的连续线性松弛。

适用阶段：列生成每轮求解当前 RMP。
上游输入：``src.model.RMP.RestrictedMasterProblem``。
下游输出：列变量值、约化成本、目标值和 ``DualSolution``，供 Price 中的
定价算法以及后续 Branch、Cut 模块使用。

本模块只求解当前受限主问题，不显式建立对偶模型，也不执行列生成循环。
"""

from __future__ import annotations

from dataclasses import dataclass

import cplex

from src.model.RMP import DualSolution, RestrictedMasterProblem


@dataclass(frozen=True, slots=True)
class RMPSolveResult:
    """一次 RMP 线性松弛的 CPLEX 求解结果。"""

    status: str
    objective_value: float | None
    column_values: tuple[float, ...]
    column_reduced_costs: tuple[float, ...]
    dual_solution: DualSolution | None


def solve_restricted_master(
    rmp: RestrictedMasterProblem,
    *,
    log_output: bool = False,
    time_limit: float | None = None,
    threads: int = 0,
    lp_method: int = 1,
    optimality_tolerance: float = 1e-6,
    numerical_emphasis: bool = True,
) -> RMPSolveResult:
    """求解连续 RMP，并仅在最优时读取可用于定价的解与对偶。

    输入：RMP、日志/数值强调 bool、可选秒数 float、线程/LP 方法 int 和最优性容差 float。
    输出：RMPSolveResult；非最优时保留真实状态，目标/对偶为 None。
    """

    model = cplex.Cplex()
    model.objective.set_sense(model.objective.sense.minimize)
    model.parameters.threads.set(threads)
    model.parameters.lpmethod.set(lp_method)
    model.parameters.simplex.tolerances.optimality.set(optimality_tolerance)
    model.parameters.emphasis.numerical.set(int(numerical_emphasis))
    if time_limit is not None:
        model.parameters.timelimit.set(max(0.001, time_limit))

    if not log_output:
        model.set_log_stream(None)
        model.set_error_stream(None)
        model.set_warning_stream(None)
        model.set_results_stream(None)

    variable_names = [f"y[{index}]" for index in range(rmp.num_variables)]
    model.variables.add(
        obj=list(rmp.objective_coefficients),
        lb=[rmp.variable_lower_bound] * rmp.num_variables,
        # 机器容量已蕴含 y<=1；去掉冗余上界，避免其对偶造成已有列负 RC。
        ub=[cplex.infinity] * rmp.num_variables,
        names=variable_names,
    )

    for job_index, job_id in enumerate(rmp.instance.job_ids):
        _add_constraint(
            model,
            variable_names,
            rmp.job_cover_matrix[job_index],
            rmp.job_constraint_sense,
            rmp.job_rhs[job_index],
            f"job[{job_id}]",
        )

    for machine_index, machine_id in enumerate(rmp.instance.machine_ids):
        _add_constraint(
            model,
            variable_names,
            rmp.machine_capacity_matrix[machine_index],
            rmp.machine_constraint_sense,
            rmp.machine_rhs[machine_index],
            f"machine[{machine_id}]",
        )

    for cut in rmp.cut_rows:
        _add_constraint(
            model,
            variable_names,
            cut.coefficients,
            cut.sense,
            cut.rhs,
            f"cut[{cut.cut_id}]",
        )

    try:
        model.solve()
    except cplex.exceptions.CplexSolverError as error:
        # 默认参数的真实 JD 测试触发 1016；只识别这一已观测的许可证规模限制。
        model.end()
        if error.args[-1] == 1016:
            return RMPSolveResult("solver_size_limit", None, (), (), None)
        raise

    status = model.solution.get_status_string()
    if model.solution.get_status() != model.solution.status.optimal:
        model.end()
        return RMPSolveResult(status, None, (), (), None)

    objective_value = model.solution.get_objective_value()
    column_values = tuple(model.solution.get_values())
    column_reduced_costs = tuple(model.solution.get_reduced_costs())
    constraint_duals = tuple(model.solution.get_dual_values())

    job_end = rmp.instance.num_jobs
    machine_end = job_end + rmp.instance.num_machines
    dual_solution = DualSolution(
        job_duals=constraint_duals[:job_end],
        machine_duals=constraint_duals[job_end:machine_end],
        cut_ids=tuple(cut.cut_id for cut in rmp.cut_rows),
        cut_duals=constraint_duals[machine_end:],
        rmp_objective_value=objective_value,
    )
    result = RMPSolveResult(
        status=status,
        objective_value=objective_value,
        column_values=column_values,
        column_reduced_costs=column_reduced_costs,
        dual_solution=dual_solution,
    )
    model.end()
    return result


def _add_constraint(
    model: cplex.Cplex,
    variable_names: list[str],
    coefficients: tuple[float, ...] | tuple[int, ...],
    sense: str,
    rhs: float,
    name: str,
) -> None:
    """按 RMP 中保存的行系数向 CPLEX 添加一条线性约束。"""

    sense_code = {">=": "G", "<=": "L", "=": "E"}[sense]
    model.linear_constraints.add(
        lin_expr=[
            cplex.SparsePair(
                ind=variable_names,
                val=[float(value) for value in coefficients],
            )
        ],
        senses=[sense_code],
        rhs=[float(rhs)],
        names=[name],
    )
