# 脚本功能与求解顺序

## 脚本功能简介

### 入口与模型

| 脚本 | 功能 |
|---|---|
| `config.py` | 定义并解析实例选择、求解参数和算法开关。 |
| `main.py` | 组织实例读取、列—割生成、固定删弧、调度枚举及分支搜索，汇总并保存求解结果。 |
| `src/model/dataload.py` | 从 Excel 读取作业权重和各机器加工时间，构造调度实例。 |
| `src/model/schedulemodel.py` | 按 SWPT 规则排列单机作业，计算调度成本并构造调度列和集合划分模型。 |
| `src/model/RMP.py` | 用当前列池构造受限主问题（RMP），定义约束、割行及对偶数据。 |
| `src/model/PP.py` | 根据 RMP 对偶构造各机器的定价问题、SWPT 有向无环图，并复核调度列的约化成本。 |
| `src/BranchPriceCut/solve.py` | 使用 CPLEX 求解当前 RMP 的线性松弛，返回列变量值、目标值和对偶。 |

### 定价与时间桶

| 脚本 | 功能 |
|---|---|
| `src/BranchPriceCut/Price/labels.py` | 定义前向标签，执行标签支配判断与非支配标签保留。 |
| `src/BranchPriceCut/Price/forward_labeling.py` | 执行非桶前向标签搜索，返回负约化成本列；也提供固定删弧所需的前向标签。 |
| `src/BranchPriceCut/Price/dynamic_upper_bound.py` | 计算动态逐作业完工时间上界（DSUB），在删弧后更新局部图与时间界。 |
| `src/BranchPriceCut/Price/cut_resource.py` | 初始化并扩展标签中的割资源，计算割对偶对约化成本的贡献。 |
| `src/BranchPriceCut/Price/rmp_integration.py` | 将新列或割池合入 RMP，重算列对应的割系数。 |
| `src/BranchPriceCut/Price/Bucket/time_bucket.py` | 根据完工时间界划分各顶点的时间桶，并定位标签所属桶。 |
| `src/BranchPriceCut/Price/Bucket/bucket_graph.py` | 将定价 DAG 与时间桶组合为桶图，提供桶弧及出弧索引。 |
| `src/BranchPriceCut/Price/Bucket/bucket_dominance.py` | 在当前桶与较小时间桶之间执行安全标签支配判断。 |
| `src/BranchPriceCut/Price/Bucket/bucket_forward_labeling.py` | 在桶图上执行前向标签精确定价，结合时间界与桶支配筛选新负列。 |
| `src/heuristic/heuristic_extension.py` | 沿桶弧随机扩展标签，构造完整单机调度路径。 |
| `src/heuristic/neighborhood_search.py` | 对调度路径进行桶弧变异与邻域搜索，收集新负约化成本列。 |
| `src/heuristic/heuristic_pricing.py` | 提供限制标签数量的启发式定价，以及邻域搜索后回退到非桶精确定价的封装接口；主流程由 `main.py` 直接调度。 |

### 割、加速与分支

| 脚本 | 功能 |
|---|---|
| `src/BranchPriceCut/Cut/lm_subset_row_cut.py` | 定义有限记忆子集行割（lm-SRC）、列系数及标签资源转移规则。 |
| `src/BranchPriceCut/Cut/separation.py` | 从当前 RMP 分数解中搜索违反的三作业 lm-SRC。 |
| `src/BranchPriceCut/Cut/cut_pool.py` | 合并候选割并去重，维护当前割池。 |
| `src/BranchPriceCut/accelerate/RootNodeColumnInitial.py` | 通过列表调度构造根节点初始列池和可行整数上界。 |
| `src/BranchPriceCut/accelerate/reduced-cost-fixing/backward_labeling.py` | 生成后向标签，为逐弧约化成本界提供后缀信息。 |
| `src/BranchPriceCut/accelerate/reduced-cost-fixing/label_joining.py` | 拼接前向与后向标签，评估经过指定弧的约化成本界。 |
| `src/BranchPriceCut/accelerate/reduced-cost-fixing/variable_fixing.py` | 在定价证书和上下界满足条件时安全删弧，更新 DSUB 并过滤失效列。 |
| `src/BranchPriceCut/accelerate/enumeration.py` | 在当前列池上求整数解，或先枚举补充调度列再求整数解，以改善可行上界；默认关闭。 |
| `src/BranchPriceCut/accelerate/strong_branching.py` | 通过 RMP 和启发式列生成试算评分，选择强分支候选；支持回退普通分支。 |
| `src/BranchPriceCut/Branch/branching.py` | 聚合作业弧、构造分支候选与子节点，同步局部图和列池，并维护上下界、节点记录和 incumbent。 |

### 包接口

| 脚本 | 功能 |
|---|---|
| `src/BranchPriceCut/Price/__init__.py` | 导出标签定价、时间桶、DSUB 和 RMP 集成接口。 |
| `src/BranchPriceCut/Price/Bucket/__init__.py` | 导出时间桶、桶图、桶支配和桶标签定价接口。 |
| `src/BranchPriceCut/Cut/__init__.py` | 导出 lm-SRC 定义、割分离与割池接口。 |
| `src/BranchPriceCut/Branch/__init__.py` | 导出分支、节点状态与求解记录接口。 |
| `src/BranchPriceCut/accelerate/__init__.py` | 标识可选加速技术包。 |
| `src/BranchPriceCut/accelerate/reduced-cost-fixing/__init__.py` | 标识约化成本固定删弧包。 |
| `src/heuristic/__init__.py` | 导出随机路径扩展、邻域搜索与启发式定价接口。 |

### 验证脚本

| 脚本 | 功能 |
|---|---|
| `tests/test_root_node_column_initial.py` | 验证根列初始化、RMP 可行性和关闭初始化时的空列池行为。 |
| `tests/test_dynamic_upper_bound.py` | 验证动态完工时间界计算与删弧后的更新。 |
| `tests/test_time_bucket_and_bucket_graph.py` | 验证时间桶划分与桶图构造。 |
| `tests/test_bucket_dominance_and_labeling.py` | 验证桶支配与桶标签定价结果。 |
| `tests/test_heuristic_neighborhood_pricing.py` | 验证随机扩展、邻域定价和精确定价回退。 |
| `tests/test_lm_src_and_pricing.py` | 验证 lm-SRC 分离、割资源及定价与 RMP 的集成。 |
| `tests/test_reduced_cost_fixing.py` | 用小型完整路径枚举检查固定删弧的安全性。 |
| `tests/test_schedule_enumeration.py` | 验证两种调度枚举模式及节点局部约束。 |
| `tests/test_branching.py` | 验证作业弧聚合、子节点状态一致性和强分支。 |
| `tests/test_solve_information.py` | 验证求解阶段记录、下界继承及 incumbent 合法性。 |
| `tests/test_main.py` | 验证完整求解流程，提供人工实例枚举对照与 JD 实例批量验收入口。 |

## 按求解逻辑执行的顺序示意

`-->` 表示前一脚本的结果进入下一脚本；`+` 表示脚本相互协同；上下横线与左右竖线组成的方框圈出协同脚本组。下图使用文件名简写，目录见上表；各算法模块由 `main.py` 调用。

入口与根节点初始化：

```text
-----------------------
| main.py + config.py | --> dataload.py
-----------------------
           -----------------------------------------------
       --> | RootNodeColumnInitial.py + schedulemodel.py |
           -----------------------------------------------
       --> RMP.py --> PP.py --> dynamic_upper_bound.py
```

节点列生成：先求 RMP，再用其对偶构造各机器定价请求。启发式找到新负列后直接回到 RMP；未找到时进入精确定价。

```text
RMP.py --> solve.py --> PP.py
           -------------------------------------------------------------
       --> | dynamic_upper_bound.py + time_bucket.py + bucket_graph.py |
           -------------------------------------------------------------
           -----------------------------------------------------
       --> | labels.py + cut_resource.py                       |
           | + heuristic_extension.py + neighborhood_search.py |
           -----------------------------------------------------

找到新负列：neighborhood_search.py --> rmp_integration.py --> RMP.py --> solve.py

未找到新负列：PP.py
           -------------------------------------------------------------
       --> | dynamic_upper_bound.py + time_bucket.py + bucket_graph.py |
           | + labels.py + cut_resource.py + bucket_dominance.py       |
           | + bucket_forward_labeling.py                              |
           -------------------------------------------------------------
       --> rmp_integration.py --> RMP.py --> solve.py
```

上述精确定价仅在找到新负列时进入 `rmp_integration.py`；全机器均确认无新负列后进入割分离。

关闭时间桶或桶图时，改用非桶标签路径；启发式阶段限制标签数量，无新负列后使用不限制标签数量的精确定价：

```text
solve.py --> PP.py
             --------------------------------------------------
         --> | labels.py + cut_resource.py                    |
             | + dynamic_upper_bound.py + forward_labeling.py |
             --------------------------------------------------
         --> rmp_integration.py --> RMP.py --> solve.py
```

该路径同样仅在找到新负列时重建 RMP，否则根据精确定价结果进入后续阶段。

割分离：从已完成精确定价的 RMP 解中分离新割；新增割后重新进行列生成，无新割或达到割迭代上限时进入后续阶段。

```text
             ----------------------------------------
solve.py --> | lm_subset_row_cut.py + separation.py | --> cut_pool.py
             ----------------------------------------
         --> rmp_integration.py --> RMP.py --> solve.py --> PP.py
```

根节点列—割生成收敛且满足固定条件后，执行约化成本固定；实际删弧后重建 RMP 并重新进行列—割生成：

```text
--------------------
| solve.py + PP.py |
--------------------
           --------------------------------------------------------------------
       --> | labels.py + cut_resource.py                                      |
           | + forward_labeling.py + backward_labeling.py                     |
           | + label_joining.py + variable_fixing.py + dynamic_upper_bound.py |
           --------------------------------------------------------------------
       --> rmp_integration.py --> RMP.py --> solve.py --> PP.py
```

启用调度枚举且满足间隙条件时，尝试改善整数上界；补充新列后重新进行列—割生成：

```text
--------------------
| solve.py + PP.py |
--------------------
           ----------------------------------------
       --> | enumeration.py + labels.py           |
           | + cut_resource.py + schedulemodel.py |
           ----------------------------------------
       --> branching.py --> main.py

enumeration.py --> rmp_integration.py --> RMP.py --> solve.py
```

节点不能按界关闭时选择分支，子节点返回列—割生成；全部节点处理结束或达到停止条件时汇总结果：

```text
             --------------------------------------
solve.py --> | strong_branching.py + branching.py |
             --------------------------------------
             -------------------------------------------
         --> | RMP.py + PP.py + dynamic_upper_bound.py |
             -------------------------------------------
         --> solve.py --> PP.py

branching.py --> main.py
```
