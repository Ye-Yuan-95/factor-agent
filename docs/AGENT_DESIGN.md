# 因子研究系统架构

本系统在受约束表达式空间中提出候选、计算因子并执行统计验证。真实数据适配范围为A股；生成器、验证器和组合评估分别承担不同职责。

`main.py` 是仓库入口，`factor_research.cli` 负责命令转发；安装后也可使用
`factor-agent` 或 `python -m factor_research`。各模块保留自身的参数检查、预算和判定规则。

## 数据与职责

```text
研究请求 / 静态候选
    → 冻结列、算子、家族及预算
    → 本地生成器或可选LLM提出表达式
    → DSL校验与计算
    → Oracle统计验证
    → FactorCard、试验台账与报告
```

生成器负责提出表达式和研究理由。主程序决定候选名称、家族、预算和验证顺序；Oracle决定候选状态。模型的文字解释不作为统计证据。

| 组件 | 文件 | 职责 |
|---|---|---|
| 数据构建 | `factor_research/build_a_share_factor_panel.py` | A股面板、公告可用性和收益标签结算 |
| 表达式语言 | `agent/factor_dsl.py` | AST白名单、列与参数检查、只向过去取值 |
| 生成契约 | `agent/generator_contract.py` | 请求、搜索空间、失败类型与反馈白名单 |
| 候选源 | `agent/baseline_generator.py`、`agent/hybrid_generator.py`、`agent/llm_generator.py` | 本地搜索或可选GLM接口 |
| 验证器 | `agent/validator_oracle.py` | 五关、DSR、控制集和FactorCard |
| 记忆 | `agent/memory.py` | 试验计数、表达式去重、收益序列及有效试验数 |
| 运行管理 | `agent/run_context.py`、`agent/l1_runner.py` | 独立运行目录、预算执行和测试锁箱 |
| 报告 | `agent/user_report.py` | 根据保存的证据生成确定性用户报告 |
| 合成评估 | `synthetic/`、`agent/realized_fdr.py` | 指定真值下的FPR、Power与经验FDP |
| 组合研究 | `factor_research/solve_unified_v2.py`、`factor_research/hybrid_walkforward.py` | 联合选择和结算约束下的开发样本外评估 |

## 验证与候选状态

Oracle检查边际显著性、冗余和家族限制、重采样稳健性、控制其他因子后的增量信息以及稀疏权重，并计算DSR。阈值由配置与Oracle参数决定。

增量检验联合控制已接受因子、表达式成分及相关基础列。至少5个可拟合日期截面且统计量有限才可能通过；不足时记录证据不足，不自动放行。输出同时记录实际拟合期数和完整观测数。日期组数不等于独立样本数，最低截面数也不是功效保证。

| 状态 | 含义 |
|---|---|
| `REJECT` | 至少一个验证条件未达到，或证据不可用 |
| `SHELVED` | 统计检查通过，但当前验证期组合增量不足或无法评价 |
| `ACCEPT` | 验证和投资增量条件达到，且具备适用的过程证书 |
| `DRAFT` | 候选有可记录的结果，但证书缺失、不匹配或不适用 |

FactorCard包含表达式、家族、各关统计量、有效样本、试验计数、证书状态、验证期投资字段，以及输入和代码标识。通过验证不等于因子经济机制或未来交易收益得到证明。

## 预算与反馈

`generation_attempt_budget`约束提案尝试，包括API请求、本地拒绝和重试；`oracle_trial_budget`约束进入验证器的试验。两者分别计数。试验收益序列进入台账；相关聚类提供有效试验数的估计。

反馈采用白名单，允许训练/验证期的拒绝原因、家族与重复信息。测试期结果、真值序列和锁箱内容不进入生成器反馈。DSR不能替代独立测试或消除测试反馈造成的选择偏差。

## 数据和测试锁箱

训练只使用在截止日已经结算的标签。运行冻结输入与配置，测试数据单独封存。锁箱仅在完成运行后通过明确的人为检查点一次性开启；测试结果用于最终报告，不回流到搜索或台账。

数据字段和可用性规则见 [DATA_CONTRACT.md](DATA_CONTRACT.md)。组合和优化入口的适用范围见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

## 证书与实证解释

过程证书评估指定Gen-1/Gen-2下的基础因子选择栈，绑定输入、配置和列出的源文件。候选DSL搜索闭环另用经验FDP评估；基础选择器的null-FPR/Power不能替代闭环FDR或投资价值证据。

合成真值定义、实验入口和适用边界见 [RESEARCH_EXPERIMENTS.md](RESEARCH_EXPERIMENTS.md)。本仓库提供代码、配置、协议与测试，原始研究面板和数值证据包由研究者自行提供。
