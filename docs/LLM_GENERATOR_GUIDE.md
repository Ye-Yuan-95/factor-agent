# LLM 候选生成与用户输出指南

LLM提出受约束的候选表达式和研究理由。本地主程序负责命名、家族、预算、表达式校验及统计判定；模型不能决定ACCEPT或修改门槛。

以下命令从仓库根目录执行；统一入口 `python main.py run` 使用相同的参数与验证流程。

## 安装与调用

GLM SDK是可选依赖：

```sh
python -m pip install -e ".[llm-glm]"
python -m factor_research.make_dummy_panel
```

在PowerShell中设置密钥：

```powershell
$env:GLM_API_KEY = "你的 API Key"
```

Bash使用 `export GLM_API_KEY="..."`。密钥只从进程环境读取，不进入配置、manifest或仓库。调用使用你的服务账号，额度与费用以服务端账号为准。

```sh
python -m agent.l1_runner --query "find a robust momentum factor" --generator glm --families momentum --synthetic gen1 --inject-ic 0.06 --generation-attempt-budget 6 --budget 2 --seed 7
```

GLM adapter配置模型 `glm-4.7-flash`。具体模型与请求参数以 `agent/llm_generator.py` 和运行快照为准。`python -m agent.l1_runner --help`列出可用参数。

## 请求与搜索空间

ResearchRequest保存原问题、研究目标、股票范围、允许/排除家族、偏好、现有因子及预算。GeneratorSpec冻结许可列、家族映射、函数签名和表达式复杂度。

```json
{
  "original_query": "寻找低冗余动量因子",
  "objective": "在规定搜索空间内评估独立增量信息",
  "universe": "synthetic_gen1",
  "allowed_families": ["momentum"],
  "excluded_families": [],
  "preferences": {"low_turnover": true},
  "existing_factors": [],
  "generation_attempt_budget": 6,
  "oracle_trial_budget": 2
}
```

模型收到许可函数及参数签名。静态校验包括JSON字段和类型、表达式列、函数参数、家族暴露、AST复杂度和重复候选；拒绝原因记录在trace中。

## Proposal 与反馈

基础proposal仅有两个字段：

```json
{
  "expression": "zscore(ts_mean(mom1m, 6))",
  "rationale": "平滑短期动量并评估其增量信息"
}
```

名称与家族由host决定。声明机制模板的实验还可依其GeneratorSpec使用 `recipe_name`；模板声明由host验证，不采信模型自报统计结果。

`generation_attempt_budget`限制API调用、本地失败与重试；`oracle_trial_budget`限制进入Oracle的有效试验。实际尝试、合法候选和统计trial分别记录，不能只用名义预算描述成本。

反馈只包含白名单允许的训练/验证期拒绝信息、已试表达式及家族/重复提示。test、lockbox、ground truth或其他禁用字段会被拒绝。真值评分属于评估端，不能作为候选搜索奖励。

## 用户看到的结果

报告包含研究请求摘要、候选清单及逐候选FactorCard：表达式、家族、来源、各关统计量、有效样本、预算计数、证书状态、拒绝原因和验证期组合增量。

| 状态 | 解释 |
|---|---|
| REJECT | 至少一项检查未达到，或证据不足 |
| SHELVED | 当前验证期组合增量不足或无法评价 |
| ACCEPT | 验证条件和适用证书满足，可进入组合候选评估 |
| DRAFT | 证书不匹配、缺失或不适用，仍可记录研究结果 |

增量门少于5个有效回归截面或统计量非有限时不通过。报告显示实际拟合期数和完整观测数；原始日期组数不能代替有效样本或独立样本量。

报告由保存的请求和卡片确定性生成，不调用LLM重新解释结果，也不编造缺失的统计量。每个实际Oracle trial对应一张候选卡，基础库的列不会自动成为新发现。

## 真实数据与证书

真实面板按 [DATA_CONTRACT.md](DATA_CONTRACT.md)准备，并显式传 `--panel-csv`。否则上述命令使用合成市场进行流程验证。

证书绑定面板、配置和列出的源文件。数据或源身份不匹配时不能把候选当作已认证ACCEPT。合成内存面板不具备真实面板认证资格。测试锁箱结果只用于完成后的评估，不回流生成器。

## 比较与指标

候选源比较使用 `agent.b2_compare` 或 `agent.factorial_ablation` 的冻结协议。记录配对面板seed、实际有效trial、提案成功率、token/延迟、接受计数及真值评分。协议中的示例、形式知识和预算差异是解释限制，不能从观察到的差异直接归因模型能力。

FDP的全部运行/非空运行均值、条件Power和全流程Power见 [结果字段](RESULT_SCHEMA.md)。流程测试不证明发现能力或真实市场收益。
