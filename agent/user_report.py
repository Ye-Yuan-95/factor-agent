"""Deterministic three-layer user report for a completed factor-research run: a pure projection
of frozen request/card metadata -- no LLM summaries, invented scores, or eligibility decisions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.generator_contract import GeneratorSpec, ResearchRequest, assign_expression_family


def portfolio_status(card: dict[str, Any]) -> str:
    decision = str(card.get("decision", "REJECT"))
    if decision == "ACCEPT" and card.get("certified") is True:
        return "可进入 portfolio 候选池"
    if decision == "SHELVED":
        return "观察名单；等待联合复选"
    if decision == "DRAFT":
        return "暂不可进入；等待有效证书或重新验证"
    return "不可进入"


def write_user_report(*, run_dir: str | Path, run_id: str, request: ResearchRequest,
                      cards: list[dict[str, Any]], spec: GeneratorSpec,
                      generator_metadata: dict[str, dict[str, Any]] | None = None,
                      data_version: str | None = None,
                      validation_scope: str = "仅训练期和验证期；测试集未打开") -> dict[str, Path]:
    """Write the summary JSON, list JSON and Markdown report before the run is sealed."""
    root = Path(run_dir)
    generator_metadata = generator_metadata or {}
    request_path = root / "research_request.json"
    factor_list_path = root / "factor_list.json"
    report_path = root / "user_report.md"
    request_path.write_text(
        json.dumps(request.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    for card in cards:
        assignment = assign_expression_family(str(card["expression"]), spec)
        rows.append({
            "name": card.get("name"),
            "expression": card.get("expression"),
            "family": card.get("family"),
            "family_exposures": list(assignment.exposures),
            "decision": card.get("decision"),
            "certified": card.get("certified"),
            "confidence": card.get("confidence"),
            "portfolio_status": portfolio_status(card),
            "primary_reason": _primary_reason(card),
            "card_file": f"cards/{card.get('name')}.json",
        })
    rows.sort(key=lambda row: (_status_order(row["portfolio_status"]), str(row["name"])))
    factor_list_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    report = _render_report(
        run_id=run_id, request=request, cards=cards, rows=rows, spec=spec,
        generator_metadata=generator_metadata, data_version=data_version,
        validation_scope=validation_scope,
    )
    report_path.write_text(report, encoding="utf-8")
    return {
        "research_request": request_path,
        "factor_list": factor_list_path,
        "user_report": report_path,
    }


def _render_report(*, run_id: str, request: ResearchRequest, cards: list[dict[str, Any]],
                   rows: list[dict[str, Any]], spec: GeneratorSpec,
                   generator_metadata: dict[str, dict[str, Any]], data_version: str | None,
                   validation_scope: str) -> str:
    """Render request, factor rows and recorded cards into a deterministic report.

    Generator metadata supplies attribution only; decisions and numerical evidence
    come from the saved cards. The renderer makes no provider call and does not
    estimate missing measurements or open evaluation data.
    """
    allowed = "、".join(request.allowed_families) or "未额外限制"
    excluded = "、".join(request.excluded_families) or "无"
    existing = "、".join(request.existing_factors) or "无"
    preferences = "、".join(_preference_labels(request.preferences)) or "无额外偏好"
    lines = [
        "# 因子研究结果",
        "",
        "## 4.1 第一层：研究请求摘要",
        "",
        f"- **你的问题**：{_md(request.original_query)}",
        f"- **系统理解**：{_md(request.objective)}",
        f"- **研究范围**：{_md(request.universe)}",
        f"- **允许的因子家族**：{_md(allowed)}",
        f"- **排除的因子家族**：{_md(excluded)}",
        f"- **主要偏好**：{_md(preferences)}",
        f"- **比较用现有因子**：{_md(existing)}",
        (f"- **研究预算**：最多 {request.generation_attempt_budget} 次生成尝试；"
         f"最多 {request.oracle_trial_budget} 个候选进入正式验证"),
        f"- **验证范围**：{_md(validation_scope)}",
        f"- **数据版本**：{_md(data_version or '未记录')}",
        f"- **运行编号**：`{_md(run_id)}`",
    ]
    if request.assumptions:
        lines.extend(["- **系统采用的解释性假设**："] +
                     [f"  - {_md(item)}" for item in request.assumptions])
    if request.unsupported_requirements:
        lines.extend(["- **本版本尚不能执行的要求**："] +
                     [f"  - {_md(item)}" for item in request.unsupported_requirements])

    lines.extend([
        "",
        "## 4.2 第二层：因子清单和筛选状态",
        "",
        "| 因子 | 表达式 | 家族暴露 | 验证结论 | 组合使用状态 | 主要原因 |",
        "|---|---|---|---|---|---|",
    ])
    if rows:
        for row in rows:
            lines.append(
                f"| `{_cell(row['name'])}` | `{_cell(row['expression'])}` | "
                f"{_cell(' + '.join(row['family_exposures']))} | {_cell(row['decision'])} | "
                f"{_cell(row['portfolio_status'])} | {_cell(row['primary_reason'])} |"
            )
    else:
        lines.append("| — | — | — | — | 不可进入 | 本次运行没有产生合法候选 |")
    lines.extend([
        "",
        "> “可进入 portfolio 候选池”不等于已经进入最终组合，也不构成收益承诺。",
        "",
        "## 4.3 第三层：每个因子的名片",
        "",
    ])
    cards_by_name = {str(card.get("name")): card for card in cards}
    for row in rows:
        card = cards_by_name[str(row["name"])]
        meta = generator_metadata.get(str(row["name"]), {})
        lines.extend(_render_card(card, row, meta, run_id, spec))
    if not rows:
        lines.append("本次没有可展示的因子名片。请查看 `trace.jsonl` 中的生成失败记录。")
    return "\n".join(lines).rstrip() + "\n"


def _render_card(card: dict[str, Any], row: dict[str, Any], meta: dict[str, Any],
                 run_id: str, spec: GeneratorSpec) -> list[str]:
    """Format one card's definition, gate evidence, usable samples and limitations.

    Display undefined statistics as missing and preserve the recorded decision,
    certificate status and refusal reasons. Explanatory text describes the fields;
    it cannot promote a candidate or substitute for absent evidence.
    """
    gate_rows = [
        ("基础有效性/FDR", bool(card.get("fdr_pass")), card.get("fdr_q"),
         "在本次比较集合中是否仍有统计证据"),
        ("冗余检查", bool(card.get("redundancy_pass")),
         card.get("max_abs_corr_with_existing"), "是否提供不同于现有因子的信息"),
        ("稳健性", bool(card.get("survival_pass")), card.get("survival_rate"),
         "重采样或扰动下是否保持表现"),
        ("增量价值", bool(card.get("incremental_pass")), card.get("incremental_t"),
         "控制邻近基础因子后是否仍有边际信息"),
        ("稀疏权重", bool(card.get("selected_by_weight")), card.get("sparse_weight"),
         "联合模型是否给予非零权重"),
        ("DSR", bool(card.get("dsr_pass")), card.get("deflated_sharpe"),
         "考虑反复搜索后是否仍达到要求"),
    ]
    lines = [
        f"### { _md(card.get('name', 'unnamed')) }",
        "",
        "#### A. 因子定义",
        "",
        f"- **表达式**：`{_md(card.get('expression', ''))}`",
        f"- **主要家族**：{_md(card.get('family', 'unclassified'))}",
        f"- **完整家族暴露**：{_md(' + '.join(row['family_exposures']))}",
        f"- **研究假设**：{_md(card.get('rationale') or '未提供')}",
        f"- **提案来源**：{_md(meta.get('provider', 'unknown'))} / "
        f"`{_md(meta.get('model', 'unknown'))}`",
        "",
        "#### B. 最终结论",
        "",
        f"- **验证结论**：{_md(card.get('decision', 'REJECT'))}",
        f"- **组合使用状态**：{_md(row['portfolio_status'])}",
        f"- **证书状态**：{'已认证' if card.get('certified') else '未认证'}",
        f"- **证据置信度**：{_md(card.get('confidence', 'low'))}",
        "",
        "#### C. 核心证据",
        "",
        "| 检查 | 结果 | 统计量 | 用户解释 |",
        "|---|---|---:|---|",
    ]
    for label, passed, value, explanation in gate_rows:
        lines.append(
            f"| {label} | {'通过' if passed else '未通过'} | {_metric(value)} | {explanation} |")
    if card.get("incremental_n_periods") is not None:
        lines.append(
            f"\n增量检验实际使用 {_md(card['incremental_n_periods'])} 个日期截面、"
            f"{_md(card.get('incremental_n_complete_rows'))} 条完整观测；"
            f"样本状态：{_md(card.get('incremental_sample_status'))}。"
            "日期截面数不等于独立样本数；少于5个有效截面时不通过增量门。")
    lines.extend([
        "",
        "#### D. 风险与限制",
        "",
    ])
    lines.extend([f"- {_md(item)}" for item in _risk_notes(card)])
    lines.extend([
        "",
        "#### E. 可复现信息",
        "",
        f"- **run_id**：`{_md(run_id)}`",
        f"- **data_version**：`{_md(card.get('data_version') or 'unknown')}`",
        f"- **GeneratorSpec hash**：`{_md(meta.get('generator_spec_hash') or spec.fingerprint())}`",
        f"- **prompt version/hash**：`{_md(meta.get('prompt_version') or 'unknown')}` / "
        f"`{_md(meta.get('prompt_hash') or 'unknown')}`",
        f"- **model**：`{_md(meta.get('model') or 'unknown')}`",
        f"- **seed**：`{_md(card.get('seed') if card.get('seed') is not None else 'unknown')}`",
        f"- **code commit**：`{_md(card.get('code_commit') or 'unknown')}`",
        f"- **generated_at**：{_md(card.get('timestamp') or 'unknown')}",
        "",
        "---",
        "",
    ])
    return lines


def _risk_notes(card: dict[str, Any]) -> list[str]:
    """Derive scope and certification caveats from artifacts rather than model-written narratives."""
    notes = ["当前研究报告不包含测试集反馈，历史统计关系不构成未来收益保证。"]
    if not card.get("certified"):
        notes.append("当前结果未绑定有效流程证书，因此不能进入 portfolio 候选池。")
    if card.get("reject_reason"):
        notes.append(f"主要拒绝或保留原因：{card['reject_reason']}")
    val_delta = card.get("val_sharpe_delta")
    if val_delta is None:
        notes.append("验证期组合增量不可用或不足以支持进入候选池。")
    elif val_delta <= 0:
        notes.append(f"验证期组合 Sharpe 增量为 {val_delta:.3f}，未显示正向改善。")
    turnover = card.get("turnover_delta")
    if turnover is not None and turnover > 0:
        notes.append(f"加入该因子后验证期换手变化为 +{turnover:.3f}，需关注交易成本。")
    return notes


def _primary_reason(card: dict[str, Any]) -> str:
    if card.get("reject_reason"):
        return str(card["reject_reason"])
    if card.get("decision") == "ACCEPT" and card.get("certified") is True:
        return "五关和 DSR 通过，且验证期投资价值满足要求"
    if card.get("decision") == "SHELVED":
        return "存在统计证据，但当前验证期增量不足"
    if card.get("decision") == "DRAFT":
        return "若有有效证书本可进入更高状态；当前仅作草稿"
    return "未通过验证流程"


def _preference_labels(preferences: dict[str, Any]) -> list[str]:
    labels = {
        "low_redundancy": "低冗余",
        "low_turnover": "低换手",
        "validation_stability": "验证期稳健",
    }
    return [label for key, label in labels.items() if preferences.get(key) is True]


def _status_order(status: str) -> int:
    for i, prefix in enumerate(("可进入 portfolio 候选池", "观察名单", "暂不可进入")):
        if status.startswith(prefix):
            return i
    return 3


def _metric(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return _cell(value)


def _md(value: Any) -> str:
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def _cell(value: Any) -> str:
    return _md(value).replace("|", "\\|")
