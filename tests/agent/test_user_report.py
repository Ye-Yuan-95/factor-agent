"""Ensure deterministic reports reflect saved cards and certification boundaries."""
import json

from agent.generator_contract import GeneratorSpec, ResearchRequest
from agent.user_report import portfolio_status, write_user_report


def _spec():
    return GeneratorSpec(
        allowed_columns=("ep", "roa"),
        column_families={"ep": "value", "roa": "profitability"},
    )


def _request():
    return ResearchRequest.from_dict({
        "original_query": "寻找低冗余、低换手的价值因子",
        "objective": "寻找可作为组合候选的价值信号",
        "universe": "synthetic_gen1_seed7_ic0.06",
        "allowed_families": ["value"],
        "excluded_families": [],
        "preferences": {"low_redundancy": True, "low_turnover": True},
        "existing_factors": ["size"],
        "generation_attempt_budget": 4,
        "oracle_trial_budget": 2,
    })


def _card(name, decision, certified):
    """Construct a minimal recorded factor card for deterministic rendering tests."""
    return {
        "name": name,
        "expression": "rank(ep)",
        "rationale": "低估值可能对应风险补偿",
        "family": "value",
        "decision": decision,
        "certified": certified,
        "confidence": "high" if certified else "low",
        "fdr_pass": decision == "ACCEPT",
        "fdr_q": 0.01,
        "redundancy_pass": True,
        "survival_pass": True,
        "incremental_pass": True,
        "selected_by_weight": True,
        "dsr_pass": True,
        "val_sharpe_delta": 0.1,
        "data_version": "data-hash",
        "seed": 7,
        "test_sharpe_delta": 99.0,
    }


def test_three_layer_report_and_portfolio_language(tmp_path):
    accept = _card("llm_0001", "ACCEPT", True)
    reject = _card("llm_0002", "REJECT", True)
    reject["reject_reason"] = "failed_synthetic_survival"
    reject.update({"incremental_pass": False, "incremental_t": None,
                   "incremental_n_periods": 4, "incremental_n_complete_rows": 320,
                   "incremental_sample_status": "insufficient_cross_sections"})
    paths = write_user_report(
        run_dir=tmp_path, run_id="run-1", request=_request(),
        cards=[reject, accept], spec=_spec(),
        generator_metadata={
            "llm_0001": {"provider": "zhipu", "model": "glm-4.7-flash",
                         "prompt_version": "glm-b1-v1", "prompt_hash": "abc"},
        }, data_version="data-hash")
    report = paths["user_report"].read_text(encoding="utf-8")
    assert "## 4.1 第一层：研究请求摘要" in report
    assert "## 4.2 第二层：因子清单和筛选状态" in report
    assert "## 4.3 第三层：每个因子的名片" in report
    assert "可进入 portfolio 候选池" in report
    assert "glm-4.7-flash" in report
    assert "99.0" not in report
    assert "test_sharpe_delta" not in report
    assert "4 个日期截面" in report and "320 条完整观测" in report
    assert "insufficient_cross_sections" in report

    rows = json.loads(paths["factor_list"].read_text(encoding="utf-8"))
    assert rows[0]["name"] == "llm_0001"
    assert rows[0]["portfolio_status"] == "可进入 portfolio 候选池"
    assert portfolio_status(_card("draft", "DRAFT", False)).startswith("暂不可进入")
