"""Stage 5 tests: the agent and the SEC data. They call OpenRouter, Qdrant and SEC."""
import json
import uuid
from pathlib import Path

from backend.agent import unique_sources
from backend.pipelines import MSG_COMPANY, answer
from backend.sec_data import annual_value

QUESTIONS = json.loads((Path(__file__).parent.parent / "test_data" / "questions.json").read_text())
NOT_REPORTED = "This number is not reported in the SEC financial data or the 10-K."


def ask(qid):
    # A new chat ID every time: the agent remembers a chat, so a reused ID would carry old answers over.
    q = next(x for x in QUESTIONS["questions"] if x["id"] == qid)
    side = q.get("sidebar", {})
    return answer(q["question"], "agent", side.get("companies", []), side.get("years", []), str(uuid.uuid4()))


def test_sec_data_ford_revenue_2024():
    value, _ = annual_value("F", "revenue", 2024)
    assert round(value / 1e9, 1) == 185.0


def test_sec_data_gm_rd_2025_is_not_returned():
    assert annual_value("GM", "rd_expense", 2025) == (None, None)


def test_q4_ford_rd_from_sec_data():
    out = ask(4)
    assert "9.4" in out["answer"]
    assert "sec_financial_data" in out["tools_used"]


def test_q5_gm_rd_from_10k_text():
    out = ask(5)
    assert "8.5" in out["answer"]
    assert any(s["type"] == "10-K" and s["company"] == "GM" and s["year"] == 2025 for s in out["sources"])


def test_q6_carvana_rd_not_reported():
    out = ask(6)
    assert NOT_REPORTED in out["answer"]
    assert "sec_financial_data" in out["tools_used"]


def test_q7_toyota_company_message():
    out = ask(7)
    assert out["answer"] == MSG_COMPANY
    assert out["sources"] == []


def test_q10_sidebar_message_no_tools():
    out = ask(10)
    assert out["answer"].startswith("This question asks about something that is not selected in the sidebar:")
    assert "General Motors" in out["answer"]
    assert out["sources"] == [] and out["tools_used"] == []


def test_repeated_sources_are_kept_once_in_order():
    a = {"type": "10-K", "company": "F", "year": 2025, "item": "7", "excerpt": "text A"}
    b = {"type": "10-K", "company": "F", "year": 2025, "item": "7", "excerpt": "text B"}
    sec = {"type": "sec", "company": "F", "year": 2025, "metric": "rd_expense", "value": 1}
    assert unique_sources([a, sec, a, b, sec, a]) == [a, sec, b]
