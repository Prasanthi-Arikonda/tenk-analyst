"""Stage 2 tests: the filters pipeline. These call OpenRouter and Qdrant."""
import json
from pathlib import Path

from backend.pipelines import answer

QUESTIONS = json.loads((Path(__file__).parent.parent / "test_data" / "questions.json").read_text())


def get(qid):
    return next(x for x in QUESTIONS["questions"] if x["id"] == qid)


def test_q2_every_source_is_gm_2024():
    q = get(2)
    out = answer(q["question"], "filters", [], [], "test-chat")
    assert out["sources"]
    assert all(s["company"] == "GM" and s["year"] == 2024 for s in out["sources"])


def test_q9_sidebar_conflict_message_and_no_sources():
    q = get(9)
    out = answer(q["question"], "filters", q["sidebar"]["companies"], q["sidebar"]["years"], "test-chat")
    assert out["answer"].startswith("This question asks about something that is not selected in the sidebar:")
    assert "General Motors" in out["answer"]
    assert out["sources"] == []


def test_sidebar_replaces_only_its_own_list():
    # Sidebar sets companies to Ford; the question's year (2025) still limits the years.
    out = answer("What risks does Ford report about tariffs in 2025?", "filters", ["F"], [], "test-chat")
    assert out["sources"]
    assert all(s["company"] == "F" and s["year"] == 2025 for s in out["sources"])
