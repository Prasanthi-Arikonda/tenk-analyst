"""Stage 3 tests: the rerank pipeline (search 20, FlashRank keeps 5)."""
import json
from pathlib import Path

from backend.pipelines import answer

QUESTIONS = json.loads((Path(__file__).parent.parent / "test_data" / "questions.json").read_text())


def test_q2_with_rerank_gives_five_gm_2024_sources():
    q = next(x for x in QUESTIONS["questions"] if x["id"] == 2)
    out = answer(q["question"], "rerank", [], [], "test-chat")
    assert len(out["sources"]) == 5
    assert all(s["company"] == "GM" and s["year"] == 2024 for s in out["sources"])
    assert out["answer"].strip()


def test_rerank_keeps_sidebar_conflict_check():
    out = answer("What was GM's R&D expense in 2025?", "rerank", ["TSLA", "F"], [], "test-chat")
    assert "General Motors" in out["answer"]
    assert out["sources"] == []
