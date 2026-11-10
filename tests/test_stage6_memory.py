"""Stage 6 test: the agent remembers the chat in Neon (Q8). Uses a new chat ID on every run."""
import json
import uuid
from pathlib import Path

from backend.pipelines import answer

QUESTIONS = json.loads((Path(__file__).parent.parent / "test_data" / "questions.json").read_text())


def test_q8_follow_up_uses_memory():
    q = next(x for x in QUESTIONS["questions"] if x["id"] == 8)
    chat_id = str(uuid.uuid4())  # new chat every run, so old runs cannot leak in
    first = answer(q["conversation"][0], "agent", [], [], chat_id)
    second = answer(q["conversation"][1], "agent", [], [], chat_id)
    assert "185.0" in first["answer"]
    assert "187.3" in second["answer"]
    # Sources and tools of the second answer come only from the second question's tool calls.
    assert all(s["year"] == 2025 for s in second["sources"] if s["type"] == "sec")
