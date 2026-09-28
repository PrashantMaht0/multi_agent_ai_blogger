"""Planner: splits the topic into 5 to 8 sub questions and pins the timeframe."""

import datetime
import json
import re

from src.common.errors import root_cause
from src.common.parsing import message_text, prompt_messages
from src.tools.sanitize import clean_title
from src.prompts import load_prompt
from src.state import AgentState, SubQuestion, Timeframe, make_timeframe

prompt_spec = load_prompt("planner")
planner_llm = prompt_spec.llm()

MIN_QUESTIONS = 5
MAX_QUESTIONS = 8
ANY_TIME = "any time"
TEMPLATE_QUESTIONS = (
    "What is {}?",
    "How does {} work?",
    "What is the current state of {}?",
    "What are the main limitations or tradeoffs of {}?",
    "What are real world examples of {}?",
)


def normalize_question(text: str) -> str:
    """Lowercase, punctuation removed, whitespace collapsed: how repeats are detected."""
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


def _clean_questions(raw) -> list[str]:
    """Distinct, non empty questions in model order, at most 8."""
    questions, seen = [], set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, str) or not item.strip():
            continue
        key = normalize_question(item)
        if key and key not in seen:
            seen.add(key)
            questions.append(item.strip())
    return questions[:MAX_QUESTIONS]


def _sub_questions(questions: list[str]) -> list[SubQuestion]:
    return [{"id": f"q{i}", "question": q, "origin": "planner"} for i, q in enumerate(questions, 1)]


def _timeframe(raw) -> Timeframe:
    timeframe = make_timeframe(raw) or {"start_year": None, "end_year": None, "label": ""}
    return {**timeframe, "label": timeframe["label"] or ANY_TIME}


def make_plan(topic: str) -> tuple[Timeframe, list[SubQuestion]]:
    """Asks the model for a plan, retrying once, then falls back to template questions."""
    prompt = prompt_spec.render(topic=topic, current_year=datetime.date.today().year)
    reason = ""
    for _ in range(2):
        try:
            data = json.loads(message_text(planner_llm.invoke(prompt_messages(prompt, "Write the plan now."))))
        except Exception as e:
            reason = root_cause(e)
            continue
        questions = _clean_questions(data.get("sub_questions")) if isinstance(data, dict) else []
        if len(questions) >= MIN_QUESTIONS:
            return _timeframe(data.get("timeframe")), _sub_questions(questions)
        reason = f"{len(questions)} usable sub questions"

    print(f"[PLAN] fallback: {reason}")
    title = clean_title(topic)
    fallback = {"start_year": None, "end_year": None, "label": ANY_TIME}
    return fallback, _sub_questions([t.format(title) for t in TEMPLATE_QUESTIONS])


def planner_node(state: AgentState) -> dict:
    timeframe, sub_questions = make_plan(state["topic"])
    print(f"[PLAN] {len(sub_questions)} sub questions, timeframe {timeframe['label']}")
    return {"timeframe": timeframe, "sub_questions": sub_questions, "sender": "planner"}
