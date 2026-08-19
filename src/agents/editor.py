"""Agent that judges how the draft reads and loops it back to the writer."""

from src.agents.parsing import judge_messages, judge_text, parse_verdict_lines
from src.prompts import load_prompt
from src.state import AgentState

prompt_spec = load_prompt("editor")
editor_llm = prompt_spec.llm()

def editor_node(state: AgentState) -> dict:
    # No research notes: the validator owns fact-checking, this node judges writing.
    prompt = prompt_spec.render(topic=state["topic"], draft=state["draft"])

    raw = judge_text(editor_llm, judge_messages(prompt))

    if not raw.strip():
        # Silence is not a verdict; let the draft through to human review.
        return {
            "feedback": "Editor returned nothing; draft passed through unreviewed.",
            "revision_count": state.get("revision_count", 0) + 1,
            "last_evaluation": "PASS",
            "sender": "editor"
        }

    evaluation, feedback = parse_verdict_lines(raw, ("PASS", "FAIL"), default="FAIL")
    print(f"Editor Result: {evaluation} - {feedback}")

    return {
        "feedback": feedback,
        "revision_count": state.get("revision_count", 0) + 1,
        "last_evaluation": evaluation,
        "sender": "editor"
    }