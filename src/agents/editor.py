"""
src/agents/editor.py
LLM-as-a-judge control loop using gemma4:12b.
"""

from src.agents.parsing import judge_messages, judge_text, parse_verdict_lines
from src.prompts import load_prompt
from src.state import AgentState

# We strictly enforce JSON format output in Ollama
prompt_spec = load_prompt("editor")
editor_llm = prompt_spec.llm()

def editor_node(state: AgentState) -> dict:
    # No research_notes: the validator owns fact-checking, this node only judges writing.
    prompt = prompt_spec.render(topic=state["topic"], draft=state["draft"])

    raw = judge_text(editor_llm, judge_messages(prompt))

    if not raw.strip():
        # A judge that answers nothing has not judged the draft. Failing here forced a
        # pointless redraft plus another review; a human still reviews before publishing.
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