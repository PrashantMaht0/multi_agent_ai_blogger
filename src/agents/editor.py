"""
src/agents/editor.py
LLM-as-a-judge control loop using gemma4:12b.
"""

import json
from langchain_core.messages import SystemMessage
from src.agents.parsing import extract_feedback, extract_verdict
from src.prompts import load_prompt
from src.state import AgentState

# We strictly enforce JSON format output in Ollama
prompt_spec = load_prompt("editor")
editor_llm = prompt_spec.llm()

def editor_node(state: AgentState) -> dict:
    prompt = prompt_spec.render(
        topic=state["topic"],
        research_notes=state["research_notes"],
        draft=state["draft"],
    )

    try:
        response = editor_llm.invoke([SystemMessage(content=prompt)])
        decision = json.loads(response.content)
        
        # Read the verdict out of the payload: the judge returns valid JSON that does not
        # always use the requested keys, and a missed PASS costs a whole revision loop.
        return {
            "feedback": extract_feedback(decision),
            "revision_count": state.get("revision_count", 0) + 1,
            "last_evaluation": extract_verdict(decision, ("PASS", "FAIL"), default="FAIL"),
            "sender": "editor"
        }
    except json.JSONDecodeError:
        # Failsafe if the model breaks formatting
        return {
            "feedback": "Editor failed to parse format. Please refine the draft structure.",
            "revision_count": state.get("revision_count", 0) + 1,
            "last_evaluation": "FAIL",
            "sender": "editor"
        }