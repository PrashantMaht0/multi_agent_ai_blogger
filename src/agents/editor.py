"""
src/agents/editor.py
LLM-as-a-judge control loop using gemma4:12b.
"""

import json
from langchain_core.messages import SystemMessage
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
        
        return {
            "feedback": decision.get("feedback", ""),
            "revision_count": state.get("revision_count", 0) + 1,
            "last_evaluation": decision.get("status", "FAIL"),
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