"""
src/agents/editor.py
LLM-as-a-judge control loop using gemma4:12b.
"""

import os
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from src.state import AgentState

# We strictly enforce JSON format output in Ollama
model_name = os.getenv("EDITOR_MODEL", "gemma4:12b")
editor_llm = ChatOllama(model=model_name, format="json", temperature=0)

def editor_node(state: AgentState) -> dict:
    prompt = f"""You are the Managing Editor. Your job is to evaluate the Writer's draft. 
    It must incorporate the provided research notes accurately and be formatted in HTML.
    
    Topic: {state['topic']}
    Research Notes: {state['research_notes']}
    Current Draft: {state['draft']}
    
    Evaluate the draft. Output your response STRICTLY as a JSON object with two keys:
    1. "status": Must be exactly "PASS" or "FAIL".
    2. "feedback": If FAIL, explain exactly what the Writer needs to fix. If PASS, leave empty.
    
    Do not output any markdown code blocks (e.g. ```json). Just the raw JSON object.
    """
    
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