"""
src/agents/validator.py
Agent responsible for validating sources and factual consistency from researcher.py.
"""

import os
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from src.state import AgentState

# Use Gemma for strong reasoning and structured output
model_name = os.getenv("EDITOR_MODEL", "gemma4:12b")
validator_llm = ChatOllama(model=model_name, format="json", temperature=0)

def validator_node(state: AgentState) -> dict:
    topic = state.get("topic", "")
    research_notes = state.get("research_notes", [])
    
    prompt = f"""You are a Fact-Checking & Source Verification Agent.
    Topic: {topic}
    Research Data Provided:
    {json.dumps(research_notes, indent=2)}

    Task:
    1. Verify if the research data directly addresses the topic.
    2. Check that the information contains concrete, credible facts (not hallucinated, contradictory, or empty).
    3. Determine if the information is sufficient for writing an in-depth article.

    Output STRICTLY as a JSON object:
    {{
        "status": "VALIDATED" or "REJECTED",
        "feedback": "Reason for rejection or verification summary"
    }}
    """
    
    try:
        response = validator_llm.invoke([SystemMessage(content=prompt)])
        decision = json.loads(response.content)
        
        status = decision.get("status", "REJECTED")
        feedback = decision.get("feedback", "")
        
        print(f"🛡️ Validator Result: {status} - {feedback}")
        return {
            "validation_status": status,
            "validation_feedback": feedback,
            "sender": "validator"
        }
    except Exception as e:
        print(f"Validator Error: {e}")
        return {
            "validation_status": "REJECTED",
            "validation_feedback": f"Failed due to system error: {str(e)}",
            "revision_count": state.get("revision_count", 0) + 1,  # Increment the counter
            "sender": "validator"
        }