"""
src/agents/validator.py
Agent responsible for validating sources and factual consistency from researcher.py.
"""

import json
from langchain_core.messages import SystemMessage
from src.agents.parsing import extract_feedback, extract_verdict
from src.prompts import load_prompt
from src.state import AgentState

# Use Gemma for strong reasoning and structured output
prompt_spec = load_prompt("validator")
validator_llm = prompt_spec.llm()

def validator_node(state: AgentState) -> dict:
    topic = state.get("topic", "")
    research_notes = state.get("research_notes", [])
    research_error = state.get("research_error")

    error_context = f"\n    Research tool failure reported: {research_error}\n" if research_error else ""

    prompt = prompt_spec.render(
        topic=topic,
        research_notes=json.dumps(research_notes, indent=2),
        error_context=error_context,
    )

    try:
        response = validator_llm.invoke([SystemMessage(content=prompt)])
        decision = json.loads(response.content)
        
        # gemma4:12b returns valid JSON that does not always match the requested shape,
        # so read the verdict out of the payload rather than trusting decision["status"].
        status = extract_verdict(decision, ("VALIDATED", "REJECTED"), default="REJECTED")
        feedback = extract_feedback(decision)

        print(f"Validator Result: {status} (raw: {decision}) - {feedback}")
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
            "sender": "validator"
        }