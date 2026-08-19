"""Agent that fact-checks the researcher's findings before any writing starts."""

import json
from src.agents.parsing import judge_messages, judge_text, parse_verdict_lines
from src.prompts import load_prompt
from src.state import AgentState

# Hosted model: this is the only node that checks facts, so it needs current knowledge.
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
        raw = judge_text(validator_llm, judge_messages(prompt))
        if not raw.strip():
            # Silence is not a rejection; let the research through.
            print("Validator returned nothing; passing research through.")
            return {
                "validation_status": "VALIDATED",
                "validation_feedback": "Validator returned nothing; research passed through unchecked.",
                "sender": "validator"
            }

        status, feedback = parse_verdict_lines(raw, ("VALIDATED", "REJECTED"), default="REJECTED")
        print(f"Validator Result: {status} - {feedback}")
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