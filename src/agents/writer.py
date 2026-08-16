"""
src/agents/writer.py
Content generation agent.
"""

from langchain_core.messages import SystemMessage
from src.prompts import load_prompt
from src.state import AgentState

prompt_spec = load_prompt("writer")
writer_llm = prompt_spec.llm()

def writer_node(state: AgentState) -> dict:
    topic = state["topic"]
    research = "\n".join(state["research_notes"])
    feedback = state.get("feedback", "None. This is the first draft.")

    prompt = prompt_spec.render(topic=topic, research=research, feedback=feedback)

    print("✍️ Writer is drafting the post...")
    response = writer_llm.invoke([SystemMessage(content=prompt)])
    
    return {
        "draft": response.content,
        "sender": "writer"
    }