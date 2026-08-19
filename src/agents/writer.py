"""Agent that turns validated research into an HTML draft."""

from src.agents.parsing import message_text, prompt_messages
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
    response = writer_llm.invoke(prompt_messages(prompt, "Write the blog post now."))
    
    return {
        "draft": message_text(response),
        "sender": "writer"
    }