"""Agent that turns validated research into an HTML draft."""

from src.common.parsing import message_text, prompt_messages
from src.prompts import load_prompt
from src.state import AgentState

prompt_spec = load_prompt("writer")
writer_llm = prompt_spec.llm()

WORDS_PER_NOTE = 40
MIN_WORDS, MAX_WORDS = 600, 1300


def target_words(notes: list[str]) -> int:
    """40 words per claim note, clamped; "Sources disagree:" lines are not claims."""
    claims = sum(1 for line in notes if line.startswith("- "))
    return min(max(WORDS_PER_NOTE * claims, MIN_WORDS), MAX_WORDS)


def writer_node(state: AgentState) -> dict:
    topic = state["topic"]
    research = "\n".join(state["research_notes"])
    feedback = state.get("feedback", "None. This is the first draft.")

    prompt = prompt_spec.render(topic=topic, research=research, feedback=feedback,
                                target_words=target_words(state["research_notes"]))

    print("✍️ Writer is drafting the post...")
    response = writer_llm.invoke(prompt_messages(prompt, "Write the blog post now."))
    
    return {
        "draft": message_text(response),
        "sender": "writer"
    }