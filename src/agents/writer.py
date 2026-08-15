"""
src/agents/writer.py
Content generation agent.
"""

import os
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from src.state import AgentState

model_name = os.getenv("WORKER_MODEL", "llama3:8b")
writer_llm = ChatOllama(model=model_name, temperature=0.7)

def writer_node(state: AgentState) -> dict:
    topic = state["topic"]
    research = "\n".join(state["research_notes"])
    feedback = state.get("feedback", "None. This is the first draft.")
    
    prompt = f"""You are an expert tech blog writer. Write a highly engaging blog post about '{topic}'.
    
    Rules:
    1. You MUST include these facts: \n{research}
    2. Format the entire post in clean HTML (e.g., <h2>, <p>, <strong>).
    3. Do NOT output a markdown wrapper (e.g., ```html), just the raw HTML elements.
    
    Previous Editor Feedback to incorporate: {feedback}
    """
    
    print("✍️ Writer is drafting the post...")
    response = writer_llm.invoke([SystemMessage(content=prompt)])
    
    return {
        "draft": response.content,
        "sender": "writer"
    }