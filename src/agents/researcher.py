"""
src/agents/researcher.py
Tool-calling agent connecting to the Search MCP Server.
"""

import os
import sys
import uuid
import asyncio
import traceback
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver  # <-- Add this import
from src.state import AgentState

model_name = os.getenv("WORKER_MODEL", "llama3:8b")
llm = ChatOllama(model=model_name, temperature=0.1)

mcp_config = {
    "research_server": {
        "command": sys.executable,
        "args": ["src/mcp_servers/search_server.py"],
        "transport": "stdio"
    }
}

async def _run_research_agent(topic: str) -> str:
    client = MultiServerMCPClient(mcp_config)
    async with client.session("research_server") as session:
        tools = await client.get_tools()
        
        system_prompt = "You are a Research Agent. Use the search tool to find 3 key facts about the given topic. Return a bulleted list of facts."
        
        # Override checkpointer inheritance with an isolated MemorySaver
        agent = create_react_agent(
            llm, 
            tools, 
            prompt=system_prompt, 
            checkpointer=MemorySaver()
        )
        
        # When using a checkpointer, we must provide a temporary thread_id
        temp_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        
        result = await agent.ainvoke({"messages": [("user", topic)]}, config=temp_config)
        
        return result["messages"][-1].content

# ... (Keep the rest of your researcher_node exactly the same) ...
def researcher_node(state: AgentState) -> dict:
    topic = state['topic']
    notes = state.get("research_notes", [])
    
    if not notes or "Error:" in notes[-1]:
        print(f"🔍 Searching the web for: {topic}...")
        try:
            findings = asyncio.run(_run_research_agent(topic))
            return {"research_notes": [findings], "sender": "researcher"}
        except Exception as e:
            print("\n" + "="*50)
            print("🚨 MCP SERVER FAILURE TRACEBACK:")
            traceback.print_exc()
            print("="*50 + "\n")
            return {"research_notes": [f"Error: {str(e)}"], "sender": "researcher"}
    
    return {"sender": "researcher"}