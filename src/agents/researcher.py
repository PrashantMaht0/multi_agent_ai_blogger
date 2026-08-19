"""Tool-calling agent connecting to the Search MCP Server."""

import sys
import uuid
import asyncio
import traceback
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from src.agents.errors import root_cause
from src.prompts import load_prompt
from src.state import AgentState

prompt_spec = load_prompt("researcher")

mcp_config = {
    "research_server": {
        "command": sys.executable,
        "args": ["src/mcp_servers/search_server.py"],
        "transport": "stdio"
    }
}

async def _run_research_agent(topic: str) -> str:
    # Built per call: an AsyncClient must not outlive the loop asyncio.run() created.
    llm = prompt_spec.llm()
    client = MultiServerMCPClient(mcp_config)
    async with client.session("research_server") as session:
        tools = await client.get_tools()

        system_prompt = prompt_spec.render()

        # Isolate this agent from the outer graph's checkpointer.
        agent = create_react_agent(
            llm, 
            tools, 
            prompt=system_prompt, 
            checkpointer=MemorySaver()
        )
        
        temp_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        
        result = await agent.ainvoke({"messages": [("user", topic)]}, config=temp_config)
        
        return result["messages"][-1].content

def researcher_node(state: AgentState) -> dict:
    topic = state['topic']
    notes = state.get("research_notes", [])
    attempts = state.get("research_attempts", 0)

    # Re-search on any unaccepted verdict, so every pass spends an attempt.
    if not notes or state.get("validation_status") != "VALIDATED":
        print(f"🔍 Searching the web for: {topic}...")
        try:
            findings = asyncio.run(_run_research_agent(topic))
            return {
                "research_notes": [findings],
                "research_error": None,
                "research_attempts": attempts + 1,
                "sender": "researcher"
            }
        except Exception as e:
            print("\n" + "="*50)
            print("🚨 MCP SERVER FAILURE TRACEBACK:")
            traceback.print_exc()
            print(f"ROOT CAUSE: {root_cause(e)}")
            print("="*50 + "\n")
            # Errors stay out of research_notes so they are never read as research.
            return {
                "research_notes": [],
                "research_error": root_cause(e),
                "research_attempts": attempts + 1,
                "sender": "researcher"
            }

    return {"sender": "researcher"}