"""
src/agents/publisher.py
Tool-calling agent connecting to the Blogger MCP Server.
"""

import sys
import uuid
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver  # <-- Add this import
from src.prompts import load_prompt
from src.state import AgentState

prompt_spec = load_prompt("publisher")

mcp_config = {
    "blogger_server": {
        "command": sys.executable,
        "args": ["src/mcp_servers/blogger_server.py"],
        "transport": "stdio"
    }
}

async def _publish_via_mcp(title: str, draft_html: str) -> str:
    # Per call, for the same reason as the researcher: an AsyncClient must not outlive
    # the event loop asyncio.run() created for it.
    llm = prompt_spec.llm()
    client = MultiServerMCPClient(mcp_config)
    async with client.session("blogger_server") as session:
        tools = await client.get_tools()

        system_prompt = prompt_spec.render()

        # Override checkpointer inheritance with an isolated MemorySaver
        agent = create_react_agent(
            llm, 
            tools, 
            prompt=system_prompt, 
            checkpointer=MemorySaver()
        )
        
        temp_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        prompt_text = f"Title: {title}\nContent: {draft_html}"
        
        result = await agent.ainvoke({"messages": [("user", prompt_text)]}, config=temp_config)
        
        return result["messages"][-1].content

# ... (Keep the rest of your publisher_node exactly the same) ...
def publisher_node(state: AgentState) -> dict:
    topic = state["topic"]
    final_draft = state["draft"]
    
    print(f"🌐 Publishing '{topic}' to Blogger...")
    try:
        live_url = asyncio.run(_publish_via_mcp(topic, final_draft))
        return {"blogger_url": live_url, "sender": "publisher"}
    except Exception as e:
        print(f"Error calling Publisher MCP Tool: {e}")
        return {"blogger_url": "Failed to publish.", "sender": "publisher"}