"""Search step: the one place research branches reach the search MCP server."""

import sys
import json
import asyncio
from typing import Optional
from typing_extensions import TypedDict
from langchain_mcp_adapters.client import MultiServerMCPClient

mcp_config = {
    "research_server": {
        "command": sys.executable,
        "args": ["src/mcp_servers/search_server.py"],
        "transport": "stdio"
    }
}


class Source(TypedDict):
    url: str
    title: str
    published_date: Optional[str]
    raw_content: str


async def _call_search(query: str) -> str:
    client = MultiServerMCPClient(mcp_config)
    async with client.session("research_server") as session:
        result = await session.call_tool("search_sources", {"query": query})
    return result.content[0].text


def search_sources(query: str) -> list[Source]:
    """One search. Each call has its own session and event loop, so threads never share one."""
    reply = json.loads(asyncio.run(_call_search(query)))
    if isinstance(reply, dict):
        raise RuntimeError(reply.get("error", "search failed"))
    return reply
