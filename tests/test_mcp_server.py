"""
tests/test_mcp_servers.py
Unit tests using FastMCP's in-memory transport for deterministic testing.
"""

import pytest
from fastmcp import Client
from src.mcp_servers.search_server import mcp as search_mcp
from src.mcp_servers.blogger_server import mcp as blogger_mcp

@pytest.mark.asyncio
async def test_search_server_in_memory():
    # Passing the FastMCP server instance directly connects via in-memory transport
    async with Client(search_mcp) as client:
        tools = await client.list_tools()
        names = [t.name for t in tools]
        assert "search_web" in names
        
        # Execute query
        result = await client.call_tool("search_web", {"query": "pytest test query"})
        output = result.data if hasattr(result, "data") else str(result)
        assert len(output) > 0
        assert "Error" not in output or "TAVILY_API_KEY is not set" in output

@pytest.mark.asyncio
async def test_blogger_server_in_memory():
    async with Client(blogger_mcp) as client:
        tools = await client.list_tools()
        names = [t.name for t in tools]
        assert "publish_to_blogger" in names