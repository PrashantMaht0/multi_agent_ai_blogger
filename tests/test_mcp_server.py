"""
tests/test_mcp_server.py
MCP server tests over FastMCP's in-memory transport. The HTTP layer is mocked,
so no Tavily credits are spent and no Google credentials are touched.
"""

import pytest
from fastmcp import Client

from src.mcp_servers.search_server import mcp as search_mcp
from src.mcp_servers.blogger_server import mcp as blogger_mcp
import src.mcp_servers.search_server as search_server


class FakeTavilyResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "answer": "MCP is an open standard.",
            "results": [
                {"title": "Spec", "url": "https://example.com/spec", "content": "Protocol details."}
            ],
        }


@pytest.mark.asyncio
async def test_search_tool_formats_results(monkeypatch):
    monkeypatch.setattr(search_server.requests, "post", lambda *a, **kw: FakeTavilyResponse())

    async with Client(search_mcp) as client:
        tools = await client.list_tools()
        assert "search_web" in [t.name for t in tools]

        result = await client.call_tool("search_web", {"query": "what is mcp"})
        output = result.data if hasattr(result, "data") else str(result)

    assert "MCP is an open standard." in output
    assert "https://example.com/spec" in output


@pytest.mark.asyncio
async def test_search_tool_reports_missing_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    async with Client(search_mcp) as client:
        result = await client.call_tool("search_web", {"query": "what is mcp"})
        output = result.data if hasattr(result, "data") else str(result)

    assert "TAVILY_API_KEY is not set" in output


@pytest.mark.asyncio
async def test_search_tool_returns_error_string_on_failure(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(search_server.requests, "post", boom)

    async with Client(search_mcp) as client:
        result = await client.call_tool("search_web", {"query": "what is mcp"})
        output = result.data if hasattr(result, "data") else str(result)

    assert "Error executing web search" in output


@pytest.mark.asyncio
async def test_blogger_server_exposes_publish_tool():
    """Tool discovery only - publishing would need real OAuth credentials."""
    async with Client(blogger_mcp) as client:
        tools = await client.list_tools()

    assert "publish_to_blogger" in [t.name for t in tools]
