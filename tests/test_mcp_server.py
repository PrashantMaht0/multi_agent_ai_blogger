"""MCP server tests over the in-memory transport, with the HTTP layer mocked."""

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
    """Tool discovery only, since publishing needs real credentials."""
    async with Client(blogger_mcp) as client:
        tools = await client.list_tools()

    assert "publish_to_blogger" in [t.name for t in tools]


@pytest.mark.asyncio
async def test_blogger_tool_returns_the_url_and_a_hash_of_what_it_sent(monkeypatch):
    """The API's own URL and a hash of the received content come back, not model text."""
    import hashlib
    import json

    import src.mcp_servers.blogger_server as blogger_server

    sent = {}

    class FakeInsert:
        def execute(self):
            return {"url": "https://example.blogspot.com/post"}

    class FakePosts:
        def insert(self, blogId, body, isDraft):
            sent.update(body)
            return FakeInsert()

    class FakeService:
        def posts(self):
            return FakePosts()

    monkeypatch.setattr(blogger_server, "get_blogger_service", lambda: FakeService())

    async with Client(blogger_mcp) as client:
        result = await client.call_tool("publish_to_blogger", {"title": "T", "content": "<p>x</p>"})

    reply = json.loads(result.data)
    assert reply["url"] == "https://example.blogspot.com/post"
    assert reply["content_sha256"] == hashlib.sha256(b"<p>x</p>").hexdigest()
    assert sent["content"] == "<p>x</p>"
