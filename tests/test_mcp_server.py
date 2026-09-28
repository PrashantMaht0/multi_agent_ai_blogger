"""MCP server tests over the in-memory transport, with the HTTP layer mocked."""

import json

import pytest
from fastmcp import Client

from src.mcp_servers.search_server import mcp as search_mcp
from src.mcp_servers.blogger_server import mcp as blogger_mcp
import src.mcp_servers.search_server as search_server


class FakeTavilyResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"results": [
            {"title": "Spec", "url": "https://example.com/spec", "raw_content": "x" * 9000,
             "published_date": "2025-04-02"},
            {"title": "Empty", "url": "https://example.com/empty", "raw_content": "   "},
        ]}


async def _search(query="what is mcp"):
    async with Client(search_mcp) as client:
        result = await client.call_tool("search_sources", {"query": query})
    return json.loads(result.content[0].text)


@pytest.mark.asyncio
async def test_search_sources_asks_for_page_text_and_drops_empty_pages(monkeypatch):
    sent = {}

    def fake_post(url, json):
        sent.update(json)
        return FakeTavilyResponse()

    monkeypatch.setattr(search_server.requests, "post", fake_post)
    sources = await _search()

    assert sent["include_raw_content"] is True and sent["max_results"] == 3
    assert [s["url"] for s in sources] == ["https://example.com/spec"]
    assert len(sources[0]["raw_content"]) == 6000
    assert sources[0]["published_date"] == "2025-04-02"


@pytest.mark.asyncio
async def test_search_sources_reports_missing_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert "TAVILY_API_KEY is not set" in (await _search())["error"]


@pytest.mark.asyncio
async def test_search_sources_returns_an_error_on_failure(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(search_server.requests, "post", boom)
    assert "connection reset" in (await _search())["error"]


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
