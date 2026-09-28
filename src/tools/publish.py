"""Publish step: posts the approved draft through the Blogger MCP tool. A tool, not an agent: no model."""

import sys
import json
import hashlib
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from src.common.errors import root_cause
from src.state import AgentState

mcp_config = {
    "blogger_server": {
        "command": sys.executable,
        "args": ["src/mcp_servers/blogger_server.py"],
        "transport": "stdio"
    }
}


def content_hash(html: str) -> str:
    return hashlib.sha256(html.encode("utf-8")).hexdigest()


async def _publish_via_mcp(title: str, draft_html: str) -> dict:
    client = MultiServerMCPClient(mcp_config)
    async with client.session("blogger_server") as session:
        result = await session.call_tool("publish_to_blogger", {"title": title, "content": draft_html})
    return json.loads(result.content[0].text)


def publish_node(state: AgentState) -> dict:
    draft = state["draft"]
    title = state["title"]

    # Refuse if the draft changed after the sanitizer approved it for review.
    if content_hash(draft) != state.get("approved_sha256"):
        print("Publish refused: draft differs from the reviewed version.")
        return {"blogger_url": "Failed to publish: draft changed after review.", "sender": "publish"}

    print(f"🌐 Publishing '{title}' to Blogger...")
    try:
        reply = asyncio.run(_publish_via_mcp(title, draft))
    except Exception as e:
        print(f"Publish tool call failed: {root_cause(e)}")
        return {"blogger_url": "Failed to publish.", "sender": "publish"}

    if reply.get("error") or not reply.get("url"):
        return {"blogger_url": f"Failed to publish: {reply.get('error', 'no URL returned')}", "sender": "publish"}

    # The server hashes what it received, so any change in transit is caught.
    if reply.get("content_sha256") != content_hash(draft):
        return {"blogger_url": f"Published, but the posted content differs from the approved draft: {reply['url']}",
                "sender": "publish"}

    return {"blogger_url": reply["url"], "sender": "publish"}
