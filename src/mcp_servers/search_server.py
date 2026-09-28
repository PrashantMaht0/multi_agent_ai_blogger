"""MCP server exposing web search to the researcher branches."""

import os
import json
import requests
from fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP(name="ResearchServer")

MAX_SOURCE_CHARS = 6000


@mcp.tool
def search_sources(query: str) -> str:
    """Searches the web and returns up to 3 sources with their page text, as JSON."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return json.dumps({"error": "TAVILY_API_KEY is not set in the environment variables."})

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "include_answer": False,
        "include_raw_content": True,
        "max_results": 3,
    }
    try:
        response = requests.post("https://api.tavily.com/search", json=payload)
        response.raise_for_status()
        results = response.json().get("results", [])
    except Exception as e:
        return json.dumps({"error": f"Error executing web search: {e}"})

    # Pages with no text cannot back a quote, so they are dropped here.
    sources = [
        {
            "url": item.get("url") or "",
            "title": item.get("title") or "",
            "published_date": item.get("published_date"),
            "raw_content": (item.get("raw_content") or "")[:MAX_SOURCE_CHARS],
        }
        for item in results
        if (item.get("raw_content") or "").strip() and item.get("url")
    ]
    return json.dumps(sources)

if __name__ == "__main__":
    mcp.run(transport="stdio")