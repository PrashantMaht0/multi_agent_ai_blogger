"""MCP server exposing web search to the Researcher agent."""

import os
import requests
from fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP(name="ResearchServer")

@mcp.tool
def search_web(query: str) -> str:
    """Searches the web and returns a summary of the top results."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY is not set in the environment variables."
    
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "include_answer": True,
        "max_results": 3
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Flatten the response into one string for the model.
        results = [f"Result: {data.get('answer', 'No direct answer generated.')}\n"]
        for item in data.get("results", []):
            results.append(f"- Source: {item.get('title')} ({item.get('url')})\n  Snippet: {item.get('content')}")
            
        return "\n".join(results)
    
    except Exception as e:
        return f"Error executing web search: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")