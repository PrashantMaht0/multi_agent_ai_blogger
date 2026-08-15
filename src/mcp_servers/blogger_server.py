"""
src/mcp_servers/blogger_server.py
FastMCP Server exposing Google Blogger publishing tools.
"""

import os
from fastmcp import FastMCP
from google.oauth2 import service_account
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the FastMCP server instance
mcp = FastMCP(name="BloggerServer")

def get_blogger_service():
    """Initializes and returns the authenticated Blogger v3 API service."""
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path or not os.path.exists(creds_path):
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS path is invalid or missing.")
    
    # Required scope for inserting posts
    scopes = ['https://www.googleapis.com/auth/blogger']
    
    creds = service_account.Credentials.from_service_account_file(
        creds_path, scopes=scopes
    )
    
    # Build the API client
    return build('blogger', 'v3', credentials=creds)

@mcp.tool
def publish_to_blogger(title: str, content_html: str, labels: list[str]) -> str:
    """
    Publishes a formatted HTML blog post to Google Blogger.
    Returns the live URL of the published post.
    """
    blog_id = os.getenv("BLOGGER_BLOG_ID")
    if not blog_id:
        return "Error: BLOGGER_BLOG_ID environment variable is missing."
    
    try:
        service = get_blogger_service()
        
        # The Blogger API requires a specific JSON body payload structure
        body = {
            "kind": "blogger#post",
            "title": title,
            "content": content_html,
            "labels": labels
        }
        
        # Execute the insert request
        request = service.posts().insert(
            blogId=blog_id, 
            body=body, 
            isDraft=False # Set to True if you want it published as a draft first
        )
        response = request.execute()
        
        # The API returns the live URL in the 'url' key of the response dictionary
        live_url = response.get("url", "URL not found in API response.")
        return live_url
        
    except Exception as e:
        return f"Failed to publish post: {str(e)}"

if __name__ == "__main__":
    # Runs on stdio for MCP clients to connect
    mcp.run(transport="stdio")
    