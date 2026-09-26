"""MCP server that publishes drafts to Google Blogger via OAuth 2.0."""

import hashlib
import json
import os
from fastmcp import FastMCP
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()
mcp = FastMCP("Blogger Publisher")

# Changing these scopes means deleting token.json.
SCOPES = ['https://www.googleapis.com/auth/blogger']

def get_blogger_service():
    """Returns an authenticated Blogger client, refreshing or requesting a token."""
    creds = None

    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # Refresh an expired token, or run the browser sign-in flow.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
            
    return build('blogger', 'v3', credentials=creds)

@mcp.tool()
def publish_to_blogger(title: str, content: str, tags: list[str] | None = None) -> str:
    """Publishes an HTML draft and returns JSON with the live URL and a hash of what was sent."""
    try:
        service = get_blogger_service()
        blog_id = os.getenv("BLOGGER_BLOG_ID")

        body = {
            "title": title,
            "content": content,
            "labels": tags or []
        }

        result = service.posts().insert(blogId=blog_id, body=body, isDraft=False).execute()

        return json.dumps({
            "url": result.get("url"),
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        })

    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    mcp.run(transport="stdio")
    get_blogger_service()