"""
src/mcp_servers/blogger_server.py
An MCP server that publishes drafts to Google Blogger via OAuth 2.0.
"""

import os
from fastmcp import FastMCP
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()
mcp = FastMCP("Blogger Publisher")

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/blogger']

def get_blogger_service():
    """Handles OAuth 2.0 authentication and token generation."""
    creds = None
    
    # The file token.json stores the user's access and refresh tokens.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # If there are no valid credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
            
    return build('blogger', 'v3', credentials=creds)

@mcp.tool()
def publish_to_blogger(title: str, content: str, tags: list[str]) -> str:
    """
    Publishes a formatted HTML draft to Google Blogger.
    Returns the live URL of the published post upon success.
    """
    try:
        service = get_blogger_service()
        blog_id = os.getenv("BLOGGER_BLOG_ID")
        
        body = {
            "title": title,
            "content": content,
            "labels": tags
        }
        
        posts = service.posts()
        result = posts.insert(blogId=blog_id, body=body, isDraft=False).execute()
        
        return f"Successfully published to Blogger. Live URL: {result.get('url')}"
        
    except Exception as e:
        return f"Error publishing to Blogger: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
    get_blogger_service()