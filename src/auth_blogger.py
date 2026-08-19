"""
auth_blogger.py
A one-time script to trigger the Google OAuth flow and generate token.json.
"""
from google_auth_oauthlib.flow import InstalledAppFlow

# The exact permission scope needed for the Blogger API
SCOPES = ['https://www.googleapis.com/auth/blogger']

def generate_token():
    print("Launching browser for Google Authentication...")
    
    # Point this to your downloaded OAuth Desktop App credentials
    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
    creds = flow.run_local_server(port=0)
    
    # Save the generated access token
    with open('token.json', 'w') as token:
        token.write(creds.to_json())
        
    print("✅ Authentication successful! 'token.json' has been saved.")
    print("You can now delete this script and run app.py normally.")

if __name__ == "__main__":
    generate_token()