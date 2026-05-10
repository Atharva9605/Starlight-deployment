import os
import base64
from google.oauth2 import service_account
from googleapiclient.discovery import build
from email import message_from_string
import logging

# --- Configuration ---
# Make sure "service_account.json" is in the same directory
# or provide the full path.
SERVICE_ACCOUNT_FILE = os.getenv("GSUITE_SERVICE_ACCOUNT_JSON", "service_account.json")
DELEGATED_USER = os.getenv("GSUITE_DELEGATED_USER", "vivek@starlightlinearled.com")
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

logging.basicConfig(level=logging.INFO)

def get_gmail_service():
    """Authenticate and return Gmail API service."""
    try:
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=SCOPES,
            subject=DELEGATED_USER
        )
        service = build("gmail", "v1", credentials=credentials)
        return service
    except Exception as e:
        logging.error(f"Failed to create GMail service: {e}")
        logging.error("Please ensure 'service_account.json' is correct and has GSuite domain-wide delegation.")
        return None

# Build the service object once when the module is loaded
service = get_gmail_service()

def send_email_gsuite(eml_path: str, sender_email: str = None):
    """
    Sends an email using Gmail API from a .eml file.
    """
    if not service:
        logging.error("Gmail service is not available. Cannot send email.")
        return False

    if not os.path.exists(eml_path):
        logging.error(f"❌ .eml file not found: {eml_path}")
        return False

    try:
        with open(eml_path, "r", encoding="utf-8") as f:
            raw_email = f.read()

        msg = message_from_string(raw_email)

        # Override sender if provided (matches DELEGATED_USER)
        if sender_email:
            msg.replace_header("From", sender_email)

        encoded_message = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")

        send_result = service.users().messages().send(
            userId="me",  # 'me' refers to the DELEGATED_USER
            body={"raw": encoded_message}
        ).execute()
        
        logging.info(f"✅ Email sent successfully: {send_result['id']} to {msg['To']}")
        return True
        
    except Exception as e:
        logging.error(f"❌ Failed to send email: {e}")
        return False