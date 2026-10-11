# Sample 4: Hardcoded Secrets and Credentials

AWS_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
DATABASE_PASSWORD = "SuperSecretPassword123!"
API_TOKEN = "sk-live-9876543210fedcba9876543210"


def authenticate_service():
    api_key = "secret_api_key_value_here"
    print("Authenticating with key:", api_key[:4] + "****")
