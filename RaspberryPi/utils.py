import requests
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context
import os

def create_backend_session():
    backend_url = os.getenv("BACKEND_URL")
    backend_ca_cert = os.getenv("CA_CERT")

    # Create a custom SSL context that trusts the backend cert
    ctx = create_urllib3_context()
    ctx.load_verify_locations(backend_ca_cert)

    # Create a persistent session
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=3, pool_connections=10, pool_maxsize=100))
    session.verify = backend_ca_cert

    return session, backend_url

def verify_otp_rest(session, backend_url, phone_number, otp_code):
    try:
        payload = {"phone_number": phone_number, "otp_code": otp_code}
        print(f"[DEBUG] Sending OTP verification request to backend: {payload}")
        
        # Send the request to the backend
        response = session.post(f"{backend_url}/check-verification-RPI", json=payload)
        print(f"[DEBUG] Backend response: {response.status_code} - {response.text}")
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "message": response.json().get("error", "Unknown error")}
    except Exception as e:
        print(f"[DEBUG] Error in verify_otp_rest: {str(e)}")
        return {"status": "error", "message": str(e)} 