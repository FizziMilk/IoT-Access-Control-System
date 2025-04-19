"""
Utility functions for the IoT Access Control System.
These functions provide common utilities used across the application.
"""
import os
import requests
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logger = logging.getLogger("Utils")

def create_backend_session(max_retries=3, backoff_factor=0.3):
    """
    Create a session for connecting to the backend with retry capabilities.
    
    Args:
        max_retries: Maximum number of retries for failed requests
        backoff_factor: Backoff factor for retry delay calculation
        
    Returns:
        tuple: (requests.Session, backend_url) - configured session and backend URL
    """
    logger.info("Creating backend session with retry capabilities")
    
    # Get backend URL from environment
    backend_url = os.getenv("BACKEND_URL")
    if not backend_url:
        logger.warning("BACKEND_URL environment variable not set")
        backend_url = "http://localhost:5000"
    
    # Set up retries for robustness
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
    )
    
    # Create session with retry adapter
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    logger.info(f"Backend session created for URL: {backend_url}")
    return session, backend_url


def is_valid_phone_number(phone_number):
    """
    Validate a phone number format.
    Currently enforces simple length validation, can be extended for more complex validation.
    
    Args:
        phone_number: String containing the phone number to validate
        
    Returns:
        bool: True if phone number is valid, False otherwise
    """
    if not phone_number:
        return False
    
    # Strip any non-numeric characters
    digits_only = ''.join(c for c in phone_number if c.isdigit())
    
    # Check length (adjust as needed for your locale)
    if len(digits_only) < 10 or len(digits_only) > 15:
        return False
        
    return True

def verify_otp_rest(session, backend_url, phone_number, otp_code):
    try:
        payload = {"phone_number": phone_number, "otp_code": otp_code}
        print(f"[DEBUG] Sending OTP verification request to backend: {payload}")
        
        # Send the request to the backend
        response = session.post(f"{backend_url}/check-verification-RPI", json=payload)
        print(f"[DEBUG] Backend response: {response.status_code} - {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "approved":
                return data
            else:
                return {"status": "error", "message": "Incorrect OTP code. Please try again."}
        else:
            return {"status": "error", "message": response.json().get("error", "Incorrect OTP code. Please try again.")}
    except Exception as e:
        print(f"[DEBUG] Error in verify_otp_rest: {str(e)}")
        return {"status": "error", "message": str(e)} 