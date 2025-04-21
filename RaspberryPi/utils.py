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
        backend_url = "https://siaudvytisbenas.dev"
    
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
    
    # We do not present a client cert for HTTP(S) calls
    # session.cert can be configured if you enforce HTTP mTLS in future

    # Use system CA certificates which include Let's Encrypt
    # This allows proper verification of the backend's HTTPS certificate
    session.verify = os.getenv("CA_CERT_PATH", "/etc/ssl/certs/ca-certificates.crt")
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    logger.info(f"Backend session created for URL: {backend_url} using Let's Encrypt certificates")
    
    # Verify backend connectivity
    backend_health_verified = False
    try:
        # Test backend connection - include a reasonable timeout
        logger.info(f"Testing connection to backend at {backend_url}/health")
        response = session.get(f"{backend_url}/health", timeout=5)
        if response.status_code == 200:
            backend_health_verified = True
            logger.info("Successfully connected to backend API")
        else:
            logger.warning(f"Backend responded with non-200 status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to backend: {e}")
        logger.warning("Application will continue, but backend-dependent features will be limited until connection can be established")
    
    if not backend_health_verified:
        # Backend is not available, log a clear warning
        logger.warning("===== BACKEND CONNECTION FAILED =====")
        logger.warning(f"Unable to connect to backend at {backend_url}")
        logger.warning("Most features requiring authentication will not work")
        logger.warning("The system will continue to try connecting for each request")
        logger.warning("=====================================")
    
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