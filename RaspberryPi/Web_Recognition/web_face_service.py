"""
Web-optimized face recognition service for web integration.
This is the main integration point for web applications.
"""
import os
import cv2
import time
import json
import base64
import numpy as np
import logging
import traceback
from .camera import WebCamera
from .recognition import WebRecognition

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WebFaceService")

class WebFaceService:
    """
    Integrated face recognition service for web applications.
    Handles camera interaction, face recognition, and resource management.
    """
    
    def __init__(self, backend_session=None, backend_url=None, headless=True):
        """
        Initialize the service with optional backend connection.
        
        Args:
            backend_session: Optional requests session for backend API calls
            backend_url: Optional backend URL for API calls
            headless: If True, no windows will be displayed (for server environments)
        """
        logger.info("Initializing WebFaceService")
        
        # Store backend connection info
        self.backend_session = backend_session
        self.backend_url = backend_url
        
        # Initialize components without claiming resources
        self.camera = WebCamera(headless=headless)
        self.recognition = WebRecognition()
        
        # Flag to track initialization status
        self.initialized = False
    
    def initialize(self):
        """
        Initialize the service by loading face data from backend.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.initialized:
                logger.info("Service already initialized")
                return True
                
            # Load face data if backend is available
            if self.backend_session and self.backend_url:
                success = self.load_faces_from_backend()
                if success:
                    logger.info("Service initialized successfully")
                    self.initialized = True
                    return True
                else:
                    logger.warning("Failed to load faces from backend")
            
            # Consider initialization successful even without backend
            # (we'll just have no known faces)
            logger.info("Service initialized without backend")
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def identify_user(self, timeout=30):
        """
        Identify a user using facial recognition with liveness detection.
        
        Args:
            timeout: Maximum seconds to allow for recognition
            
        Returns:
            list: User match information with name and confidence if found, empty list otherwise
        """
        logger.info(f"Starting facial recognition with timeout={timeout}s")
        
        try:
            # Initialize if not already done
            if not self.initialized:
                self.initialize()
            
            if not self.recognition.known_face_encodings:
                logger.warning("No known face encodings available")
                return []
                
            # Capture face with liveness check
            face_image = self.camera.capture_face_with_liveness(timeout=timeout)
            
            if face_image is None:
                logger.warning("No face captured or liveness check failed")
                return []
            
            # Try to identify the face
            match = self.recognition.identify_face(face_image)
            
            if match:
                return [match]  # Return as list for backward compatibility
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error during user identification: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def capture_face_with_liveness(self):
        """
        Capture a face with liveness checks.
        
        Returns:
            numpy.ndarray or None: Face image if successful
        """
        try:
            return self.camera.capture_face_with_liveness()
        except Exception as e:
            logger.error(f"Error during face capture: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def register_face(self, face_image):
        """
        Generate encoding from face image for registration.
        
        Args:
            face_image: OpenCV image with face
            
        Returns:
            list or None: Face encoding if successful
        """
        try:
            return self.recognition.process_face_encoding(face_image)
        except Exception as e:
            logger.error(f"Error during face registration: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def load_faces_from_backend(self):
        """
        Load face encodings from backend.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Requesting face data from backend")
            
            # Request face data from backend
            response = self.backend_session.get(f"{self.backend_url}/get-face-data")
            
            if response.status_code != 200:
                logger.warning(f"Backend returned non-200 status: {response.status_code}")
                return False
                
            data = response.json()
            
            if data.get("status") != "success":
                logger.warning(f"Backend returned error status: {data.get('status')}")
                return False
                
            face_data = data.get("face_data", [])
            logger.info(f"Retrieved {len(face_data)} faces from backend")
            
            # Process face data
            encodings = []
            names = []
            
            for entry in face_data:
                phone_number = entry.get("phone_number")
                face_encoding_b64 = entry.get("face_encoding")
                
                if not phone_number or not face_encoding_b64:
                    logger.warning("Skipping entry with missing data")
                    continue
                    
                # Decode the face encoding
                try:
                    encoding_json = base64.b64decode(face_encoding_b64).decode('utf-8')
                    encoding = json.loads(encoding_json)
                    encodings.append(encoding)
                    names.append(phone_number)
                    logger.info(f"Decoded face encoding for {phone_number}")
                except Exception as e:
                    logger.error(f"Error decoding face encoding: {e}")
            
            # Load encodings into recognition system
            if encodings:
                success = self.recognition.load_encodings(encodings, names)
                if success:
                    logger.info(f"Loaded {len(encodings)} face encodings")
                    return True
            
            logger.warning("No face encodings loaded")
            return False
            
        except Exception as e:
            logger.error(f"Error loading faces from backend: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def decode_face_data(self, face_encoding_b64):
        """
        Decode base64 face encoding.
        
        Args:
            face_encoding_b64: Base64 encoded face data
            
        Returns:
            numpy.ndarray or None: Decoded face encoding
        """
        try:
            encoding_json = base64.b64decode(face_encoding_b64).decode('utf-8')
            encoding_list = json.loads(encoding_json)
            return np.array(encoding_list)
        except Exception as e:
            logger.error(f"Error decoding face data: {e}")
            return None 