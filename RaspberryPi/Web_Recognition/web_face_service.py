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
        Initialize the face recognition service.
        
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
        self.camera = WebCamera(config={'headless': headless})
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
            # Only release resources if already initialized
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
    
    def capture_face_with_liveness(self, timeout=30):
        """
        Capture a face with liveness checks.
        
        Args:
            timeout: Maximum seconds to wait for liveness detection
        
        Returns:
            numpy.ndarray or None: Face image if successful
        """
        try:
            return self.camera.capture_face_with_liveness(timeout=timeout)
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
                    
                # Decode the face encoding using our helper method
                try:
                    encoding = self.decode_face_data(face_encoding_b64)
                    if encoding is not None:
                        encodings.append(encoding)
                        names.append(phone_number)
                        logger.info(f"Decoded face encoding for {phone_number}")
                except Exception as e:
                    logger.error(f"Error decoding face encoding for {phone_number}: {e}")
                    logger.error(traceback.format_exc())
            
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
        Decode face encoding from various possible formats.
        
        Args:
            face_encoding_b64: Face encoding data that could be in different formats:
                               - A list directly
                               - Base64 encoded JSON string
                               - Binary pickled numpy array
                               
        Returns:
            numpy.ndarray or None: Decoded face encoding
        """
        try:
            # Check if face_encoding_b64 is already a list or numpy array
            if isinstance(face_encoding_b64, (list, np.ndarray)):
                return np.array(face_encoding_b64)
                
            # Try to decode as base64
            try:
                # First attempt: Try as base64 encoded JSON string
                decoded_data = base64.b64decode(face_encoding_b64)
                
                # Try to decode as UTF-8 string (JSON)
                try:
                    encoding_json = decoded_data.decode('utf-8')
                    encoding_list = json.loads(encoding_json)
                    return np.array(encoding_list)
                except UnicodeDecodeError:
                    # Not a UTF-8 string, could be binary pickled data
                    logger.info("Face encoding is not UTF-8 encoded JSON, trying alternative decoding")
                    
                    # Try to interpret as pickled numpy array
                    try:
                        import pickle
                        encoding_array = pickle.loads(decoded_data)
                        if isinstance(encoding_array, np.ndarray):
                            return encoding_array
                    except:
                        pass
                        
                    # Last resort: try to interpret as raw numpy array data
                    try:
                        encoding_array = np.frombuffer(decoded_data, dtype=np.float64)
                        # Most face recognition models use 128-dimensional face embeddings
                        if len(encoding_array) == 128:
                            return encoding_array
                    except:
                        pass
            except:
                logger.warning("Could not decode as base64")
                
            # If we got here, we couldn't decode the data
            logger.error(f"Unable to decode face encoding data, unknown format")
            return None
                
        except Exception as e:
            logger.error(f"Error decoding face data: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def release_camera(self):
        """
        Comprehensive camera resource cleanup and service reset.
        This is the central method for ensuring all resources are properly released.
        """
        try:
            logger.info("Performing comprehensive camera cleanup")
            
            # Explicitly destroy any OpenCV windows first
            import cv2
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Process window destruction
            
            # First close camera if it exists
            if hasattr(self, 'camera') and self.camera:
                self.camera._release_camera()
            
            # Kill any stray OpenCV processes
            try:
                import subprocess
                subprocess.run("pkill -f cv2", shell=True, timeout=1)
            except:
                pass
            
            # Reset service state
            self.initialized = False
            
            # Recreate the camera instance with fresh state
            import os
            headless = getattr(self.camera, 'headless', True) if hasattr(self, 'camera') else True
            self.camera = WebCamera(config={'headless': headless})
            
            # Allow time for resources to be fully released
            import time
            time.sleep(0.5)
            
            logger.info("Camera resources fully released and service reset")
        except Exception as e:
            logger.error(f"Error during comprehensive camera cleanup: {e}")
            logger.error(traceback.format_exc()) 