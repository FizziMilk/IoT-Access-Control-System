from Test_Recognition.face_system import FaceRecognitionSystem
import numpy as np
import pickle
import base64
import requests

class FaceRecognitionService:
    def __init__(self, backend_session=None, backend_url=None):
        self.face_system = FaceRecognitionSystem()
        self.backend_session = backend_session
        self.backend_url = backend_url
        
    def identify_user(self):
        """
        Use facial recognition with liveness detection to identify a user
        
        Returns:
            dict or None: User match information with name and confidence if found
        """
        try:
            # If we have backend connection, try to load known faces first
            if self.backend_session and self.backend_url:
                self.load_faces_from_backend()
                
            results = self.face_system.recognize_face(use_liveness=True)
            if results and len(results) > 0:
                # Return the highest confidence match
                return results[0]
            return None
        finally:
            # Ensure camera is released
            self.release_camera()
        
    def release_camera(self):
        """
        Explicitly release camera resources to prevent resource locks
        """
        try:
            if hasattr(self.face_system, 'camera') and hasattr(self.face_system.camera, 'video_capture'):
                if self.face_system.camera.video_capture is not None:
                    self.face_system.camera.video_capture.release()
                    self.face_system.camera.video_capture = None
                    print("Camera resources released successfully")
        except Exception as e:
            print(f"Error releasing camera: {e}")
        
    def load_faces_from_backend(self):
        """
        Load face encodings from the backend database
        
        Returns:
            bool: True if faces were loaded successfully
        """
        try:
            # Get all users with registered faces
            response = self.backend_session.get(f"{self.backend_url}/get-face-data")
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    face_data = data.get("face_data", [])
                    print(f"Retrieved {len(face_data)} faces from backend")
                    
                    # Process each face entry
                    for entry in face_data:
                        phone_number = entry.get("phone_number")
                        face_encoding_b64 = entry.get("face_encoding")
                        
                        if phone_number and face_encoding_b64:
                            # Decode the face data
                            encoding = self.decode_face_data(face_encoding_b64)
                            if encoding is not None:
                                # Add to storage system
                                self.face_system.storage.add_encoding(phone_number, encoding)
                    
                    return True
            return False
        except Exception as e:
            print(f"Error loading faces from backend: {e}")
            return False
        
    def capture_face_with_liveness(self):
        """
        Capture a face with liveness detection
        
        Returns:
            image or None: Face image if capture and liveness check successful
        """
        try:
            return self.face_system.camera.capture_face_with_liveness()
        finally:
            self.release_camera()
        
    def register_face(self, face_image):
        """
        Generate encoding from a face image and return it as base64 string
        
        Args:
            face_image: OpenCV image containing a face
            
        Returns:
            str or None: Base64 encoded face data if successful
        """
        encoding = self.face_system.recognition.generate_encoding(face_image)
        if encoding is not None:
            # Convert numpy array to bytes using pickle
            pickled_encoding = pickle.dumps(encoding)
            # Convert bytes to base64 string for JSON compatibility
            base64_encoding = base64.b64encode(pickled_encoding).decode('utf-8')
            return base64_encoding
        return None
        
    def decode_face_data(self, base64_face_data):
        """
        Decode base64 face data back to numpy array
        
        Args:
            base64_face_data: Base64 encoded string of face data
            
        Returns:
            numpy.ndarray: Face encoding
        """
        try:
            # Convert base64 string back to bytes
            pickled_data = base64.b64decode(base64_face_data)
            # Convert bytes back to numpy array
            encoding = pickle.loads(pickled_data)
            return encoding
        except Exception as e:
            print(f"Error decoding face data: {e}")
            return None 