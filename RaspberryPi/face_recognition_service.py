from Test_Recognition.face_system import FaceRecognitionSystem
import numpy as np
import pickle
import base64
import requests
import time

class FaceRecognitionService:
    def __init__(self, backend_session=None, backend_url=None):
        print("[DEBUG] Initializing FaceRecognitionService")
        self.face_system = FaceRecognitionSystem()
        self.backend_session = backend_session
        self.backend_url = backend_url
        
    def identify_user(self, timeout=30):
        """
        Use facial recognition with liveness detection to identify a user
        
        Args:
            timeout: Number of seconds to allow for recognition process
            
        Returns:
            dict or None: User match information with name and confidence if found
        """
        print(f"[DEBUG] identify_user: Starting facial recognition with liveness (timeout={timeout}s)")
        start_time = time.time()
        try:
            # If we have backend connection, try to load known faces first
            if self.backend_session and self.backend_url:
                print("[DEBUG] Loading faces from backend...")
                self.load_faces_from_backend()
            
            # Check if we have any faces loaded
            has_faces = self.face_system.storage.has_faces()
            print(f"[DEBUG] Storage system has faces: {has_faces}")
            if not has_faces:
                print("[DEBUG] No faces found in storage - facial recognition will fail")
                
            # Set timeout for the recognition operation
            custom_timeout = max(5, min(timeout, 60))  # Between 5 and 60 seconds
            print(f"[DEBUG] Setting recognition timeout to {custom_timeout} seconds")
            
            # Pass timeout to the face system
            results = self.face_system.recognize_face(use_liveness=True, timeout=custom_timeout)
            elapsed = time.time() - start_time
            print(f"[DEBUG] Recognition completed in {elapsed:.1f} seconds with results: {results}")
            
            if results and len(results) > 0:
                # Return the highest confidence match
                print(f"[DEBUG] Returning match: {results[0]}")
                return results[0]
            return None
        finally:
            # Ensure camera is released
            elapsed = time.time() - start_time
            print(f"[DEBUG] identify_user: Cleaning up after facial recognition, elapsed time: {elapsed:.1f}s")
            self.release_camera()
        
    def release_camera(self):
        """
        Explicitly release camera resources to prevent resource locks
        """
        print("[DEBUG] release_camera: Attempting to release camera resources")
        try:
            if hasattr(self.face_system, 'camera'):
                # Use the more thorough reset method
                print("[DEBUG] Calling camera reset method")
                self.face_system.camera.reset_camera()
                print("[DEBUG] Camera resources reset successfully")
                
                # For extra cleanup, delay a bit and try to open and close the camera one more time
                time.sleep(1.0)
                try:
                    import cv2
                    print("[DEBUG] Attempting to open and immediately close camera")
                    cap = cv2.VideoCapture(0)
                    if cap.isOpened():
                        print("[DEBUG] Successfully opened camera for cleanup")
                    cap.release()
                    print("[DEBUG] Released camera after cleanup attempt")
                    cv2.destroyAllWindows()
                except Exception as e:
                    print(f"[DEBUG] Error during cleanup attempt: {e}")
        except Exception as e:
            print(f"[DEBUG] Error releasing camera: {e}")
            import traceback
            print(traceback.format_exc())
        
    def load_faces_from_backend(self):
        """
        Load face encodings from the backend database
        
        Returns:
            bool: True if faces were loaded successfully
        """
        try:
            print("[DEBUG] load_faces_from_backend: Requesting face data from backend")
            # Get all users with registered faces
            response = self.backend_session.get(f"{self.backend_url}/get-face-data")
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    face_data = data.get("face_data", [])
                    print(f"[DEBUG] Retrieved {len(face_data)} faces from backend")
                    
                    # Process each face entry
                    for entry in face_data:
                        phone_number = entry.get("phone_number")
                        face_encoding_b64 = entry.get("face_encoding")
                        
                        if phone_number and face_encoding_b64:
                            print(f"[DEBUG] Processing face data for {phone_number}")
                            # Decode the face data
                            encoding = self.decode_face_data(face_encoding_b64)
                            if encoding is not None:
                                print(f"[DEBUG] Successfully decoded face encoding for {phone_number}")
                                # Add to storage system
                                try:
                                    print(f"[DEBUG] Adding face encoding to storage for {phone_number}")
                                    result = self.face_system.storage.add_encoding(phone_number, encoding)
                                    print(f"[DEBUG] Add encoding result: {result}")
                                except Exception as e:
                                    print(f"[DEBUG] Error adding encoding: {e}")
                                    import traceback
                                    print(traceback.format_exc())
                    
                    # Verify faces were loaded
                    has_faces = self.face_system.storage.has_faces()
                    face_count = len(self.face_system.storage.known_face_names)
                    print(f"[DEBUG] After loading - has_faces: {has_faces}, face_count: {face_count}")
                    
                    return True
                else:
                    print(f"[DEBUG] Backend returned error status: {data.get('status')}")
            else:
                print(f"[DEBUG] Backend request failed with status code: {response.status_code}")
            return False
        except Exception as e:
            print(f"[DEBUG] Error loading faces from backend: {e}")
            import traceback
            print(traceback.format_exc())
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