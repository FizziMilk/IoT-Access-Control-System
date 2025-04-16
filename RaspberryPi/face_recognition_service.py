from Test_Recognition.face_system import FaceRecognitionSystem
import numpy as np
import pickle
import base64
import requests
import time
import os
import cv2

class FaceRecognitionService:
    def __init__(self, backend_session=None, backend_url=None):
        print("[DEBUG] Initializing FaceRecognitionService")
        self.face_system = FaceRecognitionSystem(web_mode=True)
        self.backend_session = backend_session
        self.backend_url = backend_url
        self.camera_open = False
        
    def identify_user(self, timeout=30):
        """
        Use facial recognition with liveness detection to identify a user
        
        Args:
            timeout: Number of seconds to allow for recognition process
            
        Returns:
            dict or None: User match information with name and confidence if found
        """
        print(f"[DEBUG] identify_user: Starting facial recognition with liveness (timeout={timeout}s)")
        try:
            # First ensure we start with a clean state
            self.reset_system()
            
            start_time = time.time()
            # Call the recognize_face method with timeout
            result = self.face_system.recognize_face(timeout=timeout)
            elapsed = time.time() - start_time
            print(f"[DEBUG] Face recognition completed in {elapsed:.2f} seconds")
            
            return result
        except Exception as e:
            print(f"[DEBUG] Error during user identification: {e}")
            return None
        finally:
            # Always release resources when done
            self.release_camera()
        
    def release_camera(self):
        """
        Explicitly release camera resources to prevent resource locks
        """
        print("[DEBUG] release_camera: Attempting to release camera resources")
        try:
            # First try the face system's release method
            try:
                if hasattr(self.face_system, 'video') and self.face_system.video is not None:
                    self.face_system.video.release()
                    print("[DEBUG] Released face system video")
            except Exception as e:
                print(f"[DEBUG] Error releasing face system video: {e}")
                
            # Then try the face system's release_resources method
            try:
                self.face_system.release_resources()
                print("[DEBUG] Released face system resources")
            except Exception as e:
                print(f"[DEBUG] Error releasing face system resources: {e}")
                
            # Close all OpenCV windows
            cv2.destroyAllWindows()
            print("[DEBUG] Destroyed all OpenCV windows")
            
            # Try to release camera directly - only try index 0
            try:
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    cap.release()
                    print(f"[DEBUG] Released camera 0 directly")
            except Exception as e:
                print(f"[DEBUG] Error releasing camera 0: {e}")
                    
            self.camera_open = False
            print("[DEBUG] Camera resources released")
            
            # Force a small delay to let resources be properly released
            time.sleep(0.5)
        except Exception as e:
            print(f"[DEBUG] Error during camera release: {e}")
        
        # Set environment variable to use available Qt plugin
        self.setup_qt_environment()
        
    def setup_qt_environment(self):
        """
        Set up the appropriate Qt environment variables based on available plugins
        """
        import os
        
        try:
            # From the error message, we know "xcb" is available
            os.environ["QT_QPA_PLATFORM"] = "xcb"
            print("[DEBUG] Using Qt platform plugin: xcb")
        except Exception as e:
            print(f"[DEBUG] Error setting up Qt environment: {e}")
        
    def reset_system(self):
        """Completely reset the face recognition system"""
        print("[DEBUG] Resetting face system")
        try:
            # First ensure any existing resources are released
            self.release_camera()
            # Close any lingering OpenCV windows
            cv2.destroyAllWindows()
            # Recreate the face system
            self.face_system = FaceRecognitionSystem(web_mode=True)
            self.camera_open = False
            print("[DEBUG] Face system reset completed")
        except Exception as e:
            print(f"[DEBUG] Error during system reset: {e}")
        
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
        print("[DEBUG] Starting face capture with liveness detection")
        try:
            # First ensure we start with a clean state
            self.reset_system()
            
            # Attempt to capture a live face
            face_image = self.face_system.get_encoding_for_backend()
            if face_image is not None:
                print("[DEBUG] Face captured successfully")
                return face_image
            else:
                print("[DEBUG] Failed to capture face image")
                return None
        except Exception as e:
            print(f"[DEBUG] Error during face capture: {e}")
            return None
        finally:
            # Always release resources when done
            self.release_camera()
        
    def register_face(self, face_image):
        """
        Generate encoding from a face image and return it as base64 string
        
        Args:
            face_image: OpenCV image containing a face
            
        Returns:
            str or None: Base64 encoded face data if successful
        """
        print("[DEBUG] Registering new face")
        try:
            # The face system will handle encoding; we just need to save the result
            encoding = self.face_system.process_face_encoding(face_image)
            if encoding is not None:
                print("[DEBUG] Face encoding generated successfully")
                return encoding.tolist()  # Convert numpy array to list for JSON serialization
            else:
                print("[DEBUG] Failed to generate face encoding")
                return None
        except Exception as e:
            print(f"[DEBUG] Error during face registration: {e}")
            return None
        finally:
            # Always release resources when done
            self.release_camera()
        
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