from .camera import CameraSystem
from .storage import StorageSystem
from .recognition import RecognitionSystem
import cv2
import os

class FaceRecognitionSystem:
    def __init__(self, web_mode=True):
        # Initialize subsystems
        self.camera = CameraSystem()
        self.storage = StorageSystem()
        self.recognition = RecognitionSystem(self.storage)
        # When in web mode, disable interactive prompts and use non-blocking behavior
        self.web_mode = web_mode
    
    def release_resources(self):
        """Release all resources including camera"""
        if hasattr(self.camera, 'video_capture') and self.camera.video_capture is not None:
            self.camera.video_capture.release()
            self.camera.video_capture = None
            print("Released camera resources in FaceRecognitionSystem")
            
        # Make sure all OpenCV windows are closed
        cv2.destroyAllWindows()
    
    def register_new_user(self, use_liveness=True):
        """Register a new user face in the system"""
        if self.web_mode:
            print("[DEBUG] register_new_user called in web_mode - using default name")
            name = "new_user"  # Default name when in web mode
        else:
            # Get user name
            name = input("Enter name for the new user: ")
            if not name.strip():
                print("Name cannot be empty")
                return False
        
        # Capture face image with liveness check if requested
        print("Capturing face image...")
        if use_liveness:
            print("With liveness detection to prevent spoofing...")
            image = self.camera.capture_face_with_liveness()
        else:
            image = self.camera.capture_face()
        
        if image is None:
            print("Image capture cancelled or failed")
            return False
        
        # Add user to storage
        result = self.recognition.add_face(name, image)
        
        if not result:
            print("No face found in the image. Please try again.")
            return False
            
        print(f"Successfully registered {name}")
        return True
    
    def recognize_face(self, use_liveness=True, timeout=15):
        """
        Recognize a face from camera
        
        Args:
            use_liveness: Whether to perform liveness detection
            timeout: Maximum seconds to wait for recognition process
            
        Returns:
            list: Recognition results if successful, None otherwise
        """
        print(f"[DEBUG] recognize_face: Starting face recognition process (timeout={timeout}s)")
        if not self.storage.has_faces():
            print("[DEBUG] No face encodings in database. Please register users first.")
            return None
        
        print(f"[DEBUG] Found {len(self.storage.known_face_names)} face(s) in storage")
        print(f"[DEBUG] Face names: {self.storage.known_face_names}")
        
        # Reset camera system to ensure it's in a clean state
        print("[DEBUG] Resetting camera system before recognition")
        self.camera.reset_camera()
        
        # Capture image with liveness check if requested
        print("[DEBUG] Capturing image for recognition...")
        if use_liveness:
            print(f"[DEBUG] With liveness detection to prevent spoofing (timeout={timeout}s)...")
            image = self.camera.capture_face_with_liveness(timeout=timeout)
        else:
            image = self.camera.capture_face()
        
        if image is None:
            print("[DEBUG] Image capture cancelled or failed")
            return None
        
        print("[DEBUG] Face image captured successfully")
        
        # Recognize faces in image
        print("[DEBUG] Identifying faces in the captured image")
        results = self.recognition.identify_faces(image)
        
        if not results:
            print("[DEBUG] No faces detected in the image")
            return None
            
        print(f"[DEBUG] Recognition successful. Results: {results}")
        return results
    
    def list_users(self):
        """List all registered users"""
        self.storage.list_users()
    
    def delete_user(self):
        """Delete a user from the database"""
        return self.storage.delete_user()
    
    def get_encoding_for_backend(self):
        """Capture face and generate encoding to send to backend"""
        # Capture image
        print("Capturing image to generate encoding...")
        image = self.camera.capture_face()
        
        if image is None:
            print("Image capture cancelled")
            return None
        
        # Generate encoding
        return self.recognition.generate_encoding(image) 