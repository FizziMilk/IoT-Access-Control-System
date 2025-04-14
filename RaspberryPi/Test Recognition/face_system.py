from camera import CameraSystem
from storage import StorageSystem
from recognition import RecognitionSystem
import cv2
import os

class FaceRecognitionSystem:
    def __init__(self):
        # Initialize subsystems
        self.camera = CameraSystem()
        self.storage = StorageSystem()
        self.recognition = RecognitionSystem(self.storage)
    
    def register_new_user(self):
        """Register a new user face in the system"""
        # Get user name
        name = input("Enter name for the new user: ")
        if not name.strip():
            print("Name cannot be empty")
            return False
        
        # Capture face image
        print("Capturing face image...")
        image = self.camera.capture_face()
        
        if image is None:
            print("Image capture cancelled")
            return False
        
        # Add user to storage
        result = self.recognition.add_face(name, image)
        
        if not result:
            print("No face found in the image. Please try again.")
            return False
            
        print(f"Successfully registered {name}")
        return True
    
    def recognize_face(self):
        """Recognize a face from camera"""
        if not self.storage.has_faces():
            print("No face encodings in database. Please register users first.")
            return None
        
        # Capture image
        print("Capturing image for recognition...")
        image = self.camera.capture_face()
        
        if image is None:
            print("Image capture cancelled")
            return None
        
        # Recognize faces in image
        results = self.recognition.identify_faces(image)
        
        if not results:
            print("No faces detected in the image")
            return None
            
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