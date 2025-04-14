import os
import pickle
import cv2
from datetime import datetime

class StorageSystem:
    def __init__(self):
        # Directory structure
        self.data_dir = "face_data"
        self.encodings_file = os.path.join(self.data_dir, "encodings.pkl")
        self.images_dir = os.path.join(self.data_dir, "images")
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Load existing encodings or create new database
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_encodings()
    
    def load_encodings(self):
        """Load saved face encodings from disk"""
        if os.path.exists(self.encodings_file):
            with open(self.encodings_file, "rb") as f:
                data = pickle.load(f)
                self.known_face_encodings = data.get("encodings", [])
                self.known_face_names = data.get("names", [])
            print(f"Loaded {len(self.known_face_names)} face encodings")
        else:
            print("No existing encodings found")
    
    def save_encodings(self):
        """Save face encodings to disk"""
        with open(self.encodings_file, "wb") as f:
            data = {"encodings": self.known_face_encodings, "names": self.known_face_names}
            pickle.dump(data, f)
        print(f"Saved {len(self.known_face_names)} face encodings")
    
    def add_face(self, name, encoding, image):
        """Add a new face to the database"""
        self.known_face_encodings.append(encoding)
        self.known_face_names.append(name)
        
        # Save the image for reference
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(self.images_dir, f"{name}_{timestamp}.jpg")
        cv2.imwrite(image_path, image)
        
        # Save updated encodings database
        self.save_encodings()
        return True
    
    def get_all_faces(self):
        """Get all face encodings and names"""
        return self.known_face_encodings, self.known_face_names
    
    def has_faces(self):
        """Check if any faces are registered"""
        return len(self.known_face_names) > 0
    
    def list_users(self):
        """List all registered users"""
        print("\nRegistered Users:")
        print("----------------")
        if not self.known_face_names:
            print("No users registered")
        else:
            for i, name in enumerate(self.known_face_names):
                print(f"{i+1}. {name}")
        print()
    
    def delete_user(self):
        """Delete a user from the database"""
        if not self.known_face_names:
            print("No users registered")
            return False
        
        self.list_users()
        try:
            idx = int(input("Enter number of user to delete (0 to cancel): ")) - 1
            if idx < 0:
                print("Cancelled")
                return False
            
            name = self.known_face_names[idx]
            self.known_face_encodings.pop(idx)
            self.known_face_names.pop(idx)
            
            # Save updated database
            self.save_encodings()
            
            print(f"Deleted user: {name}")
            return True
        except (ValueError, IndexError):
            print("Invalid selection")
            return False 