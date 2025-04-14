import os
import pickle
import cv2
import numpy as np
from datetime import datetime

class FaceDatabase:
    """
    Storage and management of enrolled face data using a more robust approach.
    Stores face encodings, names, and reference images.
    """
    def __init__(self, db_file='faces.db'):
        """
        Initialize face database with specified db file.
        
        Args:
            db_file: Path to database file
        """
        self.db_file = db_file
        self.images_dir = "face_images"
        
        # Create directory for face images if it doesn't exist
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Load existing data or create new database
        self.faces = []  # List of tuples (name, encoding, image_path)
        self.load_database()
    
    def load_database(self):
        """Load face database from disk"""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "rb") as f:
                    self.faces = pickle.load(f)
                print(f"Loaded {len(self.faces)} faces from database")
            except Exception as e:
                print(f"Error loading database: {str(e)}")
                self.faces = []
        else:
            print("No existing database found. Creating new database.")
            self.faces = []
    
    def save_database(self):
        """Save face database to disk"""
        try:
            with open(self.db_file, "wb") as f:
                pickle.dump(self.faces, f)
            print(f"Saved {len(self.faces)} faces to database")
            return True
        except Exception as e:
            print(f"Error saving database: {str(e)}")
            return False
    
    def add_face(self, name, encoding, image):
        """
        Add a new face to the database
        
        Args:
            name: Person's name
            encoding: Face encoding (numpy array)
            image: Face image for reference
            
        Returns:
            bool: Success or failure
        """
        if name is None or encoding is None or image is None:
            return False
            
        # Generate unique filename for the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(self.images_dir, f"{name}_{timestamp}.jpg")
        
        # Save the image
        cv2.imwrite(image_path, image)
        
        # Add to database (store actual encoding)
        self.faces.append((name, encoding, image_path))
        
        # Save updated database
        return self.save_database()
    
    def get_all_faces(self):
        """
        Get all faces in the database
        
        Returns:
            list: List of tuples (name, encoding, image_path)
        """
        return self.faces
    
    def get_face_by_name(self, name):
        """
        Get a face by name
        
        Args:
            name: Person's name
            
        Returns:
            tuple: (name, encoding, image_path) or None if not found
        """
        for face in self.faces:
            if face[0] == name:
                return face
        return None
    
    def delete_face(self, name):
        """
        Delete a face from the database
        
        Args:
            name: Person's name
            
        Returns:
            bool: Success or failure
        """
        initial_count = len(self.faces)
        self.faces = [face for face in self.faces if face[0] != name]
        
        # Check if any faces were removed
        if len(self.faces) < initial_count:
            return self.save_database()
        
        return False
        
    def list_all_faces(self):
        """Print a list of all enrolled faces"""
        if not self.faces:
            print("No faces enrolled in the database")
            return
            
        print("\nEnrolled Faces:")
        print("--------------")
        for i, (name, _, _) in enumerate(self.faces, 1):
            print(f"{i}. {name}")
        print()

# Keep original StorageSystem for backward compatibility
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