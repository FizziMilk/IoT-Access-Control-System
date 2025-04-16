"""
Web-optimized face recognition system for IoT access control.
Handles face detection, analysis, and identity matching.
"""
import face_recognition
import numpy as np
import cv2
import logging
import time
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WebRecognition")

class WebRecognition:
    """Face recognition handling optimized for web applications."""
    
    def __init__(self, face_encodings=None, face_names=None):
        """
        Initialize the recognition system.
        
        Args:
            face_encodings: Optional list of face encodings
            face_names: Optional list of face names corresponding to encodings
        """
        # Initialize with empty lists if not provided
        self.known_face_encodings = face_encodings or []
        self.known_face_names = face_names or []
        
        # Settings
        self.detection_model = "hog"  # Use HOG for better performance (vs CNN)
        self.recognition_tolerance = 0.6  # Threshold for face matching (lower = stricter)
    
    def identify_face(self, frame, face_location=None):
        """
        Identify a face in a frame by comparing with known encodings.
        
        Args:
            frame: Image containing a face
            face_location: Optional pre-detected face location
            
        Returns:
            dict or None: Match information (name, confidence) if found, None otherwise
        """
        try:
            if not self.known_face_encodings:
                logger.warning("No known faces to match against")
                return None
                
            # Detect face locations if not provided
            if face_location is None:
                # Resize for faster detection
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                face_locations = face_recognition.face_locations(rgb_small_frame, model=self.detection_model)
                
                if not face_locations:
                    logger.warning("No face detected in the image")
                    return None
                    
                # Use the first detected face
                small_location = face_locations[0]
                
                # Scale back to original size
                top, right, bottom, left = small_location
                face_location = (top*2, right*2, bottom*2, left*2)
            
            # Extract face encoding
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_encoding = face_recognition.face_encodings(rgb_frame, [face_location])[0]
            
            # Compare with known faces
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            if len(face_distances) == 0:
                logger.warning("No face distances computed - empty encodings list")
                return None
            
            # Find the best match
            best_match_index = np.argmin(face_distances)
            min_distance = face_distances[best_match_index]
            
            # Check if the match is good enough
            if min_distance <= self.recognition_tolerance:
                confidence = 1.0 - min_distance  # Convert distance to confidence score
                match = {
                    "name": self.known_face_names[best_match_index],
                    "confidence": confidence
                }
                logger.info(f"Face identified as {match['name']} with confidence {confidence:.2f}")
                return match
            else:
                logger.info(f"Face not recognized (best distance: {min_distance:.2f}, threshold: {self.recognition_tolerance})")
                return None
                
        except Exception as e:
            logger.error(f"Error in face identification: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def process_face_encoding(self, frame):
        """
        Generate a face encoding from an image for storage.
        
        Args:
            frame: Image containing a face
            
        Returns:
            list: Face encoding if a face is found, None otherwise
        """
        try:
            # Resize for faster detection
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_small_frame, model=self.detection_model)
            
            if not face_locations:
                logger.warning("No face detected in the image for encoding")
                return None
                
            # Convert frame to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Generate face encoding
            face_encoding = face_recognition.face_encodings(rgb_frame, [face_locations[0]])[0]
            
            # Convert numpy array to list for JSON serialization
            encoding_list = face_encoding.tolist()
            
            logger.info("Face encoding generated successfully")
            return encoding_list
            
        except Exception as e:
            logger.error(f"Error generating face encoding: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def add_face(self, name, encoding):
        """
        Add a new face to the recognition system.
        
        Args:
            name: Identifier for the face (usually phone number)
            encoding: Face encoding (from process_face_encoding)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not isinstance(encoding, (list, np.ndarray)):
                logger.error("Invalid encoding format")
                return False
                
            # Convert to numpy array if it's a list
            if isinstance(encoding, list):
                encoding = np.array(encoding)
                
            # Add to known faces
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(name)
            
            logger.info(f"Added face for {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding face: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def load_encodings(self, encodings, names):
        """
        Load multiple face encodings at once.
        
        Args:
            encodings: List of face encodings
            names: List of names corresponding to encodings
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if len(encodings) != len(names):
                logger.error("Number of encodings and names must match")
                return False
                
            # Clear existing encodings
            self.known_face_encodings = []
            self.known_face_names = []
            
            # Add each encoding
            for encoding, name in zip(encodings, names):
                # Convert to numpy array if it's a list
                if isinstance(encoding, list):
                    encoding = np.array(encoding)
                
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(name)
                
            logger.info(f"Loaded {len(encodings)} face encodings")
            return True
            
        except Exception as e:
            logger.error(f"Error loading encodings: {e}")
            logger.error(traceback.format_exc())
            return False 