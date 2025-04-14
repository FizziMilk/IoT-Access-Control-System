import face_recognition
import cv2
import numpy as np

class FaceRecognition:
    """
    Face recognition functionality for identifying faces and generating encodings.
    """
    def __init__(self):
        pass
        
    def get_face_encoding(self, image):
        """
        Extract face encoding from an image.
        
        Args:
            image: Input image containing a face
            
        Returns:
            numpy.ndarray: Face encoding or None if no face detected
        """
        if image is None:
            return None
            
        # Process the image to find face encodings
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # First get face locations
        face_locations = face_recognition.face_locations(rgb_image)
        
        if not face_locations:
            return None
        
        # Get encoding
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        if not face_encodings:
            return None
        
        return face_encodings[0]
        
    def recognize_face(self, image, known_faces):
        """
        Recognize a face in an image against a list of known faces.
        
        Args:
            image: Input image containing a face
            known_faces: List of tuples (name, encoding, face_image)
            
        Returns:
            tuple: (name, confidence) or (None, 0) if no match
        """
        if image is None or not known_faces:
            return None, 0
            
        # Extract face encoding from image
        face_encoding = self.get_face_encoding(image)
        
        if face_encoding is None:
            return None, 0
            
        # Get lists of known encodings and names
        known_names = [face[0] for face in known_faces]
        known_encodings = [face[1] for face in known_faces]
        
        # Compare against known faces
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        
        # Calculate face distances
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        
        # Find best match
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            confidence = 1.0 - face_distances[best_match_index]
            
            # Check if it's a good match
            if matches[best_match_index] and confidence > 0.6:
                return known_names[best_match_index], confidence
                
        return None, 0

# Keep the original RecognitionSystem for backward compatibility
class RecognitionSystem:
    def __init__(self, storage):
        self.storage = storage
    
    def add_face(self, name, image):
        """Process image and add face to storage"""
        # Process the image to find face encodings
        # Ensure image is the correct format (RGB and uint8)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect face locations first
        face_locations = face_recognition.face_locations(rgb_image)
        if not face_locations:
            return False
            
        # Then get encodings using the located faces
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        if not face_encodings:
            return False
        
        # Add to storage
        self.storage.add_face(name, face_encodings[0], image)
        return True
    
    def identify_faces(self, image):
        """Identify faces in an image"""
        # Get known faces
        known_face_encodings, known_face_names = self.storage.get_all_faces()
        
        if not known_face_encodings:
            return []
        
        # Process the image - ensure correct format
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # First get face locations
        face_locations = face_recognition.face_locations(rgb_image)
        
        if not face_locations:
            return []
        
        # Get encodings for all faces
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        results = []
        
        # Check each detected face
        for i, face_encoding in enumerate(face_encodings):
            # Compare face with all known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            
            # Calculate distances
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            
            # Find best match
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]
                
                # Display result on image
                top, right, bottom, left = face_locations[i]
                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Draw label
                cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(image, f"{name} ({confidence:.2f})", (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                
                results.append({"name": name, "confidence": confidence})
            else:
                # Unknown face
                top, right, bottom, left = face_locations[i]
                cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(image, "Unknown", (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                
                results.append({"name": "Unknown", "confidence": 0.0})
        
        # Show the image with results
        cv2.imshow("Recognition Results", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return results
    
    def generate_encoding(self, image):
        """Generate face encoding from image for backend use"""
        # Process the image to find face encodings
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # First get face locations
        face_locations = face_recognition.face_locations(rgb_image)
        
        if not face_locations:
            return None
        
        # Get encoding
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        if not face_encodings:
            return None
        
        # Convert to list for JSON serialization
        encoding = face_encodings[0].tolist()
        return encoding 