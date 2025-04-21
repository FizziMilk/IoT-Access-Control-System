"""
Recognition state management for face recognition
"""

class RecognitionState:
    """A centralized state manager for face recognition process"""
    
    def __init__(self):
        self.recognition_running = False
        self.face_recognition_active = False
        self.face_recognition_result = None
        self.recognition_thread = None
        self.face_recognition_progress = None
        self.temp_face_encoding = None  # Store face encoding temporarily
        self.temp_face_id = None  # Store face ID for correlation
        
    def reset(self):
        """Reset the recognition state"""
        self.recognition_running = False
        self.face_recognition_active = False
        self.face_recognition_result = None
        self.recognition_thread = None
        self.face_recognition_progress = None
        # Note: Don't reset temp_face_encoding here as it needs to persist
        
    def store_face_encoding(self, face_encoding, face_id=None):
        """Store face encoding for later use in registration"""
        self.temp_face_encoding = face_encoding
        self.temp_face_id = face_id
        
    def get_face_encoding(self):
        """Get stored face encoding"""
        return self.temp_face_encoding
        
    def clear_face_encoding(self):
        """Clear stored face encoding"""
        self.temp_face_encoding = None
        self.temp_face_id = None
        
# Global instance
recognition_state = RecognitionState() 