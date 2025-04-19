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
        
    def reset(self):
        """Reset the recognition state"""
        self.recognition_running = False
        self.face_recognition_active = False
        self.face_recognition_result = None
        self.recognition_thread = None
        
# Global instance
recognition_state = RecognitionState() 