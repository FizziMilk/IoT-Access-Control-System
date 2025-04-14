import cv2
import time
import face_recognition
import numpy as np
from collections import deque

class CameraSystem:
    def __init__(self, camera_id=0, resolution=(640, 480)):
        self.camera_id = camera_id
        self.resolution = resolution
    
    def capture_face(self):
        """Capture an image from the camera with face detection"""
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        # Wait for camera to initialize
        time.sleep(0.5)
        
        # Show camera feed until space is pressed
        print("Position face in frame and press SPACE to capture or ESC to cancel")
        face_found = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image")
                break
            
            # Show the frame
            display_frame = frame.copy()
            
            # Try to find faces - ensure correct format
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Make sure the image is uint8 (0-255)
            if rgb_small_frame.dtype != np.uint8:
                rgb_small_frame = rgb_small_frame.astype(np.uint8)
                
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_small_frame)
            
            # Draw rectangle around faces
            for (top, right, bottom, left) in face_locations:
                # Scale back up
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                face_found = True
            
            # Show instructions on the frame
            text = "SPACE: Capture, ESC: Cancel"
            if face_found:
                status = "Face detected"
                color = (0, 255, 0)
            else:
                status = "No face detected"
                color = (0, 0, 255)
            
            cv2.putText(display_frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            cv2.imshow("Capture Face", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                cap.release()
                return None
            elif key == 32:  # SPACE
                if face_found:
                    break
        
        # Get final frame and clean up
        ret, frame = cap.read()
        cap.release()
        cv2.destroyAllWindows()
        
        if not ret:
            return None
            
        return frame
        
    def detect_liveness(self, timeout=10):
        """Perform liveness detection by detecting eye blinking
        
        Args:
            timeout: Maximum time in seconds to wait for a blink
            
        Returns:
            tuple: (is_live, face_image) - is_live is boolean, face_image is the captured image if live
        """
        print("Liveness detection started. Please blink naturally...")
        
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        # Wait for camera to initialize
        time.sleep(0.5)
        
        # Parameters for blink detection
        EYE_AR_THRESH = 0.3  # Eye aspect ratio threshold for blink detection
        CONSEC_FRAMES = 2    # Number of consecutive frames eye must be below threshold
        blink_detected = False
        
        # Store eye aspect ratios for a few frames to detect blinks
        ear_history = deque(maxlen=5)
        
        # Track progress and time
        start_time = time.time()
        face_image = None
        
        # Function to calculate eye aspect ratio
        def eye_aspect_ratio(eye):
            # Compute euclidean distance between eye landmarks
            A = np.linalg.norm(eye[1] - eye[5])
            B = np.linalg.norm(eye[2] - eye[4])
            C = np.linalg.norm(eye[0] - eye[3])
            # Calculate ratio
            ear = (A + B) / (2.0 * C)
            return ear
            
        while True:
            # Check for timeout
            if time.time() - start_time > timeout:
                print("Liveness detection timeout")
                break
                
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
                
            # Convert to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process at a smaller size for speed
            small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
            
            # Detect face locations
            face_locations = face_recognition.face_locations(small_frame)
            
            # Display frame with overlays
            display_frame = frame.copy()
            
            # Calculate time remaining
            time_left = max(0, timeout - (time.time() - start_time))
            
            if face_locations:
                # Scale back face location to full size
                for (top, right, bottom, left) in face_locations:
                    # Scale back up
                    top *= 2
                    right *= 2
                    bottom *= 2
                    left *= 2
                    
                    # Draw face rectangle
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Get facial landmarks
                face_landmarks = face_recognition.face_landmarks(small_frame, face_locations)
                
                for landmarks in face_landmarks:
                    # Check if eye landmarks exist
                    if 'left_eye' in landmarks and 'right_eye' in landmarks:
                        # Convert landmarks to numpy arrays
                        left_eye = np.array(landmarks['left_eye'])
                        right_eye = np.array(landmarks['right_eye'])
                        
                        # Scale back up to full size
                        left_eye = left_eye * 2
                        right_eye = right_eye * 2
                        
                        # Draw eyes
                        for eye_point in left_eye:
                            cv2.circle(display_frame, tuple(eye_point), 2, (0, 255, 255), -1)
                        for eye_point in right_eye:
                            cv2.circle(display_frame, tuple(eye_point), 2, (0, 255, 255), -1)
                            
                        # Calculate eye aspect ratio
                        left_ear = eye_aspect_ratio(left_eye)
                        right_ear = eye_aspect_ratio(right_eye)
                        
                        # Average the eye aspect ratio
                        ear = (left_ear + right_ear) / 2.0
                        ear_history.append(ear)
                        
                        # Check for blink
                        if len(ear_history) >= 5:
                            # A blink is when eye aspect ratio drops then rises again
                            if (ear_history[2] < EYE_AR_THRESH and 
                                ear_history[0] > EYE_AR_THRESH and 
                                ear_history[4] > EYE_AR_THRESH):
                                blink_detected = True
                                face_image = frame
                                print("Blink detected! Person is live.")
                        
                        # Display EAR value
                        cv2.putText(display_frame, f"EAR: {ear:.2f}", 
                                  (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Display status and instructions
            cv2.putText(display_frame, "Liveness Detection: Please blink naturally", 
                      (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Time left: {int(time_left)}s", 
                      (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if blink_detected:
                cv2.putText(display_frame, "LIVE PERSON DETECTED!", 
                          (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            # Show the frame
            cv2.imshow("Liveness Detection", display_frame)
            
            # Break if blink detected or ESC pressed
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif blink_detected:
                # Wait a moment to show the "LIVE PERSON" message
                time.sleep(2)
                break
                
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        return blink_detected, face_image
        
    def capture_face_with_liveness(self):
        """Capture a face image with liveness verification"""
        is_live, face_image = self.detect_liveness()
        
        if not is_live:
            print("Liveness check failed. Please try again.")
            return None
            
        print("Liveness check passed!")
        
        # If we got a face during liveness detection, use it
        if face_image is not None:
            return face_image
            
        # Otherwise capture a new face
        return self.capture_face() 