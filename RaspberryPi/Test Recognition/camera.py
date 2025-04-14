import cv2
import time
import face_recognition
import numpy as np
from collections import deque
import random

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
    
    def detect_texture(self, image):
        """
        Detect if the image is a print/photo based on texture analysis
        Returns true if it appears to be a real face, false if it looks like a photo
        """
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Laplacian for edge detection - photos typically have fewer fine details
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Calculate variance of laplacian - higher variance means more texture/detail
        lap_variance = laplacian.var()
        
        # The threshold can be adjusted based on testing
        # Usually real faces have more texture detail than printed photos
        texture_threshold = 100
        
        # Debug output
        print(f"Texture variance: {lap_variance:.2f} (threshold: {texture_threshold})")
        
        return lap_variance > texture_threshold
    
    def detect_liveness_multi(self, timeout=20):
        """
        Advanced liveness detection using multiple techniques:
        1. Blink detection with higher thresholds
        2. Random head movement challenges
        3. Texture analysis to detect printed photos
        
        Args:
            timeout: Maximum time in seconds for the entire verification
            
        Returns:
            tuple: (is_live, face_image)
        """
        print("Enhanced liveness detection started...")
        
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        # Wait for camera to initialize
        time.sleep(0.5)
        
        # Parameters for blink detection - more strict thresholds
        EYE_AR_THRESH = 0.25  # Lower threshold makes it harder to trigger with printed photos
        EYE_AR_CONSEC_FRAMES = 2
        
        # Create variables to track verification stages
        stage_passed = {
            "blink": False,
            "texture": False,
            "head_movement": False
        }
        
        # Store eye aspect ratios for blink detection
        ear_history = deque(maxlen=8)  # Longer history for better analysis
        
        # For head movement challenge
        movement_type = random.choice(["left", "right", "up", "down"])
        movement_start_time = None
        movement_completed = False
        movement_positions = []  # To track nose positions
        
        # Track progress and time
        start_time = time.time()
        current_stage = "init"
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
        
        # Begin with texture analysis - this should happen first with a static face
        print("Stage 1: Looking for a real face (not a printout)...")
        current_stage = "texture"
        texture_frames = 0
        texture_passed_count = 0
        
        while True:
            # Check for overall timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
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
            
            # Display frame with overlays
            display_frame = frame.copy()
            
            # Calculate time remaining
            time_left = max(0, timeout - elapsed_time)
            
            # Detect face locations
            face_locations = face_recognition.face_locations(small_frame)
            
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
                    
                    # Extract face region for texture analysis
                    if current_stage == "texture" and texture_frames % 10 == 0:  # Check every 10 frames
                        face_region = frame[top:bottom, left:right]
                        if face_region.size > 0:  # Ensure the region is valid
                            has_texture = self.detect_texture(face_region)
                            if has_texture:
                                texture_passed_count += 1
                            # Require 3 passes to confirm it's a real face
                            if texture_passed_count >= 3:
                                stage_passed["texture"] = True
                                print("✓ Texture check passed! It appears to be a real face.")
                                # Move to blink detection
                                current_stage = "blink"
                                print("Stage 2: Please blink naturally...")
                    texture_frames += 1
                
                # Get facial landmarks for further analysis
                face_landmarks = face_recognition.face_landmarks(small_frame, face_locations)
                
                for landmarks in face_landmarks:
                    # Check if eye landmarks exist for blink detection
                    if current_stage in ["blink", "head_movement"] and 'left_eye' in landmarks and 'right_eye' in landmarks:
                        # Convert landmarks to numpy arrays
                        left_eye = np.array(landmarks['left_eye'])
                        right_eye = np.array(landmarks['right_eye'])
                        
                        # Scale back up to full size
                        left_eye = left_eye * 2
                        right_eye = right_eye * 2
                        
                        # Draw eyes
                        if current_stage == "blink":
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
                            
                            # Display EAR value
                            cv2.putText(display_frame, f"EAR: {ear:.2f}", 
                                      (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                            
                            # Check for blink - more stringent pattern matching
                            if len(ear_history) >= 8:
                                # Look for a clear pattern: open > partially closed > closed > partially open > open
                                # This is much harder to fake with a photo
                                if (ear_history[0] > EYE_AR_THRESH and
                                    ear_history[1] > EYE_AR_THRESH and
                                    ear_history[2] > EYE_AR_THRESH * 0.8 and
                                    ear_history[3] < EYE_AR_THRESH and
                                    ear_history[4] < EYE_AR_THRESH and
                                    ear_history[5] < EYE_AR_THRESH * 1.2 and
                                    ear_history[6] > EYE_AR_THRESH * 0.8 and
                                    ear_history[7] > EYE_AR_THRESH):
                                    stage_passed["blink"] = True
                                    print("✓ Blink detected! Moving to head movement test.")
                                    current_stage = "head_movement"
                                    # Save the head position challenge type
                                    print(f"Stage 3: Please move your head {movement_type.upper()}")
                                    movement_start_time = time.time()
                    
                    # Track nose position for head movement challenge
                    if current_stage == "head_movement" and 'nose_tip' in landmarks:
                        nose_point = np.array(landmarks['nose_tip'][0]) * 2  # Scale back up
                        
                        # Draw nose point
                        cv2.circle(display_frame, tuple(nose_point), 5, (0, 0, 255), -1)
                        
                        # Track nose positions
                        movement_positions.append(nose_point)
                        
                        # Only analyze after collecting enough positions
                        if len(movement_positions) > 10:
                            # Calculate movement direction
                            start_pos = np.mean(movement_positions[:5], axis=0)
                            current_pos = np.mean(movement_positions[-5:], axis=0)
                            
                            # Draw movement vector
                            cv2.arrowedLine(display_frame, 
                                          tuple(start_pos.astype(int)), 
                                          tuple(current_pos.astype(int)),
                                          (255, 0, 0), 2)
                            
                            # Calculate differences
                            x_diff = current_pos[0] - start_pos[0]
                            y_diff = current_pos[1] - start_pos[1]
                            
                            # Minimum pixel movement required
                            min_movement = 30
                            
                            # Check if movement matches the required direction
                            if movement_type == "left" and x_diff < -min_movement:
                                movement_completed = True
                            elif movement_type == "right" and x_diff > min_movement:
                                movement_completed = True
                            elif movement_type == "up" and y_diff < -min_movement:
                                movement_completed = True
                            elif movement_type == "down" and y_diff > min_movement:
                                movement_completed = True
                                
                            if movement_completed:
                                stage_passed["head_movement"] = True
                                face_image = frame
                                print(f"✓ Head movement {movement_type} detected! Liveness confirmed.")
            
            # Draw stage information
            stages_info = [
                f"1. Texture Analysis: {'✓' if stage_passed['texture'] else '...'}",
                f"2. Blink Detection: {'✓' if stage_passed['blink'] else '...' if stage_passed['texture'] else 'Waiting'}",
                f"3. Head Movement ({movement_type}): {'✓' if stage_passed['head_movement'] else '...' if stage_passed['blink'] else 'Waiting'}"
            ]
            
            for i, info in enumerate(stages_info):
                cv2.putText(display_frame, info, 
                          (10, 150 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                          (0, 255, 0) if "✓" in info else (255, 255, 255), 1)
            
            # Display time remaining
            cv2.putText(display_frame, f"Time left: {int(time_left)}s", 
                      (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display current instruction
            instruction = ""
            if current_stage == "texture":
                instruction = "Hold still while we analyze your face"
            elif current_stage == "blink":
                instruction = "Please blink naturally"
            elif current_stage == "head_movement":
                instruction = f"Please move your head {movement_type.upper()}"
                
            cv2.putText(display_frame, instruction, 
                      (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Check if all stages passed
            all_passed = all(stage_passed.values())
            if all_passed:
                cv2.putText(display_frame, "LIVENESS CONFIRMED!", 
                          (display_frame.shape[1]//2 - 150, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            # Show the frame
            cv2.imshow("Liveness Detection", display_frame)
            
            # Break if all stages passed or ESC pressed
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif all_passed:
                # Wait a moment to show the success message
                time.sleep(2)
                break
                
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        # Return the results
        return all_passed, face_image
    
    def detect_liveness(self, timeout=10):
        """
        Legacy liveness detection method - now redirects to the more secure version
        """
        print("Using enhanced liveness detection for better security...")
        return self.detect_liveness_multi(timeout)
        
    def capture_face_with_liveness(self):
        """Capture a face image with liveness verification"""
        is_live, face_image = self.detect_liveness_multi()
        
        if not is_live:
            print("Liveness check failed. Please try again.")
            return None
            
        print("Liveness check passed!")
        
        # If we got a face during liveness detection, use it
        if face_image is not None:
            return face_image
            
        # Otherwise capture a new face
        return self.capture_face() 